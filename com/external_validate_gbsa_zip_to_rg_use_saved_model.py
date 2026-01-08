#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External validation: use the *trained* GBSA model stored in Model.zip (no retraining),
run an INTERNAL sanity check against zip test_preds_*.xlsx (must match),
then apply to Radiogenomics (external) and report Harrell C-index + bootstrap CI.

Key idea:
- Your previous script retrained GBSA -> internal C-index drifted (0.6319 vs 0.6654).
- Model.zip typically contains a pickled fitted estimator (.joblib) under:
    {task}/seed_{seed}/{model}/all_train_data_{model}.joblib   (name varies)
  We load that estimator and use it directly.

Requirements (in the environment where you run this):
- pandas, numpy, pyyaml, joblib, openpyxl
- scikit-learn (ideally the same version used to create the pickles; your zip shows 1.3.2)
- scikit-survival (sksurv)

If INTERNAL sanity check fails:
- create a clean env with sklearn==1.3.2 + the same sksurv version as your baseline run, then rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import yaml

from sksurv.metrics import concordance_index_censored


# -----------------------------
# Utilities
# -----------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_txt_lines_from_zip(z: zipfile.ZipFile, member: str) -> List[str]:
    raw = z.read(member).decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in raw.splitlines()]
    return [ln for ln in lines if ln]


def _find_first_existing(z: zipfile.ZipFile, candidates: List[str]) -> Optional[str]:
    names = set(z.namelist())
    for c in candidates:
        if c in names:
            return c
    return None


def _find_by_regex(z: zipfile.ZipFile, pattern: str) -> List[str]:
    rx = re.compile(pattern)
    return [n for n in z.namelist() if rx.search(n)]


def _extract_member(z: zipfile.ZipFile, member: str, out_dir: str) -> str:
    z.extract(member, out_dir)
    return os.path.join(out_dir, member)


def _load_joblib_from_zip(z: zipfile.ZipFile, member: str, tmp_dir: str):
    path = _extract_member(z, member, tmp_dir)
    return joblib.load(path)


def _load_yaml_from_zip(z: zipfile.ZipFile, member: str) -> dict:
    raw = z.read(member).decode("utf-8", errors="ignore")
    return yaml.safe_load(raw)


def _load_selected_features_xlsx_from_zip(z: zipfile.ZipFile, member: str, tmp_dir: str) -> List[str]:
    path = _extract_member(z, member, tmp_dir)
    df = pd.read_excel(path)
    if df.shape[1] == 0:
        return []
    # try common column names, prioritizing "secondary" (final selected) over "primary"
    for col in ["secondary_selected_features", "feature", "Feature", "selected_features", "SelectedFeatures", "Selected_Features", "primary_selected_features"]:
        if col in df.columns:
            feats = df[col].dropna().astype(str).tolist()
            return feats
    # fallback: first column
    return df.iloc[:, 0].dropna().astype(str).tolist()


def _detect_risk_column(df: pd.DataFrame) -> str:
    lc = {c.lower(): c for c in df.columns}
    for cand in ["risk", "pred", "prediction", "y_pred", "pred_risk", "hazard", "log_hazard", "score"]:
        if cand in lc:
            return lc[cand]
    # fallback: numeric columns excluding time/event/id-like
    bad = set()
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["patient", "id", "time", "event", "status", "dead"]):
            bad.add(c)
    num_cols = [c for c in df.columns if c not in bad and pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) == 1:
        return num_cols[0]
    if len(num_cols) >= 1:
        return num_cols[-1]
    raise ValueError(f"Cannot detect risk column from columns={df.columns.tolist()}")


def _bootstrap_cindex(time: np.ndarray, event: np.ndarray, risk: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    n = len(time)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = concordance_index_censored(event[idx].astype(bool), time[idx].astype(float), risk[idx].astype(float))[0]
        if np.isfinite(c):
            vals.append(float(c))
    if len(vals) == 0:
        return float("nan"), float("nan"), (float("nan"), float("nan"))
    vals = np.array(vals, dtype=float)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    lo, hi = np.percentile(vals, [2.5, 97.5]).tolist()
    return mean, std, (float(lo), float(hi))


def _parse_stage_int_from_str(s: str) -> Optional[int]:
    """
    Convert 'T1a'->1, 'N2'->2, 'M1b'->1, etc.
    Returns None if cannot parse.
    """
    if s is None:
        return None
    ss = str(s).strip().upper()
    if ss in ("", "NA", "NAN", "NONE", "NOT COLLECTED", "UNKNOWN"):
        return None
    # If already a number-like string
    if re.fullmatch(r"-?\d+(\.\d+)?", ss):
        try:
            return int(float(ss))
        except Exception:
            return None
    m = re.search(r"[TNM]\s*([0-9])", ss)
    if m:
        return int(m.group(1))
    return None


def _canonicalize_gender(val) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s.startswith("m"):
        return "male"
    if s.startswith("f"):
        return "female"
    # some datasets use 1/2
    if s in ("0", "1", "2"):
        # cannot know mapping safely; return as-is
        return s
    return s if s else None


def _canonicalize_histology(val) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    return s.lower()


def _canonicalize_overall_stage(val) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    # normalize common forms
    ss = s.upper().replace(" ", "")
    ss = ss.replace("STAGE", "")
    # common typos (Illb using lowercase L)
    ss = ss.replace("ILL", "III")
    # keep as typical stage strings
    # Accept: I, II, IIIA, IIIB, IV
    if ss in ("I", "II", "III", "IIIA", "IIIB", "IV"):
        return ss
    # sometimes '3A' or '3B'
    if ss in ("3A", "3B"):
        return "III" + ss[-1]
    # sometimes 'IIIa','IIIb'
    if ss in ("IIIA", "IIIB"):
        return ss
    return s  # fallback raw


def _best_match_to_known_category(val: Optional[str], known: List[str], canon_fn=None) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if not s:
        return None
    known_set = set([str(x) for x in known])
    # direct
    if s in known_set:
        return s
    # case variants
    if s.lower() in known_set:
        return s.lower()
    if s.upper() in known_set:
        return s.upper()
    # canonicalization
    if canon_fn is not None:
        c = canon_fn(s)
        if c is None:
            return None
        if c in known_set:
            return c
        if str(c).lower() in known_set:
            return str(c).lower()
        if str(c).upper() in known_set:
            return str(c).upper()
    # space-free
    t = s.replace(" ", "")
    if t in known_set:
        return t
    if t.lower() in known_set:
        return t.lower()
    if t.upper() in known_set:
        return t.upper()
    # leave as-is (may become unknown)
    return s


@dataclass
class ZipArtifacts:
    fill_values: dict
    scaler: object
    encoder: object
    numeric_cols: List[str]
    cat_cols: List[str]
    selected_features: List[str]
    fitted_model: object
    # for sanity check
    zip_test_preds_path: Optional[str] = None


def _load_artifacts_from_zip(zip_path: str, task: str, seed: int, model: str) -> ZipArtifacts:
    """
    Load preprocessing artifacts + selected features + fitted model estimator from zip.
    Supports both:
      - {task}/seed_{seed}/preprocess/...
      - {task}/seed_{seed}/... (flat)
    Model is searched under:
      - {task}/seed_{seed}/{model}/...joblib  (largest non-surv_fn joblib)
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        names = set(z.namelist())

        # ---- preprocess base
        preprocess_base_candidates = [
            f"{task}/seed_{seed}/preprocess/",
            f"{task}/seed_{seed}/",
        ]
        preprocess_base = None
        for b in preprocess_base_candidates:
            if any(n.startswith(b) for n in names):
                preprocess_base = b
                break
        if preprocess_base is None:
            raise FileNotFoundError(f"Cannot find preprocess base for task={task} seed={seed} in zip.")

        # required members (try multiple names)
        fill_member = _find_first_existing(z, [
            preprocess_base + "train_fill_values.joblib",
            preprocess_base + "fill_values.joblib",
        ])
        if fill_member is None:
            raise FileNotFoundError("Missing train_fill_values.joblib in zip (preprocess artifacts).")

        scaler_member = _find_first_existing(z, [
            preprocess_base + "normalizer.joblib",
            preprocess_base + "numeric_scaler.joblib",
            preprocess_base + "scaler.joblib",
        ])
        if scaler_member is None:
            raise FileNotFoundError("Missing normalizer/scaler joblib in zip (preprocess artifacts).")

        encoder_member = _find_first_existing(z, [
            preprocess_base + "categorical_encoder.joblib",
            preprocess_base + "onehot_encoder.joblib",
            preprocess_base + "encoder.joblib",
        ])
        # If no separate encoder, it's inside normalizer.joblib
        use_combined_normalizer = (encoder_member is None)

        numcols_member = _find_first_existing(z, [
            preprocess_base + "numeric_columns.txt",
            preprocess_base + "numeric_cols.txt",
        ])
        catcols_member = _find_first_existing(z, [
            preprocess_base + "categorical_columns.txt",
            preprocess_base + "categorical_cols.txt",
        ])

        # ---- model base
        model_base = f"{task}/seed_{seed}/{model}/"
        if not any(n.startswith(model_base) for n in names):
            raise FileNotFoundError(f"Cannot find model dir {model_base} in zip.")

        # selected features xlsx
        sel_member = _find_first_existing(z, [
            model_base + f"selected_features_{model}.xlsx",
            model_base + "selected_features.xlsx",
        ])
        if sel_member is None:
            # fallback: any selected_features*.xlsx in model dir
            cands = [n for n in names if n.startswith(model_base) and n.lower().endswith(".xlsx") and "selected_features" in n.lower()]
            sel_member = cands[0] if cands else None
        if sel_member is None:
            raise FileNotFoundError("Missing selected_features.xlsx in zip (model artifacts).")

        # zip test preds xlsx (for sanity check)
        zip_test_preds_member = _find_first_existing(z, [
            model_base + f"test_preds_{model}.xlsx",
            model_base + "test_preds.xlsx",
        ])
        if zip_test_preds_member is None:
            # fallback: any test_preds*.xlsx
            cands = [n for n in names if n.startswith(model_base) and n.lower().endswith(".xlsx") and "test_preds" in n.lower()]
            zip_test_preds_member = cands[0] if cands else None

        # find fitted model joblib
        # We want a joblib that is NOT train/test surv_fn, and is likely the largest object.
        joblibs = []
        for n in names:
            if not n.startswith(model_base):
                continue
            if not n.lower().endswith(".joblib"):
                continue
            bn = os.path.basename(n).lower()
            if "surv_fn" in bn:
                continue
            if "pred" in bn:
                continue
            joblibs.append(n)
        if not joblibs:
            raise FileNotFoundError(
                "Cannot find a fitted model joblib under the model dir.\n"
                f"Looked in: {model_base}\n"
                "Expected something like all_train_data_*.joblib or model.joblib.\n"
                "If your zip truly doesn't store a fitted estimator, you cannot do exact external inference."
            )
        # choose largest by file size
        joblibs.sort(key=lambda n: z.getinfo(n).file_size, reverse=True)
        model_member = joblibs[0]

        # ---- extract / load
        with tempfile.TemporaryDirectory() as tmp:
            fill_obj = _load_joblib_from_zip(z, fill_member, tmp)
            
            if use_combined_normalizer:
                # normalizer.joblib contains both scaler and encoder
                normalizer = _load_joblib_from_zip(z, scaler_member, tmp)
                scaler = normalizer['normalizers']['numeric']
                encoder = normalizer['normalizers']['categorical']
                # Get columns from normalizer if not in separate files
                numeric_cols = normalizer.get('numeric_cols', [])
                cat_cols = normalizer.get('category_columns', [])
            else:
                scaler = _load_joblib_from_zip(z, scaler_member, tmp)
                encoder = _load_joblib_from_zip(z, encoder_member, tmp)
                numeric_cols = None
                cat_cols = None
            
            fitted_model = _load_joblib_from_zip(z, model_member, tmp)
            selected_features = _load_selected_features_xlsx_from_zip(z, sel_member, tmp)

            # Override with txt files if they exist
            if numcols_member is not None:
                numeric_cols = _read_txt_lines_from_zip(z, numcols_member)
            elif numeric_cols is None or not numeric_cols:
                numeric_cols = list(getattr(scaler, "feature_names_in_", []))
                if not numeric_cols:
                    raise FileNotFoundError("numeric_columns.txt missing and scaler has no feature_names_in_.")
            
            if catcols_member is not None:
                cat_cols = _read_txt_lines_from_zip(z, catcols_member)
            elif cat_cols is None or not cat_cols:
                cat_cols = list(getattr(encoder, "feature_names_in_", []))
                if not cat_cols:
                    raise FileNotFoundError("categorical_columns.txt missing and encoder has no feature_names_in_.")

            return ZipArtifacts(
                fill_values=fill_obj,
                scaler=scaler,
                encoder=encoder,
                numeric_cols=numeric_cols,
                cat_cols=cat_cols,
                selected_features=selected_features,
                fitted_model=fitted_model,
                zip_test_preds_path=zip_test_preds_member,
            )


def _unpack_fill_values(fill_obj) -> Tuple[Dict[str, object], Dict[str, object]]:
    """
    Return (clinical_fill, radiomics_fill). If the zip only stores a flat dict, return it for both.
    """
    clinical_fill = {}
    radiomics_fill = {}
    if isinstance(fill_obj, dict):
        # common nested formats
        if "clinical" in fill_obj or "radiomics" in fill_obj:
            clinical_fill = fill_obj.get("clinical", fill_obj.get("clinical_fill_values", {})) or {}
            radiomics_fill = fill_obj.get("radiomics", fill_obj.get("radiomics_fill_values", {})) or {}
        else:
            # flat dict: use for both
            clinical_fill = fill_obj
            radiomics_fill = fill_obj
    return clinical_fill, radiomics_fill


def _load_split_ids(split_json_path: str) -> Tuple[List[str], List[str]]:
    js = json.load(open(split_json_path, "r"))
    # support different keys
    train_ids = js.get("train_ids") or js.get("train") or js.get("train_index") or js.get("train_idx")
    test_ids = js.get("test_ids") or js.get("test") or js.get("test_index") or js.get("test_idx")
    if train_ids is None or test_ids is None:
        raise KeyError(f"split_json missing train/test ids keys: {list(js.keys())}")
    train_ids = [str(x) for x in train_ids]
    test_ids = [str(x) for x in test_ids]
    return train_ids, test_ids


def _load_lung1_joint(lung1_clinical_csv: str, lung1_radiomics_csv: str) -> pd.DataFrame:
    clin = pd.read_csv(lung1_clinical_csv)
    if "PatientID" not in clin.columns:
        raise KeyError("Lung1 clinical CSV must have column 'PatientID'.")
    clin["PatientID"] = clin["PatientID"].astype(str)
    clin = clin.set_index("PatientID", drop=False)

    rad = pd.read_csv(lung1_radiomics_csv)
    pid_col = None
    for c in ["PatientID", "patient_id", "patient", "case_id", "CaseID", "ID"]:
        if c in rad.columns:
            pid_col = c
            break
    if pid_col is None:
        # assume first column is id
        pid_col = rad.columns[0]
    rad[pid_col] = rad[pid_col].astype(str)
    rad = rad.set_index(pid_col, drop=False)
    if pid_col != "PatientID":
        rad.index.name = "PatientID"
        rad = rad.rename(columns={pid_col: "PatientID"})

    # drop non-feature columns if present
    if "PatientID" in rad.columns:
        rad_features = rad.drop(columns=["PatientID"], errors="ignore")
    else:
        rad_features = rad

    df = clin.join(rad_features, how="inner")
    return df


def _load_rg_joint(rg_clinical_aligned_csv: str, rg_radiomics_csv: str, id_prefix_from: Optional[str], id_prefix_to: Optional[str]) -> pd.DataFrame:
    # clinical
    df = pd.read_csv(rg_clinical_aligned_csv)
    # support both PatientID/patient_id
    if "PatientID" in df.columns:
        df["PatientID"] = df["PatientID"].astype(str)
    elif "patient_id" in df.columns:
        df["PatientID"] = df["patient_id"].astype(str)
    else:
        raise KeyError("RG clinical aligned CSV must have PatientID or patient_id.")
    # unify common field names
    rename_map = {
        "time_days": "Survival.time",
        "event": "deadstatus.event",
        "overall_stage": "Overall.Stage",
        "histology": "Histology",
        "T": "clinical.T.Stage",
        "N": "Clinical.N.Stage",
        "M": "Clinical.M.Stage",
        "gender": "gender",
        "age": "age",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]
    df = df.set_index("PatientID", drop=False)

    # radiomics
    rad = pd.read_csv(rg_radiomics_csv)
    pid_col = None
    for c in ["PatientID", "patient_id", "patient", "case_id", "CaseID", "ID"]:
        if c in rad.columns:
            pid_col = c
            break
    if pid_col is None:
        pid_col = rad.columns[0]
    rad[pid_col] = rad[pid_col].astype(str)
    rad = rad.set_index(pid_col, drop=False)
    if pid_col != "PatientID":
        rad.index.name = "PatientID"
        rad = rad.rename(columns={pid_col: "PatientID"})
    rad_features = rad.drop(columns=["PatientID"], errors="ignore")

    # optional id prefix translation on radiomics side (or both)
    if id_prefix_from and id_prefix_to:
        def _tr(x: str) -> str:
            return x.replace(id_prefix_from, id_prefix_to, 1) if x.startswith(id_prefix_from) else x
        df.index = df.index.map(_tr)
        df["PatientID"] = df.index
        rad_features.index = rad_features.index.map(_tr)

    joint = df.join(rad_features, how="inner")
    return joint

_GENDER_MAP = {"female": 0, "male": 1}

_HISTO_MAP = {
    "adenocarcinoma": 0,
    "large cell": 1,
    "nos": 2,
    "squamous cell carcinoma": 3,
}

_STAGE_MAP = {
    "I": 0,
    "II": 1,
    "IIIA": 2,
    "IIIB": 3,
}

def _canon_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _canon_gender(x):
    s = _canon_str(x)
    if s in {"m", "male", "1"}:
        return "male"
    if s in {"f", "female", "0"}:
        return "female"
    return ""  # unknown

def _canon_histology(x):
    s = _canon_str(x)
    # normalize common variants
    if "adeno" in s:
        return "adenocarcinoma"
    if "squamous" in s:
        return "squamous cell carcinoma"
    if "large" in s and "cell" in s:
        return "large cell"
    if s in {"nos", "n.o.s", "n.o.s."}:
        return "nos"
    return s  # try as-is (maybe already exact key)

def _canon_overall_stage(x):
    s = str(x).strip().upper() if not pd.isna(x) else ""
    s = s.replace("STAGE", "").replace(" ", "")
    # unify roman / substage
    if s.startswith("III"):
        if "A" in s:
            return "IIIA"
        if "B" in s:
            return "IIIB"
        # plain "III" -> treat as IIIA? (rare). leave unknown -> ""
        return ""
    if s.startswith("II"):
        return "II"
    if s.startswith("I"):
        return "I"
    return ""  # unknown

def _parse_stage_int(x):
    """Parse T/N/M stage like 'T1a'/'N2'/'M0' or numeric 0/1/2 into int, else NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    # already numeric
    try:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, (float, np.floating)) and np.isfinite(x):
            return int(x)
    except Exception:
        pass
    s = str(x).strip().upper()
    if s == "":
        return np.nan
    # capture first digit after optional TNM prefix
    m = re.search(r"(?:[TNM]\s*)?([0-9]+)", s)
    if not m:
        return np.nan
    return int(m.group(1))

def _encode_cat_int(df: pd.DataFrame, clinical_fill: dict, cat_cols: list) -> pd.DataFrame:
    """
    Convert categorical features to integers as expected by the encoder.
    This matches the training script's encode_clinical_report_like() function.
    """
    out = pd.DataFrame(index=df.index)

    # gender: female/male -> 0/1
    if "gender" in cat_cols:
        g = df["gender"].map(lambda x: str(x).strip().lower() if pd.notna(x) else "")
        out["gender"] = g.map({"female": 0, "male": 1}).fillna(1).astype(int)  # missing -> 1 (male)

    # histology: map strings -> 0..3, missing -> 4
    if "Histology" in cat_cols:
        h = df["Histology"].map(lambda x: str(x).strip().lower() if pd.notna(x) else "")
        histo_map = {
            "adenocarcinoma": 0,
            "large cell": 1,
            "nos": 2,
            "squamous cell carcinoma": 3,
        }
        out["Histology"] = h.map(histo_map).fillna(4).astype(int)  # missing -> 4

    # overall stage: I/II/IIIa/IIIb -> 0..3, missing -> 3
    if "Overall.Stage" in cat_cols:
        st = df["Overall.Stage"].map(lambda x: str(x).strip().upper() if pd.notna(x) else "")
        st = st.str.replace("STAGE", "").str.replace(" ", "")
        # Handle IIIa/IIIb variants
        st = st.map(lambda s: "IIIA" if s.startswith("III") and "A" in s else 
                           ("IIIB" if s.startswith("III") and "B" in s else s))
        stage_map = {"I": 0, "II": 1, "IIIA": 2, "IIIB": 3, "IIIa": 2, "IIIb": 3}
        out["Overall.Stage"] = st.map(stage_map).fillna(3).astype(int)  # missing -> 3

    # T stage: numeric 1..5, missing -> 0 (training script: fillna(0))
    if "clinical.T.Stage" in cat_cols:
        t = pd.to_numeric(df["clinical.T.Stage"], errors="coerce")
        out["clinical.T.Stage"] = t.fillna(0).astype(int)

    # N stage: numeric 0..4, missing -> 0 (match debug script)
    if "Clinical.N.Stage" in cat_cols:
        n = pd.to_numeric(df["Clinical.N.Stage"], errors="coerce")
        out["Clinical.N.Stage"] = n.fillna(0).astype(int)

    # M stage: numeric 0,1,3; map 3 -> 2; missing -> 0 (match debug script)
    if "Clinical.M.Stage" in cat_cols:
        m = pd.to_numeric(df["Clinical.M.Stage"], errors="coerce")
        m = m.map(lambda x: 2 if x == 3 else x)  # map 3 -> 2
        out["Clinical.M.Stage"] = m.fillna(0).astype(int)

    # Ensure all cat_cols are present
    for c in cat_cols:
        if c not in out.columns:
            fill = clinical_fill.get(c, 0)
            out[c] = fill
        # Final type: int
        out[c] = out[c].astype(int)

    # keep column order same as cat_cols
    return out[cat_cols].copy()


def _build_X_from_joint(
    joint: pd.DataFrame,
    art: ZipArtifacts,
    clinical_fill: Dict[str, object],
    radiomics_fill: Dict[str, object],
    is_external_rg: bool,
) -> pd.DataFrame:
    """
    Build full preprocessed feature matrix (numeric scaled + categorical encoded) for given joint df.
    Returns DataFrame with feature names matching training.
    """
    df = joint.copy()

    # --- target columns (time/event) are not used in X
    # ensure clinical columns exist (create if missing)
    for col in ["age", "gender", "Histology", "clinical.T.Stage", "Overall.Stage", "Clinical.N.Stage", "Clinical.M.Stage"]:
        if col not in df.columns:
            df[col] = np.nan

    # --- external-specific parsing for T/N/M strings into numeric codes
    if is_external_rg:
        # T stage
        df["clinical.T.Stage"] = df["clinical.T.Stage"].apply(lambda x: _parse_stage_int_from_str(x))
        # N/M if present (even if you don't use them, harmless)
        df["Clinical.N.Stage"] = df["Clinical.N.Stage"].apply(lambda x: _parse_stage_int_from_str(x))
        df["Clinical.M.Stage"] = df["Clinical.M.Stage"].apply(lambda x: _parse_stage_int_from_str(x))

    # --- apply fill values (ONLY for numeric features, NOT categorical)
    # Categorical features will be handled in _encode_cat_int
    for k, v in radiomics_fill.items():
        if k in df.columns and k in art.numeric_cols:
            df[k] = df[k].fillna(v)
    # Also fill numeric clinical features (e.g., age)
    for k, v in clinical_fill.items():
        if k in df.columns and k in art.numeric_cols:
            df[k] = df[k].fillna(v)

    # --- numeric matrix
    # ensure all numeric cols exist
    for c in art.numeric_cols:
        if c not in df.columns:
            df[c] = np.nan
    X_num = df[art.numeric_cols].apply(pd.to_numeric, errors="coerce")
    # fill any remaining NaNs in numeric with 0 as last resort (should be rare if fill_values complete)
    X_num = X_num.fillna(0.0)

    # --- categorical matrix (FIXED: must be integer-coded for encoder)
    for c in art.cat_cols:
        if c not in df.columns:
            df[c] = np.nan

    X_cat = _encode_cat_int(df, clinical_fill=clinical_fill, cat_cols=art.cat_cols)

    # --- transform
    # scaler/encoder may warn about feature names if we pass np arrays; pass DataFrames.
    num_scaled = art.scaler.transform(X_num[art.numeric_cols])
    X_num_df = pd.DataFrame(num_scaled, index=df.index, columns=art.numeric_cols)

    cat_enc = art.encoder.transform(X_cat[art.cat_cols])
    # handle sparse / dense
    if hasattr(cat_enc, "toarray"):
        cat_arr = cat_enc.toarray()
    else:
        cat_arr = np.asarray(cat_enc)
    cat_names = art.encoder.get_feature_names_out(art.cat_cols)
    X_cat_df = pd.DataFrame(cat_arr, index=df.index, columns=cat_names)

    X = pd.concat([X_num_df, X_cat_df], axis=1)

    return X


def _select_features(X: pd.DataFrame, selected: List[str]) -> pd.DataFrame:
    missing = [f for f in selected if f not in X.columns]
    if missing:
        # do not silently drop: this changes the model input.
        raise KeyError(
            f"{len(missing)} selected features are missing from design matrix.\n"
            f"First few: {missing[:10]}\n"
            "This indicates preprocessing mismatch (encoder categories, column naming, etc.)."
        )
    return X[selected].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip_path", required=True)
    ap.add_argument("--task", required=True, help="e.g. 'Clinic+Radiomics'")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", required=True, help="e.g. 'GradientBoostingSurvivalAnalysis'")

    ap.add_argument("--lung1_clinical_csv", required=True)
    ap.add_argument("--lung1_radiomics_csv", required=True)
    ap.add_argument("--split_json", required=True)

    ap.add_argument("--rg_clinical_aligned_csv", required=True)
    ap.add_argument("--rg_radiomics_csv", required=True)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--drop_m1", type=int, default=0, help="1 to drop M1 (after parsing M stage) in RG")
    ap.add_argument("--id_prefix_from", default=None)
    ap.add_argument("--id_prefix_to", default=None)

    # strict sanity check
    ap.add_argument("--sanity_tol", type=float, default=1e-6, help="abs diff tolerance for internal c-index")

    args = ap.parse_args()
    _ensure_dir(args.out_dir)

    # 1) Load artifacts (including *fitted model*)
    art = _load_artifacts_from_zip(args.zip_path, args.task, args.seed, args.model)
    clinical_fill, radiomics_fill = _unpack_fill_values(art.fill_values)

    # 2) Load data
    joint_lung1 = _load_lung1_joint(args.lung1_clinical_csv, args.lung1_radiomics_csv)
    train_ids, test_ids = _load_split_ids(args.split_json)
    df_tr = joint_lung1.loc[joint_lung1.index.intersection(train_ids)].copy()
    df_te = joint_lung1.loc[joint_lung1.index.intersection(test_ids)].copy()
    if len(df_te) == 0:
        raise RuntimeError("Internal test set is empty after intersecting split ids with Lung1 data.")

    # survival targets (internal)
    t_te = pd.to_numeric(df_te.get("Survival.time"), errors="coerce").values.astype(float)
    e_te = pd.to_numeric(df_te.get("deadstatus.event"), errors="coerce").fillna(0).values.astype(int)

    # 3) INTERNAL sanity check: predict with loaded model and compare with zip test_preds
    X_te_full = _build_X_from_joint(df_te, art, clinical_fill, radiomics_fill, is_external_rg=False)
    
    # Use model's feature_names_in_ if available (more reliable than selected_features.xlsx)
    if hasattr(art.fitted_model, 'feature_names_in_'):
        final_features = list(art.fitted_model.feature_names_in_)
        print(f"[INFO] Using model's feature_names_in_ ({len(final_features)} features)")
    else:
        final_features = art.selected_features
        print(f"[INFO] Using selected_features.xlsx ({len(final_features)} features)")
    
    X_te = _select_features(X_te_full, final_features)
    
    # DEBUG: Check feature matrix
    print(f"[DEBUG] X_te shape: {X_te.shape}")
    print(f"[DEBUG] X_te first 5 rows, first 5 columns:")
    print(X_te.iloc[:5, :5])
    print(f"[DEBUG] X_te stats: mean={X_te.values.mean():.6f}, std={X_te.values.std():.6f}, min={X_te.values.min():.6f}, max={X_te.values.max():.6f}")
    print(f"[DEBUG] X_te NaN count: {X_te.isna().sum().sum()}")
    print(f"[DEBUG] X_te inf count: {np.isinf(X_te.values).sum()}")
    
    # Try predicting with DataFrame (not numpy array)
    risk_te = art.fitted_model.predict(X_te)
    print(f"[DEBUG] Predicted risk_te stats: mean={risk_te.mean():.6f}, std={risk_te.std():.6f}, min={risk_te.min():.6f}, max={risk_te.max():.6f}")
    c_internal = concordance_index_censored(e_te.astype(bool), t_te, risk_te)[0]
    boot_mean, boot_std, ci95 = _bootstrap_cindex(t_te, e_te, risk_te, n_boot=args.bootstrap, seed=args.seed)

    print(f"[INTERNAL via ZIP MODEL] task={args.task} seed={args.seed} model={args.model}")
    print(f"  n_train={len(df_tr)} n_test={len(df_te)} n_features={X_te.shape[1]}")
    print(f"  test_cindex={c_internal:.6f}  boot_mean±std={boot_mean:.6f}±{boot_std:.6f}  CI95=[{ci95[0]:.6f},{ci95[1]:.6f}]")

    # Compare to zip's own test_preds if present
    c_zip = None
    if art.zip_test_preds_path is not None:
        with zipfile.ZipFile(args.zip_path, "r") as z:
            with tempfile.TemporaryDirectory() as tmp:
                p = _extract_member(z, art.zip_test_preds_path, tmp)
                df_zip = pd.read_excel(p)
        # attempt to align by PatientID
        pid_col = None
        for c in df_zip.columns:
            if c.lower() in ("patientid", "patient_id", "id"):
                pid_col = c
                break
        if pid_col is None:
            pid_col = df_zip.columns[0]
        df_zip[pid_col] = df_zip[pid_col].astype(str)
        df_zip = df_zip.set_index(pid_col, drop=False)
        risk_col = _detect_risk_column(df_zip)
        # use outcomes from df_te (truth)
        df_zip = df_zip.loc[df_zip.index.intersection(df_te.index)].copy()
        risk_zip = df_zip[risk_col].astype(float).values
        t_zip = pd.to_numeric(df_te.loc[df_zip.index, "Survival.time"], errors="coerce").values.astype(float)
        e_zip = pd.to_numeric(df_te.loc[df_zip.index, "deadstatus.event"], errors="coerce").fillna(0).values.astype(int)
        c_zip = concordance_index_censored(e_zip.astype(bool), t_zip, risk_zip)[0]
        print(f"[ZIP test_preds] cindex={c_zip:.6f}  (risk_col='{risk_col}')")

        # hard gate
        if c_zip is not None and abs(float(c_internal) - float(c_zip)) > args.sanity_tol:
            raise RuntimeError(
                "INTERNAL sanity check FAILED:\n"
                f"  cindex(zip_model_predict)={c_internal:.6f}\n"
                f"  cindex(zip_test_preds)    ={c_zip:.6f}\n"
                f"  diff={abs(c_internal-c_zip):.6f} > tol={args.sanity_tol}\n"
                "This means your runtime stack cannot reproduce the zip model inference reliably.\n"
                "Fix: run this script in an environment matching the pickles (e.g., sklearn==1.3.2).\n"
            )

    # 4) EXTERNAL (Radiogenomics)
    joint_rg = _load_rg_joint(args.rg_clinical_aligned_csv, args.rg_radiomics_csv, args.id_prefix_from, args.id_prefix_to)
    if args.drop_m1:
        # After parsing, M stage is integer-ish; drop == 1
        m_parsed = joint_rg.get("Clinical.M.Stage")
        if m_parsed is not None:
            m_code = m_parsed.apply(lambda x: _parse_stage_int_from_str(x))
            joint_rg = joint_rg.loc[~(m_code == 1)].copy()

    # external outcomes
    t_ext = pd.to_numeric(joint_rg.get("Survival.time"), errors="coerce").values.astype(float)
    e_ext = pd.to_numeric(joint_rg.get("deadstatus.event"), errors="coerce").fillna(0).values.astype(int)

    # basic external stats
    n_ext = len(joint_rg)
    n_evt = int(e_ext.sum())
    evt_rate = float(n_evt / n_ext) if n_ext else float("nan")
    print(f"[EXTERNAL] Radiogenomics  n_ext={n_ext} events={n_evt} event_rate={evt_rate:.3f}")

    X_ext_full = _build_X_from_joint(joint_rg, art, clinical_fill, radiomics_fill, is_external_rg=True)
    X_ext = _select_features(X_ext_full, final_features)
    risk_ext = art.fitted_model.predict(X_ext.values)

    c_ext = concordance_index_censored(e_ext.astype(bool), t_ext, risk_ext)[0]
    bmean_ext, bstd_ext, ci_ext = _bootstrap_cindex(t_ext, e_ext, risk_ext, n_boot=args.bootstrap, seed=args.seed)

    print(f"  cindex={c_ext:.6f}  boot_mean±std={bmean_ext:.6f}±{bstd_ext:.6f}  CI95=[{ci_ext[0]:.6f},{ci_ext[1]:.6f}]")

    # 5) Save outputs
    out_pred = pd.DataFrame({
        "PatientID": joint_rg.index.astype(str),
        "Survival.time": t_ext,
        "deadstatus.event": e_ext,
        "risk": risk_ext,
    }).set_index("PatientID", drop=False)
    pred_path = os.path.join(args.out_dir, "radiogenomics_preds.csv")
    out_pred.to_csv(pred_path, index=False)

    summary = {
        "zip_path": args.zip_path,
        "task": args.task,
        "seed": args.seed,
        "model": args.model,
        "internal": {
            "n_test": int(len(df_te)),
            "test_cindex": float(c_internal),
            "boot_mean": float(boot_mean),
            "boot_std": float(boot_std),
            "ci95": [float(ci95[0]), float(ci95[1])],
            "zip_test_preds_cindex": (float(c_zip) if c_zip is not None else None),
        },
        "external_radiogenomics": {
            "n_ext": int(n_ext),
            "events": int(n_evt),
            "event_rate": float(evt_rate),
            "cindex": float(c_ext),
            "boot_mean": float(bmean_ext),
            "boot_std": float(bstd_ext),
            "ci95": [float(ci_ext[0]), float(ci_ext[1])],
        },
        "notes": {
            "inference_mode": "LOAD_FITTED_MODEL_FROM_ZIP (no retraining)",
            "sanity_tol": args.sanity_tol,
        }
    }
    summary_path = os.path.join(args.out_dir, "external_validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved: {pred_path}")
    print(f"[OK] Saved: {summary_path}")


if __name__ == "__main__":
    main()
