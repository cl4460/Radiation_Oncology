#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External validation on NSCLC-Radiogenomics using the *exact* preprocessing/specs from Model.zip
(best internal model: Clinic+Radiomics / seed_91 / GradientBoostingSurvivalAnalysis).

What this script does (single run, no tuning on external):
1) Load preprocessing artifacts from Model.zip (train_fill_values + fitted normalizer).
2) Load selected features + best GBSA params from Model.zip.
3) Build Lung1 train/test matrices (using split_json) and refit GBSA with those exact settings.
4) Sanity-check: internal test c-index should match the zip baseline (~0.6654 for seed_91).
5) Build Radiogenomics external matrix (clinical aligned + extracted 1688 radiomics from SAME YAML).
6) Predict external risks and compute Harrell C-index + bootstrap CI.
7) Save predictions + a JSON summary.

Requirements (your conda env):
- pandas, numpy, pyyaml, joblib
- scikit-learn (any, but 1.3.2 recommended to avoid warnings)
- scikit-survival (sksurv)
"""

import argparse
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
import joblib

# scikit-survival
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import GradientBoostingSurvivalAnalysis


# -----------------------------
# utilities
# -----------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _read_text_from_zip(z: zipfile.ZipFile, inner_path: str) -> str:
    with z.open(inner_path, "r") as f:
        return f.read().decode("utf-8", errors="replace")

def _read_bytes_from_zip(z: zipfile.ZipFile, inner_path: str) -> bytes:
    with z.open(inner_path, "r") as f:
        return f.read()

def _load_joblib_from_zip(z: zipfile.ZipFile, inner_path: str):
    data = _read_bytes_from_zip(z, inner_path)
    # joblib.load expects a file-like object; BytesIO works.
    import io
    return joblib.load(io.BytesIO(data))

def _read_excel_from_zip(z: zipfile.ZipFile, inner_path: str) -> pd.DataFrame:
    data = _read_bytes_from_zip(z, inner_path)
    import io
    return pd.read_excel(io.BytesIO(data))

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def harrell_cindex(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    # event should be boolean where True indicates event occurred
    ev = event.astype(bool)
    ci = concordance_index_censored(ev, time.astype(float), risk.astype(float))[0]
    return float(ci)

def bootstrap_cindex(time: np.ndarray, event: np.ndarray, risk: np.ndarray,
                     n_boot: int = 200, seed: int = 42) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    n = len(time)
    vals = []
    idx = np.arange(n)
    for _ in range(int(n_boot)):
        samp = rng.choice(idx, size=n, replace=True)
        vals.append(harrell_cindex(time[samp], event[samp], risk[samp]))
    vals = np.asarray(vals, dtype=float)
    out = {
        "boot_mean": float(vals.mean()),
        "boot_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "ci95_lo": float(np.percentile(vals, 2.5)),
        "ci95_hi": float(np.percentile(vals, 97.5)),
    }
    return out


# -----------------------------
# clinical mapping (Radiogenomics -> Lung1 coded schema)
# NOTE: We code categories into integers because Model.zip normalizer expects ints.
# -----------------------------
def map_gender_to_code(x: str) -> Optional[int]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in ("male", "m"):
        return 0
    if s in ("female", "f"):
        return 1
    # treat "Not Collected" etc. as missing -> will be imputed by train_fill_values
    return None

def map_histology_to_code(x: str) -> Optional[int]:
    """
    Must match the training-time coding used by the zip pipeline.
    Based on observed OneHotEncoder categories_ for Histology: [0,1,2,3,4]
    We map to:
      0 = adenocarcinoma
      1 = large cell carcinoma
      2 = squamous cell carcinoma
      3 = other
      4 = nos / unknown bucket
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if "adeno" in s:
        return 0
    if "large" in s:
        return 1
    if "squamous" in s:
        return 2
    # "Not Collected" etc.
    if "not collected" in s or s == "" or s == "nan":
        return None
    return 3

def parse_t_stage_to_int_code(x: str) -> Optional[int]:
    """
    Lung1 pipeline expects clinical.T.Stage codes in {0,1,2,3,4,5}
    We'll map RG strings:
      Tis/T0 -> 0
      T1a/T1b/T1c/T1 -> 1
      T2a/T2b/T2 -> 2
      T3 -> 3
      T4 -> 4
      else -> None (will be imputed)
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    if s in ("NOT COLLECTED", "", "NAN"):
        return None
    # allow things like "T2a", "t2b", etc.
    if s.startswith("TIS") or s.startswith("T0"):
        return 0
    if s.startswith("T1"):
        return 1
    if s.startswith("T2"):
        return 2
    if s.startswith("T3"):
        return 3
    if s.startswith("T4"):
        return 4
    return None

def parse_n_stage_to_int_code(x: str) -> Optional[int]:
    """
    Lung1 pipeline expects Clinical.N.Stage codes in {0,1,2,3,4}
    Map RG:
      N0 -> 0, N1->1, N2->2, N3->3, else None
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    if s in ("NOT COLLECTED", "", "NAN"):
        return None
    if s.startswith("N0"):
        return 0
    if s.startswith("N1"):
        return 1
    if s.startswith("N2"):
        return 2
    if s.startswith("N3"):
        return 3
    return None

def parse_m_stage_to_int_code(x: str) -> Optional[int]:
    """
    Zip encoder categories for Clinical.M.Stage are [0,1,2]
    We'll map:
      M0 -> 0
      M1a/M1b/M1 -> 1
      unknown -> None (imputed)
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    if s in ("NOT COLLECTED", "", "NAN"):
        return None
    if s.startswith("M0"):
        return 0
    if s.startswith("M1"):
        return 1
    return None

def derive_overall_stage_code_from_tnm(t_code: Optional[int], n_code: Optional[int], m_code: Optional[int]) -> Optional[int]:
    """
    Zip encoder categories for Overall.Stage are [0,1,2,3] corresponding to:
      0=I, 1=II, 2=IIIa, 3=IIIb
    (no IV in training).

    We *exclude* known metastatic cases (M1) by default upstream; still, handle:
      if m_code == 1: return None  (will be imputed, but we usually filter those out)

    Very coarse TNM-to-stage grouping (approx, enough to avoid constant unknown):
      - N2 -> IIIa (unless T4 -> IIIb)
      - N1 -> II (unless T3/T4 -> IIIa)
      - N0 -> I for T0-1, II for T2-3, IIIa for T4
    If any code missing -> None (imputed by train fill, which in seed_91 is stage=3).
    """
    if t_code is None or n_code is None or m_code is None:
        return None
    if m_code == 1:
        return None
    # N3 not expected; if present treat as IIIb
    if n_code >= 3:
        return 3
    if n_code == 2:
        return 3 if t_code == 4 else 2
    if n_code == 1:
        return 2 if t_code >= 3 else 1
    # N0
    if t_code <= 1:
        return 0
    if t_code in (2, 3):
        return 1
    if t_code >= 4:
        return 2
    return None


# -----------------------------
# build matrices using zip artifacts
# -----------------------------
@dataclass
class ZipArtifacts:
    train_fill: Dict
    normalizer: Dict
    selected_features: List[str]
    gbsa_params: Dict
    exp_specs: Dict

def load_zip_artifacts(zip_path: str, task: str, seed: int, model: str) -> ZipArtifacts:
    """
    task: one of ["Clinic", "Radiomics", "Clinic+Radiomics"]
    model: e.g. "GradientBoostingSurvivalAnalysis"
    """
    seed_dir = f"{task}/seed_{seed}"
    model_dir = f"{seed_dir}/{model}"

    with zipfile.ZipFile(zip_path, "r") as z:
        # specs
        exp_specs = yaml.safe_load(_read_text_from_zip(z, f"{seed_dir}/exp_specs.yaml"))

        # preprocessing artifacts
        train_fill = _load_joblib_from_zip(z, f"{seed_dir}/train_fill_values.joblib")
        normalizer = _load_joblib_from_zip(z, f"{seed_dir}/normalizer.joblib")

        # selected features
        sel_xlsx_path = f"{model_dir}/selected_features_{model}.xlsx"
        sel_df = _read_excel_from_zip(z, sel_xlsx_path)
        # use primary list (final features); fall back to first column
        if "primary_selected_features" in sel_df.columns:
            feats = sel_df["primary_selected_features"].dropna().astype(str).tolist()
        else:
            feats = sel_df.iloc[:, 0].dropna().astype(str).tolist()

        # params
        params_path = f"{model_dir}/best_params_{model}.yaml"
        gbsa_params = yaml.safe_load(_read_text_from_zip(z, params_path))

    return ZipArtifacts(
        train_fill=train_fill,
        normalizer=normalizer,
        selected_features=feats,
        gbsa_params=gbsa_params,
        exp_specs=exp_specs,
    )

def apply_fill_values(df: pd.DataFrame, fill_values: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for col, val in fill_values.items():
        if col in out.columns:
            out[col] = out[col].fillna(val)
    return out

def build_design_matrix(df: pd.DataFrame, artifacts: ZipArtifacts) -> Tuple[np.ndarray, List[str]]:
    """
    Build full design matrix (numeric scaled + categorical one-hot) using
    fitted scaler/encoder from zip (so it's EXACT).

    Returns:
      X_all: numpy array [n, d_total]
      feature_names: list of length d_total
    """
    fill_values = artifacts.train_fill["fill_values"]
    df_filled = apply_fill_values(df, fill_values)

    # numeric
    num_cols = artifacts.normalizer["numeric_cols"]
    scaler = artifacts.normalizer["normalizers"]["numeric"]
    X_num = scaler.transform(df_filled[num_cols].astype(float).values)
    num_names = list(num_cols)

    # categorical
    cat_cols = artifacts.normalizer["category_columns"]
    encoder = artifacts.normalizer["normalizers"]["categorical"]

    # ensure integer-like dtypes (avoid 0.0 names); fill already handled
    # Check if any categorical columns contain string values and convert them
    df_cat = df_filled[cat_cols].copy()
    for col in cat_cols:
        if col in df_cat.columns:
            # If column is object type (string), try to convert to numeric
            if df_cat[col].dtype == 'object':
                # Try direct numeric conversion first
                df_cat[col] = pd.to_numeric(df_cat[col], errors='coerce')
                # Fill any NaN created by conversion with the fill value if available
                if col in fill_values:
                    df_cat[col] = df_cat[col].fillna(fill_values[col])
                else:
                    df_cat[col] = df_cat[col].fillna(0)
    
    X_cat_in = df_cat.astype(int).values
    X_cat = encoder.transform(X_cat_in)
    cat_names = encoder.get_feature_names_out(cat_cols).tolist()

    # concat
    X_all = np.concatenate([X_num, X_cat], axis=1)
    feat_names = num_names + cat_names
    return X_all, feat_names

def select_features(X_all: np.ndarray, feat_names: List[str], selected: List[str]) -> Tuple[np.ndarray, List[str]]:
    idx = []
    name_to_i = {n: i for i, n in enumerate(feat_names)}
    missing = []
    for f in selected:
        if f in name_to_i:
            idx.append(name_to_i[f])
        else:
            missing.append(f)
    if missing:
        raise ValueError(
            f"{len(missing)} selected features are missing from design matrix. "
            f"Examples: {missing[:10]}"
        )
    X_sel = X_all[:, idx]
    return X_sel, [feat_names[i] for i in idx]


# -----------------------------
# data loaders
# -----------------------------
def load_lung1_merged(clinical_csv: str, radiomics_csv: str) -> pd.DataFrame:
    clin = pd.read_csv(clinical_csv)
    rad = pd.read_csv(radiomics_csv)

    if "PatientID" not in clin.columns:
        raise ValueError("Lung1 clinical must contain PatientID column")
    if "PatientID" not in rad.columns:
        raise ValueError("Radiomics CSV must contain PatientID column")

    # Merge
    df = clin.merge(rad, on="PatientID", how="inner")

    # Minimal checks
    if "Survival.time" not in df.columns or "deadstatus.event" not in df.columns:
        raise ValueError("Lung1 clinical must contain Survival.time and deadstatus.event")

    # Drop invalid times
    df = df[df["Survival.time"].astype(float) > 0].copy()
    return df

def load_split_ids(split_json: str) -> Tuple[List[str], List[str]]:
    with open(split_json, "r") as f:
        sp = json.load(f)
    # support a few schemas
    train_ids = sp.get("train_ids") or sp.get("train") or sp.get("train_idx") or sp.get("train_index")
    test_ids = sp.get("test_ids") or sp.get("test") or sp.get("test_idx") or sp.get("test_index")
    if train_ids is None or test_ids is None:
        raise ValueError(f"Unrecognized split json schema: keys={list(sp.keys())}")
    # ids might be ints; convert to str for stable matching
    train_ids = [str(x) for x in train_ids]
    test_ids = [str(x) for x in test_ids]
    return train_ids, test_ids

def load_rg_external(clinical_aligned_csv: str, radiomics_csv: str,
                     drop_m1: bool = True,
                     id_prefix_from: str = "AMC",
                     id_prefix_to: str = "R01") -> pd.DataFrame:
    """
    Build RG merged dataframe with EXACT Lung1 column names expected by zip normalizer:
      numeric: age + 1688 radiomics
      categorical codes: gender, Histology, Overall.Stage, Clinical.M.Stage, Clinical.N.Stage, clinical.T.Stage
    plus time/event columns for scoring.

    Radiogenomics_clinical_aligned.csv is expected to have:
      patient_id, age, gender, histology, T, N, M, time_days, event, qc_time_nonpositive
    """
    rg = pd.read_csv(clinical_aligned_csv)
    rad = pd.read_csv(radiomics_csv)

    # map patient ids
    if "patient_id" not in rg.columns:
        raise ValueError("RG clinical aligned must contain patient_id")
    rg["PatientID"] = rg["patient_id"].astype(str).str.replace(f"{id_prefix_from}-", f"{id_prefix_to}-", regex=False)

    # Handle both "PatientID" and "patient_id" column names in radiomics CSV
    if "PatientID" not in rad.columns:
        if "patient_id" in rad.columns:
            rad["PatientID"] = rad["patient_id"].astype(str)
        else:
            raise ValueError("RG radiomics CSV must contain PatientID or patient_id")
    rad["PatientID"] = rad["PatientID"].astype(str)

    # survival
    if "time_days" not in rg.columns or "event" not in rg.columns:
        raise ValueError("RG clinical aligned must contain time_days and event columns")
    rg["Survival.time"] = rg["time_days"].astype(float)
    rg["deadstatus.event"] = rg["event"].astype(int)

    # filter invalid times
    rg = rg[rg["Survival.time"] > 0].copy()
    if "qc_time_nonpositive" in rg.columns:
        rg = rg[~rg["qc_time_nonpositive"].astype(bool)].copy()

    # clinical code columns expected by normalizer
    rg["age"] = rg["age"].apply(_safe_float)

    # gender / histology
    rg["gender"] = rg["gender"].apply(map_gender_to_code)
    rg["Histology"] = rg["histology"].apply(map_histology_to_code)

    # TNM codes
    rg["clinical.T.Stage"] = rg["T"].apply(parse_t_stage_to_int_code)
    rg["Clinical.N.Stage"] = rg["N"].apply(parse_n_stage_to_int_code)
    rg["Clinical.M.Stage"] = rg["M"].apply(parse_m_stage_to_int_code)

    # overall stage derived
    rg["Overall.Stage"] = [
        derive_overall_stage_code_from_tnm(t, n, m)
        for t, n, m in zip(rg["clinical.T.Stage"].tolist(),
                           rg["Clinical.N.Stage"].tolist(),
                           rg["Clinical.M.Stage"].tolist())
    ]

    # optionally drop metastatic cases (better match Lung1 which has no stage IV)
    if drop_m1:
        # keep rows where M is 0 or missing
        rg = rg[(rg["Clinical.M.Stage"].isna()) | (rg["Clinical.M.Stage"] == 0)].copy()

    # merge radiomics
    # (radiomics CSV should contain 1688 feature columns extracted from SAME YAML)
    df = rg.merge(rad, on="PatientID", how="inner")

    # Keep only columns needed by artifacts + survival cols
    return df


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip_path", type=str, required=True, help="Model.zip (experiment results)")
    ap.add_argument("--task", type=str, default="Clinic+Radiomics", choices=["Clinic", "Radiomics", "Clinic+Radiomics"])
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--model", type=str, default="GradientBoostingSurvivalAnalysis")

    ap.add_argument("--lung1_clinical_csv", type=str, required=True)
    ap.add_argument("--lung1_radiomics_csv", type=str, required=True)
    ap.add_argument("--split_json", type=str, required=True, help="split_ClinicplusRadiomics_*.json exported from zip")

    ap.add_argument("--rg_clinical_aligned_csv", type=str, required=True)
    ap.add_argument("--rg_radiomics_csv", type=str, required=True, help="1688 radiomics for Radiogenomics extracted with SAME YAML")

    ap.add_argument("--drop_m1", type=int, default=1, help="1 to exclude known metastatic (M1*) cases")
    ap.add_argument("--id_prefix_from", type=str, default="AMC")
    ap.add_argument("--id_prefix_to", type=str, default="R01")

    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    _ensure_dir(args.out_dir)

    # --- load artifacts ---
    art = load_zip_artifacts(args.zip_path, args.task, args.seed, args.model)

    # --- load Lung1 and split ---
    df_lung1 = load_lung1_merged(args.lung1_clinical_csv, args.lung1_radiomics_csv)
    train_ids, test_ids = load_split_ids(args.split_json)

    df_tr = df_lung1[df_lung1["PatientID"].astype(str).isin(train_ids)].copy()
    df_te = df_lung1[df_lung1["PatientID"].astype(str).isin(test_ids)].copy()

    if len(df_tr) == 0 or len(df_te) == 0:
        raise RuntimeError(f"Empty split after filtering. train={len(df_tr)} test={len(df_te)}")

    # --- build design matrices using EXACT zip preprocessing ---
    Xtr_all, feat_names = build_design_matrix(df_tr, art)
    Xte_all, _ = build_design_matrix(df_te, art)

    # --- select final features (primary_selected_features) ---
    Xtr, used_feats = select_features(Xtr_all, feat_names, art.selected_features)
    Xte, _ = select_features(Xte_all, feat_names, art.selected_features)

    # --- y ---
    ytr = Surv.from_arrays(event=df_tr["deadstatus.event"].astype(bool).values,
                           time=df_tr["Survival.time"].astype(float).values)
    yte = Surv.from_arrays(event=df_te["deadstatus.event"].astype(bool).values,
                           time=df_te["Survival.time"].astype(float).values)

    # --- fit GBSA with best params ---
    # Map yaml params to constructor; ignore unknown keys safely.
    params = dict(art.gbsa_params)
    # ensure deterministic
    params["random_state"] = int(args.seed)

    # sklearn-style: max_features could be 'auto' or float; keep as is.
    gbsa = GradientBoostingSurvivalAnalysis(**params)
    gbsa.fit(Xtr, ytr)

    # --- internal sanity check ---
    risk_te = gbsa.predict(Xte)
    internal_ci = harrell_cindex(df_te["Survival.time"].values,
                                 df_te["deadstatus.event"].values,
                                 risk_te)
    internal_boot = bootstrap_cindex(df_te["Survival.time"].values,
                                     df_te["deadstatus.event"].values,
                                     risk_te,
                                     n_boot=args.bootstrap,
                                     seed=42)

    print(f"[INTERNAL] task={args.task} seed={args.seed} model={args.model}")
    print(f"  n_train={len(df_tr)} n_test={len(df_te)} n_features={Xtr.shape[1]}")
    print(f"  test_cindex={internal_ci:.6f}  boot_mean±std={internal_boot['boot_mean']:.6f}±{internal_boot['boot_std']:.6f}  CI95=[{internal_boot['ci95_lo']:.6f},{internal_boot['ci95_hi']:.6f}]")

    # --- load external RG ---
    df_rg = load_rg_external(
        clinical_aligned_csv=args.rg_clinical_aligned_csv,
        radiomics_csv=args.rg_radiomics_csv,
        drop_m1=bool(args.drop_m1),
        id_prefix_from=args.id_prefix_from,
        id_prefix_to=args.id_prefix_to,
    )
    if len(df_rg) == 0:
        raise RuntimeError("No external samples after merging clinical+radiomics. Check PatientID mapping and radiomics CSV.")

    # transform external
    Xrg_all, feat_names2 = build_design_matrix(df_rg, art)
    # feature name order must match
    if feat_names2 != feat_names:
        # This should never happen if we used the same artifacts; but assert to be safe.
        raise RuntimeError("Feature name mismatch between internal/external transform. Something is inconsistent.")

    Xrg, _ = select_features(Xrg_all, feat_names, art.selected_features)

    # predict & score
    risk_rg = gbsa.predict(Xrg)
    external_ci = harrell_cindex(df_rg["Survival.time"].values,
                                 df_rg["deadstatus.event"].values,
                                 risk_rg)
    external_boot = bootstrap_cindex(df_rg["Survival.time"].values,
                                     df_rg["deadstatus.event"].values,
                                     risk_rg,
                                     n_boot=args.bootstrap,
                                     seed=42)

    print(f"[EXTERNAL] Radiogenomics")
    print(f"  n_ext={len(df_rg)} events={int(df_rg['deadstatus.event'].sum())} event_rate={df_rg['deadstatus.event'].mean():.3f}")
    print(f"  cindex={external_ci:.6f}  boot_mean±std={external_boot['boot_mean']:.6f}±{external_boot['boot_std']:.6f}  CI95=[{external_boot['ci95_lo']:.6f},{external_boot['ci95_hi']:.6f}]")

    # save preds
    out_pred = df_rg[["PatientID", "Survival.time", "deadstatus.event"]].copy()
    out_pred["risk"] = risk_rg
    pred_path = os.path.join(args.out_dir, "radiogenomics_preds.csv")
    out_pred.to_csv(pred_path, index=False)

    # save summary
    summary = {
        "zip_path": args.zip_path,
        "task": args.task,
        "seed": int(args.seed),
        "model": args.model,
        "n_train": int(len(df_tr)),
        "n_test": int(len(df_te)),
        "n_features": int(Xtr.shape[1]),
        "internal_test_cindex": float(internal_ci),
        "internal_boot": internal_boot,
        "n_external": int(len(df_rg)),
        "external_cindex": float(external_ci),
        "external_boot": external_boot,
        "drop_m1": bool(args.drop_m1),
        "id_prefix_from": args.id_prefix_from,
        "id_prefix_to": args.id_prefix_to,
        "used_features": art.selected_features,
    }
    sum_path = os.path.join(args.out_dir, "external_validation_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Saved: {pred_path}")
    print(f"[OK] Saved: {sum_path}")


if __name__ == "__main__":
    main()
