
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_multimodal_stack_v1.py

Goal: get a *real* lift in C-index on a fixed split by adding CT deep features (embedding/risk)
to the tabular radiomics+clinical pipeline, without doing any best-of-N cheating.

What this script does (single fixed split):
1) Load clinical table (LUNG1) + radiomics table (107 or 1130 features).
2) Optionally load deep embeddings (e.g., ResNet10 avgpool 512-d) as a CSV keyed by PatientID.
3) Build a simple clinical feature set (age + gender + Overall.Stage by default).
4) Train:
   - Coxnet on (clinical + radiomics)
   - Coxnet on (deep only)   [if deep_csv provided]
   - Coxnet on (clinical + radiomics + deep)  [if deep_csv provided]
   - DeepSurv (full-batch Cox loss) on (clinical + radiomics + deep)  [optional, if torch available]
   - Stacked CoxPH on risk scores (late fusion)  [if deep_csv provided]
5) Report train/val/test Harrell C-index and save artifacts.

Key design choices:
- NO CV required. We only use the provided fixed split (train/val/test).
- ALL preprocessing statistics (imputation/scaling) are fit on TRAIN only.
- Clinical preprocessing is intentionally simple and literature-aligned:
  age median-impute; gender binary; stage one-hot; (no age_is_missing).
- M-stage handling: default is binary_nonzero (M!=0 -> 1). This treats the "3" code as non-zero/unknown
  without claiming it is true metastasis.

You will NOT get to 0.70+ on Lung1 hold-out reliably with radiomics+clinical only.
If you still want 0.70+, you must feed the model *new signal* (deep features / end-to-end imaging).
This script is the lowest-effort way to do that.

Example:
python step2_multimodal_stack_v1.py \
  --clinical_csv NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
  --radiomics_csv radiomics_features/Tb_features_107.csv \
  --deep_csv deep_features/deep_resnet10_512.csv \
  --split_json splits/fixed_split_seed42.json \
  --out_dir step2_mm_seed42 \
  --radiomics_mode all \
  --deep_mode pca --deep_pca_dim 64 \
  --use_deepsurv 1

Deep CSV format:
PatientID, d0, d1, ... d511
(Any column names are fine; all non-PatientID columns are treated as deep features.)
"""

import os
import json
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

# sklearn compatibility (OneHotEncoder API changed: sparse -> sparse_output)
def make_ohe(categories):
    try:
        return OneHotEncoder(categories=categories, sparse_output=False, handle_unknown='ignore')
    except TypeError:
        return OneHotEncoder(categories=categories, sparse=False, handle_unknown='ignore')


# scikit-survival
try:
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored
    from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
except Exception as e:
    raise ImportError(
        "This script requires scikit-survival (sksurv). "
        "Install e.g. `pip install scikit-survival` following your OS instructions."
    ) from e

# torch optional (for DeepSurv)
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_split_ids(split_json: str) -> Dict[str, List[str]]:
    with open(split_json, "r") as f:
        obj = json.load(f)
    # Support both formats:
    # 1) {"train":[...], "val":[...], "test":[...], "meta": {...}}
    # 2) {"splits": {"train":[...], "val":[...], "test":[...]}, ...}
    if "train" in obj and "val" in obj and "test" in obj:
        return {
            "train": [str(x) for x in obj["train"]],
            "val": [str(x) for x in obj["val"]],
            "test": [str(x) for x in obj["test"]],
        }
    if "splits" in obj and all(k in obj["splits"] for k in ["train", "val", "test"]):
        sp = obj["splits"]
        return {"train": [str(x) for x in sp["train"]], "val": [str(x) for x in sp["val"]], "test": [str(x) for x in sp["test"]]}
    raise ValueError(f"Unrecognized split JSON format: {split_json}")


def harrell_cindex(y_struct, risk: np.ndarray) -> float:
    # concordance_index_censored expects event_indicator=True for event.
    ev = y_struct["event"]
    tm = y_struct["time"]
    # risk: higher = worse survival
    c = concordance_index_censored(ev, tm, risk)[0]
    return float(c)


def canonicalize_stage(x) -> str:
    """
    Normalize Overall.Stage strings to {I, II, IIIA, IIIB, IV, UNK}.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "UNK"
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "nan", "none"}:
        return "UNK"
    s = s.upper().replace(" ", "")
    # common variants
    s = s.replace("STAGE", "")
    s = s.replace("IIIA", "IIIA").replace("IIIB", "IIIB")  # no-op but harmless
    if s in {"0", "TIS", "IS", "IN-SITU"}:
        return "I"  # treat as earliest for stratification/encoding
    if s in {"I", "IA", "IB"}:
        return "I"
    if s in {"II", "IIA", "IIB"}:
        return "II"
    if s in {"III", "IIIA"}:
        return "IIIA"
    if s in {"IIIB"}:
        return "IIIB"
    if s in {"IV", "IVA", "IVB"}:
        return "IV"
    return "UNK"


def map_gender_binary(x) -> int:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0
    s = str(x).strip().lower()
    if s.startswith("m"):
        return 1
    if s.startswith("f"):
        return 0
    return 0


def mstage_binary_nonzero(x) -> int:
    """
    Treat any non-zero M code as 1. This merges the dataset's rare "3" code with the single "1".
    Clinically: '3' is not a valid TNM M category; it's usually an unknown/missing encoding (Mx-like).
    Modeling-wise: we keep a single indicator M_nonzero_or_unknown.
    """
    try:
        v = float(x)
        if np.isnan(v):
            return 0
        return 0 if int(v) == 0 else 1
    except Exception:
        s = str(x).strip()
        if s == "" or s.lower() in {"na", "nan"}:
            return 0
        return 0 if s == "0" else 1


@dataclass
class DesignMatrices:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]
    n_clin: int
    n_rad: int
    n_deep: int


# -----------------------------
# Build features
# -----------------------------
def build_clinical_features(
    clin_df: pd.DataFrame,
    split_ids: Dict[str, List[str]],
    clinical_mode: str = "minimal",
    mstage_mode: str = "binary_nonzero",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    clinical_mode:
      - minimal: age + gender + Overall.Stage(one-hot)
      - tn: minimal + T_stage(one-hot) + N_stage(one-hot)
      - all: tn + M_stage (binary_nonzero)  (histology excluded by design)
    """
    df = clin_df.copy()
    df["PatientID"] = df["PatientID"].astype(str)
    df = df.set_index("PatientID")

    # basic columns
    df["age_num"] = pd.to_numeric(df.get("age"), errors="coerce")
    df["gender_bin"] = df.get("gender").apply(map_gender_binary)

    df["stage_cat"] = df.get("Overall.Stage").apply(canonicalize_stage)

    # Impute age using TRAIN median (user asked to keep it simple; no missing indicator).
    train_ids = [pid for pid in split_ids["train"] if pid in df.index]
    age_median = float(df.loc[train_ids, "age_num"].median())
    df["age_num"] = df["age_num"].fillna(age_median)

    out_parts = []
    feat_names = []

    # continuous age (rename age_num to age for consistency)
    age_df = df[["age_num"]].copy()
    age_df.columns = ["age"]
    out_parts.append(age_df)
    feat_names.append("age")

    # binary gender
    out_parts.append(df[["gender_bin"]])
    feat_names.append("gender_male")

    # stage one-hot (fit categories globally to avoid "unknown category -> all zeros" surprises)
    stage_cats = ["I", "II", "IIIA", "IIIB", "IV", "UNK"]
    stage_enc = make_ohe([stage_cats])
    stage_ohe = stage_enc.fit_transform(df[["stage_cat"]])
    stage_cols = [f"stage_{c}" for c in stage_cats]
    out_parts.append(pd.DataFrame(stage_ohe, index=df.index, columns=stage_cols))
    feat_names.extend(stage_cols)

    if clinical_mode in {"tn", "all"}:
        # Treat T/N as categorical to avoid imposing linearity.
        # The dataset uses numeric codes; some rare codes (e.g., 5 for T, 4 for N) behave like unknown.
        t_raw = pd.to_numeric(df.get("clinical.T.Stage"), errors="coerce")
        n_raw = pd.to_numeric(df.get("Clinical.N.Stage"), errors="coerce")

        # mode-impute missing on TRAIN
        if t_raw.isna().any():
            t_mode = df.loc[train_ids, "clinical.T.Stage"].mode(dropna=True)
            t_fill = float(t_mode.iloc[0]) if len(t_mode) else 0.0
            t_raw = t_raw.fillna(t_fill)
        if n_raw.isna().any():
            n_mode = df.loc[train_ids, "Clinical.N.Stage"].mode(dropna=True)
            n_fill = float(n_mode.iloc[0]) if len(n_mode) else 0.0
            n_raw = n_raw.fillna(n_fill)

        # map rare out-of-range to UNK bucket
        def t_bucket(v: float) -> str:
            if pd.isna(v):
                return "UNK"
            iv = int(v)
            if iv in {0, 1, 2, 3, 4}:  # common
                return str(iv)
            return "UNK"

        def n_bucket(v: float) -> str:
            if pd.isna(v):
                return "UNK"
            iv = int(v)
            if iv in {0, 1, 2, 3}:
                return str(iv)
            return "UNK"

        df["t_cat"] = t_raw.apply(t_bucket)
        df["n_cat"] = n_raw.apply(n_bucket)

        t_cats = ["0", "1", "2", "3", "4", "UNK"]
        n_cats = ["0", "1", "2", "3", "UNK"]

        t_enc = make_ohe([t_cats])
        n_enc = make_ohe([n_cats])
        t_ohe = t_enc.fit_transform(df[["t_cat"]])
        n_ohe = n_enc.fit_transform(df[["n_cat"]])

        t_cols = [f"T_{c}" for c in t_cats]
        n_cols = [f"N_{c}" for c in n_cats]

        out_parts.append(pd.DataFrame(t_ohe, index=df.index, columns=t_cols))
        out_parts.append(pd.DataFrame(n_ohe, index=df.index, columns=n_cols))
        feat_names.extend(t_cols + n_cols)

    if clinical_mode == "all":
        if mstage_mode == "drop":
            pass
        elif mstage_mode == "binary_nonzero":
            df["m_bin"] = df.get("Clinical.M.Stage").apply(mstage_binary_nonzero)
            out_parts.append(df[["m_bin"]])
            feat_names.append("M_nonzero_or_unknown")
        else:
            raise ValueError(f"Unknown mstage_mode: {mstage_mode}")

    X_clin = pd.concat(out_parts, axis=1)
    return X_clin, feat_names


def build_surv_targets(clin_df: pd.DataFrame) -> pd.DataFrame:
    df = clin_df.copy()
    df["PatientID"] = df["PatientID"].astype(str)
    df = df.set_index("PatientID")
    time = pd.to_numeric(df["Survival.time"], errors="coerce").astype(float)
    event = pd.to_numeric(df["deadstatus.event"], errors="coerce").fillna(0).astype(int)
    out = pd.DataFrame({"time": time, "event": event.astype(bool)}, index=df.index)
    return out


def read_radiomics(radiomics_csv: str) -> pd.DataFrame:
    rad = pd.read_csv(radiomics_csv)
    if "PatientID" not in rad.columns:
        raise ValueError("Radiomics CSV must contain a PatientID column.")
    rad["PatientID"] = rad["PatientID"].astype(str)
    rad = rad.set_index("PatientID")
    # force numeric
    for c in rad.columns:
        rad[c] = pd.to_numeric(rad[c], errors="coerce")
    return rad


def read_deep(deep_csv: str) -> pd.DataFrame:
    d = pd.read_csv(deep_csv)
    if "PatientID" not in d.columns:
        raise ValueError("Deep CSV must contain a PatientID column.")
    d["PatientID"] = d["PatientID"].astype(str)
    d = d.set_index("PatientID")
    # numeric
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def corr_filter(df: pd.DataFrame, threshold: float) -> List[str]:
    """
    Simple greedy correlation filter on TRAIN only.
    Keep the first feature in an ordering; drop later features with |corr| > threshold.
    """
    cols = list(df.columns)
    if len(cols) <= 1:
        return cols
    X = df.values
    # compute corr matrix (nan-safe)
    corr = np.corrcoef(X, rowvar=False)
    keep = []
    dropped = set()
    for i, c in enumerate(cols):
        if c in dropped:
            continue
        keep.append(c)
        # drop j>i if highly correlated
        for j in range(i + 1, len(cols)):
            if cols[j] in dropped:
                continue
            if np.isfinite(corr[i, j]) and abs(corr[i, j]) > threshold:
                dropped.add(cols[j])
    return keep


def univariate_topk_by_cindex(
    X_train: pd.DataFrame,
    y_train_struct,
    k: int
) -> List[str]:
    """
    Rank by |cindex - 0.5| so both harmful and protective features are kept.
    """
    if k <= 0 or k >= X_train.shape[1]:
        return list(X_train.columns)
    scores = []
    for c in X_train.columns:
        x = X_train[c].values.astype(float)
        # if constant, score 0
        if np.nanstd(x) < 1e-12:
            scores.append((0.0, c))
            continue
        # treat x as risk (higher worse). We take abs(c-0.5) to be sign-invariant.
        try:
            ci = harrell_cindex(y_train_struct, x)
        except Exception:
            ci = 0.5
        scores.append((abs(ci - 0.5), c))
    scores.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scores[:k]]


def prepare_design(
    clinical_csv: str,
    radiomics_csv: str,
    split_json: str,
    out_dir: str,
    deep_csv: Optional[str],
    clinical_mode: str,
    mstage_mode: str,
    radiomics_mode: str,
    radiomics_topk: int,
    corr_threshold: float,
    deep_mode: str,
    deep_pca_dim: int,
) -> DesignMatrices:
    ensure_dir(out_dir)
    split_ids = load_split_ids(split_json)

    clin = pd.read_csv(clinical_csv)
    clin["PatientID"] = clin["PatientID"].astype(str)

    y_df = build_surv_targets(clin)
    X_clin, clin_feat_names = build_clinical_features(clin, split_ids, clinical_mode=clinical_mode, mstage_mode=mstage_mode)

    rad = read_radiomics(radiomics_csv)

    # align indices (intersection)
    common = X_clin.index.intersection(rad.index).intersection(y_df.index)
    X_clin = X_clin.loc[common]
    rad = rad.loc[common]
    y_df = y_df.loc[common]

    # optional deep
    deep = None
    if deep_csv is not None:
        deep = read_deep(deep_csv)
        common = common.intersection(deep.index)
        X_clin = X_clin.loc[common]
        rad = rad.loc[common]
        y_df = y_df.loc[common]
        deep = deep.loc[common]

    # split indices
    train_ids = [pid for pid in split_ids["train"] if pid in common]
    val_ids = [pid for pid in split_ids["val"] if pid in common]
    test_ids = [pid for pid in split_ids["test"] if pid in common]

    # build y structs
    y_train_struct = Surv.from_arrays(event=y_df.loc[train_ids, "event"].values, time=y_df.loc[train_ids, "time"].values)
    y_val_struct   = Surv.from_arrays(event=y_df.loc[val_ids, "event"].values, time=y_df.loc[val_ids, "time"].values)
    y_test_struct  = Surv.from_arrays(event=y_df.loc[test_ids, "event"].values, time=y_df.loc[test_ids, "time"].values)

    # radiomics selection on TRAIN only
    rad_train = rad.loc[train_ids]
    if radiomics_mode == "all":
        rad_cols = list(rad.columns)
    elif radiomics_mode == "topk":
        rad_cols = univariate_topk_by_cindex(rad_train, y_train_struct, radiomics_topk)
    elif radiomics_mode == "highest10":
        # ten signatures from the DeepSurv+deep-radiomics paper (Le et al., 2025)
        # Note: PyRadiomics uses hyphen (-) in wavelet feature names, not underscore (_)
        rad_cols = [
            "original_gldm_DependenceVariance",
            "original_firstorder_Energy",
            "original_glszm_GrayLevelNonUniformityNormalized",
            "wavelet-HLL_glcm_Correlation",
            "wavelet-LHL_glcm_MCC",
            "wavelet-LHL_firstorder_Skewness",
            "wavelet-LHH_firstorder_MeanAbsoluteDeviation",
            "wavelet-LLH_glcm_ClusterShade",
            "wavelet-LLH_firstorder_Maximum",
            "wavelet-HHH_firstorder_Energy",
        ]
        missing = [c for c in rad_cols if c not in rad.columns]
        if missing:
            raise ValueError(f"highest10 mode: missing columns in radiomics table: {missing}")
    else:
        raise ValueError(f"Unknown radiomics_mode: {radiomics_mode}")

    rad_sel = rad[rad_cols].copy()

    # impute radiomics (TRAIN median per feature)
    rad_median = rad_sel.loc[train_ids].median(axis=0)
    rad_sel = rad_sel.fillna(rad_median)

    # correlation filter (TRAIN)
    if corr_threshold is not None and 0.0 < corr_threshold < 1.0:
        keep_cols = corr_filter(rad_sel.loc[train_ids], corr_threshold)
        rad_sel = rad_sel[keep_cols]
        rad_cols = keep_cols

    # scale radiomics on TRAIN
    rad_scaler = StandardScaler()
    rad_scaler.fit(rad_sel.loc[train_ids].values)
    rad_scaled = pd.DataFrame(rad_scaler.transform(rad_sel.values), index=rad_sel.index, columns=[f"rad_{c}" for c in rad_cols])

    # scale clinical *continuous* columns (age only). We keep one-hot/binary as-is.
    # Identify clinical continuous columns by name.
    clin_cont_cols = ["age"]
    clin_cont_present = [c for c in clin_cont_cols if c in X_clin.columns]
    clin_bin_ohe_cols = [c for c in X_clin.columns if c not in clin_cont_present]

    # Handle case where no continuous features exist
    if len(clin_cont_present) > 0:
        clin_scaler = StandardScaler()
        clin_scaler.fit(X_clin.loc[train_ids, clin_cont_present].values)
        clin_cont_scaled = pd.DataFrame(
            clin_scaler.transform(X_clin[clin_cont_present].values),
            index=X_clin.index,
            columns=[f"clin_{c}" for c in clin_cont_present],
        )
    else:
        clin_cont_scaled = pd.DataFrame(index=X_clin.index)
    
    clin_rest = X_clin[clin_bin_ohe_cols].copy()
    clin_rest.columns = [f"clin_{c}" for c in clin_rest.columns]

    # deep features
    deep_block = None
    deep_feat_names = []
    if deep is not None:
        deep_df = deep.copy()
        # impute deep (TRAIN median)
        deep_median = deep_df.loc[train_ids].median(axis=0)
        deep_df = deep_df.fillna(deep_median)

        if deep_mode == "all":
            deep_used = deep_df
            deep_feat_names = [f"deep_{c}" for c in deep_used.columns]
        elif deep_mode == "pca":
            # PCA fitted on TRAIN
            pca = PCA(n_components=deep_pca_dim, random_state=42)
            pca.fit(deep_df.loc[train_ids].values)
            Z = pca.transform(deep_df.values)
            deep_used = pd.DataFrame(Z, index=deep_df.index, columns=[f"deep_pca{i:03d}" for i in range(Z.shape[1])])
            deep_feat_names = list(deep_used.columns)
        else:
            raise ValueError(f"Unknown deep_mode: {deep_mode}")

        deep_scaler = StandardScaler()
        deep_scaler.fit(deep_used.loc[train_ids].values)
        deep_block = pd.DataFrame(
            deep_scaler.transform(deep_used.values),
            index=deep_used.index,
            columns=deep_feat_names,
        )

    # final design matrices
    blocks = [clin_cont_scaled, clin_rest, rad_scaled]
    feat_names = list(clin_cont_scaled.columns) + list(clin_rest.columns) + list(rad_scaled.columns)

    if deep_block is not None:
        blocks.append(deep_block)
        feat_names += list(deep_block.columns)

    n_clin = clin_cont_scaled.shape[1] + clin_rest.shape[1]
    n_rad = rad_scaled.shape[1]
    n_deep = 0 if deep_block is None else deep_block.shape[1]

    X_all = pd.concat(blocks, axis=1)
    X_train = X_all.loc[train_ids].values.astype(np.float32)
    X_val = X_all.loc[val_ids].values.astype(np.float32)
    X_test = X_all.loc[test_ids].values.astype(np.float32)

    # save feature list
    with open(os.path.join(out_dir, "feature_names.txt"), "w") as f:
        for n in feat_names:
            f.write(n + "\n")

    # save selected radiomics list
    with open(os.path.join(out_dir, "selected_radiomics_cols.txt"), "w") as f:
        for n in rad_cols:
            f.write(n + "\n")

    return DesignMatrices(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train_struct, y_val=y_val_struct, y_test=y_test_struct,
        feature_names=feat_names,
        train_ids=train_ids, val_ids=val_ids, test_ids=test_ids,
        n_clin=int(n_clin), n_rad=int(n_rad), n_deep=int(n_deep),
    )


# -----------------------------
# Models
# -----------------------------
def fit_coxnet_select_by_val(
    X_train: np.ndarray,
    y_train,
    X_val: np.ndarray,
    y_val,
    l1_ratios: List[float],
    n_alphas: int = 80,
    alpha_min_ratio: float = 1e-4,
) -> Tuple[CoxnetSurvivalAnalysis, Dict]:
    """
    Train Coxnet for multiple l1_ratios; select alpha by best val C-index.
    Returns fitted best model and a dict of selected params.
    """
    best = {"val_cindex": -np.inf, "l1_ratio": None, "alpha": None, "model": None}
    for l1 in l1_ratios:
        model = CoxnetSurvivalAnalysis(
            l1_ratio=l1,
            alphas=None,  # let it generate path
            n_alphas=n_alphas,
            alpha_min_ratio=alpha_min_ratio,
            max_iter=100000,
        )
        model.fit(X_train, y_train)
        # model.coef_ shape: (n_features, n_alphas)
        # model.alphas_ length n_alphas
        # Evaluate each alpha on VAL
        risks = model.predict(X_val)  # for last alpha by default? Actually predict uses final model?
        # In sksurv Coxnet, predict uses coefficients at last alpha by default (I believe).
        # We need per-alpha predictions; implement manually using coef_path and intercept.
        coef_path = model.coef_.T  # (n_alphas, n_features)
        # CoxnetSurvivalAnalysis in sksurv does not have intercept; linear predictor = X @ beta
        Xv = X_val
        yv = y_val
        for ai, alpha in enumerate(model.alphas_):
            beta = coef_path[ai]
            rv = Xv @ beta
            ci = harrell_cindex(yv, rv)
            if ci > best["val_cindex"]:
                best.update({"val_cindex": ci, "l1_ratio": l1, "alpha": float(alpha), "beta": beta, "model": model})
    # build a final model object with chosen l1_ratio and alpha only (refit on TRAIN)
    final = CoxnetSurvivalAnalysis(
        l1_ratio=best["l1_ratio"],
        alphas=[best["alpha"]],
        max_iter=100000,
    )
    final.fit(X_train, y_train)
    info = {"l1_ratio": best["l1_ratio"], "alpha": best["alpha"], "val_cindex": best["val_cindex"]}
    return final, info


class DeepSurvMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int] = [128, 64], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """
    Full-batch negative log partial likelihood (Breslow ties not handled explicitly).
    time: shape (n,)
    event: shape (n,) float 0/1
    """
    # sort by decreasing time so risk set is prefix
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    # log cumulative sum exp for risk set
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    # contributions for events
    loss = -(risk - log_cumsum) * event
    return loss.sum() / (event.sum().clamp_min(1.0))


def train_deepsurv(
    X_train: np.ndarray, y_train,
    X_val: np.ndarray, y_val,
    seed: int = 42,
    hidden: List[int] = [128, 64],
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 500,
    patience: int = 30,
) -> Tuple[DeepSurvMLP, Dict]:
    if torch is None:
        raise RuntimeError("torch is not available, cannot train DeepSurv.")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepSurvMLP(X_train.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xt = torch.from_numpy(X_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)

    tt = torch.from_numpy(y_train["time"].astype(np.float32)).to(device)
    et = torch.from_numpy(y_train["event"].astype(np.float32)).to(device)
    tv = torch.from_numpy(y_val["time"].astype(np.float32)).to(device)
    ev = torch.from_numpy(y_val["event"].astype(np.float32)).to(device)

    best_state = None
    best_val = -np.inf
    best_epoch = -1
    bad = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        risk = model(Xt)
        loss = cox_ph_loss(risk, tt, et)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            rv = model(Xv).detach().cpu().numpy()
        ci = harrell_cindex(y_val, rv)

        if ci > best_val + 1e-4:
            best_val = ci
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    info = {"val_cindex": float(best_val), "best_epoch": int(best_epoch), "epochs_ran": int(epoch)}
    return model, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_csv", required=True)
    ap.add_argument("--radiomics_csv", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--deep_csv", default=None, help="Optional deep embedding CSV with PatientID column.")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--clinical_mode", default="minimal", choices=["minimal", "tn", "all"])
    ap.add_argument("--mstage_mode", default="binary_nonzero", choices=["binary_nonzero", "drop"])

    ap.add_argument("--radiomics_mode", default="all", choices=["all", "topk", "highest10"])
    ap.add_argument("--radiomics_topk", type=int, default=50)
    ap.add_argument("--corr_threshold", type=float, default=0.95)

    ap.add_argument("--deep_mode", default="pca", choices=["all", "pca"])
    ap.add_argument("--deep_pca_dim", type=int, default=64)

    ap.add_argument("--use_deepsurv", type=int, default=1)

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    set_seed(args.seed)

    design = prepare_design(
        clinical_csv=args.clinical_csv,
        radiomics_csv=args.radiomics_csv,
        split_json=args.split_json,
        out_dir=args.out_dir,
        deep_csv=args.deep_csv,
        clinical_mode=args.clinical_mode,
        mstage_mode=args.mstage_mode,
        radiomics_mode=args.radiomics_mode,
        radiomics_topk=args.radiomics_topk,
        corr_threshold=args.corr_threshold,
        deep_mode=args.deep_mode if args.deep_csv else "all",
        deep_pca_dim=args.deep_pca_dim,
    )

    Xtr, Xva, Xte = design.X_train, design.X_val, design.X_test
    ytr, yva, yte = design.y_train, design.y_val, design.y_test

    print("=== Fixed-split multimodal benchmark (stack v1) ===")
    print(f"Train/Val/Test: {Xtr.shape[0]}/{Xva.shape[0]}/{Xte.shape[0]} | dim={Xtr.shape[1]}")
    print(f"Deep provided: {args.deep_csv is not None}")

    results = {}

    # Coxnet on full design
    coxnet, info = fit_coxnet_select_by_val(
        Xtr, ytr, Xva, yva, l1_ratios=[0.05, 0.1, 0.3, 0.5, 0.8], n_alphas=80, alpha_min_ratio=1e-4
    )
    r_tr = coxnet.predict(Xtr)
    r_va = coxnet.predict(Xva)
    r_te = coxnet.predict(Xte)
    ci_tr = harrell_cindex(ytr, r_tr)
    ci_va = harrell_cindex(yva, r_va)
    ci_te = harrell_cindex(yte, r_te)
    print(f"[coxnet_fused] train={ci_tr:.4f} val={ci_va:.4f} test={ci_te:.4f} (l1={info['l1_ratio']}, alpha={info['alpha']:.3g})")
    results["coxnet_fused"] = {"train": ci_tr, "val": ci_va, "test": ci_te, **info}

    # ------------------------------------------------------------
    # Modality-specific baselines + honest late fusion (if deep exists)
    # ------------------------------------------------------------
    if design.n_deep > 0:
        n_tab = design.n_clin + design.n_rad
        Xtr_tab, Xva_tab, Xte_tab = Xtr[:, :n_tab], Xva[:, :n_tab], Xte[:, :n_tab]
        Xtr_deep, Xva_deep, Xte_deep = Xtr[:, n_tab:], Xva[:, n_tab:], Xte[:, n_tab:]

        # Tabular (clinical + radiomics)
        tab_model, tab_info = fit_coxnet_select_by_val(
            Xtr_tab, ytr, Xva_tab, yva, l1_ratios=[0.05, 0.1, 0.3, 0.5, 0.8], n_alphas=80, alpha_min_ratio=1e-4
        )
        tab_r_tr = tab_model.predict(Xtr_tab)
        tab_r_va = tab_model.predict(Xva_tab)
        tab_r_te = tab_model.predict(Xte_tab)
        tab_ci_tr = harrell_cindex(ytr, tab_r_tr)
        tab_ci_va = harrell_cindex(yva, tab_r_va)
        tab_ci_te = harrell_cindex(yte, tab_r_te)
        print(f"[coxnet_tab ] train={tab_ci_tr:.4f} val={tab_ci_va:.4f} test={tab_ci_te:.4f} (l1={tab_info['l1_ratio']}, alpha={tab_info['alpha']:.3g})")
        results["coxnet_tabular"] = {"train": tab_ci_tr, "val": tab_ci_va, "test": tab_ci_te, **tab_info}

        # Deep-only (embedding)
        deep_model, deep_info = fit_coxnet_select_by_val(
            Xtr_deep, ytr, Xva_deep, yva, l1_ratios=[0.05, 0.1, 0.3, 0.5, 0.8], n_alphas=80, alpha_min_ratio=1e-4
        )
        deep_r_tr = deep_model.predict(Xtr_deep)
        deep_r_va = deep_model.predict(Xva_deep)
        deep_r_te = deep_model.predict(Xte_deep)
        deep_ci_tr = harrell_cindex(ytr, deep_r_tr)
        deep_ci_va = harrell_cindex(yva, deep_r_va)
        deep_ci_te = harrell_cindex(yte, deep_r_te)
        print(f"[coxnet_deep] train={deep_ci_tr:.4f} val={deep_ci_va:.4f} test={deep_ci_te:.4f} (l1={deep_info['l1_ratio']}, alpha={deep_info['alpha']:.3g})")
        results["coxnet_deep"] = {"train": deep_ci_tr, "val": deep_ci_va, "test": deep_ci_te, **deep_info}

        # Late-fusion: weighted average of standardized risks, weight chosen on VAL
        def zscore(train_r, r):
            mu = float(np.mean(train_r))
            sd = float(np.std(train_r) + 1e-8)
            return (r - mu) / sd

        tab_tr_z, tab_va_z, tab_te_z = zscore(tab_r_tr, tab_r_tr), zscore(tab_r_tr, tab_r_va), zscore(tab_r_tr, tab_r_te)
        deep_tr_z, deep_va_z, deep_te_z = zscore(deep_r_tr, deep_r_tr), zscore(deep_r_tr, deep_r_va), zscore(deep_r_tr, deep_r_te)

        best_w, best_ci = 0.5, -np.inf
        for w in np.linspace(0.0, 1.0, 101):
            r_va_mix = w * tab_va_z + (1.0 - w) * deep_va_z
            ci = harrell_cindex(yva, r_va_mix)
            if ci > best_ci:
                best_ci = ci
                best_w = float(w)

        r_tr_mix = best_w * tab_tr_z + (1.0 - best_w) * deep_tr_z
        r_va_mix = best_w * tab_va_z + (1.0 - best_w) * deep_va_z
        r_te_mix = best_w * tab_te_z + (1.0 - best_w) * deep_te_z

        mix_ci_tr = harrell_cindex(ytr, r_tr_mix)
        mix_ci_va = harrell_cindex(yva, r_va_mix)
        mix_ci_te = harrell_cindex(yte, r_te_mix)
        print(f"[late_fusion]  train={mix_ci_tr:.4f} val={mix_ci_va:.4f} test={mix_ci_te:.4f} (w_tab={best_w:.2f})")
        results["late_fusion_wavg"] = {"train": mix_ci_tr, "val": mix_ci_va, "test": mix_ci_te, "w_tab": best_w}

        # Stacking: CoxPH on the 2 standardized risks (still only 2 params; usually safe)
        Ztr = np.column_stack([tab_tr_z, deep_tr_z])
        Zva = np.column_stack([tab_va_z, deep_va_z])
        Zte = np.column_stack([tab_te_z, deep_te_z])

        stack = CoxPHSurvivalAnalysis(alpha=1e-3)
        stack.fit(Ztr, ytr)

        s_tr = stack.predict(Ztr)
        s_va = stack.predict(Zva)
        s_te = stack.predict(Zte)

        ci_s_tr = harrell_cindex(ytr, s_tr)
        ci_s_va = harrell_cindex(yva, s_va)
        ci_s_te = harrell_cindex(yte, s_te)
        print(f"[stack_coxph]  train={ci_s_tr:.4f} val={ci_s_va:.4f} test={ci_s_te:.4f} (features=2)")
        results["stack_coxph"] = {"train": ci_s_tr, "val": ci_s_va, "test": ci_s_te, "alpha": 1e-3}


    # DeepSurv on fused design (often helps when deep features exist)
    if args.use_deepsurv and torch is not None:
        ds_model, ds_info = train_deepsurv(
            Xtr, ytr, Xva, yva,
            seed=args.seed,
            hidden=[128, 64],
            dropout=0.3,
            lr=1e-3,
            weight_decay=1e-4,
            max_epochs=500,
            patience=40,
        )
        ds_model.eval()
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            rt = ds_model(torch.from_numpy(Xtr).to(device)).cpu().numpy()
            rv = ds_model(torch.from_numpy(Xva).to(device)).cpu().numpy()
            rtest = ds_model(torch.from_numpy(Xte).to(device)).cpu().numpy()
        ci_tr2 = harrell_cindex(ytr, rt)
        ci_va2 = harrell_cindex(yva, rv)
        ci_te2 = harrell_cindex(yte, rtest)
        print(f"[deepsurv_fused] train={ci_tr2:.4f} val={ci_va2:.4f} test={ci_te2:.4f} (epochs_ran={ds_info['epochs_ran']}, best_epoch={ds_info['best_epoch']})")
        results["deepsurv_fused"] = {"train": ci_tr2, "val": ci_va2, "test": ci_te2, **ds_info}

    # Save results
    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    # reduce noisy warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    main()