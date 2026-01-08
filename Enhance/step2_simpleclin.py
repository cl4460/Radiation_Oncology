#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_cindex_boost_fixedsplit.py

Fixed-split, leakage-safe survival benchmarking + (optional) radiomics/deep features,
with correlation + univariate C-index feature filtering, CoxNet + GBSA, and simple ensemble.

Designed for:
- NSCLC-Radiomics (LUNG1) clinical CSV + fixed_split_seed42.json
- Optionally merge a radiomics feature matrix CSV (rows=patients, cols=features)
- Optionally merge deep feature CSV

Key design principles:
- All preprocessing/statistics/selection are fitted on TRAIN only during tuning.
- Validation is used ONLY for model/hyperparameter selection.
- Test is used ONLY once for final reporting (with bootstrap CI).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def canonicalize_stage(s: str):
    """Map raw stage strings to a stable set (keep missing as NaN).

    Why NaN?
      - We will impute missing stage using *train-mode* later (simple + avoids creating a 1-case 'UNKNOWN' category).
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).strip()
    if s == "" or s.lower() in {"na", "nan", "none"}:
        return np.nan
    s = s.replace(" ", "").replace("-", "")
    s_up = s.upper()
    # common variants
    if s_up in {"I", "1"}:
        return "I"
    if s_up in {"II", "2"}:
        return "II"
    if s_up in {"III", "3"}:
        # ambiguous, keep as III
        return "III"
    if s_up in {"IIIA", "III A", "3A"} or s_up.endswith("IIIA"):
        return "IIIA"
    if s_up in {"IIIB", "III B", "3B"} or s_up.endswith("IIIB"):
        return "IIIB"
    if s_up in {"IV", "4"}:
        return "IV"
    # some datasets have Tis
    if s_up in {"TIS"}:
        return "TIS"
    # fallback
    return s_up


def map_gender(x: str):
    # Keep missing as NaN; impute with train-mode later.
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().lower()
    if s in {"m", "male"}:
        return "Male"
    if s in {"f", "female"}:
        return "Female"
    return np.nan


def map_t_stage(x):
    """clinical.T.Stage in this CSV appears numeric: 1..5 (5 likely Tx/unknown).

    Policy:
      - missing (NaN) -> NaN (will be imputed by train-mode)
      - code 5 (or any non-{1,2,3,4}) -> 'Tx'
    """
    if pd.isna(x):
        return np.nan
    try:
        xi = int(float(x))
    except Exception:
        return np.nan
    if xi in {1, 2, 3, 4}:
        return f"T{xi}"
    return "Tx"


def map_n_stage(x):
    """Clinical.N.Stage appears numeric: 0..4 (4 likely Nx/unknown).

    Policy:
      - missing (NaN) -> NaN (will be imputed by train-mode)
      - code 4 (or any non-{0,1,2,3}) -> 'Nx'
    """
    if pd.isna(x):
        return np.nan
    try:
        xi = int(float(x))
    except Exception:
        return np.nan
    if xi in {0, 1, 2, 3}:
        return f"N{xi}"
    return "Nx"


def map_m_stage_to_cat(x) -> str:
    """
    Clinical.M.Stage appears numeric: 0, 1, 3.
    Clinically: M0=0, M1=1, Mx=unknown.
    We treat 3 (and any other non-{0,1}) as Mx by default.
    """
    if pd.isna(x):
        return "Mx"
    try:
        xi = int(float(x))
    except Exception:
        return "Mx"
    if xi == 0:
        return "M0"
    if xi == 1:
        return "M1"
    return "Mx"


def zscore(x: np.ndarray, mean_: Optional[float] = None, std_: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    if mean_ is None:
        mean_ = float(np.mean(x))
    if std_ is None:
        std_ = float(np.std(x) + 1e-12)
    return (x - mean_) / std_, mean_, std_


def bootstrap_cindex(event: np.ndarray, time: np.ndarray, risk: np.ndarray, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """
    Bootstrap percentile CI for Harrell's C-index (censored) on a fixed test set.
    Returns (lo, hi) for 95% CI.
    """
    from sksurv.metrics import concordance_index_censored

    rng = np.random.default_rng(seed)
    n = len(time)
    vals = np.empty(n_boot, dtype=float)
    idx = np.arange(n)
    for b in range(n_boot):
        sample = rng.choice(idx, size=n, replace=True)
        c = concordance_index_censored(event[sample], time[sample], risk[sample])[0]
        vals[b] = float(c)
    lo = float(np.quantile(vals, 0.025))
    hi = float(np.quantile(vals, 0.975))
    return lo, hi


def safe_numeric_df(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    """Keep numeric columns only (after coercion), excluding drop_cols."""
    out = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    # keep columns that are not all-NaN
    keep = [c for c in out.columns if not out[c].isna().all()]
    return out[keep]


def univariate_feature_score_cindex(
    event: np.ndarray,
    time: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """
    Score each feature by max(C-index(feature), 1-C-index(feature)).
    This is a cheap, direction-invariant discriminative proxy.
    """
    from sksurv.metrics import concordance_index_censored

    p = X.shape[1]
    scores = np.empty(p, dtype=float)
    for j in range(p):
        xj = X[:, j]
        # if constant, give 0.5
        if np.nanstd(xj) < 1e-12:
            scores[j] = 0.5
            continue
        c = concordance_index_censored(event, time, xj)[0]
        c = float(c)
        scores[j] = max(c, 1.0 - c)
    return scores


def greedy_corr_filter(
    X: np.ndarray,
    scores: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.9,
) -> List[str]:
    """
    Greedy redundancy removal:
    - compute |corr| on TRAIN
    - sort by univariate score descending
    - keep a feature if it's not highly correlated with any already-kept feature
    Returns kept feature names.
    """
    # Pearson correlation; works well after median imputation.
    # (Spearman is more robust but slower; you can swap if needed.)
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    abs_corr = np.abs(corr)

    order = np.argsort(-scores)  # desc
    kept: List[int] = []
    for idx in order:
        if len(kept) == 0:
            kept.append(idx)
            continue
        # vectorized check against kept features
        if np.any(abs_corr[idx, kept] > threshold):
            continue
        kept.append(idx)
    kept_names = [feature_names[i] for i in kept]
    return kept_names


@dataclass
class SplitData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


# ---------------------------
# Data loading / merging
# ---------------------------

def load_fixed_split(split_json: str) -> Dict[str, List[str]]:
    with open(split_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "splits" in obj:
        obj = obj["splits"]
    # expected keys: train/val/test
    return {k: list(obj[k]) for k in ["train", "val", "test"]}


def load_clinical(clinical_csv: str, id_col: str, time_col: str, event_col: str) -> pd.DataFrame:
    df = pd.read_csv(clinical_csv)
    if id_col not in df.columns:
        raise ValueError(f"Missing id_col={id_col} in clinical CSV columns={list(df.columns)}")
    # Ensure unique IDs
    df = df.drop_duplicates(subset=[id_col]).copy()

    # Standardize required label cols
    for c in [time_col, event_col]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} in clinical CSV")

    return df


def maybe_merge_features(
    base: pd.DataFrame,
    feat_csv: Optional[str],
    id_col: str,
    prefix: str,
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Merge a feature matrix on id_col and prefix ALL feature columns (except id_col) with `prefix`.

    Safety:
    - If the feature CSV accidentally contains label columns (time/event), pass them via drop_cols
      so they never enter the model features.
    """
    if feat_csv is None:
        return base
    feat = pd.read_csv(feat_csv)
    if id_col not in feat.columns:
        raise ValueError(f"Missing id_col={id_col} in features CSV: {feat_csv}")
    feat = feat.drop_duplicates(subset=[id_col]).copy()

    if drop_cols:
        feat = feat.drop(columns=[c for c in drop_cols if c in feat.columns], errors="ignore")

    feat_cols = [c for c in feat.columns if c != id_col]
    rename_map = {c: f"{prefix}{c}" for c in feat_cols}
    feat = feat.rename(columns=rename_map)

    merged = base.merge(feat, on=id_col, how="inner")
    return merged


# ---------------------------
# Preprocessing
# ---------------------------

def build_clinical_frame(
    df: pd.DataFrame,
    *,
    id_col: str,
    time_col: str,
    event_col: str,
    include_histology: bool,
    mstage_mode: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Returns:
      - df2: contains labels + clinical feature columns
      - num_cols: numeric clinical columns
      - cat_cols: categorical clinical columns
    """
    out = pd.DataFrame({id_col: df[id_col]})
    out[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    out[event_col] = pd.to_numeric(df[event_col], errors="coerce").astype("Int64")

    # age (numeric)
    out["age"] = pd.to_numeric(df.get("age"), errors="coerce")

    # gender (categorical)
    out["gender"] = df.get("gender").apply(map_gender)

    # overall stage (categorical)
    out["Overall.Stage"] = df.get("Overall.Stage").apply(canonicalize_stage)

    # T/N/M as categorical (safer than ordinal here)
    out["clinical.T.Stage"] = df.get("clinical.T.Stage").apply(map_t_stage)
    out["Clinical.N.Stage"] = df.get("Clinical.N.Stage").apply(map_n_stage)

    if mstage_mode == "drop":
        pass
    elif mstage_mode == "categorical":
        out["Clinical.M.Stage_cat"] = df.get("Clinical.M.Stage").apply(map_m_stage_to_cat)
    elif mstage_mode == "binary_nonzero":
        # Collapse *any* nonzero code (including '3') into 1.
        # IMPORTANT: we interpret this feature as "M is non-zero/unknown" (not as a strict metastasis label).
        m = pd.to_numeric(df.get("Clinical.M.Stage"), errors="coerce")
        out["Clinical.M.Stage_nonzero"] = ((m.notna()) & (m != 0)).astype(int)
    elif mstage_mode == "m0_m1_mx_plus_flags":
        # keep both: M category + explicit unknown flag (value==3)
        out["Clinical.M.Stage_cat"] = df.get("Clinical.M.Stage").apply(map_m_stage_to_cat)
        m = pd.to_numeric(df.get("Clinical.M.Stage"), errors="coerce")
        out["Clinical.M_is3_unknown"] = (m == 3).astype(int)
    else:
        raise ValueError(f"Unknown mstage_mode={mstage_mode}")

    # histology (optional)
    if include_histology:
        h = df.get("Histology")
        out["Histology"] = h.fillna("Unknown").astype(str).str.strip()
    # categorical + numeric columns
    num_cols = ["age"]
    cat_cols = ["gender", "Overall.Stage", "clinical.T.Stage", "Clinical.N.Stage"]

    if include_histology:
        cat_cols.append("Histology")
    if mstage_mode == "categorical":
        cat_cols.append("Clinical.M.Stage_cat")
    elif mstage_mode == "binary_nonzero":
        num_cols.append("Clinical.M.Stage_nonzero")
    elif mstage_mode == "m0_m1_mx_plus_flags":
        cat_cols.append("Clinical.M.Stage_cat")
        num_cols.append("Clinical.M_is3_unknown")

    return out, num_cols, cat_cols


def fit_transform_tabular(
    df: pd.DataFrame,
    splits: Dict[str, List[str]],
    *,
    id_col: str,
    time_col: str,
    event_col: str,
    clinical_num_cols: List[str],
    clinical_cat_cols: List[str],
    radiomics_prefixes: Tuple[str, ...] = ("R__", "D__"),
    age_impute: str = "median",
    radiomics_topk: int = 200,
    corr_threshold: float = 0.90,
    scale_radiomics: bool = True,
    scale_clinical_numeric: bool = True,
) -> Tuple[SplitData, Dict]:
    """
    Build X/y for train/val/test with leakage-safe preprocessing on TRAIN.
    Feature selection applies ONLY to radiomics/deep numeric columns (prefixed).
    Clinical columns are kept (after OHE/scaling).
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sksurv.util import Surv

    # Split masks
    id_to_row = {pid: i for i, pid in enumerate(df[id_col].tolist())}
    def rows_for(ids: List[str]) -> List[int]:
        return [id_to_row[i] for i in ids if i in id_to_row]

    tr_idx = rows_for(splits["train"])
    va_idx = rows_for(splits["val"])
    te_idx = rows_for(splits["test"])

    df_tr = df.iloc[tr_idx].copy()
    df_va = df.iloc[va_idx].copy()
    df_te = df.iloc[te_idx].copy()

    # labels
    def make_y(dfx: pd.DataFrame) -> np.ndarray:
        e = pd.to_numeric(dfx[event_col], errors="coerce").fillna(0).astype(bool).to_numpy()
        t = pd.to_numeric(dfx[time_col], errors="coerce").to_numpy()
        return Surv.from_arrays(event=e, time=t)

    y_train = make_y(df_tr)
    y_val = make_y(df_va)
    y_test = make_y(df_te)

    # ---- Clinical numeric: age impute + scale
    Xc_num_tr = df_tr[clinical_num_cols].copy()
    Xc_num_va = df_va[clinical_num_cols].copy()
    Xc_num_te = df_te[clinical_num_cols].copy()

    if "age" in clinical_num_cols:
        # impute age using TRAIN only
        if age_impute == "median":
            age_fill = float(np.nanmedian(Xc_num_tr["age"].to_numpy()))
        elif age_impute == "mean":
            age_fill = float(np.nanmean(Xc_num_tr["age"].to_numpy()))
        else:
            raise ValueError("age_impute must be 'median' or 'mean'")
        Xc_num_tr["age"] = Xc_num_tr["age"].fillna(age_fill)
        Xc_num_va["age"] = Xc_num_va["age"].fillna(age_fill)
        Xc_num_te["age"] = Xc_num_te["age"].fillna(age_fill)

    # For any other numeric clinical, median-impute
    clin_imputer = SimpleImputer(strategy="median")
    Xc_num_tr_imp = clin_imputer.fit_transform(Xc_num_tr)
    Xc_num_va_imp = clin_imputer.transform(Xc_num_va)
    Xc_num_te_imp = clin_imputer.transform(Xc_num_te)

    if scale_clinical_numeric:
        clin_scaler = StandardScaler()
        Xc_num_tr_imp = clin_scaler.fit_transform(Xc_num_tr_imp)
        Xc_num_va_imp = clin_scaler.transform(Xc_num_va_imp)
        Xc_num_te_imp = clin_scaler.transform(Xc_num_te_imp)
    else:
        clin_scaler = None

    clin_num_feature_names = [f"CNUM__{c}" for c in clinical_num_cols]

    # ---- Clinical categorical: train-mode imputation + OHE
    Xc_cat_tr = df_tr[clinical_cat_cols].copy()
    Xc_cat_va = df_va[clinical_cat_cols].copy()
    Xc_cat_te = df_te[clinical_cat_cols].copy()
    cat_impute_values = {}
    for c in clinical_cat_cols:
        # impute missing categories using TRAIN mode (simple + avoids 1-sample 'Unknown' categories)
        mode_series = Xc_cat_tr[c].dropna().mode()
        fill_value = mode_series.iloc[0] if len(mode_series) > 0 else "Unknown"
        cat_impute_values[c] = fill_value
        Xc_cat_tr[c] = Xc_cat_tr[c].fillna(fill_value).astype(str)
        Xc_cat_va[c] = Xc_cat_va[c].fillna(fill_value).astype(str)
        Xc_cat_te[c] = Xc_cat_te[c].fillna(fill_value).astype(str)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary")
    Xc_cat_tr_ohe = ohe.fit_transform(Xc_cat_tr)
    Xc_cat_va_ohe = ohe.transform(Xc_cat_va)
    Xc_cat_te_ohe = ohe.transform(Xc_cat_te)

    clin_cat_feature_names = [f"CCAT__{n}" for n in ohe.get_feature_names_out(clinical_cat_cols)]

    # ---- Radiomics/deep numeric block: columns prefixed with radiomics_prefixes
    rad_cols = [c for c in df.columns if any(c.startswith(pfx) for pfx in radiomics_prefixes)]
    meta = {
        "n_total": len(df),
        "n_train": len(df_tr),
        "n_val": len(df_va),
        "n_test": len(df_te),
        "n_rad_cols_raw": len(rad_cols),
        "age_impute": age_impute,
        "corr_threshold": corr_threshold,
        "radiomics_topk": radiomics_topk,
        "clinical_cat_impute": cat_impute_values,
    }

    if len(rad_cols) == 0:
        # Clinical-only
        X_train = np.hstack([Xc_num_tr_imp, Xc_cat_tr_ohe])
        X_val = np.hstack([Xc_num_va_imp, Xc_cat_va_ohe])
        X_test = np.hstack([Xc_num_te_imp, Xc_cat_te_ohe])
        feature_names = clin_num_feature_names + clin_cat_feature_names
        meta["selected_rad_cols"] = []
        meta["n_rad_cols_selected"] = 0
        return SplitData(X_train, X_val, X_test, y_train, y_val, y_test, feature_names), meta

    # numeric coercion
    Xr_tr_df = df_tr[rad_cols].copy()
    Xr_va_df = df_va[rad_cols].copy()
    Xr_te_df = df_te[rad_cols].copy()
    for c in rad_cols:
        Xr_tr_df[c] = pd.to_numeric(Xr_tr_df[c], errors="coerce")
        Xr_va_df[c] = pd.to_numeric(Xr_va_df[c], errors="coerce")
        Xr_te_df[c] = pd.to_numeric(Xr_te_df[c], errors="coerce")


    # Replace inf with NaN, then drop features that are all-NaN in TRAIN (otherwise SimpleImputer will crash)
    Xr_tr_df = Xr_tr_df.replace([np.inf, -np.inf], np.nan)
    Xr_va_df = Xr_va_df.replace([np.inf, -np.inf], np.nan)
    Xr_te_df = Xr_te_df.replace([np.inf, -np.inf], np.nan)

    all_nan_cols = [c for c in rad_cols if Xr_tr_df[c].isna().all()]
    if len(all_nan_cols) > 0:
        Xr_tr_df = Xr_tr_df.drop(columns=all_nan_cols)
        Xr_va_df = Xr_va_df.drop(columns=all_nan_cols)
        Xr_te_df = Xr_te_df.drop(columns=all_nan_cols)
        rad_cols = [c for c in rad_cols if c not in all_nan_cols]

    rad_imputer = SimpleImputer(strategy="median")
    Xr_tr = rad_imputer.fit_transform(Xr_tr_df)
    Xr_va = rad_imputer.transform(Xr_va_df)
    Xr_te = rad_imputer.transform(Xr_te_df)

    # remove near-zero variance on train
    var = np.var(Xr_tr, axis=0)
    keep_var = var > 1e-12
    Xr_tr = Xr_tr[:, keep_var]
    Xr_va = Xr_va[:, keep_var]
    Xr_te = Xr_te[:, keep_var]
    rad_cols_var = [c for c, kv in zip(rad_cols, keep_var) if kv]
    meta["n_rad_cols_after_var"] = len(rad_cols_var)

    # univariate score on train (direction-invariant C-index)
    event_tr = y_train["event"].astype(bool)
    time_tr = y_train["time"].astype(float)
    scores = univariate_feature_score_cindex(event_tr, time_tr, Xr_tr)

    # correlation filter on train
    kept_cols = greedy_corr_filter(Xr_tr, scores, rad_cols_var, threshold=corr_threshold)
    kept_mask = [c in set(kept_cols) for c in rad_cols_var]
    Xr_tr2 = Xr_tr[:, kept_mask]
    Xr_va2 = Xr_va[:, kept_mask]
    Xr_te2 = Xr_te[:, kept_mask]
    rad_cols_corr = [c for c in rad_cols_var if c in set(kept_cols)]
    meta["n_rad_cols_after_corr"] = len(rad_cols_corr)

    # re-score after corr filter and take top-K (if needed)
    if radiomics_topk is not None and radiomics_topk > 0 and len(rad_cols_corr) > radiomics_topk:
        scores2 = univariate_feature_score_cindex(event_tr, time_tr, Xr_tr2)
        top_idx = np.argsort(-scores2)[:radiomics_topk]
        Xr_tr3 = Xr_tr2[:, top_idx]
        Xr_va3 = Xr_va2[:, top_idx]
        Xr_te3 = Xr_te2[:, top_idx]
        rad_cols_sel = [rad_cols_corr[i] for i in top_idx]
    else:
        Xr_tr3, Xr_va3, Xr_te3 = Xr_tr2, Xr_va2, Xr_te2
        rad_cols_sel = rad_cols_corr

    meta["selected_rad_cols"] = rad_cols_sel
    meta["n_rad_cols_selected"] = len(rad_cols_sel)

    # scale radiomics/deep features (fit on train)
    if scale_radiomics:
        rad_scaler = StandardScaler()
        Xr_tr3 = rad_scaler.fit_transform(Xr_tr3)
        Xr_va3 = rad_scaler.transform(Xr_va3)
        Xr_te3 = rad_scaler.transform(Xr_te3)
    else:
        rad_scaler = None

    # concat clinical + selected radiomics
    X_train = np.hstack([Xc_num_tr_imp, Xc_cat_tr_ohe, Xr_tr3])
    X_val = np.hstack([Xc_num_va_imp, Xc_cat_va_ohe, Xr_va3])
    X_test = np.hstack([Xc_num_te_imp, Xc_cat_te_ohe, Xr_te3])

    feature_names = clin_num_feature_names + clin_cat_feature_names + [f"RAD__{c}" for c in rad_cols_sel]

    # store fitted objects for possible reuse
    meta["_fitted"] = {
        "clin_imputer": clin_imputer,
        "clin_scaler": clin_scaler,
        "ohe": ohe,
        "rad_imputer": rad_imputer,
        "rad_scaler": rad_scaler,
        "rad_cols_sel": rad_cols_sel,
        "rad_cols_var": rad_cols_var,
        "rad_cols_corr": rad_cols_corr,
        "rad_keep_var_mask": keep_var.tolist(),
        "rad_kept_corr_mask": kept_mask,
        "clinical_num_cols": clinical_num_cols,
        "clinical_cat_cols": clinical_cat_cols,
        "radiomics_prefixes": radiomics_prefixes,
    }

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test, feature_names), meta


# ---------------------------
# Models
# ---------------------------

def cindex(y: np.ndarray, risk: np.ndarray) -> float:
    from sksurv.metrics import concordance_index_censored
    e = y["event"].astype(bool)
    t = y["time"].astype(float)
    return float(concordance_index_censored(e, t, risk)[0])


def fit_coxnet_select_alpha(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    l1_ratio: float = 0.5,
    n_alphas: int = 100,
    alpha_min_ratio: float = 0.01,  # 稍微调大默认值以增加稳定性
    min_nnz: int = 3,  # 放宽最小非零特征数限制
    max_iter: int = 100000,
) -> Dict:
    from sksurv.linear_model import CoxnetSurvivalAnalysis

    # 1. Fit the full path (Warm start is handled internally here)
    model = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,
        n_alphas=n_alphas,
        alpha_min_ratio=alpha_min_ratio,
        max_iter=max_iter,
        tol=1e-7,
        fit_baseline_model=True 
    )
    
    try:
        model.fit(X_tr, y_tr)
    except Exception as e:
        print(f"  [Warning] CoxNet Path fit failed: {e}. Returning zero model.")
        return {
            "model": None, "alpha": None, "l1_ratio": l1_ratio, 
            "val_cindex": 0.5, "nnz": 0, "coef": np.zeros(X_tr.shape[1])
        }

    # 2. Evaluate each alpha on validation
    # coefs shape: (n_features, n_alphas)
    coefs = model.coef_
    alphas = model.alphas_
    
    best = {"cindex": -1.0, "alpha": None, "nnz": 0, "coef": None, "index": -1}

    # Iterate through the path
    for j in range(coefs.shape[1]):
        coef = coefs[:, j]
        nnz = int(np.sum(np.abs(coef) > 1e-6))
        
        # Skip models that are too sparse (likely noise or underfitting)
        if nnz < min_nnz:
            continue
        
        # Predict on Val
        # Linear predictor = X @ coef
        risk_va = X_va @ coef
        
        try:
            c = cindex(y_va, risk_va)
        except:
            c = 0.5 # Handle edge cases where risk is NaN or constant
            
        if c > best["cindex"]:
            best = {
                "cindex": c, 
                "alpha": float(alphas[j]), 
                "nnz": nnz, 
                "coef": coef.copy(), # Important: copy the coefs
                "index": j
            }

    # Fallback: If no alpha met min_nnz, just pick the one with max cindex regardless of nnz
    if best["alpha"] is None:
        for j in range(coefs.shape[1]):
            coef = coefs[:, j]
            nnz = int(np.sum(np.abs(coef) > 1e-6))
            risk_va = X_va @ coef
            try:
                c = cindex(y_va, risk_va)
            except: c = 0.5
            
            if c > best["cindex"]:
                 best = {"cindex": c, "alpha": float(alphas[j]), "nnz": nnz, "coef": coef.copy(), "index": j}
    
    # 3. NO REFIT NEEDED! 
    # Directly return the best coefficients found during the path search.
    # This avoids the "ArithmeticError" caused by cold-starting the optimizer on a small alpha.
    
    if best["coef"] is None:
        # Absolute fallback if everything failed (e.g. data issues)
        best["coef"] = np.zeros(X_tr.shape[1])
        best["alpha"] = 0.1

    out = {
        "model": model, # Return the path model
        "alpha": best["alpha"],
        "l1_ratio": l1_ratio,
        "val_cindex": best["cindex"],
        "nnz": best["nnz"],
        "coef": best["coef"], # Use the extracted coefficients
    }
    return out


def fit_gbsa_grid(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    *,
    grid: Optional[Dict[str, List]] = None,
    random_state: int = 42,
) -> Dict:
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis

    if grid is None:
        #grid = {
        #    "n_estimators": [100, 200, 400],
        #    "learning_rate": [0.05, 0.1],
        #    "max_depth": [1, 2, 3],
        #    "min_samples_leaf": [3, 5, 10],
        #    "subsample": [1.0],
        #}
        grid = {
            "n_estimators": [50, 100, 200],  # 减少
            "learning_rate": [0.01, 0.05],   # 更保守
            "max_depth": [1, 2],             # 限制为1-2
            "min_samples_leaf": [10, 20],    # 增加
            "subsample": [0.8, 1.0],         # 添加subsample
        }

    best = {"cindex": -1.0, "params": None, "model": None}
    for n_estimators in grid["n_estimators"]:
        for lr in grid["learning_rate"]:
            for md in grid["max_depth"]:
                for msl in grid["min_samples_leaf"]:
                    for subsample in grid.get("subsample", [1.0]):
                        m = GradientBoostingSurvivalAnalysis(
                            n_estimators=n_estimators,
                            learning_rate=lr,
                            max_depth=md,
                            min_samples_leaf=msl,
                            subsample=subsample,
                            random_state=random_state,
                        )
                        m.fit(X_tr, y_tr)
                        risk_va = m.predict(X_va)
                        c = cindex(y_va, risk_va)
                        if c > best["cindex"]:
                            best = {"cindex": c, "params": {"n_estimators": n_estimators, "learning_rate": lr, "max_depth": md,
                                                             "min_samples_leaf": msl, "subsample": subsample},
                                    "model": m}
    best["val_cindex"] = best["cindex"]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_csv", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--id_col", default="PatientID")
    ap.add_argument("--time_col", default="Survival.time")
    ap.add_argument("--event_col", default="deadstatus.event")

    ap.add_argument("--radiomics_csv", default=None, help="Optional radiomics feature CSV. Must contain id_col.")
    ap.add_argument("--deep_csv", default=None, help="Optional deep feature CSV. Must contain id_col.")

    ap.add_argument("--include_histology", action="store_true", help="Include Histology (default: off).")
    ap.add_argument(
        "--mstage_mode",
        default="binary_nonzero",
        choices=["drop", "categorical", "binary_nonzero", "m0_m1_mx_plus_flags"],
        help=(
            "How to encode Clinical.M.Stage. Default=binary_nonzero (0 vs nonzero codes, including '3'). "
            "Use categorical if you want to keep Mx separate (0/1/3 -> M0/M1/Mx)."
        ),
    )

    ap.add_argument("--age_impute", default="median", choices=["median", "mean"])
    ap.add_argument("--radiomics_topk", type=int, default=200)
    ap.add_argument("--corr_threshold", type=float, default=0.90)

    ap.add_argument("--cox_l1_ratio", type=float, default=0.5)
    ap.add_argument("--cox_min_nnz", type=int, default=5)

    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # Load data
    splits = load_fixed_split(args.split_json)
    df = load_clinical(args.clinical_csv, args.id_col, args.time_col, args.event_col)

    # Build clinical frame
    clin_df, clin_num_cols, clin_cat_cols = build_clinical_frame(
        df,
        id_col=args.id_col,
        time_col=args.time_col,
        event_col=args.event_col,
        include_histology=args.include_histology,
        mstage_mode=args.mstage_mode,
    )

    # Merge optional features
    full = clin_df.copy()
    full = maybe_merge_features(full, args.radiomics_csv, args.id_col, prefix="R__", drop_cols=[args.time_col, args.event_col])
    full = maybe_merge_features(full, args.deep_csv, args.id_col, prefix="D__", drop_cols=[args.time_col, args.event_col])

    # Intersect with available IDs from splits (important if features missing)
    all_ids = set(full[args.id_col].astype(str).tolist())
    splits2 = {k: [pid for pid in v if pid in all_ids] for k, v in splits.items()}

    # Prepare X/y
    sd, meta = fit_transform_tabular(
        full,
        splits2,
        id_col=args.id_col,
        time_col=args.time_col,
        event_col=args.event_col,
        clinical_num_cols=clin_num_cols,
        clinical_cat_cols=clin_cat_cols,
        radiomics_prefixes=("R__", "D__"),
        age_impute=args.age_impute,
        radiomics_topk=args.radiomics_topk,
        corr_threshold=args.corr_threshold,
    )

    # Fit models (tune on val)
    results = {}

    # CoxNet
    cox = fit_coxnet_select_alpha(
        sd.X_train, sd.y_train,
        sd.X_val, sd.y_val,
        l1_ratio=args.cox_l1_ratio,
        min_nnz=args.cox_min_nnz,
    )
    risk_tr_cox = sd.X_train @ cox["coef"]
    risk_va_cox = sd.X_val @ cox["coef"]
    risk_te_cox = sd.X_test @ cox["coef"]
    results["coxnet"] = {
        "alpha": cox["alpha"],
        "l1_ratio": cox["l1_ratio"],
        "nnz": cox["nnz"],
        "cindex_train": cindex(sd.y_train, risk_tr_cox),
        "cindex_val": cindex(sd.y_val, risk_va_cox),
        "cindex_test": cindex(sd.y_test, risk_te_cox),
    }

    # GBSA
    gb = fit_gbsa_grid(sd.X_train, sd.y_train, sd.X_val, sd.y_val, random_state=args.seed)
    risk_tr_gb = gb["model"].predict(sd.X_train)
    risk_va_gb = gb["model"].predict(sd.X_val)
    risk_te_gb = gb["model"].predict(sd.X_test)
    results["gbsa"] = {
        "params": gb["params"],
        "cindex_train": cindex(sd.y_train, risk_tr_gb),
        "cindex_val": cindex(sd.y_val, risk_va_gb),
        "cindex_test": cindex(sd.y_test, risk_te_gb),
    }

    # Ensemble (z-scored risks, z fitted on TRAIN only)
    z_tr_cox, m_cox, s_cox = zscore(risk_tr_cox)
    z_tr_gb, m_gb, s_gb = zscore(risk_tr_gb)
    z_va_cox, _, _ = zscore(risk_va_cox, m_cox, s_cox)
    z_va_gb, _, _ = zscore(risk_va_gb, m_gb, s_gb)
    z_te_cox, _, _ = zscore(risk_te_cox, m_cox, s_cox)
    z_te_gb, _, _ = zscore(risk_te_gb, m_gb, s_gb)

    risk_va_ens = 0.5 * (z_va_cox + z_va_gb)
    risk_te_ens = 0.5 * (z_te_cox + z_te_gb)
    risk_tr_ens = 0.5 * (z_tr_cox + z_tr_gb)

    results["ensemble"] = {
        "cindex_train": cindex(sd.y_train, risk_tr_ens),
        "cindex_val": cindex(sd.y_val, risk_va_ens),
        "cindex_test": cindex(sd.y_test, risk_te_ens),
        "zscore_params": {"cox": {"mean": m_cox, "std": s_cox}, "gbsa": {"mean": m_gb, "std": s_gb}},
    }

    # Choose best by VAL C-index
    best_name = max(results.keys(), key=lambda k: results[k]["cindex_val"])
    best = results[best_name]

    # Bootstrap CI on test for the selected model
    if best_name == "coxnet":
        risk_te_best = risk_te_cox
    elif best_name == "gbsa":
        risk_te_best = risk_te_gb
    else:
        risk_te_best = risk_te_ens

    e_test = sd.y_test["event"].astype(bool)
    t_test = sd.y_test["time"].astype(float)
    lo, hi = bootstrap_cindex(e_test, t_test, risk_te_best, n_boot=args.bootstrap, seed=args.seed)

    summary = {
        "best_model": best_name,
        "best": best,
        "bootstrap_test_ci95": {"lo": lo, "hi": hi, "n_boot": args.bootstrap},
        "all_models": results,
        "meta": {k: v for k, v in meta.items() if k != "_fitted"},
        "args": vars(args),
        "n_features_total": int(sd.X_train.shape[1]),
    }

    # Save
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save predictions (for audit)
    pred_df = pd.DataFrame({
        "split": (["train"] * len(risk_tr_cox)) + (["val"] * len(risk_va_cox)) + (["test"] * len(risk_te_cox)),
        "risk_coxnet": np.concatenate([risk_tr_cox, risk_va_cox, risk_te_cox]),
        "risk_gbsa": np.concatenate([risk_tr_gb, risk_va_gb, risk_te_gb]),
        "risk_ensemble": np.concatenate([risk_tr_ens, risk_va_ens, risk_te_ens]),
    })
    pred_df.to_csv(os.path.join(args.out_dir, "risks.csv"), index=False)

    # Save selected radiomics feature list (if any)
    if meta.get("selected_rad_cols"):
        pd.Series(meta["selected_rad_cols"]).to_csv(os.path.join(args.out_dir, "selected_radiomics_cols.txt"),
                                                    index=False, header=False)

    # Print concise report
    print("=== Fixed-split survival benchmark ===")
    print(f"Train/Val/Test: {meta['n_train']}/{meta['n_val']}/{meta['n_test']}  |  Total features used: {sd.X_train.shape[1]}")
    print(f"Radiomics cols selected: {meta.get('n_rad_cols_selected', 0)}")
    for name, r in results.items():
        print(f"[{name}] C-index train={r['cindex_train']:.4f}  val={r['cindex_val']:.4f}  test={r['cindex_test']:.4f}")
    print(f"BEST={best_name}  test_CI95=[{lo:.4f}, {hi:.4f}]")
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
