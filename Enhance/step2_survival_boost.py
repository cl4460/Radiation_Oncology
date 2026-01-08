#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_survival_boost.py

Goal
----
A *high-leverage* fixed-split benchmark for LUNG1 that is compatible with your current
workflow and avoids common pitfalls recorded in your PDFs.

Key upgrades vs step2_simpleclin.py
-----------------------------------
1) Radiomics univariate screening becomes SIGN-INVARIANT:
   - score(feature) = abs(C-index - 0.5)
   - prevents discarding "protective" features that naturally give C-index < 0.5

2) Optional "highest10" radiomics signature mode:
   - uses the 10 traditional radiomics signatures listed in your highest.pdf (Le et al.)
   - quick sanity-check: if even this doesn't help, the ceiling is likely low without deep features

3) Optional DeepSurv (pycox) model:
   - many >0.7 C-index pipelines on LUNG1 rely on DeepSurv-style non-linear tabular models

4) Minimal clinical preprocessing (as you requested):
   - age: TRAIN median impute (no age_is_missing)
   - stage/T/N: TRAIN mode impute + one-hot (fixed category vocab)
   - gender: binary 0/1 (TRAIN mode if missing)
   - M stage: binary_nonzero (0 vs nonzero; 3 is merged into nonzero)

Evaluation discipline
---------------------
- Feature selection uses TRAIN only.
- Hyperparameters use VAL only.
- TEST is evaluated once per run.

Example
-------
# Baseline (coxnet + gbsa) with sign-invariant selection
python step2_survival_boost_v2.py \
  --clinical_csv /path/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
  --radiomics_csv /path/Tb_features_107.csv \
  --split_json /path/fixed_split_seed42.json \
  --out_dir out_step2 \
  --radiomics_mode topk --radiomics_topk 50 --corr_threshold 0.75 \
  --mstage_mode binary_nonzero

# Try Le2025 highest10 signature (requires the 1130-feature table with wavelet features)
python step2_survival_boost_v2.py \
  --clinical_csv /path/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
  --radiomics_csv /path/Tb_features.csv \
  --split_json /path/fixed_split_seed42.json \
  --out_dir out_highest10 \
  --radiomics_mode highest10 \
  --mstage_mode binary_nonzero \
  --use_deepsurv 1

DeepSurv dependency
-------------------
If you enable --use_deepsurv 1 and pycox is missing:
  pip install pycox torchtuples torch

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler

# scikit-survival
try:
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
except Exception as e:
    raise RuntimeError(
        "This script requires scikit-survival (sksurv). "
        "Your environment that ran step2_simpleclin.py already has it.\n"
        f"Import error: {e}"
    )

# Optional DeepSurv (pycox)
_HAS_PYCOX = False
try:
    import torch
    import torchtuples as tt
    from pycox.models import CoxPH as PyCoxPH

    _HAS_PYCOX = True
except Exception:
    _HAS_PYCOX = False


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _fingerprint_ids(ids: Sequence[str]) -> str:
    import hashlib

    h = hashlib.sha256(",".join(map(str, ids)).encode("utf-8")
    ).hexdigest()
    return h[:16]


def _infer_id_column(cols: Sequence[str]) -> str:
    for c in cols:
        cl = c.lower()
        if "patient" in cl or "subject" in cl or cl in {"id", "patientid"}:
            return c
    return cols[0]


def _mode_ignore_na(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return s.mode().iloc[0]


# -----------------------------
# Clinical preprocessing
# -----------------------------
_STAGE_CATS = ["I", "II", "IIIA", "IIIB", "UNK"]
_T_CATS = ["T1", "T2", "T3", "T4", "Tx", "UNK"]
_N_CATS = ["N0", "N1", "N2", "N3", "UNK"]


def canonicalize_stage(x) -> str:
    if pd.isna(x):
        return "UNK"
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "nan", "none"}:
        return "UNK"
    s = s.replace(" ", "").replace("-", "")
    su = s.upper()

    if su in {"I", "1"}:
        return "I"
    if su in {"II", "2"}:
        return "II"
    # common patterns
    if su in {"IIIA", "3A"} or (su.startswith("III") and su.endswith("A")):
        return "IIIA"
    if su in {"IIIB", "3B"} or (su.startswith("III") and su.endswith("B")):
        return "IIIB"
    # LUNG1 basically has no stage IV; treat as unknown to avoid rare-category artifacts
    return "UNK"


def map_gender(x) -> int:
    if pd.isna(x):
        return -1
    s = str(x).strip().lower()
    if s in {"male", "m", "1"}:
        return 1
    if s in {"female", "f", "0"}:
        return 0
    return -1


def map_t_stage(x) -> str:
    if pd.isna(x):
        return "UNK"
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "nan", "none"}:
        return "UNK"
    try:
        v = int(float(s))
        if v in {1, 2, 3, 4}:
            return f"T{v}"
        if v in {0, 5}:
            return "Tx"
        return "UNK"
    except Exception:
        pass
    su = s.upper()
    if su in {"TX"}:
        return "Tx"
    if su in {"T1", "T2", "T3", "T4"}:
        return su
    return "UNK"


def map_n_stage(x) -> str:
    if pd.isna(x):
        return "UNK"
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "nan", "none"}:
        return "UNK"
    try:
        v = int(float(s))
        if v in {0, 1, 2, 3}:
            return f"N{v}"
        return "UNK"
    except Exception:
        pass
    su = s.upper()
    if su in {"N0", "N1", "N2", "N3"}:
        return su
    return "UNK"


def map_m_stage_binary_nonzero(x) -> int:
    """Collapse {1,3} into 1 (non-zero)."""
    if pd.isna(x):
        return -1
    try:
        v = int(float(str(x).strip()))
        return 0 if v == 0 else 1
    except Exception:
        return -1


@dataclass
class SplitData:
    ids_train: List[str]
    ids_val: List[str]
    ids_test: List[str]


def load_fixed_split(split_json: str) -> SplitData:
    d = _read_json(split_json)

    # Support nested structure from step1_lock_split.py: {"splits": {"train": [...], "val": [...], "test": [...]}}
    if "splits" in d and isinstance(d["splits"], dict):
        splits = d["splits"]
        if all(k in splits for k in ["train", "val", "test"]):
            tr = list(map(str, splits["train"]))
            va = list(map(str, splits["val"]))
            te = list(map(str, splits["test"]))
            return SplitData(tr, va, te)

    # Support flat structure: {"train": [...], "val": [...], "test": [...]}
    for keyset in [
        ("train", "val", "test"),
        ("train_ids", "val_ids", "test_ids"),
        ("train_ids", "valid_ids", "test_ids"),
    ]:
        if all(k in d for k in keyset):
            tr = list(map(str, d[keyset[0]]))
            va = list(map(str, d[keyset[1]]))
            te = list(map(str, d[keyset[2]]))
            return SplitData(tr, va, te)

    raise ValueError(f"Unrecognized split json keys: {list(d.keys())}")


def build_clinical_table(
    clinical_csv: str, split: SplitData, mstage_mode: str = "binary_nonzero"
) -> pd.DataFrame:
    df = pd.read_csv(clinical_csv)

    # patient id
    id_col = None
    for c in df.columns:
        if c.lower() in {"patientid", "patient_id"}:
            id_col = c
            break
    if id_col is None:
        id_col = _infer_id_column(df.columns)

    df[id_col] = df[id_col].astype(str)
    df = df.set_index(id_col, drop=False)

    if "Survival.time" not in df.columns or "deadstatus.event" not in df.columns:
        raise ValueError("Clinical CSV must contain Survival.time and deadstatus.event")

    out = pd.DataFrame(index=df.index.copy())
    out["time"] = pd.to_numeric(df["Survival.time"], errors="coerce")
    out["event"] = (
        pd.to_numeric(df["deadstatus.event"], errors="coerce")
        .fillna(0)
        .astype(int)
        .clip(0, 1)
        .astype(bool)
    )

    out["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    out["gender"] = df.get("gender").apply(map_gender) if "gender" in df.columns else -1

    out["Overall.Stage"] = (
        df.get("Overall.Stage").apply(canonicalize_stage)
        if "Overall.Stage" in df.columns
        else "UNK"
    )

    # T/N
    if "clinical.T.Stage" in df.columns:
        out["T_stage"] = df["clinical.T.Stage"].apply(map_t_stage)
    elif "Clinical.T.Stage" in df.columns:
        out["T_stage"] = df["Clinical.T.Stage"].apply(map_t_stage)
    else:
        out["T_stage"] = "UNK"

    if "Clinical.N.Stage" in df.columns:
        out["N_stage"] = df["Clinical.N.Stage"].apply(map_n_stage)
    elif "clinical.N.Stage" in df.columns:
        out["N_stage"] = df["clinical.N.Stage"].apply(map_n_stage)
    else:
        out["N_stage"] = "UNK"

    # M
    m_col = None
    for cand in ["Clinical.M.Stage", "clinical.M.Stage", "M.Stage", "m_stage"]:
        if cand in df.columns:
            m_col = cand
            break

    if mstage_mode == "drop" or m_col is None:
        pass
    elif mstage_mode == "binary_nonzero":
        out["m_stage_bin"] = df[m_col].apply(map_m_stage_binary_nonzero)
    else:
        raise ValueError(f"Unsupported mstage_mode: {mstage_mode}")

    # keep only split ids
    keep_ids = set(split.ids_train) | set(split.ids_val) | set(split.ids_test)
    out = out.loc[out.index.intersection(pd.Index(list(keep_ids)))].copy()

    # drop missing time
    out = out.dropna(subset=["time"])

    train_idx = out.index.isin(split.ids_train)

    # impute age by TRAIN median
    age_med = np.nanmedian(out.loc[train_idx, "age"].values.astype(float))
    out["age"] = out["age"].fillna(age_med)

    # gender by TRAIN mode
    g_train = out.loc[train_idx, "gender"].replace(-1, np.nan)
    g_mode = _mode_ignore_na(g_train)
    if pd.isna(g_mode):
        g_mode = 0
    out["gender"] = out["gender"].replace(-1, np.nan).fillna(g_mode).astype(int)

    # stage/T/N: impute UNK by TRAIN mode
    for col in ["Overall.Stage", "T_stage", "N_stage"]:
        s_train = out.loc[train_idx, col].replace("UNK", np.nan)
        md = _mode_ignore_na(s_train)
        if pd.isna(md):
            md = "UNK"
        out[col] = out[col].replace("UNK", np.nan).fillna(md)

    # M by TRAIN mode
    if "m_stage_bin" in out.columns:
        m_train = out.loc[train_idx, "m_stage_bin"].replace(-1, np.nan)
        m_mode = _mode_ignore_na(m_train)
        if pd.isna(m_mode):
            m_mode = 0
        out["m_stage_bin"] = out["m_stage_bin"].replace(-1, np.nan).fillna(m_mode).astype(int)

    # warn missing ids
    missing = sorted(list(keep_ids - set(out.index)))
    if missing:
        warnings.warn(f"{len(missing)} ids from split not found in clinical table. Example: {missing[:5]}")

    return out


# -----------------------------
# Radiomics loading + selection
# -----------------------------

def load_radiomics_table(radiomics_csv: str, id_col: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(radiomics_csv)
    if id_col is None:
        id_col = _infer_id_column(df.columns)
    df[id_col] = df[id_col].astype(str)
    df = df.set_index(id_col, drop=False)

    feat_cols = [c for c in df.columns if c != id_col]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# 10 traditional radiomics signatures from highest.pdf
_HIGHEST10 = [
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


def cindex_value(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> float:
    return float(concordance_index_censored(event.astype(bool), time.astype(float), risk.astype(float))[0])


def score_feature_strength_abs_cindex(y_event: np.ndarray, y_time: np.ndarray, x: np.ndarray) -> float:
    if np.isnan(x).any():
        med = np.nanmedian(x)
        x = np.where(np.isnan(x), med, x)
    c = cindex_value(y_event, y_time, x)
    return abs(c - 0.5)


def remove_highly_correlated(X: pd.DataFrame, scores: Dict[str, float], threshold: float) -> List[str]:
    cols = list(X.columns)
    if len(cols) <= 1:
        return cols

    M = X.values.astype(float)
    col_med = np.nanmedian(M, axis=0)
    inds = np.where(np.isnan(M))
    if len(inds[0]) > 0:
        M[inds] = np.take(col_med, inds[1])

    with np.errstate(invalid="ignore"):
        C = np.corrcoef(M, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    absC = np.abs(C)

    keep = np.ones(len(cols), dtype=bool)
    for i in range(len(cols)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(cols)):
            if not keep[j]:
                continue
            if absC[i, j] > threshold:
                fi, fj = cols[i], cols[j]
                si = scores.get(fi, 0.0)
                sj = scores.get(fj, 0.0)
                # drop the lower-score feature
                if sj > si:
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    return [c for c, k in zip(cols, keep) if k]


def select_radiomics_features(
    rad: pd.DataFrame,
    split: SplitData,
    y_event: np.ndarray,
    y_time: np.ndarray,
    mode: str,
    topk: int,
    corr_threshold: float,
    pre_topk_cap: int = 400,
) -> List[str]:
    id_col = _infer_id_column(rad.columns)
    feat_cols = [c for c in rad.columns if c != id_col]

    if mode == "all":
        return feat_cols

    if mode == "highest10":
        missing = [c for c in _HIGHEST10 if c not in rad.columns]
        if missing:
            raise ValueError(
                "radiomics_mode=highest10 but missing columns: "
                f"{missing}. Use the 1130-feature table with wavelet features."
            )
        return list(_HIGHEST10)

    if mode != "topk":
        raise ValueError(f"Unknown radiomics_mode: {mode}")

    train_ids = set(split.ids_train)
    rad_train = rad.loc[rad.index.intersection(pd.Index(list(train_ids))), feat_cols].copy()

    scores = {c: score_feature_strength_abs_cindex(y_event, y_time, rad_train[c].values.astype(float)) for c in feat_cols}

    # pre-filter by score for cheaper corr filtering
    pre_k = min(max(topk * 5, 200), pre_topk_cap, len(feat_cols))
    sorted_cols = sorted(feat_cols, key=lambda k: scores.get(k, 0.0), reverse=True)
    candidate_cols = sorted_cols[:pre_k]

    kept = remove_highly_correlated(rad_train[candidate_cols], scores, threshold=corr_threshold)
    kept_sorted = sorted(kept, key=lambda k: scores.get(k, 0.0), reverse=True)

    return kept_sorted[: min(topk, len(kept_sorted))]


# -----------------------------
# Design matrix
# -----------------------------

def make_design_matrices(
    clin: pd.DataFrame,
    rad: pd.DataFrame,
    split: SplitData,
    radiomics_cols: List[str],
    use_m_stage: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # ids present in clinical
    ids_train = [i for i in split.ids_train if i in clin.index]
    ids_val = [i for i in split.ids_val if i in clin.index]
    ids_test = [i for i in split.ids_test if i in clin.index]

    y_train = Surv.from_arrays(event=clin.loc[ids_train, "event"].values.astype(bool), time=clin.loc[ids_train, "time"].values.astype(float))
    y_val = Surv.from_arrays(event=clin.loc[ids_val, "event"].values.astype(bool), time=clin.loc[ids_val, "time"].values.astype(float))
    y_test = Surv.from_arrays(event=clin.loc[ids_test, "event"].values.astype(bool), time=clin.loc[ids_test, "time"].values.astype(float))

    # radiomics align and impute by TRAIN median
    rad_feat = rad[radiomics_cols].reindex(clin.index)
    col_med = rad_feat.loc[ids_train].median(axis=0, skipna=True)
    rad_feat = rad_feat.fillna(col_med)

    age = clin["age"].astype(float).values.reshape(-1, 1)
    X_cont_all = np.concatenate([age, rad_feat.values.astype(float)], axis=1)

    bin_cols = ["gender"]
    if use_m_stage and "m_stage_bin" in clin.columns:
        bin_cols.append("m_stage_bin")
    X_bin_all = clin[bin_cols].astype(int).values

    X_cat_all = clin[["Overall.Stage", "T_stage", "N_stage"]].astype(str)

    # fit transformers on TRAIN
    scaler = StandardScaler()
    X_cont_train = scaler.fit_transform(X_cont_all[clin.index.isin(ids_train)])
    X_cont_val = scaler.transform(X_cont_all[clin.index.isin(ids_val)])
    X_cont_test = scaler.transform(X_cont_all[clin.index.isin(ids_test)])

    ohe = OneHotEncoder(categories=[_STAGE_CATS, _T_CATS, _N_CATS], sparse_output=False, handle_unknown="ignore")
    X_cat_train = ohe.fit_transform(X_cat_all.loc[ids_train])
    X_cat_val = ohe.transform(X_cat_all.loc[ids_val])
    X_cat_test = ohe.transform(X_cat_all.loc[ids_test])

    def concat(Xc, Xb, Xo):
        return np.concatenate([Xc, Xb, Xo], axis=1)

    X_train = concat(X_cont_train, X_bin_all[clin.index.isin(ids_train)], X_cat_train)
    X_val = concat(X_cont_val, X_bin_all[clin.index.isin(ids_val)], X_cat_val)
    X_test = concat(X_cont_test, X_bin_all[clin.index.isin(ids_test)], X_cat_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


# -----------------------------
# Models
# -----------------------------

def fit_eval_coxnet(X_train, y_train, X_val, y_val, X_test, y_test) -> Tuple[dict, np.ndarray]:
    ytr_e, ytr_t = y_train["event"], y_train["time"]
    yva_e, yva_t = y_val["event"], y_val["time"]
    yte_e, yte_t = y_test["event"], y_test["time"]

    best = {"val_c": -1.0}

    for l1 in (0.1, 0.5, 0.9):
        model = CoxnetSurvivalAnalysis(l1_ratio=float(l1), alpha_min_ratio=1e-4, n_alphas=200, max_iter=100000)
        model.fit(X_train, y_train)
        for a in model.alphas_:
            pred_val = model.predict(X_val, alpha=a)
            c_val = cindex_value(yva_e, yva_t, pred_val)
            if c_val > best["val_c"]:
                pred_train = model.predict(X_train, alpha=a)
                pred_test = model.predict(X_test, alpha=a)
                best = {
                    "l1_ratio": float(l1),
                    "alpha": float(a),
                    "train_c": cindex_value(ytr_e, ytr_t, pred_train),
                    "val_c": c_val,
                    "test_c": cindex_value(yte_e, yte_t, pred_test),
                }

    # NOTE: risk on test using the best alpha (model already trained on train)
    model_best = CoxnetSurvivalAnalysis(l1_ratio=best["l1_ratio"], alpha_min_ratio=1e-4, n_alphas=200, max_iter=100000)
    model_best.fit(X_train, y_train)
    risk_test = model_best.predict(X_test, alpha=best["alpha"])

    return best, risk_test


def _gbsa_grid() -> List[dict]:
    grid = []
    for n_estimators in [100, 200, 400]:
        for lr in [0.01, 0.05]:
            for max_depth in [1, 2]:
                for min_leaf in [5, 10]:
                    grid.append(
                        dict(
                            n_estimators=n_estimators,
                            learning_rate=lr,
                            max_depth=max_depth,
                            min_samples_leaf=min_leaf,
                            random_state=0,
                        )
                    )
    return grid


def fit_eval_gbsa(X_train, y_train, X_val, y_val, X_test, y_test, do_grid: bool) -> Tuple[dict, np.ndarray]:
    ytr_e, ytr_t = y_train["event"], y_train["time"]
    yva_e, yva_t = y_val["event"], y_val["time"]
    yte_e, yte_t = y_test["event"], y_test["time"]

    params_list = _gbsa_grid() if do_grid else [dict(random_state=0)]

    best = {"val_c": -1.0}
    best_params = None

    for params in params_list:
        model = GradientBoostingSurvivalAnalysis(**params)
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        c_val = cindex_value(yva_e, yva_t, pred_val)
        if c_val > best["val_c"]:
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
            best = {
                "params": params,
                "train_c": cindex_value(ytr_e, ytr_t, pred_train),
                "val_c": c_val,
                "test_c": cindex_value(yte_e, yte_t, pred_test),
            }
            best_params = params

    # refit best on train
    model_best = GradientBoostingSurvivalAnalysis(**best_params)
    model_best.fit(X_train, y_train)
    risk_test = model_best.predict(X_test)

    return best, risk_test


def _build_mlp(in_features: int, layers: List[int], dropout: float):
    import torch.nn as nn

    modules = []
    n_in = in_features
    for n_out in layers:
        modules.append(nn.Linear(n_in, n_out))
        modules.append(nn.BatchNorm1d(n_out))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Dropout(p=dropout))
        n_in = n_out
    modules.append(nn.Linear(n_in, 1))
    return nn.Sequential(*modules)


def fit_eval_deepsurv(X_train, y_train, X_val, y_val, X_test, y_test, seed: int = 0) -> Tuple[dict, np.ndarray]:
    if not _HAS_PYCOX:
        raise RuntimeError("pycox not installed. pip install pycox torchtuples torch")

    np.random.seed(seed)
    torch.manual_seed(seed)

    net = _build_mlp(X_train.shape[1], layers=[64, 32], dropout=0.3)
    model = PyCoxPH(net, tt.optim.Adam(lr=1e-3, weight_decay=1e-4))

    ytr = (y_train["time"].astype(np.float32), y_train["event"].astype(np.float32))
    yva = (y_val["time"].astype(np.float32), y_val["event"].astype(np.float32))

    callbacks = [tt.callbacks.EarlyStopping(patience=20)]
    log = model.fit(
        X_train.astype(np.float32),
        ytr,
        batch_size=64,
        epochs=256,
        callbacks=callbacks,
        verbose=False,
        val_data=(X_val.astype(np.float32), yva),
    )

    risk_train = model.predict(X_train.astype(np.float32)).reshape(-1)
    risk_val = model.predict(X_val.astype(np.float32)).reshape(-1)
    risk_test = model.predict(X_test.astype(np.float32)).reshape(-1)

    ytr_e, ytr_t = y_train["event"], y_train["time"]
    yva_e, yva_t = y_val["event"], y_val["time"]
    yte_e, yte_t = y_test["event"], y_test["time"]

    res = {
        "train_c": cindex_value(ytr_e, ytr_t, risk_train),
        "val_c": cindex_value(yva_e, yva_t, risk_val),
        "test_c": cindex_value(yte_e, yte_t, risk_test),
        "epochs_ran": int(len(log.to_pandas())),
    }
    return res, risk_test


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_csv", required=True)
    ap.add_argument("--radiomics_csv", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--mstage_mode", default="binary_nonzero", choices=["binary_nonzero", "drop"])

    ap.add_argument("--radiomics_mode", default="topk", choices=["topk", "all", "highest10"])
    ap.add_argument("--radiomics_topk", type=int, default=50)
    ap.add_argument("--corr_threshold", type=float, default=0.75)

    ap.add_argument("--use_deepsurv", type=int, default=0)
    ap.add_argument("--gbsa_grid", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    _ensure_dir(args.out_dir)

    split = load_fixed_split(args.split_json)
    clin = build_clinical_table(args.clinical_csv, split, mstage_mode=args.mstage_mode)
    rad = load_radiomics_table(args.radiomics_csv)

    # align radiomics to clinical index
    rad = rad.reindex(clin.index)

    ids_train_present = [i for i in split.ids_train if i in clin.index]
    y_train_for_sel = Surv.from_arrays(
        event=clin.loc[ids_train_present, "event"].values.astype(bool),
        time=clin.loc[ids_train_present, "time"].values.astype(float),
    )

    radiomics_cols = select_radiomics_features(
        rad=rad,
        split=split,
        y_event=y_train_for_sel["event"],
        y_time=y_train_for_sel["time"],
        mode=args.radiomics_mode,
        topk=args.radiomics_topk,
        corr_threshold=args.corr_threshold,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = make_design_matrices(
        clin=clin,
        rad=rad,
        split=split,
        radiomics_cols=radiomics_cols,
        use_m_stage=(args.mstage_mode != "drop"),
    )

    print("=== Fixed-split survival benchmark (boost v2) ===")
    print(f"Train/Val/Test: {len(split.ids_train)}/{len(split.ids_val)}/{len(split.ids_test)}")
    print(f"Radiomics selected: {len(radiomics_cols)} | Total design dim: {X_train.shape[1]}")
    print(
        f"Fingerprints: train={_fingerprint_ids(split.ids_train)} val={_fingerprint_ids(split.ids_val)} test={_fingerprint_ids(split.ids_test)}"
    )

    results = {
        "radiomics_mode": args.radiomics_mode,
        "radiomics_topk": args.radiomics_topk,
        "corr_threshold": args.corr_threshold,
        "mstage_mode": args.mstage_mode,
        "selected_radiomics_cols": radiomics_cols,
        "models": {},
    }

    # Coxnet
    cox_res, cox_risk_test = fit_eval_coxnet(X_train, y_train, X_val, y_val, X_test, y_test)
    print(
        f"[coxnet] train={cox_res['train_c']:.4f} val={cox_res['val_c']:.4f} test={cox_res['test_c']:.4f} (l1={cox_res['l1_ratio']}, alpha={cox_res['alpha']:.3g})"
    )
    results["models"]["coxnet"] = cox_res

    # GBSA
    gbsa_res, gbsa_risk_test = fit_eval_gbsa(
        X_train, y_train, X_val, y_val, X_test, y_test, do_grid=bool(args.gbsa_grid)
    )
    print(
        f"[gbsa]   train={gbsa_res['train_c']:.4f} val={gbsa_res['val_c']:.4f} test={gbsa_res['test_c']:.4f} (grid={bool(args.gbsa_grid)})"
    )
    results["models"]["gbsa"] = gbsa_res

    # DeepSurv
    if args.use_deepsurv:
        if not _HAS_PYCOX:
            print(
                "[deepsurv] SKIPPED (pycox not installed). Install: pip install pycox torchtuples torch"
            )
        else:
            ds_res, ds_risk_test = fit_eval_deepsurv(
                X_train, y_train, X_val, y_val, X_test, y_test, seed=args.seed
            )
            print(
                f"[deepsurv] train={ds_res['train_c']:.4f} val={ds_res['val_c']:.4f} test={ds_res['test_c']:.4f} (epochs_ran={ds_res['epochs_ran']})"
            )
            results["models"]["deepsurv"] = ds_res

    # Simple 2-model ensemble: choose w on VAL
    yva_e, yva_t = y_val["event"], y_val["time"]

    # Need val risk: re-use fitted best hyperparams by re-fitting on train (cheap)
    cox_tmp = CoxnetSurvivalAnalysis(l1_ratio=cox_res["l1_ratio"], alpha_min_ratio=1e-4, n_alphas=200, max_iter=100000)
    cox_tmp.fit(X_train, y_train)
    cox_risk_val = cox_tmp.predict(X_val, alpha=cox_res["alpha"])

    gbsa_params = gbsa_res.get("params", {"random_state": 0})
    gbsa_tmp = GradientBoostingSurvivalAnalysis(**gbsa_params)
    gbsa_tmp.fit(X_train, y_train)
    gbsa_risk_val = gbsa_tmp.predict(X_val)

    best_w, best_c = 0.5, -1.0
    for w in np.linspace(0, 1, 21):
        r = w * cox_risk_val + (1 - w) * gbsa_risk_val
        c = cindex_value(yva_e, yva_t, r)
        if c > best_c:
            best_c = c
            best_w = float(w)

    yte_e, yte_t = y_test["event"], y_test["time"]
    ens_test = best_w * cox_risk_test + (1 - best_w) * gbsa_risk_test
    ens_res = {
        "w_coxnet": best_w,
        "val_c": best_c,
        "test_c": cindex_value(yte_e, yte_t, ens_test),
    }
    print(f"[ens2 ]  val={ens_res['val_c']:.4f} test={ens_res['test_c']:.4f} (w={ens_res['w_coxnet']:.2f})")
    results["models"]["ensemble2"] = ens_res

    # Save
    out_json = os.path.join(args.out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    out_feats = os.path.join(args.out_dir, "selected_radiomics_cols.txt")
    with open(out_feats, "w") as f:
        for c in radiomics_cols:
            f.write(c + "\n")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_feats}")


if __name__ == "__main__":
    main()
