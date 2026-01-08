#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
surv_2y_gbsa_tune_v4.py

- CV tune on TRAIN ONLY (Stratified by event).
- Objective: Brier@2y (IPCW) primary, Dynamic AUC@2y tie-breaker.
- Final calibration: OOF Platt scaling (logit + LogisticRegression) on TRAIN ONLY.
- Internal test evaluation on frozen TEST split.

Compatible with sklearn==1.3.2 and scikit-survival==0.22.x (handles metric return-order diffs).
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import joblib

from sksurv.util import Surv
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    cumulative_dynamic_auc,
)

# -----------------------
# Utilities
# -----------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _read_lines(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out

def _parse_split_ids(split_obj: Any) -> Tuple[List[str], List[str]]:
    """
    Supports multiple split json formats.
    Returns (train_ids, test_ids).
    """
    # Case 1: {"splits": {"train": [...], "test":[...]}}  (optionally "val")
    if isinstance(split_obj, dict) and "splits" in split_obj and isinstance(split_obj["splits"], dict):
        s = split_obj["splits"]
        if "train" in s and "test" in s:
            train = list(s["train"])
            if "val" in s and isinstance(s["val"], list):
                train = train + list(s["val"])  # use train+val as training pool
            test = list(s["test"])
            return train, test

    # Case 2: {"train": [...], "test": [...]}
    if isinstance(split_obj, dict) and "train" in split_obj and "test" in split_obj:
        return list(split_obj["train"]), list(split_obj["test"])

    # Case 3: {"train_ids": [...], "test_ids": [...]}
    if isinstance(split_obj, dict) and "train_ids" in split_obj and "test_ids" in split_obj:
        return list(split_obj["train_ids"]), list(split_obj["test_ids"])

    raise ValueError("Unrecognized split_json format. Expect keys like splits/train/test or train_ids/test_ids.")

def _make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scale: str,
) -> ColumnTransformer:
    if scale == "minmax":
        scaler = MinMaxScaler()
    elif scale == "zscore":
        scaler = StandardScaler()
    elif scale == "none":
        scaler = None
    else:
        raise ValueError(f"Unknown --scale: {scale}")

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scaler is not None:
        from sklearn.pipeline import Pipeline
        num_steps.append(("scale", scaler))
        num_pipe = Pipeline(num_steps)
    else:
        from sklearn.pipeline import Pipeline
        num_pipe = Pipeline(num_steps)

    from sklearn.pipeline import Pipeline
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_onehot()),
        ]
    )

    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ct

def _to_y_struct(event: np.ndarray, time: np.ndarray):
    return Surv.from_arrays(event.astype(bool), time.astype(float))

def _known_status_mask(event: np.ndarray, time: np.ndarray, horizon: float) -> np.ndarray:
    event = event.astype(bool)
    time = time.astype(float)
    return (event & (time <= horizon)) | (time >= horizon)

def _label_survive_at_horizon(event: np.ndarray, time: np.ndarray, horizon: float) -> np.ndarray:
    event = event.astype(bool)
    time = time.astype(float)
    died_by = event & (time <= horizon)
    return (~died_by).astype(int)

def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def _fit_platt(p_surv_raw: np.ndarray, y_survive: np.ndarray, C: float, seed: int) -> LogisticRegression:
    z = _logit(p_surv_raw)
    lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, random_state=seed)
    lr.fit(z.reshape(-1, 1), y_survive.astype(int))
    return lr

def _apply_platt(cal: LogisticRegression, p_surv_raw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    z = _logit(p_surv_raw, eps=eps)
    p = cal.predict_proba(z.reshape(-1, 1))[:, 1]
    return np.clip(p, eps, 1.0 - eps)

def _predict_surv_at_horizon(model: GradientBoostingSurvivalAnalysis, X: np.ndarray, horizon: float) -> np.ndarray:
    sf = model.predict_survival_function(X)
    
    # Handle list of callable functions (newer sksurv versions)
    if isinstance(sf, list) and len(sf) > 0 and callable(sf[0]):
        return np.asarray([float(fn(horizon)) for fn in sf], dtype=float)
    
    # Handle numpy array of StepFunction objects (sksurv 0.22.2)
    if isinstance(sf, np.ndarray) and sf.ndim == 1:
        # Check if elements are callable (StepFunction objects)
        if len(sf) > 0 and callable(sf[0]):
            return np.asarray([float(fn(horizon)) for fn in sf], dtype=float)
    
    # Handle 2D array format
    arr = np.asarray(sf)
    if arr.ndim != 2:
        raise TypeError(f"Unexpected output of predict_survival_function; expected list or 2D array. Got type={type(sf)}, shape={getattr(sf, 'shape', 'N/A')}, ndim={arr.ndim}")
    if not hasattr(model, "event_times_"):
        raise AttributeError("Model output is array but has no event_times_.")
    times = np.asarray(model.event_times_, dtype=float)
    idx = np.searchsorted(times, horizon, side="right") - 1
    idx = int(np.clip(idx, 0, len(times) - 1))
    return arr[:, idx].astype(float)

def _safe_brier_at_horizon(y_train, y_test, p_surv: np.ndarray, horizon: float) -> float:
    est = np.column_stack([p_surv.astype(float)])
    out = brier_score(y_train, y_test, est, times=[horizon])
    a, b = out
    a0 = float(np.ravel(a)[0])
    b0 = float(np.ravel(b)[0])
    if a0 > 1.0 and 0.0 <= b0 <= 1.0:
        bs = b
    elif b0 > 1.0 and 0.0 <= a0 <= 1.0:
        bs = a
    else:
        bs = b
    return float(np.ravel(bs)[0])

def _safe_auc_at_horizon(y_train, y_test, risk: np.ndarray, horizon: float) -> float:
    out = cumulative_dynamic_auc(y_train, y_test, risk.astype(float), times=[horizon])
    a, b = out
    a0 = float(np.ravel(a)[0])
    b0 = float(np.ravel(b)[0]) if np.ravel(b).size > 0 else float(b)
    if 0.0 <= a0 <= 1.0 and b0 > 1.0:
        aucs = a
    elif 0.0 <= b0 <= 1.0 and a0 > 1.0:
        aucs = b
    else:
        aucs = a if 0.0 <= a0 <= 1.0 else b
    return float(np.ravel(aucs)[0])

def _cindex(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> float:
    return float(concordance_index_censored(event.astype(bool), time.astype(float), risk.astype(float))[0])

# -----------------------
# Data loading
# -----------------------

def _load_merged_df(clinical_csv: str, radiomics_csv: str) -> pd.DataFrame:
    clin = pd.read_csv(clinical_csv)
    rad = pd.read_csv(radiomics_csv)
    if "PatientID" not in clin.columns or "PatientID" not in rad.columns:
        raise ValueError("Both clinical_csv and radiomics_csv must contain 'PatientID'.")
    df = pd.merge(clin, rad, on="PatientID", how="inner", validate="one_to_one")
    return df

def _filter_split(df: pd.DataFrame, train_ids: List[str], test_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df[df["PatientID"].isin(set(train_ids))].copy()
    df_test = df[df["PatientID"].isin(set(test_ids))].copy()
    if df_train.empty or df_test.empty:
        raise RuntimeError(f"Empty split after filtering: train={len(df_train)} test={len(df_test)}")
    return df_train, df_test

# -----------------------
# Core: CV tuning
# -----------------------

@dataclass
class CVResult:
    params: Dict[str, Any]
    mean_cindex: float
    mean_auc: float
    mean_brier: float
    std_cindex: float
    std_auc: float
    std_brier: float

def _default_param_grid() -> List[Dict[str, Any]]:
    # small, non-overfitting grid: 24 combos
    grid = []
    for lr in [0.03, 0.05, 0.1]:
        for depth in [2, 3]:
            for leaf in [10, 20]:
                for subs in [0.6, 0.8]:
                    grid.append(
                        dict(
                            learning_rate=lr,
                            max_depth=depth,
                            min_samples_leaf=leaf,
                            subsample=subs,
                            n_estimators=200,
                            max_features=0.5,
                            max_leaf_nodes=12,
                        )
                    )
    return grid

def _eval_params_cv(
    df_train: pd.DataFrame,
    selected_radiomics: List[str],
    base_params: Dict[str, Any],
    params_override: Dict[str, Any],
    horizon: float,
    scale: str,
    cv_folds: int,
    seed: int,
    platt_C_for_cv: float,
) -> CVResult:
    numeric_cols = ["age"] + selected_radiomics
    categorical_cols = ["gender", "Histology", "clinical.T.Stage", "Overall.Stage"]
    feat_cols = numeric_cols + categorical_cols

    event = df_train["deadstatus.event"].astype(int).values
    time = df_train["Survival.time"].astype(float).values

    X_raw = df_train[feat_cols].copy()
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    c_list, auc_list, b_list = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_raw, event)):
        Xtr_raw = X_raw.iloc[tr_idx].copy()
        Xva_raw = X_raw.iloc[va_idx].copy()
        ev_tr, ti_tr = event[tr_idx], time[tr_idx]
        ev_va, ti_va = event[va_idx], time[va_idx]
        y_tr = _to_y_struct(ev_tr, ti_tr)
        y_va = _to_y_struct(ev_va, ti_va)

        prep = _build_preprocessor(numeric_cols, categorical_cols, scale=scale)
        Xtr = prep.fit_transform(Xtr_raw)
        Xva = prep.transform(Xva_raw)

        params = dict(base_params)
        params.update(params_override)
        params["random_state"] = int(seed + fold)

        model = GradientBoostingSurvivalAnalysis(**params)
        model.fit(Xtr, y_tr)

        p_tr_raw = _predict_surv_at_horizon(model, Xtr, horizon)
        p_va_raw = _predict_surv_at_horizon(model, Xva, horizon)

        known_tr = _known_status_mask(ev_tr, ti_tr, horizon)
        y2y_tr = _label_survive_at_horizon(ev_tr, ti_tr, horizon)[known_tr]
        cal = _fit_platt(p_tr_raw[known_tr], y2y_tr, C=platt_C_for_cv, seed=int(seed + fold))
        p_va_cal = _apply_platt(cal, p_va_raw)

        risk_va = model.predict(Xva)
        c_list.append(_cindex(ev_va, ti_va, risk_va))
        auc_list.append(_safe_auc_at_horizon(y_tr, y_va, risk_va, horizon))
        b_list.append(_safe_brier_at_horizon(y_tr, y_va, p_va_cal, horizon))

    c_arr, auc_arr, b_arr = np.asarray(c_list), np.asarray(auc_list), np.asarray(b_list)
    return CVResult(
        params=params_override,
        mean_cindex=float(np.mean(c_arr)),
        mean_auc=float(np.mean(auc_arr)),
        mean_brier=float(np.mean(b_arr)),
        std_cindex=float(np.std(c_arr, ddof=1)) if len(c_arr) > 1 else 0.0,
        std_auc=float(np.std(auc_arr, ddof=1)) if len(auc_arr) > 1 else 0.0,
        std_brier=float(np.std(b_arr, ddof=1)) if len(b_arr) > 1 else 0.0,
    )

def _train_final_bundle(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    selected_radiomics: List[str],
    base_params: Dict[str, Any],
    best_override: Dict[str, Any],
    horizon: float,
    scale: str,
    seed: int,
    cv_folds_for_cal: int,
    platt_C: float,
) -> Dict[str, Any]:
    numeric_cols = ["age"] + selected_radiomics
    categorical_cols = ["gender", "Histology", "clinical.T.Stage", "Overall.Stage"]
    feat_cols = numeric_cols + categorical_cols

    ev_tr = df_train["deadstatus.event"].astype(int).values
    ti_tr = df_train["Survival.time"].astype(float).values
    y_tr = _to_y_struct(ev_tr, ti_tr)

    ev_te = df_test["deadstatus.event"].astype(int).values
    ti_te = df_test["Survival.time"].astype(float).values
    y_te = _to_y_struct(ev_te, ti_te)

    Xtr_raw = df_train[feat_cols].copy()
    Xte_raw = df_test[feat_cols].copy()

    prep = _build_preprocessor(numeric_cols, categorical_cols, scale=scale)
    Xtr = prep.fit_transform(Xtr_raw)
    Xte = prep.transform(Xte_raw)

    params = dict(base_params)
    params.update(best_override)
    params["random_state"] = int(seed)

    model = GradientBoostingSurvivalAnalysis(**params)
    model.fit(Xtr, y_tr)

    # OOF predictions for calibrator
    skf = StratifiedKFold(n_splits=cv_folds_for_cal, shuffle=True, random_state=seed)
    oof_p = np.full(len(df_train), np.nan, dtype=float)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(Xtr_raw, ev_tr)):
        prep_f = _build_preprocessor(numeric_cols, categorical_cols, scale=scale)
        Xtr_f = prep_f.fit_transform(Xtr_raw.iloc[tr_idx].copy())
        Xva_f = prep_f.transform(Xtr_raw.iloc[va_idx].copy())
        y_tr_f = _to_y_struct(ev_tr[tr_idx], ti_tr[tr_idx])

        params_f = dict(params)
        params_f["random_state"] = int(seed + 1000 + fold)

        m_f = GradientBoostingSurvivalAnalysis(**params_f)
        m_f.fit(Xtr_f, y_tr_f)
        oof_p[va_idx] = _predict_surv_at_horizon(m_f, Xva_f, horizon)

    known = _known_status_mask(ev_tr, ti_tr, horizon)
    y2y = _label_survive_at_horizon(ev_tr, ti_tr, horizon)[known]
    cal = _fit_platt(oof_p[known], y2y, C=platt_C, seed=seed)

    risk_te = model.predict(Xte)
    p_te_raw = _predict_surv_at_horizon(model, Xte, horizon)
    p_te_cal = _apply_platt(cal, p_te_raw)

    report = {
        "horizon_days": float(horizon),
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "internal_test": {
            "cindex": _cindex(ev_te, ti_te, risk_te),
            "auc_dynamic_2y": _safe_auc_at_horizon(y_tr, y_te, risk_te, horizon),
            "brier_ipcw_2y_raw": _safe_brier_at_horizon(y_tr, y_te, p_te_raw, horizon),
            "brier_ipcw_2y_cal": _safe_brier_at_horizon(y_tr, y_te, p_te_cal, horizon),
        },
        "best_params_override": best_override,
    }

    pred_df = pd.DataFrame(
        {
            "PatientID": df_test["PatientID"].values,
            "time": ti_te,
            "event": ev_te,
            "risk": risk_te.astype(float),
            "surv2y_raw": p_te_raw.astype(float),
            "surv2y_cal": p_te_cal.astype(float),
        }
    )

    bundle = {
        "preprocessor": prep,
        "model": model,
        "calibrator": cal,
        "meta": {
            "horizon": float(horizon),
            "scale": scale,
            "seed": int(seed),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "selected_radiomics": selected_radiomics,
            "base_params": base_params,
            "best_override": best_override,
            "cv_folds_for_cal": int(cv_folds_for_cal),
            "platt_C": float(platt_C),
        },
        "report": report,
        "internal_test_pred_df": pred_df,
    }
    return bundle

# -----------------------
# CLI
# -----------------------

def cmd_tune(args: argparse.Namespace) -> None:
    _ensure_dir(args.out_dir)

    split_obj = _read_json(args.split_json)
    train_ids, test_ids = _parse_split_ids(split_obj)

    df = _load_merged_df(args.clinical_csv, args.radiomics_csv)
    df_train, df_test = _filter_split(df, train_ids, test_ids)

    selected_radiomics = _read_lines(args.selected_radiomics_txt)
    missing = [f for f in selected_radiomics if f not in df.columns]
    if missing:
        raise RuntimeError(f"{len(missing)} selected radiomics features not found. Example: {missing[:5]}")

    base_params: Dict[str, Any] = {}
    if args.gbsa_base_params_json:
        base_params = _read_json(args.gbsa_base_params_json)
        if not isinstance(base_params, dict):
            raise ValueError("gbsa_base_params_json must be a dict-like JSON.")

    base_params.setdefault("loss", "coxph")
    base_params.setdefault("criterion", "friedman_mse")

    grid = _default_param_grid()

    cv_rows = []
    best: Optional[CVResult] = None
    for i, override in enumerate(grid):
        res = _eval_params_cv(
            df_train=df_train,
            selected_radiomics=selected_radiomics,
            base_params=base_params,
            params_override=override,
            horizon=float(args.horizon),
            scale=args.scale,
            cv_folds=int(args.cv_folds),
            seed=int(args.seed),
            platt_C_for_cv=float(args.platt_C_cv),
        )

        cv_rows.append(
            {
                "i": i,
                **override,
                "mean_brier": res.mean_brier,
                "std_brier": res.std_brier,
                "mean_auc": res.mean_auc,
                "std_auc": res.std_auc,
                "mean_cindex": res.mean_cindex,
                "std_cindex": res.std_cindex,
            }
        )

        if best is None:
            best = res
        else:
            if (res.mean_brier < best.mean_brier - 1e-6) or (
                abs(res.mean_brier - best.mean_brier) <= 1e-6 and res.mean_auc > best.mean_auc + 1e-6
            ):
                best = res

        print(
            f"[{i+1:02d}/{len(grid)}] "
            f"brier={res.mean_brier:.4f} auc={res.mean_auc:.4f} cindex={res.mean_cindex:.4f} "
            f"| {override}"
        )

    assert best is not None

    cv_df = pd.DataFrame(cv_rows).sort_values(["mean_brier", "mean_auc"], ascending=[True, False])
    cv_df.to_csv(os.path.join(args.out_dir, "cv_results.csv"), index=False)

    with open(os.path.join(args.out_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best.params, f, indent=2)

    print("\n=== BEST (CV) ===")
    print(f"mean_brier={best.mean_brier:.4f} ± {best.std_brier:.4f}")
    print(f"mean_auc  ={best.mean_auc:.4f} ± {best.std_auc:.4f}")
    print(f"mean_cidx ={best.mean_cindex:.4f} ± {best.std_cindex:.4f}")

    bundle = _train_final_bundle(
        df_train=df_train,
        df_test=df_test,
        selected_radiomics=selected_radiomics,
        base_params=base_params,
        best_override=best.params,
        horizon=float(args.horizon),
        scale=args.scale,
        seed=int(args.seed),
        cv_folds_for_cal=int(args.calibrate_oof_folds),
        platt_C=float(args.platt_C),
    )

    joblib.dump(
        {
            "preprocessor": bundle["preprocessor"],
            "model": bundle["model"],
            "calibrator": bundle["calibrator"],
            "meta": bundle["meta"],
            "report": bundle["report"],
        },
        os.path.join(args.out_dir, "gbsa_2y_bundle.joblib"),
    )
    bundle["internal_test_pred_df"].to_csv(os.path.join(args.out_dir, "internal_test_predictions.csv"), index=False)
    with open(os.path.join(args.out_dir, "internal_test_report.json"), "w", encoding="utf-8") as f:
        json.dump(bundle["report"], f, indent=2)

    it = bundle["report"]["internal_test"]
    print("\n=== INTERNAL TEST (frozen split) ===")
    print(f"cindex        = {it['cindex']:.4f}")
    print(f"auc_dynamic   = {it['auc_dynamic_2y']:.4f}")
    print(f"brier_raw     = {it['brier_ipcw_2y_raw']:.4f}")
    print(f"brier_cal     = {it['brier_ipcw_2y_cal']:.4f}")
    print(f"[OK] Saved: {args.out_dir}")

def cmd_eval(args: argparse.Namespace) -> None:
    _ensure_dir(args.out_dir)

    bundle = joblib.load(args.bundle_joblib)
    prep = bundle["preprocessor"]
    model = bundle["model"]
    cal = bundle["calibrator"]
    meta = bundle["meta"]

    horizon = float(meta["horizon"])

    split_obj = _read_json(args.split_json)
    train_ids, test_ids = _parse_split_ids(split_obj)

    df = _load_merged_df(args.clinical_csv, args.radiomics_csv)
    df_train, df_test = _filter_split(df, train_ids, test_ids)

    feat_cols = meta["numeric_cols"] + meta["categorical_cols"]

    ev_tr = df_train["deadstatus.event"].astype(int).values
    ti_tr = df_train["Survival.time"].astype(float).values
    y_tr = _to_y_struct(ev_tr, ti_tr)

    ev_te = df_test["deadstatus.event"].astype(int).values
    ti_te = df_test["Survival.time"].astype(float).values
    y_te = _to_y_struct(ev_te, ti_te)

    Xte = prep.transform(df_test[feat_cols].copy())

    risk_te = model.predict(Xte)
    p_te_raw = _predict_surv_at_horizon(model, Xte, horizon)
    p_te_cal = _apply_platt(cal, p_te_raw)

    report = {
        "internal_test": {
            "cindex": _cindex(ev_te, ti_te, risk_te),
            "auc_dynamic_2y": _safe_auc_at_horizon(y_tr, y_te, risk_te, horizon),
            "brier_ipcw_2y_raw": _safe_brier_at_horizon(y_tr, y_te, p_te_raw, horizon),
            "brier_ipcw_2y_cal": _safe_brier_at_horizon(y_tr, y_te, p_te_cal, horizon),
        }
    }

    with open(os.path.join(args.out_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pd.DataFrame(
        {
            "PatientID": df_test["PatientID"].values,
            "time": ti_te,
            "event": ev_te,
            "risk": risk_te.astype(float),
            "surv2y_raw": p_te_raw.astype(float),
            "surv2y_cal": p_te_cal.astype(float),
        }
    ).to_csv(os.path.join(args.out_dir, "eval_internal_test_predictions.csv"), index=False)

    it = report["internal_test"]
    print(f"[EVAL] cindex={it['cindex']:.4f} auc2y={it['auc_dynamic_2y']:.4f} brier2y_cal={it['brier_ipcw_2y_cal']:.4f}")
    print(f"[OK] Saved: {args.out_dir}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_tune = sub.add_parser("tune")
    p_tune.add_argument("--clinical_csv", required=True)
    p_tune.add_argument("--radiomics_csv", required=True)
    p_tune.add_argument("--split_json", required=True)
    p_tune.add_argument("--selected_radiomics_txt", required=True)
    p_tune.add_argument("--gbsa_base_params_json", default=None)
    p_tune.add_argument("--out_dir", required=True)
    p_tune.add_argument("--seed", type=int, default=91)
    p_tune.add_argument("--horizon", type=float, default=730.0)
    p_tune.add_argument("--scale", choices=["minmax", "zscore", "none"], default="minmax")
    p_tune.add_argument("--cv_folds", type=int, default=5)
    p_tune.add_argument("--calibrate_oof_folds", type=int, default=5)
    p_tune.add_argument("--platt_C", type=float, default=10.0)
    p_tune.add_argument("--platt_C_cv", type=float, default=10.0)
    p_tune.set_defaults(func=cmd_tune)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--bundle_joblib", required=True)
    p_eval.add_argument("--clinical_csv", required=True)
    p_eval.add_argument("--radiomics_csv", required=True)
    p_eval.add_argument("--split_json", required=True)
    p_eval.add_argument("--out_dir", required=True)
    p_eval.set_defaults(func=cmd_eval)

    return p

def main() -> None:
    args = build_argparser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
