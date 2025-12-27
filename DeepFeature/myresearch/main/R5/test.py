#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
R5 (final): Nested-CV Cox models with robust preprocessing and risk-direction handling.

Key points:
1) CoxnetSurvivalAnalysis requires 0 < l1_ratio <= 1 in current sksurv/sklearn validation. (So l1_ratio=0 can't go Coxnet)
2) Ridge-only (l1_ratio==0) is handled by CoxPHSurvivalAnalysis(alpha=...).
3) Risk direction:
   - C-index expects larger estimate => higher risk => shorter survival time.
   - If --allow_flip (or --risk_sign auto), we choose risk sign (+1 or -1) using INNER-CV only (no outer-val leakage).
4) Final refit uses fixed alpha (no interpolation).
5) Optional fallback for Coxnet all-zero coefficients: step to smaller alpha (less regularization) along the path.
"""

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning

from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw


# ---------------------------
# IO utils
# ---------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def mean_std(arr: List[float]) -> Tuple[float, float]:
    x = np.asarray([v for v in arr if v is not None and np.isfinite(v)], dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x.mean()), 0.0
    return float(x.mean()), float(x.std(ddof=1))


# ---------------------------
# Survival utils
# ---------------------------

def make_surv_struct(event: np.ndarray, time_arr: np.ndarray) -> np.ndarray:
    e = event.astype(bool)
    t = time_arr.astype(float)
    y = np.empty(len(t), dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = e
    y["time"] = t
    return y

def compute_tau_from_train(y_train: np.ndarray, quantile: float = 0.9) -> float:
    times = y_train["time"].astype(float)
    events = y_train["event"].astype(bool)
    ev_times = times[events]
    if ev_times.shape[0] >= 5:
        tau = float(np.quantile(ev_times, quantile))
    else:
        tau = float(np.quantile(times, quantile))
    max_time = float(np.max(times))
    if tau >= max_time:
        tau = max_time - 1e-6
    return tau

def safe_uno_cindex(y_train: np.ndarray, y_val: np.ndarray, risk_val: np.ndarray, tau: float) -> float:
    try:
        return float(concordance_index_ipcw(y_train, y_val, risk_val, tau=tau)[0])
    except Exception:
        return float("nan")

def harrell_cindex(y: np.ndarray, risk: np.ndarray) -> float:
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])

def score_with_sign(metric: str, y_tr: np.ndarray, y_va: np.ndarray, risk_raw: np.ndarray, sign: int, tau_q: float) -> float:
    r = (int(sign) * risk_raw).astype(float)
    if metric == "uno":
        tau = compute_tau_from_train(y_tr, quantile=tau_q)
        return safe_uno_cindex(y_tr, y_va, r, tau=tau)
    return harrell_cindex(y_va, r)


# ---------------------------
# Preprocess utils
# ---------------------------

def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    transformers = []

    if len(num_cols) > 0:
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))

    if len(cat_cols) > 0:
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    if len(transformers) == 0:
        raise ValueError("No num_cols/cat_cols to preprocess.")

    try:
        return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)
    except TypeError:
        return ColumnTransformer(transformers=transformers, remainder="drop")

def get_feature_names(prep: ColumnTransformer) -> Optional[List[str]]:
    try:
        if hasattr(prep, "get_feature_names_out"):
            return [str(x) for x in prep.get_feature_names_out()]
        if hasattr(prep, "get_feature_names"):
            return [str(x) for x in prep.get_feature_names()]
        return None
    except Exception:
        return None


# ---------------------------
# Data loading & typing
# ---------------------------

def load_and_merge(features_csv: Path,
                   labels_csv: Path,
                   id_col: str,
                   time_col: str,
                   event_col: str) -> pd.DataFrame:
    X_df = pd.read_csv(features_csv).copy()
    y_df = pd.read_csv(labels_csv).copy()

    X_df[id_col] = X_df[id_col].astype(str)
    y_df[id_col] = y_df[id_col].astype(str)

    if time_col not in y_df.columns or event_col not in y_df.columns:
        raise ValueError(f"labels_csv missing columns: need [{time_col}, {event_col}]")

    if time_col in X_df.columns:
        X_df = X_df.drop(columns=[time_col])
    if event_col in X_df.columns:
        X_df = X_df.drop(columns=[event_col])

    df = X_df.merge(y_df[[id_col, time_col, event_col]], on=id_col, how="inner")

    df = df.replace(r"^\s*$", np.nan, regex=True)

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")

    df = df.dropna(subset=[time_col, event_col]).copy()
    df = df[df[time_col] > 0].copy()

    df[event_col] = df[event_col].fillna(0).astype(int)
    df[event_col] = (df[event_col] > 0).astype(int)
    return df

def select_feature_cols(df: pd.DataFrame,
                        id_col: str,
                        time_col: str,
                        event_col: str,
                        feature_cols_arg: Optional[List[str]]) -> List[str]:
    if feature_cols_arg and len(feature_cols_arg) > 0:
        cols = [c for c in feature_cols_arg if c in df.columns]
        if len(cols) == 0:
            raise ValueError(f"--feature_cols provided but none exist in CSV: {feature_cols_arg}")
        return cols
    excluded = {id_col, time_col, event_col}
    return [c for c in df.columns if c not in excluded]

def infer_num_cat_cols(df: pd.DataFrame,
                       feature_cols: List[str],
                       explicit_cat_cols: Optional[List[str]],
                       numeric_frac_thresh: float = 0.9) -> Tuple[List[str], List[str]]:
    if explicit_cat_cols is not None:
        cat_cols = [c for c in explicit_cat_cols if c in feature_cols]
        num_cols = [c for c in feature_cols if c not in cat_cols]
        return num_cols, cat_cols

    num_cols, cat_cols = [], []
    for c in feature_cols:
        s_num = pd.to_numeric(df[c], errors="coerce")
        frac = float(s_num.notna().mean())
        if frac >= numeric_frac_thresh:
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols

def enforce_types_inplace(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> None:
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat_cols:
        s = df[c].astype("string").str.strip()
        df[c] = s.replace({"": pd.NA})


# ---------------------------
# Inner CV selection (supports Coxnet + Ridge-CoxPH)
# ---------------------------

@dataclass
class BestParams:
    kind: str              # "coxnet" or "coxph_ridge"
    l1_ratio: float        # 0.0 for ridge branch; >0 for coxnet
    alpha: float
    inner_score: float
    risk_sign: int         # +1 or -1
    total_conv_warnings: int = 0

def _warn_count(wlist) -> int:
    return int(sum(1 for ww in wlist if issubclass(ww.category, ConvergenceWarning)))

def fit_alpha_path_coxnet(X: np.ndarray, y: np.ndarray,
                          l1_ratio: float, n_alphas: int,
                          alpha_min_ratio: Union[str, float],
                          tol: float, max_iter: int) -> Tuple[np.ndarray, int]:
    model = CoxnetSurvivalAnalysis(
        n_alphas=n_alphas,
        alphas=None,
        alpha_min_ratio=alpha_min_ratio,
        l1_ratio=float(l1_ratio),
        tol=tol,
        max_iter=max_iter,
        normalize=False,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.fit(X, y)
        n_warn = _warn_count(w)

    alphas = np.asarray(model.alphas_, dtype=float)
    alphas = alphas[np.isfinite(alphas)]
    if alphas.size > 0:
        alphas = np.unique(alphas)
        alphas = np.sort(alphas)[::-1]
    return alphas, n_warn

def inner_select_coxnet(X_all: np.ndarray, y_all: np.ndarray,
                        l1_ratio: float, inner_folds: int, seed: int,
                        n_alphas: int, alpha_min_ratio: Union[str, float],
                        tol: float, max_iter: int,
                        metric: str, tau_q: float,
                        sign_mode: str) -> Optional[BestParams]:
    if not (float(l1_ratio) > 0.0):
        return None

    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    alphas, warn0 = fit_alpha_path_coxnet(X_all, y_all, l1_ratio, n_alphas, alpha_min_ratio, tol, max_iter)
    if alphas.size == 0:
        return None

    # accumulate mean scores for sign=+1 and sign=-1 separately
    sum_pos = np.zeros_like(alphas, dtype=float)
    cnt_pos = np.zeros_like(alphas, dtype=float)
    sum_neg = np.zeros_like(alphas, dtype=float)
    cnt_neg = np.zeros_like(alphas, dtype=float)
    total_warn = int(warn0)

    for tr_idx, va_idx in kf.split(X_all):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        model = CoxnetSurvivalAnalysis(
            n_alphas=len(alphas),
            alphas=alphas,
            l1_ratio=float(l1_ratio),
            tol=tol,
            max_iter=max_iter,
            normalize=False,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X_tr, y_tr)
            total_warn += _warn_count(w)

        # score each alpha
        for i, a in enumerate(alphas):
            try:
                risk_raw = model.predict(X_va, alpha=float(a)).astype(float)
                if sign_mode in ("auto",):
                    sc_pos = score_with_sign(metric, y_tr, y_va, risk_raw, +1, tau_q)
                    sc_neg = score_with_sign(metric, y_tr, y_va, risk_raw, -1, tau_q)
                    if np.isfinite(sc_pos):
                        sum_pos[i] += sc_pos; cnt_pos[i] += 1
                    if np.isfinite(sc_neg):
                        sum_neg[i] += sc_neg; cnt_neg[i] += 1
                elif sign_mode == "+1":
                    sc = score_with_sign(metric, y_tr, y_va, risk_raw, +1, tau_q)
                    if np.isfinite(sc):
                        sum_pos[i] += sc; cnt_pos[i] += 1
                elif sign_mode == "-1":
                    sc = score_with_sign(metric, y_tr, y_va, risk_raw, -1, tau_q)
                    if np.isfinite(sc):
                        sum_neg[i] += sc; cnt_neg[i] += 1
            except Exception:
                continue

    mean_pos = sum_pos / np.maximum(cnt_pos, 1.0)
    mean_neg = sum_neg / np.maximum(cnt_neg, 1.0)

    best = None
    # choose best (alpha, sign) by mean score
    if sign_mode in ("auto",):
        j_pos = int(np.nanargmax(mean_pos)) if np.isfinite(mean_pos).any() else None
        j_neg = int(np.nanargmax(mean_neg)) if np.isfinite(mean_neg).any() else None
        cand = []
        if j_pos is not None:
            cand.append((float(mean_pos[j_pos]), j_pos, +1))
        if j_neg is not None:
            cand.append((float(mean_neg[j_neg]), j_neg, -1))
        if len(cand) == 0:
            return None
        cand.sort(key=lambda x: x[0], reverse=True)
        score_star, j_star, sign_star = cand[0]
        best = BestParams(kind="coxnet", l1_ratio=float(l1_ratio), alpha=float(alphas[j_star]),
                          inner_score=float(score_star), risk_sign=int(sign_star),
                          total_conv_warnings=int(total_warn))
    else:
        # fixed sign
        if sign_mode == "+1":
            if not np.isfinite(mean_pos).any():
                return None
            j = int(np.nanargmax(mean_pos))
            best = BestParams(kind="coxnet", l1_ratio=float(l1_ratio), alpha=float(alphas[j]),
                              inner_score=float(mean_pos[j]), risk_sign=+1, total_conv_warnings=int(total_warn))
        else:
            if not np.isfinite(mean_neg).any():
                return None
            j = int(np.nanargmax(mean_neg))
            best = BestParams(kind="coxnet", l1_ratio=float(l1_ratio), alpha=float(alphas[j]),
                              inner_score=float(mean_neg[j]), risk_sign=-1, total_conv_warnings=int(total_warn))

    return best

def inner_select_ridge_coxph(X_all: np.ndarray, y_all: np.ndarray,
                             inner_folds: int, seed: int,
                             alpha_grid: np.ndarray,
                             coxph_tol: float, coxph_max_iter: int, ties: str,
                             metric: str, tau_q: float,
                             sign_mode: str) -> Optional[BestParams]:
    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)

    sum_pos = np.zeros(len(alpha_grid), dtype=float)
    cnt_pos = np.zeros(len(alpha_grid), dtype=float)
    sum_neg = np.zeros(len(alpha_grid), dtype=float)
    cnt_neg = np.zeros(len(alpha_grid), dtype=float)
    total_warn = 0

    for tr_idx, va_idx in kf.split(X_all):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        for j, a in enumerate(alpha_grid):
            try:
                est = CoxPHSurvivalAnalysis(alpha=float(a), ties=ties, tol=coxph_tol, n_iter=int(coxph_max_iter))
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    est.fit(X_tr, y_tr)
                    total_warn += _warn_count(w)

                risk_raw = est.predict(X_va).astype(float)

                if sign_mode in ("auto",):
                    sc_pos = score_with_sign(metric, y_tr, y_va, risk_raw, +1, tau_q)
                    sc_neg = score_with_sign(metric, y_tr, y_va, risk_raw, -1, tau_q)
                    if np.isfinite(sc_pos):
                        sum_pos[j] += sc_pos; cnt_pos[j] += 1
                    if np.isfinite(sc_neg):
                        sum_neg[j] += sc_neg; cnt_neg[j] += 1
                elif sign_mode == "+1":
                    sc = score_with_sign(metric, y_tr, y_va, risk_raw, +1, tau_q)
                    if np.isfinite(sc):
                        sum_pos[j] += sc; cnt_pos[j] += 1
                elif sign_mode == "-1":
                    sc = score_with_sign(metric, y_tr, y_va, risk_raw, -1, tau_q)
                    if np.isfinite(sc):
                        sum_neg[j] += sc; cnt_neg[j] += 1
            except Exception:
                continue

    mean_pos = sum_pos / np.maximum(cnt_pos, 1.0)
    mean_neg = sum_neg / np.maximum(cnt_neg, 1.0)

    if sign_mode in ("auto",):
        cand = []
        if np.isfinite(mean_pos).any():
            j_pos = int(np.nanargmax(mean_pos))
            cand.append((float(mean_pos[j_pos]), j_pos, +1))
        if np.isfinite(mean_neg).any():
            j_neg = int(np.nanargmax(mean_neg))
            cand.append((float(mean_neg[j_neg]), j_neg, -1))
        if len(cand) == 0:
            return None
        cand.sort(key=lambda x: x[0], reverse=True)
        score_star, j_star, sign_star = cand[0]
        return BestParams(kind="coxph_ridge", l1_ratio=0.0, alpha=float(alpha_grid[j_star]),
                          inner_score=float(score_star), risk_sign=int(sign_star),
                          total_conv_warnings=int(total_warn))
    else:
        if sign_mode == "+1":
            if not np.isfinite(mean_pos).any():
                return None
            j = int(np.nanargmax(mean_pos))
            return BestParams(kind="coxph_ridge", l1_ratio=0.0, alpha=float(alpha_grid[j]),
                              inner_score=float(mean_pos[j]), risk_sign=+1, total_conv_warnings=int(total_warn))
        else:
            if not np.isfinite(mean_neg).any():
                return None
            j = int(np.nanargmax(mean_neg))
            return BestParams(kind="coxph_ridge", l1_ratio=0.0, alpha=float(alpha_grid[j]),
                              inner_score=float(mean_neg[j]), risk_sign=-1, total_conv_warnings=int(total_warn))

def select_best_params_inner_cv(X_all: np.ndarray, y_all: np.ndarray,
                                l1_ratios: List[float],
                                inner_folds: int, seed: int,
                                n_alphas: int, alpha_min_ratio: Union[str, float],
                                tol: float, max_iter: int,
                                ridge_alpha_min: float, ridge_alpha_max: float, ridge_n_alphas: int,
                                coxph_tol: float, coxph_max_iter: int, ties: str,
                                metric: str, tau_q: float,
                                sign_mode: str,
                                enable_ridge: bool) -> BestParams:
    best: Optional[BestParams] = None

    # ridge alpha grid
    if ridge_alpha_min <= 0 or ridge_alpha_max <= 0:
        raise ValueError("ridge_alpha_min/max must be > 0")
    if ridge_alpha_min > ridge_alpha_max:
        ridge_alpha_min, ridge_alpha_max = ridge_alpha_max, ridge_alpha_min
    ridge_grid = np.logspace(np.log10(ridge_alpha_min), np.log10(ridge_alpha_max), int(ridge_n_alphas)).astype(float)

    for l1 in l1_ratios:
        l1 = float(l1)

        cand: Optional[BestParams] = None
        if l1 == 0.0:
            if not enable_ridge:
                continue
            cand = inner_select_ridge_coxph(
                X_all, y_all,
                inner_folds=inner_folds, seed=seed,
                alpha_grid=ridge_grid,
                coxph_tol=coxph_tol, coxph_max_iter=coxph_max_iter, ties=ties,
                metric=metric, tau_q=tau_q,
                sign_mode=sign_mode
            )
        else:
            # Coxnet (elastic net): requires l1_ratio > 0
            cand = inner_select_coxnet(
                X_all, y_all,
                l1_ratio=l1,
                inner_folds=inner_folds, seed=seed,
                n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio,
                tol=tol, max_iter=max_iter,
                metric=metric, tau_q=tau_q,
                sign_mode=sign_mode
            )

        if cand is None:
            continue
        if best is None or cand.inner_score > best.inner_score:
            best = cand

    if best is None:
        raise RuntimeError("Inner selection failed for all candidates.")
    return best


# ---------------------------
# Coxnet fallback for all-zero solution
# ---------------------------

def count_nonzero_coef_coxnet(model: CoxnetSurvivalAnalysis, eps: float = 1e-10) -> int:
    coef = np.asarray(model.coef_, dtype=float)
    if coef.ndim == 1:
        coef = coef.reshape(-1, 1)
    v = coef[:, 0]
    return int(np.sum(np.abs(v) > eps))

def refit_coxnet_fixed_alpha(X_tr: np.ndarray, y_tr: np.ndarray,
                             l1_ratio: float, alpha: float,
                             tol: float, max_iter: int) -> Tuple[CoxnetSurvivalAnalysis, int]:
    model = CoxnetSurvivalAnalysis(
        n_alphas=1,
        alphas=np.asarray([float(alpha)], dtype=float),
        l1_ratio=float(l1_ratio),
        tol=tol,
        max_iter=max_iter,
        normalize=False,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.fit(X_tr, y_tr)
        n_warn = _warn_count(w)
    return model, int(n_warn)

def fallback_alpha_if_all_zero_coxnet(X_tr: np.ndarray, y_tr: np.ndarray,
                                     best_l1: float, best_alpha: float,
                                     n_alphas: int, alpha_min_ratio: Union[str, float],
                                     tol: float, max_iter: int,
                                     min_nonzero: int, max_steps: int) -> Tuple[float, int, int]:
    alphas, warn0 = fit_alpha_path_coxnet(X_tr, y_tr, best_l1, n_alphas, alpha_min_ratio, tol, max_iter)
    if alphas.size == 0:
        return float(best_alpha), 0, int(warn0)

    i0 = int(np.argmin(np.abs(alphas - float(best_alpha))))
    warn_total = int(warn0)

    for k in range(i0, min(len(alphas), i0 + max_steps + 1)):
        a = float(alphas[k])
        model_k, wn = refit_coxnet_fixed_alpha(X_tr, y_tr, best_l1, a, tol, max_iter)
        warn_total += wn
        nnz = count_nonzero_coef_coxnet(model_k)
        if nnz >= int(min_nonzero):
            return a, int(k - i0), int(warn_total)

    a_last = float(alphas[min(len(alphas) - 1, i0 + max_steps)])
    return a_last, int(min(max_steps, len(alphas) - 1 - i0)), int(warn_total)


# ---------------------------
# Main outer CV
# ---------------------------

def run_r5(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    np.random.seed(args.seed)

    # effective sign mode
    sign_mode = args.risk_sign
    if args.allow_flip:
        sign_mode = "auto"

    df = load_and_merge(Path(args.features_csv), Path(args.labels_csv),
                        id_col=args.id_col, time_col=args.time_col, event_col=args.event_col)

    feature_cols = select_feature_cols(df, args.id_col, args.time_col, args.event_col, args.feature_cols)

    num_cols, cat_cols = infer_num_cat_cols(df, feature_cols, args.cat_cols, numeric_frac_thresh=args.numeric_frac_thresh)
    enforce_types_inplace(df, num_cols=num_cols, cat_cols=cat_cols)

    print(f"[R5] n={len(df)} | raw_features={len(feature_cols)} | num={len(num_cols)} cat={len(cat_cols)}")
    if len(cat_cols) > 0:
        print(f"[R5] cat_cols={cat_cols[:10]}{' ...' if len(cat_cols) > 10 else ''}")
    print(f"[R5] risk_sign_mode={sign_mode} | enable_ridge={args.enable_ridge}")

    splits = read_json(Path(args.splits_json))
    fold_keys = sorted(list(splits.keys()))

    all_fold_metrics = []
    oof_rows = []

    for fold_name in fold_keys:
        t0 = time.time()
        train_ids = [str(x) for x in splits[fold_name]["train"]]
        val_ids = [str(x) for x in splits[fold_name]["val"]]

        df_tr = df[df[args.id_col].isin(train_ids)].copy()
        df_va = df[df[args.id_col].isin(val_ids)].copy()
        if df_tr.shape[0] == 0 or df_va.shape[0] == 0:
            raise ValueError(f"[R5][{fold_name}] empty split: train={df_tr.shape[0]}, val={df_va.shape[0]}")

        X_tr_raw = df_tr[feature_cols].copy()
        X_va_raw = df_va[feature_cols].copy()
        y_tr = make_surv_struct(df_tr[args.event_col].to_numpy(), df_tr[args.time_col].to_numpy())
        y_va = make_surv_struct(df_va[args.event_col].to_numpy(), df_va[args.time_col].to_numpy())

        prep = build_preprocessor(num_cols, cat_cols)
        X_tr = prep.fit_transform(X_tr_raw)
        X_va = prep.transform(X_va_raw)

        n_feat = int(X_tr.shape[1])
        fold_idx = int(str(fold_name).split("_")[-1]) if "fold_" in str(fold_name) else fold_keys.index(fold_name)
        print(f"[R5][{fold_name}] train={len(df_tr)} val={len(df_va)} | raw={len(feature_cols)} -> preprocessed={n_feat}")

        # inner select (kind + alpha + sign)
        best = select_best_params_inner_cv(
            X_all=X_tr, y_all=y_tr,
            l1_ratios=args.l1_ratios,
            inner_folds=args.inner_folds,
            seed=args.seed + 1000 + fold_idx,
            n_alphas=args.n_alphas,
            alpha_min_ratio=args.alpha_min_ratio,
            tol=args.tol,
            max_iter=args.max_iter,
            ridge_alpha_min=args.ridge_alpha_min,
            ridge_alpha_max=args.ridge_alpha_max,
            ridge_n_alphas=args.ridge_n_alphas,
            coxph_tol=args.coxph_tol,
            coxph_max_iter=args.coxph_max_iter,
            ties=args.coxph_ties,
            metric=args.inner_metric,
            tau_q=args.tau_q,
            sign_mode=sign_mode,
            enable_ridge=args.enable_ridge,
        )

        alpha_used = float(best.alpha)
        sign_used = int(best.risk_sign)
        fallback_steps = 0
        conv_warn_fallback = 0
        conv_warn_final = 0
        nnz = None

        # final fit
        if best.kind == "coxnet":
            # optional fallback if all-zero
            if args.fallback_to_nonzero_alpha:
                tmp_model, wn0 = refit_coxnet_fixed_alpha(X_tr, y_tr, best.l1_ratio, alpha_used, args.tol, args.max_iter)
                nnz0 = count_nonzero_coef_coxnet(tmp_model)
                if nnz0 < args.fallback_min_nonzero:
                    alpha_used, fallback_steps, conv_warn_fallback = fallback_alpha_if_all_zero_coxnet(
                        X_tr, y_tr,
                        best_l1=best.l1_ratio,
                        best_alpha=best.alpha,
                        n_alphas=args.n_alphas,
                        alpha_min_ratio=args.alpha_min_ratio,
                        tol=args.tol,
                        max_iter=args.max_iter,
                        min_nonzero=args.fallback_min_nonzero,
                        max_steps=args.fallback_max_steps,
                    )
                    print(f"[WARNING][R5][{fold_name}] all-zero at alpha={best.alpha:.6g}. "
                          f"Fallback -> alpha_used={alpha_used:.6g} (steps={fallback_steps})")

            model, conv_warn_final = refit_coxnet_fixed_alpha(X_tr, y_tr, best.l1_ratio, alpha_used, args.tol, args.max_iter)
            nnz = count_nonzero_coef_coxnet(model)
            risk_raw = model.predict(X_va).astype(float)  # fixed alpha path => no need pass alpha
            kind = "coxnet"

        else:
            # ridge CoxPH
            est = CoxPHSurvivalAnalysis(alpha=float(alpha_used), ties=args.coxph_ties, tol=args.coxph_tol, n_iter=int(args.coxph_max_iter))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                est.fit(X_tr, y_tr)
                conv_warn_final = _warn_count(w)
            model = est
            # CoxPH has coef_ (dense)
            nnz = int(np.sum(np.abs(np.asarray(model.coef_, dtype=float)) > 1e-10))
            risk_raw = model.predict(X_va).astype(float)
            kind = "coxph_ridge"

        risk_pred = (sign_used * risk_raw).astype(float)

        # metrics (outer val) – diagnostics only; do NOT use these to change sign
        har_raw = harrell_cindex(y_va, risk_raw)
        har_flip = harrell_cindex(y_va, -risk_raw)
        har_final = harrell_cindex(y_va, risk_pred)

        tau = compute_tau_from_train(y_tr, quantile=args.tau_q)
        uno_final = safe_uno_cindex(y_tr, y_va, risk_pred, tau=tau)

        if abs(har_flip - har_raw) > 0.05:
            better = "(-risk_raw)" if har_flip > har_raw else "(risk_raw)"
            print(f"[DIAG][{fold_name}] C(risk_raw)={har_raw:.4f}, C(-risk_raw)={har_flip:.4f} | better={better}")
            print(f"[DIAG] C(risk)={har_raw:.4f}, C(-risk)={har_flip:.4f}")

        fold_dir = outdir / f"fold_{fold_idx}"
        ensure_dir(fold_dir)

        pred_df = pd.DataFrame({
            args.id_col: df_va[args.id_col].astype(str).to_numpy(),
            "time": y_va["time"].astype(float),
            "event": y_va["event"].astype(bool),
            "risk_raw": risk_raw.astype(float),
            "risk_pred": risk_pred.astype(float),   # this is the one you should use for fusion
        })
        pred_df.to_csv(fold_dir / "predictions.csv", index=False)

        if args.save_model:
            joblib.dump({"prep": prep, "model": model, "kind": kind, "risk_sign": int(sign_used)}, fold_dir / "model.joblib")

        # coefficients (apply sign so coef meaning aligns with risk_pred)
        try:
            feat_names = get_feature_names(prep)

            if kind == "coxnet":
                coef = np.asarray(model.coef_, dtype=float)
                if coef.ndim == 1:
                    coef = coef.reshape(-1, 1)
                coef_vec = coef[:, 0] * float(sign_used)
            else:
                coef_vec = np.asarray(model.coef_, dtype=float).reshape(-1) * float(sign_used)

            if feat_names is not None and len(feat_names) == len(coef_vec):
                coef_dict = {n: float(v) for n, v in zip(feat_names, coef_vec)}
            else:
                coef_dict = {f"feature_{i}": float(v) for i, v in enumerate(coef_vec)}

            write_json(fold_dir / "coefficients.json", {
                "kind": kind,
                "selected_l1_ratio": float(best.l1_ratio),
                "selected_alpha": float(best.alpha),
                "alpha_used_final": float(alpha_used),
                "risk_sign_used": int(sign_used),
                "fallback_steps": int(fallback_steps),
                "n_features": int(len(coef_vec)),
                "n_nonzero_coef": int(nnz),
                "sparsity_ratio": round(1.0 - nnz / max(1, len(coef_vec)), 6),
                "feature_names_available": bool(feat_names is not None),
                "coefficients": coef_dict,
            })
        except Exception as e:
            print(f"[WARNING][R5][{fold_name}] cannot dump coefficients: {e}")

        metrics = {
            "exp_id": "R5",
            "fold": int(fold_idx),
            "fold_name": str(fold_name),
            "kind": kind,
            "n_train": int(df_tr.shape[0]),
            "n_val": int(df_va.shape[0]),
            "n_features_raw": int(len(feature_cols)),
            "n_features_preprocessed": int(n_feat),
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "selected_l1_ratio": float(best.l1_ratio),
            "selected_alpha": float(best.alpha),
            "alpha_used_final": float(alpha_used),
            "risk_sign_used": int(sign_used),
            "inner_best_score": float(best.inner_score),
            "inner_metric": str(args.inner_metric),
            "conv_warnings_inner_total": int(best.total_conv_warnings),
            "conv_warnings_fallback": int(conv_warn_fallback),
            "conv_warnings_final_fit": int(conv_warn_final),
            "fallback_enabled": bool(args.fallback_to_nonzero_alpha),
            "fallback_steps": int(fallback_steps),
            "fallback_min_nonzero": int(args.fallback_min_nonzero),
            "n_nonzero_coef_final": int(nnz),
            "harrell_c_raw": round(float(har_raw), 6),
            "harrell_c_flip_raw": round(float(har_flip), 6),
            "harrell_c_final": round(float(har_final), 6),
            "uno_c_final": None if np.isnan(uno_final) else round(float(uno_final), 6),
            "tau": round(float(tau), 6),
            "tau_q": float(args.tau_q),
            "runtime_seconds": round(float(time.time() - t0), 2),
        }
        write_json(fold_dir / "metrics.json", metrics)
        all_fold_metrics.append(metrics)

        for r in pred_df.itertuples(index=False):
            oof_rows.append({
                args.id_col: getattr(r, args.id_col),
                "fold": int(fold_idx),
                "time": float(r.time),
                "event": bool(r.event),
                "risk_pred": float(r.risk_pred),
            })

        print(f"[R5][{fold_name}] HarrellC(final)={har_final:.4f} UnoC(final)={uno_final:.4f} | "
              f"kind={kind} l1={best.l1_ratio} alpha={best.alpha:.6g} sign={sign_used} nnz={nnz}")

    # oof + summary
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(outdir / "oof_predictions.csv", index=False)

    met_df = pd.DataFrame(all_fold_metrics)
    met_df.to_csv(outdir / "per_fold_metrics.csv", index=False)

    har_vals = met_df["harrell_c_final"].astype(float).tolist()
    uno_vals = [float(x) if x is not None else np.nan for x in met_df["uno_c_final"].tolist()]

    har_mean, har_std = mean_std(har_vals)
    uno_mean, uno_std = mean_std(uno_vals)

    har_oof = float(concordance_index_censored(
        oof_df["event"].to_numpy().astype(bool),
        oof_df["time"].to_numpy().astype(float),
        oof_df["risk_pred"].to_numpy().astype(float)
    )[0])

    y_oof = make_surv_struct(oof_df["event"].to_numpy().astype(int), oof_df["time"].to_numpy().astype(float))
    tau_oof = compute_tau_from_train(y_oof, quantile=args.tau_q)
    uno_oof = safe_uno_cindex(y_oof, y_oof, oof_df["risk_pred"].to_numpy().astype(float), tau=tau_oof)

    summary = {
        "exp_id": "R5",
        "n_folds": int(len(met_df)),
        "harrell_c_mean": round(float(har_mean), 6),
        "harrell_c_std": round(float(har_std), 6),
        "uno_c_mean": None if np.isnan(uno_mean) else round(float(uno_mean), 6),
        "uno_c_std": None if np.isnan(uno_std) else round(float(uno_std), 6),
        "harrell_c_oof": round(float(har_oof), 6),
        "uno_c_oof": None if np.isnan(uno_oof) else round(float(uno_oof), 6),
        "l1_ratios_candidates": [float(x) for x in args.l1_ratios],
        "inner_folds": int(args.inner_folds),
        "inner_metric": str(args.inner_metric),
        "risk_sign_mode": str("auto" if args.allow_flip else args.risk_sign),
        "enable_ridge": bool(args.enable_ridge),
        "note": "R5 final: nested-CV selects (kind, alpha, sign) using inner-CV only; outer-val is for reporting only.",
    }
    write_json(outdir / "summary.json", summary)

    print("\n" + "=" * 72)
    print("[R5] Done.")
    print(f"[R5] Per-fold avg: HarrellC={summary['harrell_c_mean']:.4f}±{summary['harrell_c_std']:.4f} | "
          f"UnoC={summary['uno_c_mean']}±{summary['uno_c_std']}")
    print(f"[R5] Overall OOF: HarrellC={summary['harrell_c_oof']:.4f} | UnoC={summary['uno_c_oof']}")
    print(f"[R5] Outputs: {outdir.resolve()}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("R5 (final): Coxnet + Ridge-CoxPH with nested CV and inner-only sign selection")

    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--id_col", type=str, default="PatientID")
    ap.add_argument("--time_col", type=str, default="Survival.time")
    ap.add_argument("--event_col", type=str, default="deadstatus.event")

    ap.add_argument("--feature_cols", type=str, nargs="*", default=None)

    ap.add_argument("--cat_cols", type=str, nargs="*", default=None,
                    help="Categorical columns. Omit flag => heuristic; pass flag alone => no categoricals.")

    ap.add_argument("--numeric_frac_thresh", type=float, default=0.9)

    ap.add_argument("--l1_ratios", type=float, nargs="+",
                    default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
                    help="Candidates. 0.0 triggers ridge CoxPH branch; >0 triggers Coxnet branch.")

    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--inner_metric", type=str, default="harrell", choices=["harrell", "uno"])

    # Coxnet path
    ap.add_argument("--n_alphas", type=int, default=100)
    ap.add_argument("--alpha_min_ratio", type=str, default="auto")
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--max_iter", type=int, default=100000)

    # Ridge CoxPH path
    ap.add_argument("--enable_ridge", dest="enable_ridge", action="store_true",
                    help="Enable ridge CoxPH branch when l1_ratio==0.0 is included.")
    ap.add_argument("--disable_ridge", dest="enable_ridge", action="store_false",
                    help="Disable ridge branch (skip l1_ratio==0.0).")
    ap.set_defaults(enable_ridge=True)

    ap.add_argument("--ridge_alpha_min", type=float, default=1e-4)
    ap.add_argument("--ridge_alpha_max", type=float, default=1e4)
    ap.add_argument("--ridge_n_alphas", type=int, default=60)

    ap.add_argument("--coxph_tol", type=float, default=1e-9)
    ap.add_argument("--coxph_max_iter", type=int, default=100)
    ap.add_argument("--coxph_ties", type=str, default="efron", choices=["efron", "breslow"])

    ap.add_argument("--tau_q", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)

    # risk sign
    ap.add_argument("--risk_sign", type=str, default="+1", choices=["auto", "+1", "-1"],
                    help="Risk sign applied to model.predict(). Use 'auto' only with inner-CV (no leakage).")
    ap.add_argument("--allow_flip", action="store_true",
                    help="Alias for --risk_sign auto (inner-CV chooses sign).")

    # Coxnet fallback controls
    ap.add_argument("--fallback_to_nonzero_alpha", dest="fallback_to_nonzero_alpha", action="store_true",
                    help="Enable fallback if Coxnet coefficients are all-zero at selected alpha.")
    ap.add_argument("--no_fallback_to_nonzero_alpha", dest="fallback_to_nonzero_alpha", action="store_false",
                    help="Disable fallback mechanism.")
    ap.set_defaults(fallback_to_nonzero_alpha=True)

    ap.add_argument("--fallback_min_nonzero", type=int, default=1)
    ap.add_argument("--fallback_max_steps", type=int, default=10)

    ap.add_argument("--save_model", action="store_true")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    try:
        if args.alpha_min_ratio != "auto":
            args.alpha_min_ratio = float(args.alpha_min_ratio)
    except Exception:
        pass
    run_r5(args)
