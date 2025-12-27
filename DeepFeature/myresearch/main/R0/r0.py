#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.feature_selection import VarianceThreshold

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis


# ----------------------------
# Logging
# ----------------------------
def setup_logger(outdir: Path, verbose: bool = True) -> logging.Logger:
    outdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("R0")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # console
    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # file
    fh = logging.FileHandler(outdir / "r0.log", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # avoid duplicate handlers if re-imported
    logger.propagate = False
    return logger


# ----------------------------
# Utils: normal CDF without scipy
# ----------------------------
def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ----------------------------
# Tau computation (based on ORIGINAL y_train)
# ----------------------------
def compute_tau_from_train(y_train, quantile: float = 0.9) -> float:
    t = y_train["time"].astype(float)
    e = y_train["event"].astype(bool)
    event_times = t[e]
    if len(event_times) >= 5:
        tau = float(np.quantile(event_times, quantile))
    else:
        tau = float(np.quantile(t, quantile))
    max_t = float(np.max(t))
    if tau >= max_t:
        tau = max_t - 1e-6
    return tau


def safe_uno_cindex(y_train, y_val, risk_val, tau: float) -> float:
    try:
        # concordance_index_ipcw(train_survival, test_survival, estimate, tau=...)
        # returns (cindex, concordant, discordant, tied_risk, tied_time) depending on version
        return float(concordance_index_ipcw(y_train, y_val, risk_val, tau=tau)[0])
    except Exception:
        return float("nan")


def count_model_nonzero_coefs(model) -> int:
    """
    Count non-zero coefficients in the fitted model.
    For Coxnet: need to count only the selected alpha column.
    For CoxPH: coef_ is 1D array.
    """
    try:
        if hasattr(model, "model_") and hasattr(model.model_, "coef_"):
            coefs = model.model_.coef_
            
            # Check if this is Coxnet (2D coef matrix)
            if coefs.ndim == 2:
                # Coxnet: coef_ is (n_features, n_alphas)
                # Need to find the selected alpha
                if hasattr(model, "alpha_"):
                    alpha_idx = np.where(np.abs(model.model_.alphas_ - model.alpha_) < 1e-10)[0]
                    if len(alpha_idx) > 0:
                        return int(np.sum(coefs[:, alpha_idx[0]] != 0.0))
                # Fallback: use the last alpha
                return int(np.sum(coefs[:, -1] != 0.0))
            else:
                # CoxPH or other: 1D coefficient array
                return int(np.sum(coefs != 0.0))
    except Exception:
        pass
    return 0


# ----------------------------
# Transformers
# ----------------------------
class SafeVarianceThreshold(BaseEstimator, TransformerMixin):
    """
    VarianceThreshold but guarantees at least 1 feature remains.
    If all removed, keep the single feature with max variance.
    """
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self._vt = VarianceThreshold(threshold=threshold)
        self.support_mask_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        self._vt.fit(X)
        mask = self._vt.get_support()
        if mask.sum() == 0:
            # keep max-variance feature
            variances = np.var(X, axis=0)
            j = int(np.argmax(variances))
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[j] = True
        self.support_mask_ = mask
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.support_mask_]


class SurvivalAwareCorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Optional: correlation filter that keeps one feature among highly correlated groups.
    For R0, NOT recommended by default (adds variance). Kept for completeness.
    Uses |corr| > corr_thresh; within a correlated cluster, keep feature with max |C-0.5|.
    """
    def __init__(self, corr_thresh: float = 0.85):
        self.corr_thresh = corr_thresh
        self.keep_indices_: Optional[np.ndarray] = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        if n_features <= 1:
            self.keep_indices_ = np.arange(n_features, dtype=int)
            return self

        # corr matrix
        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        abs_corr = np.abs(corr)

        # univariate survival "strength" score: |C - 0.5|
        strengths = np.zeros(n_features, dtype=float)
        for j in range(n_features):
            xj = X[:, j]
            try:
                c = concordance_index_censored(y["event"], y["time"], xj)[0]
                strengths[j] = abs(float(c) - 0.5)
            except Exception:
                strengths[j] = 0.0

        keep = np.ones(n_features, dtype=bool)
        for j in range(n_features):
            if not keep[j]:
                continue
            # find features strongly correlated with j
            group = np.where((abs_corr[j, :] > self.corr_thresh) & (np.arange(n_features) != j))[0]
            if group.size == 0:
                continue
            candidates = np.array([j] + group.tolist(), dtype=int)
            best = candidates[np.argmax(strengths[candidates])]
            # drop all others in candidates except best
            for k in candidates:
                if k != best:
                    keep[k] = False

        kept = np.where(keep)[0]
        if kept.size == 0:
            kept = np.array([int(np.argmax(strengths))], dtype=int)
        self.keep_indices_ = kept
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.keep_indices_]


# ----------------------------
# Coxnet wrapper to pick alpha and make .predict() work cleanly
# Coxnet predict supports alpha parameter (see docs).
# ----------------------------
class CoxnetWrapper(BaseEstimator):
    def __init__(
        self,
        l1_ratio: float = 0.5,
        alpha_min_ratio="auto",
        n_alphas: int = 100,
        min_nnz: int = 5,
        max_nnz: int = 50,
        max_iter: int = 100000,
        tol: float = 1e-7,
    ):
        self.l1_ratio = l1_ratio
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alphas = n_alphas
        self.min_nnz = min_nnz
        self.max_nnz = max_nnz
        self.max_iter = max_iter
        self.tol = tol

        self.model_: Optional[CoxnetSurvivalAnalysis] = None
        self.alpha_: Optional[float] = None

    def fit(self, X, y):
        self.model_ = CoxnetSurvivalAnalysis(
            l1_ratio=self.l1_ratio,
            alpha_min_ratio=self.alpha_min_ratio,
            n_alphas=self.n_alphas,
            max_iter=self.max_iter,
            tol=self.tol,
            normalize=False,
            fit_baseline_model=False,
        )
        self.model_.fit(X, y)

        coefs = self.model_.coef_  # (n_features, n_alphas)
        nnz = np.sum(coefs != 0.0, axis=0)

        # pick alpha with nnz within [min_nnz, max_nnz]
        ok = np.where((nnz >= self.min_nnz) & (nnz <= self.max_nnz))[0]
        if ok.size > 0:
            idx = int(ok[0])
        else:
            # fallback: choose alpha with nnz closest to mid target
            target = (self.min_nnz + self.max_nnz) / 2.0
            idx = int(np.argmin(np.abs(nnz - target)))

        self.alpha_ = float(self.model_.alphas_[idx])
        return self

    def predict(self, X):
        # CoxnetSurvivalAnalysis.predict(X, alpha=...) predicts risk scores
        return self.model_.predict(X, alpha=self.alpha_)


# ----------------------------
# IO: read splits.json
# ----------------------------
def load_splits(splits_json: Path) -> Dict[str, Dict[str, List[str]]]:
    with open(splits_json, "r") as f:
        splits = json.load(f)
    # expected: { "fold_0": {"train":[...], "val":[...]}, ...}
    return splits


# ----------------------------
# Data load and alignment
# ----------------------------
def load_and_align(
    features_csv: Path,
    labels_csv: Path,
    id_col: str,
    time_col: str,
    event_col: str,
    drop_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_df = pd.read_csv(features_csv).copy()
    y_df = pd.read_csv(labels_csv).copy()

    if id_col not in X_df.columns:
        raise ValueError(f"[features_csv] missing id_col={id_col}")
    if id_col not in y_df.columns:
        raise ValueError(f"[labels_csv] missing id_col={id_col}")
    if time_col not in y_df.columns:
        raise ValueError(f"[labels_csv] missing time_col={time_col}")
    if event_col not in y_df.columns:
        raise ValueError(f"[labels_csv] missing event_col={event_col}")

    # drop any potentially leaking columns from X
    for c in drop_cols:
        if c in X_df.columns:
            X_df.drop(columns=[c], inplace=True)

    # set index for alignment
    X_df[id_col] = X_df[id_col].astype(str)
    y_df[id_col] = y_df[id_col].astype(str)
    X_df = X_df.set_index(id_col)
    y_df = y_df.set_index(id_col)

    common = X_df.index.intersection(y_df.index)
    X_df = X_df.loc[common].copy()
    y_df = y_df.loc[common].copy()

    # keep only numeric features
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    return X_df, y_df


def make_y_struct(y_df: pd.DataFrame, time_col: str, event_col: str):
    t = pd.to_numeric(y_df[time_col], errors="coerce").astype(float).to_numpy()
    e_raw = y_df[event_col]
    # accept {0,1}, {False,True}, {"0","1"}, etc.
    e = pd.to_numeric(e_raw, errors="coerce").fillna(0).astype(int).to_numpy().astype(bool)
    return Surv.from_arrays(event=e, time=t)


# ----------------------------
# Verdict logic
# ----------------------------
@dataclass
class Verdict:
    flag: str
    reason: str
    mean: float
    std: float
    se: float
    z: float
    p_value: float


def compute_verdict(values: np.ndarray, alpha: float = 0.05, practical: float = 0.03, warning: float = 0.08) -> Verdict:
    """
    Heuristic-but-robust:
    - z-test for mean != 0.5 (normal approx)
    - practical threshold to avoid false alarms on tiny deviations
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n <= 1:
        return Verdict("WARNING", "Too few evaluations to judge.", float(np.mean(values)) if n else float("nan"),
                       float(np.std(values)) if n else float("nan"), float("nan"), 0.0, 1.0)

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    se = float(std / math.sqrt(n)) if std > 0 else 0.0
    z = float((mean - 0.5) / se) if se > 0 else 0.0
    p = float(2.0 * (1.0 - norm_cdf(abs(z)))) if se > 0 else 1.0

    if (p < alpha) and (abs(mean - 0.5) > practical):
        flag = "FLAG"
        reason = f"Significant deviation from 0.5 (p={p:.4g}, mean={mean:.4f}). Investigate leakage / fold-outside fit."
    elif abs(mean - 0.5) > warning:
        flag = "WARNING"
        reason = f"Large deviation from 0.5 (mean={mean:.4f}), but not statistically significant. Inspect pipeline."
    else:
        flag = "PASS"
        reason = f"Close to 0.5 as expected (mean={mean:.4f}). No obvious leakage signal."

    return Verdict(flag, reason, mean, std, se, z, p)


# ----------------------------
# Baselines
# ----------------------------
def eval_random_risk(y_val, n_trials: int, rng: np.random.Generator) -> Tuple[float, float]:
    vals = []
    for _ in range(n_trials):
        risk = rng.random(len(y_val))
        c = concordance_index_censored(y_val["event"], y_val["time"], risk)[0]
        vals.append(float(c))
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def build_pipeline(args, logger: logging.Logger):
    steps = []
    steps.append(("imputer", SimpleImputer(strategy=args.impute_strategy)))
    # IMPORTANT: variance filter must be BEFORE scaling; scaling makes variance ~1.
    if args.variance_thresh is not None and args.variance_thresh > 0:
        steps.append(("var", SafeVarianceThreshold(threshold=args.variance_thresh)))

    steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

    if args.use_corr_filter:
        logger.warning("Correlation filter ENABLED for R0. Not recommended unless you explicitly want to sanity-check it.")
        steps.append(("corr", SurvivalAwareCorrelationFilter(corr_thresh=args.corr_thresh)))

    if args.model == "coxph":
        # alpha is L2 penalty strength in sksurv CoxPHSurvivalAnalysis
        model = CoxPHSurvivalAnalysis(alpha=args.coxph_alpha)
    elif args.model == "coxnet":
        model = CoxnetWrapper(
            l1_ratio=args.l1_ratio,
            alpha_min_ratio=args.alpha_min_ratio,
            n_alphas=args.n_alphas,
            min_nnz=args.min_nnz,
            max_nnz=args.max_nnz,
            max_iter=args.max_iter,
            tol=args.tol,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    steps.append(("model", model))
    return Pipeline(steps)


# ----------------------------
# Main experiment loops
# ----------------------------
def run_perm_y(
    X_train, y_train, X_val, y_val,
    y_train_original,
    pipe: Pipeline,
    n_perm: int,
    tau_q: float,
    rng: np.random.Generator,
) -> List[Dict]:
    records = []
    tau = compute_tau_from_train(y_train_original, quantile=tau_q)
    
    # Track input dimensions
    n_feat_input = int(X_train.shape[1])

    for r in range(n_perm):
        perm_idx = rng.permutation(len(y_train))
        y_train_perm = y_train[perm_idx]

        pipe.fit(X_train, y_train_perm)
        risk_val = pipe.predict(X_val)

        har = float(concordance_index_censored(y_val["event"], y_val["time"], risk_val)[0])
        uno = safe_uno_cindex(y_train_original, y_val, risk_val, tau=tau)

        # Track feature dimensions at different stages
        try:
            X_val_t = pipe[:-1].transform(X_val)
            n_feat_after_pipeline = int(X_val_t.shape[1])
        except Exception:
            n_feat_after_pipeline = int(X_val.shape[1])
        
        # Count non-zero coefficients in the selected model
        model = pipe.named_steps.get("model", None)
        n_nonzero_coefs = count_model_nonzero_coefs(model) if model else 0

        records.append({
            "baseline": "perm_y",
            "rep": r,
            "harrell_c": har,
            "uno_c": uno,
            "tau": tau,
            "n_features_input": n_feat_input,
            "n_features_after_pipeline": n_feat_after_pipeline,
            "n_nonzero_coefs": n_nonzero_coefs,
        })
    return records


def run_shuffle_x(
    X_train, y_train, X_val, y_val,
    y_train_original,
    pipe: Pipeline,
    n_perm_x: int,
    tau_q: float,
    rng: np.random.Generator,
) -> List[Dict]:
    records = []
    tau = compute_tau_from_train(y_train_original, quantile=tau_q)
    
    # Track input dimensions
    n_feat_input = int(X_train.shape[1])

    for r in range(n_perm_x):
        perm_idx = rng.permutation(len(X_train))
        X_train_shuf = X_train[perm_idx, :]

        pipe.fit(X_train_shuf, y_train)
        risk_val = pipe.predict(X_val)

        har = float(concordance_index_censored(y_val["event"], y_val["time"], risk_val)[0])
        uno = safe_uno_cindex(y_train_original, y_val, risk_val, tau=tau)

        # Track feature dimensions at different stages
        try:
            X_val_t = pipe[:-1].transform(X_val)
            n_feat_after_pipeline = int(X_val_t.shape[1])
        except Exception:
            n_feat_after_pipeline = int(X_val.shape[1])
        
        # Count non-zero coefficients in the selected model
        model = pipe.named_steps.get("model", None)
        n_nonzero_coefs = count_model_nonzero_coefs(model) if model else 0

        records.append({
            "baseline": "shuffle_x",
            "rep": r,
            "harrell_c": har,
            "uno_c": uno,
            "tau": tau,
            "n_features_input": n_feat_input,
            "n_features_after_pipeline": n_feat_after_pipeline,
            "n_nonzero_coefs": n_nonzero_coefs,
        })
    return records


def main():
    ap = argparse.ArgumentParser("R0 Sanity Check (Negative Control)")

    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)

    ap.add_argument("--id_col", type=str, required=True)
    ap.add_argument("--time_col", type=str, required=True)
    ap.add_argument("--event_col", type=str, required=True)

    ap.add_argument("--drop_cols", type=str, default="", help="Comma-separated columns to drop from features to avoid leakage.")
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau_q", type=float, default=0.9)

    # pipeline args
    ap.add_argument("--impute_strategy", type=str, default="median", choices=["mean", "median", "most_frequent"])
    ap.add_argument("--variance_thresh", type=float, default=1e-6)

    ap.add_argument("--use_corr_filter", action="store_true", help="Enable survival-aware correlation filter (NOT recommended for R0).")
    ap.add_argument("--corr_thresh", type=float, default=0.85)

    # model args
    ap.add_argument("--model", type=str, default="coxnet", choices=["coxph", "coxnet"])
    ap.add_argument("--coxph_alpha", type=float, default=1e-4)

    ap.add_argument("--l1_ratio", type=float, default=0.5)
    ap.add_argument("--alpha_min_ratio", type=str, default="auto")
    ap.add_argument("--n_alphas", type=int, default=100)
    ap.add_argument("--min_nnz", type=int, default=5)
    ap.add_argument("--max_nnz", type=int, default=50)
    ap.add_argument("--max_iter", type=int, default=100000)
    ap.add_argument("--tol", type=float, default=1e-7)

    # R0 settings
    ap.add_argument("--n_perm", type=int, default=20, help="Number of permutations for permute-y baseline.")
    ap.add_argument("--n_perm_x", type=int, default=0, help="Number of permutations for shuffle-x baseline (optional).")
    ap.add_argument("--n_random", type=int, default=100, help="Trials for random-risk baseline per fold.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level for verdict.")
    ap.add_argument("--practical", type=float, default=0.03, help="Practical deviation threshold for FLAG.")
    ap.add_argument("--warning", type=float, default=0.08, help="Large deviation threshold for WARNING.")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    logger = setup_logger(outdir, verbose=True)

    # global seed hard lock (reproducibility)
    random.seed(args.seed)
    np.random.seed(args.seed)

    rng = np.random.default_rng(args.seed)

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    logger.info("Loading data...")
    X_df, y_df = load_and_align(
        features_csv=Path(args.features_csv),
        labels_csv=Path(args.labels_csv),
        id_col=args.id_col,
        time_col=args.time_col,
        event_col=args.event_col,
        drop_cols=drop_cols,
    )
    y_struct_all = make_y_struct(y_df, time_col=args.time_col, event_col=args.event_col)

    splits = load_splits(Path(args.splits_json))

    logger.info(f"Aligned patients: {len(X_df)} | Features: {X_df.shape[1]}")
    logger.info(f"Model={args.model} | n_perm={args.n_perm} | n_random={args.n_random} | n_perm_x={args.n_perm_x}")

    all_records = []
    random_records = []
    fold_summaries = []

    for fold_name, fold in splits.items():
        train_ids = [str(x) for x in fold["train"]]
        val_ids = [str(x) for x in fold["val"]]

        # intersect with available IDs
        train_ids = [pid for pid in train_ids if pid in X_df.index]
        val_ids = [pid for pid in val_ids if pid in X_df.index]

        if len(train_ids) < 10 or len(val_ids) < 5:
            logger.warning(f"{fold_name}: too few samples after intersection. train={len(train_ids)}, val={len(val_ids)}. Skipping.")
            continue

        X_train = X_df.loc[train_ids].to_numpy(dtype=float)
        X_val = X_df.loc[val_ids].to_numpy(dtype=float)

        y_train = y_struct_all[X_df.index.get_indexer(train_ids)]
        y_val = y_struct_all[X_df.index.get_indexer(val_ids)]

        logger.info(f"{fold_name}: train={len(train_ids)}, val={len(val_ids)}, n_features_input={X_train.shape[1]}")

        pipe = build_pipeline(args, logger)

        # ---- Random-risk baseline (eval-code sanity) ----
        rr_mean, rr_std = eval_random_risk(y_val, n_trials=args.n_random, rng=rng)
        random_records.append({
            "fold": fold_name,
            "random_risk_mean": rr_mean,
            "random_risk_std": rr_std,
        })

        # ---- Permute-y baseline (core R0) ----
        recs = run_perm_y(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
            y_train_original=y_train,
            pipe=pipe,
            n_perm=args.n_perm,
            tau_q=args.tau_q,
            rng=rng,
        )
        for r in recs:
            r.update({"fold": fold_name})
        all_records.extend(recs)

        # ---- Shuffle-x baseline (optional) ----
        if args.n_perm_x and args.n_perm_x > 0:
            pipe2 = build_pipeline(args, logger)
            recs_x = run_shuffle_x(
                X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                y_train_original=y_train,
                pipe=pipe2,
                n_perm_x=args.n_perm_x,
                tau_q=args.tau_q,
                rng=rng,
            )
            for r in recs_x:
                r.update({"fold": fold_name})
            all_records.extend(recs_x)

        # fold-level quick summary
        fold_perm = [x for x in recs if x["baseline"] == "perm_y"]
        fold_h = np.array([x["harrell_c"] for x in fold_perm], dtype=float)
        fold_u = np.array([x["uno_c"] for x in fold_perm], dtype=float)
        fold_summaries.append({
            "fold": fold_name,
            "perm_y_harrell_mean": float(np.nanmean(fold_h)),
            "perm_y_harrell_std": float(np.nanstd(fold_h, ddof=1)) if np.isfinite(fold_h).sum() > 1 else 0.0,
            "perm_y_uno_mean": float(np.nanmean(fold_u)),
            "perm_y_uno_std": float(np.nanstd(fold_u, ddof=1)) if np.isfinite(fold_u).sum() > 1 else 0.0,
        })

    if len(all_records) == 0:
        raise RuntimeError("No valid folds were processed. Check splits.json and ID alignment.")

    df = pd.DataFrame(all_records)
    df.to_csv(outdir / "r0_records.csv", index=False)

    df_random = pd.DataFrame(random_records)
    df_random.to_csv(outdir / "r0_random_risk_baseline.csv", index=False)

    df_fold = pd.DataFrame(fold_summaries)
    df_fold.to_csv(outdir / "r0_per_fold_summary.csv", index=False)

    # overall verdict for perm_y only
    perm_df = df[df["baseline"] == "perm_y"].copy()
    verdict_h = compute_verdict(
        perm_df["harrell_c"].to_numpy(),
        alpha=args.alpha, practical=args.practical, warning=args.warning
    )
    verdict_u = compute_verdict(
        perm_df["uno_c"].to_numpy(),
        alpha=args.alpha, practical=args.practical, warning=args.warning
    )

    summary = {
        "task": "R0 sanity check (negative control)",
        "model": args.model,
        "n_total_evaluations": int(len(perm_df)),
        "harrell": {
            "mean": round(verdict_h.mean, 6),
            "std": round(verdict_h.std, 6),
            "se": round(verdict_h.se, 6),
            "z": round(verdict_h.z, 6),
            "p_value": round(verdict_h.p_value, 6),
            "flag": verdict_h.flag,
            "reason": verdict_h.reason,
        },
        "uno_ipcw": {
            "mean": round(verdict_u.mean, 6),
            "std": round(verdict_u.std, 6),
            "se": round(verdict_u.se, 6),
            "z": round(verdict_u.z, 6),
            "p_value": round(verdict_u.p_value, 6),
            "flag": verdict_u.flag,
            "reason": verdict_u.reason,
        },
        "thresholds": {
            "alpha": args.alpha,
            "practical": args.practical,
            "warning": args.warning,
        },
        "notes": [
            "perm_y: shuffle y_train only, train on permuted labels, evaluate on real y_val.",
            "uno_ipcw uses tau computed from ORIGINAL y_train (not permuted).",
            "random-risk baseline sanity-checks the metric computation itself.",
        ],
        "config": {
            "seed": args.seed,
            "tau_q": args.tau_q,
            "impute_strategy": args.impute_strategy,
            "variance_thresh": args.variance_thresh,
            "use_corr_filter": bool(args.use_corr_filter),
            "corr_thresh": args.corr_thresh,
            "coxph_alpha": args.coxph_alpha,
            "coxnet": {
                "l1_ratio": args.l1_ratio,
                "alpha_min_ratio": args.alpha_min_ratio,
                "n_alphas": args.n_alphas,
                "min_nnz": args.min_nnz,
                "max_nnz": args.max_nnz,
                "max_iter": args.max_iter,
                "tol": args.tol,
            },
            "n_perm": args.n_perm,
            "n_perm_x": args.n_perm_x,
            "n_random": args.n_random,
        },
    }

    with open(outdir / "r0_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("==== R0 DONE ====")
    logger.info(f"Harrell perm_y: mean={summary['harrell']['mean']:.4f}, std={summary['harrell']['std']:.4f}, flag={summary['harrell']['flag']}")
    logger.info(f"Uno   perm_y: mean={summary['uno_ipcw']['mean']:.4f}, std={summary['uno_ipcw']['std']:.4f}, flag={summary['uno_ipcw']['flag']}")
    logger.info(f"Saved: {outdir/'r0_records.csv'}")
    logger.info(f"Saved: {outdir/'r0_summary.json'}")


if __name__ == "__main__":
    main()


#run command:
#bash run_all_r0_tests.sh