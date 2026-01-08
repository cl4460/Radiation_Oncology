#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R6: Late Fusion on risk predictions (OOF) with nested CV weight selection.

You pass multiple base models' OOF risk predictions (e.g., Cp / Df / Tb),
then this script:
1) For each OUTER fold (given by splits_json):
   - uses OUTER-train only to select a fusion rule via INNER CV;
   - evaluates on OUTER-val.
2) Dumps per-fold predictions + metrics, and overall OOF metrics.

Supported fusion modes:
- blend : convex combination (weights sum to 1, non-negative), with grid search.
- winner: pick the single best model per fold (no blending).

Key safeguards (fixes the common fusion pitfalls we discussed previously):
- Normalization is always fitted on train split only (inner-train inside inner CV,
  outer-train for final per-fold evaluation), then applied to val.
- Rank / quantile normalization maps VAL through the TRAIN empirical CDF
  (no "rank val by itself" leakage / range mismatch).
- Optional 1-SE rule with "prefer less mixing" tie-breaker.
- Optional R0 mode: permute labels in evaluation as sanity check.

Outputs (outdir):
- params.json
- summary.json
- per_fold_metrics.csv
- oof_predictions.csv
- fold_k/predictions.csv
- fold_k/metrics.json
- fold_k/inner_cv_grid.csv (blend only)

Note:
- This is "late fusion on risk scores", not re-training base models.
- You should feed RAW per-fold risk outputs (e.g., risk_pred_raw) rather than
  globally-normalized risk_pred_oof to avoid transductive scaling.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

from sksurv.metrics import concordance_index_censored, concordance_index_ipcw


# ---------------------------------------------------------------------
# IO utils
# ---------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def dump_args(path: Path, args: argparse.Namespace) -> None:
    obj = {}
    for k, v in vars(args).items():
        if isinstance(v, np.ndarray):
            obj[k] = v.tolist()
        else:
            obj[k] = v
    write_json(path, obj)


def mean_std(arr: Sequence[float]) -> Tuple[float, float]:
    x = np.asarray([v for v in arr if v is not None and np.isfinite(v)], dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x.mean()), 0.0
    return float(x.mean()), float(x.std(ddof=1))


# ---------------------------------------------------------------------
# Survival utils
# ---------------------------------------------------------------------

def make_surv_struct(event: np.ndarray, time_arr: np.ndarray) -> np.ndarray:
    e = event.astype(bool)
    t = time_arr.astype(float)
    y = np.empty(len(t), dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = e
    y["time"] = t
    return y


def compute_tau_from_train(y_train: np.ndarray, quantile: float = 0.9) -> float:
    """Pick tau for Uno/IPCW C-index; keep it strictly < max(time)."""
    times = y_train["time"].astype(float)
    events = y_train["event"].astype(bool)
    ev_times = times[events]
    if ev_times.shape[0] >= 5:
        tau = float(np.quantile(ev_times, quantile))
    else:
        tau = float(np.quantile(times, quantile))
    max_time = float(np.max(times))
    if not np.isfinite(tau):
        tau = max_time - 1e-6
    if tau >= max_time:
        tau = max_time - 1e-6
    return float(tau)


def safe_uno_cindex(y_train: np.ndarray, y_val: np.ndarray, risk_val: np.ndarray, tau: float) -> float:
    try:
        return float(concordance_index_ipcw(y_train, y_val, risk_val, tau=tau)[0])
    except Exception:
        return float("nan")


def harrell_cindex(y: np.ndarray, risk: np.ndarray) -> float:
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


def score_metric(metric: str, y_tr: np.ndarray, y_va: np.ndarray, risk_va: np.ndarray, tau_q: float) -> float:
    if metric == "uno":
        tau = compute_tau_from_train(y_tr, quantile=tau_q)
        return safe_uno_cindex(y_tr, y_va, risk_va, tau=tau)
    return harrell_cindex(y_va, risk_va)


# ---------------------------------------------------------------------
# Risk transform (sign align + normalization)
# ---------------------------------------------------------------------

def _safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return 1.0
    s = float(np.std(x, ddof=1))
    return s if (np.isfinite(s) and s > 0) else 1.0


def _percentile_from_sorted_train(x: np.ndarray, sorted_train: np.ndarray) -> np.ndarray:
    """
    Map values to empirical CDF percentiles using TRAIN distribution only.
    Uses right-search and (rank-0.5)/n to avoid exact 0/1.
    """
    x = np.asarray(x, dtype=float)
    st = np.asarray(sorted_train, dtype=float)
    n = st.size
    if n == 0:
        return np.full_like(x, 0.5, dtype=float)
    ranks = np.searchsorted(st, x, side="right").astype(float)  # 0..n
    p = (ranks - 0.5) / float(n)
    eps = 1e-6
    return np.clip(p, eps, 1.0 - eps)


@dataclass
class FittedRiskTransform:
    sign: float
    norm: str
    mean_: float = 0.0
    std_: float = 1.0
    sorted_train_: Optional[np.ndarray] = None
    rank_post_zscore: bool = True
    qt_: Optional[QuantileTransformer] = None

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float) * float(self.sign)

        if self.norm == "none":
            return x

        if self.norm == "zscore":
            return (x - float(self.mean_)) / float(self.std_)

        if self.norm in ("rank_traincdf", "rank_traincdf_only"):
            p = _percentile_from_sorted_train(x, self.sorted_train_) if self.sorted_train_ is not None else np.full_like(x, 0.5, dtype=float)
            if self.norm == "rank_traincdf_only" or not self.rank_post_zscore:
                return p
            return (p - float(self.mean_)) / float(self.std_)

        if self.norm in ("quantile_normal", "quantile_uniform"):
            if self.qt_ is None:
                return x
            return self.qt_.transform(x.reshape(-1, 1)).reshape(-1)

        raise ValueError(f"Unknown norm: {self.norm}")


def fit_risk_transform(x_train: np.ndarray,
                       y_train: Optional[np.ndarray],
                       norm: str,
                       auto_sign: bool,
                       seed: int) -> FittedRiskTransform:
    x_train = np.asarray(x_train, dtype=float)

    # 1) sign align (so "higher risk" => "worse survival" consistently)
    sign = 1.0
    if auto_sign and (y_train is not None):
        c = harrell_cindex(y_train, x_train)
        if np.isfinite(c) and c < 0.5:
            sign = -1.0

    x_adj = x_train * sign

    # 2) fit normalization on TRAIN ONLY
    if norm == "none":
        return FittedRiskTransform(sign=sign, norm=norm)

    if norm == "zscore":
        mu = float(np.mean(x_adj)) if x_adj.size > 0 else 0.0
        sd = _safe_std(x_adj)
        return FittedRiskTransform(sign=sign, norm=norm, mean_=mu, std_=sd)

    if norm in ("rank_traincdf", "rank_traincdf_only"):
        st = np.sort(x_adj.astype(float))
        if norm == "rank_traincdf_only":
            return FittedRiskTransform(sign=sign, norm=norm, sorted_train_=st, rank_post_zscore=False)
        p_tr = _percentile_from_sorted_train(x_adj, st)
        mu = float(np.mean(p_tr)) if p_tr.size > 0 else 0.5
        sd = _safe_std(p_tr)
        return FittedRiskTransform(sign=sign, norm=norm, mean_=mu, std_=sd, sorted_train_=st, rank_post_zscore=True)

    if norm in ("quantile_normal", "quantile_uniform"):
        out_dist = "normal" if norm == "quantile_normal" else "uniform"
        n_q = int(min(1000, max(10, x_adj.size)))
        qt = QuantileTransformer(
            n_quantiles=n_q,
            output_distribution=out_dist,
            subsample=int(1e9),
            random_state=int(seed),
        )
        qt.fit(x_adj.reshape(-1, 1))
        return FittedRiskTransform(sign=sign, norm=norm, qt_=qt)

    raise ValueError(f"Unknown norm: {norm}")


# ---------------------------------------------------------------------
# Weight grid
# ---------------------------------------------------------------------

def gen_alpha_grid(alpha_step: float) -> np.ndarray:
    step = float(alpha_step)
    if not (step > 0 and step < 1):
        raise ValueError("--alpha_step must be in (0,1)")
    grid = np.arange(0.0, 1.0 + 1e-12, step, dtype=float)
    grid = np.clip(grid, 0.0, 1.0)
    grid = np.unique(np.round(grid, 12))
    if grid[-1] != 1.0:
        grid = np.append(grid, 1.0)
    if grid[0] != 0.0:
        grid = np.insert(grid, 0, 0.0)
    return grid


def gen_simplex_grid(n_models: int, step: float, max_candidates: int = 50000) -> np.ndarray:
    """
    Generate weights on an (n_models-1) simplex with given step size.
    For n=2, it returns [[a, 1-a], ...].
    For n>=3, step must divide 1.0 exactly within tolerance.
    """
    n_models = int(n_models)
    if n_models < 2:
        raise ValueError("Need at least 2 models for fusion.")
    step = float(step)
    if not (step > 0 and step < 1):
        raise ValueError("--alpha_step must be in (0,1)")
    if n_models == 2:
        a = gen_alpha_grid(step)
        return np.stack([a, 1.0 - a], axis=1)

    inv = 1.0 / step
    N = int(round(inv))
    if not math.isclose(N * step, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"For n_models>2, alpha_step must divide 1.0. Got step={step}")

    weights_int: List[List[int]] = []

    def rec(k: int, remaining: int, cur: List[int]):
        nonlocal weights_int
        if len(weights_int) > max_candidates:
            return
        if k == n_models - 1:
            weights_int.append(cur + [remaining])
            return
        for i in range(remaining + 1):
            rec(k + 1, remaining - i, cur + [i])

    rec(0, N, [])

    if len(weights_int) > max_candidates:
        raise RuntimeError(f"Too many weight candidates ({len(weights_int)}). Increase --alpha_step.")

    W = np.asarray(weights_int, dtype=float) * step
    W = np.clip(W, 0.0, 1.0)
    W = W / np.sum(W, axis=1, keepdims=True)
    return W


def mixing_penalty(weights: np.ndarray) -> float:
    """Lower is 'less mixing' (closer to a vertex)."""
    w = np.asarray(weights, dtype=float)
    return float(1.0 - np.max(w))


# ---------------------------------------------------------------------
# Inner CV selection
# ---------------------------------------------------------------------

@dataclass
class InnerCVResult:
    selected_weights: np.ndarray
    selected_weights_best: np.ndarray
    selected_weights_1se: np.ndarray
    mean_scores: np.ndarray
    se_scores: np.ndarray
    candidate_weights: np.ndarray


def inner_select_weights_blend(df_tr: pd.DataFrame,
                               model_names: List[str],
                               weights_grid: np.ndarray,
                               inner_folds: int,
                               seed: int,
                               metric: str,
                               tau_q: float,
                               norm: str,
                               auto_sign: bool,
                               alpha_select: str,
                               prefer_less_mixing: bool,
                               mode: str) -> InnerCVResult:
    """Select blend weights by INNER CV on OUTER-train only."""
    X = df_tr[[f"risk_{m}" for m in model_names]].to_numpy(dtype=float)
    y = make_surv_struct(df_tr["event"].to_numpy().astype(int),
                         df_tr["time"].to_numpy().astype(float))
    y_event = y["event"].astype(int)

    skf = StratifiedKFold(n_splits=int(inner_folds), shuffle=True, random_state=int(seed))

    n_cand = weights_grid.shape[0]
    fold_scores = np.full((int(inner_folds), n_cand), np.nan, dtype=float)

    for k, (itr, iva) in enumerate(skf.split(X, y_event)):
        X_itr, X_iva = X[itr], X[iva]
        y_itr, y_iva = y[itr], y[iva]

        # R0: permute labels used for evaluation in this inner split
        if mode.upper() == "R0":
            rng = np.random.default_rng(int(seed) + 10000 + k)
            perm = rng.permutation(len(y_iva))
            y_iva_eval = y_iva[perm]
        else:
            y_iva_eval = y_iva

        # Fit transforms on INNER-train only
        Z_iva_list = []
        for j in range(len(model_names)):
            trf = fit_risk_transform(X_itr[:, j], y_itr, norm=norm, auto_sign=auto_sign, seed=int(seed) + 17*j + k)
            Z_iva_list.append(trf.transform(X_iva[:, j]))

        Z_iva = np.stack(Z_iva_list, axis=1)

        # Evaluate all candidates on INNER-val
        for c in range(n_cand):
            w = weights_grid[c]
            fused = Z_iva @ w
            sc = score_metric(metric, y_itr, y_iva_eval, fused, tau_q=tau_q)
            if np.isfinite(sc):
                fold_scores[k, c] = sc

    mean_sc = np.nanmean(fold_scores, axis=0)
    std_sc = np.nanstd(fold_scores, axis=0, ddof=1)
    se_sc = std_sc / math.sqrt(float(inner_folds))

    if not np.isfinite(mean_sc).any():
        w0 = np.ones(weights_grid.shape[1], dtype=float) / float(weights_grid.shape[1])
        return InnerCVResult(selected_weights=w0, selected_weights_best=w0, selected_weights_1se=w0,
                             mean_scores=mean_sc, se_scores=se_sc, candidate_weights=weights_grid)

    best_idx = int(np.nanargmax(mean_sc))
    w_best = weights_grid[best_idx].copy()

    # 1-SE rule: pick a candidate within (best - 1*SE)
    threshold = float(mean_sc[best_idx] - se_sc[best_idx]) if np.isfinite(se_sc[best_idx]) else float(mean_sc[best_idx])
    ok = np.where(mean_sc >= threshold)[0]
    if ok.size == 0:
        ok = np.asarray([best_idx], dtype=int)

    if prefer_less_mixing:
        best_ok = None
        best_pen = None
        best_score = None
        for idx in ok.tolist():
            w = weights_grid[int(idx)]
            pen = mixing_penalty(w)
            sc = mean_sc[int(idx)]
            if (best_ok is None) or (pen < best_pen - 1e-12) or (math.isclose(pen, best_pen, abs_tol=1e-12) and sc > best_score):
                best_ok = int(idx)
                best_pen = float(pen)
                best_score = float(sc)
        w_1se = weights_grid[int(best_ok)].copy()
    else:
        j = int(ok[np.nanargmax(mean_sc[ok])])
        w_1se = weights_grid[j].copy()

    w_sel = w_best if alpha_select == "best" else w_1se
    return InnerCVResult(selected_weights=w_sel,
                         selected_weights_best=w_best,
                         selected_weights_1se=w_1se,
                         mean_scores=mean_sc,
                         se_scores=se_sc,
                         candidate_weights=weights_grid)


def inner_select_winner(df_tr: pd.DataFrame,
                        model_names: List[str],
                        inner_folds: int,
                        seed: int,
                        metric: str,
                        tau_q: float,
                        norm: str,
                        auto_sign: bool,
                        mode: str) -> Tuple[str, np.ndarray]:
    """Winner-take-all: pick one model per outer fold based on inner CV mean score."""
    X = df_tr[[f"risk_{m}" for m in model_names]].to_numpy(dtype=float)
    y = make_surv_struct(df_tr["event"].to_numpy().astype(int),
                         df_tr["time"].to_numpy().astype(float))
    y_event = y["event"].astype(int)

    skf = StratifiedKFold(n_splits=int(inner_folds), shuffle=True, random_state=int(seed))
    scores = np.full((int(inner_folds), len(model_names)), np.nan, dtype=float)

    for k, (itr, iva) in enumerate(skf.split(X, y_event)):
        X_itr, X_iva = X[itr], X[iva]
        y_itr, y_iva = y[itr], y[iva]

        if mode.upper() == "R0":
            rng = np.random.default_rng(int(seed) + 20000 + k)
            perm = rng.permutation(len(y_iva))
            y_iva_eval = y_iva[perm]
        else:
            y_iva_eval = y_iva

        for j, _name in enumerate(model_names):
            trf = fit_risk_transform(X_itr[:, j], y_itr, norm=norm, auto_sign=auto_sign, seed=int(seed) + 31*j + k)
            z_iva = trf.transform(X_iva[:, j])
            scores[k, j] = score_metric(metric, y_itr, y_iva_eval, z_iva, tau_q=tau_q)

    mean_sc = np.nanmean(scores, axis=0)
    if not np.isfinite(mean_sc).any():
        return model_names[0], mean_sc

    j_best = int(np.nanargmax(mean_sc))
    return model_names[j_best], mean_sc


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def load_labels(labels_csv: Optional[Path],
                id_col: str,
                time_col: str,
                event_col: str) -> pd.DataFrame:
    """Load labels from labels_csv. If labels_csv is None, return empty df."""
    if labels_csv is None:
        return pd.DataFrame(columns=[id_col, time_col, event_col])

    df = pd.read_csv(labels_csv).copy()
    if id_col not in df.columns or time_col not in df.columns or event_col not in df.columns:
        raise ValueError(f"labels_csv must contain [{id_col}, {time_col}, {event_col}]")

    df[id_col] = df[id_col].astype(str)
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df.dropna(subset=[time_col, event_col]).copy()
    df = df[df[time_col] > 0].copy()
    df[event_col] = (df[event_col] > 0).astype(int)
    return df[[id_col, time_col, event_col]].copy()


def load_oof_csv(spec: str, id_col: str, risk_col: str) -> Tuple[str, pd.DataFrame, str]:
    """
    spec format: "Name:/path/to/oof.csv"
    Returns: (name, df(id,risk_...), path)
    """
    if ":" not in spec:
        raise ValueError(f"--oof_csv must be 'Name:/path'. Got: {spec}")
    name, path = spec.split(":", 1)
    name = name.strip()
    path = path.strip()
    if len(name) == 0:
        raise ValueError(f"Empty model name in spec: {spec}")

    df = pd.read_csv(path).copy()
    if id_col not in df.columns:
        raise ValueError(f"{path} missing id_col={id_col}")
    if risk_col not in df.columns:
        raise ValueError(f"{path} missing risk_col={risk_col}")

    df[id_col] = df[id_col].astype(str)
    df = df[[id_col, risk_col]].copy()

    if df[id_col].duplicated().any():
        df = df.drop_duplicates(subset=[id_col], keep="first").copy()

    df = df.rename(columns={risk_col: f"risk_{name}"})
    return name, df, path


def run_r6(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    print(f"[R6] exp_id={args.exp_id} | mode={args.mode} | fusion_mode={args.fusion_mode}")
    print(f"[R6] norm={args.norm} | oof_norm={args.oof_norm} | alpha_select={args.alpha_select} | prefer_less_mixing={args.prefer_less_mixing}")

    splits = read_json(Path(args.splits_json))
    fold_keys = sorted(list(splits.keys()))

    labels_df = load_labels(Path(args.labels_csv) if args.labels_csv else None,
                            id_col=args.id_col,
                            time_col=args.time_col,
                            event_col=args.event_col)

    model_names: List[str] = []
    merged_risk: Optional[pd.DataFrame] = None
    first_oof_path: Optional[str] = None

    for spec in args.oof_csv:
        name, df_r, path = load_oof_csv(spec, id_col=args.id_col, risk_col=args.risk_col)
        if first_oof_path is None:
            first_oof_path = path
        model_names.append(name)
        merged_risk = df_r if merged_risk is None else merged_risk.merge(df_r, on=args.id_col, how="inner")

    if merged_risk is None or len(model_names) < 2:
        raise ValueError("Need at least two --oof_csv inputs.")

    # If labels_csv not provided, try to fetch labels from the first oof file
    if labels_df.empty:
        assert first_oof_path is not None
        tmp = pd.read_csv(first_oof_path)
        tmp[args.id_col] = tmp[args.id_col].astype(str)

        # smart column detection
        time_col = args.time_col
        event_col = args.event_col
        if time_col not in tmp.columns and "time" in tmp.columns:
            time_col = "time"
        if event_col not in tmp.columns and "event" in tmp.columns:
            event_col = "event"

        if time_col not in tmp.columns or event_col not in tmp.columns:
            raise ValueError("labels_csv omitted, but cannot find time/event in the first oof file. "
                             f"Tried ({args.time_col},{args.event_col}) and fallback (time,event).")

        tmp[time_col] = pd.to_numeric(tmp[time_col], errors="coerce")
        tmp[event_col] = pd.to_numeric(tmp[event_col], errors="coerce")
        tmp = tmp.dropna(subset=[time_col, event_col]).copy()
        tmp = tmp[tmp[time_col] > 0].copy()
        tmp[event_col] = (tmp[event_col] > 0).astype(int)

        labels_df = tmp[[args.id_col, time_col, event_col]].copy()
        labels_df = labels_df.rename(columns={time_col: args.time_col, event_col: args.event_col})

        print(f"[R6] labels_csv omitted; using labels from {first_oof_path} with cols ({time_col},{event_col})")

    df_all = merged_risk.merge(labels_df, on=args.id_col, how="inner")
    df_all = df_all.rename(columns={args.time_col: "time", args.event_col: "event"})

    # keep only ids in splits
    all_ids_in_splits = set()
    for fk in fold_keys:
        all_ids_in_splits.update([str(x) for x in splits[fk]["train"]])
        all_ids_in_splits.update([str(x) for x in splits[fk]["val"]])

    n_before = len(df_all)
    df_all = df_all[df_all[args.id_col].isin(all_ids_in_splits)].copy()
    n_after = len(df_all)
    if n_after != n_before:
        print(f"[R6][WARN] Dropped {n_before - n_after} rows not found in splits_json.")

    print(f"[R6] n={len(df_all)} | models={model_names}")

    # weight candidates
    weights_grid = None
    if args.fusion_mode == "blend":
        weights_grid = gen_simplex_grid(len(model_names), step=float(args.alpha_step))
        print(f"[R6] weight_candidates={weights_grid.shape[0]} (step={args.alpha_step})")

    all_fold_metrics: List[Dict] = []
    oof_rows: List[Dict] = []

    for fold_idx, fold_name in enumerate(fold_keys):
        t0 = time.time()
        train_ids = [str(x) for x in splits[fold_name]["train"]]
        val_ids = [str(x) for x in splits[fold_name]["val"]]

        df_tr = df_all[df_all[args.id_col].isin(train_ids)].copy()
        df_va = df_all[df_all[args.id_col].isin(val_ids)].copy()
        if df_tr.empty or df_va.empty:
            raise ValueError(f"[R6][{fold_name}] empty split train={len(df_tr)} val={len(df_va)}")

        seed_fold = int(args.seed) + 1000 + int(fold_idx)

        # inner selection
        inner_detail = {}
        if args.fusion_mode == "winner":
            winner_name, mean_sc = inner_select_winner(
                df_tr, model_names=model_names,
                inner_folds=int(args.inner_folds), seed=seed_fold,
                metric=args.inner_metric, tau_q=float(args.tau_q),
                norm=args.norm, auto_sign=bool(args.auto_sign),
                mode=str(args.mode),
            )
            selected_weights = np.zeros(len(model_names), dtype=float)
            selected_weights[model_names.index(winner_name)] = 1.0
            selected_weights_best = selected_weights.copy()
            selected_weights_1se = selected_weights.copy()
            inner_detail = {
                "winner": winner_name,
                "mean_scores": {m: (None if not np.isfinite(s) else float(s)) for m, s in zip(model_names, mean_sc)},
            }
            inner_res = None
        else:
            assert weights_grid is not None
            inner_res = inner_select_weights_blend(
                df_tr, model_names=model_names,
                weights_grid=weights_grid,
                inner_folds=int(args.inner_folds),
                seed=seed_fold,
                metric=args.inner_metric,
                tau_q=float(args.tau_q),
                norm=args.norm,
                auto_sign=bool(args.auto_sign),
                alpha_select=str(args.alpha_select),
                prefer_less_mixing=bool(args.prefer_less_mixing),
                mode=str(args.mode),
            )
            selected_weights = inner_res.selected_weights
            selected_weights_best = inner_res.selected_weights_best
            selected_weights_1se = inner_res.selected_weights_1se
            inner_detail = {
                "selected_weights_best": {m: float(w) for m, w in zip(model_names, selected_weights_best)},
                "selected_weights_1se": {m: float(w) for m, w in zip(model_names, selected_weights_1se)},
            }

        # fit transforms on outer-train, transform train/val
        X_tr = df_tr[[f"risk_{m}" for m in model_names]].to_numpy(dtype=float)
        X_va = df_va[[f"risk_{m}" for m in model_names]].to_numpy(dtype=float)
        y_tr = make_surv_struct(df_tr["event"].to_numpy().astype(int), df_tr["time"].to_numpy().astype(float))
        y_va = make_surv_struct(df_va["event"].to_numpy().astype(int), df_va["time"].to_numpy().astype(float))

        Z_tr_list = []
        Z_va_list = []
        sign_map = {}
        for j, m in enumerate(model_names):
            trf = fit_risk_transform(X_tr[:, j], y_tr, norm=args.norm, auto_sign=bool(args.auto_sign), seed=seed_fold + 17*j)
            sign_map[m] = float(trf.sign)
            Z_tr_list.append(trf.transform(X_tr[:, j]))
            Z_va_list.append(trf.transform(X_va[:, j]))

        Z_tr = np.stack(Z_tr_list, axis=1)
        Z_va = np.stack(Z_va_list, axis=1)

        fused_tr = (Z_tr @ selected_weights).astype(float)
        fused_va = (Z_va @ selected_weights).astype(float)

        # R0: permute outer-val labels for evaluation
        if args.mode.upper() == "R0":
            rng = np.random.default_rng(seed_fold + 999)
            perm = rng.permutation(len(y_va))
            y_va_eval = y_va[perm]
        else:
            y_va_eval = y_va

        har_fused = harrell_cindex(y_va_eval, fused_va)
        tau = compute_tau_from_train(y_tr, quantile=float(args.tau_q))
        uno_fused = safe_uno_cindex(y_tr, y_va_eval, fused_va, tau=tau)

        # baseline per-model (on processed scale)
        base_scores = {}
        for j, m in enumerate(model_names):
            har_m = harrell_cindex(y_va_eval, Z_va[:, j])
            uno_m = safe_uno_cindex(y_tr, y_va_eval, Z_va[:, j], tau=tau)
            base_scores[m] = {
                "harrell": float(har_m),
                "uno": None if np.isnan(uno_m) else float(uno_m),
            }

        # OOF normalization (fold-specific) for cross-fold comparability
        if args.oof_norm == "none":
            fused_va_oof = fused_va.copy()
        else:
            trf_oof = fit_risk_transform(fused_tr, y_train=None, norm=args.oof_norm, auto_sign=False, seed=seed_fold + 12345)
            fused_va_oof = trf_oof.transform(fused_va)

        fold_dir = outdir / f"fold_{fold_idx}"
        ensure_dir(fold_dir)

        pred_df = pd.DataFrame({
            args.id_col: df_va[args.id_col].astype(str).to_numpy(),
            "time": y_va["time"].astype(float),
            "event": y_va["event"].astype(bool),
            "risk_fused_raw": fused_va.astype(float),
            "risk_fused_oof": fused_va_oof.astype(float),
        })
        for j, m in enumerate(model_names):
            pred_df[f"risk_{m}_raw"] = X_va[:, j].astype(float)
            pred_df[f"risk_{m}_proc"] = Z_va[:, j].astype(float)
        pred_df.to_csv(fold_dir / "predictions.csv", index=False)

        if args.fusion_mode == "blend" and inner_res is not None:
            inner_grid_df = pd.DataFrame(inner_res.candidate_weights, columns=[f"w_{m}" for m in model_names])
            inner_grid_df["mean_score"] = inner_res.mean_scores
            inner_grid_df["se_score"] = inner_res.se_scores
            inner_grid_df.to_csv(fold_dir / "inner_cv_grid.csv", index=False)

        metrics = {
            "exp_id": str(args.exp_id),
            "mode": str(args.mode),
            "fusion_mode": str(args.fusion_mode),
            "fold": int(fold_idx),
            "fold_name": str(fold_name),
            "n_train": int(len(df_tr)),
            "n_val": int(len(df_va)),
            "models": list(model_names),
            "norm": str(args.norm),
            "auto_sign": bool(args.auto_sign),
            "signs": sign_map,
            "inner_metric": str(args.inner_metric),
            "inner_folds": int(args.inner_folds),
            "alpha_step": float(args.alpha_step),
            "alpha_select": str(args.alpha_select),
            "prefer_less_mixing": bool(args.prefer_less_mixing),
            "selected_weights": {m: float(w) for m, w in zip(model_names, selected_weights)},
            "harrell_fused": round(float(har_fused), 6),
            "uno_fused": None if np.isnan(uno_fused) else round(float(uno_fused), 6),
            "tau": round(float(tau), 6),
            "tau_q": float(args.tau_q),
            "base_scores": base_scores,
            "oof_norm": str(args.oof_norm),
            "runtime_seconds": round(float(time.time() - t0), 3),
        }
        metrics.update(inner_detail)

        write_json(fold_dir / "metrics.json", metrics)
        all_fold_metrics.append(metrics)

        for r in pred_df.itertuples(index=False):
            row = {
                args.id_col: getattr(r, args.id_col),
                "fold": int(fold_idx),
                "time": float(r.time),
                "event": bool(r.event),
                "risk_fused_raw": float(r.risk_fused_raw),
                "risk_fused_oof": float(r.risk_fused_oof),
            }
            for m in model_names:
                row[f"risk_{m}_raw"] = float(getattr(r, f"risk_{m}_raw"))
                row[f"risk_{m}_proc"] = float(getattr(r, f"risk_{m}_proc"))
            oof_rows.append(row)

        print(f"[R6][{fold_name}] HarrellC(fused)={har_fused:.4f} UnoC={uno_fused:.4f} | weights={dict(zip(model_names, np.round(selected_weights,3)))}")

    # Save global OOF predictions
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(outdir / "oof_predictions.csv", index=False)

    # Per-fold table (include base scores for quick scan)
    rows_pf = []
    for m in all_fold_metrics:
        row = {
            "fold": m["fold"],
            "harrell_fused": m["harrell_fused"],
            "uno_fused": m["uno_fused"],
        }
        for k, v in m["selected_weights"].items():
            row[f"w_{k}"] = v
        for k, sc in m["base_scores"].items():
            row[f"harrell_{k}"] = sc["harrell"]
            row[f"uno_{k}"] = sc["uno"]
        rows_pf.append(row)
    met_df = pd.DataFrame(rows_pf)
    met_df.to_csv(outdir / "per_fold_metrics.csv", index=False)

    # Summary (fused)
    har_vals = [float(m["harrell_fused"]) for m in all_fold_metrics]
    uno_vals = [float(m["uno_fused"]) if m["uno_fused"] is not None else np.nan for m in all_fold_metrics]
    har_mean, har_std = mean_std(har_vals)
    uno_mean, uno_std = mean_std(uno_vals)

    # Overall OOF
    y_oof = make_surv_struct(oof_df["event"].to_numpy().astype(int), oof_df["time"].to_numpy().astype(float))
    tau_oof = compute_tau_from_train(y_oof, quantile=float(args.tau_q))

    def _oof_metric(col: str) -> Tuple[float, float]:
        risk = oof_df[col].to_numpy().astype(float)
        har = float(concordance_index_censored(y_oof["event"], y_oof["time"], risk)[0])
        uno = safe_uno_cindex(y_oof, y_oof, risk, tau=tau_oof)
        return har, uno

    har_oof_raw, uno_oof_raw = _oof_metric("risk_fused_raw")
    har_oof_oofnorm, uno_oof_oofnorm = _oof_metric("risk_fused_oof")

    # base per-fold mean (processed scale)
    base_means = {}
    for k in model_names:
        har_k = [float(m["base_scores"][k]["harrell"]) for m in all_fold_metrics]
        uno_k = [float(m["base_scores"][k]["uno"]) if m["base_scores"][k]["uno"] is not None else np.nan for m in all_fold_metrics]
        base_means[k] = {
            "harrell_mean": round(float(np.nanmean(har_k)), 6),
            "harrell_std": round(float(np.nanstd(har_k, ddof=1)), 6) if len(har_k) > 1 else 0.0,
            "uno_mean": None if np.isnan(np.nanmean(uno_k)) else round(float(np.nanmean(uno_k)), 6),
            "uno_std": None if np.isnan(np.nanstd(uno_k, ddof=1)) else round(float(np.nanstd(uno_k, ddof=1)), 6),
        }

    # -------------------------------------------------------------------------
    # [NEW] Dominance Analysis: Check if one modality dominates > 95% weight
    # -------------------------------------------------------------------------
    dominance_threshold = 0.95
    dominance_counts = {m: 0 for m in model_names}
    dominance_counts["Mixed"] = 0
    
    # Track per-fold dominance
    fold_dominance_info = {} 

    for m_metrics in all_fold_metrics:
        fold_k = m_metrics["fold"]
        weights = m_metrics["selected_weights"]  # e.g., {'Tb': 0.6, 'Df': 0.4}
        
        # Find the max weight in this fold
        max_w = -1.0
        dominant_model = None
        
        for name, w_val in weights.items():
            if w_val > max_w:
                max_w = w_val
                dominant_model = name
        
        # Check against threshold
        if max_w >= dominance_threshold:
            dominance_counts[dominant_model] += 1
            status = f"Dominated by {dominant_model}"
        else:
            dominance_counts["Mixed"] += 1
            status = "Mixed"
            
        fold_dominance_info[fold_k] = {
            "status": status,
            "max_weight": float(max_w),
            "dominant_model": dominant_model if max_w >= dominance_threshold else None
        }

    print("\n[Analysis] Weight Dominance (Threshold >= 0.95):")
    print(json.dumps(dominance_counts, indent=2))
    # -------------------------------------------------------------------------

    summary = {
        "exp_id": str(args.exp_id),
        "mode": str(args.mode),
        "fusion_mode": str(args.fusion_mode),
        "models": list(model_names),
        "n_folds": int(len(all_fold_metrics)),
        "per_fold_mean_harrell_fused": round(float(har_mean), 6),
        "per_fold_std_harrell_fused": round(float(har_std), 6),
        "per_fold_mean_uno_fused": None if np.isnan(uno_mean) else round(float(uno_mean), 6),
        "per_fold_std_uno_fused": None if np.isnan(uno_std) else round(float(uno_std), 6),
        "oof_harrell_raw": round(float(har_oof_raw), 6),
        "oof_uno_raw": None if np.isnan(uno_oof_raw) else round(float(uno_oof_raw), 6),
        "oof_harrell_oofnorm": round(float(har_oof_oofnorm), 6),
        "oof_uno_oofnorm": None if np.isnan(uno_oof_oofnorm) else round(float(uno_oof_oofnorm), 6),
        "norm": str(args.norm),
        "oof_norm": str(args.oof_norm),
        "alpha_step": float(args.alpha_step),
        "alpha_select": str(args.alpha_select),
        "prefer_less_mixing": bool(args.prefer_less_mixing),
        "inner_folds": int(args.inner_folds),
        "inner_metric": str(args.inner_metric),
        "tau_q": float(args.tau_q),
        "seed": int(args.seed),
        "base_per_fold_stats_proc_scale": base_means,
        "dominance_analysis": {
            "threshold": dominance_threshold,
            "counts": dominance_counts,
            "details_per_fold": fold_dominance_info
        },
        "note": "R6 late fusion on OOF risks with nested CV weight selection. Normalization fitted on train only.",
    }
    write_json(outdir / "summary.json", summary)
    dump_args(outdir / "params.json", args)

    print("\n" + "=" * 72)
    print("[R6] Done.")
    print(f"[R6] Per-fold avg fused: HarrellC={summary['per_fold_mean_harrell_fused']:.4f}±{summary['per_fold_std_harrell_fused']:.4f} | "
          f"UnoC={summary['per_fold_mean_uno_fused']}±{summary['per_fold_std_uno_fused']}")
    print(f"[R6] Overall OOF raw: HarrellC={summary['oof_harrell_raw']:.4f} | UnoC={summary['oof_uno_raw']}")
    print(f"[R6] Overall OOF ({args.oof_norm}): HarrellC={summary['oof_harrell_oofnorm']:.4f} | UnoC={summary['oof_uno_oofnorm']}")
    print(f"[R6] Outputs: {outdir.resolve()}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("R6: late fusion on OOF risk predictions (nested CV)")

    ap.add_argument("--exp_id", type=str, default="R6")
    ap.add_argument("--mode", type=str, default="R6", choices=["R6", "R0"],
                    help="R6: normal; R0: permute labels for sanity check.")
    ap.add_argument("--fusion_mode", type=str, default="blend", choices=["blend", "winner"])

    ap.add_argument("--oof_csv", type=str, action="append", required=True,
                    help="Repeatable. Format: Name:/path/to/oof_predictions.csv . "
                         "Each CSV must contain id_col and risk_col.")

    ap.add_argument("--labels_csv", type=str, default=None,
                    help="Optional. If omitted, script will try to use time/event columns from the first oof file.")

    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--id_col", type=str, default="PatientID")
    ap.add_argument("--time_col", type=str, default="Survival.time")
    ap.add_argument("--event_col", type=str, default="deadstatus.event")

    ap.add_argument("--risk_col", type=str, default="risk_pred_raw",
                    help="Column name in each oof CSV for base risk predictions.")

    ap.add_argument("--norm", type=str, default="rank_traincdf",
                    choices=["none", "zscore", "rank_traincdf", "rank_traincdf_only",
                             "quantile_normal", "quantile_uniform"],
                    help="Per-model normalization fitted on train only.")

    ap.add_argument("--auto_sign", action="store_true", help="Flip sign on train if c-index < 0.5 (direction alignment).")
    ap.set_defaults(auto_sign=True)

    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--inner_metric", type=str, default="harrell", choices=["harrell", "uno"])
    ap.add_argument("--tau_q", type=float, default=0.9)

    ap.add_argument("--alpha_step", type=float, default=0.05,
                    help="Grid step for blend weights. For >2 models, step must divide 1.0.")
    ap.add_argument("--alpha_select", type=str, default="1se", choices=["best", "1se"])
    ap.add_argument("--prefer_less_mixing", action="store_true",
                    help="When alpha_select=1se, pick the least-mixing weights within 1-SE.")
    ap.set_defaults(prefer_less_mixing=True)

    ap.add_argument("--oof_norm", type=str, default="zscore",
                    choices=["none", "zscore", "rank_traincdf", "rank_traincdf_only",
                             "quantile_normal", "quantile_uniform"],
                    help="Fold-specific normalization applied to fused risk for cross-fold comparability.")

    ap.add_argument("--seed", type=int, default=42)
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_r6(args)
