#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R6 Meta-Learner (Stacking) for survival late fusion.

What this script does
---------------------
Given multiple base-model OOF prediction CSVs (e.g., Tb, Cp, Df),
it trains a *meta* Cox model (ridge CoxPH) inside each outer fold
(using only OUTER-train) and evaluates on OUTER-val.

Compared to simple convex blending, Cox stacking:
  - allows non-convex / negative weights (if needed),
  - rescales automatically via coefficients,
  - can optionally add "Stage-aware" interactions with *regularization*
    (safer than fitting separate weights per stage on tiny subgroups).

Inputs
------
- --oof_csv name:path (repeatable): base model OOF CSVs. Each should have:
    PatientID, and the chosen --risk_col (default: risk_pred_raw).
- --labels_csv: CSV with survival labels; defaults match your R5 pipeline:
    time_col=Survival.time, event_col=deadstatus.event
- --splits_json: outer CV folds (PatientID lists)
- Optional stage info:
    --stage_csv: a CSV with PatientID and stage one-hot columns
    --stage_prefix: prefix to auto-pick stage one-hot columns (e.g., "Overall_Stage_")
    --stage_strategy: none | stage3 | onehot
        stage3: create 3 coarse groups: Early(I/II), IIIa, IIIb+
        onehot: use the picked one-hot columns directly

Outputs (outdir)
----------------
- params.json
- per_fold_metrics.csv
- summary.json
- oof_predictions.csv  (fused risk + base risks)
- fold_*/meta_coefficients.json

Dependencies
------------
Requires scikit-survival (`sksurv`). If missing, this script raises.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

# ---- scikit-survival (required)
try:
    from sksurv.util import Surv
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
except Exception as e:
    raise ImportError(
        "This script requires scikit-survival (sksurv). "
        "Install it (e.g., conda install -c conda-forge scikit-survival). "
        f"Original import error: {e}"
    )


# -------------------------
# Metrics
# -------------------------
def harrell_c_index(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C using sksurv implementation (preferred)."""
    # risk: higher -> worse survival (higher hazard)
    c = concordance_index_censored(event, time, risk)[0]
    return float(c)


def uno_c_index(event_train: np.ndarray, time_train: np.ndarray,
                event_test: np.ndarray, time_test: np.ndarray,
                risk_test: np.ndarray, tau: float) -> float:
    """Uno's C (IPCW)."""
    y_train = Surv.from_arrays(event_train, time_train)
    y_test = Surv.from_arrays(event_test, time_test)
    c = concordance_index_ipcw(y_train, y_test, risk_test, tau=tau)[0]
    return float(c)


def tau_from_train(time_train: np.ndarray, q: float) -> float:
    """Pick tau as a quantile of training times (common in IPCW C-index)."""
    q = float(q)
    q = min(max(q, 0.50), 0.99)
    return float(np.quantile(time_train, q))


# -------------------------
# Risk transforms (train-only)
# -------------------------
@dataclass
class RiskTransform:
    kind: str
    params: Dict

    def transform(self, x: np.ndarray) -> np.ndarray:
        kind = self.kind
        p = self.params
        x = np.asarray(x, dtype=float)

        if kind == "none":
            return x

        if kind == "zscore_train":
            mu = p["mu"]
            sd = p["sd"]
            if sd <= 0:
                return np.zeros_like(x)
            return (x - mu) / sd

        if kind == "rank_traincdf":
            # map to [0,1] using empirical CDF from train
            x_train_sorted = p["x_train_sorted"]
            # rank: number of train points <= x
            ranks = np.searchsorted(x_train_sorted, x, side="right")
            n = len(x_train_sorted)
            if n <= 1:
                return np.full_like(x, 0.5)
            return ranks / n

        if kind == "quantile_train":
            # sklearn QuantileTransformer would be ideal, but we implement a lightweight version:
            # map via train CDF then inverse normal (approx).
            x_train_sorted = p["x_train_sorted"]
            ranks = np.searchsorted(x_train_sorted, x, side="right")
            n = len(x_train_sorted)
            if n <= 1:
                return np.zeros_like(x)
            u = ranks / n
            # clamp
            eps = 1e-6
            u = np.clip(u, eps, 1 - eps)
            # inverse normal via erfinv
            return np.sqrt(2) * scipy_erfinv(2*u - 1)

        raise ValueError(f"Unknown transform kind: {kind}")


def scipy_erfinv(y: np.ndarray) -> np.ndarray:
    """erfinv fallback without scipy."""
    # Use approximation for inverse error function (Winitzki approximation)
    # Good enough for transforming ranks; avoids scipy dependency.
    y = np.asarray(y, dtype=float)
    a = 0.147  # magic constant
    sgn = np.sign(y)
    ln = np.log(1 - y*y)
    first = 2/(math.pi*a) + ln/2
    second = ln/a
    inside = first*first - second
    return sgn * np.sqrt(np.sqrt(inside) - first)


def fit_risk_transform(
    x_train: np.ndarray,
    y_train_event: np.ndarray,
    y_train_time: np.ndarray,
    kind: str,
    auto_sign: bool = True,
) -> Tuple[RiskTransform, float]:
    """
    Fit transform on training risk only.
    Also optionally flips sign if Harrell C < 0.5 on train (auto_sign).

    Returns:
      (transform, sign)
    """
    x_train = np.asarray(x_train, dtype=float)

    sign = 1.0
    if auto_sign:
        c = harrell_c_index(y_train_event, y_train_time, x_train)
        if c < 0.5:
            sign = -1.0
            x_train = -x_train

    kind = kind.lower()

    if kind == "none":
        return RiskTransform("none", {}), sign

    if kind == "zscore_train":
        mu = float(np.mean(x_train))
        sd = float(np.std(x_train, ddof=0))
        return RiskTransform("zscore_train", {"mu": mu, "sd": sd}), sign

    if kind == "rank_traincdf":
        xs = np.sort(x_train.copy())
        return RiskTransform("rank_traincdf", {"x_train_sorted": xs}), sign

    if kind == "quantile_train":
        xs = np.sort(x_train.copy())
        return RiskTransform("quantile_train", {"x_train_sorted": xs}), sign

    raise ValueError(f"Unknown --norm: {kind}")


def zscore_all(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    if sd <= 0:
        return np.zeros_like(x)
    return (x - mu) / sd


# -------------------------
# IO helpers
# -------------------------
def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def parse_named_paths(items: List[str]) -> Dict[str, Path]:
    """
    Parse repeated args like:
      --oof_csv Tb:/path/to/oof.csv --oof_csv Cp:/path/to/oof.csv
    """
    out: Dict[str, Path] = {}
    for it in items:
        if ":" not in it:
            raise ValueError(f"--oof_csv item must be NAME:PATH, got: {it}")
        name, p = it.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty model name in --oof_csv: {it}")
        if name in out:
            raise ValueError(f"Duplicate model name in --oof_csv: {name}")
        out[name] = Path(p).expanduser().resolve()
    if len(out) < 2:
        raise ValueError("Stacking needs at least 2 base models (>=2 --oof_csv entries).")
    return out


def load_labels(labels_csv: Path, id_col: str, time_col: str, event_col: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    for c in [id_col, time_col, event_col]:
        if c not in df.columns:
            raise ValueError(f"labels_csv missing column '{c}'. Available: {df.columns.tolist()[:20]} ...")
    out = df[[id_col, time_col, event_col]].copy()
    out = out.rename(columns={time_col: "time", event_col: "event"})
    out["PatientID"] = out[id_col].astype(str)
    out["time"] = out["time"].astype(float)
    out["event"] = out["event"].astype(int)
    out["event"] = out["event"].astype(bool)
    return out[["PatientID", "time", "event"]]


def load_oof_one(path: Path, id_col: str, risk_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if id_col not in df.columns:
        raise ValueError(f"{path} missing id_col '{id_col}'. Columns: {df.columns.tolist()[:20]} ...")
    if risk_col not in df.columns:
        raise ValueError(f"{path} missing risk_col '{risk_col}'. Columns: {df.columns.tolist()[:20]} ...")
    out = df[[id_col, risk_col]].copy()
    out["PatientID"] = out[id_col].astype(str)
    out = out.rename(columns={risk_col: "risk"})
    out["risk"] = out["risk"].astype(float)
    return out[["PatientID", "risk"]]


def build_stage_matrix(
    stage_csv: Path,
    id_col: str,
    stage_prefix: str,
    strategy: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns:
      stage_df: PatientID + stage feature columns
      stage_cols: list of stage feature columns used
    """
    df = pd.read_csv(stage_csv)
    if id_col not in df.columns:
        raise ValueError(f"stage_csv missing id_col '{id_col}'. Columns: {df.columns.tolist()[:20]} ...")
    df["PatientID"] = df[id_col].astype(str)

    # pick one-hot columns by prefix
    stage_cols = [c for c in df.columns if c.startswith(stage_prefix)]
    if len(stage_cols) == 0:
        raise ValueError(
            f"No stage columns found with prefix '{stage_prefix}' in {stage_csv}. "
            f"Available columns: {df.columns.tolist()[:30]} ..."
        )

    work = df[["PatientID"] + stage_cols].copy()
    work[stage_cols] = work[stage_cols].fillna(0).astype(float)

    strategy = strategy.lower()
    if strategy == "onehot":
        return work, stage_cols

    if strategy == "stage3":
        # Expect columns like: Overall_Stage_I / II / IIIa / IIIb / Unknown (some may be missing)
        # Build coarse 3 groups: Early(I/II), IIIa, IIIb+ (IIIb or Unknown or anything else)
        def _has(name: str) -> bool:
            return f"{stage_prefix}{name}" in stage_cols

        col_I = f"{stage_prefix}I" if _has("I") else None
        col_II = f"{stage_prefix}II" if _has("II") else None
        col_IIIa = f"{stage_prefix}IIIa" if _has("IIIa") else None
        col_IIIb = f"{stage_prefix}IIIb" if _has("IIIb") else None
        col_U = f"{stage_prefix}Unknown" if _has("Unknown") else None

        # If some are missing, we treat them as zeros (already handled)
        early = np.zeros(len(work), dtype=float)
        if col_I is not None:
            early += work[col_I].to_numpy()
        if col_II is not None:
            early += work[col_II].to_numpy()

        iii_a = work[col_IIIa].to_numpy() if col_IIIa is not None else np.zeros(len(work), dtype=float)

        iii_b = np.zeros(len(work), dtype=float)
        if col_IIIb is not None:
            iii_b += work[col_IIIb].to_numpy()
        if col_U is not None:
            iii_b += work[col_U].to_numpy()

        # if none of the known flags are set -> treat as IIIb+ (unknown bucket)
        known_sum = early + iii_a + iii_b
        iii_b = np.where(known_sum == 0, 1.0, iii_b)

        out = pd.DataFrame({
            "PatientID": work["PatientID"].values,
            "stage3_IIIa": iii_a,
            "stage3_IIIbplus": iii_b,
        })
        return out, ["stage3_IIIa", "stage3_IIIbplus"]

    raise ValueError(f"Unknown stage_strategy: {strategy}")


# -------------------------
# Meta-learner (ridge Cox) selection
# -------------------------
def make_alpha_grid(alpha_min: float, alpha_max: float, n: int) -> np.ndarray:
    alpha_min = float(alpha_min)
    alpha_max = float(alpha_max)
    n = int(n)
    if alpha_min <= 0 or alpha_max <= 0 or alpha_max <= alpha_min:
        raise ValueError("alpha_min and alpha_max must be >0 and alpha_max>alpha_min.")
    if n < 2:
        raise ValueError("n_alphas must be >=2.")
    return np.logspace(np.log10(alpha_min), np.log10(alpha_max), n)


def select_alpha_1se(mean_sc: np.ndarray, se_sc: np.ndarray) -> int:
    """Pick *largest alpha* within 1-SE of the best score."""
    best_idx = int(np.nanargmax(mean_sc))
    threshold = float(mean_sc[best_idx] - se_sc[best_idx])
    ok = np.where(mean_sc >= threshold)[0]
    if ok.size == 0:
        return best_idx
    # alpha grid is ascending => ok.max() is largest alpha (strongest regularization)
    return int(ok.max())


def inner_cv_select_alpha(
    X: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    alpha_grid: np.ndarray,
    inner_folds: int,
    metric: str,
    tau_q: float,
    seed: int,
    alpha_select: str,
) -> Tuple[float, Dict]:
    """
    X, event, time are for OUTER-train only.
    Returns:
      chosen_alpha, diagnostics dict
    """
    metric = metric.lower()
    alpha_select = alpha_select.lower()
    if alpha_select not in ("best", "1se"):
        raise ValueError("--alpha_select must be best or 1se for stacking.")

    # stratify by event status
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    y_strat = event.astype(int)

    scores = np.full((len(alpha_grid), inner_folds), np.nan, dtype=float)

    for fold_i, (itr, ival) in enumerate(skf.split(X, y_strat)):
        X_tr, X_va = X[itr], X[ival]
        e_tr, t_tr = event[itr], time[itr]
        e_va, t_va = event[ival], time[ival]

        y_tr = Surv.from_arrays(e_tr, t_tr)

        tau = tau_from_train(t_tr, tau_q)

        for a_i, alpha in enumerate(alpha_grid):
            try:
                model = CoxPHSurvivalAnalysis(alpha=float(alpha))
                model.fit(X_tr, y_tr)
                risk_va = model.predict(X_va)

                if metric == "harrell":
                    sc = harrell_c_index(e_va, t_va, risk_va)
                elif metric == "uno":
                    sc = uno_c_index(e_tr, t_tr, e_va, t_va, risk_va, tau=tau)
                else:
                    raise ValueError("--inner_metric must be harrell or uno.")
                scores[a_i, fold_i] = sc
            except Exception:
                scores[a_i, fold_i] = np.nan

    mean_sc = np.nanmean(scores, axis=1)
    se_sc = np.nanstd(scores, axis=1, ddof=1) / math.sqrt(inner_folds)

    if np.all(np.isnan(mean_sc)):
        # total failure; fallback
        chosen_alpha = float(alpha_grid[len(alpha_grid)//2])
        chosen_idx = len(alpha_grid)//2
    else:
        if alpha_select == "best":
            chosen_idx = int(np.nanargmax(mean_sc))
        else:
            chosen_idx = select_alpha_1se(mean_sc, se_sc)
        chosen_alpha = float(alpha_grid[chosen_idx])

    diag = {
        "alpha_grid": [float(a) for a in alpha_grid],
        "mean_score": [None if np.isnan(x) else float(x) for x in mean_sc],
        "se_score": [None if np.isnan(x) else float(x) for x in se_sc],
        "chosen_alpha": float(chosen_alpha),
        "chosen_idx": int(chosen_idx),
        "metric": metric,
    }
    return chosen_alpha, diag


# -------------------------
# Main
# -------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="R6: Cox stacking meta-learner for survival late fusion")

    ap.add_argument("--exp_id", type=str, default="R6_stack", help="Experiment ID for logging")
    ap.add_argument("--oof_csv", type=str, action="append", required=True,
                    help="Base OOF CSVs as NAME:PATH (repeatable). Example: Tb:/path/oof_predictions.csv")
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--id_col", type=str, default="PatientID")
    ap.add_argument("--time_col", type=str, default="Survival.time")
    ap.add_argument("--event_col", type=str, default="deadstatus.event")

    ap.add_argument("--risk_col", type=str, default="risk_pred_raw",
                    help="Which column in each base OOF CSV to use as risk feature")

    ap.add_argument("--norm", type=str, default="zscore_train",
                    choices=["none", "zscore_train", "rank_traincdf", "quantile_train"],
                    help="Train-only transform applied to each base risk per outer fold")
    ap.add_argument("--auto_sign", action="store_true", default=True,
                    help="Flip sign on train if C-index<0.5 (per model, per outer fold)")
    ap.add_argument("--no_auto_sign", action="store_false", dest="auto_sign")

    # stacking alpha grid
    ap.add_argument("--alpha_min", type=float, default=1e-4)
    ap.add_argument("--alpha_max", type=float, default=1e4)
    ap.add_argument("--n_alphas", type=int, default=50)
    ap.add_argument("--alpha_select", type=str, default="1se", choices=["best", "1se"])

    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--inner_metric", type=str, default="harrell", choices=["harrell", "uno"])
    ap.add_argument("--tau_q", type=float, default=0.9)

    # output normalization (for display / comparison)
    ap.add_argument("--oof_norm", type=str, default="zscore", choices=["none", "zscore"])

    # stage-aware interactions (optional)
    ap.add_argument("--stage_csv", type=str, default=None,
                    help="Optional CSV providing stage one-hot columns (PatientID + stage columns).")
    ap.add_argument("--stage_prefix", type=str, default="Overall_Stage_",
                    help="Prefix to auto-pick one-hot stage cols from stage_csv.")
    ap.add_argument("--stage_strategy", type=str, default="none", choices=["none", "stage3", "onehot"],
                    help="Stage features strategy. stage3 recommended if you enable stage.")
    ap.add_argument("--use_stage_interactions", action="store_true", default=False,
                    help="If set, add interactions risk*stage (regularized).")

    ap.add_argument("--seed", type=int, default=42)

    return ap


def main() -> None:
    args = build_argparser().parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    oof_map = parse_named_paths(args.oof_csv)
    model_names = list(oof_map.keys())

    print(f"[R6-STACK] exp_id={args.exp_id}")
    print(f"[R6-STACK] models={model_names}")
    print(f"[R6-STACK] risk_col={args.risk_col} | norm={args.norm} | alpha_select={args.alpha_select}")
    print(f"[R6-STACK] stage_strategy={args.stage_strategy} | use_stage_interactions={args.use_stage_interactions}")

    # Save params
    write_json(outdir / "params.json", vars(args))

    # Load labels
    labels = load_labels(Path(args.labels_csv), args.id_col, args.time_col, args.event_col)

    # Load OOF risks
    base_frames = []
    for name, path in oof_map.items():
        df_r = load_oof_one(path, args.id_col, args.risk_col)
        df_r = df_r.rename(columns={"risk": f"risk_{name}_raw"})
        base_frames.append(df_r)

    # Merge all
    df = labels.copy()
    for bf in base_frames:
        df = df.merge(bf, on="PatientID", how="inner")

    # Optional stage matrix
    stage_cols: List[str] = []
    if args.stage_strategy != "none":
        if args.stage_csv is None:
            raise ValueError("--stage_strategy requires --stage_csv")
        stage_df, stage_cols = build_stage_matrix(
            Path(args.stage_csv), args.id_col, args.stage_prefix, args.stage_strategy
        )
        df = df.merge(stage_df, on="PatientID", how="left")
        df[stage_cols] = df[stage_cols].fillna(0).astype(float)

    n = len(df)
    print(f"[R6-STACK] n={n}")

    # Load splits
    splits = read_json(Path(args.splits_json))
    fold_keys = sorted([k for k in splits.keys() if k.startswith("fold_")],
                       key=lambda x: int(x.split("_")[1]))

    # Containers
    oof_pred_raw = np.full(n, np.nan, dtype=float)
    per_fold_rows = []
    coef_rows = []

    alpha_grid = make_alpha_grid(args.alpha_min, args.alpha_max, args.n_alphas)

    # Pre-extract arrays
    pid = df["PatientID"].to_numpy()
    time_all = df["time"].to_numpy().astype(float)
    event_all = df["event"].to_numpy().astype(bool)

    base_raw = {name: df[f"risk_{name}_raw"].to_numpy().astype(float) for name in model_names}
    stage_mat = df[stage_cols].to_numpy().astype(float) if stage_cols else None

    pid_to_idx = {p: i for i, p in enumerate(pid)}

    for fk in fold_keys:
        fold_id = int(fk.split("_")[1])
        val_ids = [str(x) for x in splits[fk]["val"]]
        val_idx = np.array([pid_to_idx[x] for x in val_ids if x in pid_to_idx], dtype=int)
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        train_idx = np.where(train_mask)[0]

        # Build train-only transformed base risks
        X_train_list = []
        X_val_list = []

        sign_map = {}
        tfm_map = {}

        for name in model_names:
            x_tr_raw = base_raw[name][train_idx]
            x_va_raw = base_raw[name][val_idx]

            tfm, sgn = fit_risk_transform(
                x_tr_raw,
                event_all[train_idx],
                time_all[train_idx],
                kind=args.norm,
                auto_sign=args.auto_sign,
            )
            sign_map[name] = float(sgn)
            tfm_map[name] = {"kind": tfm.kind, "params": {k: ("<array>" if isinstance(v, np.ndarray) else v) for k, v in tfm.params.items()}}

            x_tr = tfm.transform(x_tr_raw * sgn)
            x_va = tfm.transform(x_va_raw * sgn)

            X_train_list.append(x_tr.reshape(-1, 1))
            X_val_list.append(x_va.reshape(-1, 1))

        X_tr_base = np.hstack(X_train_list)
        X_va_base = np.hstack(X_val_list)

        # Add stage features / interactions (optional)
        X_tr = X_tr_base
        X_va = X_va_base
        feature_names = [f"{name}" for name in model_names]

        if stage_cols:
            # stage main effects (small)
            X_tr_stage = stage_mat[train_idx]
            X_va_stage = stage_mat[val_idx]

            # add stage main effects
            X_tr = np.hstack([X_tr, X_tr_stage])
            X_va = np.hstack([X_va, X_va_stage])
            feature_names += stage_cols

            if args.use_stage_interactions:
                # interactions: risk_m * stage_k
                inter_tr = []
                inter_va = []
                inter_names = []
                for mi, mname in enumerate(model_names):
                    for sj, scol in enumerate(stage_cols):
                        inter_tr.append((X_tr_base[:, mi] * X_tr_stage[:, sj]).reshape(-1, 1))
                        inter_va.append((X_va_base[:, mi] * X_va_stage[:, sj]).reshape(-1, 1))
                        inter_names.append(f"{mname}*{scol}")
                if inter_tr:
                    X_tr = np.hstack([X_tr] + inter_tr)
                    X_va = np.hstack([X_va] + inter_va)
                    feature_names += inter_names

        # Inner CV select alpha on OUTER-train only
        chosen_alpha, diag = inner_cv_select_alpha(
            X_tr, event_all[train_idx], time_all[train_idx],
            alpha_grid=alpha_grid,
            inner_folds=args.inner_folds,
            metric=args.inner_metric,
            tau_q=args.tau_q,
            seed=args.seed + fold_id,  # fold-specific
            alpha_select=args.alpha_select,
        )

        # Fit final meta model and predict on OUTER-val
        y_tr = Surv.from_arrays(event_all[train_idx], time_all[train_idx])
        model = CoxPHSurvivalAnalysis(alpha=float(chosen_alpha))
        model.fit(X_tr, y_tr)
        risk_va = model.predict(X_va)

        oof_pred_raw[val_idx] = risk_va

        # Metrics on this fold
        har = harrell_c_index(event_all[val_idx], time_all[val_idx], risk_va)
        tau = tau_from_train(time_all[train_idx], args.tau_q)
        uno = uno_c_index(event_all[train_idx], time_all[train_idx],
                          event_all[val_idx], time_all[val_idx],
                          risk_va, tau=tau)

        per_fold_rows.append({
            "fold": fold_id,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "harrell_c": float(har),
            "uno_c": float(uno),
            "alpha": float(chosen_alpha),
            "norm": args.norm,
            "stage_strategy": args.stage_strategy,
            "use_stage_interactions": bool(args.use_stage_interactions),
            "sign_map": sign_map,
        })

        # Save fold coefficients
        coef = model.coef_.reshape(-1)
        coef_dict = {feature_names[i]: float(coef[i]) for i in range(len(feature_names))}
        fold_dir = outdir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_json(fold_dir / "meta_coefficients.json", {
            "fold": fold_id,
            "alpha": float(chosen_alpha),
            "feature_names": feature_names,
            "coef": [float(x) for x in coef],
            "coef_dict": coef_dict,
            "inner_cv": diag,
        })

        # also keep in memory for summary
        coef_rows.append({
            "fold": fold_id,
            "alpha": float(chosen_alpha),
            **{f"coef::{k}": v for k, v in coef_dict.items()}
        })

        print(f"[R6-STACK][fold_{fold_id}] HarrellC={har:.4f} UnoC={uno:.4f} | alpha={chosen_alpha:.3g}")

    # Sanity check
    if np.any(np.isnan(oof_pred_raw)):
        missing = int(np.isnan(oof_pred_raw).sum())
        raise RuntimeError(f"OOF predictions have {missing} NaNs. Check splits_json alignment and PatientID matching.")

    # Overall OOF metrics
    har_oof = harrell_c_index(event_all, time_all, oof_pred_raw)
    # Uno OOF: we need a "train" reference; common workaround is to use the full set as both,
    # but that's biased. We'll report per-fold Uno avg and OOF Harrell as main.
    uno_mean = float(np.mean([r["uno_c"] for r in per_fold_rows]))

    if args.oof_norm == "zscore":
        oof_pred_norm = zscore_all(oof_pred_raw)
    else:
        oof_pred_norm = oof_pred_raw.copy()

    har_oof_norm = harrell_c_index(event_all, time_all, oof_pred_norm)

    # Save outputs
    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv(outdir / "per_fold_metrics.csv", index=False)

    oof_out = df[["PatientID", "time", "event"]].copy()
    for name in model_names:
        oof_out[f"risk_{name}_raw"] = base_raw[name]
    oof_out["risk_fused_raw"] = oof_pred_raw
    oof_out["risk_fused_oof"] = oof_pred_norm
    oof_out.to_csv(outdir / "oof_predictions.csv", index=False)

    summary = {
        "exp_id": args.exp_id,
        "models": model_names,
        "n": int(n),
        "norm": args.norm,
        "auto_sign": bool(args.auto_sign),
        "inner_folds": int(args.inner_folds),
        "inner_metric": args.inner_metric,
        "alpha_grid": [float(a) for a in alpha_grid],
        "alpha_select": args.alpha_select,
        "stage_strategy": args.stage_strategy,
        "use_stage_interactions": bool(args.use_stage_interactions),
        "per_fold_harrell_mean": float(per_fold_df["harrell_c"].mean()),
        "per_fold_harrell_std": float(per_fold_df["harrell_c"].std(ddof=1)),
        "per_fold_uno_mean": float(per_fold_df["uno_c"].mean()),
        "per_fold_uno_std": float(per_fold_df["uno_c"].std(ddof=1)),
        "harrell_c_oof_raw": float(har_oof),
        "harrell_c_oof_oofnorm": float(har_oof_norm),
        "uno_c_per_fold_mean": float(uno_mean),
    }
    write_json(outdir / "summary.json", summary)

    print("=" * 72)
    print("[R6-STACK] Done.")
    print(f"[R6-STACK] Per-fold avg: HarrellC={summary['per_fold_harrell_mean']:.4f}±{summary['per_fold_harrell_std']:.4f} "
          f"| UnoC={summary['per_fold_uno_mean']:.6f}±{summary['per_fold_uno_std']:.6f}")
    print(f"[R6-STACK] Overall OOF raw: HarrellC={summary['harrell_c_oof_raw']:.4f}")
    if args.oof_norm != "none":
        print(f"[R6-STACK] Overall OOF ({args.oof_norm}): HarrellC={summary['harrell_c_oof_oofnorm']:.4f}")
    print(f"[R6-STACK] Outputs: {outdir}")


if __name__ == "__main__":
    main()
