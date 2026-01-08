#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
r6_stack_prob_kpi_fixed.py
==========================

Fixed version that handles:
1. Auto-detect risk column names (risk, risk_pred_raw, risk_pred, etc.)
2. Properly preserve time/event columns after merge
3. Robust IBS time grid calculation within validation data range
4. Better error messages
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

# ---- scikit-survival ----
try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import (
        concordance_index_censored,
        concordance_index_ipcw,
        brier_score,
        integrated_brier_score,
        cumulative_dynamic_auc,
    )
    from sksurv.util import Surv
    from sksurv.nonparametric import kaplan_meier_estimator
except Exception as e:
    raise RuntimeError(
        "Missing dependency: scikit-survival (sksurv). "
        "Install it in your conda env, e.g.:\n"
        "  conda install -c conda-forge scikit-survival\n"
        f"Original import error: {e}"
    )


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def zscore_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def zscore_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def to_surv_struct(time: np.ndarray, event: np.ndarray):
    return Surv.from_arrays(event=event.astype(bool), time=time.astype(float))


def km_survival_at_tau(time: np.ndarray, event: np.ndarray, tau: float) -> float:
    """KM estimate of S(tau) given (time, event)."""
    if len(time) == 0:
        return float("nan")
    try:
        t, s = kaplan_meier_estimator(event.astype(bool), time.astype(float))
        idx = np.searchsorted(t, tau, side="right") - 1
        if idx < 0:
            return 1.0
        return float(s[idx])
    except Exception:
        return float("nan")


def make_times_grid_safe(y_train, y_val, tau: float, n_grid: int = 50) -> np.ndarray:
    """
    Generate time grid for IBS that is STRICTLY within both train and val data range.
    scikit-survival requires: all times must be within [min(test_time), max(test_time))
    """
    # Get time ranges
    tmin_tr = float(np.min(y_train["time"]))
    tmax_tr = float(np.max(y_train["time"]))
    tmin_va = float(np.min(y_val["time"]))
    tmax_va = float(np.max(y_val["time"]))
    
    # The critical constraint: times must be < max(val_time)
    # Use a conservative margin
    upper_bound = min(tau, tmax_va - 1.0, tmax_tr - 1.0)
    lower_bound = max(1.0, tmin_tr, tmin_va)
    
    if upper_bound <= lower_bound:
        # Fallback: use a single safe point
        safe_point = min(tau, tmax_va * 0.9, tmax_tr * 0.9)
        if safe_point <= lower_bound:
            safe_point = lower_bound + 1.0
        return np.array([safe_point], dtype=float)
    
    times = np.linspace(lower_bound, upper_bound, n_grid, dtype=float)
    
    # Extra safety: ensure all times are strictly less than max val time
    times = times[times < tmax_va - 0.5]
    
    if len(times) == 0:
        return np.array([min(tau, tmax_va - 1.0)], dtype=float)
    
    return times


def eval_survival_functions_at_times(surv_funcs, times: np.ndarray) -> np.ndarray:
    """Evaluate survival functions at given time points."""
    out = np.empty((len(surv_funcs), len(times)), dtype=float)
    for i, fn in enumerate(surv_funcs):
        out[i, :] = np.array([float(fn(t)) for t in times], dtype=float)
    return out


def parse_oof_arg(s: str) -> Tuple[str, Path]:
    """Parse --oof like 'Tb:/path/to/file.csv'"""
    if ":" not in s:
        raise ValueError(f"--oof must be like TAG:/path/file.csv, got: {s}")
    tag, path = s.split(":", 1)
    tag = tag.strip()
    path = Path(path).expanduser()
    if tag == "":
        raise ValueError(f"Empty TAG in --oof {s}")
    return tag, path


def find_risk_column(df: pd.DataFrame, preferred: str = "risk_pred_raw") -> str:
    """Auto-detect risk column from common names."""
    candidates = [
        preferred,
        "risk", "risk_pred", "risk_pred_raw", "pred", "oof_pred",
        "risk_integral", "oof_risk_integral", "risk_fused", "risk_min"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find risk column. Tried: {candidates}. Available: {list(df.columns)}")


def find_id_column(df: pd.DataFrame, preferred: str = "PatientID") -> str:
    """Auto-detect ID column."""
    candidates = [preferred, "PatientID", "patient_id", "pid", "PID", "ID", "id"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find ID column. Tried: {candidates}. Available: {list(df.columns)}")


@dataclass
class FoldResult:
    fold: str
    n_train: int
    n_val: int
    alpha: float
    harrell_c: float
    uno_c: float
    auc_tau: float
    brier_tau: float
    ibs_0_tau: float
    cal_obs_tau: float
    cal_pred_mean: float
    cal_in_the_large: float


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True, type=str)
    ap.add_argument("--splits_json", required=True, type=str)
    ap.add_argument("--outdir", required=True, type=str)

    ap.add_argument("--oof", action="append", required=True,
                    help="Repeatable. Format TAG:/abs/path/oof_predictions.csv")

    ap.add_argument("--id_col", default="PatientID", type=str)
    ap.add_argument("--time_col", default="Survival.time", type=str)
    ap.add_argument("--event_col", default="deadstatus.event", type=str)
    ap.add_argument("--risk_col", default="auto", type=str,
                    help="Risk column name in OOF files. Use 'auto' for auto-detection.")

    ap.add_argument("--tau", default=730.0, type=float)

    # stacking model hyperparams
    ap.add_argument("--alpha_grid", nargs="+", type=float,
                    default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    ap.add_argument("--inner_folds", default=3, type=int)
    ap.add_argument("--inner_metric", default="brier", choices=["brier", "ibs"])
    ap.add_argument("--n_grid_ibs", default=50, type=int)

    # risk preprocessing
    ap.add_argument("--oof_norm", default="zscore", choices=["none", "zscore"])
    ap.add_argument("--auto_sign", default=1, type=int)

    # calibration
    ap.add_argument("--n_calib_bins", default=5, type=int)

    args = ap.parse_args()

    labels_csv = Path(args.labels_csv)
    splits_json = Path(args.splits_json)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # ---- load labels ----
    print(f"[INFO] Loading labels from: {labels_csv}")
    lab = pd.read_csv(labels_csv)
    
    if args.id_col not in lab.columns:
        raise ValueError(f"id_col={args.id_col} not found. Available: {list(lab.columns)}")
    if args.time_col not in lab.columns:
        raise ValueError(f"time_col={args.time_col} not found. Available: {list(lab.columns)}")
    if args.event_col not in lab.columns:
        raise ValueError(f"event_col={args.event_col} not found. Available: {list(lab.columns)}")

    # Create base dataframe with survival info
    y_df = pd.DataFrame({
        "patient_id": lab[args.id_col].astype(str).str.strip(),
        "time": pd.to_numeric(lab[args.time_col], errors="coerce"),
        "event": pd.to_numeric(lab[args.event_col], errors="coerce").fillna(0).astype(int),
    })
    
    if y_df["time"].isna().any():
        bad = y_df[y_df["time"].isna()]
        raise ValueError(f"Found NaN times in labels_csv. Example:\n{bad.head(5)}")
    
    print(f"[INFO] Loaded {len(y_df)} patients from labels")
    print(f"[INFO] Time range: [{y_df['time'].min():.1f}, {y_df['time'].max():.1f}]")
    print(f"[INFO] Events: {y_df['event'].sum()} / {len(y_df)}")

    # ---- load oof risks and merge ----
    feat_df = y_df.copy()
    tags: List[str] = []
    
    for oof_arg in args.oof:
        tag, path = parse_oof_arg(oof_arg)
        print(f"[INFO] Loading OOF '{tag}' from: {path}")
        
        df = pd.read_csv(path)
        print(f"       Columns: {list(df.columns)}")
        
        # Find ID column
        pid_col = find_id_column(df, args.id_col)
        
        # Find risk column
        if args.risk_col == "auto":
            risk_col = find_risk_column(df)
        else:
            if args.risk_col not in df.columns:
                # Try auto-detection as fallback
                print(f"[WARN] Specified risk_col={args.risk_col} not found, trying auto-detection...")
                risk_col = find_risk_column(df, args.risk_col)
            else:
                risk_col = args.risk_col
        
        print(f"       Using ID col: '{pid_col}', Risk col: '{risk_col}'")
        
        tmp = pd.DataFrame({
            "patient_id": df[pid_col].astype(str).str.strip(),
            tag: pd.to_numeric(df[risk_col], errors="coerce"),
        })
        
        if tmp[tag].isna().any():
            n_nan = tmp[tag].isna().sum()
            print(f"[WARN] OOF {tag} has {n_nan} NaN risk values, filling with median")
            tmp[tag] = tmp[tag].fillna(tmp[tag].median())
        
        feat_df = feat_df.merge(tmp, on="patient_id", how="left")
        tags.append(tag)
        print(f"       Merged {len(tmp)} rows, {feat_df[tag].notna().sum()} matched")

    # Check for missing values
    for tag in tags:
        n_missing = feat_df[tag].isna().sum()
        if n_missing > 0:
            print(f"[WARN] {n_missing} patients missing OOF '{tag}', filling with median")
            feat_df[tag] = feat_df[tag].fillna(feat_df[tag].median())

    print(f"[INFO] Final feature matrix: {len(feat_df)} patients x {len(tags)} OOF risks")

    # ---- load splits ----
    with open(splits_json, "r") as f:
        splits = json.load(f)

    fold_keys = sorted([k for k in splits.keys() if k.startswith("fold_")])
    if not fold_keys:
        raise ValueError(f"No fold_* keys found in splits_json: {splits_json}")
    
    print(f"[INFO] Found {len(fold_keys)} folds: {fold_keys}")

    fold_results: List[FoldResult] = []
    all_oof_rows = []

    # per-fold evaluation
    for fk in fold_keys:
        print(f"\n{'='*60}")
        print(f"Processing {fk}")
        print(f"{'='*60}")
        
        tr_ids = [str(x).strip() for x in splits[fk]["train"]]
        va_ids = [str(x).strip() for x in splits[fk]["val"]]

        tr_mask = feat_df["patient_id"].isin(tr_ids)
        va_mask = feat_df["patient_id"].isin(va_ids)

        tr = feat_df.loc[tr_mask].reset_index(drop=True)
        va = feat_df.loc[va_mask].reset_index(drop=True)
        
        print(f"  Train: {len(tr)}, Val: {len(va)}")
        print(f"  Train time range: [{tr['time'].min():.1f}, {tr['time'].max():.1f}]")
        print(f"  Val time range: [{va['time'].min():.1f}, {va['time'].max():.1f}]")

        y_tr = to_surv_struct(tr["time"].to_numpy(), tr["event"].to_numpy())
        y_va = to_surv_struct(va["time"].to_numpy(), va["event"].to_numpy())

        X_tr = tr[tags].to_numpy(dtype=float)
        X_va = va[tags].to_numpy(dtype=float)

        # normalize base risks
        if args.oof_norm == "zscore":
            mu, sd = zscore_fit(X_tr)
            X_tr = zscore_apply(X_tr, mu, sd)
            X_va = zscore_apply(X_va, mu, sd)

        # auto sign per feature
        if args.auto_sign == 1:
            for j, tag in enumerate(tags):
                c0 = concordance_index_censored(y_tr["event"], y_tr["time"], X_tr[:, j])[0]
                c1 = concordance_index_censored(y_tr["event"], y_tr["time"], -X_tr[:, j])[0]
                if np.isnan(c0) or np.isnan(c1):
                    continue
                if c1 > c0 + 1e-6:
                    X_tr[:, j] *= -1.0
                    X_va[:, j] *= -1.0
                    print(f"  [auto_sign] Flipped sign of '{tag}'")

        # inner-CV to select alpha
        skf = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=42)
        y_event_tr = tr["event"].to_numpy(dtype=int)

        best_alpha = None
        best_score = None

        for alpha in args.alpha_grid:
            inner_scores = []
            for tr_idx, te_idx in skf.split(X_tr, y_event_tr):
                X_in, X_te = X_tr[tr_idx], X_tr[te_idx]
                y_in, y_te = y_tr[tr_idx], y_tr[te_idx]

                try:
                    est = CoxPHSurvivalAnalysis(alpha=float(alpha))
                    est.fit(X_in, y_in)
                    surv_funcs = est.predict_survival_function(X_te)

                    if args.inner_metric == "brier":
                        # Use a safe tau for inner CV
                        tau_inner = min(float(args.tau), 
                                       float(np.max(y_in["time"])) - 1.0,
                                       float(np.max(y_te["time"])) - 1.0)
                        if tau_inner <= 0:
                            tau_inner = float(np.max(y_te["time"])) * 0.9
                        
                        times = np.array([tau_inner], dtype=float)
                        surv_hat = eval_survival_functions_at_times(surv_funcs, times)
                        _, bs = brier_score(y_in, y_te, surv_hat, times)
                        inner_scores.append(float(bs[0]))
                    else:
                        times = make_times_grid_safe(y_in, y_te, float(args.tau), int(args.n_grid_ibs))
                        surv_hat = eval_survival_functions_at_times(surv_funcs, times)
                        ibs_val = integrated_brier_score(y_in, y_te, surv_hat, times)
                        inner_scores.append(float(ibs_val))
                except Exception as e:
                    print(f"  [WARN] Inner CV failed for alpha={alpha}: {e}")
                    continue

            if len(inner_scores) > 0:
                score = float(np.mean(inner_scores))
                if (best_score is None) or (score < best_score):
                    best_score = score
                    best_alpha = float(alpha)

        if best_alpha is None:
            print(f"  [WARN] No valid alpha found, using default 0.01")
            best_alpha = 0.01

        print(f"  Selected alpha={best_alpha:.4g} (inner {args.inner_metric}={best_score:.4f})")

        # fit final on outer train
        model = CoxPHSurvivalAnalysis(alpha=best_alpha)
        model.fit(X_tr, y_tr)

        # predictions on outer val
        surv_funcs_va = model.predict_survival_function(X_va)
        
        # Calculate tau_eff: must be within val data range
        tau_eff = min(float(args.tau), 
                     float(np.max(y_tr["time"])) - 1.0,
                     float(np.max(y_va["time"])) - 1.0)
        
        S_tau = np.array([float(fn(tau_eff)) for fn in surv_funcs_va], dtype=float)
        risk_lp = model.predict(X_va).astype(float)

        # metrics
        har_c = float(concordance_index_censored(y_va["event"], y_va["time"], risk_lp)[0])
        
        try:
            uno_c = float(concordance_index_ipcw(y_tr, y_va, risk_lp, tau=tau_eff)[0])
        except Exception as e:
            print(f"  [WARN] Uno C-index failed: {e}, using Harrell")
            uno_c = har_c

        # time-dependent AUC at tau
        try:
            auc_result = cumulative_dynamic_auc(y_tr, y_va, risk_lp, times=[tau_eff])
            if isinstance(auc_result, tuple):
                _, aucs = auc_result
                auc_tau = float(aucs) if np.isscalar(aucs) else float(aucs[0])
            else:
                auc_tau = float(auc_result)
        except Exception as e:
            print(f"  [WARN] AUC failed: {e}, using 0.5")
            auc_tau = 0.5

        # Brier at tau
        try:
            times_tau = np.array([tau_eff], dtype=float)
            surv_hat_tau = S_tau.reshape(-1, 1)
            _, bs_tau = brier_score(y_tr, y_va, surv_hat_tau, times_tau)
            brier_tau = float(bs_tau[0])
        except Exception as e:
            print(f"  [WARN] Brier score failed: {e}")
            brier_tau = float("nan")

        # IBS with safe time grid
        try:
            times_ibs = make_times_grid_safe(y_tr, y_va, tau_eff, int(args.n_grid_ibs))
            surv_hat_ibs = eval_survival_functions_at_times(surv_funcs_va, times_ibs)
            ibs = float(integrated_brier_score(y_tr, y_va, surv_hat_ibs, times_ibs))
        except Exception as e:
            print(f"  [WARN] IBS failed: {e}")
            ibs = float("nan")

        # calibration at tau
        calib = pd.DataFrame({
            "patient_id": va["patient_id"].values,
            "time": va["time"].values,
            "event": va["event"].values,
            "S_tau": S_tau,
            "risk_lp": risk_lp,
        }).sort_values("S_tau", ascending=True).reset_index(drop=True)

        n_bins = int(args.n_calib_bins)
        try:
            calib["bin"] = pd.qcut(calib["S_tau"], q=n_bins, labels=False, duplicates="drop")
        except Exception:
            # Fallback to cut if qcut fails
            calib["bin"] = pd.cut(calib["S_tau"], bins=n_bins, labels=False)

        bin_rows = []
        for b in sorted(calib["bin"].dropna().unique()):
            sub = calib[calib["bin"] == b]
            obs = km_survival_at_tau(sub["time"].to_numpy(), sub["event"].to_numpy(), tau_eff)
            bin_rows.append({
                "fold": fk,
                "bin": int(b),
                "n": int(len(sub)),
                "pred_mean": float(sub["S_tau"].mean()),
                "pred_median": float(sub["S_tau"].median()),
                "obs_km_S_tau": float(obs),
                "events_le_tau": int(np.sum((sub["event"] == 1) & (sub["time"] <= tau_eff))),
                "censored_le_tau": int(np.sum((sub["event"] == 0) & (sub["time"] <= tau_eff))),
                "S_tau_min": float(sub["S_tau"].min()),
                "S_tau_max": float(sub["S_tau"].max()),
            })
        calib_bins = pd.DataFrame(bin_rows)
        calib_bins.to_csv(outdir / f"calibration_bins_{fk}.csv", index=False)

        # calibration-in-the-large
        obs_all = km_survival_at_tau(calib["time"].to_numpy(), calib["event"].to_numpy(), tau_eff)
        pred_mean_all = float(calib["S_tau"].mean())
        cal_itl = float(pred_mean_all - obs_all)

        fold_results.append(FoldResult(
            fold=fk,
            n_train=int(len(tr)),
            n_val=int(len(va)),
            alpha=float(best_alpha),
            harrell_c=har_c,
            uno_c=uno_c,
            auc_tau=auc_tau,
            brier_tau=brier_tau,
            ibs_0_tau=ibs,
            cal_obs_tau=float(obs_all),
            cal_pred_mean=pred_mean_all,
            cal_in_the_large=cal_itl,
        ))

        # store oof rows
        for i in range(len(calib)):
            all_oof_rows.append({
                "fold": fk,
                "patient_id": calib.iloc[i]["patient_id"],
                "time": float(calib.iloc[i]["time"]),
                "event": int(calib.iloc[i]["event"]),
                "S_tau": float(calib.iloc[i]["S_tau"]),
                "risk_lp": float(calib.iloc[i]["risk_lp"]),
            })

        print(
            f"  Results: HarrellC={har_c:.4f} UnoC={uno_c:.4f} AUC@tau={auc_tau:.4f} "
            f"Brier@tau={brier_tau:.4f} IBS={ibs:.4f}"
        )
        print(f"  Calibration: ObsKM={obs_all:.3f} PredMean={pred_mean_all:.3f} ITL={cal_itl:+.3f}")

    # ---- aggregate outputs ----
    fold_metrics = pd.DataFrame([fr.__dict__ for fr in fold_results])
    fold_metrics.to_csv(outdir / "fold_metrics.csv", index=False)

    oof_pred = pd.DataFrame(all_oof_rows).sort_values(["patient_id", "fold"])
    oof_pred.to_csv(outdir / "oof_pred_S_tau.csv", index=False)

    summary = {
        "tau": float(args.tau),
        "tau_eff_used": float(tau_eff),
        "models": tags,
        "inner_metric": args.inner_metric,
        "alpha_grid": [float(x) for x in args.alpha_grid],
        "n_folds": int(len(fold_results)),
        "metrics_mean": {
            "harrell_c": float(fold_metrics["harrell_c"].mean()),
            "uno_c": float(fold_metrics["uno_c"].mean()),
            "auc_tau": float(fold_metrics["auc_tau"].mean()),
            "brier_tau": float(fold_metrics["brier_tau"].mean()),
            "ibs_0_tau": float(fold_metrics["ibs_0_tau"].mean()),
            "cal_obs_tau": float(fold_metrics["cal_obs_tau"].mean()),
            "cal_pred_mean": float(fold_metrics["cal_pred_mean"].mean()),
            "cal_in_the_large": float(fold_metrics["cal_in_the_large"].mean()),
        },
        "metrics_std": {
            "harrell_c": float(fold_metrics["harrell_c"].std(ddof=1)),
            "uno_c": float(fold_metrics["uno_c"].std(ddof=1)),
            "auc_tau": float(fold_metrics["auc_tau"].std(ddof=1)),
            "brier_tau": float(fold_metrics["brier_tau"].std(ddof=1)),
            "ibs_0_tau": float(fold_metrics["ibs_0_tau"].std(ddof=1)),
            "cal_obs_tau": float(fold_metrics["cal_obs_tau"].std(ddof=1)),
            "cal_pred_mean": float(fold_metrics["cal_pred_mean"].std(ddof=1)),
            "cal_in_the_large": float(fold_metrics["cal_in_the_large"].std(ddof=1)),
        },
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Tau requested: {args.tau}, Tau effective: {tau_eff:.1f}")
    print(f"Models: {tags}")
    print("\nMetrics (mean ± std):")
    for k in ["harrell_c", "uno_c", "auc_tau", "brier_tau", "ibs_0_tau"]:
        m = summary["metrics_mean"][k]
        s = summary["metrics_std"][k]
        print(f"  {k}: {m:.4f} ± {s:.4f}")
    print(f"\nCalibration-in-the-large: {summary['metrics_mean']['cal_in_the_large']:+.3f}")
    print(f"\nSaved to: {outdir}")


if __name__ == "__main__":
    main()