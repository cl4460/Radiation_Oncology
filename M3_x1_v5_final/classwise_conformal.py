#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classwise_conformal.py

Implements:
1. Standard (Marginal) Split Conformal Prediction
2. Classwise (Mondrian) Conformal Prediction  
3. Shrinkage Classwise CP (stabilized for small event counts)
4. Baselines: probability threshold, random deferral, predict-all-survive

Also produces:
- Risk-Retention Curves (sweep alpha from 0.01 to 0.50)
- Repeated random split CI for all metrics (NOT bootstrap)
- Unified comparison table

Usage:
    python classwise_conformal.py \
        --source_oof_csv /path/to/LUNG1_TRAIN_332_predictions_CORRECT.csv \
        --external_pred_csv /path/to/external_pred.csv \
        --out_dir M3_OUT/classwise_results \
        --target_repeats 200 \
        --n_splits 1000 \
        --seed 42
"""

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# Core utilities (consistent with evaluation_protocol.py)
# ============================================================================

def safe_clip(p, eps=1e-6):
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def compute_known_and_y(df, tau, time_col, event_col):
    """Unified known@tau filter + binary label computation."""
    t = df[time_col].to_numpy(dtype=float)
    e = df[event_col].to_numpy(dtype=int)
    known = (e == 1) | (t >= tau)
    y = ((e == 1) & (t <= tau)).astype(int)
    return known, y


def infer_p_event(df, tau):
    """Infer p_event from available columns."""
    if tau == 730:
        if "p_event_730d" in df.columns:
            return safe_clip(df["p_event_730d"].to_numpy(dtype=float))
        if "S_cal_730d" in df.columns:
            return safe_clip(1.0 - df["S_cal_730d"].to_numpy(dtype=float))
    raise ValueError("Cannot infer p_event. Need p_event_730d or S_cal_730d column.")


def conformal_quantile(scores, alpha):
    """Conservative finite-sample quantile for split conformal."""
    s = np.sort(scores)
    n = len(s)
    if n == 0:
        return 1.0  # degenerate: defer everything
    k = int(math.ceil((n + 1) * (1.0 - alpha))) - 1
    k = max(0, min(k, n - 1))
    return float(s[k])


def nonconformity_scores(p_event, y):
    """s_i = 1 - p_{y_i}(x_i)"""
    p = np.asarray(p_event, dtype=float)
    y = np.asarray(y, dtype=int)
    return np.where(y == 1, 1.0 - p, p)


# ============================================================================
# Prediction set construction for different methods
# ============================================================================

def pred_sets_standard(p_event, q):
    """Standard (marginal) CP: single threshold q."""
    p = np.asarray(p_event, dtype=float)
    inc0 = (p <= q)           # include "survive" if p_event small enough
    inc1 = (p >= (1.0 - q))   # include "event" if p_event large enough
    return inc0, inc1


def pred_sets_classwise(p_event, q0, q1):
    """
    Classwise (Mondrian) CP: separate thresholds per class.
    - Include label 0 if score_for_0 = p_event <= q0
    - Include label 1 if score_for_1 = 1 - p_event <= q1, i.e., p_event >= 1 - q1
    """
    p = np.asarray(p_event, dtype=float)
    inc0 = (p <= q0)
    inc1 = (p >= (1.0 - q1))
    return inc0, inc1


# ============================================================================
# Metrics computation
# ============================================================================

@dataclass
class ConformalMetrics:
    method: str
    n_test: int
    n_event_test: int
    event_rate_test: float
    
    # Thresholds
    q_marginal: Optional[float] = None
    q_class0: Optional[float] = None
    q_class1: Optional[float] = None
    
    # Coverage
    marginal_cov: float = 0.0
    event_cov: float = 0.0      # coverage on y=1
    surv_cov: float = 0.0       # coverage on y=0
    
    # Deferral
    defer_rate: float = 0.0     # fraction with |set| != 1
    defer_ambig: float = 0.0    # fraction with |set| == 2
    defer_empty: float = 0.0    # fraction with |set| == 0
    
    # Accepted performance
    retention_rate: float = 0.0  # 1 - defer_rate
    acc_accepted: float = 0.0
    sens_accepted: float = 0.0   # sensitivity among accepted
    spec_accepted: float = 0.0   # specificity among accepted
    
    # Clinical utility
    fnr_overall: float = 0.0    # missed events / total events (including deferred)
    missed_events: int = 0       # events predicted as {0} only
    
    # For risk-retention
    auc_test: Optional[float] = None


def compute_conformal_metrics(y_test, p_event_test, inc0, inc1, method_name,
                              q_marginal=None, q0=None, q1=None):
    """Compute all metrics from prediction sets."""
    y = np.asarray(y_test, dtype=int)
    p = np.asarray(p_event_test, dtype=float)
    n = len(y)
    n_ev = int(y.sum())
    
    set_size = inc0.astype(int) + inc1.astype(int)
    covered = np.where(y == 1, inc1, inc0)
    
    # Coverage
    marg_cov = float(np.mean(covered)) if n > 0 else 0.0
    ev_cov = float(np.mean(inc1[y == 1])) if n_ev > 0 else float('nan')
    su_cov = float(np.mean(inc0[y == 0])) if (n - n_ev) > 0 else float('nan')
    
    # Deferral (anything not singleton)
    accepted = (set_size == 1)
    defer = float(np.mean(~accepted))
    defer_amb = float(np.mean(set_size == 2))
    defer_emp = float(np.mean(set_size == 0))
    retention = 1.0 - defer
    
    # Predictions on accepted
    pred = np.full(n, -1, dtype=int)
    pred[accepted & inc1 & ~inc0] = 1  # singleton {1}
    pred[accepted & inc0 & ~inc1] = 0  # singleton {0}
    
    acc_acc = 0.0
    sens_acc = 0.0
    spec_acc = 0.0
    if np.any(accepted):
        acc_acc = float(np.mean(pred[accepted] == y[accepted]))
        
        # Sensitivity among accepted events
        ev_acc = accepted & (y == 1)
        if np.any(ev_acc):
            sens_acc = float(np.mean(pred[ev_acc] == 1))
        
        # Specificity among accepted survivors
        su_acc = accepted & (y == 0)
        if np.any(su_acc):
            spec_acc = float(np.mean(pred[su_acc] == 0))
    
    # Missed events: events that got singleton {0} prediction
    missed = int(np.sum((y == 1) & (inc0) & (~inc1)))
    fnr = missed / max(1, n_ev)
    
    # AUC
    auc = None
    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, p))
    
    return ConformalMetrics(
        method=method_name,
        n_test=n,
        n_event_test=n_ev,
        event_rate_test=float(n_ev / max(1, n)),
        q_marginal=q_marginal,
        q_class0=q0,
        q_class1=q1,
        marginal_cov=marg_cov,
        event_cov=ev_cov,
        surv_cov=su_cov,
        defer_rate=defer,
        defer_ambig=defer_amb,
        defer_empty=defer_emp,
        retention_rate=retention,
        acc_accepted=acc_acc,
        sens_accepted=sens_acc,
        spec_accepted=spec_acc,
        fnr_overall=fnr,
        missed_events=missed,
        auc_test=auc,
    )


# ============================================================================
# Method implementations
# ============================================================================

def run_standard_cp(p_cal, y_cal, p_test, y_test, alpha):
    """Standard (marginal) split conformal."""
    scores = nonconformity_scores(p_cal, y_cal)
    q = conformal_quantile(scores, alpha)
    inc0, inc1 = pred_sets_standard(p_test, q)
    return compute_conformal_metrics(y_test, p_test, inc0, inc1,
                                     method_name="Standard CP (marginal)",
                                     q_marginal=q)


def run_classwise_cp(p_cal, y_cal, p_test, y_test, alpha):
    """
    Classwise (Mondrian) CP: compute separate quantiles for each class.
    - Class 0 (survive): scores = p_event for y=0 samples
    - Class 1 (event):   scores = 1 - p_event for y=1 samples
    """
    p_cal = np.asarray(p_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=int)
    
    # Separate scores by class
    scores_0 = p_cal[y_cal == 0]        # score for class 0 = p_event
    scores_1 = 1.0 - p_cal[y_cal == 1]  # score for class 1 = 1 - p_event
    
    n0 = len(scores_0)
    n1 = len(scores_1)
    
    # Quantiles per class
    q0 = conformal_quantile(scores_0, alpha) if n0 >= 2 else 1.0
    q1 = conformal_quantile(scores_1, alpha) if n1 >= 2 else 1.0
    
    inc0, inc1 = pred_sets_classwise(p_test, q0, q1)
    return compute_conformal_metrics(y_test, p_test, inc0, inc1,
                                     method_name="Classwise CP (Mondrian)",
                                     q0=q0, q1=q1)


def run_shrinkage_cp(p_cal, y_cal, p_test, y_test, alpha, n_min=15):
    """
    Shrinkage Classwise CP:
    When a class has fewer than n_min calibration samples, shrink its
    class-specific quantile toward the marginal quantile for stability.
    
    shrink_factor = min(1, n_class / n_min)
    q_class_shrunk = shrink_factor * q_class + (1 - shrink_factor) * q_marginal
    
    NOTE: This is an empirical heuristic. It does NOT preserve finite-sample
    coverage guarantees. Must be validated empirically with repeated splits + CI.
    """
    p_cal = np.asarray(p_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=int)
    
    # Marginal quantile
    scores_all = nonconformity_scores(p_cal, y_cal)
    q_marg = conformal_quantile(scores_all, alpha)
    
    # Class-specific
    scores_0 = p_cal[y_cal == 0]
    scores_1 = 1.0 - p_cal[y_cal == 1]
    
    n0 = len(scores_0)
    n1 = len(scores_1)
    
    q0_raw = conformal_quantile(scores_0, alpha) if n0 >= 2 else q_marg
    q1_raw = conformal_quantile(scores_1, alpha) if n1 >= 2 else q_marg
    
    # Shrinkage
    sf0 = min(1.0, n0 / n_min)
    sf1 = min(1.0, n1 / n_min)
    
    q0 = sf0 * q0_raw + (1.0 - sf0) * q_marg
    q1 = sf1 * q1_raw + (1.0 - sf1) * q_marg
    
    inc0, inc1 = pred_sets_classwise(p_test, q0, q1)
    return compute_conformal_metrics(y_test, p_test, inc0, inc1,
                                     method_name=f"Shrinkage CP (n_min={n_min})",
                                     q_marginal=q_marg, q0=q0, q1=q1)


def run_prob_threshold(p_test, y_test, threshold=0.5):
    """
    Baseline: probability thresholding with abstention.
    Abstain if |p_event - 0.5| < margin (using same deferral rate for fair comparison).
    For simplicity: predict 1 if p_event >= threshold, else 0. No deferral.
    """
    p = np.asarray(p_test, dtype=float)
    y = np.asarray(y_test, dtype=int)
    pred = (p >= threshold).astype(int)
    
    n = len(y)
    n_ev = int(y.sum())
    
    # No deferral in this baseline
    acc = float(np.mean(pred == y))
    sens = float(np.mean(pred[y == 1] == 1)) if n_ev > 0 else 0.0
    spec = float(np.mean(pred[y == 0] == 0)) if (n - n_ev) > 0 else 0.0
    missed = int(np.sum((y == 1) & (pred == 0)))
    fnr = missed / max(1, n_ev)
    
    auc = None
    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, p))
    
    return ConformalMetrics(
        method=f"Prob threshold ({threshold})",
        n_test=n, n_event_test=n_ev,
        event_rate_test=float(n_ev / max(1, n)),
        marginal_cov=acc, event_cov=sens, surv_cov=spec,
        defer_rate=0.0, defer_ambig=0.0, defer_empty=0.0,
        retention_rate=1.0,
        acc_accepted=acc, sens_accepted=sens, spec_accepted=spec,
        fnr_overall=fnr, missed_events=missed, auc_test=auc,
    )


def run_predict_all_survive(y_test, p_test):
    """Trivial baseline: predict survive for everyone."""
    y = np.asarray(y_test, dtype=int)
    n = len(y)
    n_ev = int(y.sum())
    acc = float(np.mean(y == 0))  # accuracy = survival rate
    
    auc = None
    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, np.asarray(p_test, dtype=float)))
    
    return ConformalMetrics(
        method="Predict all survive",
        n_test=n, n_event_test=n_ev,
        event_rate_test=float(n_ev / max(1, n)),
        marginal_cov=float(np.mean(y == 0)),  # only survivors covered
        event_cov=0.0, surv_cov=1.0,
        defer_rate=0.0, defer_ambig=0.0, defer_empty=0.0,
        retention_rate=1.0,
        acc_accepted=acc, sens_accepted=0.0, spec_accepted=1.0,
        fnr_overall=1.0, missed_events=n_ev, auc_test=auc,
    )


def run_random_deferral(y_test, p_test, defer_fraction, rng):
    """
    Baseline: randomly defer a fraction of patients, predict on rest.
    Uses p_event > 0.5 as prediction rule on accepted samples.
    """
    y = np.asarray(y_test, dtype=int)
    p = np.asarray(p_test, dtype=float)
    n = len(y)
    n_ev = int(y.sum())
    
    n_defer = int(round(n * defer_fraction))
    idx = rng.permutation(n)
    deferred = np.zeros(n, dtype=bool)
    deferred[idx[:n_defer]] = True
    accepted = ~deferred
    
    pred = (p >= 0.5).astype(int)
    
    acc_acc = 0.0
    sens_acc = 0.0
    spec_acc = 0.0
    if np.any(accepted):
        acc_acc = float(np.mean(pred[accepted] == y[accepted]))
        ev_acc = accepted & (y == 1)
        if np.any(ev_acc):
            sens_acc = float(np.mean(pred[ev_acc] == 1))
        su_acc = accepted & (y == 0)
        if np.any(su_acc):
            spec_acc = float(np.mean(pred[su_acc] == 0))
    
    # Events predicted as survive (among all, not just accepted)
    missed = int(np.sum((y == 1) & accepted & (pred == 0)))
    fnr = missed / max(1, n_ev)
    
    # Coverage: for accepted, check if prediction is correct
    covered_accepted = (pred[accepted] == y[accepted])
    # For deferred: count as "covered" (they get {0,1})
    marginal_cov = float((np.sum(covered_accepted) + n_defer) / n)
    
    auc = None
    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, p))
    
    return ConformalMetrics(
        method=f"Random deferral ({defer_fraction:.1%})",
        n_test=n, n_event_test=n_ev,
        event_rate_test=float(n_ev / max(1, n)),
        marginal_cov=marginal_cov, event_cov=float('nan'), surv_cov=float('nan'),
        defer_rate=float(defer_fraction),
        defer_ambig=float(defer_fraction), defer_empty=0.0,
        retention_rate=1.0 - defer_fraction,
        acc_accepted=acc_acc, sens_accepted=sens_acc, spec_accepted=spec_acc,
        fnr_overall=fnr, missed_events=missed, auc_test=auc,
    )


# ============================================================================
# Risk-Retention Curve: sweep alpha
# ============================================================================

def risk_retention_sweep(p_cal, y_cal, p_test, y_test, method_func, 
                         alphas=None, **kwargs):
    """
    Sweep alpha values and compute retention rate + risk metrics for each.
    Returns list of (alpha, metrics) tuples.
    """
    if alphas is None:
        alphas = np.arange(0.01, 0.51, 0.01)
    
    results = []
    for a in alphas:
        m = method_func(p_cal, y_cal, p_test, y_test, a, **kwargs)
        results.append((float(a), m))
    return results


# ============================================================================
# Repeated Random Split CI (NOT bootstrap)
# ============================================================================

def single_random_split(p_ext, y_ext, cal_frac, alpha, rng, n_min_shrink=15):
    """
    One random split iteration: split external into cal/test, run all methods.
    Returns dict of method_name -> ConformalMetrics.
    
    NOTE: This is NOT bootstrap (no replacement sampling).
    """
    n = len(y_ext)
    n_cal = int(round(n * cal_frac))
    n_cal = max(5, min(n_cal, n - 5))
    
    idx = rng.permutation(n)
    cal_idx = idx[:n_cal]
    te_idx = idx[n_cal:]
    
    p_cal = p_ext[cal_idx]
    y_cal = y_ext[cal_idx]
    p_test = p_ext[te_idx]
    y_test = y_ext[te_idx]
    
    results = {}
    results["Standard CP"] = run_standard_cp(p_cal, y_cal, p_test, y_test, alpha)
    results["Classwise CP"] = run_classwise_cp(p_cal, y_cal, p_test, y_test, alpha)
    results["Shrinkage CP"] = run_shrinkage_cp(p_cal, y_cal, p_test, y_test, alpha, n_min=n_min_shrink)
    results["Predict all survive"] = run_predict_all_survive(y_test, p_test)
    results["Prob threshold (0.5)"] = run_prob_threshold(p_test, y_test, threshold=0.5)
    results["Random deferral"] = run_random_deferral(
        y_test, p_test, 
        defer_fraction=results["Standard CP"].defer_rate,
        rng=rng
    )
    
    return results


def run_repeated_split(p_ext, y_ext, cal_frac, alpha, B, seed, n_min_shrink=15):
    """Run B repeated random splits, return aggregated results.
    
    Note: This is NOT bootstrap (no replacement sampling).
    This is repeated random split for variance estimation.
    """
    rng = np.random.default_rng(seed)
    
    all_results = {}  # method_name -> list of ConformalMetrics
    
    for b in range(B):
        single = single_random_split(p_ext, y_ext, cal_frac, alpha, rng, n_min_shrink)
        for method_name, metrics in single.items():
            if method_name not in all_results:
                all_results[method_name] = []
            all_results[method_name].append(metrics)
    
    return all_results


def aggregate_repeated_split(all_results, key_metrics=None):
    """Aggregate repeated split results into mean [2.5%, 97.5%] format."""
    if key_metrics is None:
        key_metrics = [
            "marginal_cov", "event_cov", "surv_cov",
            "defer_rate", "retention_rate",
            "acc_accepted", "sens_accepted", "spec_accepted",
            "fnr_overall", "missed_events",
        ]
    
    summary = {}
    for method_name, metrics_list in all_results.items():
        method_summary = {}
        for km in key_metrics:
            vals = []
            for m in metrics_list:
                v = getattr(m, km, None)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
            if vals:
                arr = np.array(vals)
                method_summary[km] = {
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "ci_low": float(np.percentile(arr, 2.5)),
                    "ci_high": float(np.percentile(arr, 97.5)),
                    "std": float(np.std(arr)),
                }
            else:
                method_summary[km] = {"mean": float('nan')}
        summary[method_name] = method_summary
    
    return summary


# ============================================================================
# Risk-Retention Curve with repeated random split
# ============================================================================

def risk_retention_curve_repeated_split(p_ext, y_ext, cal_frac, B, seed, 
                                         n_min_shrink=15, alphas=None):
    """
    For each alpha, run B repeated random splits and compute retention + FNR.
    Returns data suitable for plotting.
    
    Note: This uses repeated random split, NOT bootstrap.
    """
    if alphas is None:
        alphas = np.arange(0.02, 0.52, 0.02)
    
    rng = np.random.default_rng(seed)
    
    # methods to track
    method_names = ["Standard CP", "Classwise CP", "Shrinkage CP"]
    
    # Storage: method -> alpha -> list of (retention, fnr, event_cov, acc_acc)
    curves = {m: {float(a): [] for a in alphas} for m in method_names}
    
    for b in range(B):
        n = len(y_ext)
        n_cal = int(round(n * cal_frac))
        n_cal = max(5, min(n_cal, n - 5))
        
        idx = rng.permutation(n)
        p_cal = p_ext[idx[:n_cal]]
        y_cal = y_ext[idx[:n_cal]]
        p_test = p_ext[idx[n_cal:]]
        y_test = y_ext[idx[n_cal:]]
        
        for a in alphas:
            a_f = float(a)
            
            m_std = run_standard_cp(p_cal, y_cal, p_test, y_test, a_f)
            m_cw = run_classwise_cp(p_cal, y_cal, p_test, y_test, a_f)
            m_sh = run_shrinkage_cp(p_cal, y_cal, p_test, y_test, a_f, n_min=n_min_shrink)
            
            for name, met in [("Standard CP", m_std), ("Classwise CP", m_cw), ("Shrinkage CP", m_sh)]:
                curves[name][a_f].append({
                    "retention": met.retention_rate,
                    "fnr": met.fnr_overall,
                    "event_cov": met.event_cov,
                    "acc_accepted": met.acc_accepted,
                    "defer_rate": met.defer_rate,
                    "marginal_cov": met.marginal_cov,
                })
    
    # Aggregate
    curve_summary = {}
    for method_name in method_names:
        method_curve = []
        for a in alphas:
            a_f = float(a)
            reps = curves[method_name][a_f]
            if not reps:
                continue
            row = {"alpha": a_f}
            for key in ["retention", "fnr", "event_cov", "acc_accepted", "defer_rate", "marginal_cov"]:
                vals = [r[key] for r in reps if not np.isnan(r[key])]
                if vals:
                    arr = np.array(vals)
                    row[f"{key}_mean"] = float(np.mean(arr))
                    row[f"{key}_ci_low"] = float(np.percentile(arr, 2.5))
                    row[f"{key}_ci_high"] = float(np.percentile(arr, 97.5))
                else:
                    row[f"{key}_mean"] = float('nan')
            method_curve.append(row)
        curve_summary[method_name] = method_curve
    
    return curve_summary


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=730.0)
    ap.add_argument("--alpha", type=float, default=0.10, help="Primary alpha for table")
    
    ap.add_argument("--source_oof_csv", type=str, required=True)
    ap.add_argument("--external_pred_csv", type=str, required=True)
    
    ap.add_argument("--source_time_col", type=str, default="time_days")
    ap.add_argument("--source_event_col", type=str, default="event")
    ap.add_argument("--ext_time_col", type=str, default="Survival.time")
    ap.add_argument("--ext_event_col", type=str, default="deadstatus.event")
    
    ap.add_argument("--target_cal_frac", type=float, default=0.50)
    ap.add_argument("--target_repeats", type=int, default=200,
                    help="Repeats for primary table (repeated random splits)")
    ap.add_argument("--n_splits", type=int, default=500,
                    help="Number of repeated random splits for CI (NOT bootstrap)")
    ap.add_argument("--n_min_shrink", type=int, default=15,
                    help="Shrinkage threshold: shrink if n_class < n_min")
    ap.add_argument("--seed", type=int, default=42)
    
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    tau = args.tau
    alpha = args.alpha
    
    # ---- Load data ----
    df_src = pd.read_csv(args.source_oof_csv)
    df_ext = pd.read_csv(args.external_pred_csv)
    
    src_known, src_y = compute_known_and_y(
        df_src, tau, args.source_time_col, args.source_event_col)
    ext_known, ext_y = compute_known_and_y(
        df_ext, tau, args.ext_time_col, args.ext_event_col)
    
    src_p = infer_p_event(df_src.loc[src_known], tau)
    src_y_k = src_y[src_known]
    ext_p = infer_p_event(df_ext.loc[ext_known], tau)
    ext_y_k = ext_y[ext_known]
    
    print("=" * 80)
    print("CLASSWISE CONFORMAL PREDICTION EXPERIMENTS")
    print("=" * 80)
    print(f"tau={tau}, alpha={alpha}, cal_frac={args.target_cal_frac}")
    print(f"External: n={len(ext_y_k)}, n_event={int(ext_y_k.sum())}, "
          f"event_rate={ext_y_k.mean():.3f}")
    print(f"External p_event: min={ext_p.min():.4f}, max={ext_p.max():.4f}, "
          f"mean={ext_p.mean():.4f}")
    print(f"Source: n={len(src_y_k)}, n_event={int(src_y_k.sum())}, "
          f"event_rate={src_y_k.mean():.3f}")
    print("=" * 80)
    
    # ---- Part 1: Frozen comparison (source q applied to all external) ----
    print("\n[Part 1] Frozen deployment comparison (q from source, eval on all external)")
    
    # Standard CP frozen
    scores_src = nonconformity_scores(src_p, src_y_k)
    q_frozen = conformal_quantile(scores_src, alpha)
    inc0_f, inc1_f = pred_sets_standard(ext_p, q_frozen)
    m_frozen_std = compute_conformal_metrics(
        ext_y_k, ext_p, inc0_f, inc1_f, "Frozen Standard CP", q_marginal=q_frozen)
    
    # Classwise frozen (source class-specific quantiles)
    scores_src_0 = src_p[src_y_k == 0]
    scores_src_1 = 1.0 - src_p[src_y_k == 1]
    q0_frozen = conformal_quantile(scores_src_0, alpha)
    q1_frozen = conformal_quantile(scores_src_1, alpha)
    inc0_cf, inc1_cf = pred_sets_classwise(ext_p, q0_frozen, q1_frozen)
    m_frozen_cw = compute_conformal_metrics(
        ext_y_k, ext_p, inc0_cf, inc1_cf, "Frozen Classwise CP",
        q0=q0_frozen, q1=q1_frozen)
    
    # Baselines
    m_all_surv = run_predict_all_survive(ext_y_k, ext_p)
    m_prob_05 = run_prob_threshold(ext_p, ext_y_k, threshold=0.5)
    
    frozen_table = [asdict(m) for m in [m_frozen_std, m_frozen_cw, m_all_surv, m_prob_05]]
    df_frozen = pd.DataFrame(frozen_table)
    frozen_path = os.path.join(args.out_dir, "frozen_comparison.csv")
    df_frozen.to_csv(frozen_path, index=False)
    print(f"Saved frozen comparison -> {frozen_path}")
    
    for m in [m_frozen_std, m_frozen_cw, m_all_surv, m_prob_05]:
        print(f"  {m.method}: marg_cov={m.marginal_cov:.3f}, event_cov={m.event_cov:.3f}, "
              f"surv_cov={m.surv_cov:.3f}, defer={m.defer_rate:.3f}, "
              f"acc_acc={m.acc_accepted:.3f}, fnr={m.fnr_overall:.3f}")
    
    # ---- Part 2: Target-local with repeated split CI ----
    print(f"\n[Part 2] Target-local CP with repeated random split (n_splits={args.n_splits})")
    
    all_boot = run_repeated_split(
        ext_p, ext_y_k,
        cal_frac=args.target_cal_frac,
        alpha=alpha,
        B=args.n_splits,
        seed=args.seed,
        n_min_shrink=args.n_min_shrink,
    )
    
    summary = aggregate_repeated_split(all_boot)
    
    # Print summary table
    print(f"\n{'Method':<30} {'Marg_Cov':>12} {'Event_Cov':>12} {'Surv_Cov':>12} "
          f"{'Defer%':>10} {'Acc@Acc':>12} {'FNR':>10} {'Missed':>8}")
    print("-" * 120)
    
    table_rows = []
    for method_name, ms in summary.items():
        def fmt(key):
            d = ms.get(key, {})
            m = d.get("mean", float('nan'))
            lo = d.get("ci_low", float('nan'))
            hi = d.get("ci_high", float('nan'))
            if np.isnan(m):
                return "N/A"
            return f"{m:.3f} [{lo:.3f},{hi:.3f}]"
        
        def fmtv(key):
            d = ms.get(key, {})
            return d.get("mean", float('nan'))
        
        print(f"{method_name:<30} {fmt('marginal_cov'):>12} {fmt('event_cov'):>12} "
              f"{fmt('surv_cov'):>12} {fmt('defer_rate'):>10} "
              f"{fmt('acc_accepted'):>12} {fmt('fnr_overall'):>10} "
              f"{fmt('missed_events'):>8}")
        
        table_rows.append({
            "method": method_name,
            **{f"{k}_mean": ms[k]["mean"] for k in ms if "mean" in ms[k]},
            **{f"{k}_ci_low": ms[k].get("ci_low", float('nan')) for k in ms if "ci_low" in ms.get(k, {})},
            **{f"{k}_ci_high": ms[k].get("ci_high", float('nan')) for k in ms if "ci_high" in ms.get(k, {})},
        })
    
    summary_path = os.path.join(args.out_dir, "repeated_split_summary.csv")
    pd.DataFrame(table_rows).to_csv(summary_path, index=False)
    print(f"\nSaved repeated split summary -> {summary_path}")
    
    # ---- Part 3: Risk-Retention Curves ----
    print(f"\n[Part 3] Risk-Retention Curves (sweep alpha, n_splits={min(args.n_splits, 200)})")
    
    curve_summary = risk_retention_curve_repeated_split(
        ext_p, ext_y_k,
        cal_frac=args.target_cal_frac,
        B=min(args.n_splits, 200),  # cap for speed
        seed=args.seed + 1000,
        n_min_shrink=args.n_min_shrink,
    )
    
    # Save curves as CSV for plotting
    for method_name, curve_data in curve_summary.items():
        safe_name = method_name.replace(" ", "_").replace("(", "").replace(")", "")
        curve_path = os.path.join(args.out_dir, f"risk_retention_{safe_name}.csv")
        pd.DataFrame(curve_data).to_csv(curve_path, index=False)
        print(f"  Saved {method_name} curve -> {curve_path}")
    
    # ---- Part 4: Save all details as JSON ----
    all_details = {
        "config": {
            "tau": tau,
            "alpha": alpha,
            "cal_frac": args.target_cal_frac,
            "n_splits": args.n_splits,
            "n_min_shrink": args.n_min_shrink,
            "seed": args.seed,
        },
        "data_summary": {
            "external_n": int(len(ext_y_k)),
            "external_n_event": int(ext_y_k.sum()),
            "external_event_rate": float(ext_y_k.mean()),
            "external_p_event_min": float(ext_p.min()),
            "external_p_event_max": float(ext_p.max()),
            "external_p_event_mean": float(ext_p.mean()),
            "source_n": int(len(src_y_k)),
            "source_n_event": int(src_y_k.sum()),
            "source_event_rate": float(src_y_k.mean()),
        },
        "frozen_results": {m.method: asdict(m) for m in [m_frozen_std, m_frozen_cw, m_all_surv, m_prob_05]},
        "repeated_split_summary": {k: v for k, v in summary.items()},
    }
    
    json_path = os.path.join(args.out_dir, "all_details.json")
    with open(json_path, "w") as f:
        json.dump(all_details, f, indent=2, default=str)
    print(f"\nSaved all details -> {json_path}")
    
    # ---- Summary ----
    print("\n" + "=" * 80)
    print("CRITICAL FINDINGS SUMMARY")
    print("=" * 80)
    
    print(f"\n1. FROZEN DEPLOYMENT (source q on all external):")
    print(f"   Standard CP: marg_cov={m_frozen_std.marginal_cov:.1%}, "
          f"event_cov={m_frozen_std.event_cov:.1%}, surv_cov={m_frozen_std.surv_cov:.1%}")
    print(f"   → Model predicts nearly ALL patients as high-risk (p_event > {q_frozen:.3f})")
    print(f"   → Coverage collapse for survivors, NOT for events")
    
    print(f"\n2. TARGET-LOCAL (with target calibration labels):")
    std_s = summary.get("Standard CP", {})
    cw_s = summary.get("Classwise CP", {})
    sh_s = summary.get("Shrinkage CP", {})
    
    def get_mean(s, k):
        return s.get(k, {}).get("mean", float('nan'))
    
    print(f"   Standard CP: defer={get_mean(std_s, 'defer_rate'):.1%}, "
          f"event_cov={get_mean(std_s, 'event_cov'):.1%}, fnr={get_mean(std_s, 'fnr_overall'):.1%}")
    print(f"   Classwise CP: defer={get_mean(cw_s, 'defer_rate'):.1%}, "
          f"event_cov={get_mean(cw_s, 'event_cov'):.1%}, fnr={get_mean(cw_s, 'fnr_overall'):.1%}")
    print(f"   Shrinkage CP: defer={get_mean(sh_s, 'defer_rate'):.1%}, "
          f"event_cov={get_mean(sh_s, 'event_cov'):.1%}, fnr={get_mean(sh_s, 'fnr_overall'):.1%}")
    
    print(f"\n3. TRIVIAL BASELINES:")
    print(f"   Predict all survive: acc={m_all_surv.acc_accepted:.1%}, fnr=100%")
    print(f"   Prob threshold 0.5: acc={m_prob_05.acc_accepted:.1%}, fnr={m_prob_05.fnr_overall:.1%}")
    
    print("\n" + "=" * 80)
    print("DONE. Next: examine results and decide paper narrative.")
    print("=" * 80)


if __name__ == "__main__":
    main()