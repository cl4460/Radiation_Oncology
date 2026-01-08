#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
safe_fusion.py

Safe late-fusion between:
  - baseline risk predictions (e.g., zip-provided Clinic+Radiomics GBSA seed 91)
  - deep-model risk predictions (from deep embeddings or CNN)

Goal:
  - be fast
  - minimize overfitting
  - never *intentionally* harm the baseline: if fusion doesn't improve on inner validation, fall back to baseline.

Inputs must contain:
  pid, time, event, pred_risk
(train and test provided separately)

Outputs:
  out_dir/
    fusion_summary.json
    fused_test_preds.csv
"""

import os
import json
import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored
except Exception as e:
    raise ImportError(
        "This script requires scikit-survival (sksurv). "
        "Install: `pip install scikit-survival` following your OS instructions."
    ) from e


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def cindex(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> float:
    y = Surv.from_arrays(event=event.astype(bool), time=time.astype(float))
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


def read_preds(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    # normalize columns
    colmap = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ["patientid", "patient_id", "pid", "id"]:
            colmap[c] = "pid"
        elif lc in ["time", "survival_time", "survivaltime", "os_time", "survival.time"]:
            colmap[c] = "time"
        elif lc in ["event", "status", "death", "os_event", "deadstatus.event"]:
            colmap[c] = "event"
        elif lc in ["pred_risk", "risk", "prediction", "pred"]:
            colmap[c] = "pred_risk"
    df = df.rename(columns=colmap)
    required = ["pid", "time", "event", "pred_risk"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"{path} missing required column '{r}'. Got columns={list(df.columns)}")
    df = df[required].copy()
    df["pid"] = df["pid"].astype(str)
    df["time"] = pd.to_numeric(df["time"], errors="coerce").astype(float)
    df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0).astype(int)
    df["pred_risk"] = pd.to_numeric(df["pred_risk"], errors="coerce").astype(float)
    df = df.dropna(subset=["time", "pred_risk"])
    return df


def zscore_fit(x: np.ndarray) -> Tuple[float, float]:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-12:
        sd = 1.0
    return mu, sd


def zscore_apply(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (x - mu) / sd


def rankdata(x: np.ndarray) -> np.ndarray:
    # average ranks for ties
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # handle ties
    x_sorted = x[order]
    i = 0
    while i < len(x_sorted):
        j = i
        while j + 1 < len(x_sorted) and x_sorted[j + 1] == x_sorted[i]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j)
            ranks[order[i:j+1]] = avg
        i = j + 1
    # normalize to [0,1]
    if len(x) > 1:
        ranks = ranks / (len(x) - 1.0)
    return ranks


def orient_risk_by_train(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> Tuple[np.ndarray, bool, float]:
    ci_pos = cindex(event, time, risk)
    ci_neg = cindex(event, time, -risk)
    if ci_neg > ci_pos:
        return -risk, True, float(ci_neg)
    return risk, False, float(ci_pos)


def bootstrap_cindex(event: np.ndarray, time: np.ndarray, risk: np.ndarray, n_boot: int, seed: int) -> Dict:
    rng = np.random.default_rng(int(seed))
    n = len(risk)
    vals = []
    idx = np.arange(n)
    for _ in range(int(n_boot)):
        samp = rng.choice(idx, size=n, replace=True)
        vals.append(cindex(event[samp], time[samp], risk[samp]))
    vals = np.array(vals, dtype=float)
    return {
        "boot_mean": float(np.mean(vals)),
        "boot_std": float(np.std(vals)),
        "ci95_pct": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_train", required=True, help="Baseline train preds (.xlsx or .csv)")
    ap.add_argument("--base_test", required=True, help="Baseline test preds (.xlsx or .csv)")
    ap.add_argument("--deep_train", required=True, help="Deep train preds (.csv or .xlsx). You can pass comma-separated list for ensembling.")
    ap.add_argument("--deep_test", required=True, help="Deep test preds (.csv or .xlsx). Must match deep_train list order/count.")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--grid_step", type=float, default=0.01)
    ap.add_argument("--space", choices=["zscore", "rank"], default="zscore")
    ap.add_argument("--safety_eps", type=float, default=0.0, help="Require fusion_val >= base_val + eps to accept fusion, else fallback to baseline.")
    ap.add_argument("--bootstrap", type=int, default=200)

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # Load baseline
    base_tr = read_preds(args.base_train)
    base_te = read_preds(args.base_test)

    # Load deep (possibly ensemble)
    deep_train_files = [p.strip() for p in args.deep_train.split(",") if p.strip()]
    deep_test_files = [p.strip() for p in args.deep_test.split(",") if p.strip()]
    if len(deep_train_files) != len(deep_test_files):
        raise ValueError("deep_train and deep_test must have the same number of files (comma-separated).")

    deep_tr_list = [read_preds(p) for p in deep_train_files]
    deep_te_list = [read_preds(p) for p in deep_test_files]

    # Align baseline train/test
    def merge_two(a, b, suffix):
        b2 = b.rename(columns={"pred_risk": f"pred_risk{suffix}"})
        return a.merge(b2[["pid", f"pred_risk{suffix}"]], on="pid", how="inner")

    # train merge
    df_tr = base_tr.rename(columns={"pred_risk": "base_risk"}).copy()
    for j, dtr in enumerate(deep_tr_list):
        df_tr = merge_two(df_tr, dtr, suffix=f"_deep{j}")
    # test merge
    df_te = base_te.rename(columns={"pred_risk": "base_risk"}).copy()
    for j, dte in enumerate(deep_te_list):
        df_te = merge_two(df_te, dte, suffix=f"_deep{j}")

    # Sanity check: train and test should NOT overlap
    overlap = set(df_tr["pid"]).intersection(set(df_te["pid"]))
    if len(overlap) > 0:
        print(f"Warning: {len(overlap)} patients appear in both train and test!")

    if len(df_te) < 20:
        raise RuntimeError(f"Too few test samples after alignment: {len(df_te)}. Check pid formats.")

    # Prepare arrays
    e_tr = df_tr["event"].values.astype(int)
    t_tr = df_tr["time"].values.astype(float)
    e_te = df_te["event"].values.astype(int)
    t_te = df_te["time"].values.astype(float)

    base_r_tr = df_tr["base_risk"].values.astype(float)
    base_r_te = df_te["base_risk"].values.astype(float)

    # orient baseline
    base_r_tr, base_flip, base_ci_train = orient_risk_by_train(e_tr, t_tr, base_r_tr)
    base_r_te = -base_r_te if base_flip else base_r_te

    # build deep ensemble risk (standardize each deep stream on train before averaging)
    deep_r_tr_all = []
    deep_r_te_all = []
    deep_info = []
    for j in range(len(deep_train_files)):
        rtr = df_tr[f"pred_risk_deep{j}"].values.astype(float)
        rte = df_te[f"pred_risk_deep{j}"].values.astype(float)

        rtr, flipped, ci_train = orient_risk_by_train(e_tr, t_tr, rtr)
        rte = -rte if flipped else rte

        if args.space == "rank":
            rtr_s = rankdata(rtr)
            rte_s = rankdata(rte)
            mu, sd = 0.0, 1.0
        else:
            mu, sd = zscore_fit(rtr)
            rtr_s = zscore_apply(rtr, mu, sd)
            rte_s = zscore_apply(rte, mu, sd)

        deep_r_tr_all.append(rtr_s)
        deep_r_te_all.append(rte_s)
        deep_info.append({
            "file_train": os.path.abspath(deep_train_files[j]),
            "file_test": os.path.abspath(deep_test_files[j]),
            "flipped_sign": bool(flipped),
            "train_cindex": float(ci_train),
            "z_mu": float(mu),
            "z_sd": float(sd),
        })

    deep_r_tr = np.mean(np.vstack(deep_r_tr_all), axis=0)
    deep_r_te = np.mean(np.vstack(deep_r_te_all), axis=0)

    # Standardize baseline too (in same "space")
    if args.space == "rank":
        base_s_tr = rankdata(base_r_tr)
        base_s_te = rankdata(base_r_te)
    else:
        mu_b, sd_b = zscore_fit(base_r_tr)
        base_s_tr = zscore_apply(base_r_tr, mu_b, sd_b)
        base_s_te = zscore_apply(base_r_te, mu_b, sd_b)

    # inner validation split on train
    sss = StratifiedShuffleSplit(n_splits=1, test_size=float(args.val_ratio), random_state=int(args.seed))
    idx = np.arange(len(df_tr))
    tr_idx, va_idx = next(sss.split(idx, e_tr))

    base_val_ci = cindex(e_tr[va_idx], t_tr[va_idx], base_s_tr[va_idx])

    # grid search weight
    step = float(args.grid_step)
    grid = np.arange(0.0, 1.0 + 1e-12, step)
    best = {"w": 1.0, "val_ci": base_val_ci}
    for w in grid:
        fused = w * base_s_tr[va_idx] + (1.0 - w) * deep_r_tr[va_idx]
        ci = cindex(e_tr[va_idx], t_tr[va_idx], fused)
        if ci > best["val_ci"]:
            best = {"w": float(w), "val_ci": float(ci)}

    # safety fallback
    if best["val_ci"] < base_val_ci + float(args.safety_eps):
        w_final = 1.0
        accepted = False
    else:
        w_final = best["w"]
        accepted = True

    fused_test = w_final * base_s_te + (1.0 - w_final) * deep_r_te
    fused_test_ci = cindex(e_te, t_te, fused_test)

    out = {
        "n_train": int(len(df_tr)),
        "n_test": int(len(df_te)),
        "space": args.space,
        "val_ratio": float(args.val_ratio),
        "base_train_cindex": float(base_ci_train),
        "base_val_cindex": float(base_val_ci),
        "deep_ensemble_train_cindex": float(cindex(e_tr, t_tr, deep_r_tr)),
        "best_w_on_val": float(best["w"]),
        "best_val_cindex": float(best["val_ci"]),
        "accepted_fusion": bool(accepted),
        "w_final": float(w_final),
        "test_cindex_fused": float(fused_test_ci),
        "deep_streams": deep_info,
    }

    if int(args.bootstrap) > 0:
        out.update(bootstrap_cindex(e_te, t_te, fused_test, n_boot=int(args.bootstrap), seed=int(args.seed)))

    # save fused test preds
    fused_df = df_te[["pid", "time", "event"]].copy()
    fused_df["base_risk_used"] = base_s_te
    fused_df["deep_risk_used"] = deep_r_te
    fused_df["fused_risk"] = fused_test
    fused_df.to_csv(os.path.join(args.out_dir, "fused_test_preds.csv"), index=False)

    with open(os.path.join(args.out_dir, "fusion_summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=== Safe Fusion Summary ===")
    print(f"base_val_cindex={base_val_ci:.6f}")
    print(f"best_val_cindex={best['val_ci']:.6f} at w={best['w']:.2f}  accepted={accepted}  w_final={w_final:.2f}")
    print(f"test_cindex_fused={fused_test_ci:.6f}")
    if int(args.bootstrap) > 0:
        print(f"boot_mean±std={out['boot_mean']:.6f}±{out['boot_std']:.6f}  CI95={out['ci95_pct']}")
    print(f"Saved: {args.out_dir}/fusion_summary.json")
    print(f"Saved: {args.out_dir}/fused_test_preds.csv")


if __name__ == "__main__":
    main()
