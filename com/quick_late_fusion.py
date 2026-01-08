#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Late Fusion using existing Phase3 OOF predictions
========================================================

This script does NOT require retraining - it uses your existing phase3 outputs.

Strategy:
1. Aggregate 5-fold OOF predictions from phase3
2. Match with zip's test set
3. Late fusion with GBSA

Usage:
    python quick_late_fusion.py \
        --gbsa_preds /path/to/gbsa_test_preds.csv \
        --phase3_dir /path/to/phase3_outputs/exp_name \
        --out_dir /path/to/fusion_results
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def harrell_c(event, time, risk) -> float:
    """Harrell's C-index for right-censored survival."""
    event = np.asarray(event).astype(bool)
    time = np.asarray(time).astype(float)
    risk = np.asarray(risk).astype(float)
    n = len(time)
    conc = 0.0
    ties = 0.0
    perm = 0.0
    for i in range(n):
        if not event[i]:
            continue
        for j in range(n):
            if time[j] <= time[i]:
                continue
            perm += 1
            if risk[i] > risk[j]:
                conc += 1
            elif risk[i] == risk[j]:
                ties += 0.5
    return float((conc + 0.5 * ties) / perm) if perm > 0 else np.nan

def bootstrap_c(event, time, risk, n_boot=200, seed=42):
    """Bootstrap confidence interval for C-index."""
    rng = np.random.default_rng(seed)
    n = len(time)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        c = harrell_c(event[idx], time[idx], risk[idx])
        if np.isfinite(c):
            vals.append(c)
    if len(vals) == 0:
        return np.nan, np.nan, (np.nan, np.nan)
    vals = np.array(vals)
    return vals.mean(), vals.std(ddof=1), (np.percentile(vals, 2.5), np.percentile(vals, 97.5))

def aggregate_oof_predictions(phase3_dir: Path) -> pd.DataFrame:
    """Aggregate OOF predictions from all folds."""
    all_preds = []
    
    for fold_dir in sorted(phase3_dir.glob("fold_*")):
        val_pred_path = fold_dir / "val_pred.csv"
        if val_pred_path.exists():
            df = pd.read_csv(val_pred_path)
            df["fold"] = int(fold_dir.name.split("_")[1])
            all_preds.append(df)
            print(f"Loaded {val_pred_path}: {len(df)} samples")
    
    if not all_preds:
        raise ValueError(f"No val_pred.csv found in {phase3_dir}")
    
    combined = pd.concat(all_preds, ignore_index=True)
    
    # Check for duplicates (should not happen in proper CV)
    dup = combined.duplicated(subset=["patient_id"], keep=False)
    if dup.any():
        print(f"WARNING: {dup.sum()} duplicate patient_ids found, keeping first")
        combined = combined.drop_duplicates(subset=["patient_id"], keep="first")
    
    return combined

def run_fusion(
    gbsa_preds_path: str,
    phase3_dir: str,
    out_dir: str,
    risk_col: str = "risk",  # or "risk_integral" depending on phase3 output
):
    """Run late fusion pipeline."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load GBSA predictions
    df_gbsa = pd.read_csv(gbsa_preds_path)
    
    # Normalize column names
    if "pid" in df_gbsa.columns:
        df_gbsa = df_gbsa.rename(columns={"pid": "patient_id"})
    if "pred_risk" in df_gbsa.columns:
        df_gbsa = df_gbsa.rename(columns={"pred_risk": "risk_gbsa"})
    if "deadstatus.event" in df_gbsa.columns:
        df_gbsa = df_gbsa.rename(columns={"deadstatus.event": "event"})
    
    df_gbsa["patient_id"] = df_gbsa["patient_id"].astype(str)
    gbsa_test_ids = set(df_gbsa["patient_id"].tolist())
    
    print(f"GBSA test set: {len(gbsa_test_ids)} samples")
    
    # Load phase3 OOF predictions
    phase3_dir = Path(phase3_dir)
    df_cnn = aggregate_oof_predictions(phase3_dir)
    df_cnn["patient_id"] = df_cnn["patient_id"].astype(str)
    
    # Find risk column
    risk_candidates = ["risk", "risk_integral", "pred_risk", "risk_int"]
    for col in risk_candidates:
        if col in df_cnn.columns:
            risk_col = col
            break
    
    print(f"CNN OOF predictions: {len(df_cnn)} samples, using '{risk_col}' as risk")
    
    # Find overlap with GBSA test set
    cnn_ids = set(df_cnn["patient_id"].tolist())
    overlap = gbsa_test_ids & cnn_ids
    
    print(f"Overlap with GBSA test set: {len(overlap)} / {len(gbsa_test_ids)}")
    
    if len(overlap) < len(gbsa_test_ids):
        missing = gbsa_test_ids - cnn_ids
        print(f"WARNING: {len(missing)} test samples not in phase3 OOF!")
        print(f"Missing IDs (first 10): {list(missing)[:10]}")
    
    # Filter to overlap
    df_gbsa_overlap = df_gbsa[df_gbsa["patient_id"].isin(overlap)].copy()
    df_cnn_overlap = df_cnn[df_cnn["patient_id"].isin(overlap)].copy()
    
    # Merge
    df_cnn_overlap = df_cnn_overlap.rename(columns={risk_col: "risk_cnn"})
    merged = df_gbsa_overlap.merge(
        df_cnn_overlap[["patient_id", "risk_cnn"]], 
        on="patient_id", 
        how="inner"
    )
    
    print(f"\nMerged: {len(merged)} samples")
    
    # Extract arrays
    event = merged["event"].values.astype(bool)
    time = merged["time"].values.astype(float)
    risk_gbsa = merged["risk_gbsa"].values
    risk_cnn = merged["risk_cnn"].values
    
    # Standardize risks
    scaler_gbsa = StandardScaler()
    scaler_cnn = StandardScaler()
    risk_gbsa_std = scaler_gbsa.fit_transform(risk_gbsa.reshape(-1, 1)).flatten()
    risk_cnn_std = scaler_cnn.fit_transform(risk_cnn.reshape(-1, 1)).flatten()
    
    # Baseline C-index
    c_gbsa = harrell_c(event, time, risk_gbsa)
    c_cnn = harrell_c(event, time, risk_cnn)
    
    boot_gbsa = bootstrap_c(event, time, risk_gbsa)
    boot_cnn = bootstrap_c(event, time, risk_cnn)
    
    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"GBSA alone:  C-index = {c_gbsa:.4f} ± {boot_gbsa[1]:.4f} (95% CI: [{boot_gbsa[2][0]:.4f}, {boot_gbsa[2][1]:.4f}])")
    print(f"CNN alone:   C-index = {c_cnn:.4f} ± {boot_cnn[1]:.4f} (95% CI: [{boot_cnn[2][0]:.4f}, {boot_cnn[2][1]:.4f}])")
    
    # Grid search for optimal alpha
    print(f"\n{'='*60}")
    print("LATE FUSION GRID SEARCH")
    print(f"{'='*60}")
    print(f"Formula: risk_fused = α × risk_GBSA + (1-α) × risk_CNN")
    print(f"\n{'Alpha':<10} {'C-index':<12} {'vs GBSA':<12} {'vs CNN':<12}")
    print("-" * 50)
    
    results = []
    for alpha in np.arange(0.0, 1.01, 0.05):
        risk_fused = alpha * risk_gbsa_std + (1 - alpha) * risk_cnn_std
        c = harrell_c(event, time, risk_fused)
        diff_gbsa = c - c_gbsa
        diff_cnn = c - c_cnn
        results.append({
            "alpha": float(alpha),
            "c_index": float(c),
            "diff_gbsa": float(diff_gbsa),
            "diff_cnn": float(diff_cnn),
        })
        marker = "***" if c > c_gbsa else ""
        print(f"{alpha:<10.2f} {c:<12.4f} {diff_gbsa:+.4f}      {diff_cnn:+.4f}      {marker}")
    
    results_df = pd.DataFrame(results)
    best_idx = results_df["c_index"].idxmax()
    best = results_df.loc[best_idx]
    
    boot_fused = bootstrap_c(
        event, time, 
        best["alpha"] * risk_gbsa_std + (1 - best["alpha"]) * risk_cnn_std
    )
    
    print(f"\n{'='*60}")
    print("BEST FUSION RESULT")
    print(f"{'='*60}")
    print(f"Best α = {best['alpha']:.2f}")
    print(f"C-index = {best['c_index']:.4f} ± {boot_fused[1]:.4f} (95% CI: [{boot_fused[2][0]:.4f}, {boot_fused[2][1]:.4f}])")
    print(f"Improvement over GBSA: {best['diff_gbsa']:+.4f}")
    print(f"Improvement over CNN:  {best['diff_cnn']:+.4f}")
    
    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    
    if best["alpha"] == 1.0:
        print("⚠️  Best fusion uses α=1.0 (GBSA only)")
        print("   → CNN does not add useful information beyond GBSA")
        print("   → Stick with GBSA baseline")
    elif best["alpha"] == 0.0:
        print("⚠️  Best fusion uses α=0.0 (CNN only)")
        print("   → GBSA does not add useful information beyond CNN")
        print("   → This is unusual; check for data leakage")
    elif best["diff_gbsa"] > 0.005:
        print("✅ Fusion improves over GBSA baseline!")
        print(f"   → Improvement: {best['diff_gbsa']:+.4f}")
        print(f"   → Recommended α: {best['alpha']:.2f}")
    else:
        print("➖ Fusion provides marginal or no improvement")
        print("   → Improvement is within noise range")
        print("   → Consider sticking with GBSA for simplicity")
    
    # Save results
    results_df.to_csv(out_dir / "fusion_grid_search.csv", index=False)
    
    # Save fused predictions at best alpha
    merged["risk_fused"] = best["alpha"] * risk_gbsa_std + (1 - best["alpha"]) * risk_cnn_std
    merged["risk_gbsa_std"] = risk_gbsa_std
    merged["risk_cnn_std"] = risk_cnn_std
    merged.to_csv(out_dir / "fused_predictions.csv", index=False)
    
    summary = {
        "n_samples": int(len(merged)),
        "gbsa_cindex": float(c_gbsa),
        "gbsa_std": float(boot_gbsa[1]),
        "gbsa_ci95": [float(x) for x in boot_gbsa[2]],
        "cnn_cindex": float(c_cnn),
        "cnn_std": float(boot_cnn[1]),
        "cnn_ci95": [float(x) for x in boot_cnn[2]],
        "best_alpha": float(best["alpha"]),
        "best_cindex": float(best["c_index"]),
        "best_std": float(boot_fused[1]),
        "best_ci95": [float(x) for x in boot_fused[2]],
        "improvement_over_gbsa": float(best["diff_gbsa"]),
        "improvement_over_cnn": float(best["diff_cnn"]),
    }
    
    with open(out_dir / "fusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {out_dir}")
    
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gbsa_preds", required=True, help="GBSA test predictions CSV")
    parser.add_argument("--phase3_dir", required=True, help="Phase3 output directory (containing fold_* subdirs)")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--risk_col", default="risk", help="Risk column name in phase3 outputs")
    
    args = parser.parse_args()
    
    run_fusion(
        gbsa_preds_path=args.gbsa_preds,
        phase3_dir=args.phase3_dir,
        out_dir=args.out_dir,
        risk_col=args.risk_col,
    )

if __name__ == "__main__":
    main()