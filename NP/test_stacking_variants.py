#!/usr/bin/env python3
"""
test_stacking_variants.py
=========================
测试不同模态组合的stacking效果，找出最佳组合
"""

import argparse
import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score
from sksurv.util import Surv


def find_risk_column(df):
    for c in ["risk_pred_raw", "risk", "risk_pred", "pred"]:
        if c in df.columns:
            return c
    raise ValueError(f"No risk column found. Available: {list(df.columns)}")


def find_id_column(df):
    for c in ["PatientID", "patient_id", "pid"]:
        if c in df.columns:
            return c
    raise ValueError(f"No ID column found")


def zscore_fit(x):
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def zscore_apply(x, mu, sd):
    return (x - mu) / sd


def evaluate_combination(feat_df, y_df, splits, tags_to_use, tau=730.0):
    """Evaluate a specific combination of modalities"""
    fold_keys = sorted([k for k in splits if k.startswith("fold_")])
    
    results = []
    for fk in fold_keys:
        tr_ids = [str(x).strip() for x in splits[fk]["train"]]
        va_ids = [str(x).strip() for x in splits[fk]["val"]]
        
        merged = feat_df.merge(y_df, on="patient_id", how="inner")
        tr = merged[merged["patient_id"].isin(tr_ids)].reset_index(drop=True)
        va = merged[merged["patient_id"].isin(va_ids)].reset_index(drop=True)
        
        y_tr = Surv.from_arrays(tr["event"].astype(bool), tr["time"].astype(float))
        y_va = Surv.from_arrays(va["event"].astype(bool), va["time"].astype(float))
        
        X_tr = tr[list(tags_to_use)].to_numpy(dtype=float)
        X_va = va[list(tags_to_use)].to_numpy(dtype=float)
        
        # Z-score
        mu, sd = zscore_fit(X_tr)
        X_tr = zscore_apply(X_tr, mu, sd)
        X_va = zscore_apply(X_va, mu, sd)
        
        # Auto-sign
        for j in range(X_tr.shape[1]):
            c0 = concordance_index_censored(y_tr["event"], y_tr["time"], X_tr[:, j])[0]
            c1 = concordance_index_censored(y_tr["event"], y_tr["time"], -X_tr[:, j])[0]
            if c1 > c0 + 1e-6:
                X_tr[:, j] *= -1.0
                X_va[:, j] *= -1.0
        
        # Simple average fusion (no learning, just weighted average)
        risk_avg = np.mean(X_va, axis=1)
        
        # CoxPH stacking with fixed alpha
        try:
            model = CoxPHSurvivalAnalysis(alpha=0.01)
            model.fit(X_tr, y_tr)
            risk_cox = model.predict(X_va)
        except:
            risk_cox = risk_avg
        
        # Evaluate
        tau_eff = min(tau, float(np.max(y_tr["time"])) - 1, float(np.max(y_va["time"])) - 1)
        
        har_avg = concordance_index_censored(y_va["event"], y_va["time"], risk_avg)[0]
        har_cox = concordance_index_censored(y_va["event"], y_va["time"], risk_cox)[0]
        
        try:
            uno_avg = concordance_index_ipcw(y_tr, y_va, risk_avg, tau=tau_eff)[0]
            uno_cox = concordance_index_ipcw(y_tr, y_va, risk_cox, tau=tau_eff)[0]
        except:
            uno_avg = har_avg
            uno_cox = har_cox
        
        results.append({
            "fold": fk,
            "har_avg": har_avg,
            "har_cox": har_cox,
            "uno_avg": uno_avg,
            "uno_cox": uno_cox,
        })
    
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--splits_json", required=True)
    ap.add_argument("--oof", action="append", required=True, help="TAG:path")
    ap.add_argument("--tau", default=730.0, type=float)
    args = ap.parse_args()

    # Load labels
    lab = pd.read_csv(args.labels_csv)
    y_df = pd.DataFrame({
        "patient_id": lab["PatientID"].astype(str).str.strip(),
        "time": pd.to_numeric(lab["Survival.time"], errors="coerce"),
        "event": pd.to_numeric(lab["deadstatus.event"], errors="coerce").fillna(0).astype(int),
    })

    # Load splits
    with open(args.splits_json) as f:
        splits = json.load(f)

    # Load each OOF
    feat_df = pd.DataFrame({"patient_id": y_df["patient_id"]})
    all_tags = []
    
    for oof_arg in args.oof:
        tag, path = oof_arg.split(":", 1)
        df = pd.read_csv(path)
        pid_col = find_id_column(df)
        risk_col = find_risk_column(df)
        
        tmp = pd.DataFrame({
            "patient_id": df[pid_col].astype(str).str.strip(),
            tag: pd.to_numeric(df[risk_col], errors="coerce"),
        })
        feat_df = feat_df.merge(tmp, on="patient_id", how="left")
        all_tags.append(tag)
        print(f"Loaded {tag}: {len(tmp)} rows")

    print(f"\nAll modalities: {all_tags}")
    print("="*80)

    # Test all combinations
    all_results = []
    
    # Single modalities
    for tag in all_tags:
        results = evaluate_combination(feat_df, y_df, splits, [tag], args.tau)
        mean_har = np.mean([r["har_avg"] for r in results])
        mean_uno = np.mean([r["uno_avg"] for r in results])
        std_har = np.std([r["har_avg"] for r in results])
        all_results.append({
            "combination": tag,
            "n_modalities": 1,
            "har_avg_mean": mean_har,
            "har_avg_std": std_har,
            "uno_avg_mean": mean_uno,
            "har_cox_mean": mean_har,  # same for single
            "uno_cox_mean": mean_uno,
        })
        print(f"[{tag}] Harrell={mean_har:.4f}±{std_har:.4f}")

    # Pairs
    for combo in combinations(all_tags, 2):
        results = evaluate_combination(feat_df, y_df, splits, combo, args.tau)
        mean_har_avg = np.mean([r["har_avg"] for r in results])
        mean_har_cox = np.mean([r["har_cox"] for r in results])
        mean_uno_avg = np.mean([r["uno_avg"] for r in results])
        mean_uno_cox = np.mean([r["uno_cox"] for r in results])
        std_har = np.std([r["har_cox"] for r in results])
        
        combo_name = "+".join(combo)
        all_results.append({
            "combination": combo_name,
            "n_modalities": 2,
            "har_avg_mean": mean_har_avg,
            "har_avg_std": std_har,
            "uno_avg_mean": mean_uno_avg,
            "har_cox_mean": mean_har_cox,
            "uno_cox_mean": mean_uno_cox,
        })
        print(f"[{combo_name}] AvgFusion={mean_har_avg:.4f}, CoxStack={mean_har_cox:.4f}±{std_har:.4f}")

    # All three
    if len(all_tags) >= 3:
        results = evaluate_combination(feat_df, y_df, splits, all_tags, args.tau)
        mean_har_avg = np.mean([r["har_avg"] for r in results])
        mean_har_cox = np.mean([r["har_cox"] for r in results])
        mean_uno_avg = np.mean([r["uno_avg"] for r in results])
        mean_uno_cox = np.mean([r["uno_cox"] for r in results])
        std_har = np.std([r["har_cox"] for r in results])
        
        combo_name = "+".join(all_tags)
        all_results.append({
            "combination": combo_name,
            "n_modalities": 3,
            "har_avg_mean": mean_har_avg,
            "har_avg_std": std_har,
            "uno_avg_mean": mean_uno_avg,
            "har_cox_mean": mean_har_cox,
            "uno_cox_mean": mean_uno_cox,
        })
        print(f"[{combo_name}] AvgFusion={mean_har_avg:.4f}, CoxStack={mean_har_cox:.4f}±{std_har:.4f}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Best combinations")
    print("="*80)
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("har_cox_mean", ascending=False)
    print(results_df[["combination", "har_cox_mean", "har_avg_std", "uno_cox_mean"]].to_string(index=False))
    
    best = results_df.iloc[0]
    print(f"\n🏆 Best: {best['combination']} (Harrell={best['har_cox_mean']:.4f})")
    
    # Check if fusion helps
    best_single = results_df[results_df["n_modalities"] == 1]["har_cox_mean"].max()
    best_fusion = results_df[results_df["n_modalities"] > 1]["har_cox_mean"].max()
    
    if best_fusion > best_single + 0.005:
        print(f"✅ Fusion helps: {best_fusion:.4f} > {best_single:.4f}")
    else:
        print(f"⚠️ Fusion does NOT help: best_single={best_single:.4f}, best_fusion={best_fusion:.4f}")
        print("   Consider using single modality or fixing the weaker branches")


if __name__ == "__main__":
    main()