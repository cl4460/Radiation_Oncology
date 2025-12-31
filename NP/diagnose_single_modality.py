#!/usr/bin/env python3
"""
diagnose_single_modality.py
===========================
诊断每个单模态OOF的独立表现，找出问题分支
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv


def find_risk_column(df):
    candidates = ["risk_pred_raw", "risk", "risk_pred", "pred", "oof_pred", "risk_integral"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No risk column found. Available: {list(df.columns)}")


def find_id_column(df):
    candidates = ["PatientID", "patient_id", "pid", "PID"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No ID column found. Available: {list(df.columns)}")


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
    fold_keys = sorted([k for k in splits if k.startswith("fold_")])

    # Load each OOF
    oof_data = {}
    for oof_arg in args.oof:
        tag, path = oof_arg.split(":", 1)
        df = pd.read_csv(path)
        pid_col = find_id_column(df)
        risk_col = find_risk_column(df)
        oof_data[tag] = pd.DataFrame({
            "patient_id": df[pid_col].astype(str).str.strip(),
            "risk": pd.to_numeric(df[risk_col], errors="coerce"),
        })
        print(f"[{tag}] Loaded {len(df)} rows, risk col='{risk_col}'")

    print("\n" + "="*80)
    print("SINGLE MODALITY PERFORMANCE (per fold)")
    print("="*80)

    results = []
    
    for tag, oof_df in oof_data.items():
        print(f"\n--- {tag} ---")
        fold_metrics = []
        
        for fk in fold_keys:
            tr_ids = [str(x).strip() for x in splits[fk]["train"]]
            va_ids = [str(x).strip() for x in splits[fk]["val"]]
            
            # Merge with labels
            merged = y_df.merge(oof_df, on="patient_id", how="inner")
            
            tr = merged[merged["patient_id"].isin(tr_ids)]
            va = merged[merged["patient_id"].isin(va_ids)]
            
            if len(va) == 0:
                print(f"  {fk}: NO VAL DATA!")
                continue
            
            y_tr = Surv.from_arrays(tr["event"].astype(bool), tr["time"].astype(float))
            y_va = Surv.from_arrays(va["event"].astype(bool), va["time"].astype(float))
            risk_va = va["risk"].to_numpy()
            
            # Check risk variance
            risk_std = np.std(risk_va)
            if risk_std < 1e-10:
                print(f"  {fk}: CONSTANT RISK (std={risk_std:.2e}) ← PROBLEM!")
                fold_metrics.append({"fold": fk, "harrell": np.nan, "uno": np.nan, "std": risk_std})
                continue
            
            # Harrell C
            har_c = concordance_index_censored(y_va["event"], y_va["time"], risk_va)[0]
            
            # Try both directions (in case sign is flipped)
            har_c_neg = concordance_index_censored(y_va["event"], y_va["time"], -risk_va)[0]
            if har_c_neg > har_c:
                print(f"  {fk}: ⚠️ Sign flipped! Using -risk")
                risk_va = -risk_va
                har_c = har_c_neg
            
            # Uno C
            tau_eff = min(args.tau, float(np.max(y_tr["time"])) - 1, float(np.max(y_va["time"])) - 1)
            try:
                uno_c = concordance_index_ipcw(y_tr, y_va, risk_va, tau=tau_eff)[0]
            except:
                uno_c = np.nan
            
            fold_metrics.append({"fold": fk, "harrell": har_c, "uno": uno_c, "std": risk_std})
            
            # Flag problematic folds
            flag = ""
            if har_c < 0.52:
                flag = "← WEAK"
            elif har_c > 0.62:
                flag = "← GOOD"
            print(f"  {fk}: Harrell={har_c:.4f}, Uno={uno_c:.4f}, risk_std={risk_std:.4f} {flag}")
        
        # Summary for this modality
        valid = [m for m in fold_metrics if not np.isnan(m["harrell"])]
        if valid:
            mean_har = np.mean([m["harrell"] for m in valid])
            std_har = np.std([m["harrell"] for m in valid])
            mean_uno = np.nanmean([m["uno"] for m in valid])
            print(f"  MEAN: Harrell={mean_har:.4f}±{std_har:.4f}, Uno={mean_uno:.4f}")
            results.append({"tag": tag, "harrell_mean": mean_har, "harrell_std": std_har, "uno_mean": mean_uno})

    # Final comparison
    print("\n" + "="*80)
    print("MODALITY COMPARISON")
    print("="*80)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Identify best single modality
    if len(results_df) > 0:
        best = results_df.loc[results_df["harrell_mean"].idxmax()]
        print(f"\n🏆 Best single modality: {best['tag']} (Harrell={best['harrell_mean']:.4f})")
        
        # Check if stacking is likely to help
        if best["harrell_mean"] > 0.57:
            print("   → This is already better than your stacking result (0.566)")
            print("   → Something is wrong with the stacking or other modalities are hurting")


if __name__ == "__main__":
    main()