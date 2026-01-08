#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm_root", type=str, required=True, 
                    help="Root directory containing R0_tb_perm_v2_* folders")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="Optional: Output CSV file path. If not provided, only display in terminal")
    ap.add_argument("--show_all", action="store_true",
                    help="Display all data rows in terminal (default: show first 5 and last 5 only)")
    args = ap.parse_args()
    
    perm_root = Path(args.perm_root)
    rows = []
    
    # Collect all R0_tb_perm_v2_* folders (check both in root and in R0_tb_perm_v2 subdirectory)
    perm_dirs = sorted([d for d in perm_root.glob("R0_tb_perm_v2_*") if d.is_dir()])
    
    # Also check in R0_tb_perm_v2 subdirectory if it exists
    subdir = perm_root / "R0_tb_perm_v2"
    if subdir.exists() and subdir.is_dir():
        perm_dirs.extend(sorted([d for d in subdir.glob("R0_tb_perm_v2_*") if d.is_dir()]))
    
    print(f"Found {len(perm_dirs)} permutation folders")
    
    for perm_dir in perm_dirs:
        overall_json = perm_dir / "overall.json"
        if not overall_json.exists():
            print(f"Warning: {perm_dir.name}/overall.json not found, skipping")
            continue
        
        with open(overall_json, "r") as f:
            data = json.load(f)
        
        # Extract folder number from name (e.g., R0_tb_perm_v2_67 -> 67)
        folder_num = perm_dir.name.split("_")[-1]
        
        row = {
            "run": perm_dir.name,
            "run_id": int(folder_num),
            "seed": data.get("seed", None),
            "har_mean": data.get("har_mean", None),
            "har_std": data.get("har_std", None),
            "uno_mean": data.get("uno_mean", None),
            "uno_std": data.get("uno_std", None),
            "n_folds": data.get("n_folds", None),
        }
        rows.append(row)
    
    if len(rows) == 0:
        print("Error: No valid permutation folders found!")
        return
    
    df = pd.DataFrame(rows).sort_values("run_id")
    
    # Save summary CSV if requested
    if args.out_csv:
        out_path = perm_root / args.out_csv
        df.to_csv(out_path, index=False)
        print(f"[OK] Saved summary to {out_path} with {len(df)} runs")
    
    # Display data in terminal
    print("\n" + "="*80)
    print(f"Summary of {len(df)} Permutation Runs")
    print("="*80)
    
    # Configure pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}' if pd.notna(x) else '')
    
    if args.show_all:
        # Display all data
        pd.set_option('display.max_rows', None)
        print("\nAll runs:")
        print(df.to_string(index=False))
    else:
        # Display first and last few rows
        pd.set_option('display.max_rows', 10)
        print("\nFirst 5 runs:")
        print(df.head().to_string(index=False))
        print("\nLast 5 runs:")
        print(df.tail().to_string(index=False))
        print(f"\n(Use --show_all to display all {len(df)} runs)")
    
    # Print distribution statistics
    print("\n" + "="*60)
    print("Distribution Statistics (100 Permutations)")
    print("="*60)
    
    for metric in ["har_mean", "uno_mean"]:
        if metric in df.columns:
            vals = df[metric].dropna().to_numpy()
            if len(vals) > 0:
                print(f"\n{metric.upper()}:")
                print(f"  Mean: {np.mean(vals):.4f}")
                print(f"  Std:  {np.std(vals, ddof=1):.4f}")
                print(f"  Min:  {np.min(vals):.4f}")
                print(f"  Max:  {np.max(vals):.4f}")
                print(f"  Median: {np.median(vals):.4f}")
                print(f"  25th percentile: {np.percentile(vals, 25):.4f}")
                print(f"  75th percentile: {np.percentile(vals, 75):.4f}")
    
    print("\n" + "="*80)
    if args.out_csv:
        print("Summary saved successfully!")
    else:
        print("Display complete (no CSV file saved)")
    print("="*80)


if __name__ == "__main__":
    main()

