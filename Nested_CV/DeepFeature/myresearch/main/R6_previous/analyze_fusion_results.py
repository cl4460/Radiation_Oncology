#!/usr/bin/env python3
"""
分析R6 fusion结果的脚本
计算融合模型vs单独模型的性能差异
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=str, required=True,
                    help="Path to summary_by_fold.csv")
    args = ap.parse_args()
    
    df = pd.read_csv(args.summary_csv)
    
    # 计算差异：融合模型 vs 单独模型
    if "uno_T_only" in df.columns and "uno" in df.columns:
        df["delta_uno_vs_T"] = df["uno"] - df["uno_T_only"]
    
    if "har_T_only" in df.columns and "harrell" in df.columns:
        df["har_T_only"] = df["har_T_only"]  # 确保列名正确
        df["delta_har_vs_T"] = df["harrell"] - df["har_T_only"]
    
    if "uno_C_only" in df.columns and "uno" in df.columns:
        df["delta_uno_vs_C"] = df["uno"] - df["uno_C_only"]
    
    if "uno_D_only" in df.columns and "uno" in df.columns:
        df["delta_uno_vs_D"] = df["uno"] - df["uno_D_only"]
    
    # 显示关键信息
    print("=" * 80)
    print("Fusion Model Analysis")
    print("=" * 80)
    print(f"\nSummary CSV: {args.summary_csv}")
    print(f"Number of folds: {len(df)}")
    
    # 显示所有fold的详细信息
    display_cols = ["fold", "uno", "uno_T_only", "uno_C_only", "uno_D_only"]
    if "delta_uno_vs_T" in df.columns:
        display_cols.extend(["delta_uno_vs_T", "wC", "wD", "wT"])
    else:
        display_cols.extend(["wC", "wD", "wT"])
    
    available_cols = [c for c in display_cols if c in df.columns]
    print("\n" + "-" * 80)
    print("Per-fold Results:")
    print("-" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(df[available_cols].to_string(index=False))
    
    # 统计信息
    if "delta_uno_vs_T" in df.columns:
        print("\n" + "-" * 80)
        print("Fusion vs T-only (Uno C-index):")
        print("-" * 80)
        print(f"Mean delta uno vs T: {df['delta_uno_vs_T'].mean():.4f}")
        print(f"Std delta uno vs T:  {df['delta_uno_vs_T'].std():.4f}")
        print(f"Min delta uno vs T:  {df['delta_uno_vs_T'].min():.4f}")
        print(f"Max delta uno vs T:  {df['delta_uno_vs_T'].max():.4f}")
        print(f"Pos folds (fusion > T-only): {(df['delta_uno_vs_T'] > 0).sum()}")
        print(f"Neg folds (fusion < T-only): {(df['delta_uno_vs_T'] < 0).sum()}")
        print(f"Equal folds (fusion = T-only): {(df['delta_uno_vs_T'] == 0).sum()}")
    
    if "delta_har_vs_T" in df.columns:
        print("\n" + "-" * 80)
        print("Fusion vs T-only (Harrell C-index):")
        print("-" * 80)
        print(f"Mean delta har vs T: {df['delta_har_vs_T'].mean():.4f}")
        print(f"Std delta har vs T:  {df['delta_har_vs_T'].std():.4f}")
        print(f"Pos folds (fusion > T-only): {(df['delta_har_vs_T'] > 0).sum()}")
        print(f"Neg folds (fusion < T-only): {(df['delta_har_vs_T'] < 0).sum()}")
    
    # 权重统计
    if all(col in df.columns for col in ["wC", "wD", "wT"]):
        print("\n" + "-" * 80)
        print("Weight Statistics:")
        print("-" * 80)
        for mod, col in [("Clinical (C)", "wC"), ("Deep (D)", "wD"), ("Tb (T)", "wT")]:
            print(f"{mod}:")
            print(f"  Mean: {df[col].mean():.4f}")
            print(f"  Std:  {df[col].std():.4f}")
            print(f"  Min:  {df[col].min():.4f}")
            print(f"  Max:  {df[col].max():.4f}")
    
    # 平均性能
    print("\n" + "-" * 80)
    print("Average Performance:")
    print("-" * 80)
    if "uno" in df.columns:
        print(f"Fusion Uno C-index: {df['uno'].mean():.4f} ± {df['uno'].std():.4f}")
    if "uno_T_only" in df.columns:
        print(f"T-only Uno C-index: {df['uno_T_only'].mean():.4f} ± {df['uno_T_only'].std():.4f}")
    if "uno_C_only" in df.columns:
        print(f"C-only Uno C-index: {df['uno_C_only'].mean():.4f} ± {df['uno_C_only'].std():.4f}")
    if "uno_D_only" in df.columns:
        print(f"D-only Uno C-index: {df['uno_D_only'].mean():.4f} ± {df['uno_D_only'].std():.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

