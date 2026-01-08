#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查是否需要运行 make_deep_csv_from_npy.py 来转换 deep features
"""

import os
import sys
from pathlib import Path

# 检查路径
lr_7e4_dir = Path("/home/lichengze/Research/Nested_CV/DeepFeature/myresearch/phase3/phase3_outputs_ResNet10/lr_7e-4")
existing_csv = Path("/home/lichengze/Research/Nested_CV/DeepFeature/myresearch/main/R3/Df_features.csv")
clinical_csv = Path("/home/lichengze/Research/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")

print("=" * 80)
print("检查 phase3_0617.py (lr_7e-4) 输出是否需要转换")
print("=" * 80)

# 1. 检查 lr_7e-4 输出文件
print("\n1. 检查 lr_7e-4 输出文件:")
all_folds_have_files = True
for fold in range(5):
    fold_dir = lr_7e4_dir / f"fold_{fold}"
    train_emb = fold_dir / "train_embeddings.npy"
    train_pids = fold_dir / "train_pids.csv"
    val_emb = fold_dir / "val_embeddings.npy"
    val_pids = fold_dir / "val_pids.csv"
    
    has_train = train_emb.exists() and train_pids.exists()
    has_val = val_emb.exists() and val_pids.exists()
    
    print(f"  Fold {fold}: train={has_train}, val={has_val}")
    if not (has_train and has_val):
        all_folds_have_files = False

# 2. 检查现有 CSV
print("\n2. 检查现有 Df_features.csv:")
if existing_csv.exists():
    print(f"  ✅ 存在: {existing_csv}")
    print(f"  大小: {existing_csv.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  修改时间: {existing_csv.stat().st_mtime}")
    
    # 读取第一行检查格式
    with open(existing_csv, 'r') as f:
        header = f.readline().strip()
        cols = header.split(',')
        print(f"  列数: {len(cols)}")
        print(f"  第一列: {cols[0]}")
        print(f"  特征列格式: {cols[1] if len(cols) > 1 else 'N/A'}")
else:
    print(f"  ❌ 不存在: {existing_csv}")

# 3. 检查临床 CSV
print("\n3. 检查临床 CSV:")
if clinical_csv.exists():
    print(f"  ✅ 存在: {clinical_csv}")
    # 读取行数（简单检查）
    with open(clinical_csv, 'r') as f:
        lines = sum(1 for _ in f)
    print(f"  行数（包括header）: {lines}")
else:
    print(f"  ❌ 不存在: {clinical_csv}")

# 4. 结论
print("\n" + "=" * 80)
print("结论:")
print("=" * 80)

if existing_csv.exists():
    print("\n✅ 你已经有一个 Df_features.csv 文件")
    print("\n⚠️  但是需要确认以下几点：")
    print("  1. 这个 CSV 是否来自 lr_7e-4 的输出？")
    print("  2. 它是否包含了所有 fold 的数据（train + val）？")
    print("  3. PatientID 顺序是否与 clinical CSV 对齐？")
    print("\n建议：")
    print("  如果你不确定，最好重新生成以确保正确性：")
    print("  - 合并所有 fold 的 train_embeddings.npy 和 val_embeddings.npy")
    print("  - 使用 make_deep_csv_from_npy.py 转换")
else:
    print("\n❌ 没有找到 Df_features.csv")
    print("\n✅ 需要运行 make_deep_csv_from_npy.py")
    print("\n步骤：")
    print("  1. 合并所有 fold 的 embeddings.npy 和 pids.csv")
    print("  2. 运行转换脚本")

print("\n" + "=" * 80)












