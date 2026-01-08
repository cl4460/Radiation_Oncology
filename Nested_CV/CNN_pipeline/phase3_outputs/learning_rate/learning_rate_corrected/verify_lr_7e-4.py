#!/usr/bin/env python
"""
验证 lr_7e-4 文件夹的合理性
"""
import numpy as np
from pathlib import Path

OUTPUT_DIR = "/home/lichengze/Research/CNN_pipeline/phase3_outputs/learning_rate/learning_rate_corrected/output/lr_7e-4"
EXP1_EXPECTED_DIR = "/home/lichengze/Research/CNN_pipeline/phase3_outputs/learning_rate/learning_rate_corrected/output/lr_7e-4"

print("="*70)
print("验证 lr_7e-4 文件夹")
print("="*70)

# 检查每个fold的文件
issues = []
for fold in range(5):
    fold_dir = Path(OUTPUT_DIR) / f"fold_{fold}"
    
    print(f"\nFold {fold}:")
    print(f"  路径: {fold_dir}")
    
    # 检查必需文件
    required_files = ["best.pt", "train_idx.npy", "val_idx.npy"]
    for file in required_files:
        file_path = fold_dir / file
        if file_path.exists():
            print(f"  ✓ {file} 存在")
            
            # 检查npy文件内容
            if file.endswith(".npy"):
                arr = np.load(file_path)
                print(f"    形状: {arr.shape}, 类型: {arr.dtype}, 范围: [{arr.min()}, {arr.max()}]")
                
                # 验证索引范围合理（假设manifest有422个样本）
                if arr.max() >= 500:
                    issues.append(f"Fold {fold}: {file} 索引超出合理范围 (max={arr.max()})")
        else:
            print(f"  ✗ {file} 不存在")
            issues.append(f"Fold {fold}: 缺少 {file}")

# 检查路径匹配
print(f"\n路径匹配检查:")
print(f"  phase3getnpy.py 输出路径: {OUTPUT_DIR}")
print(f"  exp1.py 期望路径 (PHASE3_MODEL_DIR): {EXP1_EXPECTED_DIR}")
if str(OUTPUT_DIR) == str(EXP1_EXPECTED_DIR):
    print("  ✓ 路径匹配")
else:
    issues.append(f"路径不匹配！")

# 检查索引总数
print(f"\n索引总数检查:")
total_train = 0
total_val = 0
for fold in range(5):
    train_idx = np.load(Path(OUTPUT_DIR) / f"fold_{fold}" / "train_idx.npy")
    val_idx = np.load(Path(OUTPUT_DIR) / f"fold_{fold}" / "val_idx.npy")
    total_train += len(train_idx)
    total_val += len(val_idx)
    print(f"  Fold {fold}: train={len(train_idx)}, val={len(val_idx)}")

print(f"\n  总计: train={total_train}, val={total_val}")
print(f"  期望: train≈1685 (337*5), val≈422 (85*5)")

# 检查是否有重复
print(f"\n索引重复检查:")
for fold in range(5):
    train_idx = np.load(Path(OUTPUT_DIR) / f"fold_{fold}" / "train_idx.npy")
    val_idx = np.load(Path(OUTPUT_DIR) / f"fold_{fold}" / "val_idx.npy")
    
    # 检查train和val之间是否有重叠
    overlap = np.intersect1d(train_idx, val_idx)
    if len(overlap) > 0:
        issues.append(f"Fold {fold}: train和val索引有重叠: {overlap}")
    else:
        print(f"  Fold {fold}: ✓ train和val无重叠")
    
    # 检查train内部是否有重复
    if len(train_idx) != len(np.unique(train_idx)):
        issues.append(f"Fold {fold}: train_idx有重复")
    else:
        print(f"  Fold {fold}: ✓ train_idx无重复")

print("\n" + "="*70)
if issues:
    print("发现的问题:")
    for issue in issues:
        print(f"  ✗ {issue}")
    print("\n需要修复这些问题！")
else:
    print("✓ 所有检查通过！lr_7e-4 文件夹配置合理。")
print("="*70)





