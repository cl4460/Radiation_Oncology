#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并所有 fold 的 embeddings 并转换为 CSV 格式

用法:
python merge_fold_embeddings.py \
  --fold_dir /path/to/lr_7e-4 \
  --clinical_csv /path/to/clinical.csv \
  --out_csv /path/to/deep_resnet10_512.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple


def load_fold_embeddings(fold_dir: Path, fold: int) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """加载单个 fold 的 train 和 val embeddings"""
    fold_path = fold_dir / f"fold_{fold}"
    
    # Train
    train_emb = np.load(fold_path / "train_embeddings.npy")
    with open(fold_path / "train_pids.csv", 'r') as f:
        train_pids = [line.strip() for line in f if line.strip()]
    
    # Val
    val_emb = np.load(fold_path / "val_embeddings.npy")
    with open(fold_path / "val_pids.csv", 'r') as f:
        val_pids = [line.strip() for line in f if line.strip()]
    
    return train_emb, train_pids, val_emb, val_pids


def merge_all_folds(fold_dir: Path, n_folds: int = 5) -> Tuple[np.ndarray, List[str]]:
    """合并所有 fold 的 embeddings"""
    all_embs = []
    all_pids = []
    
    for fold in range(n_folds):
        print(f"Loading fold {fold}...")
        train_emb, train_pids, val_emb, val_pids = load_fold_embeddings(fold_dir, fold)
        
        # 合并 train 和 val
        all_embs.append(train_emb)
        all_embs.append(val_emb)
        all_pids.extend(train_pids)
        all_pids.extend(val_pids)
    
    # 堆叠所有 embeddings
    merged_embs = np.vstack(all_embs)
    
    print(f"\nMerged embeddings shape: {merged_embs.shape}")
    print(f"Total PIDs: {len(all_pids)}")
    print(f"Unique PIDs: {len(set(all_pids))}")
    
    return merged_embs, all_pids


def align_with_clinical(embs: np.ndarray, pids: List[str], clinical_csv: str) -> Tuple[np.ndarray, List[str]]:
    """根据 clinical CSV 的 PatientID 顺序对齐 embeddings"""
    df_clin = pd.read_csv(clinical_csv)
    
    if "PatientID" not in df_clin.columns:
        raise ValueError("Clinical CSV must contain 'PatientID' column")
    
    clinical_pids = df_clin["PatientID"].astype(str).tolist()
    
    # 创建 PID 到 embedding 的映射
    pid_to_emb = {pid: emb for pid, emb in zip(pids, embs)}
    
    # 按 clinical CSV 的顺序提取 embeddings
    aligned_embs = []
    aligned_pids = []
    missing_pids = []
    
    for pid in clinical_pids:
        if pid in pid_to_emb:
            aligned_embs.append(pid_to_emb[pid])
            aligned_pids.append(pid)
        else:
            missing_pids.append(pid)
    
    if missing_pids:
        print(f"\n⚠️  Warning: {len(missing_pids)} PIDs in clinical CSV not found in embeddings:")
        print(f"   Examples: {missing_pids[:5]}")
    
    aligned_embs = np.array(aligned_embs)
    
    print(f"\nAligned embeddings shape: {aligned_embs.shape}")
    print(f"Aligned PIDs: {len(aligned_pids)}")
    
    return aligned_embs, aligned_pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", required=True, help="Directory containing fold_0, fold_1, ...")
    ap.add_argument("--clinical_csv", required=True, help="Clinical CSV to align PatientID order")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--n_folds", type=int, default=5, help="Number of folds")
    args = ap.parse_args()
    
    fold_dir = Path(args.fold_dir)
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
    
    # 1. 合并所有 fold
    print("=" * 80)
    print("Step 1: Merging all folds...")
    print("=" * 80)
    merged_embs, merged_pids = merge_all_folds(fold_dir, args.n_folds)
    
    # 2. 对齐 clinical CSV 的顺序
    print("\n" + "=" * 80)
    print("Step 2: Aligning with clinical CSV...")
    print("=" * 80)
    aligned_embs, aligned_pids = align_with_clinical(merged_embs, merged_pids, args.clinical_csv)
    
    # 3. 保存为 CSV
    print("\n" + "=" * 80)
    print("Step 3: Saving to CSV...")
    print("=" * 80)
    
    N, D = aligned_embs.shape
    cols = ["PatientID"] + [f"d{i}" for i in range(D)]
    
    df_out = pd.DataFrame(aligned_embs, columns=[f"d{i}" for i in range(D)])
    df_out.insert(0, "PatientID", aligned_pids)
    
    df_out.to_csv(args.out_csv, index=False)
    
    print(f"\n✅ Saved: {args.out_csv}")
    print(f"   Shape: ({N}, {D+1})")
    print(f"   Format: PatientID, d0, d1, ..., d{D-1}")


if __name__ == "__main__":
    main()












