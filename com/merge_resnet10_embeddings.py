#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 ResNet10 所有 fold 的 train+val embeddings 成单个 CSV
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/home/lichengze/Research/Nested_CV/DeepFeature/myresearch/phase3/phase3_outputs_ResNet10/lr_6.2e-4")
OUTPUT_CSV = "/home/lichengze/Research/com/resnet10_512.csv"

def main():
    all_embeddings = []
    all_pids = []
    
    for fold in range(5):
        fold_dir = BASE_DIR / f"fold_{fold}"
        
        # Load train embeddings
        train_emb_path = fold_dir / "train_embeddings.npy"
        train_pid_path = fold_dir / "train_pids.csv"
        
        if train_emb_path.exists() and train_pid_path.exists():
            train_emb = np.load(train_emb_path)
            train_pids = pd.read_csv(train_pid_path, header=None).iloc[:, 0].astype(str).tolist()
            
            print(f"Fold {fold} - Train: {train_emb.shape}, {len(train_pids)} patients")
            all_embeddings.append(train_emb)
            all_pids.extend(train_pids)
        
        # Load val embeddings
        val_emb_path = fold_dir / "val_embeddings.npy"
        val_pid_path = fold_dir / "val_pids.csv"
        
        if val_emb_path.exists() and val_pid_path.exists():
            val_emb = np.load(val_emb_path)
            val_pids = pd.read_csv(val_pid_path, header=None).iloc[:, 0].astype(str).tolist()
            
            print(f"Fold {fold} - Val: {val_emb.shape}, {len(val_pids)} patients")
            all_embeddings.append(val_emb)
            all_pids.extend(val_pids)
    
    # Combine all embeddings
    all_embeddings = np.vstack(all_embeddings)
    print(f"\nTotal embeddings shape: {all_embeddings.shape}")
    print(f"Total patients: {len(all_pids)}")
    
    # Check for duplicates and keep first occurrence
    seen = set()
    unique_indices = []
    unique_pids = []
    unique_embs = []
    
    for i, pid in enumerate(all_pids):
        if pid not in seen:
            seen.add(pid)
            unique_indices.append(i)
            unique_pids.append(pid)
            unique_embs.append(all_embeddings[i])
    
    unique_embs = np.vstack(unique_embs)
    print(f"After removing duplicates: {unique_embs.shape}, {len(unique_pids)} unique patients")
    
    # Create DataFrame
    df = pd.DataFrame(unique_embs, columns=[f"d{i}" for i in range(512)])
    df.insert(0, "PatientID", unique_pids)
    
    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to: {OUTPUT_CSV}")
    print(f"CSV shape: {df.shape}")

if __name__ == "__main__":
    main()






