#!/usr/bin/env python3
"""
Extract embeddings from phase3 experiment folders and create CSV files for R0.

This script will:
1. Load embeddings from each fold's train/val .npy files
2. Combine all folds into a single CSV with PatientID
3. Create three CSVs:
   - deep_features.csv: PatientID + 512D embedding
   - clinical_features.csv: PatientID + clinical features  
   - fusion_features.csv: PatientID + clinical + embedding

Usage:
    python extract_embeddings_for_r0.py \
        --exp_dir ../../../phase3/phase3_seed/exp_xxxx \
        --clinical_csv /path/to/clinical.csv \
        --output_dir ./r0_feature_csvs
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def read_pid_list(path: Path):
    """Read patient IDs from a single-column CSV."""
    return pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()


def extract_clinical_features(df: pd.DataFrame, pid_list: list) -> pd.DataFrame:
    """
    Extract clinical features matching the fusion_baseline.py logic.
    Returns DataFrame with PatientID + clinical features (aligned to pid_list order).
    """
    use_cols = [
        "PatientID", "age", "gender",
        "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage",
        "Overall.Stage"
    ]
    
    # Check if all columns exist
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in clinical CSV: {missing}")
    
    d = df[use_cols].copy()
    d["PatientID"] = d["PatientID"].astype(str)
    
    # Filter to only the patients in pid_list
    d = d[d["PatientID"].isin(pid_list)].copy()
    
    # Align to pid order
    d = d.set_index("PatientID").loc[pid_list].reset_index()
    
    # Process features (simplified version - just keep raw values for R0)
    # R0 will apply its own preprocessing via the pipeline
    result = pd.DataFrame()
    result["PatientID"] = d["PatientID"]
    
    # Age (keep as-is, R0 will impute/scale)
    result["age"] = pd.to_numeric(d["age"], errors="coerce")
    
    # Gender (map to 0/1)
    gmap = {"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0}
    result["gender"] = d["gender"].astype(str).str.strip().str.lower().map(gmap)
    
    # T, N, M stages (keep as numeric)
    result["clinical_T_Stage"] = pd.to_numeric(d["clinical.T.Stage"], errors="coerce")
    result["Clinical_N_Stage"] = pd.to_numeric(d["Clinical.N.Stage"], errors="coerce")
    result["Clinical_M_Stage"] = pd.to_numeric(d["Clinical.M.Stage"], errors="coerce")
    
    # Overall Stage - use one-hot encoding
    os = d["Overall.Stage"].astype(str)
    for stage in ["I", "II", "IIIa", "IIIb"]:
        result[f"Overall_Stage_{stage}"] = (os == stage).astype(float)
    
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=str, required=True,
                    help="Path to experiment directory (e.g., phase3_seed/exp_xxx)")
    ap.add_argument("--clinical_csv", type=str, required=True,
                    help="Path to clinical CSV file")
    ap.add_argument("--output_dir", type=str, default="./r0_feature_csvs",
                    help="Output directory for generated CSV files")
    ap.add_argument("--emb_dim", type=int, default=512,
                    help="Expected embedding dimension")
    
    args = ap.parse_args()
    
    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    print(f"\n{'='*80}")
    print(f"Extracting embeddings from: {exp_dir}")
    print(f"{'='*80}\n")
    
    # Load clinical data
    clinical_df = pd.read_csv(args.clinical_csv)
    print(f"✓ Loaded clinical CSV: {clinical_df.shape}")
    
    # Collect all patients and embeddings across folds
    all_pids = []
    all_embeddings = []
    
    for fold_idx in range(5):
        fold_dir = exp_dir / f"fold_{fold_idx}"
        
        if not fold_dir.exists():
            print(f"⚠ Warning: fold_{fold_idx} directory not found, skipping")
            continue
        
        # Read patient IDs
        train_pids_file = fold_dir / "train_pids.csv"
        val_pids_file = fold_dir / "val_pids.csv"
        
        if not train_pids_file.exists() or not val_pids_file.exists():
            print(f"⚠ Warning: Missing PID files in fold_{fold_idx}, skipping")
            continue
        
        train_pids = read_pid_list(train_pids_file)
        val_pids = read_pid_list(val_pids_file)
        
        # Try to find embedding files (different naming conventions)
        train_emb_candidates = [
            fold_dir / "train_embeddings.npy",
            fold_dir / "train_embeddings_extracted.npy"
        ]
        val_emb_candidates = [
            fold_dir / "val_embeddings.npy",
            fold_dir / "val_embeddings_extracted.npy"
        ]
        
        train_emb_path = None
        for p in train_emb_candidates:
            if p.exists():
                train_emb_path = p
                break
        
        val_emb_path = None
        for p in val_emb_candidates:
            if p.exists():
                val_emb_path = p
                break
        
        if train_emb_path is None or val_emb_path is None:
            print(f"⚠ Warning: Missing embedding files in fold_{fold_idx}, skipping")
            continue
        
        # Load embeddings
        train_emb = np.load(train_emb_path).astype(np.float32)
        val_emb = np.load(val_emb_path).astype(np.float32)
        
        # Verify dimensions
        assert train_emb.shape[0] == len(train_pids), \
            f"fold_{fold_idx}: train embedding count mismatch: {train_emb.shape[0]} vs {len(train_pids)}"
        assert val_emb.shape[0] == len(val_pids), \
            f"fold_{fold_idx}: val embedding count mismatch: {val_emb.shape[0]} vs {len(val_pids)}"
        
        print(f"✓ fold_{fold_idx}: train={len(train_pids)}, val={len(val_pids)}, "
              f"emb_dim={train_emb.shape[1]}")
        
        # Collect
        all_pids.extend(train_pids)
        all_pids.extend(val_pids)
        all_embeddings.append(train_emb)
        all_embeddings.append(val_emb)
    
    if len(all_pids) == 0:
        raise RuntimeError("No valid folds found! Check your exp_dir path.")
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    print(f"\n{'='*80}")
    print(f"Collected data:")
    print(f"  Total patients: {len(all_pids)}")
    print(f"  Embedding shape: {all_embeddings.shape}")
    print(f"  Unique patients: {len(set(all_pids))}")
    print(f"{'='*80}\n")
    
    # Check for duplicates (patients appearing in multiple folds)
    unique_pids, unique_indices = np.unique(all_pids, return_index=True)
    
    if len(unique_pids) < len(all_pids):
        print(f"⚠ Found {len(all_pids) - len(unique_pids)} duplicate patients (expected with CV)")
        print(f"  Keeping first occurrence of each patient")
        all_pids = [all_pids[i] for i in unique_indices]
        all_embeddings = all_embeddings[unique_indices]
    
    # Create deep features CSV
    print("Creating deep_features.csv...")
    deep_df = pd.DataFrame(all_embeddings, 
                          columns=[f"emb_{i:03d}" for i in range(all_embeddings.shape[1])])
    deep_df.insert(0, "PatientID", all_pids)
    
    # Add survival info for R0 (it needs time/event columns)
    survival_cols = ["Survival.time", "deadstatus.event"]
    for col in survival_cols:
        if col in clinical_df.columns:
            deep_df[col] = clinical_df.set_index("PatientID").loc[all_pids][col].values
    
    deep_csv_path = output_dir / "deep_features.csv"
    deep_df.to_csv(deep_csv_path, index=False)
    print(f"✓ Saved: {deep_csv_path}")
    print(f"  Shape: {deep_df.shape}")
    print(f"  Columns: PatientID + {deep_df.shape[1]-3} embedding features + survival info")
    
    # Create clinical features CSV (only for patients we have)
    print("\nCreating clinical_features.csv...")
    clinical_subset = extract_clinical_features(clinical_df, all_pids)
    
    # Also add survival info for R0 (it needs time/event columns)
    survival_cols = ["Survival.time", "deadstatus.event"]
    for col in survival_cols:
        if col in clinical_df.columns:
            clinical_subset[col] = clinical_df.set_index("PatientID").loc[all_pids][col].values
    
    clinical_csv_path = output_dir / "clinical_features.csv"
    clinical_subset.to_csv(clinical_csv_path, index=False)
    print(f"✓ Saved: {clinical_csv_path}")
    print(f"  Shape: {clinical_subset.shape}")
    print(f"  Features: {[c for c in clinical_subset.columns if c not in ['PatientID', 'Survival.time', 'deadstatus.event']]}")
    
    # Create fusion features CSV
    print("\nCreating fusion_features.csv...")
    
    # Remove survival columns from clinical_subset temporarily
    clinical_features_only = clinical_subset.drop(columns=["Survival.time", "deadstatus.event"], errors='ignore')
    
    # Merge clinical and deep
    fusion_df = clinical_features_only.merge(deep_df, on="PatientID", how="inner")
    
    # Add back survival info
    for col in survival_cols:
        if col in clinical_df.columns:
            fusion_df[col] = clinical_df.set_index("PatientID").loc[fusion_df["PatientID"].values][col].values
    
    fusion_csv_path = output_dir / "fusion_features.csv"
    fusion_df.to_csv(fusion_csv_path, index=False)
    print(f"✓ Saved: {fusion_csv_path}")
    print(f"  Shape: {fusion_df.shape}")
    print(f"  Clinical features: {clinical_features_only.shape[1]-1}")
    print(f"  Deep features: {all_embeddings.shape[1]}")
    print(f"  Total features: {fusion_df.shape[1]-1-2} (excluding PatientID + survival)")
    
    # Create metadata file
    metadata = {
        "exp_dir": str(exp_dir),
        "clinical_csv": str(args.clinical_csv),
        "n_patients": len(all_pids),
        "embedding_dim": int(all_embeddings.shape[1]),
        "clinical_features": int(clinical_features_only.shape[1] - 1),  # -1 for PatientID
        "fusion_features": int(fusion_df.shape[1] - 3),  # -3 for PatientID + survival cols
        "output_files": {
            "deep": str(deep_csv_path.name),
            "clinical": str(clinical_csv_path.name),
            "fusion": str(fusion_csv_path.name)
        }
    }
    
    metadata_path = output_dir / "extraction_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata: {metadata_path}")
    
    # Print R0 command examples
    print(f"\n{'='*80}")
    print("✅ All done! Now you can run R0 with these commands:")
    print(f"{'='*80}\n")
    
    print("# 1. Clinical R0:")
    print(f"python r0.py \\")
    print(f"  --features_csv {clinical_csv_path} \\")
    print(f"  --labels_csv {clinical_csv_path} \\")
    print(f"  --splits_json ./splits.json \\")
    print(f"  --id_col PatientID \\")
    print(f"  --time_col Survival.time \\")
    print(f"  --event_col deadstatus.event \\")
    print(f"  --drop_cols \"Survival.time,deadstatus.event\" \\")
    print(f"  --model coxnet \\")
    print(f"  --n_perm 100 \\")
    print(f"  --n_random 200 \\")
    print(f"  --outdir ./R0_clinical_output \\")
    print(f"  --seed 42\n")
    
    print("# 2. Deep R0:")
    print(f"python r0.py \\")
    print(f"  --features_csv {deep_csv_path} \\")
    print(f"  --labels_csv {deep_csv_path} \\")
    print(f"  --splits_json ./splits.json \\")
    print(f"  --id_col PatientID \\")
    print(f"  --time_col Survival.time \\")
    print(f"  --event_col deadstatus.event \\")
    print(f"  --drop_cols \"Survival.time,deadstatus.event\" \\")
    print(f"  --model coxnet \\")
    print(f"  --n_perm 100 \\")
    print(f"  --n_random 200 \\")
    print(f"  --n_perm_x 50 \\")
    print(f"  --outdir ./R0_deep_output \\")
    print(f"  --seed 42\n")
    
    print("# 3. Fusion R0:")
    print(f"python r0.py \\")
    print(f"  --features_csv {fusion_csv_path} \\")
    print(f"  --labels_csv {fusion_csv_path} \\")
    print(f"  --splits_json ./splits.json \\")
    print(f"  --id_col PatientID \\")
    print(f"  --time_col Survival.time \\")
    print(f"  --event_col deadstatus.event \\")
    print(f"  --drop_cols \"Survival.time,deadstatus.event\" \\")
    print(f"  --model coxnet \\")
    print(f"  --n_perm 100 \\")
    print(f"  --n_random 200 \\")
    print(f"  --n_perm_x 50 \\")
    print(f"  --outdir ./R0_fusion_output \\")
    print(f"  --seed 42\n")
    
    print(f"{'='*80}")
    print("Note: The deep/fusion CSVs include Survival.time and deadstatus.event columns")
    print("so they can be used as both --features_csv and --labels_csv")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

