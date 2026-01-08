#!/usr/bin/env python3
"""
Extract OOF (val) embeddings from phase3 experiment folders and create CSV files for R0/R1/R2.

Key fixes vs your previous version:
1) Use ONLY val (OOF) embeddings from each fold to avoid leakage.
2) Add Clinical_M_binary = 1 if M>0 else 0, and DO NOT feed raw Clinical_M_Stage by default.
3) Keep Overall Stage as one-hot numeric columns (0/1). These MUST NOT be put into cat_cols later.

Usage:
    python extract_embeddings_for_r0.py \
        --exp_dir ../../../phase3/phase3_seed/exp_xxxx \
        --clinical_csv /path/to/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
        --output_dir ./r0_feature_csvs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_pid_list(path: Path):
    """Read patient IDs from a single-column CSV (no header)."""
    return pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()


def extract_clinical_features(df: pd.DataFrame, pid_list: list) -> pd.DataFrame:
    """
    Build clinical feature table aligned to pid_list order.
    Output columns:
      PatientID
      age
      gender (0/1)
      clinical_T_Stage (numeric ordinal)
      Clinical_N_Stage (numeric ordinal)
      Clinical_M_binary (0/1)
      Overall_Stage_II, Overall_Stage_IIIa, Overall_Stage_IIIb  (I as baseline)
      Overall_Stage_Unknown (rare safety net)
    """
    use_cols = [
        "PatientID", "age", "gender",
        "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage",
        "Overall.Stage",
        "Survival.time", "deadstatus.event",
    ]

    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in clinical CSV: {missing}")

    d = df[use_cols].copy()
    d["PatientID"] = d["PatientID"].astype(str)

    # Filter & align
    d = d[d["PatientID"].isin(pid_list)].copy()
    d = d.set_index("PatientID").loc[pid_list].reset_index()

    out = pd.DataFrame()
    out["PatientID"] = d["PatientID"]

    # age
    out["age"] = pd.to_numeric(d["age"], errors="coerce")

    # gender -> 0/1
    gmap = {"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0}
    out["gender"] = d["gender"].astype(str).str.strip().str.lower().map(gmap)

    # T/N as ordinal numeric
    out["clinical_T_Stage"] = pd.to_numeric(d["clinical.T.Stage"], errors="coerce")
    out["Clinical_N_Stage"] = pd.to_numeric(d["Clinical.N.Stage"], errors="coerce")

    # M -> binary (robust against rare categories)
    m_raw = pd.to_numeric(d["Clinical.M.Stage"], errors="coerce").fillna(0)
    out["Clinical_M_binary"] = (m_raw > 0).astype(float)

    # Overall.Stage -> one-hot numeric, baseline = I (so we keep only II/IIIa/IIIb)
    os = d["Overall.Stage"].astype(str).str.strip()

    out["Overall_Stage_II"] = (os == "II").astype(float)
    out["Overall_Stage_IIIa"] = (os == "IIIa").astype(float)
    out["Overall_Stage_IIIb"] = (os == "IIIb").astype(float)

    # Safety: any row that is not I/II/IIIa/IIIb becomes Unknown
    known = (os == "I") | (os == "II") | (os == "IIIa") | (os == "IIIb")
    out["Overall_Stage_Unknown"] = (~known).astype(float)

    # Survival labels (keep for R0/R1/R2)
    out["Survival.time"] = pd.to_numeric(d["Survival.time"], errors="coerce")
    out["deadstatus.event"] = pd.to_numeric(d["deadstatus.event"], errors="coerce")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=str, required=True,
                    help="Path to experiment directory (e.g., phase3_seed/exp_xxx)")
    ap.add_argument("--clinical_csv", type=str, required=True,
                    help="Path to clinical CSV file (contains survival columns)")
    ap.add_argument("--output_dir", type=str, default="./r0_feature_csvs",
                    help="Output directory for generated CSV files")
    ap.add_argument("--emb_dim", type=int, default=512,
                    help="Expected embedding dimension")
    ap.add_argument("--n_folds", type=int, default=5,
                    help="Number of CV folds to read (default 5)")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    clinical_df = pd.read_csv(args.clinical_csv)
    print(f"✓ Loaded clinical CSV: {clinical_df.shape}")

    # Collect ONLY OOF/val embeddings across folds
    all_pids = []
    all_embeddings = []

    for fold_idx in range(args.n_folds):
        fold_dir = exp_dir / f"fold_{fold_idx}"
        if not fold_dir.exists():
            print(f"⚠ Warning: fold_{fold_idx} directory not found, skipping")
            continue

        val_pids_file = fold_dir / "val_pids.csv"
        if not val_pids_file.exists():
            print(f"⚠ Warning: Missing val_pids.csv in fold_{fold_idx}, skipping")
            continue

        val_pids = read_pid_list(val_pids_file)

        val_emb_candidates = [
            fold_dir / "val_embeddings.npy",
            fold_dir / "val_embeddings_extracted.npy",
        ]
        val_emb_path = next((p for p in val_emb_candidates if p.exists()), None)
        if val_emb_path is None:
            print(f"⚠ Warning: Missing val embedding file in fold_{fold_idx}, skipping")
            continue

        val_emb = np.load(val_emb_path).astype(np.float32)
        if val_emb.shape[0] != len(val_pids):
            raise RuntimeError(
                f"fold_{fold_idx}: val embedding count mismatch: "
                f"{val_emb.shape[0]} vs {len(val_pids)}"
            )
        if val_emb.shape[1] != args.emb_dim:
            raise RuntimeError(
                f"fold_{fold_idx}: emb_dim mismatch: {val_emb.shape[1]} vs expected {args.emb_dim}"
            )

        print(f"✓ fold_{fold_idx}: val={len(val_pids)}, emb_dim={val_emb.shape[1]}")
        all_pids.extend(val_pids)
        all_embeddings.append(val_emb)

    if not all_pids:
        raise RuntimeError("No valid folds found! Check exp_dir and fold outputs.")

    all_embeddings = np.vstack(all_embeddings)

    # In proper KFold, each patient should appear exactly once in val across folds
    n_unique = len(set(all_pids))
    print(f"\nCollected OOF data: n={len(all_pids)}, unique={n_unique}, emb={all_embeddings.shape}")
    if n_unique != len(all_pids):
        # still handle gracefully if something odd happened
        print("⚠ Warning: Duplicate PatientID found across val folds. Keeping first occurrence.")
        uniq_pids, uniq_idx = np.unique(all_pids, return_index=True)
        all_pids = [all_pids[i] for i in uniq_idx]
        all_embeddings = all_embeddings[uniq_idx]

    # deep_features.csv
    deep_df = pd.DataFrame(all_embeddings, columns=[f"emb_{i:03d}" for i in range(all_embeddings.shape[1])])
    deep_df.insert(0, "PatientID", all_pids)

    # Attach survival columns
    idx = clinical_df.set_index(clinical_df["PatientID"].astype(str))
    deep_df["Survival.time"] = idx.loc[all_pids]["Survival.time"].values
    deep_df["deadstatus.event"] = idx.loc[all_pids]["deadstatus.event"].values

    deep_csv_path = output_dir / "deep_features.csv"
    deep_df.to_csv(deep_csv_path, index=False)
    print(f"✓ Saved: {deep_csv_path}  shape={deep_df.shape}")

    # clinical_features.csv
    clinical_subset = extract_clinical_features(clinical_df, all_pids)
    clinical_csv_path = output_dir / "clinical_features.csv"
    clinical_subset.to_csv(clinical_csv_path, index=False)
    print(f"✓ Saved: {clinical_csv_path}  shape={clinical_subset.shape}")

    # fusion_features.csv (clinical + deep)
    clin_no_surv = clinical_subset.drop(columns=["Survival.time", "deadstatus.event"], errors="ignore")
    fusion_df = clin_no_surv.merge(deep_df, on="PatientID", how="inner")
    fusion_csv_path = output_dir / "fusion_features.csv"
    fusion_df.to_csv(fusion_csv_path, index=False)
    print(f"✓ Saved: {fusion_csv_path}  shape={fusion_df.shape}")

    metadata = {
        "exp_dir": str(exp_dir),
        "clinical_csv": str(args.clinical_csv),
        "n_patients": len(all_pids),
        "embedding_dim": int(all_embeddings.shape[1]),
        "output_files": {
            "deep": deep_csv_path.name,
            "clinical": clinical_csv_path.name,
            "fusion": fusion_csv_path.name,
        },
        "notes": [
            "deep_features uses ONLY OOF/val embeddings to avoid leakage",
            "Clinical_M_binary used instead of raw Clinical_M_Stage",
            "Overall Stage kept as numeric one-hot (II/IIIa/IIIb + Unknown), do NOT put into cat_cols",
        ],
    }
    metadata_path = output_dir / "extraction_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
