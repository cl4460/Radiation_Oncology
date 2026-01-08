#!/usr/bin/env python3
"""
Extract embeddings from ALL patients using a SINGLE fold model.
This ensures all embeddings are in the same feature space.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from monai.networks.nets import resnet as monai_resnet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConcatItemsd, DeleteItemsd
)
from torch.utils.data import Dataset, DataLoader

# Paths - UPDATE THESE
BASE_DIR = Path("/home/lichengze/Research/Nested_CV/CNN_pipeline/phase3_outputs/learning_rate/learning_rate_corrected/output/lr_7e-4")
CROP_LOG_CSV = "/home/lichengze/Research/Nested_CV/CNN_pipeline/phase2_outputs/crop_log.csv"
OUTPUT_CSV = "/home/lichengze/Research/com/phase3_0617_512.csv"

# Which fold model to use (fold_3 has highest C-index: 0.645)
FOLD_TO_USE = 3  # Change to 0, 1, 2, 3, or 4

# GPU selection: use GPU 1 instead of default GPU 0
GPU_ID = 1
DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # Reduced from 8 to avoid OOM


class CTDataset(Dataset):
    def __init__(self, manifest, transform):
        self.manifest = manifest
        self.transform = transform
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        data = {
            "image": row["ct_path"],
            "mask": row["mask_path"],
            "patient_id": row["patient_id"]
        }
        data = self.transform(data)
        return data


def _replace_bn_with_gn(module, num_groups=8):
    """Replace BatchNorm with GroupNorm (must match training)."""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            num_channels = child.num_features
            groups = min(num_groups, num_channels)
            while num_channels % groups != 0:
                groups = groups // 2
                if groups == 0:
                    groups = 1
                    break
            setattr(module, name, nn.GroupNorm(groups, num_channels))
        else:
            _replace_bn_with_gn(child, num_groups)


def build_backbone(ckpt_path):
    """Load backbone from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, uno={ckpt['uno_integral']:.4f}")
    
    # Build backbone
    backbone = monai_resnet.resnet10(
        spatial_dims=3,
        n_input_channels=2,  # CT + mask
        num_classes=512
    )
    
    # CRITICAL: Replace BatchNorm with GroupNorm (must match training)
    _replace_bn_with_gn(backbone)
    
    # Extract backbone weights
    state_dict = {}
    for k, v in ckpt['state_dict'].items():
        if k.startswith('backbone.'):
            state_dict[k.replace('backbone.', '')] = v
    
    backbone.load_state_dict(state_dict, strict=True)
    backbone.eval()
    return backbone.to(DEVICE)


def main():
    # Load crop log (contains paths to all CT/mask files)
    crop = pd.read_csv(CROP_LOG_CSV)
    manifest = crop[["patient_id", "out_ct", "out_mask"]].rename(columns={
        "out_ct": "ct_path",
        "out_mask": "mask_path"
    })
    
    # Fix paths: update old path to new Nested_CV path
    old_prefix = "/home/lichengze/Research/CNN_pipeline"
    new_prefix = "/home/lichengze/Research/Nested_CV/CNN_pipeline"
    manifest["ct_path"] = manifest["ct_path"].str.replace(old_prefix, new_prefix, regex=False)
    manifest["mask_path"] = manifest["mask_path"].str.replace(old_prefix, new_prefix, regex=False)
    
    # Filter out rows where files don't exist
    from pathlib import Path
    manifest["ct_exists"] = manifest["ct_path"].apply(lambda p: Path(p).exists())
    manifest["mask_exists"] = manifest["mask_path"].apply(lambda p: Path(p).exists())
    manifest = manifest[manifest["ct_exists"] & manifest["mask_exists"]].copy()
    manifest = manifest.drop(columns=["ct_exists", "mask_exists"])
    
    print(f"Total patients: {len(manifest)}")
    
    # Build transform
    transform = Compose([
        LoadImaged(keys=["image", "mask"], image_only=True),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ConcatItemsd(keys=["image", "mask"], name="image", dim=0),
        DeleteItemsd(keys=["mask"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])
    
    # Load model
    ckpt_path = BASE_DIR / f"fold_{FOLD_TO_USE}" / "best.pt"
    backbone = build_backbone(ckpt_path)
    
    # Create dataset and dataloader
    dataset = CTDataset(manifest, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Extract embeddings
    all_embeddings = []
    all_patient_ids = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            images = batch["image"].to(DEVICE)
            patient_ids = batch["patient_id"]
            
            # Forward through backbone
            embeddings = backbone(images)  # [B, 512]
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_patient_ids.extend(patient_ids)
    
    # Combine
    all_embeddings = np.vstack(all_embeddings)
    print(f"Extracted embeddings shape: {all_embeddings.shape}")
    
    # Save
    df = pd.DataFrame(all_embeddings, columns=[f"d{i}" for i in range(512)])
    df.insert(0, "PatientID", all_patient_ids)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to: {OUTPUT_CSV}")
    
    # Quick validation: check PC1 correlation with survival
    print("\n=== Quick Validation ===")
    from sklearn.decomposition import PCA
    from sksurv.metrics import concordance_index_censored
    
    clinical = pd.read_csv("/home/lichengze/Research/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
    clinical = clinical.set_index("PatientID")
    
    common = df.set_index("PatientID").index.intersection(clinical.index)
    X = df.set_index("PatientID").loc[common].values
    y_e = clinical.loc[common, "deadstatus.event"].values.astype(bool)
    y_t = clinical.loc[common, "Survival.time"].values.astype(float)
    
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X).ravel()
    
    c_pos = concordance_index_censored(y_e, y_t, pc1)[0]
    c_neg = concordance_index_censored(y_e, y_t, -pc1)[0]
    
    print(f"PC1 C-index (positive): {c_pos:.4f}")
    print(f"PC1 C-index (negative): {c_neg:.4f}")
    print(f"Best direction: {max(c_pos, c_neg):.4f}")
    print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")


if __name__ == "__main__":
    main()