#!/usr/bin/env python3
"""
从ResNet10的单个最佳fold模型提取所有患者的embeddings
类似于CNN pipeline的extract_unified_embeddings.py
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
    Lambdad, ConcatItemsd, DeleteItemsd, ToTensord
)
from torch.utils.data import Dataset, DataLoader

# Paths
BASE_DIR = Path("/home/lichengze/Research/Nested_CV/DeepFeature/myresearch/phase3/phase3_outputs_ResNet10/lr_6.2e-4")
CROP_LOG_CSV = "/home/lichengze/Research/Nested_CV/CNN_pipeline/phase2_outputs/crop_log.csv"
OUTPUT_CSV = "/home/lichengze/Research/com/resnet10_512_unified.csv"

# 选择最佳fold: fold_4 (best_uno=0.6188)
FOLD_TO_USE = 4

# GPU selection
GPU_ID = 1
DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2


class CTDataset(Dataset):
    def __init__(self, manifest, transform):
        self.manifest = manifest
        self.transform = transform
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        data = {
            "ct": row["ct_path"],
            "mask": row["mask_path"],
            "patient_id": row["patient_id"]
        }
        data = self.transform(data)
        return data


class ResNet10_ImageOnly_GN(nn.Module):
    def __init__(self, in_channels: int, n_bins: int, dropout: float=0.35):
        super().__init__()
        self.backbone = monai_resnet.resnet10(
            spatial_dims=3, n_input_channels=in_channels, num_classes=512
        )
        self._bn_to_gn(self.backbone)
        self.head = nn.Sequential(
            nn.Linear(512, 256), 
            nn.LayerNorm(256), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_bins)
        )

    def _bn_to_gn(self, module, max_groups=8):
        for name, child in module.named_children():
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                C = child.num_features
                g = min(max_groups, C)
                while C % g != 0 and g > 1:
                    g //= 2
                setattr(module, name, nn.GroupNorm(g, C))
            else:
                self._bn_to_gn(child, max_groups)


def build_model(ckpt_path):
    """Load ResNet10 model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    n_bins = ckpt.get("n_time_bins", 15)
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, uno={ckpt.get('uno_integral', 'N/A')}, n_bins={n_bins}")
    
    # Build full model
    model = ResNet10_ImageOnly_GN(in_channels=2, n_bins=n_bins)
    
    # Load state dict
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(DEVICE)


def main():
    # Load crop log (contains paths to all CT/mask files)
    crop = pd.read_csv(CROP_LOG_CSV)
    
    # Use out_ct and out_mask columns
    manifest = crop[["patient_id", "out_ct", "out_mask"]].rename(columns={
        "out_ct": "ct_path",
        "out_mask": "mask_path"
    })
    
    # Fix paths: update old path prefix to Nested_CV
    old_prefix = "/home/lichengze/Research/CNN_pipeline"
    new_prefix = "/home/lichengze/Research/Nested_CV/CNN_pipeline"
    manifest["ct_path"] = manifest["ct_path"].str.replace(old_prefix, new_prefix, regex=False)
    manifest["mask_path"] = manifest["mask_path"].str.replace(old_prefix, new_prefix, regex=False)
    
    # Check if files exist
    manifest["ct_exists"] = manifest["ct_path"].apply(lambda p: Path(p).exists())
    manifest["mask_exists"] = manifest["mask_path"].apply(lambda p: Path(p).exists())
    
    print(f"Total patients in crop_log: {len(manifest)}")
    print(f"CT files exist: {manifest['ct_exists'].sum()}")
    print(f"Mask files exist: {manifest['mask_exists'].sum()}")
    
    manifest = manifest[manifest["ct_exists"] & manifest["mask_exists"]].copy()
    manifest = manifest.drop(columns=["ct_exists", "mask_exists"])
    
    print(f"Patients with valid files: {len(manifest)}")
    
    # Build transform (match phase3 training)
    transform = Compose([
        LoadImaged(keys=["ct", "mask"]),
        EnsureChannelFirstd(keys=["ct", "mask"]),
        EnsureTyped(keys=["ct", "mask"], dtype=torch.float32, track_meta=False),
        Lambdad(keys=["mask"], func=lambda x: (x > 0).float()),
        ConcatItemsd(keys=["ct", "mask"], name="image"),
        DeleteItemsd(keys=["ct", "mask"]),
        ToTensord(keys=["image"]),
    ])
    
    # Load model
    ckpt_path = BASE_DIR / f"fold_{FOLD_TO_USE}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    model = build_model(ckpt_path)
    
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
            
            # Forward through backbone only (not head)
            embeddings = model.backbone(images)  # [B, 512]
            
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

