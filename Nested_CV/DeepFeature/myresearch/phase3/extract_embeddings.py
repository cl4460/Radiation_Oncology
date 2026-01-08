#!/usr/bin/env python
"""从已训练模型提取train/val embeddings - 复制phase3逻辑"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Lambdad, ConcatItemsd, DeleteItemsd, ToTensord,
)
from monai.networks.nets import resnet as monai_resnet

# Paths
PHASE2_CROP_LOG = "/home/lichengze/Research/DeepFeature/myresearch/phase2_outputs/crop_log.csv"
CLINICAL_CSV    = "/home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageForEmbeddingDS(Dataset):
    def __init__(self, manifest: pd.DataFrame, indices: np.ndarray, transform: Compose):
        self.df = manifest.iloc[indices].reset_index(drop=True)
        self.tfm = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        d = {"ct": row["ct_path"], "mask": row["mask_path"]}
        d = self.tfm(d)
        d["patient_id"] = row["patient_id"]
        return d


def build_transforms() -> Compose:
    return Compose([
        LoadImaged(keys=["ct","mask"]),
        EnsureChannelFirstd(keys=["ct","mask"]),
        EnsureTyped(keys=["ct","mask"], dtype=torch.float32, track_meta=False),
        Lambdad(keys=["mask"], func=lambda x: (x > 0).float()),
        ConcatItemsd(keys=["ct","mask"], name="image"),
        DeleteItemsd(keys=["ct","mask"]),
        ToTensord(keys=["image"]),
    ])


class ResNet10_ImageOnly_GN(nn.Module):
    def __init__(self, in_channels: int, n_bins: int):
        super().__init__()
        self.backbone = monai_resnet.resnet10(
            spatial_dims=3, n_input_channels=in_channels, num_classes=512
        )
        self._bn_to_gn(self.backbone)
        self.head = nn.Linear(512, n_bins)

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


def build_manifest():
    crop = pd.read_csv(PHASE2_CROP_LOG)
    if "ct_path" not in crop.columns and "out_ct" in crop.columns:
        crop = crop.rename(columns={"out_ct": "ct_path", "out_mask": "mask_path"})
    manifest = crop[["patient_id","ct_path","mask_path"]].copy()

    clin = pd.read_csv(CLINICAL_CSV)
    ren = {}
    for c in clin.columns:
        s = c.lower().strip()
        if "patient" in s or "case" in s:
            ren[c] = "patient_id"
        elif "survival" in s and "time" in s:
            ren[c] = "time"
        elif "event" in s or "dead" in s or "status" in s:
            ren[c] = "event"
    
    clin = clin.rename(columns=ren)
    clin = clin[["patient_id","time","event"]].dropna(subset=["time","event"])
    
    manifest = manifest.merge(clin, on="patient_id", how="inner").copy()
    manifest["time"] = manifest["time"].astype(float)
    manifest["event"] = manifest["event"].astype(int)
    
    return manifest


def extract_for_experiment(exp_dir: Path, seed: int = 42):
    from sklearn.model_selection import StratifiedKFold

    manifest = build_manifest()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    tfm = build_transforms()

    for fold, (idx_tr, idx_va) in enumerate(skf.split(np.zeros(len(manifest)), manifest["event"])):
        print(f"\n=== {exp_dir.name} | Fold {fold} ===")
        fold_dir = exp_dir / f"fold_{fold}"
        ckpt_path = fold_dir / "best.pt"
        
        if not ckpt_path.exists():
            print(f"  Skip: checkpoint not found")
            continue

        # ✅ 修正：PyTorch 2.6+ 需要显式设置 weights_only=False
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        n_bins = ckpt.get("n_time_bins", 15)
        
        model = ResNet10_ImageOnly_GN(in_channels=2, n_bins=n_bins).to(DEVICE)
        
        # 只加载backbone
        state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
        backbone_state = {k.replace("backbone.", ""): v 
                         for k, v in state_dict.items() 
                         if k.startswith("backbone.")}
        model.backbone.load_state_dict(backbone_state, strict=True)
        model.eval()

        # Train embeddings
        ds_tr = ImageForEmbeddingDS(manifest, idx_tr, tfm)
        dl_tr = DataLoader(ds_tr, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

        train_feats, train_pids = [], []
        with torch.no_grad():
            for batch in tqdm(dl_tr, desc="Train", leave=False):
                feat = model.backbone(batch["image"].to(DEVICE))
                train_feats.append(feat.cpu().numpy())
                train_pids.extend(batch["patient_id"])

        train_feats = np.concatenate(train_feats, axis=0)
        np.save(fold_dir / "train_embeddings.npy", train_feats)
        pd.Series(train_pids).to_csv(fold_dir / "train_pids.csv", index=False, header=False)
        print(f"  Train: {train_feats.shape}")

        # Val embeddings - 使用不同文件名避免覆盖phase3训练时保存的版本
        ds_va = ImageForEmbeddingDS(manifest, idx_va, tfm)
        dl_va = DataLoader(ds_va, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

        val_feats, val_pids = [], []
        with torch.no_grad():
            for batch in tqdm(dl_va, desc="Val", leave=False):
                feat = model.backbone(batch["image"].to(DEVICE))
                val_feats.append(feat.cpu().numpy())
                val_pids.extend(batch["patient_id"])

        val_feats = np.concatenate(val_feats, axis=0)
        # 保存为新文件名，不覆盖原始的 val_embeddings.npy
        np.save(fold_dir / "val_embeddings_extracted.npy", val_feats)
        pd.Series(val_pids).to_csv(fold_dir / "val_pids_extracted.csv", index=False, header=False)
        print(f"  Val: {val_feats.shape}")
        
        # 验证与原始文件的一致性（如果原始文件存在）
        orig_emb_path = fold_dir / "val_embeddings.npy"
        if orig_emb_path.exists():
            orig_emb = np.load(orig_emb_path)
            if np.allclose(orig_emb, val_feats, atol=1e-5):
                print(f"  ✅ Val embeddings match original (max diff: {np.abs(orig_emb - val_feats).max():.2e})")
            else:
                print(f"  ⚠️  Val embeddings differ from original (max diff: {np.abs(orig_emb - val_feats).max():.2e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(exp_dir)

    extract_for_experiment(exp_dir)
    print("\nDone!")