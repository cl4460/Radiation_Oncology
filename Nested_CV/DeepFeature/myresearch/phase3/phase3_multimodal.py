"""
Phase 3: Multimodal Survival Prediction (Image + Clinical)
Combining 3D CT/EDT features with clinical covariates (age, gender, overall stage)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import json
from typing import List, Tuple

# MONAI imports
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, Spacingd, ConcatItemsd,
    RandAffined, RandGaussianNoised, RandGaussianSmoothd,
    EnsureTyped
)
import monai.networks.nets as monai_nets

# Import clinical feature preparation
sys.path.append('/home/lichengze/Research/DeepFeature/myresearch')
from clinical_features import prepare_clinical_stage_age_gender

# pycox for survival analysis
from pycox.models.loss import NLLLogistiHazardLoss
from pycox.models import utils as pycox_utils


# ============================================================
# Configuration
# ============================================================
CROP_LOG_CSV = "/home/lichengze/Research/DeepFeature/myresearch/phase2_outputs/crop_log.csv"
CLINICAL_CSV = "/home/lichengze/Research/DeepFeature/myresearch/NSCLC-Radiomics-Lung1_clinical-version3-Oct-2019.csv"
OUTPUT_DIR = "/home/lichengze/Research/DeepFeature/myresearch/phase3_multimodal_outputs"

N_FOLDS = 5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Data Preparation
# ============================================================
def prepare_manifest():
    """合并crop_log和clinical数据"""
    # 读取crop_log
    crop = pd.read_csv(CROP_LOG_CSV)
    
    # 读取clinical
    clinical_raw = pd.read_csv(CLINICAL_CSV)
        clinical_raw = clinical_raw.rename(columns={
        "PatientID": "patient_id",
        "Overall.Stage": "overall_stage",
        "Survival.time": "time",
        "deadstatus.event": "event",
    })
    
    # 列名映射
    col_map = {}
    for col in clinical_raw.columns:
        col_clean = col.lower().strip()
        if "patient" in col_clean or "case" in col_clean:
            col_map[col] = "patient_id"
        elif "survival" in col_clean and "time" in col_clean:
            col_map[col] = "time"
        elif "dead" in col_clean or "event" in col_clean:
            col_map[col] = "event"
        elif "overall" in col_clean and "stage" in col_clean:
            col_map[col] = "overall_stage"
        elif "age" in col_clean and "stage" not in col_clean:
            col_map[col] = "age"
        elif "gender" in col_clean or "sex" in col_clean:
            col_map[col] = "gender"
    
    clinical = clinical_raw.rename(columns=col_map)
    
    # 合并
    manifest = crop[["patient_id", "out_ct", "out_edt"]].rename(columns={
        "out_ct": "ct_path",
        "out_edt": "edt_path",
    })
    manifest = manifest.merge(clinical, on="patient_id", how="inner")
    
    # 清理
    manifest = manifest.dropna(subset=["time", "event"])
    manifest["time"] = manifest["time"].astype(float)
    manifest["event"] = manifest["event"].astype(int)
    manifest = manifest.reset_index(drop=True)
    
    print(f"\n[Data] Total patients after merge: {len(manifest)}")
    print(f"[Data] Columns: {manifest.columns.tolist()}")
    print(f"[Data] Events: {manifest['event'].sum()}/{len(manifest)}")
    
    return manifest


# ============================================================
# Survival Time Discretization (LabTrans)
# ============================================================
def make_time_bins(times: np.ndarray, n_bins: int = 15) -> np.ndarray:
    """创建时间分bin"""
    cuts = np.percentile(times[times > 0], np.linspace(0, 100, n_bins + 1))
    cuts = np.unique(cuts)
    return cuts


def discretize_time(times: np.ndarray, events: np.ndarray, 
                    cuts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """离散化生存时间"""
    y_idx = np.digitize(times, cuts, right=True)
    y_idx = np.clip(y_idx, 0, len(cuts) - 1)
    y_evt = events.astype(np.float32)
    return y_idx, y_evt


# ============================================================
# Dataset
# ============================================================
class SurvivalDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        indices: np.ndarray,
        clinical_mat: np.ndarray,
        y_idx: np.ndarray,
        y_evt: np.ndarray,
        transform: Compose,
    ):
        """
        参数:
            manifest: 完整manifest
            indices: 当前fold的索引
            clinical_mat: 临床特征矩阵 [N, clinical_dim]
            y_idx: 离散化的时间bin索引
            y_evt: 事件指示
            transform: MONAI transforms
        """
        self.df = manifest.iloc[indices].reset_index(drop=True)
        self.clinical = clinical_mat.astype(np.float32)
        self.y_idx = y_idx.astype(np.int64)
        self.y_evt = y_evt.astype(np.float32)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        row = self.df.iloc[i]
        
        data = {
            "ct": row["ct_path"],
            "edt": row["edt_path"],
        }
        data = self.transform(data)
        img = data["image"]  # [2, D, H, W]
        
        clin = torch.from_numpy(self.clinical[i])  # [clinical_dim]
        
        return {
            "image": img,
            "clinical": clin,
            "label_idx": torch.tensor(self.y_idx[i], dtype=torch.long),
            "label_event": torch.tensor(self.y_evt[i], dtype=torch.float32),
        }


# ============================================================
# Model: ResNet10 + Clinical Encoder
# ============================================================
class ResNet10_Clinical_GN(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        clinical_dim: int = 8,
        n_bins: int = 15,
        dropout: float = 0.35,
    ):
        super().__init__()
        
        # 3D ResNet10 backbone for images
        self.backbone = monai_nets.resnet10(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=512,
        )
        
        # Replace BatchNorm with GroupNorm
        self._replace_bn_with_gn(self.backbone)
        
        # Clinical encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )
        
        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_bins),
        )
    
    def _replace_bn_with_gn(self, module, num_groups=8):
        """递归替换BatchNorm为GroupNorm"""
        for name, child in module.named_children():
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                num_channels = child.num_features
                setattr(module, name, nn.GroupNorm(num_groups, num_channels))
            else:
                self._replace_bn_with_gn(child, num_groups)
    
    def forward(self, x_img, x_clinical):
        """
        x_img: [B, 2, D, H, W]
        x_clinical: [B, clinical_dim]
        """
        f_img = self.backbone(x_img)  # [B, 512]
        f_clin = self.clinical_encoder(x_clinical)  # [B, 64]
        f = torch.cat([f_img, f_clin], dim=1)  # [B, 576]
        logits = self.head(f)  # [B, n_bins]
        return logits


# ============================================================
# Transforms
# ============================================================
def get_train_transforms():
    return Compose([
        LoadImaged(keys=["ct", "edt"], ensure_channel_first=False, image_only=True),
        EnsureChannelFirstd(keys=["ct", "edt"], channel_dim="no_channel"),
        Orientationd(keys=["ct", "edt"], axcodes="RAS"),
        Spacingd(keys=["ct", "edt"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"]),
        RandAffined(
            keys=["ct", "edt"],
            prob=0.5,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            mode=["bilinear", "nearest"],
            padding_mode="border",
        ),
        RandGaussianNoised(keys=["ct"], prob=0.3, mean=0.0, std=0.1),
        RandGaussianSmoothd(keys=["ct"], prob=0.3, sigma_x=(0.5, 1.0)),
        ConcatItemsd(keys=["ct", "edt"], name="image"),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])


def get_val_transforms():
    return Compose([
        LoadImaged(keys=["ct", "edt"], ensure_channel_first=False, image_only=True),
        EnsureChannelFirstd(keys=["ct", "edt"], channel_dim="no_channel"),
        Orientationd(keys=["ct", "edt"], axcodes="RAS"),
        Spacingd(keys=["ct", "edt"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"]),
        ConcatItemsd(keys=["ct", "edt"], name="image"),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ])


# ============================================================
# Training & Evaluation
# ============================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        clin = batch["clinical"].to(device, non_blocking=True)
        label_idx = batch["label_idx"].to(device)
        label_event = batch["label_event"].to(device)
        
        logits = model(imgs, clin)
        loss = loss_fn(logits, label_idx, label_event)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_idx, all_evt, all_time = [], [], [], []
    
    for batch in loader:
        imgs = batch["image"].to(device)
        clin = batch["clinical"].to(device)
        label_idx = batch["label_idx"].to(device)
        label_event = batch["label_event"].to(device)
        
        logits = model(imgs, clin)
        loss = loss_fn(logits, label_idx, label_event)
        
        total_loss += loss.item() * imgs.size(0)
        all_logits.append(logits.cpu())
        all_idx.append(label_idx.cpu())
        all_evt.append(label_event.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_idx = torch.cat(all_idx, dim=0)
    all_evt = torch.cat(all_evt, dim=0)
    
    avg_loss = total_loss / len(loader.dataset)
    
    # 计算C-index
    pmf = torch.softmax(all_logits, dim=1).numpy()
    surv = 1 - pmf.cumsum(axis=1)
    risk_scores = -(surv.sum(axis=1))  # 负的期望生存时间作为风险分数
    
    # 使用pycox的concordance
    from pycox.evaluation import EvalSurv
    # 需要真实时间（这里用idx近似）
    durations = all_idx.numpy().astype(float)
    events = all_evt.numpy().astype(bool)
    
    # 简单的concordance计算
    from lifelines.utils import concordance_index
    c_index = concordance_index(durations, -risk_scores, events)
    
    return avg_loss, c_index


# ============================================================
# Main Training Loop
# ============================================================
def main(args):
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 准备数据
    manifest = prepare_manifest()
    print(manifest.columns)
    # 时间离散化
    times = manifest["time"].values
    events = manifest["event"].values
    cuts = make_time_bins(times[events == 1], n_bins=args.n_bins)
    n_time_bins = len(cuts)
    print(f"\n[Time Bins] Number of bins: {n_time_bins}")
    print(f"[Time Bins] Cuts: {cuts[:5]}... (showing first 5)")
    
    # K-Fold交叉验证
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    # Transforms
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()
    
    # 训练每个fold
    fold_results = []
    
    for fold, (idx_train, idx_val) in enumerate(skf.split(np.zeros(len(manifest)), manifest["event"])):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")
        print(f"Train: {len(idx_train)}, Val: {len(idx_val)}")
        
        # 准备临床特征
        Xtr_clin, Xva_clin, clin_names = prepare_clinical_stage_age_gender(
            manifest,
            idx_train=idx_train,
            idx_val=idx_val,
            zscore_age=True,
            verbose=(fold == 0),  # 只在第一个fold打印详细信息
        )
        clinical_dim = Xtr_clin.shape[1]
        print(f"\n[Clinical] Dimension: {clinical_dim}")
        
        # 离散化时间
        y_train_idx, y_train_evt = discretize_time(
            times[idx_train], events[idx_train], cuts
        )
        y_val_idx, y_val_evt = discretize_time(
            times[idx_val], events[idx_val], cuts
        )
        
        # 创建Dataset
        train_ds = SurvivalDataset(
            manifest, idx_train, Xtr_clin, y_train_idx, y_train_evt, train_transforms
        )
        val_ds = SurvivalDataset(
            manifest, idx_val, Xva_clin, y_val_idx, y_val_evt, val_transforms
        )
        
        # DataLoader
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        # 创建模型
        model = ResNet10_Clinical_GN(
            in_channels=2,
            clinical_dim=clinical_dim,
            n_bins=n_time_bins,
            dropout=args.dropout,
        ).to(DEVICE)
        
        print(f"\n[Model] Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        
        # 损失函数
        loss_fn = NLLLogistiHazardLoss()
        
        # 训练
        best_c_index = 0.0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
            val_loss, val_c_index = evaluate(model, val_loader, loss_fn, DEVICE)
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{args.epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val C-index: {val_c_index:.4f}")
            
            # 保存最佳模型
            if val_c_index > best_c_index:
                best_c_index = val_c_index
                best_epoch = epoch + 1
                patience_counter = 0
                
                # 保存checkpoint
                fold_dir = Path(OUTPUT_DIR) / f"fold_{fold}"
                fold_dir.mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'c_index': val_c_index,
                    'clinical_names': clin_names,
                }, fold_dir / "best_model.pt")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"\n[Fold {fold}] Best C-index: {best_c_index:.4f} (epoch {best_epoch})")
        
        fold_results.append({
            'fold': fold,
            'best_c_index': best_c_index,
            'best_epoch': best_epoch,
        })
    
    # 汇总结果
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    
    c_indices = [r['best_c_index'] for r in fold_results]
    print(f"Mean C-index: {np.mean(c_indices):.4f} ± {np.std(c_indices):.4f}")
    
    for r in fold_results:
        print(f"Fold {r['fold']}: {r['best_c_index']:.4f} (epoch {r['best_epoch']})")
    
    # 保存结果
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(Path(OUTPUT_DIR) / "fold_results.csv", index=False)
    
    # 保存配置
    with open(Path(OUTPUT_DIR) / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--patience", type=int, default=20)
    
    # 数据参数
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Multimodal Survival Prediction")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    main(args)