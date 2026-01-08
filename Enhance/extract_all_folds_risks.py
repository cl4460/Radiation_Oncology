#!/usr/bin/env python3
"""
从所有 5 个 fold 的模型中提取所有患者（train/val/test）的 risk 值
然后对 5 个 fold 的 risk 进行平均，生成 deep_risk_ens.csv
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from monai.networks.nets import resnet as monai_resnet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConcatItemsd, DeleteItemsd, Lambdad
)
from monai.data import Dataset, DataLoader

# 配置
CROP_LOG_CSV = "/home/lichengze/Research/Nested_CV/CNN_pipeline/phase2_outputs/crop_log.csv"
CLINICAL_CSV = "/home/lichengze/Research/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
FOLD_DIR = "/home/lichengze/Research/Enhance/phase3_7e4_output/fixed_split_seed42_lr7e-04"
OUTPUT_DIR = FOLD_DIR
N_FOLDS = 5

BATCH_SIZE = 8
NUM_WORKERS = 4
GPU_ID = 1
DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")


def fix_path(path_str):
    """Fix paths that may be missing Nested_CV in the middle."""
    if pd.isna(path_str) or not path_str:
        return path_str
    path_str = str(path_str).strip()
    if not Path(path_str).exists() and "/Research/CNN_pipeline/" in path_str:
        fixed = path_str.replace("/Research/CNN_pipeline/", "/Research/Nested_CV/CNN_pipeline/")
        if Path(fixed).exists():
            return fixed
    return path_str


def _replace_bn_with_gn(module, num_groups=8):
    """Replace BatchNorm with GroupNorm (must match training)"""
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


def build_model(ckpt, device):
    """从 checkpoint 构建完整的模型（与训练代码一致）"""
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("phase3_0617_update", 
                                                  "/home/lichengze/Research/Enhance/phase3_0617_update.py")
    phase3_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase3_module)
    
    n_time_bins = ckpt['n_time_bins']
    clinical_dim = ckpt['clinical_dim']
    
    # 使用训练代码中的模型类
    model = phase3_module.ResNet10_Clinical_GN(
        in_channels=2,
        clinical_dim=clinical_dim,
        n_bins=n_time_bins,
        dropout=0.35
    )
    
    # 加载权重
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.to(device).eval()
    
    return model


@torch.no_grad()
def predict_with_tta4(model, img_batch, clin_batch, device, n_time_bins):
    """4-view TTA prediction（与训练代码一致）"""
    views = [
        img_batch,
        torch.flip(img_batch, dims=[-2]),
        torch.flip(img_batch, dims=[-1]),
        torch.flip(img_batch, dims=[-2, -1]),
    ]
    
    model.eval()
    outs = []
    for v in views:
        logits = model(v.to(device), clin_batch.to(device))
        haz = torch.sigmoid(logits[:, :n_time_bins])
        surv = torch.cumprod(1.0 - haz, dim=1)
        outs.append(surv)
    
    return torch.stack(outs, 0).mean(0).cpu().numpy()


def extract_risk_from_fold(fold_id, model, records, clinical_data, patient_ids, cuts, device, batch_size=8, num_workers=4):
    """从单个 fold 的模型中提取所有患者的 risk"""
    # 构建 transform
    tfm = Compose([
        LoadImaged(keys=["ct", "edt"], image_only=True),
        EnsureChannelFirstd(keys=["ct", "edt"]),
        EnsureTyped(keys=["ct", "edt"], dtype=torch.float32, track_meta=False),
        Lambdad(keys=["ct", "edt"], func=lambda x: torch.nan_to_num(x).clamp_(0, 1)),
        ConcatItemsd(keys=["ct", "edt"], name="image", dim=0),
        DeleteItemsd(keys=["ct", "edt"]),
    ])
    
    ds = Dataset(records, transform=tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    n_time_bins = len(cuts) - 1
    bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
    
    # 提取所有患者的 risk
    all_risks = []
    idx = 0
    
    model.eval()
    for batch in tqdm(dl, desc=f"Fold {fold_id}"):
        img = batch["image"].to(device)
        clin_batch = clinical_data[idx:idx+img.size(0)].to(device)
        
        # 使用 TTA 预测
        surv = predict_with_tta4(model, img, clin_batch, device, n_time_bins)
        
        # 计算 risk: -积分(survival curve)
        risk = -np.trapz(surv, x=bin_mids, axis=1)
        all_risks.append(risk)
        
        idx += img.size(0)
    
    all_risks = np.concatenate(all_risks, axis=0)
    return all_risks


def prepare_clinical_features_all(manifest):
    """准备所有患者的临床特征（与训练时一致）"""
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("phase3_0617_update", 
                                                  "/home/lichengze/Research/Enhance/phase3_0617_update.py")
    phase3_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase3_module)
    
    # 使用所有数据作为 train，然后提取特征（用于计算统计量）
    # 然后用这些统计量处理所有数据
    idx_all = np.arange(len(manifest))
    n = len(idx_all)
    
    # 使用前一半作为 train（用于计算统计量），后一半作为 val（用于验证）
    # 但实际上我们会用 train 的统计量处理所有数据
    dummy_train_idx = idx_all[:max(1, n//2)]
    dummy_val_idx = idx_all[max(1, n//2):] if n > 1 else idx_all
    
    clinical_train, clinical_val, feature_names = phase3_module.prepare_clinical_features_fold(
        manifest, dummy_train_idx, dummy_val_idx
    )
    
    # 合并 train 和 val 的特征
    clinical_all = np.concatenate([clinical_train, clinical_val], axis=0)
    
    return clinical_all, feature_names


def main():
    print("="*80)
    print("从所有 fold 提取 risk 值并生成 ensemble")
    print("="*80)
    print()
    
    # 加载数据
    print("加载数据...")
    crop = pd.read_csv(CROP_LOG_CSV)
    clin = pd.read_csv(CLINICAL_CSV)
    
    # 查找 patient_id 列
    clin_id_col = None
    for col in clin.columns:
        if col.lower() in ['patientid', 'patient_id', 'case']:
            clin_id_col = col
            break
    
    if clin_id_col is None:
        raise ValueError(f"clinical_csv missing patient ID column")
    
    clin_renamed = clin.rename(columns={clin_id_col: "patient_id"})
    
    # 查找生存时间和事件列
    time_col = None
    event_col = None
    for col in clin_renamed.columns:
        col_lower = col.lower()
        if 'survival' in col_lower and 'time' in col_lower:
            time_col = col
        elif 'dead' in col_lower or 'event' in col_lower:
            event_col = col
    
    if time_col is None or event_col is None:
        raise ValueError(f"clinical_csv missing time/event columns")
    
    # 合并数据
    manifest = crop.merge(clin_renamed[[col for col in ["patient_id", time_col, event_col] if col in clin_renamed.columns]], 
                         on="patient_id", how="inner")
    manifest = manifest.dropna(subset=[time_col, event_col]).copy()
    manifest = manifest.sort_values("patient_id").reset_index(drop=True)
    
    patient_ids = manifest["patient_id"].tolist()
    
    # 修复路径
    manifest["out_ct"] = manifest["out_ct"].apply(fix_path)
    manifest["out_edt"] = manifest["out_edt"].apply(fix_path)
    
    # 构建 records
    records = []
    for _, r in manifest.iterrows():
        ct_path = fix_path(r["out_ct"])
        edt_path = fix_path(r["out_edt"])
        records.append({
            "ct": ct_path,
            "edt": edt_path,
        })
    
    print(f"总患者数: {len(patient_ids)}")
    print()
    
    # 准备临床特征（所有患者）
    print("准备临床特征...")
    clinical_all, feature_names = prepare_clinical_features_all(manifest)
    clinical_all_tensor = torch.tensor(clinical_all, dtype=torch.float32)
    print(f"临床特征维度: {clinical_all.shape}")
    print(f"临床特征名称: {feature_names}")
    print()
    
    # 从每个 fold 提取 risk
    all_fold_risks = []
    fold_dir_path = Path(FOLD_DIR)
    
    print("="*80)
    print("从每个 fold 提取 risk...")
    print("="*80)
    
    for fold_id in range(N_FOLDS):
        fold_path = fold_dir_path / f"fold_{fold_id}" / "best.pt"
        
        if not fold_path.exists():
            print(f"⚠️  警告: {fold_path} 不存在，跳过 fold_{fold_id}")
            continue
        
        print(f"\n处理 fold_{fold_id}...")
        
        # 加载 checkpoint
        ckpt = torch.load(fold_path, map_location="cpu", weights_only=False)
        cuts = np.array(ckpt['cuts'])
        n_time_bins = ckpt['n_time_bins']
        clinical_dim = ckpt['clinical_dim']
        
        print(f"  Epoch: {ckpt.get('epoch')}, Uno: {ckpt.get('uno_integral', 'N/A'):.4f}")
        print(f"  Time bins: {n_time_bins}, Clinical dim: {clinical_dim}")
        
        # 构建模型（使用训练代码中的模型类）
        model = build_model(ckpt, DEVICE)
        
        # 提取 risk
        risks = extract_risk_from_fold(
            fold_id, model, records, clinical_all_tensor, patient_ids,
            cuts, DEVICE, BATCH_SIZE, NUM_WORKERS
        )
        
        all_fold_risks.append(risks)
        
        # 保存单个 fold 的 risk CSV
        fold_risk_df = pd.DataFrame({
            "PatientID": patient_ids,
            "risk": risks
        })
        fold_risk_csv = fold_dir_path / f"fold_{fold_id}_risk_all.csv"
        fold_risk_df.to_csv(fold_risk_csv, index=False)
        print(f"  ✅ 已保存: {fold_risk_csv}")
    
    if len(all_fold_risks) == 0:
        raise ValueError("没有找到任何有效的 fold 模型！")
    
    # 对 5 个 fold 的 risk 进行平均
    print()
    print("="*80)
    print("对 5 个 fold 的 risk 进行平均...")
    print("="*80)
    
    all_fold_risks = np.stack(all_fold_risks, axis=0)  # (n_folds, n_patients)
    averaged_risks = np.mean(all_fold_risks, axis=0)  # (n_patients,)
    
    print(f"使用的 fold 数: {len(all_fold_risks)}")
    print(f"患者数: {len(averaged_risks)}")
    print(f"Risk 范围: [{averaged_risks.min():.2f}, {averaged_risks.max():.2f}]")
    
    # 保存 ensemble risk
    ensemble_df = pd.DataFrame({
        "PatientID": patient_ids,
        "risk": averaged_risks
    })
    ensemble_csv = Path(OUTPUT_DIR) / "deep_risk_ens.csv"
    ensemble_df.to_csv(ensemble_csv, index=False)
    
    print()
    print("="*80)
    print(f"✅ 已保存 ensemble risk 到: {ensemble_csv}")
    print(f"   格式: PatientID, risk")
    print(f"   患者数: {len(patient_ids)}")
    print("="*80)


if __name__ == "__main__":
    main()

