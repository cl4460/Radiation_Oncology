#!/usr/bin/env python3
"""
从所有 5 个 fold 的模型中提取 embeddings，然后平均合并成一个 CSV
用于 step2_multimodal_stack_v1.py 的 deep_csv 参数
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConcatItemsd, DeleteItemsd, Lambdad
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import resnet10
import torch.nn as nn


def build_backbone(in_channels=2, embedding_dim=512):
    backbone = resnet10(
        spatial_dims=3,
        n_input_channels=in_channels,
        num_classes=embedding_dim,
    )
    return backbone


def replace_bn_with_gn(module, num_groups=8):
    """替换 BatchNorm 为 GroupNorm，与训练代码一致"""
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
            replace_bn_with_gn(child, num_groups)
    return module


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


def load_ckpt_backbone(backbone, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    filtered = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
    missing, unexpected = backbone.load_state_dict(filtered, strict=False)
    return ckpt, missing, unexpected


def extract_embeddings_from_fold(ckpt_path, records, patient_ids, batch_size=8, num_workers=4, device_str="cuda:0"):
    """从单个 fold 的模型中提取所有患者的 embeddings"""
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
    
    # 加载模型
    backbone = build_backbone(in_channels=2, embedding_dim=512)
    backbone = replace_bn_with_gn(backbone, num_groups=8)  # 与训练代码一致
    ckpt, missing, unexpected = load_ckpt_backbone(backbone, ckpt_path)
    
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    backbone.to(device).eval()
    
    # 提取 embeddings
    all_emb = []
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"Extracting from {Path(ckpt_path).parent.name}"):
            img = batch["image"].to(device)
            emb = backbone(img).detach().cpu().numpy()  # (B,512)
            all_emb.append(emb)
    
    all_emb = np.concatenate(all_emb, axis=0)
    return all_emb, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold_dir", required=True, help="Directory containing fold_0, fold_1, ..., fold_4")
    ap.add_argument("--crop_log_csv", required=True)
    ap.add_argument("--clinical_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    
    # 设置 GPU
    if args.gpu is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device_str = "cuda:0"
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    fold_dir = Path(args.fold_dir)
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
    
    # 加载数据
    crop = pd.read_csv(args.crop_log_csv)
    clin = pd.read_csv(args.clinical_csv)
    
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
    df = crop.merge(clin_renamed[[col for col in ["patient_id", time_col, event_col] if col in clin_renamed.columns]], 
                    on="patient_id", how="inner")
    df = df.dropna(subset=[time_col, event_col]).copy()
    df = df.sort_values("patient_id").reset_index(drop=True)
    
    patient_ids = df["patient_id"].tolist()
    
    # 修复路径
    df["out_ct"] = df["out_ct"].apply(fix_path)
    df["out_edt"] = df["out_edt"].apply(fix_path)
    
    records = []
    for _, r in df.iterrows():
        ct_path = fix_path(r["out_ct"])
        edt_path = fix_path(r["out_edt"])
        records.append({
            "ct": ct_path,
            "edt": edt_path,
        })
    
    print("="*80)
    print(f"从 {args.n_folds} 个 fold 中提取 embeddings")
    print("="*80)
    print(f"总患者数: {len(patient_ids)}")
    print(f"Fold 目录: {fold_dir}")
    print()
    
    # 从每个 fold 提取 embeddings
    all_fold_embeddings = []
    for fold_id in range(args.n_folds):
        fold_path = fold_dir / f"fold_{fold_id}" / "best.pt"
        if not fold_path.exists():
            print(f"⚠️  警告: {fold_path} 不存在，跳过 fold_{fold_id}")
            continue
        
        print(f"处理 fold_{fold_id}...")
        emb, ckpt = extract_embeddings_from_fold(
            fold_path, records, patient_ids, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            device_str=device_str
        )
        all_fold_embeddings.append(emb)
        
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            print(f"  Epoch: {ckpt.get('epoch')}, Uno: {ckpt.get('uno_integral', 'N/A')}")
    
    if len(all_fold_embeddings) == 0:
        raise ValueError("没有找到任何有效的 fold 模型！")
    
    print()
    print("="*80)
    print("合并所有 fold 的 embeddings (平均)")
    print("="*80)
    
    # 平均所有 fold 的 embeddings
    all_fold_embeddings = np.stack(all_fold_embeddings, axis=0)  # (n_folds, n_patients, 512)
    averaged_embeddings = np.mean(all_fold_embeddings, axis=0)  # (n_patients, 512)
    
    print(f"合并后的 embeddings shape: {averaged_embeddings.shape}")
    print(f"使用的 fold 数: {len(all_fold_embeddings)}")
    
    # 保存为 CSV
    out_df = pd.DataFrame(averaged_embeddings, columns=[f"d{i}" for i in range(averaged_embeddings.shape[1])])
    out_df.insert(0, "PatientID", patient_ids)
    out_df.to_csv(args.out_csv, index=False)
    
    print()
    print("="*80)
    print(f"✅ 已保存到: {args.out_csv}")
    print(f"   格式: PatientID, d0, d1, ..., d511")
    print(f"   患者数: {len(patient_ids)}")
    print("="*80)


if __name__ == "__main__":
    main()

