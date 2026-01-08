#!/usr/bin/env python3
"""
find_correct_fold_split.py

目标：通过比较checkpoint中保存的val_uno，找到正确的fold划分方式

原理：
1. 尝试不同的seed和排序方式
2. 对每种划分，计算每个fold的验证集Uno
3. 与checkpoint中的val_uno比较，找到匹配的划分

用法：
  python find_correct_fold_split.py \
    --lung1_crop_log /path/to/crop_log.csv \
    --lung1_clinical /path/to/clinical.csv \
    --ckpt_root /path/to/lr_7e-04_29night \
    --gpu 0
"""

import argparse
import os
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

try:
    import phase4_update1_new as p4
except ImportError:
    raise RuntimeError("需要 phase4_update1_new.py")

try:
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_ipcw
except ImportError:
    raise RuntimeError("需要 scikit-survival")


def list_fold_dirs(ckpt_root: str) -> List[str]:
    return sorted([
        os.path.join(ckpt_root, d) for d in os.listdir(ckpt_root)
        if d.startswith("fold_") and os.path.isdir(os.path.join(ckpt_root, d))
    ])


def load_fold_ckpt(fold_dir: str) -> Dict:
    best = os.path.join(fold_dir, "best.pt")
    if not os.path.exists(best):
        raise FileNotFoundError(f"Missing best.pt in {fold_dir}")
    return torch.load(best, map_location="cpu")


def get_ckpt_val_unos(ckpt_root: str) -> List[float]:
    """获取checkpoint中保存的val_uno"""
    fold_dirs = list_fold_dirs(ckpt_root)
    val_unos = []
    for fold_dir in fold_dirs:
        ckpt = load_fold_ckpt(fold_dir)
        # 尝试多种可能的key
        val_uno = (
            ckpt.get("val_uno") or 
            ckpt.get("metrics", {}).get("uno") or
            ckpt.get("metrics", {}).get("val_uno") or
            ckpt.get("uno") or
            0.5
        )
        val_unos.append(float(val_uno))
    return val_unos


def uno_cindex(train_df: pd.DataFrame, test_df: pd.DataFrame, risk: np.ndarray) -> float:
    y_train = Surv.from_arrays(event=train_df["event"].astype(bool).values, time=train_df["time"].values)
    y_test = Surv.from_arrays(event=test_df["event"].astype(bool).values, time=test_df["time"].values)
    return float(concordance_index_ipcw(y_train, y_test, risk)[0])


def compute_risk_for_fold(
    df_val: pd.DataFrame,
    df_train: pd.DataFrame,
    ckpt: Dict,
    device: torch.device,
) -> np.ndarray:
    """快速计算risk（不用TTA，加速搜索）"""
    model = p4.ResNet10_Clinical_GN(
        in_channels=2,
        clinical_dim=int(ckpt.get("clinical_dim", 12)),
        n_bins=int(ckpt.get("n_time_bins", 14)),
        dropout=float(ckpt.get("dropout", 0.35)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cuts = np.array(ckpt["cuts"], dtype=np.float64)
    bin_mids = (cuts[:-1] + cuts[1:]) / 2.0

    use_clin = bool(ckpt.get("use_clinical", True))
    feat_names = ckpt.get("clinical_features", [])
    defaults = ckpt.get("clinical_defaults", {})

    if use_clin and feat_names:
        _, X = p4.prepare_clinical_external(df_train, df_val, feature_names=feat_names, defaults=defaults)
    else:
        X = np.zeros((len(df_val), max(1, int(ckpt.get("clinical_dim", 1)))), dtype=np.float32)

    tfm = p4.build_test_transforms()
    records = [{"ct": r["out_ct"], "edt": r["out_edt"], "patient_id": str(r["patient_id"])} 
               for _, r in df_val.iterrows()]
    ds = p4.ExternalTestDataset(records, X, tfm)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    risks = []
    with torch.no_grad():
        for batch in dl:
            img = batch["image"].to(device)
            clin = batch["clinical"].to(device)
            logits = model(img, clin)
            hazards = torch.sigmoid(logits)
            surv = torch.cumprod(1.0 - hazards, dim=1).cpu().numpy()
            risk = -np.trapz(surv, bin_mids[None, :], axis=1)
            risks.append(risk)

    return np.concatenate(risks, axis=0)


def try_split_configuration(
    df: pd.DataFrame,
    ckpt_root: str,
    seed: int,
    sort_by: Optional[str],
    device: torch.device,
) -> Tuple[List[float], float]:
    """
    尝试一种划分配置，返回每个fold的computed Uno和与ckpt的总差异
    """
    fold_dirs = list_fold_dirs(ckpt_root)
    n_folds = len(fold_dirs)
    
    # 准备数据
    if sort_by:
        df_sorted = df.sort_values(sort_by).reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    computed_unos = []
    ckpt_unos = get_ckpt_val_unos(ckpt_root)
    
    for k, (train_idx, val_idx) in enumerate(skf.split(df_sorted, df_sorted["event"].astype(int))):
        fold_dir = fold_dirs[k]
        ckpt = load_fold_ckpt(fold_dir)
        
        df_val = df_sorted.iloc[val_idx].reset_index(drop=True)
        df_train = df_sorted.iloc[train_idx].reset_index(drop=True)
        
        try:
            risk = compute_risk_for_fold(df_val, df_sorted, ckpt, device)
            uno = uno_cindex(df_sorted, df_val, risk)
        except Exception as e:
            uno = 0.5
        
        computed_unos.append(uno)
    
    # 计算总差异
    total_diff = sum(abs(c - t) for c, t in zip(computed_unos, ckpt_unos))
    
    return computed_unos, total_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lung1_crop_log", required=True)
    ap.add_argument("--lung1_clinical", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 42, 1, 2, 3, 123, 1234, 2024])
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load data
    lung1 = p4.load_manifest(args.lung1_crop_log, args.lung1_clinical, dataset_label="LUNG1")
    print(f"LUNG1: n={len(lung1)}, events={lung1['event'].sum()}")

    # Get checkpoint val_unos for reference
    ckpt_unos = get_ckpt_val_unos(args.ckpt_root)
    print(f"\nCheckpoint val_unos: {[f'{u:.4f}' for u in ckpt_unos]}")
    print(f"Mean: {np.mean(ckpt_unos):.4f}")

    # Try different configurations
    results = []
    
    # 排序方式
    sort_options = [
        ("patient_id", "sorted by patient_id"),
        (None, "original order (no sort)"),
        ("time", "sorted by time"),
    ]
    
    print("\n" + "="*80)
    print("Searching for correct fold split configuration...")
    print("="*80)
    
    for seed in args.seeds:
        for sort_by, sort_desc in sort_options:
            print(f"\n  Testing: seed={seed}, {sort_desc}")
            
            computed_unos, total_diff = try_split_configuration(
                lung1, args.ckpt_root, seed, sort_by, device
            )
            
            print(f"    Computed: {[f'{u:.4f}' for u in computed_unos]}")
            print(f"    Ckpt:     {[f'{u:.4f}' for u in ckpt_unos]}")
            print(f"    Total diff: {total_diff:.4f}")
            
            results.append({
                "seed": seed,
                "sort_by": sort_by,
                "sort_desc": sort_desc,
                "computed_unos": computed_unos,
                "total_diff": total_diff,
            })
            
            # 如果差异很小，可能找到了正确配置
            if total_diff < 0.05:
                print(f"    ✓ POSSIBLE MATCH!")

    # 排序找到最佳配置
    results.sort(key=lambda x: x["total_diff"])
    
    print("\n" + "="*80)
    print("TOP 5 BEST MATCHES:")
    print("="*80)
    
    for i, r in enumerate(results[:5]):
        print(f"\n{i+1}. seed={r['seed']}, {r['sort_desc']}")
        print(f"   Total diff: {r['total_diff']:.4f}")
        print(f"   Computed: {[f'{u:.4f}' for u in r['computed_unos']]}")

    best = results[0]
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION:")
    print(f"{'='*80}")
    print(f"  seed={best['seed']}")
    print(f"  sort_by={best['sort_by']}")
    print(f"  Total diff: {best['total_diff']:.4f}")
    
    if best["total_diff"] > 0.1:
        print(f"\n  ⚠️ WARNING: 最佳匹配的差异仍然较大({best['total_diff']:.4f})")
        print(f"  这可能意味着：")
        print(f"    1. 训练时使用了不同的划分逻辑（不是StratifiedKFold）")
        print(f"    2. 数据在训练时有不同的过滤/处理")
        print(f"    3. 需要检查phase3训练脚本的具体实现")


if __name__ == "__main__":
    main()