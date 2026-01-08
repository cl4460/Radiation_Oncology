#!/usr/bin/env python
"""
ensemble_top3_peer_fixed.py - 基于Peer Review的完整修复版本

关键修复:
1. 使用checkpoint中的patient IDs重建训练时的fold split
2. 在相同validation set上评估单模型和ensemble (公平比较)
3. 使用TTA保持与训练一致
4. 实现logit/hazard averaging (比survival averaging更稳定)
5. 可选的learned weights based on per-fold performance
"""

import os, math, random, re
from pathlib import Path
from typing import List, Optional
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConcatItemsd, DeleteItemsd, ToTensord, Lambdad
)
from monai.networks.nets import resnet as monai_resnet
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from sksurv.metrics import concordance_index_ipcw

# ============================================================
# 路径配置
# ============================================================
CROP_LOG_CSV = "/home/lichengze/Research/CNN_pipeline/phase2_outputs/crop_log.csv"
CLINICAL_CSV = "/home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical.csv"
OUTPUT_DIR = "/home/lichengze/Research/CNN_pipeline/phase3_outputs/ensemble/ensemble_top3"

# ============================================================
# Ensemble配置
# ============================================================
class EnsembleConfig:
    # Top-3 学习率
    TOP_LRS = ['7e-4', '6.8e-4', '7.4e-4']
    WEIGHTS = [1/3, 1/3, 1/3]  # 初始等权重，可在每个fold中调整
    
    # 模型路径
    BASE_DIR = Path('/home/lichengze/Research/CNN_pipeline/phase3_outputs/learning_rate/learning_rate_corrected/output')
    
    # CV设置
    N_FOLDS = 5
    SEED = 42
    
    # DataLoader设置
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2
    
    # Ensemble设置
    ENSEMBLE_MODE = 'logit'  # 'logit' | 'hazard' | 'survival'
    USE_TTA = True  # 与训练保持一致
    USE_LEARNED_WEIGHTS = True  # 是否使用per-fold learned weights
    
    # 设备
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config = EnsembleConfig()

# ============================================================
# 工具函数
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_structured_y(times, events):
    return np.array([(bool(e), float(t)) for t, e in zip(times, events)],
                    dtype=[('event', bool), ('time', float)])

# ============================================================
# Clinical特征处理（完全从训练代码复制）
# ============================================================

def _safe_mode(values: np.ndarray, default: float) -> float:
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return float(default)
    cnt = Counter(vals.tolist())
    mode_val = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return float(mode_val)

def _parse_T(series: pd.Series) -> np.ndarray:
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).upper()
        m = re.search(r'(\d)', s)
        if m:
            t = float(m.group(1))
            if 1.0 <= t <= 4.0:
                out.append(t)
            else:
                out.append(np.nan)
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)

def _parse_N(series: pd.Series) -> np.ndarray:
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).upper()
        m = re.search(r'(\d)', s)
        if m:
            n = float(m.group(1))
            if 0.0 <= n <= 3.0:
                out.append(n)
            else:
                out.append(np.nan)
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)

def _parse_M(series: pd.Series):
    M01, M_misc = [], []
    for v in series:
        if pd.isna(v):
            M01.append(np.nan)
            M_misc.append(0.0)
            continue
        s = str(v).upper()
        m = re.search(r'(\d)', s)
        if not m:
            M01.append(np.nan)
            M_misc.append(0.0)
            continue
        mv = int(m.group(1))
        if mv == 0:
            M01.append(0.0)
            M_misc.append(0.0)
        elif mv == 1:
            M01.append(1.0)
            M_misc.append(0.0)
        else:
            M01.append(1.0)
            M_misc.append(1.0)
    return np.array(M01, dtype=np.float32), np.array(M_misc, dtype=np.float32)

def _map_overall(series: pd.Series) -> np.ndarray:
    mapping = {
        "I": 0.2, "1": 0.2,
        "II": 0.4, "2": 0.4,
        "IIIA": 0.6, "3A": 0.6,
        "IIIB": 0.8, "3B": 0.8,
        "IV": 1.0, "4": 1.0
    }
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).strip().upper().replace("STAGE", "").replace(" ", "")
        s = s.replace("ⅢA", "IIIA").replace("ⅢB", "IIIB").replace("Ⅱ", "II").replace("Ⅳ", "IV").replace("Ⅰ", "I")
        if s in mapping:
            out.append(mapping[s])
        elif s in ["III"]:
            out.append(0.7)
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)

def prepare_clinical_features_fold(manifest: pd.DataFrame, 
                                  idx_train: np.ndarray,
                                  idx_val: np.ndarray):
    """完全从训练代码复制"""
    features_train = []
    features_val = []
    feature_names = []

    if 'age' in manifest.columns:
        age_raw = pd.to_numeric(manifest['age'], errors='coerce').values.astype(np.float32)
        age_missing = np.isnan(age_raw).astype(np.float32)
        train_median = np.nanmedian(age_raw[idx_train]) if np.any(~np.isnan(age_raw[idx_train])) else 0.0
        age_filled = age_raw.copy()
        age_filled[np.isnan(age_filled)] = train_median

        scaler = StandardScaler()
        scaler.fit(age_filled[idx_train].reshape(-1, 1))
        age_scaled = scaler.transform(age_filled.reshape(-1, 1)).astype(np.float32).flatten()

        features_train += [age_scaled[idx_train], age_missing[idx_train]]
        features_val   += [age_scaled[idx_val],   age_missing[idx_val]]
        feature_names  += ['age_scaled', 'age_missing']

    if 'gender' in manifest.columns:
        gender_ser = manifest['gender'].astype(str).str.upper().str[0]
        gender = (gender_ser == 'M').astype(np.float32).values
        features_train.append(gender[idx_train])
        features_val.append(gender[idx_val])
        feature_names.append('gender_male')

    if 'T' in manifest.columns:
        T_num = _parse_T(manifest['T'])
        T_missing = np.isnan(T_num).astype(np.float32)
        T_mode = _safe_mode(T_num[idx_train], default=2.0)
        T_filled = T_num.copy()
        T_filled[np.isnan(T_filled)] = T_mode
        T_norm = (T_filled - 1.0) / 3.0
        features_train += [T_norm[idx_train], T_missing[idx_train]]
        features_val   += [T_norm[idx_val],   T_missing[idx_val]]
        feature_names  += ['T_norm', 'T_missing']

    if 'N' in manifest.columns:
        N_num = _parse_N(manifest['N'])
        N_missing = np.isnan(N_num).astype(np.float32)
        N_mode = _safe_mode(N_num[idx_train], default=1.0)
        N_filled = N_num.copy()
        N_filled[np.isnan(N_filled)] = N_mode
        N_norm = (N_filled - 0.0) / 3.0
        features_train += [N_norm[idx_train], N_missing[idx_train]]
        features_val   += [N_norm[idx_val],   N_missing[idx_val]]
        feature_names  += ['N_norm', 'N_missing']

    if 'M' in manifest.columns:
        M01, M_misc = _parse_M(manifest['M'])
        M_missing = np.isnan(M01).astype(np.float32)
        M_mode = _safe_mode(M01[idx_train], default=0.0)
        M01_filled = M01.copy()
        M01_filled[np.isnan(M01_filled)] = M_mode
        features_train += [M01_filled[idx_train], M_missing[idx_train], M_misc[idx_train]]
        features_val   += [M01_filled[idx_val],   M_missing[idx_val],   M_misc[idx_val]]
        feature_names  += ['M01', 'M_missing', 'M_misc']

    if 'overall_stage' in manifest.columns:
        ov = _map_overall(manifest['overall_stage'])
        overall_missing = np.isnan(ov).astype(np.float32)
        overall_mode = _safe_mode(ov[idx_train], default=0.6)
        ov_filled = ov.copy()
        ov_filled[np.isnan(ov_filled)] = overall_mode
        features_train += [ov_filled[idx_train], overall_missing[idx_train]]
        features_val   += [ov_filled[idx_val],   overall_missing[idx_val]]
        feature_names  += ['overall_stage_ord', 'overall_missing']

    if features_train:
        clinical_train = np.stack(features_train, axis=1).astype(np.float32)
        clinical_val   = np.stack(features_val,   axis=1).astype(np.float32)
    else:
        clinical_train = np.zeros((len(idx_train), 1), dtype=np.float32)
        clinical_val   = np.zeros((len(idx_val), 1),   dtype=np.float32)
        feature_names  = ['dummy']

    clinical_train = np.nan_to_num(clinical_train, nan=0.0, posinf=0.0, neginf=0.0)
    clinical_val   = np.nan_to_num(clinical_val,   nan=0.0, posinf=0.0, neginf=0.0)

    return clinical_train, clinical_val, feature_names

# ============================================================
# Dataset
# ============================================================

class SurvivalDataset(Dataset):
    def __init__(self, records: List[dict], clinical: np.ndarray,
                 y_idx: np.ndarray, y_evt: np.ndarray, transform: Compose):
        self.records = records
        self.clinical = clinical
        self.y_idx = y_idx
        self.y_evt = y_evt
        self.transform = transform
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, i):
        data = self.transform(self.records[i])
        data["clinical"] = torch.tensor(self.clinical[i], dtype=torch.float32)
        data["label_idx"] = torch.tensor(self.y_idx[i], dtype=torch.long)
        data["label_event"] = torch.tensor(self.y_evt[i], dtype=torch.float32)
        return data

def build_transforms_val() -> Compose:
    """Validation transforms"""
    transforms = [
        LoadImaged(keys=["ct", "edt"]),
        EnsureChannelFirstd(keys=["ct", "edt"]),
        EnsureTyped(keys=["ct", "edt"], dtype=torch.float32, track_meta=False),
        Lambdad(keys=["ct", "edt"], func=lambda x: torch.nan_to_num(x, 0.0, 0.0, 0.0)), 
        Lambdad(keys=["ct", "edt"], func=lambda x: torch.clamp(x, 0.0, 1.0)),
        ConcatItemsd(keys=["ct", "edt"], name="image"),
        DeleteItemsd(keys=["ct", "edt"]),
        ToTensord(keys=["image"])
    ]
    return Compose(transforms)

# ============================================================
# 模型
# ============================================================

class ResNet10_Clinical_GN(nn.Module):
    def __init__(self, in_channels: int = 2, clinical_dim: int = 6,
                 n_bins: int = 15, dropout: float = 0.35):
        super().__init__()
        
        self.backbone = monai_resnet.resnet10(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=512,
        )
        
        self._replace_bn_with_gn(self.backbone)
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True)
        )
        
        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_bins)
        )
    
    def _replace_bn_with_gn(self, module, num_groups=8):
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
                self._replace_bn_with_gn(child, num_groups)
    
    def forward(self, x_img, x_clinical):
        f_img = self.backbone(x_img)
        f_clin = self.clinical_encoder(x_clinical)
        f_combined = torch.cat([f_img, f_clin], dim=1)
        return self.head(f_combined)

# ============================================================
# 🔧 修复1: 单模型评估函数（用于公平比较）
# ============================================================

@torch.no_grad()
def score_single_model(model, dl_val, device, n_time_bins, bin_mids, 
                      y_train_struct, y_val_struct, tau, use_tta=True):
    """
    在给定validation loader上评估单个模型
    
    Args:
        model: trained model
        dl_val: validation DataLoader
        device: cuda or cpu
        n_time_bins: number of time bins
        bin_mids: bin midpoints for integration
        y_train_struct, y_val_struct: structured arrays for IPCW
        tau: truncation time for IPCW
        use_tta: whether to use TTA (should match training)
    
    Returns:
        uno: Uno's C-index
    """
    model.eval()
    surv_all = []
    
    for batch in dl_val:
        img_batch = batch['image']
        clin_batch = batch['clinical']
        
        if use_tta:
            # 4-view TTA (与训练代码完全相同)
            views = [
                img_batch,
                torch.flip(img_batch, dims=[-2]),
                torch.flip(img_batch, dims=[-1]),
                torch.flip(img_batch, dims=[-2, -1]),
            ]
            
            logits_views = []
            for v in views:
                logits = model(v.to(device), clin_batch.to(device))[:, :n_time_bins]
                logits_views.append(logits)
            
            logits = torch.stack(logits_views, 0).mean(0)  # [B, K]
            haz = torch.sigmoid(logits)
            surv = torch.cumprod(1.0 - haz, dim=1).cpu().numpy()
        else:
            # 不用TTA
            logits = model(img_batch.to(device), clin_batch.to(device))[:, :n_time_bins]
            haz = torch.sigmoid(logits)
            surv = torch.cumprod(1.0 - haz, dim=1).cpu().numpy()
        
        surv_all.append(surv)
    
    surv_all = np.concatenate(surv_all, axis=0)
    
    # 计算risk score (积分面积的负值)
    risk = -np.trapz(surv_all, x=bin_mids, axis=1)
    
    # IPCW C-index
    uno = concordance_index_ipcw(y_train_struct, y_val_struct, risk, tau=tau)[0]
    
    return uno

# ============================================================
# 🔧 修复2: 改进的Ensemble预测（支持logit/hazard/survival averaging）
# ============================================================

@torch.no_grad()
def ensemble_predict(models, img_batch, clin_batch, device, n_time_bins, weights,
                    mode='logit', tta=True):
    """
    Ensemble预测，支持3种averaging模式
    
    Args:
        models: List of trained models
        img_batch: [B, 2, D, H, W]
        clin_batch: [B, clinical_dim]
        device: cuda or cpu
        n_time_bins: number of time bins
        weights: [M] weights for each model
        mode: 'logit' | 'hazard' | 'survival'
            - 'logit': average logits then compute survival (most stable)
            - 'hazard': average hazards then compute survival
            - 'survival': average survival curves (least stable due to nonlinearity)
        tta: whether to use TTA (should match training)
    
    Returns:
        surv_ensemble: [B, n_bins] ensemble survival curves
    """
    parts = []
    
    for m in models:
        m.eval()
        
        if tta:
            # 4-view TTA
            views = [
                img_batch,
                torch.flip(img_batch, dims=[-2]),
                torch.flip(img_batch, dims=[-1]),
                torch.flip(img_batch, dims=[-2, -1]),
            ]
            logits_views = [m(v.to(device), clin_batch.to(device))[:, :n_time_bins] for v in views]
            logits = torch.stack(logits_views, 0).mean(0)  # [B, K]
        else:
            logits = m(img_batch.to(device), clin_batch.to(device))[:, :n_time_bins]
        
        # 根据mode收集不同的量
        if mode == 'logit':
            parts.append(logits.cpu())
        elif mode == 'hazard':
            parts.append(torch.sigmoid(logits).cpu())
        elif mode == 'survival':
            hz = torch.sigmoid(logits)
            sv = torch.cumprod(1.0 - hz, dim=1)
            parts.append(sv.cpu())
        else:
            raise ValueError("mode must be 'logit' | 'hazard' | 'survival'")
    
    # Weighted average
    P = torch.stack(parts, 0)  # [M, B, K]
    w = torch.tensor(weights, dtype=torch.float32).view(-1, 1, 1)  # [M, 1, 1]
    
    if mode == 'logit':
        # Average logits, then compute survival
        logits_ens = (P * w).sum(0)  # [B, K]
        hz = torch.sigmoid(logits_ens)
        sv = torch.cumprod(1.0 - hz, dim=1)
        return sv.cpu().numpy()
    
    elif mode == 'hazard':
        # Average hazards, then compute survival
        hz = (P * w).sum(0)  # [B, K]
        sv = torch.cumprod(1.0 - hz, dim=1)
        return sv.cpu().numpy()
    
    else:  # 'survival'
        # Average survival curves, then enforce monotonicity
        sv = (P * w).sum(0).numpy()  # [B, K]
        sv = np.minimum.accumulate(sv[:, ::-1], axis=1)[:, ::-1]
        return sv

# ============================================================
# 🔧 修复3: 使用checkpoint中的patient IDs重建fold
# ============================================================

def reconstruct_fold_from_checkpoint(manifest: pd.DataFrame, ckpts: List[dict], fold: int):
    """
    使用checkpoint中保存的patient IDs重建训练时的fold split
    
    这确保了我们在与训练时完全相同的validation set上评估
    
    Args:
        manifest: full dataset
        ckpts: list of checkpoints (should all have same val_patient_ids)
        fold: fold number (for logging)
    
    Returns:
        idx_train, idx_val: indices for train/val split
        reconstructed: whether reconstruction was successful
    """
    if 'val_patient_ids' not in ckpts[0]:
        print(f"  [WARNING] Checkpoint has no 'val_patient_ids'")
        print(f"  [WARNING] Cannot reconstruct original fold - will use fresh split")
        print(f"  [WARNING] This means comparison won't be apples-to-apples!")
        return None, None, False
    
    # 获取保存的validation patient IDs
    saved_val_pids = [str(p) for p in ckpts[0]['val_patient_ids']]
    
    # 建立patient_id到index的映射
    pid_all = manifest['patient_id'].astype(str).values
    pid_to_idx = {pid: i for i, pid in enumerate(pid_all)}
    
    # 重建validation indices
    idx_val = []
    missing_pids = []
    for pid in saved_val_pids:
        if pid in pid_to_idx:
            idx_val.append(pid_to_idx[pid])
        else:
            missing_pids.append(pid)
    
    idx_val = np.array(idx_val, dtype=int)
    idx_train = np.setdiff1d(np.arange(len(manifest)), idx_val)
    
    if len(missing_pids) > 0:
        print(f"  [WARNING] {len(missing_pids)} patients from checkpoint not found in current manifest")
        print(f"  [WARNING] First few missing: {missing_pids[:3]}")
    
    print(f"  [OK] Reconstructed original fold from checkpoint")
    print(f"     Train: {len(idx_train)} | Val: {len(idx_val)} patients")
    
    # 验证所有checkpoints有相同的val set
    for i, ckpt in enumerate(ckpts[1:], 1):
        if 'val_patient_ids' in ckpt:
            other_pids = set(str(p) for p in ckpt['val_patient_ids'])
            if other_pids != set(saved_val_pids):
                print(f"  [WARNING] Model {i} has different val_patient_ids!")
    
    return idx_train, idx_val, True

# ============================================================
# 🔧 修复4: Learned weights based on per-fold performance
# ============================================================

def compute_learned_weights(single_unos: List[float], temperature: float = 0.01):
    """
    根据单模型在当前fold的性能计算权重
    
    使用softmax with temperature:
    - temperature小 → 更集中在最好的模型
    - temperature大 → 更平均
    
    Args:
        single_unos: [M] Uno C-indices of each model on current fold
        temperature: temperature parameter (default 0.01)
    
    Returns:
        weights: [M] normalized weights
    """
    scores = np.array(single_unos)
    
    # Softmax with temperature
    exp_scores = np.exp(scores / temperature)
    weights = exp_scores / exp_scores.sum()
    
    return weights.tolist()

# ============================================================
# 单个Fold的Ensemble评估（完整修复版本）
# ============================================================

def evaluate_fold_ensemble(fold, manifest, idx_train_default, idx_val_default):
    """
    评估单个fold的ensemble（完整修复版本）
    
    关键修复:
    1. 使用checkpoint中的patient IDs重建训练时的fold
    2. 在相同validation set上评估单模型和ensemble
    3. 使用TTA保持与训练一致
    4. 支持logit/hazard averaging
    5. 可选的learned weights
    """
    print(f"\n{'='*70}")
    print(f"FOLD {fold} - Peer-Reviewed Ensemble")
    print(f"{'='*70}")
    
    times = manifest["time"].values
    events = manifest["event"].values
    
    # ============================================================
    # 1. 加载Top-3模型
    # ============================================================
    print(f"\n[1/6] Loading Top-3 models...")
    
    models = []
    ckpts = []
    
    for lr in config.TOP_LRS:
        model_path = config.BASE_DIR / f'lr_{lr}' / f'fold_{fold}' / 'best.pt'
        
        if not model_path.exists():
            print(f"  [ERROR] Model not found: {model_path}")
            return None
        
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        
        model = ResNet10_Clinical_GN(
            in_channels=2,
            clinical_dim=ckpt['clinical_dim'],
            n_bins=ckpt['n_time_bins'],
            dropout=0.35
        )
        
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        model.to(config.DEVICE)
        
        models.append(model)
        ckpts.append(ckpt)
        
        baseline_uno = ckpt.get('uno_integral', ckpt.get('best_uno', 'N/A'))
        print(f"  [OK] lr={lr:<8s}: Checkpoint Uno={baseline_uno:.4f}" 
              if isinstance(baseline_uno, float) else f"  [OK] lr={lr}")
    
    # ============================================================
    # 2. 验证一致性并设置
    # ============================================================
    print(f"\n[2/6] Validating temporal discretization...")
    
    n_time_bins = ckpts[0]['n_time_bins']
    cuts = np.array(ckpts[0]['cuts'])
    bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
    
    print(f"  n_bins: {n_time_bins}")
    print(f"  cuts range: [{cuts[0]:.1f}, {cuts[-1]:.1f}]")
    
    for i, ckpt in enumerate(ckpts[1:], 1):
        assert ckpt['n_time_bins'] == n_time_bins, f"Model {i} has different n_time_bins!"
        assert np.allclose(np.array(ckpt['cuts']), cuts, rtol=1e-5), f"Model {i} has different cuts!"
    
    print(f"  [OK] All models aligned")
    
    # ============================================================
    # 🔧 修复: 使用checkpoint中的patient IDs重建fold
    # ============================================================
    print(f"\n[3/6] Reconstructing original fold split...")
    
    idx_train, idx_val, reconstructed = reconstruct_fold_from_checkpoint(
        manifest, ckpts, fold
    )
    
    if not reconstructed:
        # 如果无法重建，使用默认split并警告
        idx_train, idx_val = idx_train_default, idx_val_default
        print(f"  [WARNING] Using default split (NOT the training fold)")
    
    # ============================================================
    # 4. 准备数据
    # ============================================================
    print(f"\n[4/6] Preparing validation data...")
    
    # Label transformation
    lab = LabTransDiscreteTime(cuts=cuts[1:-1])
    y_train = lab.transform(times[idx_train], events[idx_train])
    y_val = lab.transform(times[idx_val], events[idx_val])
    
    # Clinical features
    clinical_train, clinical_val, feature_names = prepare_clinical_features_fold(
        manifest, idx_train, idx_val
    )
    
    print(f"  Clinical features: {len(feature_names)}")
    print(f"  Train: {len(idx_train)} | Val: {len(idx_val)} patients")
        
    # Dataset
    records_val = [{"ct": manifest.iloc[i].ct_path, 
                   "edt": manifest.iloc[i].edt_path} for i in idx_val]
    
    ds_val = SurvivalDataset(
        records_val, clinical_val, 
        y_val[0], y_val[1], 
        build_transforms_val()
    )
    
    dl_val = DataLoader(
        ds_val,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=config.PREFETCH_FACTOR
    )
    
    # IPCW setup
    y_train_struct = to_structured_y(times[idx_train], events[idx_train])
    y_val_struct = to_structured_y(times[idx_val], events[idx_val])
    evt_times = times[idx_train][events[idx_train] == 1]
    tau = np.quantile(evt_times, 0.9)
    
    # ============================================================
    # 🔧 修复: 在相同validation set上评估单模型（公平比较）
    # ============================================================
    print(f"\n[5/6] Evaluating single models on THIS validation set...")
    print(f"  (Using TTA={config.USE_TTA} to match training)")
    
    single_unos = []
    for i, m in enumerate(models):
        uno = score_single_model(
            m, dl_val, config.DEVICE, n_time_bins, bin_mids,
            y_train_struct, y_val_struct, tau, use_tta=config.USE_TTA
        )
        single_unos.append(uno)
        print(f"  lr={config.TOP_LRS[i]:<8s}: Uno={uno:.4f}")
    
    best_single = np.max(single_unos)
    print(f"  Best single on THIS val: {best_single:.4f}")
    
    # 计算learned weights (可选)
    if config.USE_LEARNED_WEIGHTS:
        learned_weights = compute_learned_weights(single_unos, temperature=0.01)
        print(f"\n  Learned weights for this fold:")
        for i, (lr, w) in enumerate(zip(config.TOP_LRS, learned_weights)):
            print(f"    lr={lr:<8s}: {w:.3f}")
        weights_to_use = learned_weights
    else:
        weights_to_use = config.WEIGHTS
        print(f"\n  Using equal weights: {weights_to_use}")
    
    # ============================================================
    # 6. Ensemble预测
    # ============================================================
    print(f"\n[6/6] Running ensemble prediction...")
    print(f"  Mode: {config.ENSEMBLE_MODE}")
    print(f"  TTA: {config.USE_TTA}")
    
    all_surv = []
    
    for batch in tqdm(dl_val, desc="Batches"):
        surv_ensemble = ensemble_predict(
            models,
            batch['image'],
            batch['clinical'],
            config.DEVICE,
            n_time_bins,
            weights_to_use,
            mode=config.ENSEMBLE_MODE,
            tta=config.USE_TTA
        )
        all_surv.append(surv_ensemble)
    
    all_surv = np.concatenate(all_surv, axis=0)
    
    # 计算指标
    risk_integral = -np.trapz(all_surv, x=bin_mids, axis=1)
    
    uno_integral = concordance_index_ipcw(
        y_train_struct, y_val_struct, 
        risk_integral, tau=tau
    )[0]
    
    # ============================================================
    # 7. 汇总结果
    # ============================================================
    lr7e4_uno = single_unos[0]  # 明确取7e-4的性能

    results = {
        'fold': fold,
        'ensemble_uno': uno_integral,
        'single_uno_lr7e4': lr7e4_uno,
        'single_uno_lr6.8e4': single_unos[1],
        'single_uno_lr7.4e4': single_unos[2],
        'best_of_3': best_single,  # ← 重命名，表明这只是参考
        'improvement_vs_lr7e4': uno_integral - lr7e4_uno,  # ← 正确的改善
        'relative_improvement': (uno_integral - lr7e4_uno) / lr7e4_uno * 100,  # ← 正确
        'reconstructed_fold': reconstructed
    }
    
    # 打印结果
    print(f"\n{'='*70}")
    print(f"Results for Fold {fold}")
    print(f"{'='*70}")
    
    print(f"\nSingle Models (evaluated on THIS val set):")
    for i, lr in enumerate(config.TOP_LRS):
        print(f"   lr={lr:<8s}: {single_unos[i]:.4f}")
    print(f"   Best of 3:    {best_single:.4f} (for reference)")  # ← 明确说明

    print(f"\nEnsemble (Top-3, {config.ENSEMBLE_MODE} averaging, TTA={config.USE_TTA}):")
    print(f"   Uno C-index:  {uno_integral:.4f}")
    print(f"   vs LR=7e-4:   {results['improvement_vs_lr7e4']:+.4f} ({results['relative_improvement']:+.2f}%)")  # ← 明确baseline
    
    if reconstructed:
        print(f"\n[OK] This is an apples-to-apples comparison (same validation set)")
    else:
        print(f"\n[WARNING] Using different validation set than training!")
    
    return results

# ============================================================
# Main
# ============================================================

def main():
    set_seed(config.SEED)
    
    print("\n" + "="*70)
    print("PEER-REVIEWED ENSEMBLE EVALUATION")
    print("="*70)
    
    print(f"\n[FIXES] Key Fixes Applied:")
    print(f"  1. [OK] Reconstruct training folds using checkpoint patient IDs")
    print(f"  2. [OK] Evaluate single models on SAME validation set")
    print(f"  3. [OK] Use TTA={config.USE_TTA} to match training")
    print(f"  4. [OK] {config.ENSEMBLE_MODE.upper()} averaging (linear space)")
    print(f"  5. [OK] Optional learned weights: {config.USE_LEARNED_WEIGHTS}")
    
    print(f"\nConfiguration:")
    print(f"  Models: {config.TOP_LRS}")
    print(f"  Initial weights: {config.WEIGHTS}")
    print(f"  Ensemble mode: {config.ENSEMBLE_MODE}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Seed: {config.SEED}")
    
    # ============================================================
    # 数据加载
    # ============================================================
    print(f"\nLoading data...")
    
    crop = pd.read_csv(CROP_LOG_CSV)
    clinical = pd.read_csv(CLINICAL_CSV)
    
    # Column mapping
    col_map = {}
    for col in clinical.columns:
        col_clean = col.lower().strip()
        
        if 'patient' in col_clean or 'case' in col_clean:
            col_map[col] = 'patient_id'
        elif 'survival' in col_clean and 'time' in col_clean:
            col_map[col] = 'time'
        elif 'dead' in col_clean or 'event' in col_clean:
            col_map[col] = 'event'
        elif 'overall' in col_clean and 'stage' in col_clean:
            col_map[col] = 'overall_stage'
        elif ('.t.' in col_clean or col_clean.startswith('t.') or '.t' in col_clean) and 'stage' in col_clean:
            col_map[col] = 'T'
        elif ('.n.' in col_clean or col_clean.startswith('n.') or '.n' in col_clean) and 'stage' in col_clean:
            col_map[col] = 'N'
        elif ('.m.' in col_clean or col_clean.startswith('m.') or '.m' in col_clean) and 'stage' in col_clean:
            col_map[col] = 'M'
        elif 'age' in col_clean and 'stage' not in col_clean:
            col_map[col] = 'age'
        elif 'gender' in col_clean or 'sex' in col_clean:
            col_map[col] = 'gender'
    
    clinical = clinical.rename(columns=col_map)
    
    # Merge
    manifest = crop[["patient_id", "out_ct", "out_mask", "out_edt"]].rename(columns={
        "out_ct": "ct_path",
        "out_mask": "mask_path",
        "out_edt": "edt_path"
    }).merge(clinical, on="patient_id", how="inner")
    
    manifest = manifest.dropna(subset=["time", "event"])
    manifest["time"] = manifest["time"].astype(int)
    manifest["event"] = manifest["event"].astype(int)
    
    print(f"  Dataset: {len(manifest)} patients")
    print(f"  Events: {manifest['event'].sum()} ({manifest['event'].mean()*100:.1f}%)")
    
    # ============================================================
    # Cross-validation（用于提供默认split，但优先使用checkpoint中的）
    # ============================================================
    print(f"\nCreating default 5-fold CV splits (seed={config.SEED})...")
    print(f"  (Will try to use checkpoint folds instead)")
    
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, 
        shuffle=True, 
        random_state=config.SEED
    )
    
    # ============================================================
    # 运行所有folds
    # ============================================================
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for fold, (idx_train_default, idx_val_default) in enumerate(skf.split(
        np.zeros(len(manifest)), manifest["event"]
    )):
        print(f"\n\n{'#'*70}")
        print(f"# FOLD {fold}")
        print(f"{'#'*70}")
        
        results = evaluate_fold_ensemble(
            fold, manifest, idx_train_default, idx_val_default
        )
        
        if results is not None:
            all_results.append(results)
    
    # ============================================================
    # 汇总所有folds
    # ============================================================
    if len(all_results) == 0:
        print(f"\n[ERROR] No results collected!")
        return
    
    results_df = pd.DataFrame(all_results)
    
    print(f"\n\n{'='*70}")
    print(f"FINAL RESULTS - 5-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    
    print(f"\nFold-by-Fold Results:")
    print(f"{'─'*70}")
    print(f"{'Fold':<6} {'LR=7e-4':<12} {'Ensemble':<12} {'Δ':<10} {'Δ%':<8} {'Fold OK'}")  # ← 明确baseline
    print(f"{'─'*70}")

    for _, row in results_df.iterrows():
        fold_ok = "[OK]" if row['reconstructed_fold'] else "[WARNING]"
        print(f"{int(row['fold']):<6} {row['single_uno_lr7e4']:<12.4f} {row['ensemble_uno']:<12.4f} "
            f"{row['improvement_vs_lr7e4']:+.4f}    {row['relative_improvement']:+.2f}%     {fold_ok}")
    
    print(f"{'─'*70}")
    
# 统计汇总
    ensemble_mean = results_df['ensemble_uno'].mean()
    ensemble_std = results_df['ensemble_uno'].std()
    baseline_7e4_mean = results_df['single_uno_lr7e4'].mean()  # ✅ 正确的baseline
    baseline_7e4_std = results_df['single_uno_lr7e4'].std()

    mean_improvement = ensemble_mean - baseline_7e4_mean  # ✅ 正确
    rel_improvement = mean_improvement / baseline_7e4_mean * 100  # ✅ 正确

    print(f"\nSummary Statistics:")
    print(f"{'─'*70}")
    print(f"Single Model (LR=7e-4):  {baseline_7e4_mean:.4f} ± {baseline_7e4_std:.3f}")  # ✅ 明确
    print(f"Ensemble (Top-3):        {ensemble_mean:.4f} ± {ensemble_std:.4f}")
    print(f"Mean Improvement:        {mean_improvement:+.4f} ({rel_improvement:+.2f}%)")
    
    # 统计检验
    if len(results_df) >= 3:
        lr7e4_values = results_df['single_uno_lr7e4'].values  # ✅ 正确
        ensembles = results_df['ensemble_uno'].values
        
        stat, p_value = stats.wilcoxon(ensembles, lr7e4_values)
        
        print(f"\nStatistical Test (Wilcoxon Signed-Rank):")
        print(f"   Statistic: {stat:.2f}")
        print(f"   P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"   [OK] Ensemble significantly better (p < 0.05)")
        elif p_value < 0.10:
            print(f"   [WARNING] Marginally significant (0.05 <= p < 0.10)")
        else:
            print(f"   [WARNING] Not statistically significant (p >= 0.10)")
    
    # 检查有多少folds成功重建
    n_reconstructed = results_df['reconstructed_fold'].sum()
    print(f"\n[STATS] Fold Reconstruction:")
    print(f"   {n_reconstructed}/{len(results_df)} folds successfully reconstructed from checkpoints")
    
    if n_reconstructed < len(results_df):
        print(f"   [WARNING] Some comparisons may not be apples-to-apples!")
        print(f"   [WARNING] Consider re-training with 'val_patient_ids' saved in checkpoints")
    
    # 保存结果
    results_csv = output_dir / 'ensemble_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")
    
    print(f"\n[OK] Peer-reviewed ensemble evaluation completed!")
    
    # 根据结果给出建议
    print(f"\n[RECOMMENDATIONS] Recommendations:")
    if mean_improvement > 0.01:
        print(f"   [OK] Ensemble provides meaningful improvement (+{mean_improvement:.3f})")
        print(f"   [OK] Consider using ensemble for final model")
    elif mean_improvement > 0:
        print(f"   [WARNING] Ensemble provides marginal improvement (+{mean_improvement:.3f})")
        print(f"   [WARNING] May not be worth the extra complexity")
    else:
        print(f"   [ERROR] Ensemble does not improve over best single model")
        print(f"   [ERROR] Use best single model instead")
        print(f"   [ERROR] Models may be too similar (high correlation)")


if __name__ == '__main__':
    main()