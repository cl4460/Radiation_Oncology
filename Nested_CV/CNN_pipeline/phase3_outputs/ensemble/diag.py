#!/usr/bin/env python
"""
ensemble_diagnostics.py - 模型相关性与Temperature诊断工具

用途:
1. 量化3个模型之间的risk correlation
2. 测试不同temperature对learned weights的影响
3. 测试不同averaging域 (logit/hazard/survival)

运行时间: ~30-40分钟 (5 folds)
"""

import os, math, random, re
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import spearmanr
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
# 路径配置 - 与ensemble.py完全相同
# ============================================================
CROP_LOG_CSV = "/home/lichengze/Research/CNN_pipeline/phase2_outputs/crop_log.csv"
CLINICAL_CSV = "/home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical.csv"
OUTPUT_DIR = "/home/lichengze/Research/CNN_pipeline/phase3_outputs/ensemble/ensemble_diagnostics"

# ============================================================
# 配置
# ============================================================
class DiagConfig:
    TOP_LRS = ['7e-4', '6.8e-4', '7.4e-4']
    BASE_DIR = Path('/home/lichengze/Research/CNN_pipeline/phase3_outputs/learning_rate/learning_rate_corrected/output')
    
    N_FOLDS = 5
    SEED = 42
    
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2
    
    # 测试配置
    TEMPERATURES = [0.01, 0.05, 0.10, 0.20]  # 要测试的temperature值
    AVERAGING_MODES = ['logit', 'hazard', 'survival']  # 要测试的averaging方法
    USE_TTA = True  # 与训练保持一致
    
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config = DiagConfig()

# ============================================================
# 工具函数 - 从ensemble.py复制
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
# Clinical特征处理 - 从ensemble.py复制（简化版）
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
    """从ensemble.py完整复制"""
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
# Dataset - 从ensemble.py复制
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
# 模型 - 从ensemble.py复制
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
# 🔬 诊断1: 提取每个模型的risk scores
# ============================================================

@torch.no_grad()
def extract_model_risks(model, dl_val, device, n_time_bins, bin_mids, use_tta=True):
    """
    提取单个模型对所有validation样本的risk scores
    
    Returns:
        risks: [N] array of risk scores
    """
    model.eval()
    surv_all = []
    
    for batch in dl_val:
        img_batch = batch['image']
        clin_batch = batch['clinical']
        
        if use_tta:
            # 4-view TTA
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
            
            logits = torch.stack(logits_views, 0).mean(0)
            haz = torch.sigmoid(logits)
            surv = torch.cumprod(1.0 - haz, dim=1).cpu().numpy()
        else:
            logits = model(img_batch.to(device), clin_batch.to(device))[:, :n_time_bins]
            haz = torch.sigmoid(logits)
            surv = torch.cumprod(1.0 - haz, dim=1).cpu().numpy()
        
        surv_all.append(surv)
    
    surv_all = np.concatenate(surv_all, axis=0)  # [N, K]
    
    # 计算risk: -AUC(survival)
    risks = -np.trapz(surv_all, x=bin_mids, axis=1)  # [N]
    
    return risks

# ============================================================
# 🔬 诊断2: 计算模型相关性
# ============================================================

def compute_model_correlations(risks_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算模型之间的Pearson和Spearman相关性
    
    Args:
        risks_dict: {'7e-4': risks1, '6.8e-4': risks2, '7.4e-4': risks3}
    
    Returns:
        pearson_corr: [3, 3] Pearson correlation matrix
        spearman_corr: [3, 3] Spearman correlation matrix
    """
    lrs = list(risks_dict.keys())
    risks_matrix = np.column_stack([risks_dict[lr] for lr in lrs])  # [N, 3]
    
    # Pearson correlation
    pearson_corr = np.corrcoef(risks_matrix, rowvar=False)
    
    # Spearman correlation
    spearman_corr = spearmanr(risks_matrix, axis=0).correlation
    
    return pearson_corr, spearman_corr

# ============================================================
# 🔬 诊断3: 测试不同temperature
# ============================================================

def compute_learned_weights(single_unos: List[float], temperature: float = 0.01):
    """
    根据单模型性能计算learned weights
    
    Args:
        single_unos: [3] Uno C-indices
        temperature: temperature parameter
    
    Returns:
        weights: [3] normalized weights
    """
    scores = np.array(single_unos)
    exp_scores = np.exp(scores / temperature)
    weights = exp_scores / exp_scores.sum()
    return weights.tolist()

def test_temperature_effects(single_unos: List[float], lrs: List[str]):
    """
    测试不同temperature对weights的影响
    
    Args:
        single_unos: [3] Uno C-indices for the 3 models
        lrs: ['7e-4', '6.8e-4', '7.4e-4']
    
    Returns:
        results: dict with temperature -> weights mapping
    """
    results = {}
    
    print(f"\n{'─'*70}")
    print(f"Temperature Test Results")
    print(f"{'─'*70}")
    print(f"\nSingle Model Performance:")
    for lr, uno in zip(lrs, single_unos):
        print(f"  lr={lr:<8s}: {uno:.4f}")
    
    print(f"\nLearned Weights at Different Temperatures:")
    print(f"{'─'*70}")
    print(f"{'Temp':<10} {'lr=7e-4':<15} {'lr=6.8e-4':<15} {'lr=7.4e-4':<15}")
    print(f"{'─'*70}")
    
    for T in config.TEMPERATURES:
        weights = compute_learned_weights(single_unos, temperature=T)
        results[T] = weights
        
        print(f"{T:<10.2f} {weights[0]:<15.3f} {weights[1]:<15.3f} {weights[2]:<15.3f}")
    
    print(f"{'─'*70}")
    
    return results

# ============================================================
# 🔬 诊断4: 使用不同weights进行ensemble并评估
# ============================================================

@torch.no_grad()
def ensemble_predict_with_weights(models, img_batch, clin_batch, device, 
                                 n_time_bins, weights, mode='logit', tta=True):
    """
    使用给定weights进行ensemble预测
    """
    parts = []
    
    for m in models:
        m.eval()
        
        if tta:
            views = [
                img_batch,
                torch.flip(img_batch, dims=[-2]),
                torch.flip(img_batch, dims=[-1]),
                torch.flip(img_batch, dims=[-2, -1]),
            ]
            logits_views = [m(v.to(device), clin_batch.to(device))[:, :n_time_bins] for v in views]
            logits = torch.stack(logits_views, 0).mean(0)
        else:
            logits = m(img_batch.to(device), clin_batch.to(device))[:, :n_time_bins]
        
        if mode == 'logit':
            parts.append(logits.cpu())
        elif mode == 'hazard':
            parts.append(torch.sigmoid(logits).cpu())
        elif mode == 'survival':
            hz = torch.sigmoid(logits)
            sv = torch.cumprod(1.0 - hz, dim=1)
            parts.append(sv.cpu())
    
    P = torch.stack(parts, 0)
    w = torch.tensor(weights, dtype=torch.float32).view(-1, 1, 1)
    
    if mode == 'logit':
        logits_ens = (P * w).sum(0)
        hz = torch.sigmoid(logits_ens)
        sv = torch.cumprod(1.0 - hz, dim=1)
        return sv.cpu().numpy()
    elif mode == 'hazard':
        hz = (P * w).sum(0)
        sv = torch.cumprod(1.0 - hz, dim=1)
        return sv.cpu().numpy()
    else:  # 'survival'
        sv = (P * w).sum(0).numpy()
        sv = np.minimum.accumulate(sv[:, ::-1], axis=1)[:, ::-1]
        return sv

def evaluate_ensemble_with_config(models, dl_val, device, n_time_bins, bin_mids,
                                  y_train_struct, y_val_struct, tau,
                                  weights, mode='logit', tta=True):
    """
    使用给定配置评估ensemble
    
    Returns:
        uno: Uno C-index
    """
    all_surv = []
    
    for batch in dl_val:
        surv = ensemble_predict_with_weights(
            models, batch['image'], batch['clinical'],
            device, n_time_bins, weights, mode=mode, tta=tta
        )
        all_surv.append(surv)
    
    all_surv = np.concatenate(all_surv, axis=0)
    risk = -np.trapz(all_surv, x=bin_mids, axis=1)
    
    uno = concordance_index_ipcw(y_train_struct, y_val_struct, risk, tau=tau)[0]
    
    return uno

# ============================================================
# 🔬 主诊断函数
# ============================================================

def diagnose_fold(fold, manifest, idx_train, idx_val):
    """
    对单个fold进行完整诊断
    """
    print(f"\n{'='*70}")
    print(f"🔬 DIAGNOSTIC FOR FOLD {fold}")
    print(f"{'='*70}")
    
    times = manifest["time"].values
    events = manifest["event"].values
    
    # ============================================================
    # 1. 加载模型
    # ============================================================
    print(f"\n[1/5] Loading models...")
    
    models = []
    ckpts = []
    
    for lr in config.TOP_LRS:
        model_path = config.BASE_DIR / f'lr_{lr}' / f'fold_{fold}' / 'best.pt'
        
        if not model_path.exists():
            print(f"  ❌ Model not found: {model_path}")
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
        
        print(f"  ✅ lr={lr:<8s}: loaded")
    
    # 时间分箱设置
    n_time_bins = ckpts[0]['n_time_bins']
    cuts = np.array(ckpts[0]['cuts'])
    bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
    
    # ============================================================
    # 2. 准备数据
    # ============================================================
    print(f"\n[2/5] Preparing data...")
    
    lab = LabTransDiscreteTime(cuts=cuts[1:-1])
    y_train = lab.transform(times[idx_train], events[idx_train])
    y_val = lab.transform(times[idx_val], events[idx_val])
    
    clinical_train, clinical_val, feature_names = prepare_clinical_features_fold(
        manifest, idx_train, idx_val
    )
    
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
    
    y_train_struct = to_structured_y(times[idx_train], events[idx_train])
    y_val_struct = to_structured_y(times[idx_val], events[idx_val])
    evt_times = times[idx_train][events[idx_train] == 1]
    tau = np.quantile(evt_times, 0.9)
    
    print(f"  Val samples: {len(idx_val)}")
    
    # ============================================================
    # 3. 提取每个模型的risks并计算相关性
    # ============================================================
    print(f"\n[3/5] Computing model correlations...")
    
    risks_dict = {}
    single_unos = []
    
    for i, (lr, model) in enumerate(zip(config.TOP_LRS, models)):
        print(f"  Extracting risks for lr={lr}...", end='')
        
        risks = extract_model_risks(
            model, dl_val, config.DEVICE, n_time_bins, bin_mids, 
            use_tta=config.USE_TTA
        )
        risks_dict[lr] = risks
        
        # 同时计算Uno
        uno = concordance_index_ipcw(y_train_struct, y_val_struct, risks, tau=tau)[0]
        single_unos.append(uno)
        
        print(f" Uno={uno:.4f}")
    
    # 计算相关性
    pearson_corr, spearman_corr = compute_model_correlations(risks_dict)
    
    print(f"\n{'─'*70}")
    print(f"🔍 Model Correlation Analysis")
    print(f"{'─'*70}")
    
    print(f"\nPearson Correlation Matrix:")
    print(f"{'':>12} {'7e-4':>12} {'6.8e-4':>12} {'7.4e-4':>12}")
    for i, lr1 in enumerate(config.TOP_LRS):
        row = f"{lr1:>12}"
        for j, lr2 in enumerate(config.TOP_LRS):
            row += f" {pearson_corr[i,j]:>12.4f}"
        print(row)
    
    print(f"\nSpearman Correlation Matrix:")
    print(f"{'':>12} {'7e-4':>12} {'6.8e-4':>12} {'7.4e-4':>12}")
    for i, lr1 in enumerate(config.TOP_LRS):
        row = f"{lr1:>12}"
        for j, lr2 in enumerate(config.TOP_LRS):
            row += f" {spearman_corr[i,j]:>12.4f}"
        print(row)
    
    # 计算平均相关性（排除对角线）
    mask = ~np.eye(3, dtype=bool)
    mean_pearson = pearson_corr[mask].mean()
    mean_spearman = spearman_corr[mask].mean()
    
    print(f"\n📊 Average Pairwise Correlation:")
    print(f"  Pearson:  {mean_pearson:.4f}")
    print(f"  Spearman: {mean_spearman:.4f}")
    
    # 判断相关性水平
    if mean_pearson > 0.9:
        corr_level = "🔴 VERY HIGH"
        advice = "Ensemble unlikely to help - models are too similar"
    elif mean_pearson > 0.8:
        corr_level = "🟠 HIGH"
        advice = "Limited ensemble benefit expected"
    elif mean_pearson > 0.7:
        corr_level = "🟡 MODERATE"
        advice = "Some ensemble benefit possible"
    else:
        corr_level = "🟢 LOW"
        advice = "Good diversity - ensemble should help"
    
    print(f"\n  Correlation Level: {corr_level}")
    print(f"  Recommendation: {advice}")
    
    # ============================================================
    # 4. Temperature测试
    # ============================================================
    print(f"\n[4/5] Testing temperature effects...")
    
    temp_results = test_temperature_effects(single_unos, config.TOP_LRS)
    
    # ============================================================
    # 5. 测试不同配置的ensemble
    # ============================================================
    print(f"\n[5/5] Testing ensemble configurations...")
    
    print(f"\n{'─'*70}")
    print(f"📊 Ensemble Configuration Tests")
    print(f"{'─'*70}")
    print(f"\nBaseline - Best Single Model: {max(single_unos):.4f}")
    print(f"\n{'Config':<35} {'Uno':<10} {'Δ':<10} {'Δ%':<10}")
    print(f"{'─'*70}")
    
    results = []
    
    # 测试不同temperature + mode组合
    for mode in config.AVERAGING_MODES:
        for T in config.TEMPERATURES:
            weights = temp_results[T]
            
            uno = evaluate_ensemble_with_config(
                models, dl_val, config.DEVICE, n_time_bins, bin_mids,
                y_train_struct, y_val_struct, tau,
                weights=weights, mode=mode, tta=config.USE_TTA
            )
            
            best_single = max(single_unos)
            delta = uno - best_single
            delta_pct = delta / best_single * 100
            
            config_name = f"{mode}, T={T:.2f}"
            print(f"{config_name:<35} {uno:<10.4f} {delta:+.4f}    {delta_pct:+.2f}%")
            
            results.append({
                'mode': mode,
                'temperature': T,
                'weights': weights,
                'uno': uno,
                'delta': delta,
                'delta_pct': delta_pct
            })
    
    # 找到最佳配置
    best_result = max(results, key=lambda x: x['uno'])
    
    print(f"{'─'*70}")
    print(f"\n🏆 Best Configuration:")
    print(f"  Mode: {best_result['mode']}")
    print(f"  Temperature: {best_result['temperature']:.2f}")
    print(f"  Weights: {[f'{w:.3f}' for w in best_result['weights']]}")
    print(f"  Uno: {best_result['uno']:.4f}")
    print(f"  Improvement: {best_result['delta']:+.4f} ({best_result['delta_pct']:+.2f}%)")
    
    # ============================================================
    # 返回汇总结果
    # ============================================================
    lr_7e4_uno = single_unos[config.TOP_LRS.index('7e-4')] 
    return {
        'fold': fold,
        'mean_pearson': mean_pearson,
        'mean_spearman': mean_spearman,
        'single_unos': single_unos,  # [7e-4, 6.8e-4, 7.4e-4]的性能列表
        'lr_7e4_uno': lr_7e4_uno,  # ← 新增：7e-4的性能（正确的baseline）
        'best_of_3_uno': max(single_unos),  # ← 保留：3个中最好的（仅供参考）
        'best_ensemble_uno': best_result['uno'],
        'best_ensemble_mode': best_result['mode'],
        'best_ensemble_temp': best_result['temperature'],
        'best_improvement': best_result['delta'],
        'correlation_level': corr_level
    }

# ============================================================
# Main
# ============================================================

def main():
    set_seed(config.SEED)
    
    print("\n" + "="*70)
    print("🔬 ENSEMBLE DIAGNOSTICS - CORRELATION & TEMPERATURE TESTS")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Models: {config.TOP_LRS}")
    print(f"  Temperatures to test: {config.TEMPERATURES}")
    print(f"  Averaging modes to test: {config.AVERAGING_MODES}")
    print(f"  Use TTA: {config.USE_TTA}")
    print(f"  Device: {config.DEVICE}")
    
    # 加载数据
    print(f"\n📁 Loading data...")
    
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
    
    manifest = crop[["patient_id", "out_ct", "out_mask", "out_edt"]].rename(columns={
        "out_ct": "ct_path",
        "out_mask": "mask_path",
        "out_edt": "edt_path"
    }).merge(clinical, on="patient_id", how="inner")
    
    manifest = manifest.dropna(subset=["time", "event"])
    manifest["time"] = manifest["time"].astype(int)
    manifest["event"] = manifest["event"].astype(int)
    
    print(f"  Dataset: {len(manifest)} patients")
    
    # CV splits
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, 
        shuffle=True, 
        random_state=config.SEED
    )
    
    # 诊断所有folds
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for fold, (idx_train, idx_val) in enumerate(skf.split(
        np.zeros(len(manifest)), manifest["event"]
    )):
        results = diagnose_fold(fold, manifest, idx_train, idx_val)
        
        if results is not None:
            all_results.append(results)
    
    # ============================================================
    # 汇总所有folds的结果
    # ============================================================
    if len(all_results) == 0:
        print(f"\n❌ No results collected!")
        return
    
    results_df = pd.DataFrame(all_results)
    
    print(f"\n\n{'='*70}")
    print(f"📊 SUMMARY ACROSS ALL FOLDS")
    print(f"{'='*70}")
    
    # 相关性汇总
    print(f"\n🔍 Correlation Summary:")
    print(f"{'─'*70}")
    print(f"{'Fold':<6} {'Pearson':<12} {'Spearman':<12} {'Level':<20} {'Best Δ%':<10}")
    print(f"{'─'*70}")
    
    for _, row in results_df.iterrows():
        # 修正：用lr_7e4_uno作为baseline计算改善百分比
        delta_pct = (row['best_improvement'] / row['lr_7e4_uno']) * 100
        print(f"{int(row['fold']):<6} {row['mean_pearson']:<12.4f} "
            f"{row['mean_spearman']:<12.4f} {row['correlation_level']:<20} "
            f"{delta_pct:+.2f}%")
    
    print(f"{'─'*70}")
    
    avg_pearson = results_df['mean_pearson'].mean()
    avg_spearman = results_df['mean_spearman'].mean()

    # ← 新增：显示正确的baseline性能
    mean_single_7e4 = results_df['lr_7e4_uno'].mean()
    std_single_7e4 = results_df['lr_7e4_uno'].std()
    mean_ensemble = results_df['best_ensemble_uno'].mean()
    std_ensemble = results_df['best_ensemble_uno'].std()

    print(f"\n📈 Overall Statistics:")
    print(f"  Mean Single (LR=7e-4):     {mean_single_7e4:.4f} ± {std_single_7e4:.3f}")
    print(f"  Mean Ensemble (Top-3):     {mean_ensemble:.4f} ± {std_ensemble:.3f}")
    print(f"  Mean Improvement:          {mean_ensemble - mean_single_7e4:+.4f} ({(mean_ensemble - mean_single_7e4)/mean_single_7e4*100:+.2f}%)")
    print(f"")
    print(f"  Mean Pearson Correlation:  {avg_pearson:.4f}")
    print(f"  Mean Spearman Correlation: {avg_spearman:.4f}")
    
    # 判断总体相关性
    if avg_pearson > 0.9:
        print(f"\n  🔴 CONCLUSION: Models are VERY HIGHLY correlated")
        print(f"     → Ensemble is unlikely to provide benefits")
        print(f"     → Recommend using best single model")
    elif avg_pearson > 0.8:
        print(f"\n  🟠 CONCLUSION: Models are HIGHLY correlated")
        print(f"     → Limited ensemble benefit expected (<1%)")
        print(f"     → May not be worth the complexity")
    elif avg_pearson > 0.7:
        print(f"\n  🟡 CONCLUSION: Models have MODERATE correlation")
        print(f"     → Some ensemble benefit possible (1-2%)")
        print(f"     → Worth trying with optimal configuration")
    else:
        print(f"\n  🟢 CONCLUSION: Models have LOW correlation")
        print(f"     → Good diversity - ensemble should help (2-3%+)")
        print(f"     → Strongly recommend ensemble")
    
    # Ensemble改善汇总
    print(f"\n🏆 Best Ensemble Configuration Per Fold:")
    print(f"{'─'*70}")
    print(f"{'Fold':<6} {'Mode':<12} {'Temp':<8} {'Single(7e-4)':<14} {'Ensemble':<10} {'Δ%':<10}")
    print(f"{'─'*70}")

    for _, row in results_df.iterrows():
        # 修正：用lr_7e4_uno作为baseline
        delta_pct = (row['best_ensemble_uno'] - row['lr_7e4_uno']) / row['lr_7e4_uno'] * 100
        print(f"{int(row['fold']):<6} {row['best_ensemble_mode']:<12} "
            f"{row['best_ensemble_temp']:<8.2f} {row['lr_7e4_uno']:<14.4f} "
            f"{row['best_ensemble_uno']:<10.4f} {delta_pct:+.2f}%")
    
    print(f"{'─'*70}")
    
    # 统计检验
    single_7e4_values = results_df['lr_7e4_uno'].values  # ← 正确
    best_ensembles = results_df['best_ensemble_uno'].values
        
    if len(results_df) >= 3:
        stat, p_value = stats.wilcoxon(best_ensembles, single_7e4_values)  # ← 修正
        
        print(f"\n📊 Statistical Test (Wilcoxon):")
        print(f"  P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✅ Ensemble significantly better (p < 0.05)")
        elif p_value < 0.10:
            print(f"  ⚠️  Marginally significant (0.05 <= p < 0.10)")
        else:
            print(f"  ❌ Not statistically significant (p >= 0.10)")
    
    # 保存结果
    results_csv = output_dir / 'diagnostic_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\n💾 Results saved to: {results_csv}")
    
    # 最终建议
    print(f"\n{'='*70}")
    print(f"💡 FINAL RECOMMENDATIONS")
    print(f"{'='*70}")
    
    mean_improvement = mean_ensemble - mean_single_7e4  # 已在前面计算
    mean_improvement_pct = (mean_improvement / mean_single_7e4) * 100
    
    if avg_pearson > 0.9:
        print(f"\n❌ DO NOT use ensemble:")
        print(f"   - Models are too highly correlated (ρ={avg_pearson:.3f})")
        print(f"   - Best single (7e-4): {mean_single_7e4:.4f}")  # ← 修正
        print(f"   - Best ensemble: {mean_ensemble:.4f}")  # ← 修正
        print(f"   - Improvement: {mean_improvement_pct:+.2f}% (not significant)")
        print(f"   - Action: Use single model with LR=7e-4")
        print(f"   - Paper: Report correlation analysis as negative result")
    elif avg_pearson > 0.8 and mean_improvement_pct < 0.5:
        print(f"\n⚠️  Ensemble provides minimal benefit:")
        print(f"   - Models are highly correlated (ρ={avg_pearson:.3f})")
        print(f"   - Average improvement: {mean_improvement_pct:+.2f}%")
        print(f"   - May not be worth the added complexity")
        print(f"   - Consider: Use single model, or explore architectural diversity")
    else:
        print(f"\n✅ Ensemble shows promise:")
        print(f"   - Correlation: ρ={avg_pearson:.3f}")
        print(f"   - Average improvement: {mean_improvement_pct:+.2f}%")
        print(f"   - Recommended configuration:")
        
        # 找出最常见的最优配置
        mode_counts = results_df['best_ensemble_mode'].value_counts()
        temp_mean = results_df['best_ensemble_temp'].mean()
        
        print(f"     - Mode: {mode_counts.index[0]} (most frequent)")
        print(f"     - Temperature: {temp_mean:.2f} (average best)")
    
    print(f"\n✅ Diagnostic completed!")


if __name__ == '__main__':
    main()