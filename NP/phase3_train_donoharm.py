#!/usr/bin/env python3
"""
phase3_train_donoharm.py
========================
NSCLC Survival Prediction - Optimized Training Script

Key Features:
1. Do-No-Harm Selection: Train both image-only and image+clinical per fold,
   select winner based on validation Uno C-index
2. Missing Augmentation: Simulate high missing rate during training to improve
   robustness on external validation (RG has ~80% stage missing vs LUNG1 ~5%)
3. Clinical Feature Processing: Ordinal encoding for stage (more robust for OOD)
4. GroupNorm for small batch stability
5. 4-view flip TTA for inference

Based on best configuration from previous experiments:
- lr=7e-4, dropout=0.35, weight_decay=2e-4
- CT+EDT dual-channel input (HU window normalized to [0,1])
- 12-dim clinical features with missing indicators

Author: Research Pipeline
"""

import os
import sys
import math
import random
import argparse
import warnings
import json
import shutil
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConcatItemsd, DeleteItemsd, ToTensord,
    RandAffined, RandGaussianNoised, RandAdjustContrastd, RandFlipd,
    Lambdad,
)
from monai.networks.nets import resnet as monai_resnet

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.loss import NLLLogistiHazardLoss

from sksurv.metrics import concordance_index_ipcw, concordance_index_censored, brier_score

warnings.filterwarnings("ignore")
EPS = 1e-8


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class TrainConfig:
    """Training configuration with optimized hyperparameters."""
    
    # Cross-validation
    N_FOLDS: int = 5
    SEED: int = 42
    
    # Training
    BATCH_SIZE: int = 4
    ACCUMULATION_STEPS: int = 2  # Effective batch = 8
    MAX_EPOCHS: int = 100
    EARLY_STOP: int = 30
    
    # Optimizer (from best experiments)
    LR: float = 7e-4
    WEIGHT_DECAY: float = 2e-4
    DROPOUT: float = 0.35
    WARMUP_EPOCHS: int = 25
    
    # Data augmentation
    AUG_PROB: float = 0.5
    FLIP_PROB: float = 0.5
    ROT_DEG: Tuple[float, float, float] = (5.0, 5.0, 10.0)
    SCALE_RANGE: float = 0.12
    NOISE_STD: float = 0.02
    GAMMA_RANGE: Tuple[float, float] = (0.85, 1.15)
    
    # Data loading
    NUM_WORKERS: int = 4
    PREFETCH_FACTOR: int = 2
    
    # Survival model
    N_BINS_MAX: int = 15
    N_BINS_MIN: int = 12
    MIN_EVENTS_PER_BIN: int = 20
    
    # Evaluation
    USE_TTA: bool = True
    EVAL_24M_DAYS: int = 730


@dataclass
class ClinicalConfig:
    """Clinical feature processing configuration."""
    
    # Imputation method
    age_impute: str = "median"  # "median" is more robust to outliers
    
    # Encoding method for overall_stage
    # "ordinal" is more robust for external validation with distribution shift
    stage_encoding: str = "ordinal"
    
    # Missing augmentation probabilities (training only)
    # Key insight: RG has ~80% stage missing, LUNG1 has ~5%
    # We need to simulate this during training
    drop_stage_prob: float = 0.50  # High prob to match RG's missing rate
    drop_tnm_prob: float = 0.25    # TNM also has higher missing in RG
    drop_all_clin_prob: float = 0.15  # Full modality dropout
    
    # Stage ordinal mapping (normalized to [0,1])
    stage_ordinal_map: Dict[str, float] = field(default_factory=lambda: {
        "I": 0.2, "1": 0.2, "IA": 0.15, "IB": 0.25,
        "II": 0.4, "2": 0.4, "IIA": 0.35, "IIB": 0.45,
        "IIIA": 0.6, "3A": 0.6,
        "III": 0.7,  # Ambiguous III
        "IIIB": 0.8, "3B": 0.8,
        "IV": 1.0, "4": 1.0,
    })


# Global configs
cfg = TrainConfig()
clin_cfg = ClinicalConfig()


# ============================================================================
# Utility Functions
# ============================================================================
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def to_structured_y(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Convert to structured array for sksurv."""
    return np.array(
        [(bool(e), float(t)) for t, e in zip(times, events)],
        dtype=[('event', bool), ('time', float)]
    )


def adaptive_bins(times: np.ndarray, events: np.ndarray) -> int:
    """Determine optimal number of time bins based on event count."""
    n_events = int(events.sum())
    ideal = n_events // cfg.MIN_EVENTS_PER_BIN
    return int(np.clip(ideal, cfg.N_BINS_MIN, cfg.N_BINS_MAX))


def make_time_bins(times: np.ndarray, events: np.ndarray, n_bins: int) -> np.ndarray:
    """Create time bin edges using event time quantiles."""
    event_times = times[events == 1]
    if len(event_times) < n_bins:
        return np.linspace(times.min(), times.max(), n_bins + 1)
    
    quantiles = np.linspace(0, 1, n_bins + 1)
    cuts = np.unique(np.quantile(event_times, quantiles))
    
    if len(cuts) != n_bins + 1:
        return np.linspace(times.min(), times.max(), n_bins + 1)
    return cuts


# ============================================================================
# Clinical Feature Processing
# ============================================================================
def _safe_mode(values: np.ndarray, default: float) -> float:
    """Compute mode of non-NaN values, or return default."""
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return float(default)
    cnt = Counter(vals.tolist())
    mode_val = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return float(mode_val)


def _parse_T(series: pd.Series) -> np.ndarray:
    """Parse T stage: extract numeric 1-4, else NaN."""
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).upper()
        m = re.search(r'(\d)', s)
        if m:
            t = float(m.group(1))
            out.append(t if 1.0 <= t <= 4.0 else np.nan)
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)


def _parse_N(series: pd.Series) -> np.ndarray:
    """Parse N stage: extract numeric 0-3, else NaN."""
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).upper()
        m = re.search(r'(\d)', s)
        if m:
            n = float(m.group(1))
            out.append(n if 0.0 <= n <= 3.0 else np.nan)
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)


def _parse_M(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Parse M stage: return (M01 binary, M_misc indicator for M>1)."""
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


def _parse_overall_stage_ordinal(series: pd.Series, mapping: Dict[str, float]) -> np.ndarray:
    """Parse overall stage to ordinal value using mapping."""
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).strip().upper().replace("STAGE", "").replace(" ", "")
        # Handle unicode roman numerals
        s = (s.replace("ⅢA", "IIIA").replace("ⅢB", "IIIB")
              .replace("Ⅲ", "III").replace("Ⅱ", "II")
              .replace("Ⅳ", "IV").replace("Ⅰ", "I"))
        if s in mapping:
            out.append(mapping[s])
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)


def prepare_clinical_features_fold(
    manifest: pd.DataFrame,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, float]]:
    """
    Prepare clinical features for a single CV fold.
    
    Returns:
        X_train: [n_train, n_features] clinical features
        X_val: [n_val, n_features] clinical features
        feature_names: List of feature names
        defaults: Dict of default values for missing augmentation
    """
    feats_tr, feats_va = [], []
    names = []
    defaults = {}
    
    n_train, n_val = len(idx_train), len(idx_val)
    
    # ==================== AGE ====================
    if 'age' in manifest.columns:
        age_raw = pd.to_numeric(manifest['age'], errors='coerce').values.astype(np.float32)
        age_missing = np.isnan(age_raw).astype(np.float32)
        
        # Compute imputation value from training set
        if clin_cfg.age_impute == "median":
            age_impute_val = np.nanmedian(age_raw[idx_train])
        else:
            age_impute_val = np.nanmean(age_raw[idx_train])
        
        if np.isnan(age_impute_val):
            age_impute_val = 65.0
        
        age_filled = age_raw.copy()
        age_filled[np.isnan(age_filled)] = age_impute_val
        
        # StandardScaler fit on training
        scaler = StandardScaler()
        scaler.fit(age_filled[idx_train].reshape(-1, 1))
        age_scaled = scaler.transform(age_filled.reshape(-1, 1)).astype(np.float32).flatten()
        
        # Default value for augmentation
        age_scaled_default = float(scaler.transform(np.array([[age_impute_val]]))[0, 0])
        
        feats_tr.extend([age_scaled[idx_train], age_missing[idx_train]])
        feats_va.extend([age_scaled[idx_val], age_missing[idx_val]])
        names.extend(['age_scaled', 'age_missing'])
        defaults['age_scaled_default'] = age_scaled_default
        
        if verbose:
            miss_tr = age_missing[idx_train].mean() * 100
            miss_va = age_missing[idx_val].mean() * 100
            print(f"    age: impute={clin_cfg.age_impute}={age_impute_val:.1f} | "
                  f"missing: tr={miss_tr:.1f}% va={miss_va:.1f}%")
    
    # ==================== GENDER ====================
    if 'gender' in manifest.columns:
        gender_ser = manifest['gender'].astype(str).str.upper().str[0]
        gender = (gender_ser == 'M').astype(np.float32).values
        
        feats_tr.append(gender[idx_train])
        feats_va.append(gender[idx_val])
        names.append('gender_male')
        defaults['gender_male_default'] = float(_safe_mode(gender[idx_train], 1.0))
        
        if verbose:
            male_tr = gender[idx_train].mean() * 100
            print(f"    gender: male={male_tr:.1f}% (train)")
    
    # ==================== T STAGE ====================
    if 'T' in manifest.columns:
        T_num = _parse_T(manifest['T'])
        T_missing = np.isnan(T_num).astype(np.float32)
        T_mode = _safe_mode(T_num[idx_train], default=2.0)
        
        T_filled = T_num.copy()
        T_filled[np.isnan(T_filled)] = T_mode
        T_norm = (T_filled - 1.0) / 3.0
        
        feats_tr.extend([T_norm[idx_train], T_missing[idx_train]])
        feats_va.extend([T_norm[idx_val], T_missing[idx_val]])
        names.extend(['T_norm', 'T_missing'])
        defaults['T_norm_default'] = float((T_mode - 1.0) / 3.0)
        
        if verbose:
            miss_tr = T_missing[idx_train].mean() * 100
            miss_va = T_missing[idx_val].mean() * 100
            print(f"    T: mode={T_mode:.0f} | missing: tr={miss_tr:.1f}% va={miss_va:.1f}%")
    
    # ==================== N STAGE ====================
    if 'N' in manifest.columns:
        N_num = _parse_N(manifest['N'])
        N_missing = np.isnan(N_num).astype(np.float32)
        N_mode = _safe_mode(N_num[idx_train], default=1.0)
        
        N_filled = N_num.copy()
        N_filled[np.isnan(N_filled)] = N_mode
        N_norm = N_filled / 3.0
        
        feats_tr.extend([N_norm[idx_train], N_missing[idx_train]])
        feats_va.extend([N_norm[idx_val], N_missing[idx_val]])
        names.extend(['N_norm', 'N_missing'])
        defaults['N_norm_default'] = float(N_mode / 3.0)
        
        if verbose:
            miss_tr = N_missing[idx_train].mean() * 100
            miss_va = N_missing[idx_val].mean() * 100
            print(f"    N: mode={N_mode:.0f} | missing: tr={miss_tr:.1f}% va={miss_va:.1f}%")
    
    # ==================== M STAGE ====================
    if 'M' in manifest.columns:
        M01, M_misc = _parse_M(manifest['M'])
        M_missing = np.isnan(M01).astype(np.float32)
        M_mode = _safe_mode(M01[idx_train], default=0.0)
        
        M01_filled = M01.copy()
        M01_filled[np.isnan(M01_filled)] = M_mode
        
        feats_tr.extend([M01_filled[idx_train], M_missing[idx_train], M_misc[idx_train]])
        feats_va.extend([M01_filled[idx_val], M_missing[idx_val], M_misc[idx_val]])
        names.extend(['M01', 'M_missing', 'M_misc'])
        defaults['M01_default'] = float(M_mode)
        
        if verbose:
            miss_tr = M_missing[idx_train].mean() * 100
            miss_va = M_missing[idx_val].mean() * 100
            print(f"    M: mode={M_mode:.0f} | missing: tr={miss_tr:.1f}% va={miss_va:.1f}%")
    
    # ==================== OVERALL STAGE (Ordinal) ====================
    if 'overall_stage' in manifest.columns:
        ov = _parse_overall_stage_ordinal(manifest['overall_stage'], clin_cfg.stage_ordinal_map)
        ov_missing = np.isnan(ov).astype(np.float32)
        ov_mode = _safe_mode(ov[idx_train], default=0.6)
        
        ov_filled = ov.copy()
        ov_filled[np.isnan(ov_filled)] = ov_mode
        
        feats_tr.extend([ov_filled[idx_train], ov_missing[idx_train]])
        feats_va.extend([ov_filled[idx_val], ov_missing[idx_val]])
        names.extend(['overall_stage_ord', 'overall_missing'])
        defaults['overall_stage_default'] = float(ov_mode)
        
        if verbose:
            miss_tr = ov_missing[idx_train].mean() * 100
            miss_va = ov_missing[idx_val].mean() * 100
            print(f"    overall_stage: ordinal mode={ov_mode:.2f} | missing: tr={miss_tr:.1f}% va={miss_va:.1f}%")
    
    # ==================== ASSEMBLE ====================
    if feats_tr:
        X_tr = np.stack(feats_tr, axis=1).astype(np.float32)
        X_va = np.stack(feats_va, axis=1).astype(np.float32)
    else:
        X_tr = np.zeros((n_train, 1), dtype=np.float32)
        X_va = np.zeros((n_val, 1), dtype=np.float32)
        names = ['dummy']
        defaults['dummy'] = 0.0
    
    # Clean up any remaining NaN/Inf
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)
    
    if verbose:
        print(f"    Total clinical dim: {X_tr.shape[1]} features")
    
    return X_tr, X_va, names, defaults


def clinical_missing_augment(
    x: np.ndarray,
    feature_names: List[str],
    defaults: Dict[str, float],
) -> np.ndarray:
    """
    Apply missing augmentation during training.
    
    This simulates the high missing rate in external validation (RG).
    Key insight: RG has ~80% stage missing vs LUNG1 ~5%.
    """
    x = x.copy()
    
    # 1) Full clinical modality dropout (most aggressive)
    if np.random.rand() < clin_cfg.drop_all_clin_prob:
        for i, name in enumerate(feature_names):
            if name.endswith("_missing"):
                x[i] = 1.0
            else:
                x[i] = 0.0
        return x
    
    # 2) Stage-specific dropout (most important for RG)
    if np.random.rand() < clin_cfg.drop_stage_prob:
        if 'overall_stage_ord' in feature_names:
            i_val = feature_names.index('overall_stage_ord')
            i_mis = feature_names.index('overall_missing')
            x[i_val] = defaults.get('overall_stage_default', 0.6)
            x[i_mis] = 1.0
    
    # 3) TNM dropout
    if np.random.rand() < clin_cfg.drop_tnm_prob:
        # T stage
        if 'T_norm' in feature_names and 'T_missing' in feature_names:
            x[feature_names.index('T_norm')] = defaults.get('T_norm_default', 0.33)
            x[feature_names.index('T_missing')] = 1.0
        
        # N stage
        if 'N_norm' in feature_names and 'N_missing' in feature_names:
            x[feature_names.index('N_norm')] = defaults.get('N_norm_default', 0.33)
            x[feature_names.index('N_missing')] = 1.0
        
        # M stage
        if 'M01' in feature_names and 'M_missing' in feature_names:
            x[feature_names.index('M01')] = defaults.get('M01_default', 0.0)
            x[feature_names.index('M_missing')] = 1.0
    
    return x


# ============================================================================
# Dataset and Transforms
# ============================================================================
class SurvivalDataset(Dataset):
    """Dataset for survival prediction with optional clinical augmentation."""
    
    def __init__(
        self,
        records: List[dict],
        clinical: np.ndarray,
        y_idx: np.ndarray,
        y_evt: np.ndarray,
        transform: Compose,
        feature_names: List[str],
        defaults: Dict[str, float],
        train: bool,
        use_clinical: bool,
    ):
        self.records = records
        self.clinical = clinical
        self.y_idx = y_idx
        self.y_evt = y_evt
        self.transform = transform
        self.feature_names = feature_names
        self.defaults = defaults
        self.train = train
        self.use_clinical = use_clinical
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, i):
        data = self.transform(self.records[i])
        x = self.clinical[i].copy()
        
        if not self.use_clinical:
            # Image-only variant: zero out clinical features
            x = np.zeros_like(x, dtype=np.float32)
        else:
            # Apply missing augmentation during training
            if self.train:
                x = clinical_missing_augment(x, self.feature_names, self.defaults)
        
        data["clinical"] = torch.tensor(x, dtype=torch.float32)
        data["label_idx"] = torch.tensor(self.y_idx[i], dtype=torch.long)
        data["label_event"] = torch.tensor(self.y_evt[i], dtype=torch.float32)
        return data


def build_transforms(train: bool) -> Compose:
    """Build MONAI transforms for CT+EDT input."""
    t = [
        LoadImaged(keys=["ct", "edt"]),
        EnsureChannelFirstd(keys=["ct", "edt"]),
        EnsureTyped(keys=["ct", "edt"], dtype=torch.float32, track_meta=False),
        Lambdad(keys=["ct", "edt"], func=lambda x: torch.nan_to_num(x, 0.0, 0.0, 0.0)),
        Lambdad(keys=["ct", "edt"], func=lambda x: torch.clamp(x, 0.0, 1.0)),
    ]
    
    if train:
        rx, ry, rz = map(math.radians, cfg.ROT_DEG)
        t += [
            RandFlipd(keys=["ct", "edt"], prob=cfg.FLIP_PROB, spatial_axis=1),
            RandFlipd(keys=["ct", "edt"], prob=cfg.FLIP_PROB, spatial_axis=2),
            RandAffined(
                keys=["ct", "edt"],
                prob=cfg.AUG_PROB,
                rotate_range=(rx, ry, rz),
                scale_range=(cfg.SCALE_RANGE, cfg.SCALE_RANGE, cfg.SCALE_RANGE),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),
            RandGaussianNoised(keys=["ct"], prob=cfg.AUG_PROB, std=cfg.NOISE_STD),
            RandAdjustContrastd(keys=["ct"], prob=cfg.AUG_PROB, gamma=cfg.GAMMA_RANGE),
        ]
    
    t += [
        ConcatItemsd(keys=["ct", "edt"], name="image"),
        DeleteItemsd(keys=["ct", "edt"]),
        ToTensord(keys=["image"]),
    ]
    return Compose(t)


# ============================================================================
# Model Architecture
# ============================================================================
class ResNet10_Clinical_GN(nn.Module):
    """
    ResNet10 backbone with GroupNorm + Clinical encoder.
    
    Architecture:
    - Backbone: ResNet10 (3D) -> 512-d image features
    - Clinical encoder: MLP -> 64-d clinical features
    - Fusion head: (512+64) -> 256 -> 128 -> n_bins (hazard logits)
    """
    
    def __init__(
        self,
        in_channels: int,
        clinical_dim: int,
        n_bins: int,
        dropout: float,
    ):
        super().__init__()
        
        # Backbone: ResNet10 with GroupNorm
        self.backbone = monai_resnet.resnet10(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=512,
        )
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
        """Replace BatchNorm with GroupNorm for small batch stability."""
        for name, child in module.named_children():
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                num_channels = child.num_features
                groups = min(num_groups, num_channels)
                while groups > 1 and num_channels % groups != 0:
                    groups //= 2
                setattr(module, name, nn.GroupNorm(groups, num_channels))
            else:
                self._replace_bn_with_gn(child, num_groups)
    
    def forward(self, x_img: torch.Tensor, x_clin: torch.Tensor) -> torch.Tensor:
        f_img = self.backbone(x_img)
        f_clin = self.clinical_encoder(x_clin)
        f = torch.cat([f_img, f_clin], dim=1)
        return self.head(f)


# ============================================================================
# Inference with TTA
# ============================================================================
@torch.no_grad()
def predict_surv_tta4(
    model: nn.Module,
    img_batch: torch.Tensor,
    clin_batch: torch.Tensor,
    device: torch.device,
    n_time_bins: int,
    use_tta: bool = True,
) -> np.ndarray:
    """Predict survival curves with optional 4-view flip TTA."""
    model.eval()
    
    if not use_tta:
        logits = model(img_batch.to(device), clin_batch.to(device))
        haz = torch.sigmoid(logits[:, :n_time_bins])
        surv = torch.cumprod(1.0 - haz, dim=1)
        return surv.cpu().numpy()
    
    # 4-view flip augmentation
    views = [
        img_batch,
        torch.flip(img_batch, dims=[-2]),
        torch.flip(img_batch, dims=[-1]),
        torch.flip(img_batch, dims=[-2, -1]),
    ]
    
    outs = []
    for v in views:
        logits = model(v.to(device), clin_batch.to(device))
        haz = torch.sigmoid(logits[:, :n_time_bins])
        surv = torch.cumprod(1.0 - haz, dim=1)
        outs.append(surv)
    
    return torch.stack(outs, 0).mean(0).cpu().numpy()


# ============================================================================
# Evaluation Metrics
# ============================================================================
def compute_metrics(
    times_tr: np.ndarray,
    events_tr: np.ndarray,
    times_va: np.ndarray,
    events_va: np.ndarray,
    surv: np.ndarray,
    cuts: np.ndarray,
) -> Dict[str, Any]:
    """Compute survival evaluation metrics."""
    y_tr = to_structured_y(times_tr, events_tr)
    y_va = to_structured_y(times_va, events_va)
    
    n_time_bins = surv.shape[1]
    bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
    
    # Risk score: negative integral of survival curve
    risk_integral = -np.trapz(surv, x=bin_mids, axis=1)
    
    # Tau for IPCW (90th percentile of event times)
    evt_times = times_tr[events_tr == 1]
    tau = float(np.quantile(evt_times, 0.9)) if len(evt_times) > 0 else float(np.max(times_tr))
    
    # Uno C-index (IPCW)
    try:
        uno = float(concordance_index_ipcw(y_tr, y_va, risk_integral, tau=tau)[0])
    except Exception:
        uno = float("nan")
    
    # Harrell C-index
    try:
        harrell = float(concordance_index_censored(y_va["event"], y_va["time"], risk_integral)[0])
    except Exception:
        harrell = float("nan")
    
    # Brier score at 24 months
    t_idx_24m = int(np.argmin(np.abs(bin_mids - cfg.EVAL_24M_DAYS)))
    surv_24m = surv[:, t_idx_24m] if t_idx_24m < n_time_bins else surv[:, -1]
    try:
        _, brier = brier_score(y_tr, y_va, surv_24m.reshape(-1, 1), np.array([cfg.EVAL_24M_DAYS]))
        brier_24m = float(brier[0])
    except Exception:
        brier_24m = float("nan")
    
    return {
        "uno": uno,
        "harrell": harrell,
        "tau": tau,
        "brier_24m": brier_24m,
        "risk": risk_integral,
        "surv_24m": surv_24m,
    }


# ============================================================================
# Training One Variant
# ============================================================================
def train_one_variant(
    variant_name: str,
    use_clinical: bool,
    fold_dir: Path,
    records_tr: List[dict],
    records_va: List[dict],
    X_tr: np.ndarray,
    X_va: np.ndarray,
    y_tr_idx: np.ndarray,
    y_tr_evt: np.ndarray,
    y_va_idx: np.ndarray,
    y_va_evt: np.ndarray,
    feature_names: List[str],
    defaults: Dict[str, float],
    cuts: np.ndarray,
    times_tr: np.ndarray,
    events_tr: np.ndarray,
    times_va: np.ndarray,
    events_va: np.ndarray,
) -> Dict[str, Any]:
    """Train one model variant (image-only or image+clinical)."""
    
    out_dir = fold_dir / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create datasets
    ds_tr = SurvivalDataset(
        records_tr, X_tr, y_tr_idx, y_tr_evt,
        transform=build_transforms(train=True),
        feature_names=feature_names, defaults=defaults,
        train=True, use_clinical=use_clinical,
    )
    ds_va = SurvivalDataset(
        records_va, X_va, y_va_idx, y_va_evt,
        transform=build_transforms(train=False),
        feature_names=feature_names, defaults=defaults,
        train=False, use_clinical=use_clinical,
    )
    
    # Create dataloaders
    dl_tr = DataLoader(
        ds_tr, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
        prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None,
    )
    dl_va = DataLoader(
        ds_va, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        persistent_workers=(cfg.NUM_WORKERS > 0),
        prefetch_factor=cfg.PREFETCH_FACTOR if cfg.NUM_WORKERS > 0 else None,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    n_time_bins = len(cuts) - 1
    model = ResNet10_Clinical_GN(
        in_channels=2,
        clinical_dim=X_tr.shape[1],
        n_bins=n_time_bins,
        dropout=cfg.DROPOUT,
    ).to(device)
    
    # Optimizer with weight decay only on non-bias/norm params
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or 'bias' in name or 'norm' in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.LR,
    )
    
    # Learning rate schedule: warmup + cosine annealing
    def lr_schedule(epoch):
        if epoch < cfg.WARMUP_EPOCHS:
            return (epoch + 1) / cfg.WARMUP_EPOCHS
        progress = (epoch - cfg.WARMUP_EPOCHS) / max(1, (cfg.MAX_EPOCHS - cfg.WARMUP_EPOCHS))
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    criterion = NLLLogistiHazardLoss()
    
    best_uno = -1.0
    best_epoch = -1
    no_improve = 0
    best_path = out_dir / "best.pt"
    
    print(f"\n  [{variant_name}] use_clinical={use_clinical} | "
          f"train={len(ds_tr)} val={len(ds_va)} | bins={n_time_bins}", flush=True)
    
    for epoch in range(cfg.MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        
        for step, batch in enumerate(dl_tr):
            x_img = batch["image"].to(device)
            x_clin = batch["clinical"].to(device)
            y_idx = batch["label_idx"].to(device)
            y_evt = batch["label_event"].to(device)
            
            logits = model(x_img, x_clin)
            loss = criterion(logits, y_idx, y_evt) / cfg.ACCUMULATION_STEPS
            
            if torch.isnan(loss).item():
                optimizer.zero_grad()
                continue
            
            loss.backward()
            
            if (step + 1) % cfg.ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * cfg.ACCUMULATION_STEPS * x_img.size(0)
        
        # Handle remaining gradients
        if (step + 1) % cfg.ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()
        train_loss = running_loss / max(1, len(ds_tr))
        
        # Validation
        model.eval()
        all_surv = []
        for batch in dl_va:
            surv = predict_surv_tta4(
                model, batch["image"], batch["clinical"], device,
                n_time_bins=n_time_bins, use_tta=cfg.USE_TTA,
            )
            all_surv.append(surv)
        surv = np.concatenate(all_surv, axis=0)
        
        met = compute_metrics(times_tr, events_tr, times_va, events_va, surv, cuts)
        
        brier_str = f"{met['brier_24m']:.3f}" if not np.isnan(met['brier_24m']) else "N/A"
        print(f"    Ep {epoch+1:03d} | loss={train_loss:.4f} | "
              f"Uno={met['uno']:.3f} Harrell={met['harrell']:.3f} Brier@24m={brier_str}", flush=True)
        
        # Check for improvement
        if np.isnan(met["uno"]):
            no_improve += 1
        elif met["uno"] > best_uno + 1e-5:
            best_uno = met["uno"]
            best_epoch = epoch
            no_improve = 0
            
            # Save checkpoint
            ckpt = {
                "variant": variant_name,
                "use_clinical": bool(use_clinical),
                "state_dict": model.state_dict(),
                "cuts": cuts.tolist(),
                "epoch": int(epoch),
                "uno": float(met["uno"]),
                "harrell": float(met["harrell"]),
                "tau": float(met["tau"]),
                "lr": float(cfg.LR),
                "use_tta": bool(cfg.USE_TTA),
                "clinical_features": feature_names,
                "clinical_dim": int(X_tr.shape[1]),
                "n_time_bins": int(n_time_bins),
                "clinical_defaults": defaults,
                "cfg": asdict(cfg),
                "clin_cfg": asdict(clin_cfg),
            }
            torch.save(ckpt, best_path)
            
            # Save validation predictions
            pd.DataFrame({
                "patient_id": [r["patient_id"] for r in records_va],
                "risk": met["risk"],
                "surv_24m": met["surv_24m"],
                "time": times_va,
                "event": events_va,
            }).to_csv(out_dir / "val_pred.csv", index=False)
        else:
            no_improve += 1
        
        if no_improve >= cfg.EARLY_STOP:
            print(f"    Early stop at epoch {epoch+1} (no improvement for {no_improve} epochs)", flush=True)
            break
    
    return {
        "variant": variant_name,
        "use_clinical": bool(use_clinical),
        "best_uno": float(best_uno),
        "best_epoch": int(best_epoch),
        "ckpt_path": str(best_path),
    }


# ============================================================================
# Column Name Mapping
# ============================================================================
def standardize_clinical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize clinical column names."""
    col_map = {}
    for col in df.columns:
        c = col.lower().strip()
        if 'patient' in c or 'case' in c:
            col_map[col] = 'patient_id'
        elif 'survival' in c and 'time' in c:
            col_map[col] = 'time'
        elif 'dead' in c or ('event' in c and 'time' not in c):
            col_map[col] = 'event'
        elif 'overall' in c and 'stage' in c:
            col_map[col] = 'overall_stage'
        elif ('.t.' in c or c.startswith('t.') or '.t' in c or c == 't') and 'stage' in c:
            col_map[col] = 'T'
        elif ('.n.' in c or c.startswith('n.') or '.n' in c or c == 'n') and 'stage' in c:
            col_map[col] = 'N'
        elif ('.m.' in c or c.startswith('m.') or '.m' in c or c == 'm') and 'stage' in c:
            col_map[col] = 'M'
        elif 'age' in c and 'stage' not in c:
            col_map[col] = 'age'
        elif 'gender' in c or 'sex' in c:
            col_map[col] = 'gender'
    
    return df.rename(columns=col_map)


# ============================================================================
# Main Training Function
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase3 Do-No-Harm Training")
    parser.add_argument("--crop_log", required=True, help="LUNG1 Phase2 crop_log.csv")
    parser.add_argument("--clinical_csv", required=True, help="LUNG1 clinical CSV")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--exp_name", default=None, help="Experiment name")
    parser.add_argument("--fold", type=int, default=None, help="Run single fold")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device ID")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--n_folds", type=int, default=None, help="Number of CV folds (default: 5)")
    parser.add_argument("--no_tta", action="store_true", help="Disable TTA")
    parser.add_argument("--drop_stage_prob", type=float, default=None)
    parser.add_argument("--drop_tnm_prob", type=float, default=None)
    parser.add_argument("--drop_all_clin_prob", type=float, default=None)
    args = parser.parse_args()
    
    # GPU selection
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Override configs
    if args.lr is not None:
        cfg.LR = float(args.lr)
    if args.n_folds is not None:
        cfg.N_FOLDS = int(args.n_folds)
    if args.no_tta:
        cfg.USE_TTA = False
    if args.drop_stage_prob is not None:
        clin_cfg.drop_stage_prob = float(args.drop_stage_prob)
    if args.drop_tnm_prob is not None:
        clin_cfg.drop_tnm_prob = float(args.drop_tnm_prob)
    if args.drop_all_clin_prob is not None:
        clin_cfg.drop_all_clin_prob = float(args.drop_all_clin_prob)
    
    set_seed(cfg.SEED)
    
    # Setup output directory
    exp_name = args.exp_name or f"lr_{cfg.LR:.0e}_donoharm"
    out_root = Path(args.out_dir) / exp_name
    out_root.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("PHASE3 DO-NO-HARM TRAINING")
    print("=" * 80)
    print(f"Experiment: {exp_name}")
    print(f"Crop log: {args.crop_log}")
    print(f"Clinical: {args.clinical_csv}")
    print(f"Output: {out_root}")
    print(f"LR: {cfg.LR}")
    print(f"TTA: {cfg.USE_TTA}")
    print(f"Missing Aug - stage: {clin_cfg.drop_stage_prob}, TNM: {clin_cfg.drop_tnm_prob}, all: {clin_cfg.drop_all_clin_prob}")
    print("=" * 80)
    
    # Load data
    crop = pd.read_csv(args.crop_log)
    clin = pd.read_csv(args.clinical_csv)
    
    # Validate crop_log
    for c in ["patient_id", "out_ct", "out_edt"]:
        if c not in crop.columns:
            raise ValueError(f"crop_log missing column: {c}")
    
    if "norm" in crop.columns:
        norms = crop["norm"].astype(str).unique().tolist()
        if len(norms) != 1 or norms[0] != "window":
            print(f"WARNING: Expected norm=='window', got: {norms}")
    
    # Standardize clinical columns
    clin = standardize_clinical_columns(clin)
    
    # Merge
    manifest = crop[["patient_id", "out_ct", "out_edt"]].rename(columns={
        "out_ct": "ct_path",
        "out_edt": "edt_path",
    }).merge(clin, on="patient_id", how="inner")
    
    manifest = manifest.dropna(subset=["time", "event"]).copy()
    manifest["time"] = pd.to_numeric(manifest["time"], errors="coerce").astype(float)
    manifest["event"] = pd.to_numeric(manifest["event"], errors="coerce").astype(int)
    
    print(f"\nDataset: n={len(manifest)} | events={manifest['event'].sum()} ({manifest['event'].mean()*100:.1f}%)")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    
    results = []
    selection = {}
    
    for fold, (idx_tr, idx_va) in enumerate(skf.split(np.zeros(len(manifest)), manifest["event"].values)):
        if args.fold is not None and fold != args.fold:
            continue
        
        fold_dir = out_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        times = manifest["time"].values.astype(float)
        events = manifest["event"].values.astype(int)
        
        # Determine time bins
        n_bins = adaptive_bins(times[idx_tr], events[idx_tr])
        cuts = make_time_bins(times[idx_tr], events[idx_tr], n_bins)
        
        # Create survival labels
        lab = LabTransDiscreteTime(cuts=cuts[1:-1])
        y_tr = lab.fit_transform(times[idx_tr], events[idx_tr])
        y_va = lab.transform(times[idx_va], events[idx_va])
        
        n_time_bins = len(cuts) - 1
        print(f"\n{'='*80}")
        print(f"FOLD {fold} | train={len(idx_tr)} val={len(idx_va)} | bins={n_time_bins}")
        print(f"{'='*80}")
        
        # Prepare clinical features
        print("  Clinical feature processing:")
        X_tr, X_va, feat_names, defaults = prepare_clinical_features_fold(
            manifest, idx_tr, idx_va, verbose=True
        )
        
        # Create record lists
        rec_tr = [
            {"patient_id": manifest.iloc[i].patient_id,
             "ct": manifest.iloc[i].ct_path,
             "edt": manifest.iloc[i].edt_path}
            for i in idx_tr
        ]
        rec_va = [
            {"patient_id": manifest.iloc[i].patient_id,
             "ct": manifest.iloc[i].ct_path,
             "edt": manifest.iloc[i].edt_path}
            for i in idx_va
        ]
        
        # Train image-only variant
        res_img = train_one_variant(
            variant_name="model_img",
            use_clinical=False,
            fold_dir=fold_dir,
            records_tr=rec_tr, records_va=rec_va,
            X_tr=X_tr, X_va=X_va,
            y_tr_idx=y_tr[0], y_tr_evt=y_tr[1],
            y_va_idx=y_va[0], y_va_evt=y_va[1],
            feature_names=feat_names, defaults=defaults,
            cuts=cuts,
            times_tr=times[idx_tr], events_tr=events[idx_tr],
            times_va=times[idx_va], events_va=events[idx_va],
        )
        
        # Train image+clinical variant
        res_imgclin = train_one_variant(
            variant_name="model_imgclin",
            use_clinical=True,
            fold_dir=fold_dir,
            records_tr=rec_tr, records_va=rec_va,
            X_tr=X_tr, X_va=X_va,
            y_tr_idx=y_tr[0], y_tr_evt=y_tr[1],
            y_va_idx=y_va[0], y_va_evt=y_va[1],
            feature_names=feat_names, defaults=defaults,
            cuts=cuts,
            times_tr=times[idx_tr], events_tr=events[idx_tr],
            times_va=times[idx_va], events_va=events[idx_va],
        )
        
        # Select winner (do-no-harm)
        if res_imgclin["best_uno"] >= res_img["best_uno"]:
            chosen = res_imgclin
        else:
            chosen = res_img
        
        # Copy winner to fold root
        shutil.copy2(Path(chosen["ckpt_path"]), fold_dir / "best.pt")
        
        # Save selection info
        sel_info = {
            "fold": fold,
            "chosen_variant": chosen["variant"],
            "chosen_uno": chosen["best_uno"],
            "img_uno": res_img["best_uno"],
            "imgclin_uno": res_imgclin["best_uno"],
            "improvement": res_imgclin["best_uno"] - res_img["best_uno"],
        }
        with open(fold_dir / "selection.json", "w") as f:
            json.dump(sel_info, f, indent=2)
        
        selection[str(fold)] = sel_info
        results.append({
            "fold": fold,
            "uno_img": res_img["best_uno"],
            "uno_imgclin": res_imgclin["best_uno"],
            "chosen": chosen["variant"],
            "chosen_uno": chosen["best_uno"],
        })
        
        print(f"\n  >> FOLD {fold} Winner: {chosen['variant']} (Uno={chosen['best_uno']:.3f})")
        print(f"     img={res_img['best_uno']:.3f} vs imgclin={res_imgclin['best_uno']:.3f}")
    
    # Save overall results
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_root / "results.csv", index=False)
    
    with open(out_root / "selection_summary.json", "w") as f:
        json.dump(selection, f, indent=2)
    
    # Save configs
    with open(out_root / "config.json", "w") as f:
        json.dump({
            "train_cfg": asdict(cfg),
            "clin_cfg": asdict(clin_cfg),
        }, f, indent=2)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    if len(results_df) > 0:
        mean_img = results_df['uno_img'].mean()
        mean_imgclin = results_df['uno_imgclin'].mean()
        mean_chosen = results_df['chosen_uno'].mean()
        std_chosen = results_df['chosen_uno'].std()
        
        print(f"\nImage-only mean: {mean_img:.3f}")
        print(f"Image+Clinical mean: {mean_imgclin:.3f}")
        print(f"Do-No-Harm chosen mean: {mean_chosen:.3f} ± {std_chosen:.3f}")
        
        imgclin_wins = (results_df['chosen'] == 'model_imgclin').sum()
        print(f"Clinical helped in {imgclin_wins}/{len(results_df)} folds")
    
    print(f"\nOutput saved to: {out_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()