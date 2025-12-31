#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 Training (Do-No-Harm) - v3 Complete
===========================================================
完整版本，包含以下改进：
1. 使用MONAI LoadImaged读取.nii.gz文件
2. 严格对齐LUNG1 clinical CSV列名
3. Overall Stage映射只包含实际存在的值 (I, II, IIIa, IIIb)
4. 确定性4-view flip TTA
5. 保持32→64 with LayerNorm的Clinical Encoder
6. 在ckpt中保存val_risk_mean/std
7. Warmup (5 epochs) + Cosine Annealing LR调度
8. 梯度累积 (effective batch = 4)
9. 温和的Missing Augmentation (15%/5%/5%)
10. Brier Score计算
11. GroupNorm和Do-No-Harm选择

Usage:
    python phase3_train_v3.py \
        --crop_log /path/to/lung1/crop_log.csv \
        --clinical_csv /path/to/lung1/clinical.csv \
        --out_dir /path/to/output \
        --exp_name lr_7e-4_v3 \
        --gpu 0
"""

import os
import sys
import json
import math
import time
import random
import shutil
import argparse
import warnings
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, asdict, field
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score

# MONAI
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    RandFlipd, RandAffined, RandGaussianNoised, RandShiftIntensityd,
    ConcatItemsd, DeleteItemsd, Lambdad,
)
from monai.networks.nets import resnet as monai_resnet

# pycox
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
try:
    from pycox.models.loss import NLLLogisticHazardLoss as _Loss
except ImportError:
    from pycox.models.loss import NLLLogistiHazardLoss as _Loss

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class CFG:
    n_folds: int = 5
    seed: int = 0
    
    # training
    lr: float = 7e-4
    lr_min: float = 7e-6  # Cosine最小LR (1/100 of lr)
    weight_decay: float = 1e-4
    batch_size: int = 4
    accumulation_steps: int = 2  # 有效batch = 4
    num_workers: int = 4
    max_epochs: int = 100
    patience: int = 30
    grad_clip: float = 2.0
    
    # LR schedule
    warmup_epochs: int = 5
    
    # survival
    n_time_bins: int = 13
    
    # augmentation
    use_aug: bool = True
    
    # TTA (确定性4-view)
    use_tta: bool = True
    
    # dropout
    dropout: float = 0.35
    
    # Missing Augmentation (温和版本)
    drop_stage_prob: float = 0.70  # 之前是0.50 刚才是0.10
    drop_tnm_prob: float = 0.05    # 之前是0.25
    drop_all_clin_prob: float = 0.03  # 之前是0.15


@dataclass
class ClinicalConfig:
    """Clinical feature configuration matching actual LUNG1 CSV"""
    # Stage ordinal mapping - ONLY values that exist in LUNG1
    # Distribution: IIIb=176, IIIa=112, I=93, II=40
    stage_ordinal_map: Dict[str, float] = field(default_factory=lambda: {
        "I": 0.2,
        "II": 0.4,
        "IIIA": 0.6,
        "IIIB": 0.8,
    })
    
    # T/N/M ranges (from actual data)
    t_min: float = 1.0
    t_max: float = 4.0
    n_min: float = 0.0
    n_max: float = 3.0


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# LR Scheduler with Warmup + Cosine Annealing
# ============================================================================
def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
):
    """
    Linear warmup + Cosine Annealing scheduler.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Clinical Feature Processing
# ============================================================================
def standardize_lung1_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize LUNG1 clinical CSV column names"""
    col_map = {
        "PatientID": "patient_id",
        "age": "age",
        "clinical.T.Stage": "T",
        "Clinical.N.Stage": "N",
        "Clinical.M.Stage": "M",
        "Overall.Stage": "overall_stage",
        "gender": "gender",
        "Survival.time": "time",
        "deadstatus.event": "event",
    }
    return df.rename(columns=col_map)


def _safe_mode(values: np.ndarray, default: float) -> float:
    """Get mode of values, return default if empty"""
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return float(default)
    cnt = Counter(vals.tolist())
    mode_val = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return float(mode_val)


def _parse_T(series: pd.Series, t_max: float = 4.0) -> np.ndarray:
    """Parse T stage (numeric 1-4, clamp 5 to 4)"""
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        try:
            t = float(v)
            if t < 1.0:
                out.append(np.nan)
            elif t > t_max:
                out.append(t_max)
            else:
                out.append(t)
        except (ValueError, TypeError):
            out.append(np.nan)
    return np.array(out, dtype=np.float32)


def _parse_N(series: pd.Series, n_max: float = 3.0) -> np.ndarray:
    """Parse N stage (numeric 0-3, clamp 4 to 3)"""
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        try:
            n = float(v)
            if n < 0.0:
                out.append(np.nan)
            elif n > n_max:
                out.append(n_max)
            else:
                out.append(n)
        except (ValueError, TypeError):
            out.append(np.nan)
    return np.array(out, dtype=np.float32)


def _parse_M(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Parse M stage: returns (M01 binary, M_misc for M>1)"""
    M01, M_misc = [], []
    for v in series:
        if pd.isna(v):
            M01.append(np.nan)
            M_misc.append(0.0)
            continue
        try:
            mv = float(v)
            if mv == 0:
                M01.append(0.0)
                M_misc.append(0.0)
            elif mv == 1:
                M01.append(1.0)
                M_misc.append(0.0)
            else:
                M01.append(1.0)
                M_misc.append(1.0)
        except (ValueError, TypeError):
            M01.append(np.nan)
            M_misc.append(0.0)
    return np.array(M01, dtype=np.float32), np.array(M_misc, dtype=np.float32)


def _parse_overall_stage_ordinal(series: pd.Series, stage_map: Dict[str, float]) -> np.ndarray:
    """Parse overall stage to ordinal values"""
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).strip().upper()
        if s in stage_map:
            out.append(stage_map[s])
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)


def prepare_clinical_features_fold(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    clin_cfg: ClinicalConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[str]]:
    """
    Build clinical features for training and validation.
    
    Features (12-dim):
      0. age_scaled
      1. age_missing
      2. gender_male
      3. T_norm
      4. T_missing
      5. N_norm
      6. N_missing
      7. M01
      8. M_missing
      9. M_misc
      10. overall_stage_ord
      11. overall_missing
    """
    feats_tr, feats_va = [], []
    defaults = {}
    
    # ==================== AGE ====================
    age_tr = pd.to_numeric(df_train["age"], errors="coerce").values.astype(np.float32)
    age_va = pd.to_numeric(df_val["age"], errors="coerce").values.astype(np.float32)
    
    age_impute = np.nanmedian(age_tr)
    if np.isnan(age_impute):
        age_impute = 65.0
    defaults["age_impute"] = float(age_impute)
    
    age_tr_missing = np.isnan(age_tr).astype(np.float32)
    age_va_missing = np.isnan(age_va).astype(np.float32)
    
    age_tr_filled = np.where(np.isnan(age_tr), age_impute, age_tr)
    age_va_filled = np.where(np.isnan(age_va), age_impute, age_va)
    
    scaler = StandardScaler()
    age_tr_scaled = scaler.fit_transform(age_tr_filled.reshape(-1, 1)).flatten().astype(np.float32)
    age_va_scaled = scaler.transform(age_va_filled.reshape(-1, 1)).flatten().astype(np.float32)
    
    defaults["age_mean"] = float(scaler.mean_[0])
    defaults["age_std"] = float(scaler.scale_[0])
    
    feats_tr.extend([age_tr_scaled, age_tr_missing])
    feats_va.extend([age_va_scaled, age_va_missing])
    
    # ==================== GENDER ====================
    def parse_gender(df):
        return (df["gender"].astype(str).str.upper().str[0] == "M").astype(np.float32).values
    
    feats_tr.append(parse_gender(df_train))
    feats_va.append(parse_gender(df_val))
    
    # ==================== T STAGE ====================
    T_tr = _parse_T(df_train["T"], clin_cfg.t_max)
    T_va = _parse_T(df_val["T"], clin_cfg.t_max)
    
    T_mode = _safe_mode(T_tr, default=2.0)
    defaults["T_mode"] = float(T_mode)
    
    T_tr_missing = np.isnan(T_tr).astype(np.float32)
    T_va_missing = np.isnan(T_va).astype(np.float32)
    
    T_tr_filled = np.where(np.isnan(T_tr), T_mode, T_tr)
    T_va_filled = np.where(np.isnan(T_va), T_mode, T_va)
    
    T_tr_norm = (T_tr_filled - clin_cfg.t_min) / (clin_cfg.t_max - clin_cfg.t_min)
    T_va_norm = (T_va_filled - clin_cfg.t_min) / (clin_cfg.t_max - clin_cfg.t_min)
    
    feats_tr.extend([T_tr_norm, T_tr_missing])
    feats_va.extend([T_va_norm, T_va_missing])
    
    # ==================== N STAGE ====================
    N_tr = _parse_N(df_train["N"], clin_cfg.n_max)
    N_va = _parse_N(df_val["N"], clin_cfg.n_max)
    
    N_mode = _safe_mode(N_tr, default=0.0)
    defaults["N_mode"] = float(N_mode)
    
    N_tr_missing = np.isnan(N_tr).astype(np.float32)
    N_va_missing = np.isnan(N_va).astype(np.float32)
    
    N_tr_filled = np.where(np.isnan(N_tr), N_mode, N_tr)
    N_va_filled = np.where(np.isnan(N_va), N_mode, N_va)
    
    N_tr_norm = N_tr_filled / clin_cfg.n_max
    N_va_norm = N_va_filled / clin_cfg.n_max
    
    feats_tr.extend([N_tr_norm, N_tr_missing])
    feats_va.extend([N_va_norm, N_va_missing])
    
    # ==================== M STAGE ====================
    M01_tr, M_misc_tr = _parse_M(df_train["M"])
    M01_va, M_misc_va = _parse_M(df_val["M"])
    
    M_mode = _safe_mode(M01_tr, default=0.0)
    defaults["M_mode"] = float(M_mode)
    
    M_tr_missing = np.isnan(M01_tr).astype(np.float32)
    M_va_missing = np.isnan(M01_va).astype(np.float32)
    
    M01_tr_filled = np.where(np.isnan(M01_tr), M_mode, M01_tr)
    M01_va_filled = np.where(np.isnan(M01_va), M_mode, M01_va)
    
    feats_tr.extend([M01_tr_filled, M_tr_missing, M_misc_tr])
    feats_va.extend([M01_va_filled, M_va_missing, M_misc_va])
    
    # ==================== OVERALL STAGE ====================
    ov_tr = _parse_overall_stage_ordinal(df_train["overall_stage"], clin_cfg.stage_ordinal_map)
    ov_va = _parse_overall_stage_ordinal(df_val["overall_stage"], clin_cfg.stage_ordinal_map)
    
    ov_mode = _safe_mode(ov_tr, default=0.6)
    defaults["overall_mode"] = float(ov_mode)
    
    ov_tr_missing = np.isnan(ov_tr).astype(np.float32)
    ov_va_missing = np.isnan(ov_va).astype(np.float32)
    
    ov_tr_filled = np.where(np.isnan(ov_tr), ov_mode, ov_tr)
    ov_va_filled = np.where(np.isnan(ov_va), ov_mode, ov_va)
    
    feats_tr.extend([ov_tr_filled, ov_tr_missing])
    feats_va.extend([ov_va_filled, ov_va_missing])
    
    # Stack features
    X_tr = np.stack(feats_tr, axis=1).astype(np.float32)
    X_va = np.stack(feats_va, axis=1).astype(np.float32)
    
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)
    
    feature_names = [
        "age_scaled", "age_missing",
        "gender_male",
        "T_norm", "T_missing",
        "N_norm", "N_missing",
        "M01", "M_missing", "M_misc",
        "overall_stage_ord", "overall_missing",
    ]
    
    return X_tr, X_va, defaults, feature_names


# ============================================================================
# Missing Augmentation (温和版本)
# ============================================================================
def clinical_missing_augment(
    x: np.ndarray,
    defaults: Dict[str, Any],
    drop_stage_prob: float = 0.15,
    drop_tnm_prob: float = 0.05,
    drop_all_clin_prob: float = 0.05,
) -> np.ndarray:
    """
    Apply missing augmentation to clinical features during training.
    
    Feature indices:
      0: age_scaled, 1: age_missing
      2: gender_male
      3: T_norm, 4: T_missing
      5: N_norm, 6: N_missing
      7: M01, 8: M_missing, 9: M_misc
      10: overall_stage_ord, 11: overall_missing
    """
    x = x.copy()
    
    # Small probability to drop ALL clinical features
    if random.random() < drop_all_clin_prob:
        # Zero all features except gender (always observable)
        x[0] = 0.0  # age_scaled
        x[1] = 1.0  # age_missing
        x[3] = (defaults["T_mode"] - 1.0) / 3.0  # T_norm default
        x[4] = 1.0  # T_missing
        x[5] = defaults["N_mode"] / 3.0  # N_norm default
        x[6] = 1.0  # N_missing
        x[7] = defaults["M_mode"]  # M01 default
        x[8] = 1.0  # M_missing
        x[9] = 0.0  # M_misc
        x[10] = defaults["overall_mode"]  # overall_stage_ord default
        x[11] = 1.0  # overall_missing
        return x
    
    # Drop overall stage with probability
    if random.random() < drop_stage_prob:
        x[10] = defaults["overall_mode"]
        x[11] = 1.0
    
    # Drop T/N/M with probability
    if random.random() < drop_tnm_prob:
        x[3] = (defaults["T_mode"] - 1.0) / 3.0
        x[4] = 1.0
    
    if random.random() < drop_tnm_prob:
        x[5] = defaults["N_mode"] / 3.0
        x[6] = 1.0
    
    if random.random() < drop_tnm_prob:
        x[7] = defaults["M_mode"]
        x[8] = 1.0
        x[9] = 0.0
    
    return x


# ============================================================================
# Data Transforms
# ============================================================================
def build_train_transforms(use_aug: bool = True) -> Compose:
    """Build training transforms with MONAI LoadImaged for .nii.gz"""
    keys = ["ct", "edt"]
    
    base_tfms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        EnsureTyped(keys=keys, dtype=torch.float32, track_meta=False),
        Lambdad(keys=keys, func=lambda x: torch.nan_to_num(x, 0.0, 0.0, 0.0)),
        Lambdad(keys=keys, func=lambda x: torch.clamp(x, 0.0, 1.0)),
    ]
    
    if use_aug:
        aug_tfms = [
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandAffined(
                keys=keys,
                prob=0.25,
                rotate_range=(0.15, 0.15, 0.15),
                translate_range=(5, 5, 5),
                scale_range=(0.08, 0.08, 0.08),
                padding_mode="border",
            ),
            RandGaussianNoised(keys=keys, prob=0.15, mean=0.0, std=0.02),
            RandShiftIntensityd(keys=keys, prob=0.15, offsets=0.05),
        ]
    else:
        aug_tfms = []
    
    final_tfms = [
        ConcatItemsd(keys=keys, name="image"),
        DeleteItemsd(keys=keys),
    ]
    
    return Compose(base_tfms + aug_tfms + final_tfms)


def build_val_transforms() -> Compose:
    """Build validation transforms (no augmentation)"""
    keys = ["ct", "edt"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        EnsureTyped(keys=keys, dtype=torch.float32, track_meta=False),
        Lambdad(keys=keys, func=lambda x: torch.nan_to_num(x, 0.0, 0.0, 0.0)),
        Lambdad(keys=keys, func=lambda x: torch.clamp(x, 0.0, 1.0)),
        ConcatItemsd(keys=keys, name="image"),
        DeleteItemsd(keys=keys),
    ])


# ============================================================================
# Dataset
# ============================================================================
class SurvivalDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        x_clin: np.ndarray,
        y_disc: np.ndarray,
        y_event: np.ndarray,
        transforms: Compose,
        use_clinical: bool = True,
        is_train: bool = False,
        defaults: Optional[Dict] = None,
        cfg: Optional[CFG] = None,
    ):
        self.records = records
        self.X = x_clin.astype(np.float32)
        self.y_disc = y_disc.astype(np.int64)
        self.y_event = y_event.astype(np.float32)
        self.transforms = transforms
        self.use_clinical = use_clinical
        self.is_train = is_train
        self.defaults = defaults or {}
        self.cfg = cfg
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        data = {"ct": record["ct_path"], "edt": record["edt_path"]}
        data = self.transforms(data)
        
        x = self.X[idx].copy()
        
        # Apply missing augmentation only during training for imgclin variant
        if self.is_train and self.use_clinical and self.cfg is not None and self.defaults:
            x = clinical_missing_augment(
                x, self.defaults,
                drop_stage_prob=self.cfg.drop_stage_prob,
                drop_tnm_prob=self.cfg.drop_tnm_prob,
                drop_all_clin_prob=self.cfg.drop_all_clin_prob,
            )
        
        if not self.use_clinical:
            x = np.zeros_like(x, dtype=np.float32)
        
        return {
            "image": data["image"],
            "x": torch.from_numpy(x),
            "t": torch.tensor(self.y_disc[idx], dtype=torch.long),
            "e": torch.tensor(self.y_event[idx], dtype=torch.float32),
        }


# ============================================================================
# Model Architecture
# ============================================================================
def replace_bn_with_gn(module: nn.Module, num_groups: int = 8) -> nn.Module:
    """Replace BatchNorm with GroupNorm for small batch stability"""
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            num_channels = child.num_features
            groups = min(num_groups, num_channels)
            while groups > 1 and num_channels % groups != 0:
                groups //= 2
            setattr(module, name, nn.GroupNorm(groups, num_channels))
        else:
            replace_bn_with_gn(child, num_groups)
    return module


class ResNet10_Clinical_GN(nn.Module):
    """
    ResNet10 backbone with GroupNorm + Clinical encoder.
    Clinical encoder: 32→64 with LayerNorm
    """
    
    def __init__(self, in_channels: int, clinical_dim: int, n_bins: int, dropout: float = 0.35):
        super().__init__()
        
        self.backbone = monai_resnet.resnet10(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=512,
        )
        replace_bn_with_gn(self.backbone)
        
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
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
            nn.Linear(128, n_bins),
        )
    
    def forward(self, x_img: torch.Tensor, x_clin: torch.Tensor) -> torch.Tensor:
        f_img = self.backbone(x_img)
        f_clin = self.clinical_encoder(x_clin)
        f = torch.cat([f_img, f_clin], dim=1)
        return self.head(f)


# ============================================================================
# Inference with Deterministic 4-View TTA
# ============================================================================
@torch.no_grad()
def predict_with_tta4(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_bins: int,
    use_tta: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inference with deterministic 4-view flip TTA.
    Returns (survival curves, risk scores)
    """
    model.eval()
    all_surv = []
    
    for batch in loader:
        img = batch["image"].to(device)
        x = batch["x"].to(device)
        
        if use_tta:
            views = [
                img,
                torch.flip(img, dims=[-2]),
                torch.flip(img, dims=[-1]),
                torch.flip(img, dims=[-2, -1]),
            ]
            
            survs = []
            for v in views:
                logits = model(v, x)
                haz = torch.sigmoid(logits[:, :n_bins])
                surv = torch.cumprod(1.0 - haz, dim=1)
                survs.append(surv)
            
            surv = torch.stack(survs, dim=0).mean(dim=0)
        else:
            logits = model(img, x)
            haz = torch.sigmoid(logits[:, :n_bins])
            surv = torch.cumprod(1.0 - haz, dim=1)
        
        all_surv.append(surv.cpu().numpy())
    
    surv = np.concatenate(all_surv, axis=0)
    risk = -np.sum(surv, axis=1)
    
    return surv, risk


# ============================================================================
# Metrics (including Brier Score)
# ============================================================================
def to_structured_y(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    return np.array(
        [(bool(e), float(t)) for t, e in zip(times, events)],
        dtype=[("event", bool), ("time", float)]
    )


def compute_metrics(
    times_tr: np.ndarray,
    events_tr: np.ndarray,
    times_va: np.ndarray,
    events_va: np.ndarray,
    surv: np.ndarray,
    risk: np.ndarray,
    cuts: np.ndarray,
) -> Dict[str, float]:
    """Compute Harrell, Uno C-index, and Brier Score"""
    y_tr = to_structured_y(times_tr, events_tr)
    y_va = to_structured_y(times_va, events_va)
    
    # Harrell C-index
    try:
        harrell = float(concordance_index_censored(
            events_va.astype(bool), times_va.astype(float), risk
        )[0])
    except Exception:
        harrell = float("nan")
    
    # Uno C-index (IPCW)
    evt_times = times_tr[events_tr == 1]
    tau = float(np.quantile(evt_times, 0.9)) if len(evt_times) > 0 else float(np.max(times_tr))
    
    try:
        uno = float(concordance_index_ipcw(y_tr, y_va, risk, tau=tau)[0])
    except Exception:
        uno = float("nan")
    
    # Brier Score at 24 months (730 days)
    try:
        n_bins = surv.shape[1]
        bin_mids = (cuts[:-1] + cuts[1:]) / 2.0 if len(cuts) > 1 else cuts
        
        t_24m = 730.0
        t_idx = int(np.argmin(np.abs(bin_mids - t_24m)))
        surv_24m = surv[:, t_idx] if t_idx < n_bins else surv[:, -1]
        
        _, brier_scores = brier_score(y_tr, y_va, surv_24m.reshape(-1, 1), np.array([t_24m]))
        brier_24m = float(brier_scores[0])
    except Exception:
        brier_24m = float("nan")
    
    return {
        "harrell": harrell,
        "uno": uno,
        "tau": tau,
        "brier_24m": brier_24m,
    }


# ============================================================================
# Training One Variant
# ============================================================================
def train_one_variant(
    fold_idx: int,
    variant_name: str,
    use_clinical: bool,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    x_tr: np.ndarray,
    x_va: np.ndarray,
    cuts: np.ndarray,
    defaults: Dict[str, Any],
    cfg: CFG,
    out_dir: Path,
    device: torch.device,
) -> Tuple[Path, Dict[str, Any]]:
    """Train a single variant and return (best_ckpt_path, best_metrics)"""
    
    # Discrete-time labels
    durations_tr = df_train["time"].values.astype(np.float32)
    events_tr = df_train["event"].values.astype(np.int32)
    durations_va = df_val["time"].values.astype(np.float32)
    events_va = df_val["event"].values.astype(np.int32)
    
    labtrans = LabTransDiscreteTime(cuts)
    y_tr = labtrans.transform(durations_tr, events_tr)
    y_va = labtrans.transform(durations_va, events_va)
    
    # Build records
    records_tr = [
        {"ct_path": row["out_ct"], "edt_path": row["out_edt"]}
        for _, row in df_train.iterrows()
    ]
    records_va = [
        {"ct_path": row["out_ct"], "edt_path": row["out_edt"]}
        for _, row in df_val.iterrows()
    ]
    
    # Datasets and loaders
    ds_tr = SurvivalDataset(
        records_tr, x_tr, y_tr[0], y_tr[1],
        build_train_transforms(cfg.use_aug),
        use_clinical=use_clinical,
        is_train=True,
        defaults=defaults,
        cfg=cfg,
    )
    ds_va = SurvivalDataset(
        records_va, x_va, y_va[0], y_va[1],
        build_val_transforms(),
        use_clinical=use_clinical,
        is_train=False,
    )
    
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True)
    
    # Model
    clinical_dim = x_tr.shape[1]
    model = ResNet10_Clinical_GN(
        in_channels=2,
        clinical_dim=clinical_dim,
        n_bins=cfg.n_time_bins,
        dropout=cfg.dropout
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = _Loss()
    
    # LR Scheduler: Warmup + Cosine Annealing
    steps_per_epoch = len(dl_tr) // cfg.accumulation_steps
    total_steps = steps_per_epoch * cfg.max_epochs
    warmup_steps = steps_per_epoch * cfg.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, min_lr_ratio=cfg.lr_min / cfg.lr
    )
    
    # Training loop
    best_uno = -1e9
    best_epoch = -1
    best_path = out_dir / "best.pt"
    bad_epochs = 0
    global_step = 0
    
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(dl_tr):
            img = batch["image"].to(device)
            x = batch["x"].to(device)
            t_disc = batch["t"].to(device)
            e = batch["e"].to(device)
            
            logits = model(img, x)
            loss = criterion(logits, t_disc, e)
            loss = loss / cfg.accumulation_steps
            loss.backward()
            
            losses.append(loss.item() * cfg.accumulation_steps)
            
            # Gradient accumulation
            if (batch_idx + 1) % cfg.accumulation_steps == 0:
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        
        # Handle remaining gradients
        if (batch_idx + 1) % cfg.accumulation_steps != 0:
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Validation
        surv_va, risk_va = predict_with_tta4(model, dl_va, device, cfg.n_time_bins, cfg.use_tta)
        met = compute_metrics(durations_tr, events_tr, durations_va, events_va, surv_va, risk_va, cuts)
        
        current_lr = optimizer.param_groups[0]["lr"]
        ep_time = time.time() - t0
        print(f"  [fold {fold_idx}][{variant_name}] epoch {epoch:03d} "
              f"loss={np.mean(losses):.4f} uno={met['uno']:.4f} harrell={met['harrell']:.4f} "
              f"brier={met['brier_24m']:.4f} lr={current_lr:.2e} time={ep_time:.1f}s")
        
        # Best model selection
        if np.isfinite(met["uno"]) and met["uno"] > best_uno + 1e-6:
            best_uno = met["uno"]
            best_epoch = epoch
            bad_epochs = 0
            
            ckpt = {
                "state_dict": model.state_dict(),
                "fold": int(fold_idx),
                "variant": variant_name,
                "use_clinical": bool(use_clinical),
                "cfg": asdict(cfg),
                "cuts": cuts.astype(float).tolist(),
                "n_time_bins": int(cfg.n_time_bins),
                "clinical_dim": int(clinical_dim),
                "best_epoch": int(best_epoch),
                "metrics": {
                    "uno": float(met["uno"]),
                    "harrell": float(met["harrell"]),
                    "tau": float(met["tau"]),
                    "brier_24m": float(met["brier_24m"]),
                },
                "val_risk_mean": float(np.mean(risk_va)),
                "val_risk_std": float(np.std(risk_va) + 1e-8),
                "val_risk_median": float(np.median(risk_va)),
                "val_risk_iqr": float(np.subtract(*np.percentile(risk_va, [75, 25])) + 1e-8),
                "clinical_defaults": defaults,
                "clinical_features": [
                    "age_scaled", "age_missing", "gender_male",
                    "T_norm", "T_missing", "N_norm", "N_missing",
                    "M01", "M_missing", "M_misc",
                    "overall_stage_ord", "overall_missing",
                ],
            }
            torch.save(ckpt, best_path)
            
            val_pred = df_val[["patient_id", "event", "time"]].copy()
            val_pred["risk"] = risk_va.astype(np.float32)
            val_pred.to_csv(out_dir / "val_pred.csv", index=False)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"  [fold {fold_idx}][{variant_name}] Early stop at epoch {epoch} (best={best_epoch})")
                break
    
    # Load best metrics
    best_ckpt = torch.load(best_path, map_location="cpu")
    best_met = {
        "uno": float(best_ckpt["metrics"]["uno"]),
        "harrell": float(best_ckpt["metrics"]["harrell"]),
        "tau": float(best_ckpt["metrics"]["tau"]),
        "brier_24m": float(best_ckpt["metrics"]["brier_24m"]),
        "best_epoch": int(best_ckpt["best_epoch"]),
        "use_clinical": bool(best_ckpt["use_clinical"]),
        "variant": str(best_ckpt["variant"]),
        "val_risk_mean": float(best_ckpt.get("val_risk_mean", float("nan"))),
        "val_risk_std": float(best_ckpt.get("val_risk_std", float("nan"))),
    }
    
    return best_path, best_met


# ============================================================================
# Main Training
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 3 Training (Do-No-Harm) v3")
    parser.add_argument("--crop_log", required=True, help="LUNG1 phase2 crop_log.csv")
    parser.add_argument("--clinical_csv", required=True, help="LUNG1 clinical CSV")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--exp_name", default="lr_7e-4_v3", help="Experiment name")
    
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_time_bins", type=int, default=13)
    
    # Missing augmentation probabilities
    parser.add_argument("--drop_stage_prob", type=float, default=0.15)
    parser.add_argument("--drop_tnm_prob", type=float, default=0.05)
    parser.add_argument("--drop_all_clin_prob", type=float, default=0.05)
    
    parser.add_argument("--no_tta", action="store_true", help="Disable TTA")
    parser.add_argument("--no_aug", action="store_true", help="Disable augmentation")
    parser.add_argument("--no_missing_aug", action="store_true", help="Disable missing augmentation")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index")
    parser.add_argument("--fold", type=int, default=None, help="Run single fold only")
    parser.add_argument("--n_folds", type=int, default=None, help="Number of CV folds (default: 5)")
    
    args = parser.parse_args()
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # If no_missing_aug, set all drop probs to 0
    drop_stage = 0.0 if args.no_missing_aug else args.drop_stage_prob
    drop_tnm = 0.0 if args.no_missing_aug else args.drop_tnm_prob
    drop_all = 0.0 if args.no_missing_aug else args.drop_all_clin_prob
    
    # Override n_folds if provided
    n_folds_override = args.n_folds if args.n_folds is not None else 5
    
    cfg = CFG(
        n_folds=n_folds_override,
        lr=args.lr,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        max_epochs=args.epochs,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        seed=args.seed,
        n_time_bins=args.n_time_bins,
        use_tta=(not args.no_tta),
        use_aug=(not args.no_aug),
        drop_stage_prob=drop_stage,
        drop_tnm_prob=drop_tnm,
        drop_all_clin_prob=drop_all,
    )
    clin_cfg = ClinicalConfig()
    
    seed_everything(cfg.seed)
    
    exp_dir = Path(args.out_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    with open(exp_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    
    # Load data
    df_crop = pd.read_csv(args.crop_log)
    df_clin = pd.read_csv(args.clinical_csv)
    
    df_clin = standardize_lung1_columns(df_clin)
    
    df = df_crop.merge(df_clin, on="patient_id", how="inner")
    df = df.dropna(subset=["out_ct", "out_edt", "time", "event"]).reset_index(drop=True)
    df["time"] = pd.to_numeric(df["time"], errors="coerce").astype(float)
    df["event"] = pd.to_numeric(df["event"], errors="coerce").astype(int)
    
    print("\n" + "=" * 80)
    print(f"LUNG1 Training Data: n={len(df)} | events={df['event'].sum()} ({df['event'].mean()*100:.1f}%)")
    print(f"Missing Augmentation: drop_stage={cfg.drop_stage_prob:.0%}, drop_tnm={cfg.drop_tnm_prob:.0%}")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    y_strat = df["event"].astype(int).values
    
    fold_summaries = []
    
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(df, y_strat)):
        if args.fold is not None and fold_idx != args.fold:
            continue
        
        print("\n" + "=" * 80)
        print(f"FOLD {fold_idx}")
        print("=" * 80)
        
        df_tr = df.iloc[tr_idx].copy().reset_index(drop=True)
        df_va = df.iloc[va_idx].copy().reset_index(drop=True)
        
        durations_tr = df_tr["time"].values.astype(np.float32)
        events_tr = df_tr["event"].values.astype(np.int32)
        
        labtrans = LabTransDiscreteTime(cfg.n_time_bins)
        labtrans.fit(durations_tr, events_tr)
        cuts = labtrans.cuts
        
        x_tr, x_va, defaults, feat_names = prepare_clinical_features_fold(df_tr, df_va, clin_cfg)
        
        fold_dir = exp_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        with open(fold_dir / "clinical_info.json", "w") as f:
            json.dump({
                "defaults": defaults,
                "feature_names": feat_names,
                "cuts": cuts.tolist(),
            }, f, indent=2)
        
        results = {}
        
        for variant_name, use_clin in [("model_img", False), ("model_imgclin", True)]:
            var_dir = fold_dir / variant_name
            var_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n[{variant_name}] use_clinical={use_clin}")
            best_path, best_met = train_one_variant(
                fold_idx=fold_idx,
                variant_name=variant_name,
                use_clinical=use_clin,
                df_train=df_tr,
                df_val=df_va,
                x_tr=x_tr,
                x_va=x_va,
                cuts=cuts,
                defaults=defaults,
                cfg=cfg,
                out_dir=var_dir,
                device=device,
            )
            results[variant_name] = {"ckpt_path": str(best_path), "metrics": best_met}
        
        # Do-No-Harm selection
        uno_img = results["model_img"]["metrics"]["uno"]
        uno_imgclin = results["model_imgclin"]["metrics"]["uno"]
        
        if np.isfinite(uno_imgclin) and uno_imgclin >= uno_img:
            winner = "model_imgclin"
        else:
            winner = "model_img"
        
        win_met = results[winner]["metrics"]
        
        print(f"\nFOLD {fold_idx} Winner: {winner} (Uno={win_met['uno']:.4f}, Brier={win_met['brier_24m']:.4f})")
        print(f"  img: Uno={uno_img:.4f}")
        print(f"  imgclin: Uno={uno_imgclin:.4f}")
        
        selection = {
            "fold": int(fold_idx),
            "winner": winner,
            "val_uno": float(win_met["uno"]),
            "val_harrell": float(win_met["harrell"]),
            "val_brier_24m": float(win_met["brier_24m"]),
            "tau": float(win_met["tau"]),
            "best_epoch": int(win_met["best_epoch"]),
            "use_clinical": bool(win_met["use_clinical"]),
            "img_uno": float(uno_img),
            "imgclin_uno": float(uno_imgclin),
            "val_risk_mean": float(win_met.get("val_risk_mean", float("nan"))),
            "val_risk_std": float(win_met.get("val_risk_std", float("nan"))),
        }
        
        with open(fold_dir / "selection.json", "w") as f:
            json.dump(selection, f, indent=2)
        
        src = fold_dir / winner / "best.pt"
        dst = fold_dir / "best.pt"
        if dst.exists():
            dst.unlink()
        shutil.copy(src, dst)
        
        fold_summaries.append(selection)
    
    if fold_summaries:
        pd.DataFrame(fold_summaries).to_csv(exp_dir / "fold_selection_summary.csv", index=False)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        for s in fold_summaries:
            print(f"  Fold {s['fold']}: {s['winner']} Uno={s['val_uno']:.4f} Brier={s['val_brier_24m']:.4f} "
                  f"(img={s['img_uno']:.4f}, imgclin={s['imgclin_uno']:.4f})")
        
        mean_uno = np.mean([s["val_uno"] for s in fold_summaries])
        std_uno = np.std([s["val_uno"] for s in fold_summaries])
        mean_brier = np.mean([s["val_brier_24m"] for s in fold_summaries if np.isfinite(s["val_brier_24m"])])
        
        print(f"\nMean Uno: {mean_uno:.4f} ± {std_uno:.4f}")
        print(f"Mean Brier@24m: {mean_brier:.4f}")
        print(f"\nSaved to: {exp_dir}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()