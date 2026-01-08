#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 External Validation - v3 Complete
===========================================================
完整版本，包含以下改进：
1. 使用MONAI LoadImaged读取.nii.gz文件
2. 严格对齐两个数据集的Clinical CSV列名
3. 确定性4-view flip TTA
4. 保持32→64 with LayerNorm的Clinical Encoder（与Phase 3一致）
5. 多种Ensemble策略：
   - equal: 等权平均
   - zavg: z-score标准化后等权平均
   - zavg_weighted: z-score标准化后按val_uno加权平均
6. Z-score来源选项：
   - lung1_val: 使用ckpt中保存的val_risk_mean/std（更严谨）
   - rg: 使用RG测试集自身的mean/std（可能更高）
7. Brier Score计算
8. 详细的per-fold和ensemble指标报告

Usage:
    python phase4_external_test_v3.py \
        --lung1_crop_log /path/to/lung1/crop_log.csv \
        --lung1_clinical /path/to/lung1/clinical.csv \
        --ckpt_root /path/to/phase3_outputs/exp_name \
        --rg_crop_log /path/to/rg/phase2_crop_log.csv \
        --rg_clinical /path/to/rg/clinical.csv \
        --out_dir /path/to/output \
        --ensemble zavg_weighted \
        --z_source lung1_val \
        --gpu 0
"""

import os
import sys
import math
import argparse
import warnings
import json
import re
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConcatItemsd, DeleteItemsd, Lambdad,
)
from monai.networks.nets import resnet as monai_resnet

from sksurv.metrics import concordance_index_ipcw, concordance_index_censored, brier_score

warnings.filterwarnings("ignore")


# ============================================================================
# Model Architecture (must match Phase 3)
# ============================================================================
def replace_bn_with_gn(module: nn.Module, num_groups: int = 8) -> nn.Module:
    """Replace BatchNorm with GroupNorm"""
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
    """ResNet10 backbone with GroupNorm + Clinical encoder (32→64 with LayerNorm)."""
    
    def __init__(self, in_channels: int, clinical_dim: int, n_bins: int, dropout: float = 0.35):
        super().__init__()
        
        self.backbone = monai_resnet.resnet10(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=512,
        )
        replace_bn_with_gn(self.backbone)
        
        # Clinical encoder: 32→64 with LayerNorm (与Phase 3一致)
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
# Clinical Feature Processing (must match Phase 3)
# ============================================================================
def _safe_mode(values: np.ndarray, default: float) -> float:
    """Get mode of values, return default if empty"""
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return float(default)
    cnt = Counter(vals.tolist())
    mode_val = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return float(mode_val)


def _parse_T(series: pd.Series, t_max: float = 4.0) -> np.ndarray:
    """Parse T stage (numeric 1-4)"""
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).upper()
        m = re.search(r'(\d)', s)
        if m:
            t = float(m.group(1))
            if t < 1.0:
                out.append(np.nan)
            elif t > t_max:
                out.append(t_max)
            else:
                out.append(t)
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)


def _parse_N(series: pd.Series, n_max: float = 3.0) -> np.ndarray:
    """Parse N stage (numeric 0-3)"""
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).upper()
        m = re.search(r'(\d)', s)
        if m:
            n = float(m.group(1))
            if n < 0.0:
                out.append(np.nan)
            elif n > n_max:
                out.append(n_max)
            else:
                out.append(n)
        else:
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


# Stage ordinal mapping - 只包含实际存在的值
STAGE_ORDINAL_MAP = {
    "I": 0.2,
    "II": 0.4,
    "IIIA": 0.6,
    "IIIB": 0.8,
    "IV": 1.0,  # RG可能有IV期
}


def _parse_overall_stage_ordinal(series: pd.Series) -> np.ndarray:
    """Parse overall stage to ordinal values"""
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).strip().upper().replace("STAGE", "").replace(" ", "")
        # Handle roman numerals
        s = (s.replace("Ⅲ", "III").replace("Ⅱ", "II")
              .replace("Ⅳ", "IV").replace("Ⅰ", "I")
              .replace("ⅢA", "IIIA").replace("ⅢB", "IIIB"))
        if s in STAGE_ORDINAL_MAP:
            out.append(STAGE_ORDINAL_MAP[s])
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)


def prepare_clinical_external(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: List[str],
    defaults: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare clinical features for external validation.
    Uses training set statistics for imputation and scaling.
    
    Features (12-dim):
      0. age_scaled, 1. age_missing
      2. gender_male
      3. T_norm, 4. T_missing
      5. N_norm, 6. N_missing
      7. M01, 8. M_missing, 9. M_misc
      10. overall_stage_ord, 11. overall_missing
    """
    feats_tr, feats_te = [], []
    
    # ==================== AGE ====================
    if 'age_scaled' in feature_names:
        age_tr = pd.to_numeric(train_df['age'], errors='coerce').values.astype(np.float32)
        age_te = pd.to_numeric(test_df['age'], errors='coerce').values.astype(np.float32)
        
        age_impute = defaults.get('age_impute', np.nanmedian(age_tr))
        if np.isnan(age_impute):
            age_impute = 65.0
        
        age_tr_missing = np.isnan(age_tr).astype(np.float32)
        age_te_missing = np.isnan(age_te).astype(np.float32)
        
        age_tr_filled = np.where(np.isnan(age_tr), age_impute, age_tr)
        age_te_filled = np.where(np.isnan(age_te), age_impute, age_te)
        
        age_mean = defaults.get('age_mean', np.mean(age_tr_filled))
        age_std = defaults.get('age_std', np.std(age_tr_filled) + 1e-8)
        
        age_tr_scaled = ((age_tr_filled - age_mean) / age_std).astype(np.float32)
        age_te_scaled = ((age_te_filled - age_mean) / age_std).astype(np.float32)
        
        feats_tr.extend([age_tr_scaled, age_tr_missing])
        feats_te.extend([age_te_scaled, age_te_missing])
    
    # ==================== GENDER ====================
    if 'gender_male' in feature_names:
        def parse_gender(df):
            return (df['gender'].astype(str).str.upper().str[0] == 'M').astype(np.float32).values
        
        feats_tr.append(parse_gender(train_df))
        feats_te.append(parse_gender(test_df))
    
    # ==================== T STAGE ====================
    if 'T_norm' in feature_names:
        T_col_tr = 'T' if 'T' in train_df.columns else None
        T_col_te = 'T' if 'T' in test_df.columns else None
        
        T_tr = _parse_T(train_df[T_col_tr] if T_col_tr else pd.Series([np.nan]*len(train_df)))
        T_te = _parse_T(test_df[T_col_te] if T_col_te else pd.Series([np.nan]*len(test_df)))
        
        T_mode = defaults.get('T_mode', _safe_mode(T_tr, default=2.0))
        
        T_tr_missing = np.isnan(T_tr).astype(np.float32)
        T_te_missing = np.isnan(T_te).astype(np.float32)
        
        T_tr_filled = np.where(np.isnan(T_tr), T_mode, T_tr)
        T_te_filled = np.where(np.isnan(T_te), T_mode, T_te)
        
        T_tr_norm = (T_tr_filled - 1.0) / 3.0
        T_te_norm = (T_te_filled - 1.0) / 3.0
        
        feats_tr.extend([T_tr_norm, T_tr_missing])
        feats_te.extend([T_te_norm, T_te_missing])
    
    # ==================== N STAGE ====================
    if 'N_norm' in feature_names:
        N_col_tr = 'N' if 'N' in train_df.columns else None
        N_col_te = 'N' if 'N' in test_df.columns else None
        
        N_tr = _parse_N(train_df[N_col_tr] if N_col_tr else pd.Series([np.nan]*len(train_df)))
        N_te = _parse_N(test_df[N_col_te] if N_col_te else pd.Series([np.nan]*len(test_df)))
        
        N_mode = defaults.get('N_mode', _safe_mode(N_tr, default=0.0))
        
        N_tr_missing = np.isnan(N_tr).astype(np.float32)
        N_te_missing = np.isnan(N_te).astype(np.float32)
        
        N_tr_filled = np.where(np.isnan(N_tr), N_mode, N_tr)
        N_te_filled = np.where(np.isnan(N_te), N_mode, N_te)
        
        N_tr_norm = N_tr_filled / 3.0
        N_te_norm = N_te_filled / 3.0
        
        feats_tr.extend([N_tr_norm, N_tr_missing])
        feats_te.extend([N_te_norm, N_te_missing])
    
    # ==================== M STAGE ====================
    if 'M01' in feature_names:
        M_col_tr = 'M' if 'M' in train_df.columns else None
        M_col_te = 'M' if 'M' in test_df.columns else None
        
        M01_tr, M_misc_tr = _parse_M(train_df[M_col_tr] if M_col_tr else pd.Series([np.nan]*len(train_df)))
        M01_te, M_misc_te = _parse_M(test_df[M_col_te] if M_col_te else pd.Series([np.nan]*len(test_df)))
        
        M_mode = defaults.get('M_mode', _safe_mode(M01_tr, default=0.0))
        
        M_tr_missing = np.isnan(M01_tr).astype(np.float32)
        M_te_missing = np.isnan(M01_te).astype(np.float32)
        
        M01_tr_filled = np.where(np.isnan(M01_tr), M_mode, M01_tr)
        M01_te_filled = np.where(np.isnan(M01_te), M_mode, M01_te)
        
        feats_tr.extend([M01_tr_filled, M_tr_missing, M_misc_tr])
        feats_te.extend([M01_te_filled, M_te_missing, M_misc_te])
    
    # ==================== OVERALL STAGE ====================
    if 'overall_stage_ord' in feature_names:
        ov_col_tr = 'overall_stage' if 'overall_stage' in train_df.columns else None
        ov_col_te = 'overall_stage' if 'overall_stage' in test_df.columns else None
        
        ov_tr = _parse_overall_stage_ordinal(
            train_df[ov_col_tr] if ov_col_tr else pd.Series([np.nan]*len(train_df))
        )
        ov_te = _parse_overall_stage_ordinal(
            test_df[ov_col_te] if ov_col_te else pd.Series([np.nan]*len(test_df))
        )
        
        ov_mode = defaults.get('overall_mode', _safe_mode(ov_tr, default=0.6))
        
        ov_tr_missing = np.isnan(ov_tr).astype(np.float32)
        ov_te_missing = np.isnan(ov_te).astype(np.float32)
        
        ov_tr_filled = np.where(np.isnan(ov_tr), ov_mode, ov_tr)
        ov_te_filled = np.where(np.isnan(ov_te), ov_mode, ov_te)
        
        feats_tr.extend([ov_tr_filled, ov_tr_missing])
        feats_te.extend([ov_te_filled, ov_te_missing])
    
    # Assemble
    if feats_tr:
        X_tr = np.stack(feats_tr, axis=1).astype(np.float32)
        X_te = np.stack(feats_te, axis=1).astype(np.float32)
    else:
        X_tr = np.zeros((len(train_df), 1), dtype=np.float32)
        X_te = np.zeros((len(test_df), 1), dtype=np.float32)
    
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_tr, X_te


# ============================================================================
# Dataset and Transforms
# ============================================================================
class ExternalTestDataset(Dataset):
    def __init__(self, records: List[dict], clinical: np.ndarray, transform: Compose):
        self.records = records
        self.clinical = clinical
        self.transform = transform
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, i):
        data = {"ct": self.records[i]["ct"], "edt": self.records[i]["edt"]}
        data = self.transform(data)
        data["clinical"] = torch.tensor(self.clinical[i], dtype=torch.float32)
        data["patient_id"] = self.records[i]["patient_id"]
        return data


def build_test_transforms() -> Compose:
    """Build test transforms with MONAI LoadImaged for .nii.gz"""
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
# Inference with Deterministic 4-View TTA
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
    """
    Inference with deterministic 4-view flip TTA.
    Returns survival curves.
    """
    model.eval()
    
    if not use_tta:
        logits = model(img_batch.to(device), clin_batch.to(device))
        haz = torch.sigmoid(logits[:, :n_time_bins])
        surv = torch.cumprod(1.0 - haz, dim=1)
        return surv.cpu().numpy()
    
    # Deterministic 4-view flip TTA
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
# Metrics
# ============================================================================
def to_structured_y(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    return np.array(
        [(bool(e), float(t)) for t, e in zip(times, events)],
        dtype=[('event', bool), ('time', float)]
    )


def compute_external_metrics(
    times_tr: np.ndarray,
    events_tr: np.ndarray,
    times_te: np.ndarray,
    events_te: np.ndarray,
    surv: np.ndarray,
    cuts: np.ndarray,
) -> Dict[str, Any]:
    """Compute Harrell, Uno C-index, and Brier Score"""
    y_tr = to_structured_y(times_tr, events_tr)
    y_te = to_structured_y(times_te, events_te)
    
    n_time_bins = surv.shape[1]
    bin_mids = (cuts[:-1] + cuts[1:]) / 2.0 if len(cuts) > 1 else cuts
    
    # Risk = negative integral of survival
    risk = -np.trapz(surv, x=bin_mids, axis=1) if len(bin_mids) == n_time_bins else -np.sum(surv, axis=1)
    
    # Tau for IPCW
    evt_times = times_tr[events_tr == 1]
    tau = float(np.quantile(evt_times, 0.9)) if len(evt_times) > 0 else float(np.max(times_tr))
    
    # Uno C-index
    try:
        uno = float(concordance_index_ipcw(y_tr, y_te, risk, tau=tau)[0])
    except Exception:
        uno = float("nan")
    
    # Harrell C-index
    try:
        harrell = float(concordance_index_censored(y_te["event"], y_te["time"], risk)[0])
    except Exception:
        harrell = float("nan")
    
    # Brier Score at 24 months
    try:
        t_idx_24m = int(np.argmin(np.abs(bin_mids - 730))) if len(bin_mids) == n_time_bins else n_time_bins // 2
        surv_24m = surv[:, t_idx_24m] if t_idx_24m < n_time_bins else surv[:, -1]
        _, brier = brier_score(y_tr, y_te, surv_24m.reshape(-1, 1), np.array([730]))
        brier_24m = float(brier[0])
    except Exception:
        brier_24m = float("nan")
        surv_24m = surv[:, n_time_bins // 2]
    
    return {
        "uno": uno,
        "harrell": harrell,
        "tau": tau,
        "brier_24m": brier_24m,
        "risk": risk,
        "surv_24m": surv_24m,
    }


# ============================================================================
# Column Standardization
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


def standardize_rg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Radiogenomics clinical CSV column names (flexible)"""
    col_map = {}
    for col in df.columns:
        c = col.lower().strip()
        if 'patient' in c or 'case' in c:
            col_map[col] = 'patient_id'
        elif 'survival' in c and 'time' in c:
            col_map[col] = 'time'
        elif c in ['time', 'time_days', 'os', 'os_days', 'survival_time']:
            col_map[col] = 'time'
        elif 'dead' in c or ('event' in c and 'time' not in c) or c == 'status':
            col_map[col] = 'event'
        elif 'overall' in c and 'stage' in c:
            col_map[col] = 'overall_stage'
        elif ('clinical' in c or 'pathologic' in c) and '.t' in c:
            col_map[col] = 'T'
        elif ('clinical' in c or 'pathologic' in c) and '.n' in c:
            col_map[col] = 'N'
        elif ('clinical' in c or 'pathologic' in c) and '.m' in c:
            col_map[col] = 'M'
        elif c == 't' or c == 't_stage':
            col_map[col] = 'T'
        elif c == 'n' or c == 'n_stage':
            col_map[col] = 'N'
        elif c == 'm' or c == 'm_stage':
            col_map[col] = 'M'
        elif 'age' in c and 'stage' not in c:
            col_map[col] = 'age'
        elif 'gender' in c or 'sex' in c:
            col_map[col] = 'gender'
    return df.rename(columns=col_map)


# ============================================================================
# Ensemble Methods
# ============================================================================
def ensemble_risks(
    all_risks: List[np.ndarray],
    all_val_unos: List[float],
    all_val_risk_means: List[float],
    all_val_risk_stds: List[float],
    method: str = "zavg_weighted",
    z_source: str = "lung1_val",
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Ensemble multiple fold risks.
    
    Methods:
    - equal: Simple average
    - zavg: Z-score normalize then average
    - zavg_weighted: Z-score normalize then weighted average by val_uno
    
    Z-source:
    - lung1_val: Use val_risk_mean/std from ckpt (training fold validation)
    - rg: Use RG test set's own mean/std
    """
    if len(all_risks) == 0:
        return np.array([])
    
    if len(all_risks) == 1:
        return all_risks[0]
    
    risks_arr = np.stack(all_risks, axis=0)  # (n_folds, n_samples)
    n_folds = len(all_risks)
    
    if method == "equal":
        # Simple average
        return np.mean(risks_arr, axis=0)
    
    # Z-score normalization
    z_risks = np.zeros_like(risks_arr)
    for i in range(n_folds):
        if z_source == "lung1_val":
            mu = all_val_risk_means[i]
            std = all_val_risk_stds[i]
        else:  # rg
            mu = np.mean(risks_arr[i])
            std = np.std(risks_arr[i]) + 1e-8
        z_risks[i] = (risks_arr[i] - mu) / std
    
    if method == "zavg":
        return np.mean(z_risks, axis=0)
    
    elif method == "zavg_weighted":
        # Softmax weights based on val_uno
        valid_unos = np.array([u if np.isfinite(u) else 0.5 for u in all_val_unos])
        weights = np.exp((valid_unos - np.max(valid_unos)) / temperature)
        weights = weights / np.sum(weights)
        
        print(f"    Ensemble weights (temperature={temperature}):")
        for i, (w, u) in enumerate(zip(weights, all_val_unos)):
            print(f"      Fold {i}: weight={w:.3f} (val_uno={u:.4f})")
        
        # Weighted average
        return np.sum(z_risks * weights[:, np.newaxis], axis=0)
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


# ============================================================================
# Load Manifest (for reuse in other scripts)
# ============================================================================
def load_manifest(crop_log_path: str, clinical_csv_path: str, dataset_label: str = "LUNG1") -> pd.DataFrame:
    """
    Load and merge crop_log and clinical data, returning a standardized DataFrame.
    
    Args:
        crop_log_path: Path to crop_log.csv (phase2 output)
        clinical_csv_path: Path to clinical CSV file
        dataset_label: "LUNG1" or "RG" (affects column standardization)
    
    Returns:
        DataFrame with columns: patient_id, time, event, out_ct, out_edt, out_mask 
        (or ct_path, edt_path for RG), and all clinical features
    """
    crop_df = pd.read_csv(crop_log_path)
    
    if dataset_label == "LUNG1":
        clin_df = standardize_lung1_columns(pd.read_csv(clinical_csv_path))
        # Merge
        merged = crop_df[["patient_id"]].merge(clin_df, on="patient_id", how="inner")
        merged = merged.dropna(subset=["time", "event"])
        merged["time"] = pd.to_numeric(merged["time"], errors="coerce").astype(float)
        merged["event"] = pd.to_numeric(merged["event"], errors="coerce").astype(int)
        # Ensure out_ct, out_edt, and out_mask columns exist (needed for radiomics extraction)
        cols_to_merge = ["patient_id"]
        for col in ["out_ct", "out_edt", "out_mask"]:
            if col in crop_df.columns:
                cols_to_merge.append(col)
        if len(cols_to_merge) > 1:  # More than just patient_id
            merged = merged.merge(crop_df[cols_to_merge], on="patient_id", how="left")
    else:  # RG
        clin_df = standardize_rg_columns(pd.read_csv(clinical_csv_path))
        # Merge and rename columns (include out_mask if available)
        cols_to_merge = ["patient_id", "out_ct", "out_edt"]
        if "out_mask" in crop_df.columns:
            cols_to_merge.append("out_mask")
        merged = crop_df[cols_to_merge].rename(columns={
            "out_ct": "ct_path",
            "out_edt": "edt_path",
        }).merge(clin_df, on="patient_id", how="inner")
        # Check required columns
        if "time" not in merged.columns:
            raise ValueError(f"RG missing 'time' column. Available: {list(merged.columns)[:20]}")
        if "event" not in merged.columns:
            raise ValueError(f"RG missing 'event' column. Available: {list(merged.columns)[:20]}")
        merged = merged.dropna(subset=["time", "event"])
        merged["time"] = pd.to_numeric(merged["time"], errors="coerce").astype(float)
        merged["event"] = pd.to_numeric(merged["event"], errors="coerce").astype(int)
        # For RG, also add out_ct, out_edt, and out_mask for compatibility
        if "out_ct" not in merged.columns:
            merged["out_ct"] = merged["ct_path"]
        if "out_edt" not in merged.columns:
            merged["out_edt"] = merged["edt_path"]
        # out_mask should already be in merged if it was in crop_df (not renamed)
        # If it was renamed to mask_path, restore it
        if "out_mask" not in merged.columns:
            if "mask_path" in merged.columns:
                merged["out_mask"] = merged["mask_path"]
            elif "out_mask" in crop_df.columns:
                # Merge it back from original crop_df
                merged = merged.merge(crop_df[["patient_id", "out_mask"]], on="patient_id", how="left")
    
    return merged


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="External Validation on Radiogenomics v3")
    parser.add_argument("--lung1_crop_log", required=True)
    parser.add_argument("--lung1_clinical", required=True)
    parser.add_argument("--ckpt_root", required=True, help="Path to do-no-harm experiment folder")
    parser.add_argument("--rg_crop_log", required=True)
    parser.add_argument("--rg_clinical", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--no_tta", action="store_true")
    
    # Ensemble options
    parser.add_argument("--ensemble", type=str, default="zavg_weighted",
                        choices=["equal", "zavg", "zavg_weighted"],
                        help="Ensemble method")
    parser.add_argument("--z_source", type=str, default="lung1_val",
                        choices=["lung1_val", "rg"],
                        help="Source for z-score normalization")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature for weighted ensemble")
    
    args = parser.parse_args()
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("EXTERNAL VALIDATION: LUNG1 -> Radiogenomics")
    print(f"Ensemble: {args.ensemble} | Z-source: {args.z_source}")
    print("=" * 80)
    
    # Load LUNG1 data (for IPCW weighting)
    lung1_crop = pd.read_csv(args.lung1_crop_log)
    lung1_clin = standardize_lung1_columns(pd.read_csv(args.lung1_clinical))
    
    lung1 = lung1_crop[["patient_id"]].merge(lung1_clin, on="patient_id", how="inner")
    lung1 = lung1.dropna(subset=["time", "event"])
    lung1["time"] = pd.to_numeric(lung1["time"], errors="coerce").astype(float)
    lung1["event"] = pd.to_numeric(lung1["event"], errors="coerce").astype(int)
    
    times_tr = lung1["time"].values
    events_tr = lung1["event"].values
    
    print(f"LUNG1: n={len(lung1)} events={events_tr.sum()}")
    
    # Load RG data
    rg_crop = pd.read_csv(args.rg_crop_log)
    rg_clin = standardize_rg_columns(pd.read_csv(args.rg_clinical))
    
    rg = rg_crop[["patient_id", "out_ct", "out_edt"]].rename(columns={
        "out_ct": "ct_path",
        "out_edt": "edt_path",
    }).merge(rg_clin, on="patient_id", how="inner")
    
    # Check columns
    if "time" not in rg.columns:
        raise ValueError(f"RG missing 'time' column. Available: {list(rg.columns)[:20]}")
    if "event" not in rg.columns:
        raise ValueError(f"RG missing 'event' column. Available: {list(rg.columns)[:20]}")
    
    rg = rg.dropna(subset=["time", "event"])
    rg["time"] = pd.to_numeric(rg["time"], errors="coerce").astype(float)
    rg["event"] = pd.to_numeric(rg["event"], errors="coerce").astype(int)
    
    times_te = rg["time"].values
    events_te = rg["event"].values
    
    print(f"RG: n={len(rg)} events={events_te.sum()}")
    
    # Find folds
    ckpt_root = Path(args.ckpt_root)
    fold_dirs = sorted([d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    
    print(f"Found {len(fold_dirs)} folds")
    
    all_results = []
    all_risks = []
    all_val_unos = []
    all_val_risk_means = []
    all_val_risk_stds = []
    
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        ckpt_path = fold_dir / "best.pt"
        
        if not ckpt_path.exists():
            print(f"  {fold_name}: No checkpoint found, skipping")
            continue
        
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        use_clinical = ckpt.get("use_clinical", True)
        n_time_bins = ckpt.get("n_time_bins", 13)
        clinical_dim = ckpt.get("clinical_dim", 12)
        cuts = np.array(ckpt.get("cuts", []))
        feature_names = ckpt.get("clinical_features", [
            "age_scaled", "age_missing", "gender_male",
            "T_norm", "T_missing", "N_norm", "N_missing",
            "M01", "M_missing", "M_misc",
            "overall_stage_ord", "overall_missing",
        ])
        defaults = ckpt.get("clinical_defaults", {})
        use_tta = ckpt.get("cfg", {}).get("use_tta", True) and not args.no_tta
        variant = ckpt.get("variant", "unknown")
        
        # Val risk statistics (for z-score)
        # Try multiple ways to get val_uno (compatible with different checkpoint structures)
        # 1. phase3_fixed_previous.py: ckpt["metrics"]["uno"]
        # 2. phase3_train_donoharm.py: ckpt["uno"] (top-level)
        # 3. Fallback: selection.json["val_uno"] or selection.json["chosen_uno"]
        val_uno = float("nan")
        if "metrics" in ckpt and isinstance(ckpt["metrics"], dict) and "uno" in ckpt["metrics"]:
            val_uno = float(ckpt["metrics"]["uno"])
        elif "uno" in ckpt:
            val_uno = float(ckpt["uno"])
        else:
            # Try reading from selection.json as fallback
            selection_path = fold_dir / "selection.json"
            if selection_path.exists():
                try:
                    sel = json.load(open(selection_path))
                    val_uno = float(sel.get("val_uno", sel.get("chosen_uno", float("nan"))))
                except Exception:
                    pass
        
        val_risk_mean = ckpt.get("val_risk_mean", 0.0)
        val_risk_std = ckpt.get("val_risk_std", 1.0)
        
        print(f"\n  {fold_name}: variant={variant}, use_clinical={use_clinical}, "
              f"val_uno={val_uno:.4f}, bins={n_time_bins}")
        
        # Prepare clinical features
        _, X_te = prepare_clinical_external(lung1, rg, feature_names, defaults)
        
        # If checkpoint is image-only variant, zero out clinical
        if not use_clinical:
            X_te = np.zeros_like(X_te, dtype=np.float32)
            print(f"    -> Image-only variant: zeroing clinical features")
        
        # Create model and load weights
        model = ResNet10_Clinical_GN(
            in_channels=2,
            clinical_dim=clinical_dim,
            n_bins=n_time_bins,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        
        # Create dataset and loader
        records = [
            {"patient_id": row.patient_id, "ct": row.ct_path, "edt": row.edt_path}
            for _, row in rg.iterrows()
        ]
        ds = ExternalTestDataset(records, X_te, build_test_transforms())
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        # Inference
        all_surv = []
        for batch in dl:
            surv = predict_surv_tta4(
                model, batch["image"], batch["clinical"],
                device, n_time_bins, use_tta=use_tta,
            )
            all_surv.append(surv)
        surv = np.concatenate(all_surv, axis=0)
        
        # Compute metrics
        met = compute_external_metrics(times_tr, events_tr, times_te, events_te, surv, cuts)
        
        print(f"    Uno={met['uno']:.4f} Harrell={met['harrell']:.4f} Brier@24m={met['brier_24m']:.4f}")
        
        all_results.append({
            "fold": fold_name,
            "variant": variant,
            "use_clinical": use_clinical,
            "val_uno": val_uno,
            "rg_uno": met["uno"],
            "rg_harrell": met["harrell"],
            "rg_brier_24m": met["brier_24m"],
        })
        all_risks.append(met["risk"])
        all_val_unos.append(val_uno)
        all_val_risk_means.append(val_risk_mean)
        all_val_risk_stds.append(val_risk_std)
        
        # Save per-fold predictions
        pd.DataFrame({
            "patient_id": rg["patient_id"].values,
            "risk": met["risk"],
            "surv_24m": met["surv_24m"],
            "time": times_te,
            "event": events_te,
        }).to_csv(out_dir / f"{fold_name}_predictions.csv", index=False)
    
    # Ensemble
    if len(all_risks) > 0:
        print("\n" + "=" * 80)
        print(f"ENSEMBLE RESULTS ({args.ensemble}, z_source={args.z_source})")
        print("=" * 80)
        
        ensemble_risk = ensemble_risks(
            all_risks,
            all_val_unos,
            all_val_risk_means,
            all_val_risk_stds,
            method=args.ensemble,
            z_source=args.z_source,
            temperature=args.temperature,
        )
        
        # Compute ensemble metrics
        y_tr = to_structured_y(times_tr, events_tr)
        y_te = to_structured_y(times_te, events_te)
        
        evt_times = times_tr[events_tr == 1]
        tau = float(np.quantile(evt_times, 0.9))
        
        try:
            ensemble_uno = float(concordance_index_ipcw(y_tr, y_te, ensemble_risk, tau=tau)[0])
        except Exception:
            ensemble_uno = float("nan")
        
        try:
            ensemble_harrell = float(concordance_index_censored(y_te["event"], y_te["time"], ensemble_risk)[0])
        except Exception:
            ensemble_harrell = float("nan")
        
        print(f"\nEnsemble Uno C-index: {ensemble_uno:.4f}")
        print(f"Ensemble Harrell C-index: {ensemble_harrell:.4f}")
        
        # Save ensemble predictions
        pd.DataFrame({
            "patient_id": rg["patient_id"].values,
            "risk_ensemble": ensemble_risk,
            "time": times_te,
            "event": events_te,
        }).to_csv(out_dir / "ensemble_predictions.csv", index=False)
        
        # Also compute metrics for other ensemble methods for comparison
        print("\n--- Ensemble Method Comparison ---")
        for method in ["equal", "zavg", "zavg_weighted"]:
            for z_src in ["lung1_val", "rg"]:
                if method == "equal" and z_src == "rg":
                    continue  # equal doesn't use z_source
                
                risk_cmp = ensemble_risks(
                    all_risks, all_val_unos,
                    all_val_risk_means, all_val_risk_stds,
                    method=method, z_source=z_src, temperature=args.temperature,
                )
                try:
                    uno_cmp = float(concordance_index_ipcw(y_tr, y_te, risk_cmp, tau=tau)[0])
                except Exception:
                    uno_cmp = float("nan")
                
                key = f"{method}" if method == "equal" else f"{method}_{z_src}"
                print(f"  {key}: Uno={uno_cmp:.4f}")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "fold_results.csv", index=False)
    
    summary = {
        "n_test": len(rg),
        "n_events": int(events_te.sum()),
        "ensemble_method": args.ensemble,
        "z_source": args.z_source,
        "ensemble_uno": float(ensemble_uno) if len(all_risks) > 0 else None,
        "ensemble_harrell": float(ensemble_harrell) if len(all_risks) > 0 else None,
        "per_fold_rg_uno_mean": float(results_df["rg_uno"].mean()),
        "per_fold_rg_uno_std": float(results_df["rg_uno"].std()),
        "per_fold_rg_brier_mean": float(results_df["rg_brier_24m"].mean()),
    }
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("PER-FOLD RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\nPer-fold RG Uno: {results_df['rg_uno'].mean():.4f} ± {results_df['rg_uno'].std():.4f}")
    print(f"Per-fold RG Brier@24m: {results_df['rg_brier_24m'].mean():.4f}")
    print(f"\nResults saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()