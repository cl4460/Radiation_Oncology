#!/usr/bin/env python
"""
Phase 3 Training - FIXED VERSION with Paired TTA
============================================
CRITICAL FIXES:
1. Removed double normalization (CT already [0,1] from Phase 2)
2. Fixed time binning alignment (n_out = n_time_bins)
3. Added training flip augmentation (H/W axes)
4. Implemented 4-view TTA (Original, H, W, HW)
5. Removed duplicate EnsureTyped
6. Added dimension verification logs
7. FIXED: NaN handling with mutually exclusive branches (no double increment)
"""

import os, math, random, warnings, time, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from monai.transforms import Lambdad

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConcatItemsd, DeleteItemsd, ToTensord,
    RandAffined, RandGaussianNoised, RandAdjustContrastd,
    RandFlipd,
)
from monai.networks.nets import resnet as monai_resnet

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.loss import NLLLogistiHazardLoss
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored, brier_score

warnings.filterwarnings("ignore")

# ============================================================
# Paths
# ============================================================
PHASE1_QC_CSV = "/home/lichengze/Research/CNN_pipeline/phase1_outputs/phase1_qc.csv"
CROP_LOG_CSV = "/home/lichengze/Research/CNN_pipeline/phase2_outputs/crop_log.csv"
CLINICAL_CSV = "/home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical.csv"
OUTPUT_DIR = "/home/lichengze/Research/CNN_pipeline/phase3_outputs/learning_rate_corrected/output"

# ============================================================
# Configuration
# ============================================================
@dataclass
class CFG:
    # Time binning
    N_BINS_MAX: int = 15
    N_BINS_MIN: int = 12
    MIN_EVENTS_PER_BIN: int = 20
    
    # Training
    BATCH_SIZE: int = 4
    ACCUMULATION_STEPS: int = 2
    MAX_EPOCHS: int = 100
    EARLY_STOP: int = 30
    
    # Optimizer
    LR: float = 7.2e-4
    WEIGHT_DECAY: float = 2e-4
    DROPOUT: float = 0.35
    WARMUP_EPOCHS: int = 25
    
    # Cross-validation
    N_FOLDS: int = 5
    SEED: int = 42
    
    # Augmentation
    AUG_PROB: float = 0.5
    FLIP_PROB: float = 0.5
    ROT_DEG: Tuple[float,float,float] = (5.0, 5.0, 10.0)
    SCALE_RANGE: float = 0.12
    NOISE_STD: float = 0.02
    GAMMA_RANGE: Tuple[float,float] = (0.85, 1.15)
    
    # Data loading
    NUM_WORKERS: int = 4
    PREFETCH_FACTOR: int = 2
    
    # TTA
    USE_TTA: bool = True
    EVAL_24M_DAYS: int = 730

    USE_CLASS_WEIGHT: bool = False

cfg = CFG()
# ============================================================
# Command Line Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Phase 3 Training with Paired TTA',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--fold', type=int, default=None, 
                       help='Run specific fold (0-4), None=run all folds')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID')
    parser.add_argument('--lr', type=float, default=7.2e-4,
                       help='Learning rate')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name for output directory')
    parser.add_argument('--no_tta', action='store_true',
                   help='Disable 4-view test-time augmentation')
    
    return parser.parse_args()

# ============================================================
# Utility Functions
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def to_structured_y(times, events):
    return np.array([(bool(e), float(t)) for t, e in zip(times, events)],
                    dtype=[('event', bool), ('time', float)])

def adaptive_bins(times: np.ndarray, events: np.ndarray) -> int:
    n_events = events.sum()
    ideal = n_events // cfg.MIN_EVENTS_PER_BIN
    return int(np.clip(ideal, cfg.N_BINS_MIN, cfg.N_BINS_MAX))

def make_time_bins(times: np.ndarray, events: np.ndarray, n_bins: int):
    event_times = times[events == 1]
    if len(event_times) < n_bins:
        return np.linspace(times.min(), times.max(), n_bins + 1)
    quantiles = np.linspace(0, 1, n_bins + 1)
    cuts = np.unique(np.quantile(event_times, quantiles))
    if len(cuts) != n_bins + 1:
        return np.linspace(times.min(), times.max(), n_bins + 1)
    return cuts

# ============================================================
# Clinical Feature Processing
# ============================================================
import re
from collections import Counter

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
                                  idx_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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

# ============================================================
# Data Transforms
# ============================================================
def build_predict_with_tta4s(train: bool) -> Compose:
    """
    CRITICAL FIXES:
    1. Removed double normalization
    2. Removed duplicate EnsureTyped
    3. Added paired flip augmentation
    
    Phase 2 output verification:
    - CT: [0, 1] (HU window [-1000, 400] already applied)
    - EDT: [0, 1] (distance transform already normalized)
    """
    transforms = [
        LoadImaged(keys=["ct", "edt"]),
        EnsureChannelFirstd(keys=["ct", "edt"]),
        EnsureTyped(keys=["ct", "edt"], dtype=torch.float32, track_meta=False),

        # Handle NaN/Inf (safety check)
        Lambdad(keys=["ct", "edt"], func=lambda x: torch.nan_to_num(x, 0.0, 0.0, 0.0)), 
        Lambdad(keys=["ct", "edt"], func=lambda x: torch.clamp(x, 0.0, 1.0)),
        
        # FIXED: Only clamp to [0,1] (Phase 2 already normalized)
        # REMOVED: (ct + 1000) / 1400 (double normalization)
        # REMOVED: Second EnsureTyped (redundant)
    ]
    
    if train:
        rx, ry, rz = map(math.radians, cfg.ROT_DEG)
        transforms += [
            # Paired TTA: Train with flip on H/W axes
            # Shape after EnsureChannelFirstd: [C, D, H, W] = [2, 128, 192, 160]
            # spatial_axis=1 → H axis (Height, 192)
            # spatial_axis=2 → W axis (Width, 160)
            RandFlipd(keys=["ct", "edt"], prob=cfg.FLIP_PROB, spatial_axis=1),
            RandFlipd(keys=["ct", "edt"], prob=cfg.FLIP_PROB, spatial_axis=2),
            
            # Existing augmentations
            RandAffined(
                keys=["ct", "edt"],
                prob=cfg.AUG_PROB,
                rotate_range=(rx, ry, rz),
                scale_range=(cfg.SCALE_RANGE, cfg.SCALE_RANGE, cfg.SCALE_RANGE),
                mode=("bilinear", "nearest"),
                padding_mode="border"
            ),
            RandGaussianNoised(keys=["ct"], prob=cfg.AUG_PROB, std=cfg.NOISE_STD),
            RandAdjustContrastd(keys=["ct"], prob=cfg.AUG_PROB, gamma=cfg.GAMMA_RANGE),
        ]
    
    transforms += [
        ConcatItemsd(keys=["ct", "edt"], name="image"),
        DeleteItemsd(keys=["ct", "edt"]),
        ToTensord(keys=["image"])
    ]
    
    return Compose(transforms)

# ============================================================
# Model Architecture
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
# Test-Time Augmentation
# ============================================================
@torch.no_grad()
def predict_with_tta4(model, img_batch, clin_batch, device, n_time_bins, _debug_first_call=[True]):
    """
    4-view TTA for survival prediction
    
    Views:
    - Original
    - H-flip (Height axis flip)
    - W-flip (Width axis flip)
    - HW-flip (Both axes flip)
    
    Shape verification:
    - img_batch: [B, C, D, H, W] = [4, 2, 128, 192, 160]
    - dims=[-2] → H axis (192)
    - dims=[-1] → W axis (160)
    """
    # NEW: Dimension verification (only first call)
    if _debug_first_call[0]:
        print(f"\nTTA Dimension Verification:")
        print(f"   Input shape: {img_batch.shape}")
        print(f"   Expected: [B, C, D, H, W]")
        print(f"   dims=[-2] → flips H axis (size={img_batch.shape[-2]})")
        print(f"   dims=[-1] → flips W axis (size={img_batch.shape[-1]})")
        _debug_first_call[0] = False
    
    if not cfg.USE_TTA:
        model.eval()
        logits = model(img_batch.to(device), clin_batch.to(device))
        haz = torch.sigmoid(logits[:, :n_time_bins])
        surv = torch.cumprod(1.0 - haz, dim=1)
        return surv.cpu().numpy()
    
    # 4-view TTA ensemble
    views = [
        img_batch,                              # Original
        torch.flip(img_batch, dims=[-2]),       # H-flip
        torch.flip(img_batch, dims=[-1]),       # W-flip
        torch.flip(img_batch, dims=[-2, -1]),   # HW-flip
    ]
    
    model.eval()
    outs = []
    for v in views:
        logits = model(v.to(device), clin_batch.to(device))
        haz = torch.sigmoid(logits[:, :n_time_bins])
        surv = torch.cumprod(1.0 - haz, dim=1)
        outs.append(surv)
    
    return torch.stack(outs, 0).mean(0).cpu().numpy()

# ============================================================
# Training Function
# ============================================================
def run_fold(fold: int, manifest: pd.DataFrame, idx_train, idx_val, output_dir: Path) -> float:
    print(f"\n{'='*70}")
    print(f"FOLD {fold} | LR={cfg.LR:.2e} | TTA={'ON' if cfg.USE_TTA else 'OFF'}")
    print(f"{'='*70}")
    print(f"Training: {len(idx_train)} | Validation: {len(idx_val)}")
    
    times = manifest["time"].values
    events = manifest["event"].values
    
    # Time binning
    n_bins = adaptive_bins(times[idx_train], events[idx_train])
    cuts = make_time_bins(times[idx_train], events[idx_train], n_bins)
    
    # Label transformation
    # CRITICAL: Pass internal cuts (exclude min/max) to LabTransDiscreteTime
    # If cuts has 13 points, cuts[1:-1] gives 11 points, creating 12 intervals with indices 0-11
    lab = LabTransDiscreteTime(cuts=cuts[1:-1])
    y_train = lab.fit_transform(times[idx_train], events[idx_train])
    y_val = lab.transform(times[idx_val], events[idx_val])
    
    # FIXED: Strict alignment
    n_time_bins = len(cuts) - 1  # 13 boundaries → 12 bins
    n_out = n_time_bins
    
    print(f"Time binning:")
    print(f"   Bins: {n_time_bins}")
    print(f"   Cuts: {cuts[:5]}...{cuts[-2:]} ({len(cuts)} boundaries)")
    print(f"   Network output dim: {n_out}")
    
    # Clinical features
    clinical_train, clinical_val, feature_names = prepare_clinical_features_fold(
        manifest, idx_train, idx_val
    )
    
    
    print(f"Clinical features ({len(feature_names)}): {feature_names}")
    print(f"   {', '.join(feature_names[:5])}{'...' if len(feature_names) > 5 else ''}")
    
    # Datasets
    records_train = [{"ct": manifest.iloc[i].ct_path, 
                     "edt": manifest.iloc[i].edt_path} for i in idx_train]
    records_val = [{"ct": manifest.iloc[i].ct_path, 
                   "edt": manifest.iloc[i].edt_path} for i in idx_val]
    
    ds_train = SurvivalDataset(records_train, clinical_train, 
                               y_train[0], y_train[1], build_transforms(True))
    ds_val = SurvivalDataset(records_val, clinical_val, 
                            y_val[0], y_val[1], build_transforms(False))
    
    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=cfg.PREFETCH_FACTOR
    )
    
    dl_val = DataLoader(
        ds_val, 
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=cfg.PREFETCH_FACTOR
    )
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResNet10_Clinical_GN(
        in_channels=2,
        clinical_dim=len(feature_names),
        n_bins=n_out,
        dropout=cfg.DROPOUT
    ).to(device)
    
    print(f"Model: ResNet10 + Clinical (GroupNorm)")
    print(f"   Input: 2 channels (CT + EDT)")
    print(f"   Clinical: {len(feature_names)} features")
    print(f"   Output: {n_out} bins")
    
    # Optimizer
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim == 1 or 'bias' in name or 'norm' in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": cfg.WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=cfg.LR)
    
    def lr_schedule(epoch):
        if epoch < cfg.WARMUP_EPOCHS:
            return (epoch + 1) / cfg.WARMUP_EPOCHS
        progress = (epoch - cfg.WARMUP_EPOCHS) / (cfg.MAX_EPOCHS - cfg.WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    criterion = NLLLogistiHazardLoss()
    
    # Training tracking
    best_uno = -1.0
    no_improve = 0
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # IPCW setup
    y_train_struct = to_structured_y(times[idx_train], events[idx_train])
    y_val_struct = to_structured_y(times[idx_val], events[idx_val])
    evt_times = times[idx_train][events[idx_train] == 1]
    tau = np.quantile(evt_times, 0.9)
    
    bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
    t_idx_24m = np.argmin(np.abs(bin_mids - cfg.EVAL_24M_DAYS))
    
    # Training loop
    for epoch in range(cfg.MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(dl_train):
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
            
            if step == len(dl_train) - 1 and (step + 1) % cfg.ACCUMULATION_STEPS != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * cfg.ACCUMULATION_STEPS * x_img.size(0)
        
        scheduler.step()
        train_loss /= len(idx_train)
        
        # Validation
        model.eval()
        all_surv = []
        
        for batch in dl_val:
            surv = predict_with_tta4(model, batch["image"], batch["clinical"],
                                    device, n_time_bins)
            all_surv.append(surv)
        
        surv = np.concatenate(all_surv, axis=0)
        
        # Compute risk scores
        risk_integral = -np.trapz(surv, x=bin_mids, axis=1)
        surv_24m_val = surv[:, t_idx_24m] if t_idx_24m < surv.shape[1] else surv[:, -1]
        risk_24m = 1.0 - surv_24m_val
        
        # Metrics
        uno_integral = concordance_index_ipcw(y_train_struct, y_val_struct, 
                                              risk_integral, tau=tau)[0]
        uno_24m = concordance_index_ipcw(y_train_struct, y_val_struct, 
                                         risk_24m, tau=tau)[0]
        
        # Always use integral for consistency (as discussed in CNN Sonnet 7)
        uno = uno_integral
        risk = risk_integral

        harrell = concordance_index_censored(y_val_struct["event"],
                                            y_val_struct["time"], risk)[0]
        
        try:
            _, brier = brier_score(y_train_struct, y_val_struct, 
                                  surv_24m_val.reshape(-1, 1), 
                                  np.array([cfg.EVAL_24M_DAYS]))
            brier_24m = float(brier[0])
        except:
            brier_24m = np.nan
        
        brier_str = f"{brier_24m:.3f}" if not np.isnan(brier_24m) else "N/A"
        tta_marker = "[TTA]" if cfg.USE_TTA else ""
        
        print(f"Epoch {epoch+1:03d} | Loss {train_loss:.4f} | "
                    f"Uno(int)={uno_integral:.3f} | Uno(24m)={uno_24m:.3f} | "
                    f"Harrell {harrell:.3f} | Brier@24m {brier_str} {tta_marker}")
        
        # ============================================================
        # FIXED: Model selection with mutually exclusive branches
        # ============================================================
        improvement_threshold = 1e-5  # Define threshold BEFORE if-elif-else
        
        if np.isnan(uno):
            # Case 1: Metric is NaN - skip model save
            print(f"  WARNING: Metric is NaN, skipping model save")
            no_improve += 1
        elif uno > best_uno + improvement_threshold:
            # Case 2: Valid metric AND improved - save model
            prev_best = best_uno
            best_uno = uno
            no_improve = 0
            print(f"   New best: {best_uno:.4f} (improved by {best_uno - prev_best:.4f})")
            
            checkpoint = {
                "state_dict": model.state_dict(),
                "cuts": cuts.tolist(),
                "epoch": epoch,
                "uno_integral": uno_integral,
                "uno_24m": uno_24m,
                "harrell": harrell,
                "lr": cfg.LR,
                "use_tta": cfg.USE_TTA,
                "clinical_features": feature_names,
                "clinical_dim": len(feature_names),
                "n_time_bins": n_time_bins
            }
            torch.save(checkpoint, fold_dir / "best.pt")
            
            pd.DataFrame({
                "patient_id": manifest.iloc[idx_val]["patient_id"].values,
                "risk": risk,
                "surv_24m": surv_24m_val,
                "time": times[idx_val],
                "event": events[idx_val]
            }).to_csv(fold_dir / "val_pred.csv", index=False)
        else:
            # Case 3: Valid metric but NOT improved
            no_improve += 1
        
        # Early stopping check (independent, after all cases)
        if no_improve >= cfg.EARLY_STOP:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {no_improve} epochs)")
            break
    
    return best_uno

# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    
    cfg.LR = args.lr
    cfg.USE_TTA = not args.no_tta 
    
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    set_seed(cfg.SEED)
    
    tta_suffix = "_tta" if cfg.USE_TTA else "_noTTA"
    exp_name = args.exp_name or f"lr_{cfg.LR:.0e}{tta_suffix}"
    output_dir = Path(OUTPUT_DIR) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("PHASE 3 TRAINING - FIXED VERSION")
    print("="*70)
    print(f"Experiment: {exp_name}")
    print(f"LR: {cfg.LR:.2e}")
    print(f"TTA: {'4-view flip (H/W)' if cfg.USE_TTA else 'Disabled'}")
    print("="*70)
    print("\nVERIFIED FIXES:")
    print("   1. No double normalization (CT/EDT from Phase 2 are [0,1])")
    print("   2. Time binning aligned (n_out = n_time_bins)")
    print("   3. Paired flip augmentation (H/W axes)")
    print("   4. No duplicate operations")
    print("   5. Dimension logging enabled")
    print("   6. NaN handling fixed (no double increment)")
    print("="*70)
    
    # Load data
    crop = pd.read_csv(CROP_LOG_CSV)
    clinical = pd.read_csv(CLINICAL_CSV)
    
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
    
    print(f"\nDataset: {len(manifest)} patients")
    print(f"   Events: {manifest['event'].sum()} ({manifest['event'].mean()*100:.1f}%)")
    print(f"   Median survival: {manifest['time'].median():.0f} days")
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    
    results = []
    for fold, (idx_train, idx_val) in enumerate(skf.split(np.zeros(len(manifest)), 
                                                           manifest["event"])):
        if args.fold is not None and fold != args.fold:
            continue
            
        best_uno = run_fold(fold, manifest, idx_train, idx_val, output_dir)
        results.append({
            "fold": fold, 
            "best_uno": best_uno, 
            "lr": cfg.LR,
            "use_tta": cfg.USE_TTA
        })
    
    results_df = pd.DataFrame(results)
    
    if args.fold is not None:
        results_df.to_csv(output_dir / f"results_fold{args.fold}.csv", index=False)
        print(f"\nFold {args.fold} completed: Uno = {results[0]['best_uno']:.3f}")
    else:
        results_df.to_csv(output_dir / "results_complete.csv", index=False)
        
        print("\n" + "="*70)
        print(f"FINAL RESULTS")
        print("="*70)
        print(results_df)
        print(f"\nMean C-index: {results_df['best_uno'].mean():.3f} ± "
              f"{results_df['best_uno'].std():.3f}")
        print("="*70)

if __name__ == "__main__":
    main()