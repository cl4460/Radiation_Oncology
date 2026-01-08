#!/usr/bin/env python
"""
Phase 3 Training - FIXED SPLIT VERSION (Final)
===============================================
Based on phase3_0617.py with all fixes:
1. Support for fixed split JSON (no CV)
2. Simplified clinical features (6 dims: age, gender, T, N, M, stage)
3. All features normalized to [0, 1] for consistency
4. Fixed T stage handling (includes T5)
5. Fixed N stage handling (includes N4)
6. Fixed M stage handling (1,3 -> 1)
7. Fixed Overall stage (only I, II, IIIa, IIIb in this dataset)
"""

import os, math, random, warnings, time, argparse, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
from collections import Counter
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConcatItemsd, DeleteItemsd, ToTensord, Lambdad,
    RandAffined, RandGaussianNoised, RandAdjustContrastd,
    RandFlipd,
)
from monai.networks.nets import resnet as monai_resnet

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.loss import NLLLogistiHazardLoss
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored

warnings.filterwarnings("ignore")

# ============================================================
# Paths - VERIFIED CORRECT
# ============================================================
CROP_LOG_CSV = "/home/lichengze/Research/Nested_CV/CNN_pipeline/phase2_outputs/crop_log.csv"
CLINICAL_CSV = "/home/lichengze/Research/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
OUTPUT_DIR = "/home/lichengze/Research/Enhance/phase3_7e4_output"

# ============================================================
# Configuration
# ============================================================
@dataclass
class CFG:
    N_BINS_MAX: int = 15
    N_BINS_MIN: int = 12
    MIN_EVENTS_PER_BIN: int = 20
    
    BATCH_SIZE: int = 4
    ACCUMULATION_STEPS: int = 2
    MAX_EPOCHS: int = 100
    EARLY_STOP: int = 30
    
    LR: float = 7e-4
    WEIGHT_DECAY: float = 2e-4
    DROPOUT: float = 0.35
    WARMUP_EPOCHS: int = 25
    
    N_FOLDS: int = 5
    SEED: int = 42
    
    AUG_PROB: float = 0.5
    FLIP_PROB: float = 0.5
    ROT_DEG: Tuple[float, float, float] = (5.0, 5.0, 10.0)
    SCALE_RANGE: float = 0.12
    NOISE_STD: float = 0.02
    GAMMA_RANGE: Tuple[float, float] = (0.85, 1.15)
    
    NUM_WORKERS: int = 4
    PREFETCH_FACTOR: int = 2
    
    USE_TTA: bool = True
    EVAL_24M_DAYS: int = 730

cfg = CFG()

# ============================================================
# Command Line Arguments
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--no_tta', action='store_true')
    parser.add_argument("--split_json", type=str, default=None,
        help="Fixed split JSON. If provided, run single fixed-split training instead of CV")
    parser.add_argument('--seed', type=int, default=None,
        help="Random seed (overrides CFG.SEED)")
    parser.add_argument('--run_id', type=int, default=0,
        help="Run ID for multiple runs with same split (for output directory naming)")
    return parser.parse_args()

# ============================================================
# Utilities
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_structured_y(times, events):
    return np.array([(bool(e), float(t)) for t, e in zip(times, events)],
                    dtype=[('event', bool), ('time', float)])

def adaptive_bins(times, events):
    n_events = events.sum()
    ideal = n_events // cfg.MIN_EVENTS_PER_BIN
    return int(np.clip(ideal, cfg.N_BINS_MIN, cfg.N_BINS_MAX))

def make_time_bins(times, events, n_bins):
    event_times = times[events == 1]
    if len(event_times) < n_bins:
        return np.linspace(times.min(), times.max(), n_bins + 1)
    quantiles = np.linspace(0, 1, n_bins + 1)
    cuts = np.unique(np.quantile(event_times, quantiles))
    if len(cuts) != n_bins + 1:
        return np.linspace(times.min(), times.max(), n_bins + 1)
    return cuts

def _safe_mode(values: np.ndarray, default: float) -> float:
    """Get mode of array, return default if empty."""
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return float(default)
    cnt = Counter(vals.tolist())
    mode_val = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return float(mode_val)

def fix_path(path_str):
    """Fix paths that may be missing Nested_CV in the middle."""
    if pd.isna(path_str) or not path_str:
        return path_str
    path_str = str(path_str).strip()
    # If path doesn't exist, try adding Nested_CV after /Research/
    if not Path(path_str).exists() and "/Research/CNN_pipeline/" in path_str:
        fixed = path_str.replace("/Research/CNN_pipeline/", "/Research/Nested_CV/CNN_pipeline/")
        if Path(fixed).exists():
            return fixed
    return path_str

# ============================================================
# Clinical Feature Parsing Functions
# ============================================================
def _parse_T(series: pd.Series) -> np.ndarray:
    """
    Parse T stage: T1-T4 (and T5 if present, which exists in this dataset: 2 cases)
    Output: numeric value 1-5, NaN for missing/invalid
    """
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).upper()
        m = re.search(r'(\d)', s)
        if m:
            t = float(m.group(1))
            if 1.0 <= t <= 5.0:  # Allow T5
                out.append(t)
            else:
                out.append(np.nan)
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)

def _parse_N(series: pd.Series) -> np.ndarray:
    """
    Parse N stage: N0-N3 (and N4 if present, which exists in this dataset: 3 cases)
    Output: numeric value 0-4, NaN for missing/invalid
    """
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).upper()
        m = re.search(r'(\d)', s)
        if m:
            n = float(m.group(1))
            if 0.0 <= n <= 4.0:  # Allow N4
                out.append(n)
            else:
                out.append(np.nan)
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)

def _parse_M(series: pd.Series) -> np.ndarray:
    """
    Parse M stage: M0 -> 0, M1/M3 -> 1 (binary: no metastasis vs metastasis)
    This dataset has: 0 (417), 1 (1), 3 (4)
    Output: 0 or 1
    """
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).upper()
        m = re.search(r'(\d)', s)
        if not m:
            out.append(np.nan)
            continue
        mv = int(m.group(1))
        if mv == 0:
            out.append(0.0)
        else:  # 1, 3, or any other -> metastasis present
            out.append(1.0)
    return np.array(out, dtype=np.float32)

def _parse_overall_stage(series: pd.Series) -> np.ndarray:
    """
    Parse Overall Stage for NSCLC-Radiomics dataset.
    This dataset only has: I (93), II (40), IIIa (112), IIIb (176)
    
    Mapping to [0, 1] with equal spacing:
    - I    -> 0.0
    - II   -> 0.333
    - IIIa -> 0.667
    - IIIb -> 1.0
    """
    mapping = {
        "I": 0.0,
        "II": 0.333,
        "IIIA": 0.667,
        "IIIB": 1.0,
    }
    
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        # Clean and standardize
        s = str(v).strip().upper()
        # Remove common prefixes
        s = s.replace("STAGE", "").replace(" ", "")
        # Handle special characters
        s = s.replace("ⅢA", "IIIA").replace("ⅢB", "IIIB")
        s = s.replace("Ⅱ", "II").replace("Ⅰ", "I")
        
        if s in mapping:
            out.append(mapping[s])
        else:
            out.append(np.nan)
    
    return np.array(out, dtype=np.float32)

# ============================================================
# Clinical Feature Processing - FINAL VERSION
# ============================================================
def prepare_clinical_features_fold(manifest: pd.DataFrame, 
                                  idx_train: np.ndarray,
                                  idx_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Simplified clinical features - all normalized to [0, 1]:
    - age: median impute + MinMax [0, 1] (1 dim)
    - gender: binary M=1 (1 dim)
    - T: mode impute + normalize [0, 1] (1 dim)
    - N: normalize [0, 1] (1 dim)
    - M: binary 0/1 (1 dim)
    - overall_stage: ordinal [0, 1] (1 dim)
    
    Total: 6 dimensions, all in [0, 1] range
    """
    features_train = []
    features_val = []
    feature_names = []

    # ========== Age: median impute + MinMax [0, 1] ==========
    if 'age' in manifest.columns:
        age_raw = pd.to_numeric(manifest['age'], errors='coerce').values.astype(np.float32)
        
        # Median impute from train
        train_age = age_raw[idx_train]
        train_median = np.nanmedian(train_age) if np.any(~np.isnan(train_age)) else 65.0
        age_filled = np.where(np.isnan(age_raw), train_median, age_raw)
        
        # MinMax scale to [0, 1] based on train
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(age_filled[idx_train].reshape(-1, 1))
        age_scaled = scaler.transform(age_filled.reshape(-1, 1)).flatten().astype(np.float32)
        
        # Clip to [0, 1] in case val has values outside train range
        age_scaled = np.clip(age_scaled, 0.0, 1.0)
        
        features_train.append(age_scaled[idx_train])
        features_val.append(age_scaled[idx_val])
        feature_names.append('age')

    # ========== Gender: binary (M=1, F=0) ==========
    if 'gender' in manifest.columns:
        gender_ser = manifest['gender'].astype(str).str.upper().str[0]
        gender = (gender_ser == 'M').astype(np.float32).values
        
        features_train.append(gender[idx_train])
        features_val.append(gender[idx_val])
        feature_names.append('gender')

    # ========== T stage: mode impute + normalize [0, 1] ==========
    if 'T' in manifest.columns:
        T_num = _parse_T(manifest['T'])
        
        # Mode impute from train (1 missing value in this dataset)
        train_T = T_num[idx_train]
        T_mode = _safe_mode(train_T, default=2.0)
        T_filled = np.where(np.isnan(T_num), T_mode, T_num)
        
        # Normalize: T1-T5 -> [0, 1]
        # T1=0.0, T2=0.25, T3=0.5, T4=0.75, T5=1.0
        T_norm = (T_filled - 1.0) / 4.0  # (T - 1) / (5 - 1)
        T_norm = np.clip(T_norm, 0.0, 1.0)
        
        features_train.append(T_norm[idx_train].astype(np.float32))
        features_val.append(T_norm[idx_val].astype(np.float32))
        feature_names.append('T')

    # ========== N stage: normalize [0, 1] ==========
    if 'N' in manifest.columns:
        N_num = _parse_N(manifest['N'])
        
        # Mode impute (0 missing in this dataset, but just in case)
        train_N = N_num[idx_train]
        N_mode = _safe_mode(train_N, default=0.0)
        N_filled = np.where(np.isnan(N_num), N_mode, N_num)
        
        # Normalize: N0-N4 -> [0, 1]
        # N0=0.0, N1=0.25, N2=0.5, N3=0.75, N4=1.0
        N_norm = N_filled / 4.0
        N_norm = np.clip(N_norm, 0.0, 1.0)
        
        features_train.append(N_norm[idx_train].astype(np.float32))
        features_val.append(N_norm[idx_val].astype(np.float32))
        feature_names.append('N')

    # ========== M stage: binary (0 or 1) ==========
    if 'M' in manifest.columns:
        M_binary = _parse_M(manifest['M'])
        
        # No missing in this dataset, but just in case
        M_filled = np.where(np.isnan(M_binary), 0.0, M_binary)
        
        features_train.append(M_filled[idx_train].astype(np.float32))
        features_val.append(M_filled[idx_val].astype(np.float32))
        feature_names.append('M')

    # ========== Overall Stage: ordinal [0, 1] ==========
    if 'overall_stage' in manifest.columns:
        stage_ord = _parse_overall_stage(manifest['overall_stage'])
        
        # Mode impute (1 missing in this dataset)
        train_stage = stage_ord[idx_train]
        stage_mode = _safe_mode(train_stage, default=0.667)  # Default to IIIa (most common)
        stage_filled = np.where(np.isnan(stage_ord), stage_mode, stage_ord)
        
        features_train.append(stage_filled[idx_train].astype(np.float32))
        features_val.append(stage_filled[idx_val].astype(np.float32))
        feature_names.append('stage')

    # ========== Stack and finalize ==========
    if features_train:
        clinical_train = np.stack(features_train, axis=1).astype(np.float32)
        clinical_val = np.stack(features_val, axis=1).astype(np.float32)
    else:
        clinical_train = np.zeros((len(idx_train), 1), dtype=np.float32)
        clinical_val = np.zeros((len(idx_val), 1), dtype=np.float32)
        feature_names = ['dummy']

    # Safety: replace any remaining NaN/Inf
    clinical_train = np.nan_to_num(clinical_train, nan=0.0, posinf=1.0, neginf=0.0)
    clinical_val = np.nan_to_num(clinical_val, nan=0.0, posinf=1.0, neginf=0.0)

    return clinical_train, clinical_val, feature_names

# ============================================================
# Dataset
# ============================================================
class SurvivalDataset(Dataset):
    def __init__(self, records, clinical, y_idx, y_evt, transform):
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
# Transforms - FIXED function name: build_transforms
# ============================================================
def build_transforms(train: bool) -> Compose:
    """Build data transforms for CT+EDT input."""
    transforms = [
        LoadImaged(keys=["ct", "edt"]),
        EnsureChannelFirstd(keys=["ct", "edt"]),
        EnsureTyped(keys=["ct", "edt"], dtype=torch.float32, track_meta=False),
        Lambdad(keys=["ct", "edt"], func=lambda x: torch.nan_to_num(x, 0.0, 0.0, 0.0)), 
        Lambdad(keys=["ct", "edt"], func=lambda x: torch.clamp(x, 0.0, 1.0)),
    ]
    
    if train:
        rx, ry, rz = map(math.radians, cfg.ROT_DEG)
        transforms += [
            RandFlipd(keys=["ct", "edt"], prob=cfg.FLIP_PROB, spatial_axis=1),
            RandFlipd(keys=["ct", "edt"], prob=cfg.FLIP_PROB, spatial_axis=2),
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
# Model
# ============================================================
class ResNet10_Clinical_GN(nn.Module):
    def __init__(self, in_channels=2, clinical_dim=6, n_bins=15, dropout=0.35):
        super().__init__()
        
        self.backbone = monai_resnet.resnet10(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=512,
        )
        self._replace_bn_with_gn(self.backbone)
        
        # Clinical encoder for 6 features
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
# TTA (4-view)
# ============================================================
@torch.no_grad()
def predict_with_tta4(model, img_batch, clin_batch, device, n_time_bins):
    if not cfg.USE_TTA:
        model.eval()
        logits = model(img_batch.to(device), clin_batch.to(device))
        haz = torch.sigmoid(logits[:, :n_time_bins])
        surv = torch.cumprod(1.0 - haz, dim=1)
        return surv.cpu().numpy()
    
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

# ============================================================
# Training
# ============================================================
def run_fold(fold, manifest, idx_train, idx_val, output_dir):
    print(f"\n{'='*70}", flush=True)
    print(f"FOLD {fold} | LR={cfg.LR:.2e} | TTA={'ON' if cfg.USE_TTA else 'OFF'}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Training: {len(idx_train)} | Validation: {len(idx_val)}", flush=True)
    
    times = manifest["time"].values
    events = manifest["event"].values
    
    # Time binning
    n_bins = adaptive_bins(times[idx_train], events[idx_train])
    cuts = make_time_bins(times[idx_train], events[idx_train], n_bins)
    
    lab = LabTransDiscreteTime(cuts=cuts[1:-1])
    y_train = lab.fit_transform(times[idx_train], events[idx_train])
    y_val = lab.transform(times[idx_val], events[idx_val])
    
    n_time_bins = len(cuts) - 1
    
    print(f"Time bins: {n_time_bins}", flush=True)
    
    # Clinical features (6 dims)
    clinical_train, clinical_val, feature_names = prepare_clinical_features_fold(
        manifest, idx_train, idx_val
    )
    print(f"Clinical features ({len(feature_names)}): {feature_names}", flush=True)
    
    # Verify all features in [0, 1]
    print(f"   Train range: [{clinical_train.min():.3f}, {clinical_train.max():.3f}]", flush=True)
    print(f"   Val range: [{clinical_val.min():.3f}, {clinical_val.max():.3f}]", flush=True)
    
    # Datasets
    records_train = [{"ct": manifest.iloc[i].ct_path, 
                     "edt": manifest.iloc[i].edt_path} for i in idx_train]
    records_val = [{"ct": manifest.iloc[i].ct_path, 
                   "edt": manifest.iloc[i].edt_path} for i in idx_val]
    
    ds_train = SurvivalDataset(records_train, clinical_train, 
                               y_train[0], y_train[1], build_transforms(True))
    ds_val = SurvivalDataset(records_val, clinical_val, 
                            y_val[0], y_val[1], build_transforms(False))
    
    dl_train = DataLoader(ds_train, batch_size=cfg.BATCH_SIZE, shuffle=True,
                          num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
                          persistent_workers=True, prefetch_factor=cfg.PREFETCH_FACTOR)
    dl_val = DataLoader(ds_val, batch_size=cfg.BATCH_SIZE, shuffle=False,
                        num_workers=cfg.NUM_WORKERS, pin_memory=True,
                        persistent_workers=True, prefetch_factor=cfg.PREFETCH_FACTOR)
    
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet10_Clinical_GN(
        in_channels=2,
        clinical_dim=len(feature_names),
        n_bins=n_time_bins,
        dropout=cfg.DROPOUT
    ).to(device)
    
    print(f"Model: ResNet10 + Clinical (GroupNorm)", flush=True)
    print(f"   Clinical dim: {len(feature_names)}", flush=True)
    print(f"   Output bins: {n_time_bins}", flush=True)
    
    # Optimizer with weight decay separation
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
            
            # Handle last batch
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
        risk = -np.trapz(surv, x=bin_mids, axis=1)
        
        # Metrics
        try:
            uno = concordance_index_ipcw(y_train_struct, y_val_struct, risk, tau=tau)[0]
        except:
            uno = np.nan
        
        harrell = concordance_index_censored(y_val_struct["event"], y_val_struct["time"], risk)[0]
        
        print(f"Epoch {epoch+1:03d} | Loss {train_loss:.4f} | Uno {uno:.4f} | Harrell {harrell:.4f}", flush=True)
        
        # Model selection
        if np.isnan(uno):
            no_improve += 1
        elif uno > best_uno + 1e-5:
            best_uno = uno
            no_improve = 0
            print(f"   *** New best: {best_uno:.4f}", flush=True)
            
            checkpoint = {
                "state_dict": model.state_dict(),
                "cuts": cuts.tolist(),
                "epoch": epoch,
                "uno_integral": uno,
                "harrell": harrell,
                "clinical_features": feature_names,
                "clinical_dim": len(feature_names),
                "n_time_bins": n_time_bins
            }
            torch.save(checkpoint, fold_dir / "best.pt")
            
            # Save validation predictions
            pd.DataFrame({
                "patient_id": manifest.iloc[idx_val]["patient_id"].values,
                "risk": risk,
                "time": times[idx_val],
                "event": events[idx_val]
            }).to_csv(fold_dir / "val_pred.csv", index=False)
        else:
            no_improve += 1
        
        if no_improve >= cfg.EARLY_STOP:
            print(f"Early stopping at epoch {epoch+1}", flush=True)
            break
    
    return best_uno

# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    
    cfg.LR = args.lr
    cfg.USE_TTA = not args.no_tta 
    
    if args.seed is not None:
        cfg.SEED = args.seed
    
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    set_seed(cfg.SEED)
    
    # Output directory
    if args.split_json:
        split_name = Path(args.split_json).stem
        exp_name = args.exp_name or f"{split_name}_lr{cfg.LR:.0e}"
    else:
        tta_suffix = "_tta" if cfg.USE_TTA else "_noTTA"
        exp_name = args.exp_name or f"lr_{cfg.LR:.0e}{tta_suffix}"
    
    output_dir = Path(OUTPUT_DIR) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("PHASE 3 TRAINING - FIXED SPLIT VERSION (Final)")
    print("="*70)
    print(f"Experiment: {exp_name}")
    print(f"LR: {cfg.LR:.2e}")
    print(f"TTA: {'4-view flip' if cfg.USE_TTA else 'OFF'}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Load data
    crop = pd.read_csv(CROP_LOG_CSV)
    clinical = pd.read_csv(CLINICAL_CSV)
    
    # Column mapping (handle case sensitivity)
    col_map = {}
    for col in clinical.columns:
        col_lower = col.lower().strip()
        if 'patient' in col_lower:
            col_map[col] = 'patient_id'
        elif 'survival' in col_lower and 'time' in col_lower:
            col_map[col] = 'time'
        elif 'dead' in col_lower or 'event' in col_lower:
            col_map[col] = 'event'
        elif 'overall' in col_lower and 'stage' in col_lower:
            col_map[col] = 'overall_stage'
        elif '.t.' in col_lower and 'stage' in col_lower:
            col_map[col] = 'T'
        elif '.n.' in col_lower and 'stage' in col_lower:
            col_map[col] = 'N'
        elif '.m.' in col_lower and 'stage' in col_lower:
            col_map[col] = 'M'
        elif 'age' in col_lower and 'stage' not in col_lower:
            col_map[col] = 'age'
        elif 'gender' in col_lower:
            col_map[col] = 'gender'
    
    clinical = clinical.rename(columns=col_map)
    
    # Merge crop log with clinical
    manifest = crop[["patient_id", "out_ct", "out_edt"]].rename(columns={
        "out_ct": "ct_path",
        "out_edt": "edt_path"
    }).merge(clinical, on="patient_id", how="inner")
    
    # Fix paths (CSV may have paths missing Nested_CV)
    manifest["ct_path"] = manifest["ct_path"].apply(fix_path)
    manifest["edt_path"] = manifest["edt_path"].apply(fix_path)
    
    manifest = manifest.dropna(subset=["time", "event"])
    manifest["time"] = manifest["time"].astype(int)
    manifest["event"] = manifest["event"].astype(int)
    
    print(f"\nDataset: {len(manifest)} patients", flush=True)
    print(f"   Events: {manifest['event'].sum()} ({manifest['event'].mean()*100:.1f}%)", flush=True)
    print(f"   Median survival: {manifest['time'].median():.0f} days", flush=True)
    
    # Fixed split mode
    if args.split_json:
        with open(args.split_json, "r") as f:
            sp = json.load(f)
        
        # Handle nested "splits" structure
        if "splits" in sp:
            sp = sp["splits"]
        
        train_ids = set(map(str, sp.get("train", sp.get("train_ids", []))))
        val_ids = set(map(str, sp.get("val", sp.get("val_ids", []))))
        test_ids = set(map(str, sp.get("test", sp.get("test_ids", []))))
        
        # Sort manifest for reproducibility
        manifest = manifest.sort_values("patient_id").reset_index(drop=True)
        
        idx_train = manifest.index[manifest["patient_id"].astype(str).isin(train_ids)].to_numpy()
        idx_val = manifest.index[manifest["patient_id"].astype(str).isin(val_ids)].to_numpy()
        idx_test = manifest.index[manifest["patient_id"].astype(str).isin(test_ids)].to_numpy()
        
        print(f"\n[Fixed Split] train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}", flush=True)
        
        # Use run_id as fold number for multiple runs with same split
        fold_num = args.run_id if hasattr(args, 'run_id') else 0
        best_uno = run_fold(fold_num, manifest, idx_train, idx_val, output_dir)
        
        print(f"\n" + "="*70, flush=True)
        print(f"TRAINING COMPLETE", flush=True)
        print(f"="*70, flush=True)
        print(f"Best Val Uno C-index: {best_uno:.4f}", flush=True)
        fold_num = args.run_id if hasattr(args, 'run_id') else 0
        print(f"Model saved to: {output_dir / f'fold_{fold_num}' / 'best.pt'}", flush=True)
        print(f"="*70, flush=True)
        return
    
    # CV mode (fallback if no split_json)
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    
    results = []
    for fold, (idx_train, idx_val) in enumerate(skf.split(np.zeros(len(manifest)), manifest["event"])):
        if args.fold is not None and fold != args.fold:
            continue
        best_uno = run_fold(fold, manifest, idx_train, idx_val, output_dir)
        results.append({"fold": fold, "best_uno": best_uno})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)
    print(f"\nMean C-index: {results_df['best_uno'].mean():.3f} ± {results_df['best_uno'].std():.3f}")

if __name__ == "__main__":
    main()