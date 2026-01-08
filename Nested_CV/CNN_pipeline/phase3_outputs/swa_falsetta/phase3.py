import os, math, random, warnings, time, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConcatItemsd, DeleteItemsd, ToTensord,
    RandAffined, RandGaussianNoised, RandAdjustContrastd,
    Lambda
)
from monai.networks.nets import resnet as monai_resnet

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.loss import NLLLogistiHazardLoss  # Correct spelling in pycox library
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored, brier_score

warnings.filterwarnings("ignore")

# Paths
PHASE1_QC_CSV = "/home/lichengze/Research/CNN_pipeline/phase1_outputs/phase1_qc.csv"
CROP_LOG_CSV = "/home/lichengze/Research/CNN_pipeline/phase2_outputs/crop_log.csv"
CLINICAL_CSV = "/home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical.csv"
OUTPUT_DIR = "/home/lichengze/Research/CNN_pipeline/phase3_outputs/swa_falsetta"

@dataclass
class CFG:
    # Time binning
    N_BINS_MAX: int = 12
    N_BINS_MIN: int = 12
    MIN_EVENTS_PER_BIN: int = 24
    
    # Training
    BATCH_SIZE: int = 4
    ACCUMULATION_STEPS: int = 2
    MAX_EPOCHS: int = 100
    EARLY_STOP: int = 25
    
    # LR will be set via command line
    LR: float = 7e-4  # Default, will be overridden
    WEIGHT_DECAY: float = 2e-4
    DROPOUT: float = 0.35
    WARMUP_EPOCHS: int = 20
    
    # SWA configuration (OPTIMIZED)
    USE_SWA: bool = True
    SWA_START_RATIO: float = 0.65  # Start earlier for more samples
    SWA_LR_FACTOR: float = 0.4  # Conservative but effective
    SWA_MIN_EPOCHS: int = 10  # Minimum epochs for SWA
    
    # Primary metric for model selection
    PRIMARY_METRIC: str = "uno_integral"  # or "uno_24m"

    N_FOLDS: int = 5
    SEED: int = 42
    
    # Augmentation
    AUG_PROB: float = 0.5
    ROT_DEG: Tuple[float,float,float] = (5.0, 5.0, 10.0)
    SCALE_RANGE: float = 0.12
    NOISE_STD: float = 0.02
    GAMMA_RANGE: Tuple[float,float] = (0.85, 1.15)
    
    # Data loading
    NUM_WORKERS: int = 4
    PREFETCH_FACTOR: int = 2
    
    # TTA
    USE_TTA: bool = False
    EVAL_24M_DAYS: int = 730

cfg = CFG()

def parse_args():
    """Parse command line arguments with improved interface"""
    parser = argparse.ArgumentParser(
        description='SWA-enhanced survival model training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--fold', type=int, default=None, 
                       help='Run specific fold (0-4), None=run all folds')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID')
    parser.add_argument('--lr', type=float, default=7e-4,
                       help='Learning rate')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name for output directory')
    
    # Better SWA control
    swa_group = parser.add_mutually_exclusive_group()
    swa_group.add_argument('--use_swa', dest='use_swa', action='store_true',
                          help='Enable SWA (default)')
    swa_group.add_argument('--no_swa', dest='use_swa', action='store_false',
                          help='Disable SWA')
    parser.set_defaults(use_swa=True)
    
    parser.add_argument('--swa_start', type=float, default=0.65,
                       help='SWA start ratio (0.0-1.0)')
    
    return parser.parse_args()

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
    """✅ VERIFIED: Scaler fit only on training data"""
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
        scaler.fit(age_filled[idx_train].reshape(-1, 1))  # ✅ Fit on train only
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

def build_transforms(train: bool) -> Compose:
    """
    ✅ FIXED: Removed double normalization
    Phase 2 already normalized CT to [0, 1], so we don't re-normalize here
    """
    transforms = [
        LoadImaged(keys=["ct", "edt"]),  # ✅ FIXED: Removed ensure_channel_first kwarg
        EnsureChannelFirstd(keys=["ct", "edt"]),
        EnsureTyped(keys=["ct", "edt"], dtype=torch.float32, track_meta=False),

        # ✅ REMOVED: CT normalization (already done in phase 2)
        # Phase 2 already scales CT to [0, 1] with HU window [-1000, 400]
        # EDT is already normalized to [0, 1] in phase 2
        
        Lambda(func=lambda d: {
            **d,
            "ct": torch.nan_to_num(d["ct"], nan=0.0, posinf=0.0, neginf=0.0),
            "edt": torch.nan_to_num(d["edt"], nan=0.0, posinf=0.0, neginf=0.0)
        }),
        
        # Just ensure values are in valid range (sanity check)
        Lambda(func=lambda d: {
            **d,
            "ct": torch.clamp(d["ct"], min=0.0, max=1.0),
            "edt": torch.clamp(d["edt"], min=0.0, max=1.0)
        }),
        
        EnsureTyped(keys=["ct", "edt"], dtype=torch.float32, track_meta=False),
    ]
    
    if train:
        rx, ry, rz = map(math.radians, cfg.ROT_DEG)
        transforms += [
            RandAffined(
                keys=["ct", "edt"],
                prob=cfg.AUG_PROB,
                rotate_range=(rx, ry, rz),
                scale_range=cfg.SCALE_RANGE,
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

class ResNet10_Clinical_GN(nn.Module):
    def __init__(self, in_channels: int = 2, clinical_dim: int = 6,
                 n_bins: int = 15, dropout: float = 0.35):
        super().__init__()
        
        self.backbone = monai_resnet.resnet10(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=512
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

def tta_predict(model, img_batch, clin_batch, device, n_time_bins):
    """✅ Short-circuits when TTA is disabled"""
    if not cfg.USE_TTA:
        with torch.no_grad():
            logits = model(img_batch.to(device), clin_batch.to(device))
            haz = torch.sigmoid(logits[:, :n_time_bins])
            return torch.cumprod(1.0 - haz, dim=1).cpu().numpy()
    
    augmented = [
        img_batch,
        torch.flip(img_batch, dims=[-1]),
        torch.flip(img_batch, dims=[-2])
    ]
    
    survs = []
    with torch.no_grad():
        for aug_img in augmented:
            logits = model(aug_img.to(device), clin_batch.to(device))
            haz = torch.sigmoid(logits[:, :n_time_bins])
            surv = torch.cumprod(1.0 - haz, dim=1)
            survs.append(surv)
    
    return torch.stack(survs).mean(dim=0).cpu().numpy()

def has_batchnorm(model):
    """Check if model contains any BatchNorm layers"""
    return any(isinstance(m, nn.modules.batchnorm._BatchNorm) 
               for m in model.modules())


def run_fold_with_swa(fold: int, manifest: pd.DataFrame, 
                      idx_train: np.ndarray, idx_val: np.ndarray,
                      output_dir: Path) -> dict:
    """
    Train one fold with optimized SWA support and all fixes applied.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ===== Data preparation =====
    times = manifest["time"].values
    events = manifest["event"].values
    
    # Time binning
    n_bins = adaptive_bins(times[idx_train], events[idx_train])
    cuts = make_time_bins(times[idx_train], events[idx_train], n_bins)
    
    # Label transformation
    lab = LabTransDiscreteTime(cuts=cuts[1:-1])
    y_train_discrete = lab.fit_transform(times[idx_train], events[idx_train])
    y_val_discrete = lab.transform(times[idx_val], events[idx_val])
    
    # CRITICAL: Get actual number of time bins
    n_time_bins = len(cuts) - 1
    print(f"Time bins: {n_time_bins} (cuts: {len(cuts)} points)", flush=True)
    
    # ✅ FIXED: Compute bin constants ONCE before training loop
    bin_edges = cuts
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    t_idx_24m = int(np.argmin(np.abs(bin_mids - cfg.EVAL_24M_DAYS)))
    print(f"24-month evaluation: bin index {t_idx_24m}, time {bin_mids[t_idx_24m]:.1f} days", flush=True)
    
    # Clinical features
    X_train_clin, X_val_clin, feature_names = prepare_clinical_features_fold(
        manifest, idx_train, idx_val
    )
    
    # Datasets
    train_records = [
        {"ct": manifest.iloc[i]["out_ct"], 
         "edt": manifest.iloc[i]["out_edt"]}
        for i in idx_train
    ]
    val_records = [
        {"ct": manifest.iloc[i]["out_ct"], 
         "edt": manifest.iloc[i]["out_edt"]}
        for i in idx_val
    ]
    
    ds_train = SurvivalDataset(
        train_records, X_train_clin, 
        y_train_discrete[0], y_train_discrete[1],
        build_transforms(train=True)
    )
    ds_val = SurvivalDataset(
        val_records, X_val_clin,
        y_val_discrete[0], y_val_discrete[1],
        build_transforms(train=False)
    )
    
    # ✅ OPTIONAL: Add WeightedRandomSampler with generator for reproducibility
    # Uncomment if you want to use weighted sampling
    # weights = torch.as_tensor(
    #     np.where(y_train_discrete[1] > 0.5, 2.0, 1.0), 
    #     dtype=torch.double
    # )
    # sampler = WeightedRandomSampler(
    #     weights=weights,
    #     num_samples=len(weights),
    #     replacement=True,
    #     generator=torch.Generator().manual_seed(cfg.SEED)
    # )
    # dl_train = DataLoader(
    #     ds_train, batch_size=cfg.BATCH_SIZE, sampler=sampler,
    #     num_workers=cfg.NUM_WORKERS, prefetch_factor=cfg.PREFETCH_FACTOR,
    #     pin_memory=True, persistent_workers=True
    # )
    
    # Standard DataLoader (current approach)
    dl_train = DataLoader(
        ds_train, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, prefetch_factor=cfg.PREFETCH_FACTOR,
        pin_memory=True, persistent_workers=True, drop_last=True
    )
    dl_val = DataLoader(
        ds_val, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, prefetch_factor=cfg.PREFETCH_FACTOR,
        pin_memory=True, persistent_workers=True
    )
    
    # ===== Model initialization =====
    model = ResNet10_Clinical_GN(
        in_channels=2,
        clinical_dim=len(feature_names),
        n_bins=n_time_bins,
        dropout=cfg.DROPOUT
    ).to(device)
    
    print(f"\n📊 Model architecture:", flush=True)
    print(f"   Input channels: 2 (CT + EDT)", flush=True)
    print(f"   Clinical features: {len(feature_names)}", flush=True)
    print(f"   Output bins: {n_time_bins}", flush=True)
    print(f"   Dropout: {cfg.DROPOUT}", flush=True)
    print(f"   Using GroupNorm (no BatchNorm)", flush=True)
    
    # ===== Optimizer setup =====
    no_decay_params = []
    decay_params = []
    for name, param in model.named_parameters():
        if "norm" in name.lower() or "bias" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": cfg.WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0}
    ], lr=cfg.LR)
    
    # ===== Learning rate scheduler =====
    def lr_schedule(epoch):
        if epoch < cfg.WARMUP_EPOCHS:
            return (epoch + 1) / cfg.WARMUP_EPOCHS
        progress = (epoch - cfg.WARMUP_EPOCHS) / (cfg.MAX_EPOCHS - cfg.WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    criterion = NLLLogistiHazardLoss()
    
    # ===== SWA initialization =====
    swa_model = None
    swa_scheduler = None
    swa_start = int(cfg.SWA_START_RATIO * cfg.MAX_EPOCHS)
    
    if cfg.USE_SWA:
        from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
        
        print(f"\n🔵 SWA Configuration:", flush=True)
        print(f"   Start epoch: {swa_start}/{cfg.MAX_EPOCHS}", flush=True)
        print(f"   SWA epochs: {cfg.MAX_EPOCHS - swa_start}", flush=True)
        print(f"   Base LR: {cfg.LR:.2e}", flush=True)
        print(f"   SWA LR factor: {cfg.SWA_LR_FACTOR}", flush=True)
        print(f"   Has BatchNorm: No (using GroupNorm)", flush=True)
    
    # ===== Training tracking =====
    best_uno = -1.0
    best_swa_uno = -1.0
    no_improve = 0
    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # For IPCW calculation
    y_train_struct = to_structured_y(times[idx_train], events[idx_train])
    y_val_struct = to_structured_y(times[idx_val], events[idx_val])
    evt_times = times[idx_train][events[idx_train] == 1]
    tau = np.quantile(evt_times, 0.9)
    
    # ===== Training loop with optimized SWA =====
    for epoch in range(cfg.MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        # ===== Initialize SWA at the right epoch =====
        if cfg.USE_SWA and epoch == swa_start:
            print(f"\n🔵 Initializing SWA at epoch {epoch + 1}", flush=True)
            swa_model = AveragedModel(model)
            swa_scheduler = SWALR(optimizer, swa_lr=cfg.LR * cfg.SWA_LR_FACTOR)
        
        # Training step
        for step, batch in enumerate(dl_train):
            x_img = batch["image"].to(device)
            x_clin = batch["clinical"].to(device)
            y_idx = batch["label_idx"].to(device)
            y_evt = batch["label_event"].to(device)
            
            logits = model(x_img, x_clin)
            loss = criterion(logits, y_idx, y_evt) / cfg.ACCUMULATION_STEPS
            
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue
            
            loss.backward()
            
            if (step + 1) % cfg.ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * cfg.ACCUMULATION_STEPS
            
            if step == len(dl_train) - 1 and (step + 1) % cfg.ACCUMULATION_STEPS != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        train_loss /= len(dl_train)
        
        # ===== Update SWA model =====
        if cfg.USE_SWA and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        # ===== Validation =====
        model.eval()
        all_surv = []
        
        with torch.no_grad():
            for batch in dl_val:
                surv = tta_predict(model, batch["image"], batch["clinical"],
                                 device, n_time_bins)
                all_surv.append(surv)
        
        surv = np.concatenate(all_surv, axis=0)
        
        # ✅ FIXED: Use pre-computed bin constants
        risk_integral = -np.trapz(surv, x=bin_mids, axis=1)
        surv_24m_val = surv[:, t_idx_24m] if t_idx_24m < surv.shape[1] else surv[:, -1]
        risk_24m = 1.0 - surv_24m_val
        
        uno_integral = concordance_index_ipcw(y_train_struct, y_val_struct, 
                                              risk_integral, tau=tau)[0]
        uno_24m = concordance_index_ipcw(y_train_struct, y_val_struct, 
                                         risk_24m, tau=tau)[0]
        
        # Fixed metric selection
        if cfg.PRIMARY_METRIC == "uno_integral":
            uno = uno_integral
            risk = risk_integral
            risk_method = "int"
        else:  # uno_24m
            uno = uno_24m
            risk = risk_24m
            risk_method = "24m"
        
        harrell = concordance_index_censored(y_val_struct["event"],
                                            y_val_struct["time"], risk)[0]
        
        # Print progress
        swa_indicator = "🔵 SWA" if (cfg.USE_SWA and epoch >= swa_start) else ""
        print(f"Epoch {epoch+1:03d} | Loss {train_loss:.4f} | "
              f"Uno-int {uno_integral:.3f} | Uno-24m {uno_24m:.3f} | "
              f"Harrell {harrell:.3f} {swa_indicator}", flush=True)
        
        # ===== Early stopping (IMPROVED for SWA) =====
        min_epoch_for_stop = swa_start + cfg.SWA_MIN_EPOCHS if cfg.USE_SWA else 0
        
        if uno > best_uno + 0.0005:
            best_uno = uno
            no_improve = 0
            checkpoint = {
                "state_dict": model.state_dict(),
                "cuts": cuts.tolist(),
                "epoch": epoch,
                "uno": uno,
                "uno_integral": uno_integral,
                "uno_24m": uno_24m,
                "lr": cfg.LR,
                "clinical_features": feature_names,
                "clinical_dim": len(feature_names)
            }
            torch.save(checkpoint, fold_dir / "best_regular.pt")
        else:
            no_improve += 1
            if no_improve >= cfg.EARLY_STOP and epoch >= min_epoch_for_stop:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                print(f"(Ensured {epoch - swa_start + 1} SWA epochs)", flush=True)
                break
    
    # ===== Finalize SWA model =====
    if cfg.USE_SWA and swa_model is not None:
        from torch.optim.swa_utils import update_bn
        
        print("\n" + "="*60, flush=True)
        print("🔵 Finalizing SWA model...", flush=True)
        
        has_bn = has_batchnorm(swa_model)
        print(f"   Has BatchNorm: {has_bn}", flush=True)
        
        if has_bn:
            print("   Updating BatchNorm statistics with training data...", flush=True)
            update_bn(dl_train, swa_model, device=device)
        else:
            print("   Skipping update_bn (model uses GroupNorm)", flush=True)
        
        # Evaluate SWA model
        print("   Evaluating SWA model on validation set...", flush=True)
        swa_model.eval()
        all_surv_swa = []
        
        with torch.no_grad():
            for batch in dl_val:
                surv_swa = tta_predict(swa_model, batch["image"], batch["clinical"],
                                      device, n_time_bins)
                all_surv_swa.append(surv_swa)
        
        surv_swa = np.concatenate(all_surv_swa, axis=0)
        
        # Compute SWA metrics using pre-computed bin constants
        risk_integral_swa = -np.trapz(surv_swa, x=bin_mids, axis=1)
        surv_24m_swa = surv_swa[:, t_idx_24m] if t_idx_24m < surv_swa.shape[1] else surv_swa[:, -1]
        risk_24m_swa = 1.0 - surv_24m_swa
        
        uno_integral_swa = concordance_index_ipcw(y_train_struct, y_val_struct, 
                                                  risk_integral_swa, tau=tau)[0]
        uno_24m_swa = concordance_index_ipcw(y_train_struct, y_val_struct, 
                                             risk_24m_swa, tau=tau)[0]
        
        if cfg.PRIMARY_METRIC == "uno_integral":
            best_swa_uno = uno_integral_swa
        else:
            best_swa_uno = uno_24m_swa
        
        # Save SWA model
        swa_checkpoint = {
            "state_dict": swa_model.state_dict(),
            "cuts": cuts.tolist(),
            "uno": best_swa_uno,
            "uno_integral": uno_integral_swa,
            "uno_24m": uno_24m_swa,
            "lr": cfg.LR,
            "clinical_features": feature_names,
            "clinical_dim": len(feature_names),
            "swa_start": swa_start,
            "swa_epochs": epoch - swa_start + 1
        }
        torch.save(swa_checkpoint, fold_dir / "best_swa.pt")
        
        # Report results
        improvement = best_swa_uno - best_uno
        print(f"\n📊 FOLD {fold} RESULTS:", flush=True)
        print(f"   Regular model:", flush=True)
        print(f"      Uno-int: {uno_integral:.3f}", flush=True)
        print(f"      Uno-24m: {uno_24m:.3f}", flush=True)
        print(f"      Primary ({cfg.PRIMARY_METRIC}): {best_uno:.3f}", flush=True)
        print(f"   SWA model:", flush=True)
        print(f"      Uno-int: {uno_integral_swa:.3f}", flush=True)
        print(f"      Uno-24m: {uno_24m_swa:.3f}", flush=True)
        print(f"      Primary ({cfg.PRIMARY_METRIC}): {best_swa_uno:.3f}", flush=True)
        print(f"   Improvement: {improvement:+.3f} ({improvement/best_uno*100:+.2f}%)", flush=True)
        print("="*60, flush=True)
    
    return {
        "regular_uno": best_uno,
        "swa_uno": best_swa_uno if cfg.USE_SWA else None
    }

def main():
    args = parse_args()
    
    # Update config from args
    cfg.LR = args.lr
    cfg.USE_SWA = args.use_swa
    cfg.SWA_START_RATIO = args.swa_start
    
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    set_seed(cfg.SEED)
    
    # Create experiment directory
    swa_suffix = "_swa" if cfg.USE_SWA else ""
    exp_name = args.exp_name or f"lr_{cfg.LR:.0e}{swa_suffix}"
    output_dir = Path(OUTPUT_DIR) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70, flush=True)
    print("FIXED & OPTIMIZED SWA TRAINING", flush=True)
    print("="*70, flush=True)
    print(f"Experiment: {exp_name}", flush=True)
    print(f"Learning Rate: {cfg.LR:.2e}", flush=True)
    print(f"SWA: {'Enabled' if cfg.USE_SWA else 'Disabled'}", flush=True)
    if cfg.USE_SWA:
        print(f"  Start ratio: {cfg.SWA_START_RATIO} ({int(cfg.SWA_START_RATIO * cfg.MAX_EPOCHS)} epochs)", flush=True)
        print(f"  SWA LR factor: {cfg.SWA_LR_FACTOR}", flush=True)
        print(f"  Min SWA epochs: {cfg.SWA_MIN_EPOCHS}", flush=True)
    print(f"TTA: {'Enabled' if cfg.USE_TTA else 'Disabled (for model selection)'}", flush=True)
    print(f"Primary metric: {cfg.PRIMARY_METRIC}", flush=True)
    print("="*70, flush=True)
    
    # Load data
    crop = pd.read_csv(CROP_LOG_CSV)
    phase1 = pd.read_csv(PHASE1_QC_CSV)
    clinical = pd.read_csv(CLINICAL_CSV)
    
    # Merge data
    manifest = crop.merge(phase1[["patient_id", "mask_voxels"]], on="patient_id", how="left")
    manifest = manifest[manifest["mask_voxels"] > 0].reset_index(drop=True)
    
    # Map clinical columns
    col_map = {
        "PatientID": "patient_id",
        "age": "age",
        "clinical.T.Stage": "T",
        "Clinical.N.Stage": "N",
        "Clinical.M.Stage": "M",
        "Overall.Stage": "overall_stage",
        "gender": "gender",
        "Survival.time": "time",
        "deadstatus.event": "event"
    }
    clinical = clinical.rename(columns=col_map)
    
    manifest = manifest.merge(
        clinical[["patient_id", "age", "T", "N", "M", "overall_stage", "gender", "time", "event"]],
        on="patient_id",
        how="left"
    )
    
    print(f"\nDataset: {len(manifest)} patients", flush=True)
    print(f"Events: {manifest['event'].sum()} ({manifest['event'].mean()*100:.1f}%)", flush=True)
    print(f"Median survival: {manifest['time'].median():.0f} days", flush=True)
    
    # Run folds
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    
    results = []
    for fold, (idx_train, idx_val) in enumerate(skf.split(np.zeros(len(manifest)), 
                                                           manifest["event"])):
        if args.fold is not None and fold != args.fold:
            continue
        
        print(f"\n{'='*70}", flush=True)
        print(f"FOLD {fold}", flush=True)
        print(f"{'='*70}", flush=True)
        
        fold_results = run_fold_with_swa(fold, manifest, idx_train, idx_val, output_dir)
        
        results.append({
            "fold": fold, 
            "regular_uno": fold_results["regular_uno"],
            "swa_uno": fold_results["swa_uno"],
            "lr": cfg.LR
        })
    
    # Save and report results
    results_df = pd.DataFrame(results)
    
    if args.fold is None:
        results_df.to_csv(output_dir / "results_complete.csv", index=False)
        
        print("\n" + "="*70, flush=True)
        print(f"FINAL RESULTS (LR={cfg.LR:.2e})", flush=True)
        print("="*70, flush=True)
        print(results_df, flush=True)
        print(flush=True)
        
        reg_mean = results_df['regular_uno'].mean()
        reg_std = results_df['regular_uno'].std()
        print(f"Regular Model: {reg_mean:.3f} ± {reg_std:.3f}", flush=True)
        
        if cfg.USE_SWA:
            swa_mean = results_df['swa_uno'].mean()
            swa_std = results_df['swa_uno'].std()
            improvement = swa_mean - reg_mean
            pct_improvement = (improvement / reg_mean) * 100
            print(f"SWA Model:     {swa_mean:.3f} ± {swa_std:.3f}", flush=True)
            print(f"Improvement:   {improvement:+.3f} ({pct_improvement:+.2f}%)", flush=True)
            print(flush=True)
            print(f"✅ SWA {'succeeded' if improvement > 0.003 else 'helped'} in improving performance!", flush=True)
        
        print("="*70, flush=True)

if __name__ == "__main__":
    main()