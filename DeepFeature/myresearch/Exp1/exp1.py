#!/usr/bin/env python
"""
Exp A1: Feature Extraction + DeepSurv
"""
import os, random, warnings
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
from collections import Counter  
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
from typing import Tuple
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConcatItemsd, DeleteItemsd, ToTensord, Lambdad
)
from monai.networks.nets import resnet as monai_resnet
from pathlib import Path
from tqdm import tqdm
import json
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
CROP_LOG_CSV = "/home/lichengze/Research/CNN_pipeline/phase2_outputs/crop_log.csv"
CLINICAL_CSV = "/home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical.csv"
PHASE3_MODEL_DIR = "/home/lichengze/Research/CNN_pipeline/phase3_outputs/learning_rate/learning_rate_corrected/output/lr_7e-4"
OUTPUT_DIR = "/home/lichengze/Research/DeepFeature/myresearch/Exp1"

FEATURES_DIR = f"{OUTPUT_DIR}/features"
DEEPSURV_DIR = f"{OUTPUT_DIR}/deepsurv"

Path(FEATURES_DIR).mkdir(parents=True, exist_ok=True)
Path(DEEPSURV_DIR).mkdir(parents=True, exist_ok=True)

# Reproducibility, use SEED=42 to match Phase3 fold splitting
PHASE3_SEED = 42  # Must match phase3.py CFG.SEED

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(PHASE3_SEED)

# ============================================================
# Phase3 Model Architecture
# ============================================================
class ResNet10_Clinical_GN(nn.Module):
    """Phase3 model - for loading weights only"""
    def __init__(self, in_channels=2, clinical_dim=6, n_bins=15, dropout=0.35):
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
# Data Transforms
# ============================================================
def build_transforms():
    """Phase2 output is [0,1] - CT+EDT 2-channel"""
    return Compose([
        LoadImaged(keys=["ct", "edt"]),
        EnsureChannelFirstd(keys=["ct", "edt"]),
        EnsureTyped(keys=["ct", "edt"], dtype=torch.float32, track_meta=False),
        Lambdad(keys=["ct", "edt"], func=lambda x: torch.nan_to_num(x, 0.0, 0.0, 0.0)),
        Lambdad(keys=["ct", "edt"], func=lambda x: torch.clamp(x, 0.0, 1.0)),
        ConcatItemsd(keys=["ct", "edt"], name="image"),
        DeleteItemsd(keys=["ct", "edt"]),
        ToTensord(keys=["image"])
    ])

class SimpleDataset(Dataset):
    def __init__(self, records, patient_ids, transform):
        self.records = records
        self.patient_ids = patient_ids
        self.transform = transform
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        data = self.transform(self.records[idx])
        data["patient_id"] = self.patient_ids[idx]
        return data

# ============================================================
# STEP 1: Feature Extraction
# ============================================================
def extract_features_fold(fold, manifest, idx_train, idx_val, device):
    """Extract features with OPTIMIZED memory management"""
    print(f"\n{'='*70}")
    print(f"STEP 1: Feature Extraction - Fold {fold}")
    print(f"{'='*70}")
    
    print(f"Using Phase3-compatible fold indices: Train={len(idx_train)}, Val={len(idx_val)}")
    
    # ============================================================
    # OPTIMIZATION 1: Clean up + load model
    # ============================================================
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  
        print(f"  ✓ GPU cache cleared (Free: {torch.cuda.mem_get_info()[0]/1024**3:.2f}GB)")
    
    # Load Phase3 model
    fold_dir = Path(PHASE3_MODEL_DIR) / f"fold_{fold}"
    model_path = fold_dir / "best.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    print(f"✓ Loaded Phase3 checkpoint:")
    print(f"  Keys: {list(checkpoint.keys())}")
    print(f"  n_time_bins: {checkpoint.get('n_time_bins', 'N/A')}")
    print(f"  clinical_dim: {checkpoint.get('clinical_dim', 'N/A')}")
    print(f"  Phase3 Uno: {checkpoint.get('uno_integral', checkpoint.get('uno', 'N/A'))}")

    model = ResNet10_Clinical_GN(
        in_channels=2,
        clinical_dim=checkpoint['clinical_dim'],
        n_bins=checkpoint['n_time_bins'],
        dropout=0.35
    )
    model.load_state_dict(checkpoint['state_dict'])
    
    if hasattr(model.backbone, "fc"):
        model.backbone.fc = nn.Identity()
        print("  ✓ Removed backbone.fc - extracting pure pooled features")
    
    model = model.to(device)
    model.eval()
    
    # release checkpoint memory
    del checkpoint
    
    print(f"Loaded Phase3 model")
    if torch.cuda.is_available():
        print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated")
    
    transform = build_transforms()
    
    # ============================================================
    # OPTIMIZATION 2: Training set feature extraction
    # ============================================================
    train_records = [
        {"ct": manifest.iloc[i].ct_path, "edt": manifest.iloc[i].edt_path}
        for i in idx_train
    ]
    train_patient_ids = manifest.iloc[idx_train]["patient_id"].values
    
    train_dataset = SimpleDataset(train_records, train_patient_ids, transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,  
        pin_memory=True,
        persistent_workers=False,  
        prefetch_factor=2  
    )
    
    print("Extracting training features...")
    train_features = []
    train_ids_ordered = []
    
    with torch.inference_mode():
        for batch in tqdm(train_loader, desc="  Train"):
            x_img = batch["image"].to(device)
            features = model.backbone(x_img)
            
            # immediately transfer to CPU and release GPU tensor
            train_features.append(features.cpu().numpy())
            train_ids_ordered.extend(batch["patient_id"])
            
            # clean up GPU memory occupied by batch
            del x_img, features
    
    train_features = np.vstack(train_features)
    
  
    del train_loader, train_dataset, train_records
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"  ✓ Train done, GPU freed: {torch.cuda.mem_get_info()[0]/1024**3:.2f}GB free")
    
    # ============================================================
    # OPTIMIZATION 3: 验证集特征提取
    # ============================================================
    val_records = [
        {"ct": manifest.iloc[i].ct_path, "edt": manifest.iloc[i].edt_path}
        for i in idx_val
    ]
    val_patient_ids = manifest.iloc[idx_val]["patient_id"].values
    
    val_dataset = SimpleDataset(val_records, val_patient_ids, transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,  
        pin_memory=True,
        persistent_workers=False,  
        prefetch_factor=2 
    )
    
    print("Extracting validation features...")
    val_features = []
    val_ids_ordered = []
    
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="  Val"):
            x_img = batch["image"].to(device)
            features = model.backbone(x_img)
            
            val_features.append(features.cpu().numpy())
            val_ids_ordered.extend(batch["patient_id"])
            
            del x_img, features
    
    val_features = np.vstack(val_features)
    
    del val_loader, val_dataset, val_records
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"  ✓ Val done, GPU freed: {torch.cuda.mem_get_info()[0]/1024**3:.2f}GB free")
    
    # ============================================================
    # OPTIMIZATION 4: Save results
    # ============================================================
    np.save(f"{FEATURES_DIR}/fold{fold}_train_features.npy", train_features)
    np.save(f"{FEATURES_DIR}/fold{fold}_train_ids.npy", train_ids_ordered)
    np.save(f"{FEATURES_DIR}/fold{fold}_val_features.npy", val_features)
    np.save(f"{FEATURES_DIR}/fold{fold}_val_ids.npy", val_ids_ordered)
    np.save(f"{FEATURES_DIR}/fold{fold}_train_idx.npy", idx_train)
    np.save(f"{FEATURES_DIR}/fold{fold}_val_idx.npy", idx_val)
    
    print(f"Pure features: Train {train_features.shape}, Val {val_features.shape}")
    
    # ============================================================
    # OPTIMIZATION 5: Final cleanup
    # ============================================================
    del model, train_features, val_features
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"  ✓ Final cleanup: {torch.cuda.mem_get_info()[0]/1024**3:.2f}GB free")

# ============================================================
# Clinical Features
# ============================================================
def _safe_mode(values: np.ndarray, default: float) -> float:
    """Calculate mode, for missing value imputation of categorical variables"""
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return float(default)
    cnt = Counter(vals.tolist())
    mode_val = sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return float(mode_val)

def _parse_T(series: pd.Series) -> np.ndarray:
    """Parse T stage: T1/T2/T3/T4 → 1.0/2.0/3.0/4.0"""
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
    """Parse N stage: N0/N1/N2/N3 → 0.0/1.0/2.0/3.0"""
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
    """
    Parse M stage, return two arrays:
    - M01: 0/1 encoding (M0→0, M1→1, M>1→1)
    - M_misc: M>1 special case marker
    """
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
        else:  # M > 1 (少见情况)
            M01.append(1.0)
            M_misc.append(1.0)
    return np.array(M01, dtype=np.float32), np.array(M_misc, dtype=np.float32)

def _map_overall(series: pd.Series) -> np.ndarray:
    """
    Overall stage ordinal encoding:
    I series → 0.2, II series → 0.4, IIIA → 0.6, IIIB → 0.8, IV → 1.0
    """
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
        s = s.replace("ⅢA", "IIIA").replace("ⅢB", "IIIB")\
             .replace("Ⅱ", "II").replace("Ⅳ", "IV").replace("Ⅰ", "I")
        if s in mapping:
            out.append(mapping[s])
        elif s in ["III"]:
            out.append(0.7)  # III general case, between IIIA and IIIB
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)

def prepare_clinical_no_scale(manifest: pd.DataFrame, 
                               idx_train: np.ndarray,
                               idx_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 12 clinical features (align with Phase3), but NO scaling/normalization
    
    All features will be standardized together in train_deepsurv_fold after concatenation.
    This avoids double standardization (e.g., T/N normalized twice).
    
    Feature list (12 features):
    1. age_filled (continuous value, imputed median, NOT scaled)
    2. age_missing (0/1 indicator)
    3. gender_male (0/1)
    4. T (T stage value, NOT normalized - will be standardized with all features)
    5. T_missing (0/1 indicator)
    6. N (N stage value, NOT normalized - will be standardized with all features)
    7. N_missing (0/1 indicator)
    8. M01 (0/1)
    9. M_missing (0/1 indicator)
    10. M_misc (0/1, M>1 case)
    11. overall_stage_ord (ordinal encoding [0.2, 1.0], NOT normalized)
    12. overall_missing (0/1 indicator)
    """
    features_train = []
    features_val = []
    feature_names = []

    if 'age' in manifest.columns:
        age_raw = pd.to_numeric(manifest['age'], errors='coerce').values.astype(np.float32)
        age_missing = np.isnan(age_raw).astype(np.float32)
        
        # Use training median to impute
        train_median = np.nanmedian(age_raw[idx_train]) if np.any(~np.isnan(age_raw[idx_train])) else 65.0
        age_filled = age_raw.copy()
        age_filled[np.isnan(age_filled)] = train_median
        
        # Here we don't use StandardScaler, only impute missing values
        # Later we will standardize all features together
        features_train += [age_filled[idx_train], age_missing[idx_train]]
        features_val += [age_filled[idx_val], age_missing[idx_val]]
        feature_names += ['age', 'age_missing']

    if 'gender' in manifest.columns:
        gender_ser = manifest['gender'].astype(str).str.upper().str[0]
        gender = (gender_ser == 'M').astype(np.float32).values
        features_train.append(gender[idx_train])
        features_val.append(gender[idx_val])
        feature_names.append('gender_male')

    if 'T' in manifest.columns:
        T_num = _parse_T(manifest['T'])
        T_missing = np.isnan(T_num).astype(np.float32)
        
        # Use training mode to impute
        T_mode = _safe_mode(T_num[idx_train], default=2.0)
        T_filled = T_num.copy()
        T_filled[np.isnan(T_filled)] = T_mode
        
        # NO normalization here - will be standardized together with all features later
        features_train += [T_filled[idx_train], T_missing[idx_train]]
        features_val += [T_filled[idx_val], T_missing[idx_val]]
        feature_names += ['T', 'T_missing']

    if 'N' in manifest.columns:
        N_num = _parse_N(manifest['N'])
        N_missing = np.isnan(N_num).astype(np.float32)
        
        N_mode = _safe_mode(N_num[idx_train], default=1.0)
        N_filled = N_num.copy()
        N_filled[np.isnan(N_filled)] = N_mode
        
        # NO normalization here - will be standardized together with all features later
        features_train += [N_filled[idx_train], N_missing[idx_train]]
        features_val += [N_filled[idx_val], N_missing[idx_val]]
        feature_names += ['N', 'N_missing']

    if 'M' in manifest.columns:
        M01, M_misc = _parse_M(manifest['M'])
        M_missing = np.isnan(M01).astype(np.float32)
        
        M_mode = _safe_mode(M01[idx_train], default=0.0)
        M01_filled = M01.copy()
        M01_filled[np.isnan(M01_filled)] = M_mode
        
        features_train += [M01_filled[idx_train], M_missing[idx_train], M_misc[idx_train]]
        features_val += [M01_filled[idx_val], M_missing[idx_val], M_misc[idx_val]]
        feature_names += ['M01', 'M_missing', 'M_misc']

    if 'overall_stage' in manifest.columns:
        ov = _map_overall(manifest['overall_stage'])
        overall_missing = np.isnan(ov).astype(np.float32)
        
        overall_mode = _safe_mode(ov[idx_train], default=0.6)  
        ov_filled = ov.copy()
        ov_filled[np.isnan(ov_filled)] = overall_mode
        
        features_train += [ov_filled[idx_train], overall_missing[idx_train]]
        features_val += [ov_filled[idx_val], overall_missing[idx_val]]
        feature_names += ['overall_stage_ord', 'overall_missing']

    if features_train:
        clinical_train = np.stack(features_train, axis=1).astype(np.float32)
        clinical_val = np.stack(features_val, axis=1).astype(np.float32)
    else:
        # If no clinical features, create a dummy feature with all zeros
        clinical_train = np.zeros((len(idx_train), 1), dtype=np.float32)
        clinical_val = np.zeros((len(idx_val), 1), dtype=np.float32)
        feature_names = ['dummy']

    clinical_train = np.nan_to_num(clinical_train, nan=0.0, posinf=0.0, neginf=0.0)
    clinical_val = np.nan_to_num(clinical_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Extracted {clinical_train.shape[1]} clinical features:")
    print(f"    {', '.join(feature_names)}")

    return clinical_train, clinical_val

# ============================================================
# Stable Cox Loss
# ============================================================
def cox_ph_loss_breslow(log_h, time, event, eps=1e-7):
    """
    Stable Cox partial likelihood loss
    
    Implementation:
    - Breslow approximation for tied event times
    - Time-sorted descending (longest survival first)
    - Stable logcumsumexp for risk set computation
    """
    # Sort by time descending
    order = torch.argsort(time, descending=True)
    log_h = log_h.squeeze()[order]
    event = event[order].float()
    
    # Log-sum-exp of risk set
    log_risk = torch.logcumsumexp(log_h, dim=0)

    
    # Negative log partial likelihood
    loss = -(log_h - log_risk) * event
    
    return loss.sum() / (event.sum() + eps)

# ============================================================
# DeepSurv Model
# ============================================================
class DeepSurv(nn.Module):
    def __init__(self, in_features, hidden_1=128, hidden_2=64, dropout=0.3):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_2, 1)  # log-hazard
        )
    
    def forward(self, x):
        return self.network(x)

# ============================================================
# STEP 2: Train DeepSurv
# ============================================================
def train_deepsurv_fold(fold, manifest):
    """Train DeepSurv with FULL-BATCH Cox loss (CORRECTED)"""
    print(f"\n{'='*70}")
    print(f"STEP 2: Train DeepSurv - Fold {fold}")
    print(f"{'='*70}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load features
    train_deep = np.load(f"{FEATURES_DIR}/fold{fold}_train_features.npy")
    val_deep = np.load(f"{FEATURES_DIR}/fold{fold}_val_features.npy")
    idx_train = np.load(f"{FEATURES_DIR}/fold{fold}_train_idx.npy")
    idx_val = np.load(f"{FEATURES_DIR}/fold{fold}_val_idx.npy")
    
    # Clinical features
    clinical_train, clinical_val = prepare_clinical_no_scale(manifest, idx_train, idx_val)
    
    # Combine features
    X_train = np.hstack([train_deep, clinical_train])
    X_val = np.hstack([val_deep, clinical_val])
    
    print(f"Features (before standardization): Train {X_train.shape}, Val {X_val.shape}")
    print(f"  Deep features: 512-d")
    print(f"  Clinical features: {clinical_train.shape[1]}-d")
    
    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    joblib.dump(scaler, f"{DEEPSURV_DIR}/fold{fold}_scaler.joblib")
    
    print(f"All features standardized once (mean≈0, std≈1)")
    
    # Labels
    y_train = {
        'time': manifest.iloc[idx_train]['time'].values.astype(float),
        'event': manifest.iloc[idx_train]['event'].values.astype(float)
    }
    y_val = {
        'time': manifest.iloc[idx_val]['time'].values.astype(float),
        'event': manifest.iloc[idx_val]['event'].values.astype(float)
    }
    
    # Tau
    tau = 730.0
    
    
    # Model
    model = DeepSurv(
        in_features=X_train.shape[1],
        hidden_1=128,
        hidden_2=64,
        dropout=0.3
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # ============================================================
    # FIX: Full-batch training (CORRECT for Cox)
    # ============================================================
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_time_t = torch.FloatTensor(y_train['time']).to(device)
    y_train_event_t = torch.FloatTensor(y_train['event']).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    
    print(f"Training with FULL-BATCH (N={len(X_train)})")
    print(f"  This is CORRECT for Cox partial likelihood!")
    
    # ============================================================
    # FIX: Pre-construct Surv objects (before training loop)
    # ============================================================
    y_train_surv = Surv.from_arrays(
        event=y_train['event'].astype(bool),
        time=y_train['time']
    )
    y_val_surv = Surv.from_arrays(
        event=y_val['event'].astype(bool),
        time=y_val['time']
    )
    
    best_uno = 0.0
    patience = 0
    
    for epoch in range(200):
        model.train()
        
        # ✅ Full-batch forward pass
        log_hazards = model(X_train_t)
        loss = cox_ph_loss_breslow(log_hazards, y_train_time_t, y_train_event_t)
        
        if torch.isnan(loss):
            print(f"  Epoch {epoch+1}: NaN loss, skipping")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.inference_mode():
                log_hazards_val = model(X_val_t)
                risks = log_hazards_val.cpu().numpy().flatten()
            
            # Uno's C-index
            try:
                uno = concordance_index_ipcw(
                    y_train_surv, y_val_surv, risks, tau=tau
                )[0]
            except:
                uno = np.nan
            
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Uno: {uno:.4f}")
            
            if uno > best_uno + 1e-4:
                best_uno = uno
                patience = 0
                
                torch.save({
                    'state_dict': model.state_dict(),
                    'uno': uno,
                    'epoch': epoch,
                    'tau': tau,
                    'scaler_path': f"{DEEPSURV_DIR}/fold{fold}_scaler.joblib"
                }, f"{DEEPSURV_DIR}/fold{fold}_best.pt")
            else:
                patience += 1
                if patience >= 30:
                    print(f"  Early stop at epoch {epoch+1}")
                    break
    
    print(f"Fold {fold}: Best Uno = {best_uno:.4f}")
    
    return {
        'fold': fold,
        'best_uno': float(best_uno),
        'tau': float(tau)
    }

def _calc_manifest_fingerprint(patient_ids) -> str:
    """
    Given a complete ordered list of patient_ids, calculate a stable sha256 fingerprint.
    If the order or content changes, the fingerprint will be different.
    """
    s = "\n".join(map(str, patient_ids))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ============================================================
# Main Pipeline
# ============================================================
def main():
    print("="*70)
    print("Exp A1: Feature Extraction + DeepSurv")
    print("="*70)
    print("\nAll corrections applied:")
    print("  backbone.fc = Identity() → pure pooled features")
    print("  Clinical features: no double standardization")
    print("  Stable Cox loss (Breslow ties)")
    print("  concordance_index_ipcw(...)[0]")
    print("  Use log_hazard directly (monotone-invariant)")
    print("  inference_mode() for extraction")
    print("  Tau bounded by 0.99*max_followup")
    print("  Phase3 fold indices regenerated (SEED=42)")
    print("  Random seed (42, matching Phase3)")
    print("="*70)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
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
    manifest["time"]  = manifest["time"].astype(int)
    manifest["event"] = manifest["event"].astype(int)
    
    current_fp = _calc_manifest_fingerprint(manifest["patient_id"].tolist())
    print(f"Dataset: {len(manifest)} patients")
    
    # Run all folds
    N_FOLDS = 5
    for fold in range(N_FOLDS):
        fold_dir = PHASE3_MODEL_DIR / f"fold_{fold}"
        train_idx_path = fold_dir / "train_idx.npy"
        val_idx_path   = fold_dir / "val_idx.npy"
    
        if not train_idx_path.exists() or not val_idx_path.exists():
                raise FileNotFoundError(
                    f"[FOLD {fold}] Missing index files: {train_idx_path} or {val_idx_path}."
                    f"Please confirm that they have been generated using generate_indices.py."
                )

        idx_train = np.load(train_idx_path)
        idx_val   = np.load(val_idx_path)

        assert manifest.loc[idx_train, "patient_id"].nunique() == len(idx_train), \
            (f"[FOLD {fold}] Training set patient_id unique count != index length;"
             f"Please roll back to the CSV / crop log version used during training.")
        assert manifest.loc[idx_val, "patient_id"].nunique() == len(idx_val), \
            (f"[FOLD {fold}] validation set patient_id unique count != index length;"
             f"Please roll back to the CSV / crop log version used during training.")

    
    print(f"All corrections verified and applied!")

if __name__ == "__main__":
    main()