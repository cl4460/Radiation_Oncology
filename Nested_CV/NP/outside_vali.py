#!/usr/bin/env python3
"""
phase4_external_test_donoharm.py
================================
External validation on NSCLC-Radiogenomics using do-no-harm trained models.

Key Features:
1. Reads `use_clinical` from checkpoint to decide whether to use clinical features
2. Uses same clinical processing as training (ordinal encoding, missing indicators)
3. 4-view flip TTA for inference
4. Computes Uno C-index with proper IPCW weighting

Usage:
    python phase4_external_test_donoharm.py \
        --lung1_crop_log /path/to/lung1/crop_log.csv \
        --lung1_clinical /path/to/lung1/clinical.csv \
        --ckpt_root /path/to/phase3_outputs/exp_name \
        --rg_crop_log /path/to/rg/phase2_crop_log.csv \
        --rg_clinical /path/to/rg/clinical_aligned.csv \
        --out_dir /path/to/output
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
    ConcatItemsd, DeleteItemsd, ToTensord, Lambdad,
)
from monai.networks.nets import resnet as monai_resnet

from sksurv.metrics import concordance_index_ipcw, concordance_index_censored, brier_score

warnings.filterwarnings("ignore")


# ============================================================================
# Model Architecture (must match training)
# ============================================================================
class ResNet10_Clinical_GN(nn.Module):
    """ResNet10 backbone with GroupNorm + Clinical encoder."""
    
    def __init__(self, in_channels: int, clinical_dim: int, n_bins: int, dropout: float = 0.35):
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
    
    def _replace_bn_with_gn(self, module, num_groups=8):
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
# Clinical Feature Processing (must match training)
# ============================================================================
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
            out.append(t if 1.0 <= t <= 4.0 else np.nan)
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
            out.append(n if 0.0 <= n <= 3.0 else np.nan)
        else:
            out.append(np.nan)
    return np.array(out, dtype=np.float32)


def _parse_M(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
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


STAGE_ORDINAL_MAP = {
    "I": 0.2, "1": 0.2, "IA": 0.15, "IB": 0.25,
    "II": 0.4, "2": 0.4, "IIA": 0.35, "IIB": 0.45,
    "IIIA": 0.6, "3A": 0.6,
    "III": 0.7,
    "IIIB": 0.8, "3B": 0.8,
    "IV": 1.0, "4": 1.0,
}


def _parse_overall_stage_ordinal(series: pd.Series) -> np.ndarray:
    out = []
    for v in series:
        if pd.isna(v):
            out.append(np.nan)
            continue
        s = str(v).strip().upper().replace("STAGE", "").replace(" ", "")
        s = (s.replace("ⅢA", "IIIA").replace("ⅢB", "IIIB")
              .replace("Ⅲ", "III").replace("Ⅱ", "II")
              .replace("Ⅳ", "IV").replace("Ⅰ", "I"))
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
    """
    feats_tr, feats_te = [], []
    
    # ==================== AGE ====================
    if 'age_scaled' in feature_names:
        age_tr = pd.to_numeric(train_df['age'], errors='coerce').values.astype(np.float32)
        age_te = pd.to_numeric(test_df['age'], errors='coerce').values.astype(np.float32)
        
        age_impute = np.nanmedian(age_tr)
        if np.isnan(age_impute):
            age_impute = 65.0
        
        age_tr_filled = np.where(np.isnan(age_tr), age_impute, age_tr)
        age_te_filled = np.where(np.isnan(age_te), age_impute, age_te)
        
        scaler = StandardScaler()
        scaler.fit(age_tr_filled.reshape(-1, 1))
        
        age_tr_scaled = scaler.transform(age_tr_filled.reshape(-1, 1)).flatten()
        age_te_scaled = scaler.transform(age_te_filled.reshape(-1, 1)).flatten()
        
        age_tr_missing = np.isnan(age_tr).astype(np.float32)
        age_te_missing = np.isnan(age_te).astype(np.float32)
        
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
        T_tr = _parse_T(train_df['T'] if 'T' in train_df.columns else pd.Series([np.nan]*len(train_df)))
        T_te = _parse_T(test_df['T'] if 'T' in test_df.columns else pd.Series([np.nan]*len(test_df)))
        
        T_mode = _safe_mode(T_tr, default=2.0)
        T_tr_filled = np.where(np.isnan(T_tr), T_mode, T_tr)
        T_te_filled = np.where(np.isnan(T_te), T_mode, T_te)
        
        T_tr_norm = (T_tr_filled - 1.0) / 3.0
        T_te_norm = (T_te_filled - 1.0) / 3.0
        
        T_tr_missing = np.isnan(T_tr).astype(np.float32)
        T_te_missing = np.isnan(T_te).astype(np.float32)
        
        feats_tr.extend([T_tr_norm, T_tr_missing])
        feats_te.extend([T_te_norm, T_te_missing])
    
    # ==================== N STAGE ====================
    if 'N_norm' in feature_names:
        N_tr = _parse_N(train_df['N'] if 'N' in train_df.columns else pd.Series([np.nan]*len(train_df)))
        N_te = _parse_N(test_df['N'] if 'N' in test_df.columns else pd.Series([np.nan]*len(test_df)))
        
        N_mode = _safe_mode(N_tr, default=1.0)
        N_tr_filled = np.where(np.isnan(N_tr), N_mode, N_tr)
        N_te_filled = np.where(np.isnan(N_te), N_mode, N_te)
        
        N_tr_norm = N_tr_filled / 3.0
        N_te_norm = N_te_filled / 3.0
        
        N_tr_missing = np.isnan(N_tr).astype(np.float32)
        N_te_missing = np.isnan(N_te).astype(np.float32)
        
        feats_tr.extend([N_tr_norm, N_tr_missing])
        feats_te.extend([N_te_norm, N_te_missing])
    
    # ==================== M STAGE ====================
    if 'M01' in feature_names:
        M01_tr, M_misc_tr = _parse_M(train_df['M'] if 'M' in train_df.columns else pd.Series([np.nan]*len(train_df)))
        M01_te, M_misc_te = _parse_M(test_df['M'] if 'M' in test_df.columns else pd.Series([np.nan]*len(test_df)))
        
        M_mode = _safe_mode(M01_tr, default=0.0)
        M01_tr_filled = np.where(np.isnan(M01_tr), M_mode, M01_tr)
        M01_te_filled = np.where(np.isnan(M01_te), M_mode, M01_te)
        
        M_tr_missing = np.isnan(M01_tr).astype(np.float32)
        M_te_missing = np.isnan(M01_te).astype(np.float32)
        
        feats_tr.extend([M01_tr_filled, M_tr_missing, M_misc_tr])
        feats_te.extend([M01_te_filled, M_te_missing, M_misc_te])
    
    # ==================== OVERALL STAGE ====================
    if 'overall_stage_ord' in feature_names:
        ov_tr = _parse_overall_stage_ordinal(
            train_df['overall_stage'] if 'overall_stage' in train_df.columns else pd.Series([np.nan]*len(train_df))
        )
        ov_te = _parse_overall_stage_ordinal(
            test_df['overall_stage'] if 'overall_stage' in test_df.columns else pd.Series([np.nan]*len(test_df))
        )
        
        ov_mode = _safe_mode(ov_tr, default=0.6)
        ov_tr_filled = np.where(np.isnan(ov_tr), ov_mode, ov_tr)
        ov_te_filled = np.where(np.isnan(ov_te), ov_mode, ov_te)
        
        ov_tr_missing = np.isnan(ov_tr).astype(np.float32)
        ov_te_missing = np.isnan(ov_te).astype(np.float32)
        
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
        data = self.transform(self.records[i])
        data["clinical"] = torch.tensor(self.clinical[i], dtype=torch.float32)
        return data


def build_test_transforms() -> Compose:
    return Compose([
        LoadImaged(keys=["ct", "edt"]),
        EnsureChannelFirstd(keys=["ct", "edt"]),
        EnsureTyped(keys=["ct", "edt"], dtype=torch.float32, track_meta=False),
        Lambdad(keys=["ct", "edt"], func=lambda x: torch.nan_to_num(x, 0.0, 0.0, 0.0)),
        Lambdad(keys=["ct", "edt"], func=lambda x: torch.clamp(x, 0.0, 1.0)),
        ConcatItemsd(keys=["ct", "edt"], name="image"),
        DeleteItemsd(keys=["ct", "edt"]),
        ToTensord(keys=["image"]),
    ])


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
    model.eval()
    
    if not use_tta:
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
) -> Dict[str, float]:
    y_tr = to_structured_y(times_tr, events_tr)
    y_te = to_structured_y(times_te, events_te)
    
    n_time_bins = surv.shape[1]
    bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
    
    risk = -np.trapz(surv, x=bin_mids, axis=1)
    
    evt_times = times_tr[events_tr == 1]
    tau = float(np.quantile(evt_times, 0.9)) if len(evt_times) > 0 else float(np.max(times_tr))
    
    try:
        uno = float(concordance_index_ipcw(y_tr, y_te, risk, tau=tau)[0])
    except Exception:
        uno = float("nan")
    
    try:
        harrell = float(concordance_index_censored(y_te["event"], y_te["time"], risk)[0])
    except Exception:
        harrell = float("nan")
    
    t_idx_24m = int(np.argmin(np.abs(bin_mids - 730)))
    surv_24m = surv[:, t_idx_24m] if t_idx_24m < n_time_bins else surv[:, -1]
    try:
        _, brier = brier_score(y_tr, y_te, surv_24m.reshape(-1, 1), np.array([730]))
        brier_24m = float(brier[0])
    except Exception:
        brier_24m = float("nan")
    
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
def standardize_clinical_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        c = col.lower().strip()
        if 'patient' in c or 'case' in c:
            col_map[col] = 'patient_id'
        elif 'survival' in c and 'time' in c:
            col_map[col] = 'time'
        elif c in ['time', 'time_days', 'os', 'os_days']:
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
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="External Validation on Radiogenomics")
    parser.add_argument("--lung1_crop_log", required=True)
    parser.add_argument("--lung1_clinical", required=True)
    parser.add_argument("--ckpt_root", required=True, help="Path to do-no-harm experiment folder")
    parser.add_argument("--rg_crop_log", required=True)
    parser.add_argument("--rg_clinical", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--no_tta", action="store_true")
    args = parser.parse_args()
    
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("EXTERNAL VALIDATION: LUNG1 -> Radiogenomics")
    print("=" * 80)
    
    # Load LUNG1 data (for IPCW weighting)
    lung1_crop = pd.read_csv(args.lung1_crop_log)
    lung1_clin = standardize_clinical_columns(pd.read_csv(args.lung1_clinical))
    
    lung1 = lung1_crop[["patient_id"]].merge(lung1_clin, on="patient_id", how="inner")
    lung1 = lung1.dropna(subset=["time", "event"])
    lung1["time"] = pd.to_numeric(lung1["time"], errors="coerce").astype(float)
    lung1["event"] = pd.to_numeric(lung1["event"], errors="coerce").astype(int)
    
    times_tr = lung1["time"].values
    events_tr = lung1["event"].values
    
    print(f"LUNG1: n={len(lung1)} events={events_tr.sum()}")
    
    # Load RG data
    rg_crop = pd.read_csv(args.rg_crop_log)
    rg_clin = standardize_clinical_columns(pd.read_csv(args.rg_clinical))
    
    rg = rg_crop[["patient_id", "out_ct", "out_edt"]].rename(columns={
        "out_ct": "ct_path",
        "out_edt": "edt_path",
    }).merge(rg_clin, on="patient_id", how="inner")
    
    # Check if time and event columns exist after standardization
    if "time" not in rg.columns:
        # Try alternative column names
        if "time_days" in rg.columns:
            rg["time"] = rg["time_days"]
        else:
            raise ValueError(f"Radiogenomics data missing 'time' column. Available columns: {list(rg.columns)[:20]}")
    if "event" not in rg.columns:
        raise ValueError(f"Radiogenomics data missing 'event' column. Available columns: {list(rg.columns)[:20]}")
    
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
    
    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        ckpt_path = fold_dir / "best.pt"
        
        if not ckpt_path.exists():
            print(f"  {fold_name}: No checkpoint found, skipping")
            continue
        
        ckpt = torch.load(ckpt_path, map_location="cpu")
        
        use_clinical = ckpt.get("use_clinical", True)
        n_time_bins = ckpt.get("n_time_bins", 13)
        clinical_dim = ckpt.get("clinical_dim", 12)
        cuts = np.array(ckpt.get("cuts", []))
        feature_names = ckpt.get("clinical_features", [])
        defaults = ckpt.get("clinical_defaults", {})
        use_tta = ckpt.get("use_tta", True) and not args.no_tta
        variant = ckpt.get("variant", "unknown")
        
        print(f"\n  {fold_name}: variant={variant}, use_clinical={use_clinical}, bins={n_time_bins}")
        
        # Prepare clinical features
        X_tr_full, X_te = prepare_clinical_external(lung1, rg, feature_names, defaults)
        
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
        
        print(f"    Uno={met['uno']:.3f} Harrell={met['harrell']:.3f} Brier@24m={met['brier_24m']:.3f}")
        
        all_results.append({
            "fold": fold_name,
            "variant": variant,
            "use_clinical": use_clinical,
            "uno": met["uno"],
            "harrell": met["harrell"],
            "brier_24m": met["brier_24m"],
        })
        all_risks.append(met["risk"])
        
        # Save per-fold predictions
        pd.DataFrame({
            "patient_id": rg["patient_id"].values,
            "risk": met["risk"],
            "surv_24m": met["surv_24m"],
            "time": times_te,
            "event": events_te,
        }).to_csv(out_dir / f"{fold_name}_predictions.csv", index=False)
    
    # Ensemble: average risk scores
    if len(all_risks) > 0:
        ensemble_risk = np.mean(all_risks, axis=0)
        
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
        
        print("\n" + "=" * 80)
        print("ENSEMBLE RESULTS (averaged risk)")
        print("=" * 80)
        print(f"Uno C-index: {ensemble_uno:.4f}")
        print(f"Harrell C-index: {ensemble_harrell:.4f}")
        
        # Save ensemble predictions
        pd.DataFrame({
            "patient_id": rg["patient_id"].values,
            "risk_ensemble": ensemble_risk,
            "time": times_te,
            "event": events_te,
        }).to_csv(out_dir / "ensemble_predictions.csv", index=False)
    
    # Save summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "fold_results.csv", index=False)
    
    summary = {
        "n_test": len(rg),
        "n_events": int(events_te.sum()),
        "ensemble_uno": float(ensemble_uno) if len(all_risks) > 0 else None,
        "ensemble_harrell": float(ensemble_harrell) if len(all_risks) > 0 else None,
        "per_fold_uno_mean": float(results_df["uno"].mean()),
        "per_fold_uno_std": float(results_df["uno"].std()),
    }
    
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("PER-FOLD RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print(f"\nMean Uno: {results_df['uno'].mean():.4f} ± {results_df['uno'].std():.4f}")
    print(f"\nResults saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()