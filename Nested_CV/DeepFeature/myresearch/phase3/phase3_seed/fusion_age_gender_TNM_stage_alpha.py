#!/usr/bin/env python
# -*- coding: utf-8 -*-
# fusion_age_gender_TNM_stage_v2.py
#
# Additive Fusion Protocol (frozen image head + clinical residual)
# Clinical: age + gender + T + N + M + overall_stage  (6-dim)
#
# Supports:
#   - zero_init         : last layer W,b = 0  (clinical logits start at 0)
#   - small_init        : last layer W ~ N(0, init_std), b = 0
#   - gate_alpha_fixed  : logits = img + alpha * clin_logits, alpha = softplus(alpha_raw) + alpha_min
#                         and clin last layer uses small_init (init_std) for stable magnitude
#
# Reproducibility:
#   - per-fold seed reset: fold_seed = base_seed + fold_seed_stride * fold
#   - deterministic DataLoader: generator + worker_init_fn
#   - TF32 disabled by default
#   - best-effort deterministic algorithms (optional strict mode)
#
# JSONL output includes per-fold:
#   stop_ep, last_uno, first_over_baseline_ep
#   plus rms diagnostics and alpha stats (if gate mode)

import argparse
import json
import os
import platform
import random
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.loss import NLLLogistiHazardLoss
from sksurv.metrics import concordance_index_ipcw

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Seed / determinism utils
# =========================
def set_seed(seed: int, deterministic: bool = True, strict_deterministic: bool = False):
    """
    Best-effort determinism. For strict determinism, set strict_deterministic=True to raise if not supported.
    """
    # Note: ideally set before python starts; still useful as a record.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # cuBLAS determinism (needs to be set before CUDA context; we set anyway + print it)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = (not deterministic)

    # TF32 off by default for reproducibility (can be toggled by args)
    # (will be set in main after args are parsed)

    # Deterministic algorithms (best-effort)
    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception as e:
        if strict_deterministic:
            raise
        # otherwise ignore (still seed-controlled)


def seed_worker(worker_id: int):
    """
    Ensure each DataLoader worker has deterministic numpy/random state.
    Works when num_workers > 0.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==================
# Config
# ==================
@dataclass
class CFG:
    BATCH_SIZE: int = 64
    EPOCHS: int = 100
    PATIENCE: int = 25
    LR: float = 1e-3
    WD: float = 5e-4
    CLIN_DROPOUT: float = 0.3
    GRAD_CLIP: float = 1.0


# ==================
# Clinical utils
# ==================
def _map_gender_to_float(s) -> float:
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    if s in ["m", "male", "1"]:
        return 1.0
    if s in ["f", "female", "0"]:
        return 0.0
    return np.nan


def _map_overall_stage_to_group(s) -> float:
    """
    overall stage → ordinal 1..4
    I* → 1, II* → 2, III* → 3, IV* → 4
    """
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    if not s:
        return np.nan

    for tok in ["stage", "stg"]:
        s = s.replace(tok, "")
    s = s.replace(" ", "").replace("-", "")

    if s.startswith("iv"):
        return 4.0
    if s.startswith("iii"):
        return 3.0
    if s.startswith("ii"):
        return 2.0
    if s.startswith("i"):
        return 1.0

    for ch in s:
        if ch.isdigit():
            v = int(ch)
            if 1 <= v <= 4:
                return float(v)
            break
    return np.nan


def load_clinical_data(csv_path: str) -> pd.DataFrame:
    """
    Loader: patient_id, time, event, age, gender, t_stage, n_stage, m_stage, overall_stage
    """
    df = pd.read_csv(csv_path)

    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if "patientid" in cl or cl == "patient_id":
            col_map[col] = "patient_id"
        elif "survival" in cl and "time" in cl:
            col_map[col] = "time"
        elif "deadstatus" in cl or "event" in cl:
            col_map[col] = "event"
        elif cl == "age":
            col_map[col] = "age"
        elif cl in ["gender", "sex"]:
            col_map[col] = "gender"
        elif "t.stage" in cl or "t_stage" in cl:
            col_map[col] = "t_stage"
        elif "n.stage" in cl or "n_stage" in cl:
            col_map[col] = "n_stage"
        elif "m.stage" in cl or "m_stage" in cl:
            col_map[col] = "m_stage"
        elif ("overall" in cl and "stage" in cl) or cl in ["stage", "overall_stage", "ajcc_stage", "stage_group"]:
            col_map[col] = "overall_stage"

    df = df.rename(columns=col_map)

    required = ["patient_id", "time", "event", "age", "gender"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in clinical CSV: {missing}")

    df["patient_id"] = df["patient_id"].astype(str)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce").astype(int)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # optional columns
    if "t_stage" in df.columns:
        df["t_stage"] = pd.to_numeric(df["t_stage"], errors="coerce")
        print(f"[Clinical] T stage valid: {df['t_stage'].notna().sum()} / {len(df)}")
    else:
        df["t_stage"] = np.nan
        print("[Clinical] T stage column not found, filled as NaN")

    if "n_stage" in df.columns:
        df["n_stage"] = pd.to_numeric(df["n_stage"], errors="coerce")
        print(f"[Clinical] N stage valid: {df['n_stage'].notna().sum()} / {len(df)}")
    else:
        df["n_stage"] = np.nan
        print("[Clinical] N stage column not found, filled as NaN")

    if "m_stage" in df.columns:
        df["m_stage"] = pd.to_numeric(df["m_stage"], errors="coerce")
        print(f"[Clinical] M stage valid: {df['m_stage'].notna().sum()} / {len(df)}")
        print(f"[Clinical] M stage distribution: {df['m_stage'].value_counts().to_dict()}")
    else:
        df["m_stage"] = np.nan
        print("[Clinical] M stage column not found, filled as NaN")

    if "overall_stage" in df.columns:
        df["overall_stage"] = df["overall_stage"].apply(_map_overall_stage_to_group)
        print(f"[Clinical] Overall stage valid: {df['overall_stage'].notna().sum()} / {len(df)}")
        print(f"[Clinical] Overall stage distribution: {df['overall_stage'].value_counts().sort_index().to_dict()}")
    else:
        df["overall_stage"] = np.nan
        print("[Clinical] Overall stage column not found, filled as NaN")

    print(f"[Clinical] Total patients: {len(df)}")
    print(f"[Clinical] Age missing: {df['age'].isna().sum()}")

    return df.set_index("patient_id")


def build_clinical_features(
    df_indexed: pd.DataFrame,
    t_pids: List[str],
    v_pids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    6-dim: age_z, gender, t_z, n_z, m_bin, overall_stage_z
    """
    tr = df_indexed.loc[t_pids].copy()
    va = df_indexed.loc[v_pids].copy()

    feats_tr: List[np.ndarray] = []
    feats_va: List[np.ndarray] = []

    # Age z-score (train stats)
    age_mean = tr["age"].mean()
    tr["age"] = tr["age"].fillna(age_mean)
    va["age"] = va["age"].fillna(age_mean)
    mu = tr["age"].mean()
    std = tr["age"].std() + 1e-8
    feats_tr.append(((tr["age"].values - mu) / std).astype(np.float32))
    feats_va.append(((va["age"].values - mu) / std).astype(np.float32))

    # Gender (mode impute from train)
    g_tr = np.array([_map_gender_to_float(x) for x in tr["gender"]], dtype=float)
    g_va = np.array([_map_gender_to_float(x) for x in va["gender"]], dtype=float)
    g_valid = g_tr[~np.isnan(g_tr)]
    g_mode = pd.Series(g_valid).mode()[0] if len(g_valid) > 0 else 0.5
    g_tr = np.nan_to_num(g_tr, nan=g_mode).astype(np.float32)
    g_va = np.nan_to_num(g_va, nan=g_mode).astype(np.float32)
    feats_tr.append(g_tr)
    feats_va.append(g_va)

    # T z-score
    t_valid = tr["t_stage"].dropna()
    t_mode = t_valid.mode()[0] if len(t_valid) > 0 else 2.0
    tr["t_stage"] = tr["t_stage"].fillna(t_mode)
    va["t_stage"] = va["t_stage"].fillna(t_mode)
    mu = tr["t_stage"].mean()
    std = tr["t_stage"].std() + 1e-8
    feats_tr.append(((tr["t_stage"].values - mu) / std).astype(np.float32))
    feats_va.append(((va["t_stage"].values - mu) / std).astype(np.float32))

    # N z-score
    n_valid = tr["n_stage"].dropna()
    n_mode = n_valid.mode()[0] if len(n_valid) > 0 else 1.0
    tr["n_stage"] = tr["n_stage"].fillna(n_mode)
    va["n_stage"] = va["n_stage"].fillna(n_mode)
    mu = tr["n_stage"].mean()
    std = tr["n_stage"].std() + 1e-8
    feats_tr.append(((tr["n_stage"].values - mu) / std).astype(np.float32))
    feats_va.append(((va["n_stage"].values - mu) / std).astype(np.float32))

    # M binarized
    m_tr_raw = tr["m_stage"].fillna(0).values
    m_va_raw = va["m_stage"].fillna(0).values
    m_tr = (m_tr_raw >= 1).astype(np.float32)
    m_va = (m_va_raw >= 1).astype(np.float32)
    feats_tr.append(m_tr)
    feats_va.append(m_va)
    print(f"[Features] M binarized: train M1+={int(m_tr.sum())}/{len(m_tr)}, val M1+={int(m_va.sum())}/{len(m_va)}")

    # Overall stage z-score
    st_valid = tr["overall_stage"].dropna()
    st_mode = st_valid.mode()[0] if len(st_valid) > 0 else 2.0
    tr["overall_stage"] = tr["overall_stage"].fillna(st_mode)
    va["overall_stage"] = va["overall_stage"].fillna(st_mode)
    mu = tr["overall_stage"].mean()
    std = tr["overall_stage"].std() + 1e-8
    feats_tr.append(((tr["overall_stage"].values - mu) / std).astype(np.float32))
    feats_va.append(((va["overall_stage"].values - mu) / std).astype(np.float32))

    X_tr = np.stack(feats_tr, axis=1).astype(np.float32)
    X_va = np.stack(feats_va, axis=1).astype(np.float32)
    print(f"[Features] Shape: train={X_tr.shape}, val={X_va.shape}")

    return (
        X_tr,
        X_va,
        tr["time"].values.astype(float),
        tr["event"].values.astype(int),
        va["time"].values.astype(float),
        va["event"].values.astype(int),
    )


# ==================
# UNO helpers
# ==================
def _to_struct(times, events):
    return np.array([(bool(e), float(t)) for t, e in zip(times, events)], dtype=[("event", bool), ("time", float)])


def _uno_from_logits(logits: torch.Tensor, bin_mids: np.ndarray, y_tr_struct, y_va_struct, tau: float) -> float:
    haz = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1e-7)
    log_surv = torch.cumsum(torch.log(1.0 - haz), dim=1)
    surv = torch.exp(log_surv)  # [N, n_bins]
    risk = -np.trapz(surv.detach().cpu().numpy(), x=bin_mids, axis=1)
    return float(concordance_index_ipcw(y_tr_struct, y_va_struct, risk, tau=tau)[0])


def _rms(x: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(x.float() ** 2)).detach().cpu().item())


# ==================
# Image artifacts
# ==================
class ImageHead(nn.Module):
    def __init__(self, n_bins: int, dropout: float = 0.35):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_bins),
        )

    def forward(self, x):
        return self.head(x)


def _load_image_head_from_ckpt(ckpt_path: Path, dropout: float = 0.35) -> Tuple[ImageHead, np.ndarray, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "cuts" not in ckpt:
        raise ValueError(f"{ckpt_path} missing 'cuts'")
    cuts = np.array(ckpt["cuts"], dtype=float)
    n_bins = ckpt.get("n_time_bins", len(cuts) - 1)

    head = ImageHead(n_bins=n_bins, dropout=dropout)

    state_dict = ckpt.get("state_dict", ckpt)
    head_state = {}
    for k, v in state_dict.items():
        if "head." in k:
            k_clean = k.replace("module.", "") if k.startswith("module.") else k
            head_state[k_clean] = v

    if not head_state:
        raise ValueError(f"No 'head.' parameters in {ckpt_path}")

    head.load_state_dict(head_state, strict=False)
    head.eval()
    for p in head.parameters():
        p.requires_grad = False

    print(f"[{ckpt_path.name}] ImageHead loaded: {len(head_state)} params")
    return head.to(DEVICE), cuts, n_bins


def load_phase3_artifacts(exp_dir: Path, fold: int, dropout_head: float = 0.35):
    fold_dir = exp_dir / f"fold_{fold}"
    ckpt_path = fold_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found.")

    t_emb = np.load(fold_dir / "train_embeddings.npy")
    v_emb = np.load(fold_dir / "val_embeddings.npy")
    t_pids = pd.read_csv(fold_dir / "train_pids.csv", header=None)[0].astype(str).tolist()
    v_pids = pd.read_csv(fold_dir / "val_pids.csv", header=None)[0].astype(str).tolist()
    img_head, cuts, n_bins = _load_image_head_from_ckpt(ckpt_path, dropout=dropout_head)
    return t_emb, v_emb, t_pids, v_pids, img_head, cuts, n_bins


# ==================
# Fusion model
# ==================
class ClinicalNet(nn.Module):
    """
    in_dim -> 32 -> n_bins
    init modes are handled by caller (zero_init/small_init)
    """
    def __init__(self, in_dim: int, n_bins: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_bins),
        )

    def forward(self, x):
        return self.net(x)


class GateAlpha(nn.Module):
    """
    alpha = softplus(alpha_raw) + alpha_min, alpha >= alpha_min
    """
    def __init__(self, alpha_min: float = 1e-2, alpha_raw_init: float = -6.0):
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_raw_init)))

    def alpha(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha_raw) + self.alpha_min

    def forward(self, clin_logits: torch.Tensor) -> torch.Tensor:
        return self.alpha() * clin_logits


class AdditiveFusion(nn.Module):
    """
    - zero_init / small_init: logits = img + clin
    - gate_alpha_fixed      : logits = img + alpha * clin
    """
    def __init__(self, img_head: nn.Module, clin_net: nn.Module, gate: Optional[GateAlpha] = None):
        super().__init__()
        self.img_head = img_head
        self.clin_net = clin_net
        self.gate = gate

        self.img_head.eval()
        for p in self.img_head.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # keep img head frozen & eval
        self.img_head.eval()
        for p in self.img_head.parameters():
            p.requires_grad = False
        return self

    def forward(self, emb: torch.Tensor, clin: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits_img = self.img_head(emb)
        logits_clin = self.clin_net(clin)
        if self.gate is not None:
            logits_clin_eff = self.gate(logits_clin)
            return logits_img + logits_clin_eff
        return logits_img + logits_clin


class FusionDS(Dataset):
    def __init__(self, emb, clin, y_idx, y_evt):
        self.emb = torch.from_numpy(emb).float()
        self.clin = torch.from_numpy(clin).float()
        self.y_idx = torch.from_numpy(y_idx).long()
        self.y_evt = torch.from_numpy(y_evt).float()

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, i):
        return self.emb[i], self.clin[i], self.y_idx[i], self.y_evt[i]


def init_clinical_last_layer(clin_net: ClinicalNet, mode: str, init_std: float):
    """
    Apply only to the last Linear(32 -> n_bins).
    """
    last = clin_net.net[-1]
    assert isinstance(last, nn.Linear), "Expected last layer to be nn.Linear"
    if mode == "zero_init":
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    elif mode in ["small_init", "gate_alpha_fixed"]:
        nn.init.normal_(last.weight, mean=0.0, std=float(init_std))
        nn.init.zeros_(last.bias)
    else:
        raise ValueError(f"Unknown init mode: {mode}")


@torch.no_grad()
def compute_init_diag(
    model: AdditiveFusion,
    v_emb_f: np.ndarray,
    X_va: np.ndarray,
    fusion_mode: str,
    max_n: int = 128,
) -> Dict[str, float]:
    n = min(len(v_emb_f), max_n)
    emb = torch.from_numpy(v_emb_f[:n]).float().to(DEVICE)
    clin = torch.from_numpy(X_va[:n]).float().to(DEVICE)

    model.eval()
    # compute components
    logits_img = model.img_head(emb)
    logits_clin = model.clin_net(clin)
    rms_img = _rms(logits_img)
    rms_clin_logits = _rms(logits_clin)

    if model.gate is not None:
        alpha_val = float(model.gate.alpha().detach().cpu().item())
        clin_eff = model.gate(logits_clin)
        rms_clin_eff = _rms(clin_eff)
    else:
        alpha_val = 1.0
        clin_eff = logits_clin
        if fusion_mode == "zero_init":
            # should be exactly 0 (up to float noise)
            pass
        rms_clin_eff = _rms(clin_eff)

    ratio = (rms_clin_eff / (rms_img + 1e-12)) if rms_img > 0 else 0.0
    out = {
        "rms_img": float(rms_img),
        "rms_clin_logits": float(rms_clin_logits),
        "rms_clin_eff": float(rms_clin_eff),
        "ratio": float(ratio),
        "alpha_init": float(alpha_val),
    }
    return out


def append_jsonl(path: Optional[str], record: Dict):
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_run_header(args):
    print("\n" + "=" * 80)
    print(f"[Run] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Run] Python: {sys.version.replace(os.linesep, ' ')}")
    print(f"[Run] Platform: {platform.system().lower()}")
    print(f"[Run] Torch: {torch.__version__}")
    print(f"[Run] DEVICE: {DEVICE}")
    print(f"[Run] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"[Run] CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
        try:
            print(f"[Run] cuDNN: {torch.backends.cudnn.version()}")
        except Exception:
            pass
    print(f"[Run] CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--clinical_csv", type=str, required=True)

    # training config
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout_clin", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--print_every", type=int, default=1)

    # reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Single base seed (if --seed_start not used).")
    parser.add_argument("--seed_start", type=int, default=None, help="Start of base seed range (e.g., 42).")
    parser.add_argument("--n_seeds", type=int, default=1, help="How many base seeds (e.g., 20).")
    parser.add_argument("--fold_seed_stride", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no_deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--strict_deterministic", action="store_true", default=False)
    parser.add_argument("--allow_tf32", action="store_true", default=False)

    # fusion modes
    parser.add_argument("--fusion_mode", type=str, required=True,
                        choices=["zero_init", "small_init", "gate_alpha_fixed"])
    parser.add_argument("--init_std", type=float, default=0.01,
                        help="Used by small_init and gate_alpha_fixed to init last layer.")
    # gate-alpha params
    parser.add_argument("--alpha_min", type=float, default=1e-2)
    parser.add_argument("--alpha_raw_init", type=float, default=-6.0)
    parser.add_argument("--alpha_lr_mult", type=float, default=1.0)

    # logging
    parser.add_argument("--save_jsonl", type=str, default=None)

    args = parser.parse_args()

    cfg = CFG(
        BATCH_SIZE=args.batch_size,
        EPOCHS=args.epochs,
        PATIENCE=args.patience,
        LR=args.lr,
        WD=args.weight_decay,
        CLIN_DROPOUT=args.dropout_clin,
        GRAD_CLIP=args.grad_clip,
    )

    # TF32 controls
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    print_run_header(args)

    run_id = str(uuid.uuid4())[:8]
    run_meta = {
        "type": "run",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "cfg": vars(cfg),
    }
    append_jsonl(args.save_jsonl, run_meta)

    # load clinical once
    clin_df = load_clinical_data(args.clinical_csv)
    valid_pids = set(clin_df.index)

    exp_dir = Path(args.exp_dir)

    print("\n" + "=" * 80)
    print("ADDITIVE FUSION PROTOCOL")
    print("- Clinical: age + gender + T + N + M + overall_stage (6-dim)")
    print(f"- fusion_mode={args.fusion_mode}")
    print(f"- LR={cfg.LR:.3e}  WD={cfg.WD:.3e}  dropout={cfg.CLIN_DROPOUT:.2f}")
    print(f"- epochs={cfg.EPOCHS}  patience={cfg.PATIENCE}  batch={cfg.BATCH_SIZE}")
    print(f"- deterministic={args.deterministic}  num_workers={args.num_workers}  allow_tf32={args.allow_tf32}")
    if args.fusion_mode in ["small_init", "gate_alpha_fixed"]:
        print(f"- init_std={args.init_std}")
    if args.fusion_mode == "gate_alpha_fixed":
        print(f"- alpha_min={args.alpha_min}  alpha_raw_init={args.alpha_raw_init}  alpha_lr_mult={args.alpha_lr_mult}")
    print("=" * 80)

    # base seeds list
    if args.seed_start is not None:
        base_seeds = [int(args.seed_start) + i for i in range(int(args.n_seeds))]
    else:
        base_seeds = [int(args.seed)]

    for base_seed in base_seeds:
        print("\n" + "-" * 80)
        print(f"[SeedRun] base_seed={base_seed}")
        print("-" * 80)

        seed_fold_final: List[float] = []
        seed_fold_base: List[float] = []
        seed_fold_delta: List[float] = []
        improved_folds = 0
        fallback_folds = 0

        for fold in range(5):
            fold_seed = int(base_seed) + int(args.fold_seed_stride) * int(fold)
            set_seed(fold_seed, deterministic=args.deterministic, strict_deterministic=args.strict_deterministic)

            print(f"\n=== Fold {fold} ===")
            print(f"[Seed] base_seed={base_seed}, stride={args.fold_seed_stride} -> fold_seed={fold_seed}")

            try:
                t_emb, v_emb, t_pids, v_pids, img_head, cuts, n_bins = load_phase3_artifacts(exp_dir, fold, dropout_head=0.35)
            except Exception as e:
                print(f"[Fold {fold}] Skip: {e}")
                continue

            # filter to pids that exist in clinical
            tr_mask = [p in valid_pids for p in t_pids]
            va_mask = [p in valid_pids for p in v_pids]
            t_emb_f = t_emb[tr_mask]
            v_emb_f = v_emb[va_mask]
            t_pids_f = [p for p, m in zip(t_pids, tr_mask) if m]
            v_pids_f = [p for p, m in zip(v_pids, va_mask) if m]
            print(f"[Data] N_train={len(t_emb_f)} / {len(t_emb)}, N_val={len(v_emb_f)} / {len(v_emb)}")

            X_tr, X_va, t_tr, e_tr, t_va, e_va = build_clinical_features(clin_df, t_pids_f, v_pids_f)

            # label transform consistent with image model cuts
            lab = LabTransDiscreteTime(cuts=cuts[1:-1])
            y_tr_idx, y_tr_evt = lab.transform(t_tr, e_tr)
            y_va_idx, y_va_evt = lab.transform(t_va, e_va)

            # UNO setup
            bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
            y_tr_struct = _to_struct(t_tr, e_tr)
            y_va_struct = _to_struct(t_va, e_va)
            evt_times = t_tr[e_tr == 1]
            tau = float(np.quantile(evt_times, 0.9)) if len(evt_times) > 0 else float(np.max(t_tr))
            print(f"[Data] Event rate train={e_tr.mean():.2%}, val={e_va.mean():.2%}")
            print(f"[UNO] tau(90% event time)={tau:.4f}  (n_events_train={int((e_tr==1).sum())})")

            # baseline
            img_head.eval()
            with torch.no_grad():
                logits_base = img_head(torch.from_numpy(v_emb_f).float().to(DEVICE))
                uno_base = _uno_from_logits(logits_base, bin_mids, y_tr_struct, y_va_struct, tau)
            print(f"[Fold {fold}] Image-only baseline Uno={uno_base:.4f}")

            # build fusion model
            clin_net = ClinicalNet(in_dim=X_tr.shape[1], n_bins=n_bins, dropout=cfg.CLIN_DROPOUT).to(DEVICE)
            gate = None
            if args.fusion_mode == "gate_alpha_fixed":
                gate = GateAlpha(alpha_min=args.alpha_min, alpha_raw_init=args.alpha_raw_init).to(DEVICE)
            init_clinical_last_layer(clin_net, mode=args.fusion_mode, init_std=args.init_std)
            model = AdditiveFusion(img_head, clin_net, gate=gate).to(DEVICE)

            # init diagnostics
            init_diag = compute_init_diag(model, v_emb_f, X_va, fusion_mode=args.fusion_mode, max_n=128)
            if args.fusion_mode == "gate_alpha_fixed":
                print(f"[InitDiag] RMS(img)={init_diag['rms_img']:.6f}  RMS(clin_logits)={init_diag['rms_clin_logits']:.6f}  "
                      f"RMS(clin_eff)={init_diag['rms_clin_eff']:.6f}  ratio={init_diag['ratio']:.6f}  alpha_init={init_diag['alpha_init']:.6f}")
            else:
                print(f"[InitDiag] RMS(img)={init_diag['rms_img']:.6f}  RMS(clin_logits)={init_diag['rms_clin_logits']:.6f}  "
                      f"RMS(clin_eff)={init_diag['rms_clin_eff']:.6f}  ratio={init_diag['ratio']:.6f}")

            # loaders (deterministic shuffle)
            g = torch.Generator()
            g.manual_seed(fold_seed)

            dl_tr = DataLoader(
                FusionDS(t_emb_f, X_tr, y_tr_idx, y_tr_evt),
                batch_size=cfg.BATCH_SIZE,
                shuffle=True,
                generator=g,
                worker_init_fn=seed_worker,
                num_workers=args.num_workers,
            )
            dl_va = DataLoader(
                FusionDS(v_emb_f, X_va, y_va_idx, y_va_evt),
                batch_size=cfg.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                num_workers=args.num_workers,
            )

            # optimizer: separate group for gate alpha (no WD)
            params = [{"params": clin_net.parameters(), "lr": cfg.LR, "weight_decay": cfg.WD}]
            if gate is not None:
                params.append({"params": gate.parameters(), "lr": cfg.LR * float(args.alpha_lr_mult), "weight_decay": 0.0})
            optimizer = torch.optim.Adam(params)
            loss_fn = NLLLogistiHazardLoss()

            best_uno = float(uno_base)
            best_ep = 0
            no_improve = 0

            stop_ep = 0
            last_uno = float("nan")
            first_over_baseline_ep: Optional[int] = None
            alpha_best = float(init_diag.get("alpha_init", 1.0))

            for ep in range(1, cfg.EPOCHS + 1):
                model.train()
                total_loss = 0.0

                for emb_b, clin_b, yi_b, ye_b in dl_tr:
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(emb_b.to(DEVICE), clin_b.to(DEVICE))
                    loss = loss_fn(logits, yi_b.to(DEVICE), ye_b.to(DEVICE))
                    loss.backward()
                    nn.utils.clip_grad_norm_(clin_net.parameters(), cfg.GRAD_CLIP)
                    optimizer.step()
                    total_loss += float(loss.item()) * emb_b.size(0)

                avg_loss = total_loss / max(1, len(X_tr))

                model.eval()
                with torch.no_grad():
                    logits_list = []
                    for emb_b, clin_b, _, _ in dl_va:
                        logits_list.append(model(emb_b.to(DEVICE), clin_b.to(DEVICE)))
                    logits_va = torch.cat(logits_list, dim=0)
                    uno_curr = _uno_from_logits(logits_va, bin_mids, y_tr_struct, y_va_struct, tau)

                    # alpha tracking
                    alpha_curr = 1.0
                    if gate is not None:
                        alpha_curr = float(gate.alpha().detach().cpu().item())

                last_uno = float(uno_curr)
                stop_ep = ep

                if first_over_baseline_ep is None and (uno_curr > uno_base + 1e-4):
                    first_over_baseline_ep = ep

                if (ep % max(1, args.print_every)) == 0:
                    if gate is not None:
                        print(f"[Fold {fold}] Ep {ep:03d} | loss={avg_loss:.4f} | Uno={uno_curr:.4f} | alpha={alpha_curr:.5f}")
                    else:
                        print(f"[Fold {fold}] Ep {ep:03d} | loss={avg_loss:.4f} | Uno={uno_curr:.4f}")

                if uno_curr > best_uno + 1e-4:
                    best_uno = float(uno_curr)
                    best_ep = ep
                    no_improve = 0
                    alpha_best = float(alpha_curr)
                else:
                    no_improve += 1
                    if no_improve >= cfg.PATIENCE:
                        break

            # report final (baseline-safe)
            final_uno = max(best_uno, float(uno_base))
            delta = float(final_uno - float(uno_base))
            improved = bool(final_uno > float(uno_base) + 1e-3)
            status = "IMPROVED" if improved else "FALLBACK/FLAT"

            if improved:
                improved_folds += 1
            else:
                fallback_folds += 1

            alpha_final = 1.0
            if gate is not None:
                alpha_final = float(gate.alpha().detach().cpu().item())

            print(f"[Fold {fold}] Early stopping at ep {stop_ep} (best_ep={best_ep}, best_uno={best_uno:.4f})")
            print(f"[Fold {fold}] Final Uno={final_uno:.4f} | baseline={uno_base:.4f} | Δ={delta:+.4f} | best_ep={best_ep} | {status}")
            if gate is not None:
                print(f"[Fold {fold}] alpha_best={alpha_best:.6f}  alpha_final={alpha_final:.6f}")
            print(f"[Fold {fold}] first_over_baseline_ep={first_over_baseline_ep}  last_uno={last_uno:.4f}  stop_ep={stop_ep}")

            # json record per fold
            rec = {
                "type": "fold",
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "fusion_mode": args.fusion_mode,
                "base_seed": int(base_seed),
                "fold": int(fold),
                "fold_seed": int(fold_seed),
                "cfg": {
                    "lr": float(cfg.LR),
                    "wd": float(cfg.WD),
                    "dropout": float(cfg.CLIN_DROPOUT),
                    "epochs": int(cfg.EPOCHS),
                    "patience": int(cfg.PATIENCE),
                    "batch": int(cfg.BATCH_SIZE),
                },
                "mode_params": {
                    "init_std": float(args.init_std) if args.fusion_mode in ["small_init", "gate_alpha_fixed"] else None,
                    "alpha_min": float(args.alpha_min) if args.fusion_mode == "gate_alpha_fixed" else None,
                    "alpha_raw_init": float(args.alpha_raw_init) if args.fusion_mode == "gate_alpha_fixed" else None,
                    "alpha_lr_mult": float(args.alpha_lr_mult) if args.fusion_mode == "gate_alpha_fixed" else None,
                },
                "data": {
                    "n_train": int(len(t_tr)),
                    "n_val": int(len(t_va)),
                    "event_rate_train": float(e_tr.mean()),
                    "event_rate_val": float(e_va.mean()),
                    "tau": float(tau),
                },
                "metrics": {
                    "baseline_uno": float(uno_base),
                    "best_uno": float(best_uno),
                    "best_ep": int(best_ep),
                    "final_uno": float(final_uno),
                    "delta": float(delta),
                    "improved": bool(improved),
                    # The 3 fields you explicitly asked to add:
                    "stop_ep": int(stop_ep),
                    "last_uno": float(last_uno),
                    "first_over_baseline_ep": (int(first_over_baseline_ep) if first_over_baseline_ep is not None else None),
                },
                "init_diag": init_diag,
                "alpha": {
                    "alpha_best": float(alpha_best),
                    "alpha_final": float(alpha_final),
                } if args.fusion_mode == "gate_alpha_fixed" else None,
            }
            append_jsonl(args.save_jsonl, rec)

            seed_fold_final.append(final_uno)
            seed_fold_base.append(float(uno_base))
            seed_fold_delta.append(delta)

        # seed-level summary
        if seed_fold_final:
            seed_summary = {
                "type": "seed_summary",
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "fusion_mode": args.fusion_mode,
                "base_seed": int(base_seed),
                "mean_final_uno": float(np.mean(seed_fold_final)),
                "std_final_uno": float(np.std(seed_fold_final)),
                "mean_baseline_uno": float(np.mean(seed_fold_base)),
                "std_baseline_uno": float(np.std(seed_fold_base)),
                "mean_delta": float(np.mean(seed_fold_delta)),
                "std_delta": float(np.std(seed_fold_delta)),
                "improved_folds": int(improved_folds),
                "fallback_folds": int(fallback_folds),
            }
            append_jsonl(args.save_jsonl, seed_summary)

            print("\n" + "=" * 80)
            print(f"[SeedRun Summary] base_seed={base_seed}  fusion_mode={args.fusion_mode}")
            print(f"Final Uno per fold: {[f'{x:.4f}' for x in seed_fold_final]}")
            print(f"Mean Final Uno: {np.mean(seed_fold_final):.4f} ± {np.std(seed_fold_final):.4f}")
            print(f"Mean Baseline Uno: {np.mean(seed_fold_base):.4f} ± {np.std(seed_fold_base):.4f}")
            print(f"Mean Δ (Final - Baseline): {np.mean(seed_fold_delta):+.4f} ± {np.std(seed_fold_delta):.4f}")
            print(f"Improved folds: {improved_folds}/5  | Fallback/flat folds: {fallback_folds}/5")
            print("=" * 80)

    # run footer
    append_jsonl(args.save_jsonl, {"type": "run_end", "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds")})


if __name__ == "__main__":
    main()
