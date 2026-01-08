#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
phase3_fusion_age_gender_TNM_stage_solid.py

Frozen ImageHead (from fold_k/best.pt) + Clinical residual (6-dim):
    age_z, gender, t_z, n_z, m_bin, overall_stage_z

Key goals:
1) Reproducibility:
   - Per-fold seed reset: seed_fold = base_seed + stride * fold
   - Deterministic DataLoader shuffling (generator) + worker_init_fn
   - Best-effort deterministic torch/cudnn configs (optional switch)
2) Baseline safety:
   - best_uno starts from image-only baseline
   - final_uno = max(best_uno, baseline)  -> never worse than baseline
3) Diagnostics printing:
   - environment / torch / cuda / cudnn info
   - per-fold: seed, N(train/val), event rate, tau
   - baseline uno, best uno, delta, best epoch, fallback flag
   - initial logits RMS ratio (img vs clin) for init sanity checks
4) Future paired comparisons:
   - fusion_mode: zero_init | small_init | gate_alpha
"""

import os
import sys
import time
import json
import math
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.loss import NLLLogistiHazardLoss
from sksurv.metrics import concordance_index_ipcw

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Reproducibility utilities
# =========================
def set_seed(seed: int, deterministic: bool = True, allow_tf32: bool = False) -> None:
    """
    Best-effort reproducibility seed setter.
    Note:
      - PYTHONHASHSEED is only fully effective if set before interpreter starts.
      - Deterministic algorithms may raise errors for some ops on some versions/hardware.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = (not deterministic)

    try:
        torch.use_deterministic_algorithms(bool(deterministic))
    except Exception:
        pass

    try:
        torch.set_deterministic_debug_mode("warn" if deterministic else "default")
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker seed hook.
    Ensures numpy/random are aligned with torch.initial_seed().
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def print_env_banner() -> None:
    print("\n" + "=" * 80)
    print(f"[Run] Time: {now_str()}")
    print(f"[Run] Python: {sys.version.splitlines()[0]}")
    print(f"[Run] Platform: {sys.platform}")
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
    print(f"[Run] CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', None)}")
    print("=" * 80 + "\n")


# ==================
# Config
# ==================
@dataclass
class CFG:
    BATCH_SIZE: int = 64
    EPOCHS: int = 100
    LR: float = 1e-3
    WD: float = 5e-4
    CLIN_DROPOUT: float = 0.3
    PATIENCE: int = 25
    CLIP_NORM: float = 1.0


cfg = CFG()


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
    Loader aligned with your baseline:
    - patient_id, time, event, age, gender, t_stage, n_stage, m_stage, overall_stage
    """
    df = pd.read_csv(csv_path)

    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if "patientid" in col_lower or col_lower == "patient_id":
            col_map[col] = "patient_id"
        elif "survival" in col_lower and "time" in col_lower:
            col_map[col] = "time"
        elif "deadstatus" in col_lower or "event" in col_lower:
            col_map[col] = "event"
        elif col_lower == "age":
            col_map[col] = "age"
        elif col_lower in ["gender", "sex"]:
            col_map[col] = "gender"
        elif "t.stage" in col_lower or "t_stage" in col_lower:
            col_map[col] = "t_stage"
        elif "n.stage" in col_lower or "n_stage" in col_lower:
            col_map[col] = "n_stage"
        elif "m.stage" in col_lower or "m_stage" in col_lower:
            col_map[col] = "m_stage"
        elif ("overall" in col_lower and "stage" in col_lower) or col_lower in [
            "stage", "overall_stage", "ajcc_stage", "stage_group"
        ]:
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
        print("[Clinical] T stage column not found -> filled NaN")

    if "n_stage" in df.columns:
        df["n_stage"] = pd.to_numeric(df["n_stage"], errors="coerce")
        print(f"[Clinical] N stage valid: {df['n_stage'].notna().sum()} / {len(df)}")
    else:
        df["n_stage"] = np.nan
        print("[Clinical] N stage column not found -> filled NaN")

    if "m_stage" in df.columns:
        df["m_stage"] = pd.to_numeric(df["m_stage"], errors="coerce")
        print(f"[Clinical] M stage valid: {df['m_stage'].notna().sum()} / {len(df)}")
        print(f"[Clinical] M stage distribution: {df['m_stage'].value_counts().to_dict()}")
    else:
        df["m_stage"] = np.nan
        print("[Clinical] M stage column not found -> filled NaN")

    if "overall_stage" in df.columns:
        df["overall_stage"] = df["overall_stage"].apply(_map_overall_stage_to_group)
        print(f"[Clinical] Overall stage valid: {df['overall_stage'].notna().sum()} / {len(df)}")
        print(f"[Clinical] Overall stage distribution: {df['overall_stage'].value_counts().sort_index().to_dict()}")
    else:
        df["overall_stage"] = np.nan
        print("[Clinical] Overall stage column not found -> filled NaN")

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

    # Age (mean-impute on train, then z-score with train stats)
    age_mean_impute = tr["age"].mean()
    tr["age"] = tr["age"].fillna(age_mean_impute)
    va["age"] = va["age"].fillna(age_mean_impute)

    age_mu = tr["age"].mean()
    age_std = tr["age"].std() + 1e-8
    feats_tr.append(((tr["age"].values - age_mu) / age_std).astype(np.float32))
    feats_va.append(((va["age"].values - age_mu) / age_std).astype(np.float32))

    # Gender (mode-impute on train)
    g_tr = np.array([_map_gender_to_float(x) for x in tr["gender"]], dtype=float)
    g_va = np.array([_map_gender_to_float(x) for x in va["gender"]], dtype=float)
    g_valid = g_tr[~np.isnan(g_tr)]
    g_mode = pd.Series(g_valid).mode()[0] if len(g_valid) > 0 else 0.5
    feats_tr.append(np.nan_to_num(g_tr, nan=g_mode).astype(np.float32))
    feats_va.append(np.nan_to_num(g_va, nan=g_mode).astype(np.float32))

    # T (mode-impute on train, z-score with train stats)
    t_valid_tr = tr["t_stage"].dropna()
    t_mode = t_valid_tr.mode()[0] if len(t_valid_tr) > 0 else 2.0
    tr["t_stage"] = tr["t_stage"].fillna(t_mode)
    va["t_stage"] = va["t_stage"].fillna(t_mode)
    t_mu = tr["t_stage"].mean()
    t_std = tr["t_stage"].std() + 1e-8
    feats_tr.append(((tr["t_stage"].values - t_mu) / t_std).astype(np.float32))
    feats_va.append(((va["t_stage"].values - t_mu) / t_std).astype(np.float32))

    # N (mode-impute on train, z-score with train stats)
    n_valid_tr = tr["n_stage"].dropna()
    n_mode = n_valid_tr.mode()[0] if len(n_valid_tr) > 0 else 1.0
    tr["n_stage"] = tr["n_stage"].fillna(n_mode)
    va["n_stage"] = va["n_stage"].fillna(n_mode)
    n_mu = tr["n_stage"].mean()
    n_std = tr["n_stage"].std() + 1e-8
    feats_tr.append(((tr["n_stage"].values - n_mu) / n_std).astype(np.float32))
    feats_va.append(((va["n_stage"].values - n_mu) / n_std).astype(np.float32))

    # M (binarized)
    m_tr_raw = tr["m_stage"].fillna(0).values
    m_va_raw = va["m_stage"].fillna(0).values
    m_tr = (m_tr_raw >= 1).astype(np.float32)
    m_va = (m_va_raw >= 1).astype(np.float32)
    feats_tr.append(m_tr)
    feats_va.append(m_va)

    print(
        f"[Features] M binarized: train M1+={m_tr.sum():.0f}/{len(m_tr)}, "
        f"val M1+={m_va.sum():.0f}/{len(m_va)}"
    )

    # Overall stage (mode-impute on train, z-score with train stats)
    st_valid_tr = tr["overall_stage"].dropna()
    st_mode = st_valid_tr.mode()[0] if len(st_valid_tr) > 0 else 2.0
    tr["overall_stage"] = tr["overall_stage"].fillna(st_mode)
    va["overall_stage"] = va["overall_stage"].fillna(st_mode)
    st_mu = tr["overall_stage"].mean()
    st_std = tr["overall_stage"].std() + 1e-8
    feats_tr.append(((tr["overall_stage"].values - st_mu) / st_std).astype(np.float32))
    feats_va.append(((va["overall_stage"].values - st_mu) / st_std).astype(np.float32))

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
# Uno C-index utils
# ==================
def _to_struct(times: np.ndarray, events: np.ndarray) -> np.ndarray:
    return np.array(
        [(bool(e), float(t)) for t, e in zip(times, events)],
        dtype=[("event", bool), ("time", float)],
    )


def _uno_from_logits(
    logits: torch.Tensor,
    bin_mids: np.ndarray,
    y_tr_struct: np.ndarray,
    y_va_struct: np.ndarray,
    tau: float,
) -> float:
    haz = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1e-7)
    log_surv = torch.cumsum(torch.log(1.0 - haz), dim=1)
    surv = torch.exp(log_surv)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def _load_image_head_from_ckpt(ckpt_path: Path, dropout: float = 0.35) -> Tuple[ImageHead, np.ndarray, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "cuts" not in ckpt:
        raise ValueError(f"{ckpt_path} missing 'cuts'")

    cuts = np.array(ckpt["cuts"], dtype=float)
    n_bins = int(ckpt.get("n_time_bins", len(cuts) - 1))

    head = ImageHead(n_bins=n_bins, dropout=dropout)

    state_dict = ckpt.get("state_dict", ckpt)
    head_state: Dict[str, torch.Tensor] = {}
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
# Fusion model & Dataset
# ==================
class ClinicalNet(nn.Module):
    """
    Clinical residual head: in_dim -> 32 -> n_bins
    init_mode:
      - zero: last layer weights/bias = 0
      - small: last layer weights ~ N(0, init_std), bias=0
      - default: leave PyTorch default init (rarely used here)
    """
    def __init__(self, in_dim: int, n_bins: int, dropout: float, init_mode: str, init_std: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_bins),
        )

        last = self.net[-1]
        assert isinstance(last, nn.Linear)

        if init_mode == "zero":
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif init_mode == "small":
            nn.init.normal_(last.weight, mean=0.0, std=float(init_std))
            nn.init.zeros_(last.bias)
        elif init_mode == "default":
            pass
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdditiveFusion(nn.Module):
    """
    fusion_mode:
      - zero_init: logits = img + clin   (clin last layer init_mode=zero)
      - small_init: logits = img + clin  (clin last layer init_mode=small)
      - gate_alpha: logits = img + alpha * clin, alpha = softplus(a) (a learnable)
                   We recommend init alpha small-but-nonzero to avoid gradient blocking.
    """
    def __init__(self, img_head: nn.Module, clin_net: nn.Module, fusion_mode: str, alpha_init: float):
        super().__init__()
        self.img_head = img_head
        self.clin_net = clin_net
        self.fusion_mode = fusion_mode

        # Freeze image head
        self.img_head.eval()
        for p in self.img_head.parameters():
            p.requires_grad = False

        # Gate alpha (softplus to keep alpha >= 0)
        if fusion_mode == "gate_alpha":
            alpha0 = float(alpha_init)
            alpha0 = max(alpha0, 1e-8)
            # inverse softplus: a = log(exp(alpha)-1)
            a0 = math.log(math.expm1(alpha0))
            self.alpha_param = nn.Parameter(torch.tensor(a0, dtype=torch.float32))
        else:
            self.alpha_param = None

    def alpha(self) -> torch.Tensor:
        if self.alpha_param is None:
            return torch.tensor(1.0, device=DEVICE)
        return torch.nn.functional.softplus(self.alpha_param)

    def train(self, mode: bool = True):
        super().train(mode)
        # keep img_head always eval/frozen
        self.img_head.eval()
        for p in self.img_head.parameters():
            p.requires_grad = False
        return self

    @torch.no_grad()
    def logits_img(self, emb: torch.Tensor) -> torch.Tensor:
        self.img_head.eval()
        return self.img_head(emb)

    def logits_clin(self, clin: torch.Tensor) -> torch.Tensor:
        return self.clin_net(clin)

    def forward(self, emb: torch.Tensor, clin: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            li = self.img_head(emb)
        lc = self.clin_net(clin)

        if self.fusion_mode == "gate_alpha":
            a = self.alpha()
            return li + a * lc
        else:
            return li + lc


class FusionDS(Dataset):
    def __init__(self, emb: np.ndarray, clin: np.ndarray, y_idx: np.ndarray, y_evt: np.ndarray):
        self.emb = torch.from_numpy(emb).float()
        self.clin = torch.from_numpy(clin).float()
        self.y_idx = torch.from_numpy(y_idx).long()
        self.y_evt = torch.from_numpy(y_evt).float()

    def __len__(self) -> int:
        return len(self.emb)

    def __getitem__(self, i: int):
        return self.emb[i], self.clin[i], self.y_idx[i], self.y_evt[i]


# ==================
# Main
# ==================
def main():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--clinical_csv", type=str, required=True)

    # Optim
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--weight_decay", type=float, default=cfg.WD)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--dropout_clin", type=float, default=cfg.CLIN_DROPOUT)
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=cfg.PATIENCE)
    parser.add_argument("--clip_norm", type=float, default=cfg.CLIP_NORM)

    # Repro
    parser.add_argument("--seed", type=int, default=42, help="Base seed (paired comparisons use same seeds).")
    parser.add_argument("--fold_seed_stride", type=int, default=1000, help="seed_fold = seed + stride*fold")
    parser.add_argument("--num_workers", type=int, default=0, help="0 is strictest reproducibility.")
    parser.add_argument("--deterministic", action="store_true", help="Enable best-effort deterministic mode.")
    parser.add_argument("--no_deterministic", dest="deterministic", action="store_false", help="Disable deterministic mode.")
    parser.set_defaults(deterministic=True)
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 (faster, less strict reproducibility).")

    # Fusion protocol switches (for later paired tests)
    parser.add_argument("--fusion_mode", type=str, default="zero_init",
                        choices=["zero_init", "small_init", "gate_alpha"],
                        help="Protocol to compare under same seeds.")
    parser.add_argument("--init_std", type=float, default=0.01,
                        help="For small_init: last-layer weight std.")
    parser.add_argument("--alpha_init", type=float, default=0.01,
                        help="For gate_alpha: initial alpha (softplus) > 0, small keeps baseline.")

    # Logging
    parser.add_argument("--print_every", type=int, default=1, help="Print every N epochs.")
    parser.add_argument("--save_jsonl", type=str, default="",
                        help="Optional: append per-fold results to this JSONL file.")

    args = parser.parse_args()

    print_env_banner()

    exp_dir = Path(args.exp_dir)
    clin_df = load_clinical_data(args.clinical_csv)
    valid_pids = set(clin_df.index)

    print("\n" + "=" * 80)
    print("ADDITIVE FUSION PROTOCOL")
    print("- Clinical: age + gender + T + N + M + overall_stage (6-dim)")
    print(f"- fusion_mode={args.fusion_mode}")
    if args.fusion_mode == "small_init":
        print(f"- init_std={args.init_std}")
    if args.fusion_mode == "gate_alpha":
        print(f"- alpha_init={args.alpha_init}")
    print(f"- LR={args.lr:.3e}  WD={args.weight_decay:.3e}  dropout={args.dropout_clin:.2f}")
    print(f"- epochs={args.epochs}  patience={args.patience}  batch={args.batch_size}")
    print(f"- deterministic={args.deterministic}  num_workers={args.num_workers}  allow_tf32={args.allow_tf32}")
    print("=" * 80 + "\n")

    fold_records: List[Dict[str, Any]] = []

    for fold in range(5):
        print(f"\n=== Fold {fold} ===")

        # --- Per-fold seed reset (critical for “clean” folds)
        fold_seed = int(args.seed + args.fold_seed_stride * fold)
        set_seed(fold_seed, deterministic=args.deterministic, allow_tf32=args.allow_tf32)
        print(f"[Seed] base_seed={args.seed}, stride={args.fold_seed_stride} -> fold_seed={fold_seed}")

        # --- Load artifacts
        try:
            t_emb, v_emb, t_pids, v_pids, img_head, cuts, n_bins = load_phase3_artifacts(exp_dir, fold, dropout_head=0.35)
        except Exception as e:
            print(f"[Fold {fold}] Skip (artifact load error): {e}")
            continue

        # --- Filter pids
        tr_mask = [p in valid_pids for p in t_pids]
        va_mask = [p in valid_pids for p in v_pids]
        t_emb_f = t_emb[tr_mask]
        v_emb_f = v_emb[va_mask]
        t_pids_f = [p for p, m in zip(t_pids, tr_mask) if m]
        v_pids_f = [p for p, m in zip(v_pids, va_mask) if m]
        print(f"[Data] N_train={len(t_pids_f)} / {len(t_pids)}, N_val={len(v_pids_f)} / {len(v_pids)}")

        # --- Build clinical features
        X_tr, X_va, t_tr, e_tr, t_va, e_va = build_clinical_features(clin_df, t_pids_f, v_pids_f)
        print(f"[Data] Event rate train={e_tr.mean():.2%}, val={e_va.mean():.2%}")

        # --- Labels / bins
        lab = LabTransDiscreteTime(cuts=cuts[1:-1])
        y_tr_idx, y_tr_evt = lab.transform(t_tr, e_tr)
        y_va_idx, y_va_evt = lab.transform(t_va, e_va)

        bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
        y_tr_struct = _to_struct(t_tr, e_tr)
        y_va_struct = _to_struct(t_va, e_va)

        evt_times = t_tr[e_tr == 1]
        tau = float(np.quantile(evt_times, 0.9)) if len(evt_times) > 0 else float(np.max(t_tr))
        print(f"[UNO] tau(90% event time)={tau:.4f}  (n_events_train={len(evt_times)})")

        # --- Baseline (image-only)
        img_head.eval()
        with torch.no_grad():
            logits_base = img_head(torch.from_numpy(v_emb_f).float().to(DEVICE))
            uno_base = _uno_from_logits(logits_base, bin_mids, y_tr_struct, y_va_struct, tau)
        print(f"[Fold {fold}] Image-only baseline Uno={uno_base:.4f}")

        # --- Clinical net init mode determined by fusion_mode
        if args.fusion_mode == "zero_init":
            init_mode = "zero"
        elif args.fusion_mode == "small_init":
            init_mode = "small"
        elif args.fusion_mode == "gate_alpha":
            # gate_alpha typically pairs well with default/small.
            # We'll keep default init for clin weights + alpha small (you can change to 'small' if needed).
            init_mode = "default"
        else:
            raise ValueError("Unexpected fusion_mode")

        clin_net = ClinicalNet(
            in_dim=X_tr.shape[1],
            n_bins=n_bins,
            dropout=float(args.dropout_clin),
            init_mode=init_mode,
            init_std=float(args.init_std),
        ).to(DEVICE)

        model = AdditiveFusion(
            img_head=img_head,
            clin_net=clin_net,
            fusion_mode=args.fusion_mode,
            alpha_init=float(args.alpha_init),
        ).to(DEVICE)

        # --- Deterministic DataLoader shuffling
        g = torch.Generator()
        g.manual_seed(fold_seed)

        dl_tr = DataLoader(
            FusionDS(t_emb_f, X_tr, y_tr_idx, y_tr_evt),
            batch_size=int(args.batch_size),
            shuffle=True,
            generator=g,
            num_workers=int(args.num_workers),
            worker_init_fn=seed_worker,
            pin_memory=(DEVICE == "cuda"),
        )
        dl_va = DataLoader(
            FusionDS(v_emb_f, X_va, y_va_idx, y_va_evt),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            worker_init_fn=seed_worker,
            pin_memory=(DEVICE == "cuda"),
        )

        # --- Init diagnostics: RMS of logits contributions (first val batch)
        model.eval()
        with torch.no_grad():
            emb0, clin0, _, _ = next(iter(dl_va))
            emb0 = emb0.to(DEVICE)
            clin0 = clin0.to(DEVICE)
            li0 = model.logits_img(emb0)
            lc0 = model.logits_clin(clin0)
            if args.fusion_mode == "gate_alpha":
                a0 = float(model.alpha().detach().cpu().item())
                lc_eff0 = model.alpha() * lc0
                print(f"[InitDiag] alpha0={a0:.6f}")
            else:
                lc_eff0 = lc0

            rms_img = _rms(li0)
            rms_clin = _rms(lc_eff0)
            ratio = (rms_clin / (rms_img + 1e-12))
            print(f"[InitDiag] RMS(img)={rms_img:.6f}  RMS(clin_eff)={rms_clin:.6f}  ratio={ratio:.6f}")

        # --- Train
        optimizer = torch.optim.Adam(clin_net.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        loss_fn = NLLLogistiHazardLoss()

        best_uno = float(uno_base)  # start from baseline so we never “accept” worse
        best_ep = 0
        no_improve = 0

        for ep in range(1, int(args.epochs) + 1):
            model.train()
            total_loss = 0.0

            for emb_b, clin_b, yi_b, ye_b in dl_tr:
                optimizer.zero_grad(set_to_none=True)
                logits = model(emb_b.to(DEVICE), clin_b.to(DEVICE))
                loss = loss_fn(logits, yi_b.to(DEVICE), ye_b.to(DEVICE))
                loss.backward()
                nn.utils.clip_grad_norm_(clin_net.parameters(), float(args.clip_norm))
                optimizer.step()
                total_loss += float(loss.item()) * emb_b.size(0)

            avg_loss = total_loss / max(1, len(X_tr))

            # Validation Uno
            model.eval()
            with torch.no_grad():
                logits_list = []
                for emb_b, clin_b, _, _ in dl_va:
                    logits_list.append(model(emb_b.to(DEVICE), clin_b.to(DEVICE)))
                logits_va = torch.cat(logits_list, dim=0)
                uno_curr = _uno_from_logits(logits_va, bin_mids, y_tr_struct, y_va_struct, tau)

            if (ep % int(args.print_every)) == 0:
                print(f"[Fold {fold}] Ep {ep:03d} | loss={avg_loss:.4f} | Uno={uno_curr:.4f}")

            # Early stopping on improvement beyond baseline-anchored best_uno
            if uno_curr > best_uno + 1e-4:
                best_uno = float(uno_curr)
                best_ep = ep
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= int(args.patience):
                    print(f"[Fold {fold}] Early stopping at ep {ep} (best_ep={best_ep}, best_uno={best_uno:.4f})")
                    break

        final_uno = max(best_uno, float(uno_base))  # baseline safety
        improved = (best_uno > float(uno_base) + 1e-3)
        status = "IMPROVED" if improved else "FALLBACK/FLAT"
        delta = final_uno - float(uno_base)

        print(f"[Fold {fold}] Final Uno={final_uno:.4f} | baseline={uno_base:.4f} | Δ={delta:+.4f} | best_ep={best_ep} | {status}")

        rec = {
            "time": now_str(),
            "exp_dir": str(exp_dir),
            "clinical_csv": str(args.clinical_csv),
            "fold": int(fold),
            "fold_seed": int(fold_seed),
            "fusion_mode": str(args.fusion_mode),
            "init_std": float(args.init_std),
            "alpha_init": float(args.alpha_init),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "dropout_clin": float(args.dropout_clin),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "deterministic": bool(args.deterministic),
            "num_workers": int(args.num_workers),
            "allow_tf32": bool(args.allow_tf32),
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "event_rate_train": float(e_tr.mean()),
            "event_rate_val": float(e_va.mean()),
            "tau": float(tau),
            "uno_base": float(uno_base),
            "best_uno": float(best_uno),
            "final_uno": float(final_uno),
            "delta": float(delta),
            "best_ep": int(best_ep),
            "status": status,
        }
        fold_records.append(rec)

        if args.save_jsonl:
            try:
                with open(args.save_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                print(f"[Warn] Failed to append JSONL: {e}")

    # --- Summary
    if fold_records:
        finals = np.array([r["final_uno"] for r in fold_records], dtype=float)
        bases = np.array([r["uno_base"] for r in fold_records], dtype=float)
        deltas = np.array([r["delta"] for r in fold_records], dtype=float)
        improved_cnt = int(np.sum(deltas > 1e-3))
        total_cnt = int(len(fold_records))

        print("\n" + "=" * 80)
        print("SUMMARY (6-dim fusion)")
        print("=" * 80)
        print(f"fusion_mode={args.fusion_mode} | base_seed={args.seed} | stride={args.fold_seed_stride}")
        print(f"Final Uno per fold: {[f'{x:.4f}' for x in finals.tolist()]}")
        print(f"Mean Final Uno: {finals.mean():.4f} ± {finals.std():.4f}")
        print(f"Mean Baseline Uno: {bases.mean():.4f} ± {bases.std():.4f}")
        print(f"Mean Δ (Final - Baseline): {deltas.mean():+.4f} ± {deltas.std():.4f}")
        print(f"Improved folds: {improved_cnt}/{total_cnt}  | Fallback/flat folds: {total_cnt - improved_cnt}/{total_cnt}")
        print("=" * 80 + "\n")
    else:
        print("[Error] No folds completed. Check artifact paths / clinical CSV columns.")


if __name__ == "__main__":
    main()
