#!/usr/bin/env python
# -*- coding: utf-8 -*-
# phase3_fusion_age_gender_TNM_stage_histology.py
# Frozen image head + Clinical residual
# Clinical = Age + Gender + T + N + M + OverallStage + Histology(4 one-hot + missing) => 11-dim

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.loss import NLLLogistiHazardLoss
from sksurv.metrics import concordance_index_ipcw

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================== Seed Setting ==================
def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================== Config ==================
@dataclass
class CFG:
    BATCH_SIZE: int = 64
    EPOCHS: int = 100
    LR: float = 7e-5
    WD: float = 5e-4
    CLIN_DROPOUT: float = 0.3
    PATIENCE: int = 25

cfg = CFG()


# ================== Clinical utils ==================
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


def _normalize_histology(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ["", "na", "nan", "none", "null"]:
        return None
    if "adeno" in s:
        return "adeno"
    if "squamous" in s:
        return "squamous"
    if "large" in s:
        return "large"
    if "nos" in s:
        return "nos"
    return None


def load_clinical_data(csv_path: str) -> pd.DataFrame:
    """
    Keep your old loader style. Only add mapping for overall_stage + histology.
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
        elif "histology" in col_lower:
            col_map[col] = "histology"

    df = df.rename(columns=col_map)

    required = ["patient_id", "time", "event", "age", "gender"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in clinical CSV: {missing}")

    # ensure optional cols exist
    for opt in ["t_stage", "n_stage", "m_stage", "overall_stage", "histology"]:
        if opt not in df.columns:
            df[opt] = np.nan

    df["patient_id"] = df["patient_id"].astype(str)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce").astype(int)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # keep same numeric parsing behavior
    df["t_stage"] = pd.to_numeric(df["t_stage"], errors="coerce")
    df["n_stage"] = pd.to_numeric(df["n_stage"], errors="coerce")
    df["m_stage"] = pd.to_numeric(df["m_stage"], errors="coerce")
    df["overall_stage"] = df["overall_stage"].apply(_map_overall_stage_to_group)

    return df.set_index("patient_id")


def build_clinical_features(
    df_indexed: pd.DataFrame,
    t_pids: List[str],
    v_pids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    11-dim:
    age_z, gender, t_z, n_z, m_bin, overall_stage_z,
    hist_adeno, hist_squamous, hist_large, hist_nos, hist_missing
    """
    tr = df_indexed.loc[t_pids].copy()
    va = df_indexed.loc[v_pids].copy()

    feats_tr = []
    feats_va = []

    # Age
    age_mean_impute = tr["age"].mean()
    tr["age"] = tr["age"].fillna(age_mean_impute)
    va["age"] = va["age"].fillna(age_mean_impute)

    age_mu = tr["age"].mean()
    age_std = tr["age"].std() + 1e-8
    feats_tr.append(((tr["age"].values - age_mu) / age_std).astype(np.float32))
    feats_va.append(((va["age"].values - age_mu) / age_std).astype(np.float32))

    # Gender
    g_tr = np.array([_map_gender_to_float(x) for x in tr["gender"]], dtype=float)
    g_va = np.array([_map_gender_to_float(x) for x in va["gender"]], dtype=float)
    g_valid = g_tr[~np.isnan(g_tr)]
    g_mode = pd.Series(g_valid).mode()[0] if len(g_valid) > 0 else 0.5
    g_tr = np.nan_to_num(g_tr, nan=g_mode).astype(np.float32)
    g_va = np.nan_to_num(g_va, nan=g_mode).astype(np.float32)
    feats_tr.append(g_tr)
    feats_va.append(g_va)

    # T
    t_valid_tr = tr["t_stage"].dropna()
    t_mode = t_valid_tr.mode()[0] if len(t_valid_tr) > 0 else 2.0
    tr["t_stage"] = tr["t_stage"].fillna(t_mode)
    va["t_stage"] = va["t_stage"].fillna(t_mode)

    t_mu = tr["t_stage"].mean()
    t_std = tr["t_stage"].std() + 1e-8
    feats_tr.append(((tr["t_stage"].values - t_mu) / t_std).astype(np.float32))
    feats_va.append(((va["t_stage"].values - t_mu) / t_std).astype(np.float32))

    # N
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

    # Overall stage (z-score)
    st_valid_tr = tr["overall_stage"].dropna()
    st_mode = st_valid_tr.mode()[0] if len(st_valid_tr) > 0 else 2.0
    tr["overall_stage"] = tr["overall_stage"].fillna(st_mode)
    va["overall_stage"] = va["overall_stage"].fillna(st_mode)

    st_mu = tr["overall_stage"].mean()
    st_std = tr["overall_stage"].std() + 1e-8
    feats_tr.append(((tr["overall_stage"].values - st_mu) / st_std).astype(np.float32))
    feats_va.append(((va["overall_stage"].values - st_mu) / st_std).astype(np.float32))

    # Histology (4 one-hot + missing)
    h_tr_norm = [_normalize_histology(x) for x in tr["histology"]]
    h_va_norm = [_normalize_histology(x) for x in va["histology"]]

    h_tr_missing = np.array([1.0 if x is None else 0.0 for x in h_tr_norm], dtype=np.float32)
    h_va_missing = np.array([1.0 if x is None else 0.0 for x in h_va_norm], dtype=np.float32)

    # ===== Histology Statistics (Step 2) =====
    n_missing = int(h_tr_missing.sum())
    missing_rate = n_missing / len(h_tr_missing)
    print(f"[Histology] Missing rate: {n_missing}/{len(h_tr_missing)} = {missing_rate:.2%}")
    
    # Count each category in train fold
    h_tr_vals = [x for x in h_tr_norm if x is not None]
    if len(h_tr_vals) > 0:
        c = Counter(h_tr_vals)
        print(f"[Histology] Category counts (train): {dict(c)}")
        h_mode = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    else:
        print("[Histology] Category counts (train): ALL MISSING")
        h_mode = "squamous"
    
    # Correlation between missing flag and event
    e_tr = tr["event"].values
    event_rate_missing = e_tr[h_tr_missing == 1].mean() if (h_tr_missing == 1).sum() > 0 else 0
    event_rate_nonmissing = e_tr[h_tr_missing == 0].mean() if (h_tr_missing == 0).sum() > 0 else 0
    print(f"[Histology] Event rate: missing={event_rate_missing:.2%} vs non-missing={event_rate_nonmissing:.2%}")
    print()

    h_tr_fill = [x if x is not None else h_mode for x in h_tr_norm]
    h_va_fill = [x if x is not None else h_mode for x in h_va_norm]

    cats = ["adeno", "squamous", "large", "nos"]
    for cat in cats:
        feats_tr.append(np.array([1.0 if x == cat else 0.0 for x in h_tr_fill], dtype=np.float32))
        feats_va.append(np.array([1.0 if x == cat else 0.0 for x in h_va_fill], dtype=np.float32))

    feats_tr.append(h_tr_missing)
    feats_va.append(h_va_missing)

    X_tr = np.stack(feats_tr, axis=1).astype(np.float32)
    X_va = np.stack(feats_va, axis=1).astype(np.float32)

    return (
        X_tr,
        X_va,
        tr["time"].values.astype(float),
        tr["event"].values.astype(int),
        va["time"].values.astype(float),
        va["event"].values.astype(int),
    )


def _to_struct(times, events):
    return np.array(
        [(bool(e), float(t)) for t, e in zip(times, events)],
        dtype=[("event", bool), ("time", float)],
    )


def _uno_from_logits(logits, bin_mids, y_tr, y_va, tau):
    haz = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    log_surv = torch.cumsum(torch.log(1.0 - haz), dim=1)
    surv = torch.exp(log_surv)
    risk = -np.trapz(surv.cpu().numpy(), x=bin_mids, axis=1)
    return concordance_index_ipcw(y_tr, y_va, risk, tau=tau)[0]


# ================== Image artifacts ==================
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


def _load_image_head_from_ckpt(ckpt_path: Path, dropout: float = 0.35):
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


# ================== Fusion model & Dataset ==================
class ClinicalNet(nn.Module):
    """
    Clinical residual head: in_dim -> 32 -> n_bins
    zero-init last layer
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
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class AdditiveFusion(nn.Module):
    def __init__(self, img_head: nn.Module, clin_net: nn.Module):
        super().__init__()
        self.img_head = img_head
        self.clin_net = clin_net

        self.img_head.eval()
        for p in self.img_head.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.img_head.eval()
        for p in self.img_head.parameters():
            p.requires_grad = False
        return self

    def forward(self, emb, clin):
        with torch.no_grad():
            logits_img = self.img_head(emb)
        logits_clin = self.clin_net(clin)
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


# ================== Main ==================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--clinical_csv", type=str, required=True)
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--weight_decay", type=float, default=cfg.WD)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--dropout_clin", type=float, default=cfg.CLIN_DROPOUT)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"[Seed] Set random seed to {args.seed}")

    cfg.LR = args.lr
    cfg.WD = args.weight_decay
    cfg.EPOCHS = args.epochs
    cfg.CLIN_DROPOUT = args.dropout_clin

    exp_dir = Path(args.exp_dir)
    clin_df = load_clinical_data(args.clinical_csv)
    valid_pids = set(clin_df.index)

    fold_results = []

    for fold in range(5):
        print(f"\n=== Fold {fold} ===")
        t_emb, v_emb, t_pids, v_pids, img_head, cuts, n_bins = load_phase3_artifacts(exp_dir, fold, dropout_head=0.35)

        # filter to clinical pids
        tr_mask = [p in valid_pids for p in t_pids]
        va_mask = [p in valid_pids for p in v_pids]
        t_emb_f = t_emb[tr_mask]
        v_emb_f = v_emb[va_mask]
        t_pids_f = [p for p, m in zip(t_pids, tr_mask) if m]
        v_pids_f = [p for p, m in zip(v_pids, va_mask) if m]

        X_tr, X_va, t_tr, e_tr, t_va, e_va = build_clinical_features(clin_df, t_pids_f, v_pids_f)

        # keep same label behavior as your old fusion code
        lab = LabTransDiscreteTime(cuts=cuts[1:-1])
        y_tr_idx, y_tr_evt = lab.transform(t_tr, e_tr)
        y_va_idx, y_va_evt = lab.transform(t_va, e_va)

        bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
        y_tr_struct = _to_struct(t_tr, e_tr)
        y_va_struct = _to_struct(t_va, e_va)
        tau = np.quantile(t_tr[e_tr == 1], 0.9) if np.any(e_tr == 1) else np.max(t_tr)

        # image-only baseline
        img_head.eval()
        with torch.no_grad():
            logits_base = img_head(torch.from_numpy(v_emb_f).float().to(DEVICE))
            uno_base = _uno_from_logits(logits_base, bin_mids, y_tr_struct, y_va_struct, tau)
        print(f"[Fold {fold}] Image-only baseline: {uno_base:.4f}")

        clin_net = ClinicalNet(in_dim=X_tr.shape[1], n_bins=n_bins, dropout=cfg.CLIN_DROPOUT).to(DEVICE)
        model = AdditiveFusion(img_head, clin_net).to(DEVICE)

        dl_tr = DataLoader(FusionDS(t_emb_f, X_tr, y_tr_idx, y_tr_evt),
                           batch_size=cfg.BATCH_SIZE, shuffle=True)
        dl_va = DataLoader(FusionDS(v_emb_f, X_va, y_va_idx, y_va_evt),
                           batch_size=cfg.BATCH_SIZE, shuffle=False)

        optimizer = torch.optim.Adam(clin_net.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
        loss_fn = NLLLogistiHazardLoss()

        best_uno = uno_base
        best_ep = 0
        no_improve = 0

        for ep in range(1, cfg.EPOCHS + 1):
            model.train()
            total_loss = 0.0
            for emb_b, clin_b, yi_b, ye_b in dl_tr:
                optimizer.zero_grad()
                logits = model(emb_b.to(DEVICE), clin_b.to(DEVICE))
                loss = loss_fn(logits, yi_b.to(DEVICE), ye_b.to(DEVICE))
                loss.backward()
                nn.utils.clip_grad_norm_(clin_net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * emb_b.size(0)

            model.eval()
            with torch.no_grad():
                logits_list = []
                for emb_b, clin_b, _, _ in dl_va:
                    logits_list.append(model(emb_b.to(DEVICE), clin_b.to(DEVICE)))
                logits_va = torch.cat(logits_list, dim=0)
                uno_curr = _uno_from_logits(logits_va, bin_mids, y_tr_struct, y_va_struct, tau)

            if uno_curr > best_uno + 1e-4:
                best_uno = float(uno_curr)
                best_ep = ep
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.PATIENCE:
                    break

        # baseline fallback (keep same as your old code)
        final_uno = max(best_uno, uno_base)
        print(f"[Fold {fold}] Final Uno={final_uno:.4f} | Δ={final_uno-uno_base:+.4f} | best_ep={best_ep}")
        fold_results.append(final_uno)

    print("\n" + "=" * 60)
    print("SUMMARY (Fusion, aligned except histology)")
    print("=" * 60)
    print("Fold unos:", [f"{u:.4f}" for u in fold_results])
    print(f"Mean Uno: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")


if __name__ == "__main__":
    main()
