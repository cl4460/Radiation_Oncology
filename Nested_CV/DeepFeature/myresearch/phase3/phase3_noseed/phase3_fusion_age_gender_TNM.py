#!/usr/bin/env python
# -*- coding: utf-8 -*-
# phase3_fusion_age_gender_TNM.py
# Image embedding (frozen) + Clinical residual (Age + Gender + T + N + M)
# M stage 使用二值化编码 (M0→0, M≥1→1)

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.loss import NLLLogistiHazardLoss
from sksurv.metrics import concordance_index_ipcw

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def load_clinical_data(csv_path: str) -> pd.DataFrame:
    """
    与 baseline 完全一致的 clinical loader:
    - patient_id, time, event, age, gender, t_stage, n_stage, m_stage
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

    df = df.rename(columns=col_map)

    required = ["patient_id", "time", "event", "age", "gender"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in clinical CSV: {missing}")

    df["patient_id"] = df["patient_id"].astype(str)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce").astype(int)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    if "t_stage" in df.columns:
        df["t_stage"] = pd.to_numeric(df["t_stage"], errors="coerce")
        print(f"[Clinical] T stage valid: {df['t_stage'].notna().sum()} / {len(df)}")
    else:
        print("[Clinical] T stage column not found, filled as NaN")
        df["t_stage"] = np.nan

    if "n_stage" in df.columns:
        df["n_stage"] = pd.to_numeric(df["n_stage"], errors="coerce")
        print(f"[Clinical] N stage valid: {df['n_stage'].notna().sum()} / {len(df)}")
    else:
        print("[Clinical] N stage column not found, filled as NaN")
        df["n_stage"] = np.nan

    if "m_stage" in df.columns:
        df["m_stage"] = pd.to_numeric(df["m_stage"], errors="coerce")
        print(f"[Clinical] M stage valid: {df['m_stage'].notna().sum()} / {len(df)}")
        m_dist = df["m_stage"].value_counts().to_dict()
        print(f"[Clinical] M stage distribution: {m_dist}")
    else:
        print("[Clinical] M stage column not found, filled as NaN")
        df["m_stage"] = np.nan

    print(f"[Clinical] Total patients: {len(df)}")
    print(f"[Clinical] Age missing: {df['age'].isna().sum()}")

    return df.set_index("patient_id")


def build_clinical_features(
    df_indexed: pd.DataFrame,
    t_pids: List[str],
    v_pids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    5 维特征: age + gender + T + N + M
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
    age_tr = (tr["age"].values - age_mu) / age_std
    age_va = (va["age"].values - age_mu) / age_std
    feats_tr.append(age_tr)
    feats_va.append(age_va)

    # Gender
    g_tr = np.array([_map_gender_to_float(x) for x in tr["gender"]], dtype=float)
    g_va = np.array([_map_gender_to_float(x) for x in va["gender"]], dtype=float)
    g_valid = g_tr[~np.isnan(g_tr)]
    g_mode = pd.Series(g_valid).mode()[0] if len(g_valid) > 0 else 0.5
    g_tr = np.nan_to_num(g_tr, nan=g_mode)
    g_va = np.nan_to_num(g_va, nan=g_mode)
    feats_tr.append(g_tr)
    feats_va.append(g_va)

    # T
    t_valid_tr = tr["t_stage"].dropna()
    t_mode = t_valid_tr.mode()[0] if len(t_valid_tr) > 0 else 2.0
    tr["t_stage"] = tr["t_stage"].fillna(t_mode)
    va["t_stage"] = va["t_stage"].fillna(t_mode)

    t_mu = tr["t_stage"].mean()
    t_std = tr["t_stage"].std() + 1e-8
    t_tr = (tr["t_stage"].values - t_mu) / t_std
    t_va = (va["t_stage"].values - t_mu) / t_std
    feats_tr.append(t_tr)
    feats_va.append(t_va)

    # N
    n_valid_tr = tr["n_stage"].dropna()
    n_mode = n_valid_tr.mode()[0] if len(n_valid_tr) > 0 else 1.0
    tr["n_stage"] = tr["n_stage"].fillna(n_mode)
    va["n_stage"] = va["n_stage"].fillna(n_mode)

    n_mu = tr["n_stage"].mean()
    n_std = tr["n_stage"].std() + 1e-8
    n_tr = (tr["n_stage"].values - n_mu) / n_std
    n_va = (va["n_stage"].values - n_mu) / n_std
    feats_tr.append(n_tr)
    feats_va.append(n_va)

    # M (二值化: M0→0, M≥1→1)
    m_tr_raw = tr["m_stage"].fillna(0).values
    m_va_raw = va["m_stage"].fillna(0).values

    m_tr = (m_tr_raw >= 1).astype(np.float32)
    m_va = (m_va_raw >= 1).astype(np.float32)

    feats_tr.append(m_tr)
    feats_va.append(m_va)

    print(
        f"[Features] M stage binarized: train M1+={m_tr.sum():.0f}/{len(m_tr)}, "
        f"val M1+={m_va.sum():.0f}/{len(m_va)}"
    )

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


def _load_image_head_from_ckpt(
    ckpt_path: Path, dropout: float = 0.35
) -> Tuple[ImageHead, np.ndarray, int]:
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


# ================== Fusion model & Dataset ==================
class ClinicalNet(nn.Module):
    """
    Clinical residual head: in_dim (=5) -> 32 -> n_bins
    最后一层 zero-init，使初始预测 ≈ 纯 image-only。
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
        # Zero-init 最后一层，做 residual
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
    args = parser.parse_args()

    cfg.LR = args.lr
    cfg.WD = args.weight_decay
    cfg.EPOCHS = args.epochs
    cfg.CLIN_DROPOUT = args.dropout_clin

    exp_dir = Path(args.exp_dir)
    clin_df = load_clinical_data(args.clinical_csv)
    valid_pids = set(clin_df.index)

    fold_results = []

    print("\n" + "=" * 60)
    print("ADDITIVE FUSION: Age + Gender + T + N + M (5-dim)")
    print("Note: M stage is binarized (M0→0, M≥1→1)")
    print(f"LR={cfg.LR:.2e}, WD={cfg.WD:.1e}, dropout={cfg.CLIN_DROPOUT:.2f}")
    print("=" * 60)

    for fold in range(5):
        print(f"\n=== Fold {fold} ===")
        try:
            t_emb, v_emb, t_pids, v_pids, img_head, cuts, n_bins = load_phase3_artifacts(
                exp_dir, fold, dropout_head=0.35
            )
        except Exception as e:
            print(f"[Fold {fold}] Skip: {e}")
            continue

        # 过滤到 clinical 中存在的 pid
        tr_mask = [p in valid_pids for p in t_pids]
        va_mask = [p in valid_pids for p in v_pids]
        t_emb_f = t_emb[tr_mask]
        v_emb_f = v_emb[va_mask]
        t_pids_f = [p for p, m in zip(t_pids, tr_mask) if m]
        v_pids_f = [p for p, m in zip(v_pids, va_mask) if m]

        # Clinical 特征
        X_tr, X_va, t_tr, e_tr, t_va, e_va = build_clinical_features(
            clin_df, t_pids_f, v_pids_f
        )

        # Labels（保持与 Phase3 一致）
        lab = LabTransDiscreteTime(cuts=cuts[1:-1])
        y_tr_idx, y_tr_evt = lab.transform(t_tr, e_tr)
        y_va_idx, y_va_evt = lab.transform(t_va, e_va)

        bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
        y_tr_struct = _to_struct(t_tr, e_tr)
        y_va_struct = _to_struct(t_va, e_va)
        tau = np.quantile(t_tr[e_tr == 1], 0.9)

        # Image-only baseline
        img_head.eval()
        with torch.no_grad():
            logits_base = img_head(torch.from_numpy(v_emb_f).float().to(DEVICE))
            uno_base = _uno_from_logits(
                logits_base, bin_mids, y_tr_struct, y_va_struct, tau
            )
        print(f"[Fold {fold}] Image-only baseline: {uno_base:.4f}")

        # Fusion model
        in_dim = X_tr.shape[1]  # 应为 5
        clin_net = ClinicalNet(
            in_dim=in_dim, n_bins=n_bins, dropout=cfg.CLIN_DROPOUT
        ).to(DEVICE)
        model = AdditiveFusion(img_head, clin_net).to(DEVICE)

        dl_tr = DataLoader(
            FusionDS(t_emb_f, X_tr, y_tr_idx, y_tr_evt),
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
        )
        dl_va = DataLoader(
            FusionDS(v_emb_f, X_va, y_va_idx, y_va_evt),
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
        )

        optimizer = torch.optim.Adam(
            clin_net.parameters(), lr=cfg.LR, weight_decay=cfg.WD
        )
        loss_fn = NLLLogistiHazardLoss()

        best_uno = uno_base
        best_ep = 0
        no_improve = 0

        for ep in range(1, cfg.EPOCHS + 1):
            # Train
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

            avg_loss = total_loss / len(X_tr)

            # Eval
            model.eval()
            with torch.no_grad():
                logits_list = []
                for emb_b, clin_b, _, _ in dl_va:
                    logits_list.append(model(emb_b.to(DEVICE), clin_b.to(DEVICE)))
                logits_va = torch.cat(logits_list, dim=0)
                uno_curr = _uno_from_logits(
                    logits_va, bin_mids, y_tr_struct, y_va_struct, tau
                )

            print(f"[Fold {fold}] Ep {ep:03d} | loss={avg_loss:.4f} | Uno={uno_curr:.4f}")

            if uno_curr > best_uno + 1e-4:
                best_uno = uno_curr
                best_ep = ep
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= cfg.PATIENCE:
                print(f"[Fold {fold}] Early stopping at ep {ep}")
                break

        final_uno = max(best_uno, uno_base)
        status = "IMPROVED" if best_uno > uno_base + 1e-3 else "FALLBACK/FLAT"
        delta = final_uno - uno_base
        print(
            f"[Fold {fold}] Final: {final_uno:.4f} ({status}), "
            f"Δ={delta:+.4f}, best_ep={best_ep}"
        )
        fold_results.append(final_uno)

    if fold_results:
        print("\n" + "=" * 60)
        print("SUMMARY: Age + Gender + T + N + M")
        print("=" * 60)
        for i, u in enumerate(fold_results):
            print(f"Fold {i}: Uno={u:.4f}")
        print("-" * 60)
        print(f"Mean Uno: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")


if __name__ == "__main__":
    main()
