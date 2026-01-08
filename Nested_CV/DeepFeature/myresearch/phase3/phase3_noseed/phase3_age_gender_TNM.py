#!/usr/bin/env python
# -*- coding: utf-8 -*-
# phase3_clinical_age_gender_TNM.py
# Clinical-only baseline: Age + Gender + T + N + M (5-dim)
# M stage 使用二值化编码 (M0→0, M≥1→1)

import argparse
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


# ================== Clinical utils ==================
def load_clinical_data(csv_path: str) -> pd.DataFrame:
    """
    读取 NSCLC 临床 CSV，并统一列名:
    - patient_id, time, event, age, gender, t_stage, n_stage, m_stage
    - T/N/M 一律用 pd.to_numeric 解析
    """
    df = pd.read_csv(csv_path)

    # 1. 统一列名（大小写 / 前缀不敏感）
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

    # 2. 检查必要列
    required = ["patient_id", "time", "event", "age", "gender"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in clinical CSV: {missing}")

    # 3. 数据清洗与转换
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


def _map_gender_to_float(s) -> float:
    """Gender: M/male→1, F/female→0，其它视为缺失"""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    if s in ["m", "male", "1"]:
        return 1.0
    if s in ["f", "female", "0"]:
        return 0.0
    return np.nan


def build_clinical_features(
    df_indexed: pd.DataFrame,
    t_pids: List[str],
    v_pids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构造 5 维特征: age + gender + T + N + M

    - age: train 均值填充 + z-score
    - gender: M/F → 1/0, train 众数填充（不标准化）
    - T/N: train 众数填充 + z-score
    - M: 二值化 (M0→0, M≥1→1)，不做 z-score
    """
    tr = df_indexed.loc[t_pids].copy()
    va = df_indexed.loc[v_pids].copy()

    feats_tr = []
    feats_va = []

    # ---- Age ----
    age_mean_impute = tr["age"].mean()
    tr["age"] = tr["age"].fillna(age_mean_impute)
    va["age"] = va["age"].fillna(age_mean_impute)

    age_mu = tr["age"].mean()
    age_std = tr["age"].std() + 1e-8
    age_tr = (tr["age"].values - age_mu) / age_std
    age_va = (va["age"].values - age_mu) / age_std
    feats_tr.append(age_tr)
    feats_va.append(age_va)

    # ---- Gender ----
    g_tr = np.array([_map_gender_to_float(x) for x in tr["gender"]], dtype=float)
    g_va = np.array([_map_gender_to_float(x) for x in va["gender"]], dtype=float)
    g_valid = g_tr[~np.isnan(g_tr)]
    g_mode = pd.Series(g_valid).mode()[0] if len(g_valid) > 0 else 0.5
    g_tr = np.nan_to_num(g_tr, nan=g_mode)
    g_va = np.nan_to_num(g_va, nan=g_mode)
    feats_tr.append(g_tr)
    feats_va.append(g_va)

    # ---- T stage ----
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

    # ---- N stage ----
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

    # ---- M stage (二值化: M0→0, M≥1→1) ----
    # 数据分布极端不平衡，直接 z-score 会产生大异常值 → 改为 0/1
    m_tr_raw = tr["m_stage"].fillna(0).values  # 缺失视为 M0（保守）
    m_va_raw = va["m_stage"].fillna(0).values

    m_tr = (m_tr_raw >= 1).astype(np.float32)
    m_va = (m_va_raw >= 1).astype(np.float32)

    feats_tr.append(m_tr)
    feats_va.append(m_va)

    print(
        f"[Features] M stage binarized: train M1+={m_tr.sum():.0f}/{len(m_tr)}, "
        f"val M1+={m_va.sum():.0f}/{len(m_va)}"
    )

    # 组合特征: [age, gender, T, N, M]
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
    """构造 sksurv 需要的 structured array"""
    return np.array(
        [(bool(e), float(t)) for t, e in zip(times, events)],
        dtype=[("event", bool), ("time", float)],
    )


def _uno_from_logits(
    logits: torch.Tensor,
    bin_mids: np.ndarray,
    y_tr_struct,
    y_va_struct,
    tau: float,
) -> float:
    """从 logits 计算 Uno C-index"""
    haz = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1.0e-7)
    log_surv = torch.cumsum(torch.log(1.0 - haz), dim=1)
    surv = torch.exp(log_surv)  # [N, n_bins]
    risk = -np.trapz(surv.cpu().numpy(), x=bin_mids, axis=1)
    uno = concordance_index_ipcw(y_tr_struct, y_va_struct, risk, tau=tau)[0]
    return float(uno)


# ================== Dataset & Model ==================
class ClinicalDataset(Dataset):
    def __init__(self, X: np.ndarray, y_idx: np.ndarray, y_evt: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y_idx = torch.from_numpy(y_idx).long()
        self.y_evt = torch.from_numpy(y_evt).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.y_idx[i], self.y_evt[i]


class ClinicalNet(nn.Module):
    """
    Clinical-only MLP:
    in_dim (=5) -> 32 -> 32 -> n_bins
    """

    def __init__(self, in_dim: int, n_bins: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, n_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ================== Main ==================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Phase3 image-only 实验目录（用于读取 cuts/n_bins）",
    )
    parser.add_argument(
        "--clinical_csv",
        type=str,
        required=True,
        help="NSCLC-Radiomics-Lung1 临床表格 CSV 路径",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    clin_df = load_clinical_data(args.clinical_csv)
    valid_pids = set(clin_df.index)

    fold_unos: List[float] = []

    print("\n" + "=" * 60)
    print("CLINICAL-ONLY BASELINE: Age + Gender + T + N + M (5-dim)")
    print("Note: M stage is binarized (M0→0, M≥1→1)")
    print(f"LR={args.lr:.1e}, WD={args.weight_decay:.1e}, dropout={args.dropout}")
    print("=" * 60)

    for fold in range(5):
        print(f"\n=== Clinical-only | Fold {fold} ===")
        fold_dir = Path(args.exp_dir) / f"fold_{fold}"

        ckpt = torch.load(fold_dir / "best.pt", map_location="cpu", weights_only=False)
        cuts = np.asarray(ckpt["cuts"], dtype=float)
        n_bins = int(ckpt.get("n_time_bins", len(cuts) - 1))
        print(f"[Fold {fold}] n_bins = {n_bins}")

        t_pids = pd.read_csv(fold_dir / "train_pids.csv", header=None)[0].astype(str).tolist()
        v_pids = pd.read_csv(fold_dir / "val_pids.csv", header=None)[0].astype(str).tolist()

        t_pids = [p for p in t_pids if p in valid_pids]
        v_pids = [p for p in v_pids if p in valid_pids]
        print(f"[Fold {fold}] Train: {len(t_pids)}, Val: {len(v_pids)}")

        X_tr, X_va, t_tr, e_tr, t_va, e_va = build_clinical_features(
            clin_df, t_pids, v_pids
        )

        lab = LabTransDiscreteTime(cuts=cuts[1:-1])
        y_tr_idx, y_tr_evt = lab.fit_transform(t_tr, e_tr)
        y_va_idx, y_va_evt = lab.transform(t_va, e_va)

        bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
        y_tr_struct = _to_struct(t_tr, e_tr)
        y_va_struct = _to_struct(t_va, e_va)
        evt_times_tr = t_tr[e_tr == 1]
        tau = np.quantile(evt_times_tr, 0.9) if len(evt_times_tr) > 0 else t_tr.max()

        train_ds = ClinicalDataset(X_tr, y_tr_idx, y_tr_evt)
        val_ds = ClinicalDataset(X_va, y_va_idx, y_va_evt)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        model = ClinicalNet(in_dim=X_tr.shape[1], n_bins=n_bins, dropout=args.dropout).to(
            DEVICE
        )
        loss_fn = NLLLogistiHazardLoss().to(DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        best_uno = -1.0
        best_epoch = -1
        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            # Train
            model.train()
            total_loss = 0.0
            n_samples = 0
            for x, y_idx, y_evt in train_loader:
                x = x.to(DEVICE)
                y_idx = y_idx.to(DEVICE)
                y_evt = y_evt.to(DEVICE)

                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y_idx, y_evt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                n_samples += x.size(0)

            train_loss = total_loss / max(n_samples, 1)

            # Eval
            model.eval()
            all_logits = []
            with torch.no_grad():
                for x, y_idx, y_evt in val_loader:
                    x = x.to(DEVICE)
                    logits = model(x)
                    all_logits.append(logits.cpu())

            logits_full = torch.cat(all_logits, dim=0)
            val_uno = _uno_from_logits(
                logits_full, bin_mids, y_tr_struct, y_va_struct, tau
            )

            print(
                f"[Fold {fold}] Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | val_Uno={val_uno:.4f}"
            )

            if val_uno > best_uno + 1e-4:
                best_uno = val_uno
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"[Fold {fold}] Early stopping at epoch {epoch}")
                    break

        print(
            f"[Fold {fold}] ✅ Best clinical-only Uno = {best_uno:.4f} "
            f"(epoch {best_epoch})"
        )
        fold_unos.append(best_uno)

    mean_uno = float(np.mean(fold_unos))
    std_uno = float(np.std(fold_unos, ddof=1)) if len(fold_unos) > 1 else 0.0

    print("\n" + "=" * 60)
    print("CLINICAL-ONLY SUMMARY: Age + Gender + T + N + M")
    print("=" * 60)
    for i, u in enumerate(fold_unos):
        print(f"Fold {i}: Uno={u:.4f}")
    print("--------------------------------")
    print(f"Clinical-only Uno: {mean_uno:.4f} ± {std_uno:.4f}")


if __name__ == "__main__":
    main()
