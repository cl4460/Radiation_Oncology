#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clinical-only NSCLC-Radiomics baseline (age + gender + Overall.Stage).

- 使用 Phase3 的 patient split (train_pids.csv / val_pids.csv)
- 使用 Phase3 best.pt 中的 cuts / n_time_bins
- 只用 3 个 clinical 特征做 Logistic-Hazard 模型
- 评估 Uno C-index (concordance_index_ipcw)
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.loss import NLLLogistiHazardLoss
from sksurv.metrics import concordance_index_ipcw

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. Clinical 预处理：StageNumeric + 缺失值填补
# ============================================================

def _clean_stage_text(s: str | float) -> str | None:
    """把各种写法归一到 I / II / IIIa / IIIb，其他返回 None。"""
    if pd.isna(s):
        return None
    s = str(s).strip().upper()
    s = s.replace("STAGE", "").replace(" ", "")

    if s in ["I", "IA", "IB", "1"]:
        return "I"
    if s in ["II", "IIA", "IIB", "2"]:
        return "II"
    if s in ["IIIA", "3A"]:
        return "IIIa"
    if s in ["IIIB", "3B"]:
        return "IIIb"
    return None


def load_clinical_with_stage_numeric(csv_path: str) -> pd.DataFrame:
    """
    读取 NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv
    并把 Overall.Stage → stage_numeric, 保留 age/gender/time/event。

    此函数只做基本清洗，不做缺失值填补（填补在每个 fold 内完成）。
    """
    raw = pd.read_csv(csv_path)
    print(f"[Clinical] 原始 CSV 患者数: {len(raw)}")

    df = raw.rename(columns={
        "PatientID": "patient_id",
        "Survival.time": "time",
        "deadstatus.event": "event",
        "Overall.Stage": "overall_stage",
        "gender": "gender",
        "age": "age",
    }).copy()

    # 归一化 stage 文本
    df["stage_clean"] = df["overall_stage"].apply(_clean_stage_text)

    print(f"[Clinical] Age 缺失: {df['age'].isna().sum()}, "
          f"Stage 缺失: {df['stage_clean'].isna().sum()}")

    # StageNumeric: I=1, II=2, IIIa=3, IIIb=4
    stage_map = {"I": 1.0, "II": 2.0, "IIIa": 3.0, "IIIb": 4.0}
    df["stage_numeric_raw"] = df["stage_clean"].map(stage_map)

    df["event"] = df["event"].astype(int)
    df["time"] = df["time"].astype(float)
    df["patient_id"] = df["patient_id"].astype(str)

    # 保留必要列；age / stage_numeric_raw 仍可能有 NaN，后续在 fold 内填补
    df = df[["patient_id", "time", "event", "age", "gender", "stage_numeric_raw"]]
    return df


# ============================================================
# 2. 使用 Phase3 的 best.pt 读取 cuts / n_bins
# ============================================================

def load_cuts_for_fold(exp_dir: str, fold: int) -> Tuple[np.ndarray, int]:
    """
    从 Phase3 的 best.pt 中读取 cuts 和 n_time_bins。
    这样 clinical-only 的离散时间网格和 image-only 完全一致。
    """
    fold_dir = Path(exp_dir) / f"fold_{fold}"
    ckpt_path = fold_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} 不存在")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "cuts" not in ckpt:
        raise KeyError(f"{ckpt_path} 中没有 'cuts' 字段")

    cuts = np.asarray(ckpt["cuts"], dtype=float)
    n_bins = int(ckpt.get("n_time_bins", len(cuts) - 1))
    print(f"[Fold {fold}] 载入 cuts, n_bins={n_bins}")
    return cuts, n_bins


def load_pids_for_fold(
    exp_dir: str,
    fold: int,
    clinical_df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    读取 Phase3 的 train/val pids，并过滤到 clinical_df 中存在的病例。
    """
    fold_dir = Path(exp_dir) / f"fold_{fold}"
    train_pids = pd.read_csv(fold_dir / "train_pids.csv", header=None)[0].astype(str).tolist()
    val_pids = pd.read_csv(fold_dir / "val_pids.csv", header=None)[0].astype(str).tolist()

    valid_pids = set(clinical_df["patient_id"].astype(str).tolist())

    tr_keep = [p for p in train_pids if p in valid_pids]
    va_keep = [p for p in val_pids if p in valid_pids]

    print(f"[Fold {fold}] train: {len(tr_keep)}, val: {len(va_keep)} (过滤到 clinical 中存在的病例)")
    return tr_keep, va_keep


def _to_struct(times: np.ndarray, events: np.ndarray):
    """构造 sksurv 需要的 structured array。"""
    return np.array(
        [(bool(e), float(t)) for t, e in zip(times, events)],
        dtype=[("event", bool), ("time", float)],
    )


def prepare_fold_data(
    clinical_df: pd.DataFrame,
    train_pids: List[str],
    val_pids: List[str],
    cuts: np.ndarray,
):
    """
    根据 train/val pid，从 clinical_df 中抽取：
    - age_z, gender_male, stage_z 特征（在每个 fold 内做缺失值填补 + 标准化）
    - logistic-hazard 的离散标签
    - Uno C-index 所需的 survival 结构
    """
    df_idx = clinical_df.set_index("patient_id")

    tr = df_idx.loc[train_pids].copy()
    va = df_idx.loc[val_pids].copy()

    # ===== 在每个 fold 内进行缺失值填补 =====
    # Age: 用 train 中的均值填补
    age_mean = tr["age"].mean(skipna=True)
    tr["age"] = tr["age"].fillna(age_mean)
    va["age"] = va["age"].fillna(age_mean)

    # StageNumeric: 用 train 中的众数填补
    stage_tr_series = tr["stage_numeric_raw"]
    stage_mode = stage_tr_series.mode(dropna=True)
    if len(stage_mode) == 0:
        raise RuntimeError("StageNumeric 在 train 中全部缺失，这是不正常的。")
    stage_mode = stage_mode.iloc[0]

    tr["stage_numeric_raw"] = tr["stage_numeric_raw"].fillna(stage_mode)
    va["stage_numeric_raw"] = va["stage_numeric_raw"].fillna(stage_mode)

    # ===== 提取 time/event =====
    times_tr = tr["time"].to_numpy(float)
    events_tr = tr["event"].to_numpy(int)
    times_va = va["time"].to_numpy(float)
    events_va = va["event"].to_numpy(int)

    # ===== Stage z-score =====
    stage_tr = tr["stage_numeric_raw"].to_numpy(float)
    stage_va = va["stage_numeric_raw"].to_numpy(float)
    mu_stage = stage_tr.mean()
    std_stage = stage_tr.std() + 1e-8
    stage_tr_z = (stage_tr - mu_stage) / std_stage
    stage_va_z = (stage_va - mu_stage) / std_stage

    # ===== Age z-score =====
    age_tr = tr["age"].to_numpy(float)
    age_va = va["age"].to_numpy(float)
    mu_age = age_tr.mean()
    std_age = age_tr.std() + 1e-8
    age_tr_z = (age_tr - mu_age) / std_age
    age_va_z = (age_va - mu_age) / std_age

    # ===== Gender: male=1, female=0 =====
    gender_tr = (tr["gender"].str.lower() == "male").astype(float).to_numpy()
    gender_va = (va["gender"].str.lower() == "male").astype(float).to_numpy()

    # ===== 组合特征矩阵 =====
    X_tr = np.stack([age_tr_z, gender_tr, stage_tr_z], axis=1).astype(np.float32)
    X_va = np.stack([age_va_z, gender_va, stage_va_z], axis=1).astype(np.float32)

    # ===== Logistic-hazard 离散标签 =====
    lab = LabTransDiscreteTime(cuts=cuts[1:-1])
    y_tr_idx, y_tr_evt = lab.fit_transform(times_tr, events_tr)
    y_va_idx, y_va_evt = lab.transform(times_va, events_va)

    # ===== Uno C-index 所需信息 =====
    y_tr_struct = _to_struct(times_tr, events_tr)
    y_va_struct = _to_struct(times_va, events_va)
    bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
    evt_times_tr = times_tr[events_tr == 1]
    tau = np.quantile(evt_times_tr, 0.9) if len(evt_times_tr) > 0 else times_tr.max()

    return (X_tr, X_va,
            y_tr_idx, y_tr_evt,
            y_va_idx, y_va_evt,
            y_tr_struct, y_va_struct,
            bin_mids, tau)


# ============================================================
# 3. Dataset + Clinical-only 网络 + Uno 计算
# ============================================================

class ClinicalDataset(Dataset):
    def __init__(self, X: np.ndarray, y_idx: np.ndarray, y_evt: np.ndarray):
        """
        X: [N, 3]  (age_z, gender_male, stage_z)
        """
        self.X = torch.from_numpy(X).float()
        self.y_idx = torch.from_numpy(y_idx).long()
        self.y_evt = torch.from_numpy(y_evt).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return {
            "clin": self.X[i],
            "label_idx": self.y_idx[i],
            "label_evt": self.y_evt[i],
        }


class ClinicalOnlyNet(nn.Module):
    """
    简单的 clinical-only MLP：
    3 -> 32 -> 32 -> n_bins
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


def _uno_from_logits(
    logits: torch.Tensor,
    bin_mids: np.ndarray,
    y_tr_struct,
    y_va_struct,
    tau: float,
) -> float:
    """从 logits 计算 Uno C-index。"""
    haz = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1.0e-7)
    log_surv = torch.cumsum(torch.log(1.0 - haz), dim=1)
    surv = torch.exp(log_surv)  # [N, n_bins]
    risk = -np.trapz(surv.cpu().numpy(), x=bin_mids, axis=1)
    uno = concordance_index_ipcw(y_tr_struct, y_va_struct, risk, tau=tau)[0]
    return float(uno)


def train_one_epoch(model, loader, loss_fn, optimizer) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["clin"].to(DEVICE)
        y_idx = batch["label_idx"].to(DEVICE)
        y_evt = batch["label_evt"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y_idx, y_evt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    loss_fn,
    bin_mids: np.ndarray,
    y_tr_struct,
    y_va_struct,
    tau: float,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_logits = []

    for batch in loader:
        x = batch["clin"].to(DEVICE)
        y_idx = batch["label_idx"].to(DEVICE)
        y_evt = batch["label_evt"].to(DEVICE)

        logits = model(x)
        loss = loss_fn(logits, y_idx, y_evt)

        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        all_logits.append(logits.cpu())

    total_loss /= max(n, 1)

    logits_full = torch.cat(all_logits, dim=0)
    uno = _uno_from_logits(logits_full, bin_mids, y_tr_struct, y_va_struct, tau)
    return total_loss, uno


# ============================================================
# 4. 主逻辑：5-fold CV clinical-only baseline
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True,
                        help="Phase3 image-only 实验目录，例如 phase3_outputs/lr_6.2e-4")
    parser.add_argument("--clinical_csv", type=str, required=True,
                        help="NSCLC-Radiomics-Lung1 临床表格 CSV 路径")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    clinical_df = load_clinical_with_stage_numeric(args.clinical_csv)

    fold_unos: List[float] = []

    for fold in range(5):
        print(f"\n=== Clinical-only | Fold {fold} ===")

        cuts, n_bins = load_cuts_for_fold(args.exp_dir, fold)
        train_pids, val_pids = load_pids_for_fold(args.exp_dir, fold, clinical_df)

        (
            X_tr, X_va,
            y_tr_idx, y_tr_evt,
            y_va_idx, y_va_evt,
            y_tr_struct, y_va_struct,
            bin_mids, tau,
        ) = prepare_fold_data(clinical_df, train_pids, val_pids, cuts)

        train_ds = ClinicalDataset(X_tr, y_tr_idx, y_tr_evt)
        val_ds = ClinicalDataset(X_va, y_va_idx, y_va_evt)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        model = ClinicalOnlyNet(in_dim=3, n_bins=n_bins, dropout=args.dropout).to(DEVICE)
        loss_fn = NLLLogistiHazardLoss().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_uno = -1.0
        best_epoch = -1
        epochs_no_improve = 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
            val_loss, val_uno = evaluate(
                model, val_loader, loss_fn, bin_mids, y_tr_struct, y_va_struct, tau
            )

            print(f"[Fold {fold}] Epoch {epoch:03d} | "
                  f"train {train_loss:.4f} | val {val_loss:.4f} | Uno {val_uno:.4f}")

            if val_uno > best_uno + 1e-4:
                best_uno = val_uno
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"[Fold {fold}] Early stopping at epoch {epoch}")
                    break

        print(f"[Fold {fold}] ✅ Clinical-only best Uno = {best_uno:.4f} (epoch={best_epoch})")
        fold_unos.append(best_uno)

    mean_uno = float(np.mean(fold_unos))
    std_uno = float(np.std(fold_unos, ddof=1)) if len(fold_unos) > 1 else 0.0

    print("\n========== CLINICAL-ONLY SUMMARY ==========")
    for i, u in enumerate(fold_unos):
        print(f"Fold {i}: Uno={u:.4f}")
    print("--------------------------------")
    print(f"Clinical-only Uno: {mean_uno:.4f} ± {std_uno:.4f}")


if __name__ == "__main__":
    main()
