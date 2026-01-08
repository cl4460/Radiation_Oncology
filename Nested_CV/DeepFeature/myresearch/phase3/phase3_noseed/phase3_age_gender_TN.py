#!/usr/bin/env python
# phase3_clinical_only_tn.py
# Clinical-only baseline: Age + Gender + T + N (4-dim),
# using the SAME clinical preprocessing as phase3_fusion_additive_tn.py

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


# ================== Clinical mapping utilities (COPY from fusion) ==================
def _map_gender_to_float(s):
    """Gender映射：male → 1, female → 0，其它/缺失 → NaN"""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    if s in ["m", "male", "1"]:
        return 1.0
    if s in ["f", "female", "0"]:
        return 0.0
    return np.nan


def _to_struct(times, events):
    """构造 sksurv 需要的 structured array"""
    return np.array(
        [(bool(e), float(t)) for t, e in zip(times, events)],
        dtype=[("event", bool), ("time", float)],
    )


def compute_uno(
    logits: torch.Tensor,
    bin_mids: np.ndarray,
    y_tr_struct,
    y_va_struct,
    tau: float,
) -> float:
    """从 logits 计算 Uno C-index（和 fusion 里 _uno_from_logits 完全一致）"""
    haz = torch.sigmoid(logits).clamp(1e-7, 1.0 - 1.0e-7)
    log_surv = torch.cumsum(torch.log(1.0 - haz), dim=1)
    surv = torch.exp(log_surv)  # [N, n_bins]
    risk = -np.trapz(surv.cpu().numpy(), x=bin_mids, axis=1)
    uno = concordance_index_ipcw(y_tr_struct, y_va_struct, risk, tau=tau)[0]
    return float(uno)


# ================== Clinical loading & feature building (COPY from fusion) ==================
def load_clinical_data(csv_path: str) -> pd.DataFrame:
    """
    读取 NSCLC 临床 CSV，并统一列名:
    - patient_id, time, event, age, gender, t_stage, n_stage
    - T/N 一律用 pd.to_numeric 解析 (自动处理 1 / 1.0 / "1" / "1.0" / "NA")
    """
    df = pd.read_csv(csv_path)

    # 1. 统一列名（大小写、点号都不敏感）
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
            col_map[col] = "t_stage"   # 注意：直接映射到最终名字
        elif "n.stage" in col_lower or "n_stage" in col_lower:
            col_map[col] = "n_stage"

    df = df.rename(columns=col_map)

    # 2. 检查必要列
    required = ["patient_id", "time", "event", "age", "gender"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in clinical CSV: {missing}")

    # 3. 基本类型转换
    df["patient_id"] = df["patient_id"].astype(str)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["event"] = pd.to_numeric(df["event"], errors="coerce").astype(int)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # 4. T / N stage 统一用 to_numeric 解析
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

    print(f"[Clinical] Total patients: {len(df)}")
    print(f"[Clinical] Age missing: {df['age'].isna().sum()}")

    return df.set_index("patient_id")



def build_clinical_features(
    df_indexed: pd.DataFrame,
    t_pids: List[str],
    v_pids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构造 4 维特征: age + gender + T + N
    - age: 均值填充 + z-score
    - T/N: 众数填充 + z-score
    - gender: M/F → 1/0, 众数填充（不标准化）
    """
    tr = df_indexed.loc[t_pids].copy()
    va = df_indexed.loc[v_pids].copy()

    feats_tr = []
    feats_va = []

    # ---- Age ----
    age_mean_imp = tr["age"].mean()
    tr["age"] = tr["age"].fillna(age_mean_imp)
    va["age"] = va["age"].fillna(age_mean_imp)

    mu_age = tr["age"].mean()
    std_age = tr["age"].std() + 1e-8
    age_tr = (tr["age"].values - mu_age) / std_age
    age_va = (va["age"].values - mu_age) / std_age
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

    mu_t = tr["t_stage"].mean()
    std_t = tr["t_stage"].std() + 1e-8
    t_tr = (tr["t_stage"].values - mu_t) / std_t
    t_va = (va["t_stage"].values - mu_t) / std_t
    feats_tr.append(t_tr)
    feats_va.append(t_va)

    # ---- N stage ----
    n_valid_tr = tr["n_stage"].dropna()
    n_mode = n_valid_tr.mode()[0] if len(n_valid_tr) > 0 else 1.0
    tr["n_stage"] = tr["n_stage"].fillna(n_mode)
    va["n_stage"] = va["n_stage"].fillna(n_mode)

    mu_n = tr["n_stage"].mean()
    std_n = tr["n_stage"].std() + 1e-8
    n_tr = (tr["n_stage"].values - mu_n) / std_n
    n_va = (va["n_stage"].values - mu_n) / std_n
    feats_tr.append(n_tr)
    feats_va.append(n_va)

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


# ================== Model & Dataset ==================
class ClinicalNet(nn.Module):
    """Clinical-only 网络：4 -> 32 -> 32 -> n_bins"""
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


class ClinicalDataset(Dataset):
    def __init__(self, X: np.ndarray, y_idx: np.ndarray, y_evt: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y_idx = torch.from_numpy(y_idx).long()
        self.y_evt = torch.from_numpy(y_evt).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.y_idx[i], self.y_evt[i]


# ================== Main ==================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir", type=str, required=True,
        help="Phase3 outputs dir (contains fold_0..4 with best.pt, train_pids.csv, val_pids.csv)",
    )
    parser.add_argument("--clinical_csv", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    clin_df = load_clinical_data(args.clinical_csv)
    valid_pids = set(clin_df.index)

    fold_unos: List[float] = []

    print("\n" + "=" * 60)
    print("CLINICAL-ONLY BASELINE: Age + Gender + T + N")
    print(f"LR={args.lr:.2e}, WD={args.weight_decay:.1e}, dropout={args.dropout:.2f}")
    print("=" * 60)

    for fold in range(5):
        print(f"\n=== Clinical-only | Fold {fold} ===")
        fold_dir = Path(args.exp_dir) / f"fold_{fold}"

        # 1) 载入 cuts / n_bins
        ckpt = torch.load(fold_dir / "best.pt", map_location="cpu", weights_only=False)
        cuts = np.asarray(ckpt["cuts"], dtype=float)
        n_bins = int(ckpt.get("n_time_bins", len(cuts) - 1))
        print(f"[Fold {fold}] n_bins = {n_bins}")

        # 2) 载入 train/val pids 并过滤到 clinical 存在的病例
        t_pids_all = pd.read_csv(fold_dir / "train_pids.csv", header=None)[0].astype(str).tolist()
        v_pids_all = pd.read_csv(fold_dir / "val_pids.csv", header=None)[0].astype(str).tolist()
        t_pids = [p for p in t_pids_all if p in valid_pids]
        v_pids = [p for p in v_pids_all if p in valid_pids]
        print(f"[Fold {fold}] Train: {len(t_pids)}, Val: {len(v_pids)}")

        # 3) 构造 clinical 特征
        (
            X_tr,
            X_va,
            t_tr,
            e_tr,
            t_va,
            e_va,
        ) = build_clinical_features(clin_df, t_pids, v_pids)

        # 4) 离散时间标签（注意：clinical-only 用 fit_transform）
        lab = LabTransDiscreteTime(cuts=cuts[1:-1])
        y_tr_idx, y_tr_evt = lab.fit_transform(t_tr, e_tr)
        y_va_idx, y_va_evt = lab.transform(t_va, e_va)

        bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
        y_tr_struct = _to_struct(t_tr, e_tr)
        y_va_struct = _to_struct(t_va, e_va)
        evt_times_tr = t_tr[e_tr == 1]
        tau = np.quantile(evt_times_tr, 0.9) if len(evt_times_tr) > 0 else t_tr.max()

        # 5) DataLoader & Model
        train_ds = ClinicalDataset(X_tr, y_tr_idx, y_tr_evt)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)

        model = ClinicalNet(in_dim=X_tr.shape[1], n_bins=n_bins, dropout=args.dropout).to(DEVICE)
        loss_fn = NLLLogistiHazardLoss().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_uno = -1.0
        best_epoch = -1
        no_improve = 0

        for epoch in range(1, args.epochs + 1):
            # ---- Train ----
            model.train()
            total_loss = 0.0
            n_samples = 0

            for x_b, yi_b, ye_b in train_loader:
                x_b = x_b.to(DEVICE)
                yi_b = yi_b.to(DEVICE)
                ye_b = ye_b.to(DEVICE)

                optimizer.zero_grad()
                logits = model(x_b)
                loss = loss_fn(logits, yi_b, ye_b)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item() * x_b.size(0)
                n_samples += x_b.size(0)

            avg_loss = total_loss / max(1, n_samples)

            # ---- Eval on val ----
            model.eval()
            with torch.no_grad():
                logits_va = model(torch.from_numpy(X_va).float().to(DEVICE))
                try:
                    uno_val = compute_uno(
                        logits_va, bin_mids, y_tr_struct, y_va_struct, tau
                    )
                except Exception as e:
                    print(f"[Fold {fold}] WARN: Uno computation failed: {e}")
                    uno_val = 0.0

            print(
                f"[Fold {fold}] Epoch {epoch:03d} | "
                f"train_loss={avg_loss:.4f} | val_Uno={uno_val:.4f}"
            )

            # ---- Early stopping on Uno ----
            if uno_val > best_uno + 1e-4:
                best_uno = uno_val
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"[Fold {fold}] Early stopping at epoch {epoch}")
                    break

        print(f"[Fold {fold}] ✅ Best clinical-only Uno = {best_uno:.4f} (epoch {best_epoch})")
        fold_unos.append(best_uno)

    # Summary
    if fold_unos:
        mean_uno = float(np.mean(fold_unos))
        std_uno = float(np.std(fold_unos, ddof=1)) if len(fold_unos) > 1 else 0.0

        print("\n" + "=" * 60)
        print("CLINICAL-ONLY SUMMARY: Age + Gender + T + N")
        print("=" * 60)
        for i, u in enumerate(fold_unos):
            print(f"Fold {i}: Uno={u:.4f}")
        print("--------------------------------")
        print(f"Clinical-only Uno: {mean_uno:.4f} ± {std_uno:.4f}")
    else:
        print("[WARN] No folds were successfully evaluated.")


if __name__ == "__main__":
    main()
