#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Clinical-only baseline:
# Age + Gender + T + N + M + OverallStage + Histology(4 one-hot + missing) => 11-dim

import argparse
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
    """
    overall stage -> ordinal group 1~4:
    I*->1, II*->2, III*->3, IV*->4, else NaN
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


def _normalize_histology(x):
    """
    Return one of: adeno/squamous/large/nos, or None if missing/unknown.
    """
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

    # numeric parsing (keep identical behavior to your old code)
    df["t_stage"] = pd.to_numeric(df["t_stage"], errors="coerce")
    df["n_stage"] = pd.to_numeric(df["n_stage"], errors="coerce")
    df["m_stage"] = pd.to_numeric(df["m_stage"], errors="coerce")

    df["overall_stage"] = df["overall_stage"].apply(_map_overall_stage_to_group)

    print(f"[Clinical] Total patients: {len(df)}")
    print(f"[Clinical] Histology counts (raw):\n{df['histology'].value_counts(dropna=False)}")

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

    # ---- Age ----
    age_mean_impute = tr["age"].mean()
    tr["age"] = tr["age"].fillna(age_mean_impute)
    va["age"] = va["age"].fillna(age_mean_impute)

    mu = tr["age"].mean()
    sd = tr["age"].std() + 1e-8
    feats_tr.append(((tr["age"].values - mu) / sd).astype(np.float32))
    feats_va.append(((va["age"].values - mu) / sd).astype(np.float32))

    # ---- Gender ----
    g_tr = np.array([_map_gender_to_float(x) for x in tr["gender"]], dtype=float)
    g_va = np.array([_map_gender_to_float(x) for x in va["gender"]], dtype=float)

    g_valid = g_tr[~np.isnan(g_tr)]
    g_mode = pd.Series(g_valid).mode()[0] if len(g_valid) > 0 else 0.5
    g_tr = np.nan_to_num(g_tr, nan=g_mode).astype(np.float32)
    g_va = np.nan_to_num(g_va, nan=g_mode).astype(np.float32)

    feats_tr.append(g_tr)
    feats_va.append(g_va)

    # ---- T stage ----
    t_valid_tr = tr["t_stage"].dropna()
    t_mode = t_valid_tr.mode()[0] if len(t_valid_tr) > 0 else 2.0
    tr["t_stage"] = tr["t_stage"].fillna(t_mode)
    va["t_stage"] = va["t_stage"].fillna(t_mode)

    mu = tr["t_stage"].mean()
    sd = tr["t_stage"].std() + 1e-8
    feats_tr.append(((tr["t_stage"].values - mu) / sd).astype(np.float32))
    feats_va.append(((va["t_stage"].values - mu) / sd).astype(np.float32))

    # ---- N stage ----
    n_valid_tr = tr["n_stage"].dropna()
    n_mode = n_valid_tr.mode()[0] if len(n_valid_tr) > 0 else 1.0
    tr["n_stage"] = tr["n_stage"].fillna(n_mode)
    va["n_stage"] = va["n_stage"].fillna(n_mode)

    mu = tr["n_stage"].mean()
    sd = tr["n_stage"].std() + 1e-8
    feats_tr.append(((tr["n_stage"].values - mu) / sd).astype(np.float32))
    feats_va.append(((va["n_stage"].values - mu) / sd).astype(np.float32))

    # ---- M stage (binarized) ----
    m_tr_raw = tr["m_stage"].fillna(0).values
    m_va_raw = va["m_stage"].fillna(0).values
    m_tr = (m_tr_raw >= 1).astype(np.float32)
    m_va = (m_va_raw >= 1).astype(np.float32)
    feats_tr.append(m_tr)
    feats_va.append(m_va)

    # ---- Overall stage (ordinal + z-score) ----
    st_valid_tr = tr["overall_stage"].dropna()
    st_mode = st_valid_tr.mode()[0] if len(st_valid_tr) > 0 else 2.0
    tr["overall_stage"] = tr["overall_stage"].fillna(st_mode)
    va["overall_stage"] = va["overall_stage"].fillna(st_mode)

    mu = tr["overall_stage"].mean()
    sd = tr["overall_stage"].std() + 1e-8
    feats_tr.append(((tr["overall_stage"].values - mu) / sd).astype(np.float32))
    feats_va.append(((va["overall_stage"].values - mu) / sd).astype(np.float32))

    # ---- Histology (4 one-hot + missing flag) ----
    h_tr_norm = [ _normalize_histology(x) for x in tr["histology"] ]
    h_va_norm = [ _normalize_histology(x) for x in va["histology"] ]

    h_tr_missing = np.array([1.0 if x is None else 0.0 for x in h_tr_norm], dtype=np.float32)
    h_va_missing = np.array([1.0 if x is None else 0.0 for x in h_va_norm], dtype=np.float32)

    h_tr_vals = [x for x in h_tr_norm if x is not None]
    if len(h_tr_vals) > 0:
        # deterministic mode
        c = Counter(h_tr_vals)
        h_mode = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    else:
        h_mode = "squamous"

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


# ================== Dataset & Model ==================

class ClinicalDataset(Dataset):
    def __init__(self, X, y_idx, y_evt):
        self.X = torch.from_numpy(X).float()
        self.y_idx = torch.from_numpy(y_idx).long()
        self.y_evt = torch.from_numpy(y_evt).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y_idx[i], self.y_evt[i]


class ClinicalNet(nn.Module):
    """
    Keep your previous clinical-only depth (2-layer MLP):
    in_dim -> 32 -> 32 -> n_bins
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

    def forward(self, x):
        return self.net(x)


# ================== Main ==================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--clinical_csv", type=str, required=True)
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

    fold_unos = []

    for fold in range(5):
        fold_dir = Path(args.exp_dir) / f"fold_{fold}"
        ckpt = torch.load(fold_dir / "best.pt", map_location="cpu", weights_only=False)
        cuts = np.asarray(ckpt["cuts"], dtype=float)
        n_bins = int(ckpt.get("n_time_bins", len(cuts) - 1))

        t_pids = pd.read_csv(fold_dir / "train_pids.csv", header=None)[0].astype(str).tolist()
        v_pids = pd.read_csv(fold_dir / "val_pids.csv", header=None)[0].astype(str).tolist()
        t_pids = [p for p in t_pids if p in valid_pids]
        v_pids = [p for p in v_pids if p in valid_pids]

        X_tr, X_va, t_tr, e_tr, t_va, e_va = build_clinical_features(clin_df, t_pids, v_pids)

        lab = LabTransDiscreteTime(cuts=cuts[1:-1])
        y_tr_idx, y_tr_evt = lab.fit_transform(t_tr, e_tr)
        y_va_idx, y_va_evt = lab.transform(t_va, e_va)

        bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
        y_tr_struct = _to_struct(t_tr, e_tr)
        y_va_struct = _to_struct(t_va, e_va)
        evt_times_tr = t_tr[e_tr == 1]
        tau = np.quantile(evt_times_tr, 0.9) if len(evt_times_tr) > 0 else t_tr.max()

        train_loader = DataLoader(ClinicalDataset(X_tr, y_tr_idx, y_tr_evt),
                                  batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(ClinicalDataset(X_va, y_va_idx, y_va_evt),
                                batch_size=args.batch_size, shuffle=False, num_workers=4)

        model = ClinicalNet(in_dim=X_tr.shape[1], n_bins=n_bins, dropout=args.dropout).to(DEVICE)
        loss_fn = NLLLogistiHazardLoss().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_uno = -1.0
        no_imp = 0

        for ep in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            n = 0
            for x, yi, ye in train_loader:
                x, yi, ye = x.to(DEVICE), yi.to(DEVICE), ye.to(DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, yi, ye)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item() * x.size(0)
                n += x.size(0)

            model.eval()
            with torch.no_grad():
                logits_list = []
                for x, _, _ in val_loader:
                    logits_list.append(model(x.to(DEVICE)).cpu())
                logits_va = torch.cat(logits_list, dim=0)
                uno = _uno_from_logits(logits_va, bin_mids, y_tr_struct, y_va_struct, tau)

            if uno > best_uno + 1e-4:
                best_uno = float(uno)
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= args.patience:
                    break

        fold_unos.append(best_uno)
        print(f"[Fold {fold}] Best clinical-only Uno={best_uno:.4f} | in_dim={X_tr.shape[1]}")

    print("Fold unos:", [f"{u:.4f}" for u in fold_unos])
    print(f"Mean Uno: {np.mean(fold_unos):.4f} ± {np.std(fold_unos):.4f}")


if __name__ == "__main__":
    main()
