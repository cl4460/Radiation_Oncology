#!/usr/bin/env python
"""
Exp A1 V2 (TNM-only): Feature Extraction + DeepSurv
- 仅使用 TNM (T/N/M) one-hot，不使用 overall stage（避免信息重复）
- 可选加入 Histology（数据驱动 one-hot；NOS 可保留为独立类或并入缺失）
- Tau 固定为 730 天（24 个月），对齐 uno@24m
- Mini-batch 训练，适度正则化
"""

import os, random, warnings, json
from pathlib import Path
from collections import Counter
from typing import Tuple, List, Optional, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# Paths / Config
# ============================================================
CROP_LOG_CSV   = "/home/lichengze/Research/CNN_pipeline/phase2_outputs/crop_log.csv"
CLINICAL_CSV   = "/home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical.csv"

# 这里假设你已完成特征提取（512-d deep features）并保存到 Exp1/features 下
FEATURES_DIR   = "/home/lichengze/Research/DeepFeature/myresearch/Exp1/features"
DEEPSURV_DIR   = "/home/lichengze/Research/DeepFeature/myresearch/Exp1/deepsurv_v2_tnm_only"

Path(DEEPSURV_DIR).mkdir(parents=True, exist_ok=True)

N_FOLDS        = 5
SEED           = 42
TAU_DAYS       = 730.0          # 固定 24 个月
BATCH_SIZE     = 32
LR             = 3e-4
WEIGHT_DECAY   = 1e-5
DROPOUT        = 0.15
MAX_EPOCHS     = 200
EARLY_PATIENCE = 20              # 每 2 epoch 验证一次

# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ============================================================
# Utils for Clinical Features (TNM + optional Histology)
# ============================================================
def _safe_mode(values: np.ndarray, default: float) -> float:
    vals = values[~np.isnan(values)]
    if vals.size == 0:
        return float(default)
    cnt = Counter(vals.tolist())
    return sorted(cnt.items(), key=lambda x: (-x[1], x[0]))[0][0]

def _num_clean(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").astype(np.float32).values

def _norm_gender(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.upper().str[0]
    return (s == "M").astype(np.float32).values

def _canonical_histology(text: str) -> Optional[str]:
    """
    归一化 Histology：
      - 'adenocarcinoma'
      - 'squamous cell carcinoma'
      - 'large cell'
      - 'nos'（Not Otherwise Specified）
    未识别或 NA/空 → None（缺失）
    """
    if text is None:
        return None
    s = str(text).strip().lower()
    if s in {"", "na", "n/a", "none", "null"}:
        return None
    import re as _re
    s = _re.sub(r"[^a-z0-9\s]", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    if "adenocarcinoma" in s:
        return "adenocarcinoma"
    if "squamous" in s:
        return "squamous cell carcinoma"
    if "large cell" in s:
        return "large cell"
    if s == "nos" or s.endswith(" nos"):
        return "nos"
    return None

def _one_hot_from_labels(labels: np.ndarray, classes: List[str]) -> np.ndarray:
    idx_map = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(labels), len(classes)), dtype=np.float32)
    for i, lab in enumerate(labels):
        if lab in idx_map:
            out[i, idx_map[lab]] = 1.0
    return out

def prepare_clinical_tnm_only(
    manifest: pd.DataFrame,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    include_histology: bool = True,
    histology_policy: Literal["keep_nos", "nos_as_missing"] = "keep_nos",
    min_histology_count: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    只使用 TNM（T/N/M）+ 可选 Histology 的临床特征（全部 one-hot + 缺失指示）。
    返回：X_train, X_val, feature_names
    """
    feats_tr, feats_va, names = [], [], []

    # ---- Age ----
    if "age" in manifest.columns:
        age = _num_clean(manifest["age"])
        age_missing = np.isnan(age).astype(np.float32)
        med = np.nanmedian(age[idx_train]) if np.any(~np.isnan(age[idx_train])) else 65.0
        age_filled = age.copy(); age_filled[np.isnan(age_filled)] = med
        feats_tr += [age_filled[idx_train], age_missing[idx_train]]
        feats_va += [age_filled[idx_val],   age_missing[idx_val]]
        names += ["age", "age_missing"]

    # ---- Gender ----
    if "gender" in manifest.columns:
        g_male = _norm_gender(manifest["gender"])
        feats_tr.append(g_male[idx_train]); feats_va.append(g_male[idx_val])
        names.append("gender_male")

    # ---- T (1/2/3/4) ----
    if "T" in manifest.columns:
        t = _num_clean(manifest["T"])
        t = np.where(np.isin(t, [1, 2, 3, 4]), t, np.nan).astype(np.float32)
        t_missing = np.isnan(t).astype(np.float32)
        t_mode = _safe_mode(t[idx_train], default=2.0)
        t_fill = t.copy(); t_fill[np.isnan(t_fill)] = t_mode
        present_t = sorted(set(t_fill[~np.isnan(t_fill)].astype(int).tolist()))
        if verbose: print(f"[TNM] T present: {present_t}")
        for tv in present_t:
            feats_tr.append((t_fill[idx_train] == tv).astype(np.float32))
            feats_va.append((t_fill[idx_val]   == tv).astype(np.float32))
            names.append(f"T{tv}")
        feats_tr.append(t_missing[idx_train]); feats_va.append(t_missing[idx_val]); names.append("T_missing")

    # ---- N (0/1/2/3) ----
    if "N" in manifest.columns:
        n = _num_clean(manifest["N"])
        n = np.where(np.isin(n, [0, 1, 2, 3]), n, np.nan).astype(np.float32)
        n_missing = np.isnan(n).astype(np.float32)
        n_mode = _safe_mode(n[idx_train], default=1.0)
        n_fill = n.copy(); n_fill[np.isnan(n_fill)] = n_mode
        present_n = sorted(set(n_fill[~np.isnan(n_fill)].astype(int).tolist()))
        if verbose: print(f"[TNM] N present: {present_n}")
        for nv in present_n:
            feats_tr.append((n_fill[idx_train] == nv).astype(np.float32))
            feats_va.append((n_fill[idx_val]   == nv).astype(np.float32))
            names.append(f"N{nv}")
        feats_tr.append(n_missing[idx_train]); feats_va.append(n_missing[idx_val]); names.append("N_missing")

    # ---- M (0/1) ----
    if "M" in manifest.columns:
        m = _num_clean(manifest["M"])
        m = np.where(np.isin(m, [0, 1]), m, np.nan).astype(np.float32)
        m_missing = np.isnan(m).astype(np.float32)
        m_mode = _safe_mode(m[idx_train], default=0.0)
        m_fill = m.copy(); m_fill[np.isnan(m_fill)] = m_mode
        present_m = sorted(set(m_fill[~np.isnan(m_fill)].astype(int).tolist()))
        if verbose: print(f"[TNM] M present: {present_m}")
        for mv in present_m:
            feats_tr.append((m_fill[idx_train] == mv).astype(np.float32))
            feats_va.append((m_fill[idx_val]   == mv).astype(np.float32))
            names.append(f"M{mv}")
        feats_tr.append(m_missing[idx_train]); feats_va.append(m_missing[idx_val]); names.append("M_missing")

    # ---- Histology （可选）----
    if include_histology and ("histology" in manifest.columns):
        raw = manifest["histology"].astype(str).tolist()
        canon = np.array([_canonical_histology(x) for x in raw], dtype=object)
        if histology_policy == "nos_as_missing":
            canon = np.array([None if c == "nos" else c for c in canon], dtype=object)

        tr_vals = [c for i, c in enumerate(canon) if (i in idx_train) and (c is not None)]
        vc = Counter(tr_vals)
        classes = [c for c, k in vc.items() if k >= min_histology_count]
        classes = sorted(classes)
        if verbose: print(f"[Histology] kept classes: {classes} (min_count={min_histology_count})")

        oh_all = _one_hot_from_labels(canon, classes)
        feats_tr.append(oh_all[idx_train]); feats_va.append(oh_all[idx_val])
        names += [f"Histology::{c}" for c in classes]

        hist_missing = np.array([1.0 if c is None else 0.0 for c in canon], dtype=np.float32)
        feats_tr.append(hist_missing[idx_train]); feats_va.append(hist_missing[idx_val]); names.append("Histology_missing")

    # ---- Stack ----
    if len(feats_tr) == 0:
        Xtr = np.zeros((len(idx_train), 1), dtype=np.float32)
        Xva = np.zeros((len(idx_val), 1),   dtype=np.float32)
        names = ["dummy"]
    else:
        Xtr = np.stack(feats_tr, axis=1).astype(np.float32)
        Xva = np.stack(feats_va, axis=1).astype(np.float32)

    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xva = np.nan_to_num(Xva, nan=0.0, posinf=0.0, neginf=0.0)

    if verbose:
        print(f"[Clinical] TNM-only, include_histology={include_histology}")
        print(f"[Clinical] X_train: {Xtr.shape}, X_val: {Xva.shape}")
        print(f"[Clinical] features: {', '.join(names)}")

    return Xtr, Xva, names

# ============================================================
# Stable Cox Loss (Breslow ties)
# ============================================================
def cox_ph_loss_breslow(log_h, time, event, eps=1e-7):
    order = torch.argsort(time, descending=True)
    log_h = log_h.squeeze()[order]
    event = event[order].float()
    log_risk = torch.logcumsumexp(log_h, dim=0)
    loss = -(log_h - log_risk) * event
    return loss.sum() / (event.sum() + eps)

# ============================================================
# DeepSurv
# ============================================================
class DeepSurv(nn.Module):
    def __init__(self, in_features, hidden_1=128, hidden_2=64, dropout=DROPOUT):
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
            nn.Linear(hidden_2, 1)    # log-hazard
        )
    def forward(self, x):
        return self.network(x)

# ============================================================
# Train one fold
# ============================================================
def train_deepsurv_fold(fold: int, manifest: pd.DataFrame):
    print(f"\n{'='*70}\nTrain DeepSurv (TNM-only) - Fold {fold}\n{'='*70}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load deep features & indices
    train_deep = np.load(f"{FEATURES_DIR}/fold{fold}_train_features.npy")
    val_deep   = np.load(f"{FEATURES_DIR}/fold{fold}_val_features.npy")
    idx_train  = np.load(f"{FEATURES_DIR}/fold{fold}_train_idx.npy")
    idx_val    = np.load(f"{FEATURES_DIR}/fold{fold}_val_idx.npy")

    # --- Safety: ID 对齐检查 ---
    assert manifest.loc[idx_train, "patient_id"].nunique() == len(idx_train), \
        f"[FOLD {fold}] Training set patient_id unique count != len(idx_train)"
    assert manifest.loc[idx_val, "patient_id"].nunique() == len(idx_val), \
        f"[FOLD {fold}] Validation set patient_id unique count != len(idx_val)"

    # Clinical features: TNM-only (+ Histology)
    clinical_train, clinical_val, clinical_names = prepare_clinical_tnm_only(
        manifest, idx_train, idx_val,
        include_histology=True,            # 如不需要，改为 False
        histology_policy="keep_nos",       # 或 'nos_as_missing'
        min_histology_count=1,
        verbose=True
    )

    # Combine & standardize
    X_train = np.hstack([train_deep, clinical_train])
    X_val   = np.hstack([val_deep,   clinical_val])
    print(f"Features: Train {X_train.shape}, Val {X_val.shape}  (Deep 512 + Clinical {clinical_train.shape[1]})")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    joblib.dump(scaler, f"{DEEPSURV_DIR}/fold{fold}_scaler.joblib")

    # Labels
    y_train = {
        "time":  manifest.iloc[idx_train]["time"].values.astype(float),
        "event": manifest.iloc[idx_train]["event"].values.astype(float),
    }
    y_val = {
        "time":  manifest.iloc[idx_val]["time"].values.astype(float),
        "event": manifest.iloc[idx_val]["event"].values.astype(float),
    }

    # Tau 固定为 730 天
    tau = TAU_DAYS
    print(f"Tau fixed: {tau:.1f} days (24 months)")

    # Model/optim
    model = DeepSurv(in_features=X_train.shape[1]).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Dataloaders
    ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train["time"]),
        torch.FloatTensor(y_train["event"]),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    print(f"Training with mini-batch (batch={BATCH_SIZE}, {len(loader)} batches/epoch)")

    # Surv objects
    y_train_surv = Surv.from_arrays(event=y_train["event"].astype(bool), time=y_train["time"])
    y_val_surv   = Surv.from_arrays(event=y_val["event"].astype(bool),   time=y_val["time"])
    X_val_t      = torch.FloatTensor(X_val).to(device)

    best_uno, patience = 0.0, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for xb, tb, eb in loader:
            xb = xb.to(device); tb = tb.to(device); eb = eb.to(device)
            log_h = model(xb)
            loss = cox_ph_loss_breslow(log_h, tb, eb)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            loss_sum += loss.item(); n_batches += 1

        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.inference_mode():
                risks = model(X_val_t).cpu().numpy().ravel()
            try:
                uno = concordance_index_ipcw(y_train_surv, y_val_surv, risks, tau=tau)[0]
            except Exception:
                uno = np.nan
            avg_loss = loss_sum / max(n_batches, 1)
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Uno@24m: {uno:.4f}")

            if np.isfinite(uno) and (uno > best_uno + 1e-4):
                best_uno = uno; patience = 0
                torch.save({
                    "state_dict": model.state_dict(),
                    "uno": float(uno),
                    "epoch": epoch,
                    "tau": float(tau),
                    "scaler_path": f"{DEEPSURV_DIR}/fold{fold}_scaler.joblib",
                    "clinical_feature_names": clinical_names,
                }, f"{DEEPSURV_DIR}/fold{fold}_best.pt")
            else:
                patience += 1
                if patience >= EARLY_PATIENCE:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

    print(f"Fold {fold}: Best Uno@24m = {best_uno:.4f}")
    return {"fold": fold, "best_uno": float(best_uno), "tau": float(tau)}

# ============================================================
# Main
# ============================================================
def main():
    print("="*70)
    print("Exp A1 V2 (TNM-only): Feature Extraction + DeepSurv")
    print("="*70)
    print("  ✓ Only TNM (T/N/M) one-hot; NO overall stage")
    print("  ✓ Optional Histology one-hot (data-driven)")
    print(f"  ✓ Tau fixed @ {TAU_DAYS} days (uno@24m)")
    print("  ✓ Mini-batch + AdamW + moderate regularization")
    print("="*70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # --- Load & unify columns ---
    crop = pd.read_csv(CROP_LOG_CSV)
    clinical = pd.read_csv(CLINICAL_CSV)

    col_map = {}
    for col in clinical.columns:
        lc = col.lower().strip()
        if ("patient" in lc) or ("case" in lc):
            col_map[col] = "patient_id"
        elif ("survival" in lc and "time" in lc) or (lc == "time"):
            col_map[col] = "time"
        elif ("dead" in lc) or ("event" in lc):
            col_map[col] = "event"
        elif ("age" in lc) and ("stage" not in lc):
            col_map[col] = "age"
        elif ("gender" in lc) or ("sex" in lc):
            col_map[col] = "gender"
        elif (lc == "t") or ("t.stage" in lc) or (".t." in lc):
            col_map[col] = "T"
        elif (lc == "n") or ("n.stage" in lc) or (".n." in lc):
            col_map[col] = "N"
        elif (lc == "m") or ("m.stage" in lc) or (".m." in lc):
            col_map[col] = "M"
        elif "histology" in lc:
            col_map[col] = "histology"

    clinical = clinical.rename(columns=col_map)

    manifest = crop[["patient_id", "out_ct", "out_edt"]].rename(columns={
        "out_ct": "ct_path",
        "out_edt": "edt_path"
    }).merge(clinical, on="patient_id", how="inner")

    manifest = manifest.dropna(subset=["time", "event"])
    manifest["time"]  = manifest["time"].astype(float)
    manifest["event"] = manifest["event"].astype(float)

    print(f"Dataset: {len(manifest)} patients")

    # --- Run all folds ---
    results = []
    for fold in range(N_FOLDS):
        r = train_deepsurv_fold(fold, manifest)
        results.append(r)

    # --- Summaries ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{DEEPSURV_DIR}/results.csv", index=False)

    mean_uno = results_df["best_uno"].mean()
    std_uno  = results_df["best_uno"].std()

    print("\n" + "="*70)
    print("Exp A1 V2 (TNM-only) FINAL RESULTS")
    print("="*70)
    print(f"Uno@24m: {mean_uno:.4f} ± {std_uno:.4f}")
    for _, row in results_df.iterrows():
        print(f"  Fold {int(row['fold'])}: {row['best_uno']:.4f}")
    print("="*70)

    with open(f"{DEEPSURV_DIR}/summary.json", "w") as f:
        json.dump({
            "exp_id": "A1_V2_TNM_ONLY",
            "method": "Deep features (512) + TNM-only clinical (one-hot) + optional Histology",
            "tau_days": TAU_DAYS,
            "mean_uno_24m": float(mean_uno),
            "std_uno_24m": float(std_uno),
            "folds": results
        }, f, indent=2)

    print(f"\nResults saved to: {DEEPSURV_DIR}/")

if __name__ == "__main__":
    main()
