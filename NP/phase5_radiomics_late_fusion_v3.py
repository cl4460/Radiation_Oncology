#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase5_radiomics_late_fusion_v3.py

修正版 v3 - 确保checkpoint和val_pred.csv来自同一个variant

关键改进：
1. 从selection.json读取winner variant
2. 从winner对应的目录读取val_pred.csv和best.pt
3. 保持完全一致性

用法：
  python phase5_radiomics_late_fusion_v3.py \
    --lung1_crop_log /path/to/LUNG1/crop_log.csv \
    --lung1_clinical /path/to/NSCLC-Radiomics-Lung1.clinical.csv \
    --rg_crop_log /path/to/RG/phase2_crop_log.csv \
    --rg_clinical /path/to/Radiogenomics_clinical_aligned.csv \
    --ckpt_root /path/to/lr_7e-04_29night \
    --out_dir /path/to/out \
    --w_grid 0.7 1.0 0.02 \
    --batch_size 2 --gpu 0
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# --- 依赖检查 ---
try:
    import phase4_update1_new as p4
except ImportError as e:
    raise RuntimeError(f"无法 import phase4_update1_new.py: {e}")

try:
    import SimpleITK as sitk
except ImportError:
    raise RuntimeError("缺少 SimpleITK")

try:
    import torchtuples as tt
    from pycox.models import CoxPH
except ImportError:
    raise RuntimeError("缺少 pycox/torchtuples")

try:
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
except ImportError:
    raise RuntimeError("缺少 scikit-survival")


# ============================================================================
# 从checkpoint读取winner和对应的val_pred.csv（关键修复）
# ============================================================================

def get_fold_info(fold_dir: str) -> Dict:
    """
    获取fold的完整信息：
    - winner variant (img or imgclin)
    - winner checkpoint path
    - val_pred.csv path
    - val_patient_ids
    """
    fold_path = fold_dir
    
    # 1. 读取selection.json确定winner
    sel_json = os.path.join(fold_path, "selection.json")
    winner = None
    if os.path.exists(sel_json):
        try:
            with open(sel_json) as f:
                sel = json.load(f)
            winner = sel.get("winner", sel.get("selected", sel.get("variant")))
        except Exception:
            pass
    
    # 2. 如果没有selection.json，从best.pt读取variant
    if winner is None:
        best_pt = os.path.join(fold_path, "best.pt")
        if os.path.exists(best_pt):
            try:
                ckpt = torch.load(best_pt, map_location="cpu")
                winner = ckpt.get("variant")
            except Exception:
                pass
    
    # 3. 如果仍然没有，默认使用model_img
    if winner is None:
        winner = "model_img"
    
    # 4. 获取winner对应的文件路径
    winner_dir = os.path.join(fold_path, winner)
    winner_ckpt = os.path.join(winner_dir, "best.pt")
    winner_val_pred = os.path.join(winner_dir, "val_pred.csv")
    
    # 5. 读取val_patient_ids
    val_patient_ids = None
    if os.path.exists(winner_val_pred):
        try:
            df = pd.read_csv(winner_val_pred)
            val_patient_ids = df["patient_id"].astype(str).tolist()
        except Exception:
            pass
    
    # 6. 如果winner目录没有val_pred.csv，尝试另一个
    if val_patient_ids is None:
        other = "model_imgclin" if winner == "model_img" else "model_img"
        other_val_pred = os.path.join(fold_path, other, "val_pred.csv")
        if os.path.exists(other_val_pred):
            try:
                df = pd.read_csv(other_val_pred)
                val_patient_ids = df["patient_id"].astype(str).tolist()
                print(f"  ⚠️ {os.path.basename(fold_path)}: val_pred from {other}, not {winner}")
            except Exception:
                pass
    
    # 7. 如果还没有，使用fold_X/best.pt
    if not os.path.exists(winner_ckpt):
        winner_ckpt = os.path.join(fold_path, "best.pt")
    
    return {
        "fold_dir": fold_path,
        "winner": winner,
        "ckpt_path": winner_ckpt,
        "val_pred_path": winner_val_pred if os.path.exists(winner_val_pred) else None,
        "val_patient_ids": val_patient_ids,
    }


def list_fold_dirs(ckpt_root: str) -> List[str]:
    return sorted([
        os.path.join(ckpt_root, d) for d in os.listdir(ckpt_root)
        if d.startswith("fold_") and os.path.isdir(os.path.join(ckpt_root, d))
    ])


def load_ckpt(ckpt_path: str) -> Dict:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing: {ckpt_path}")
    return torch.load(ckpt_path, map_location="cpu")


# ============================================================================
# Radiomics特征提取
# ============================================================================

def _safe_entropy(x: np.ndarray, n_bins: int = 64) -> float:
    if x.size == 0:
        return float("nan")
    lo = np.percentile(x, 0.5)
    hi = np.percentile(x, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(x)), float(np.max(x))
        if hi <= lo:
            return 0.0
    hist, _ = np.histogram(x, bins=n_bins, range=(lo, hi), density=False)
    p = hist.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def extract_simple_radiomics_one(ct_path: str, mask_path: str) -> Dict[str, float]:
    img = sitk.ReadImage(ct_path)
    msk = sitk.ReadImage(mask_path)
    ct = sitk.GetArrayFromImage(img).astype(np.float32)
    mk = sitk.GetArrayFromImage(msk).astype(np.uint8)

    if ct.shape != mk.shape:
        raise ValueError(f"CT/mask shape mismatch")

    vox = (mk > 0)
    n_vox = int(vox.sum())
    
    if n_vox == 0:
        return {k: float("nan") if k != "n_vox" else 0 for k in [
            "n_vox", "vol_mm3", "bbox_z_mm", "bbox_y_mm", "bbox_x_mm",
            "hu_mean", "hu_std", "hu_median", "hu_min", "hu_max",
            "hu_p10", "hu_p25", "hu_p75", "hu_p90", "hu_skew", "hu_kurt", "hu_entropy"
        ]}

    spacing = img.GetSpacing()
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    vol = n_vox * sx * sy * sz

    coords = np.argwhere(vox)
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0)

    vals = ct[vox].astype(np.float64)
    mean, std = float(np.mean(vals)), float(np.std(vals))
    
    if std > 1e-8:
        centered = (vals - mean) / std
        skew = float(np.mean(centered ** 3))
        kurt = float(np.mean(centered ** 4) - 3.0)
    else:
        skew, kurt = 0.0, 0.0

    return {
        "n_vox": float(n_vox),
        "vol_mm3": float(vol),
        "bbox_z_mm": float((z1 - z0 + 1) * sz),
        "bbox_y_mm": float((y1 - y0 + 1) * sy),
        "bbox_x_mm": float((x1 - x0 + 1) * sx),
        "hu_mean": mean,
        "hu_std": std,
        "hu_median": float(np.median(vals)),
        "hu_min": float(np.min(vals)),
        "hu_max": float(np.max(vals)),
        "hu_p10": float(np.percentile(vals, 10)),
        "hu_p25": float(np.percentile(vals, 25)),
        "hu_p75": float(np.percentile(vals, 75)),
        "hu_p90": float(np.percentile(vals, 90)),
        "hu_skew": skew,
        "hu_kurt": kurt,
        "hu_entropy": _safe_entropy(vals, n_bins=64),
    }


def extract_simple_radiomics_df(df: pd.DataFrame, cache_csv: Optional[str] = None) -> pd.DataFrame:
    if cache_csv and os.path.exists(cache_csv):
        feats = pd.read_csv(cache_csv)
        feats["patient_id"] = feats["patient_id"].astype(str)
        return feats

    rows = []
    for _, r in df.iterrows():
        pid = str(r["patient_id"])
        ct_path = r.get("out_ct")
        mask_path = r.get("out_mask")
        
        if not ct_path or not os.path.exists(ct_path):
            raise FileNotFoundError(f"[{pid}] missing out_ct")
        if not mask_path or not os.path.exists(mask_path):
            raise FileNotFoundError(f"[{pid}] missing out_mask")

        feat = extract_simple_radiomics_one(ct_path, mask_path)
        feat["patient_id"] = pid
        
        for col in ["retained_mask_ratio", "truncated", "qc_warning"]:
            if col in df.columns:
                feat[col] = r[col]
        rows.append(feat)

    feats = pd.DataFrame(rows)
    if cache_csv:
        os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
        feats.to_csv(cache_csv, index=False)
    return feats


# ============================================================================
# C-index计算
# ============================================================================

def uno_cindex(train_df: pd.DataFrame, test_df: pd.DataFrame, risk: np.ndarray) -> float:
    y_train = Surv.from_arrays(event=train_df["event"].astype(bool).values, time=train_df["time"].values)
    y_test = Surv.from_arrays(event=test_df["event"].astype(bool).values, time=test_df["time"].values)
    return float(concordance_index_ipcw(y_train, y_test, risk)[0])


def harrell_cindex(test_df: pd.DataFrame, risk: np.ndarray) -> float:
    y = Surv.from_arrays(event=test_df["event"].astype(bool).values, time=test_df["time"].values)
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])


# ============================================================================
# Deep Risk: OOF + Ensemble
# ============================================================================

def compute_deep_risk_for_subset(
    df: pd.DataFrame,
    clin_train_df: pd.DataFrame,
    ckpt: Dict,
    device: torch.device,
    batch_size: int = 2,
    num_workers: int = 4,
    use_tta4: bool = True,
) -> np.ndarray:
    model = p4.ResNet10_Clinical_GN(
        in_channels=2,
        clinical_dim=int(ckpt.get("clinical_dim", 12)),
        n_bins=int(ckpt.get("n_time_bins", 14)),
        dropout=float(ckpt.get("dropout", 0.35)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    cuts = np.array(ckpt["cuts"], dtype=np.float64)
    bin_mids = (cuts[:-1] + cuts[1:]) / 2.0

    use_clin = bool(ckpt.get("use_clinical", True))
    feat_names = ckpt.get("clinical_features", [])
    defaults = ckpt.get("clinical_defaults", {})

    if use_clin and feat_names:
        _, X = p4.prepare_clinical_external(clin_train_df, df, feature_names=feat_names, defaults=defaults)
    else:
        X = np.zeros((len(df), max(1, int(ckpt.get("clinical_dim", 1)))), dtype=np.float32)

    tfm = p4.build_test_transforms()
    records = [{"ct": r["out_ct"], "edt": r["out_edt"], "patient_id": str(r["patient_id"])} 
               for _, r in df.iterrows()]
    ds = p4.ExternalTestDataset(records, X, tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    risks = []
    with torch.no_grad():
        for batch in dl:
            img = batch["image"].to(device)
            clin = batch["clinical"].to(device)
            
            if use_tta4:
                surv = p4.predict_surv_tta4(model, img, clin, device, len(bin_mids), use_tta=True)
            else:
                logits = model(img, clin)
                hazards = torch.sigmoid(logits)
                surv = torch.cumprod(1.0 - hazards, dim=1).cpu().numpy()
            
            risk = -np.trapz(surv, bin_mids[None, :], axis=1)
            risks.append(risk)

    return np.concatenate(risks, axis=0)


def deep_risk_oof_consistent(
    lung1_df: pd.DataFrame,
    ckpt_root: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_tta4: bool = True,
) -> Tuple[pd.DataFrame, List[List[str]]]:
    """
    一致性OOF计算：使用winner对应的checkpoint和val_pred.csv
    """
    fold_dirs = list_fold_dirs(ckpt_root)
    
    df = lung1_df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    
    print(f"\n  检查 {len(fold_dirs)} 个fold的配置...")
    
    fold_infos = []
    for fold_dir in fold_dirs:
        info = get_fold_info(fold_dir)
        fold_infos.append(info)
        print(f"  {os.path.basename(fold_dir)}: winner={info['winner']}, "
              f"val_patients={len(info['val_patient_ids']) if info['val_patient_ids'] else 0}")
    
    # 验证
    all_val_ids = set()
    for info in fold_infos:
        if info['val_patient_ids']:
            all_val_ids.update(info['val_patient_ids'])
    
    all_pids = set(df["patient_id"].tolist())
    missing = all_pids - all_val_ids
    if missing:
        print(f"  ⚠️ {len(missing)}个患者不在任何验证集中")

    # 计算OOF risk
    risks = {}
    val_ids_per_fold = []
    
    for i, info in enumerate(fold_infos):
        if not info['val_patient_ids']:
            print(f"  ⚠️ fold_{i}: 没有val_patient_ids!")
            val_ids_per_fold.append([])
            continue
        
        val_ids_per_fold.append(info['val_patient_ids'])
        
        # 加载winner对应的checkpoint
        ckpt = load_ckpt(info['ckpt_path'])
        
        df_val = df[df["patient_id"].isin(info['val_patient_ids'])].reset_index(drop=True)
        
        if len(df_val) == 0:
            print(f"  ⚠️ fold_{i}: 验证集为空！")
            continue
        
        r = compute_deep_risk_for_subset(
            df_val, clin_train_df=df, ckpt=ckpt, device=device,
            batch_size=batch_size, num_workers=num_workers, use_tta4=use_tta4
        )
        
        for pid, risk in zip(df_val["patient_id"].tolist(), r):
            risks[pid] = risk
        
        # Sanity check
        ckpt_val_uno = ckpt.get("val_uno") or ckpt.get("metrics", {}).get("uno")
        if ckpt_val_uno is not None:
            try:
                est = uno_cindex(df, df_val, r)
                diff = abs(est - float(ckpt_val_uno))
                status = "✓" if diff < 0.02 else f"⚠️ diff={diff:.4f}"
                print(f"  fold_{i} ({info['winner']}): Uno={est:.4f} vs ckpt={float(ckpt_val_uno):.4f} {status}")
            except Exception as e:
                print(f"  fold_{i}: 无法计算Uno: {e}")

    # 构建输出
    out_rows = []
    for _, row in df.iterrows():
        pid = row["patient_id"]
        out_rows.append({
            "patient_id": pid,
            "time": row["time"],
            "event": row["event"],
            "deep_risk_oof": risks.get(pid, np.nan),
        })
    
    out = pd.DataFrame(out_rows)
    n_valid = out["deep_risk_oof"].notna().sum()
    print(f"\n  OOF risk: {n_valid}/{len(out)} patients")
    
    return out, val_ids_per_fold


def deep_risk_rg_ensemble(
    rg_df: pd.DataFrame,
    clin_train_df: pd.DataFrame,
    ckpt_root: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    z_source: str = "lung1_val",
    weighted: bool = True,
    use_tta4: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    fold_dirs = list_fold_dirs(ckpt_root)
    n_folds = len(fold_dirs)
    df = rg_df.sort_values("patient_id").reset_index(drop=True).copy()

    per_fold, val_unos, val_mus, val_stds, use_clins = [], [], [], [], []
    
    for fold_dir in fold_dirs:
        info = get_fold_info(fold_dir)
        ckpt = load_ckpt(info['ckpt_path'])
        
        r = compute_deep_risk_for_subset(
            df, clin_train_df=clin_train_df, ckpt=ckpt, device=device,
            batch_size=batch_size, num_workers=num_workers, use_tta4=use_tta4
        )
        per_fold.append(r)
        val_unos.append(float(ckpt.get("val_uno") or ckpt.get("metrics", {}).get("uno", 0.5)))
        val_mus.append(float(ckpt.get("val_risk_mean", np.mean(r))))
        val_stds.append(float(ckpt.get("val_risk_std", np.std(r) + 1e-6)))
        use_clins.append(bool(ckpt.get("use_clinical", True)))

    R = np.stack(per_fold, axis=1)

    if z_source == "lung1_val":
        mu, sd = np.array(val_mus), np.array(val_stds)
    else:
        mu, sd = R.mean(axis=0), R.std(axis=0) + 1e-6

    Z = (R - mu[None, :]) / sd[None, :]

    if weighted:
        v = np.array(val_unos)
        ex = np.exp(v - v.max())
        w = ex / ex.sum()
    else:
        w = np.ones(n_folds) / n_folds

    deep_ens = (Z * w[None, :]).sum(axis=1)

    out = df[["patient_id", "time", "event"]].copy()
    out["deep_risk_ens"] = deep_ens
    
    meta = {
        "z_source": z_source,
        "weighted": weighted,
        "val_unos": val_unos,
        "weights": w.tolist(),
        "use_clinical": use_clins,
    }
    return out, meta


# ============================================================================
# Radiomics CoxPH
# ============================================================================

@dataclass
class CoxCfg:
    hidden: List[int]
    dropout: float
    lr: float
    weight_decay: float
    epochs: int
    batch_size: int
    patience: int


def train_coxph(
    X_train: np.ndarray, durations: np.ndarray, events: np.ndarray,
    X_val: Optional[np.ndarray] = None, d_val: Optional[np.ndarray] = None, e_val: Optional[np.ndarray] = None,
    cfg: CoxCfg = None,
    seed: int = 0
) -> CoxPH:
    if cfg is None:
        cfg = CoxCfg([64, 32], 0.1, 1e-3, 1e-4, 256, 64, 20)
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    net = tt.practical.MLPVanilla(X_train.shape[1], cfg.hidden, 1, batch_norm=True, dropout=cfg.dropout)
    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(cfg.lr)

    callbacks = [tt.callbacks.EarlyStopping(patience=cfg.patience)]
    val_data = (X_val, (d_val, e_val)) if X_val is not None else None

    model.fit(X_train, (durations, events), batch_size=cfg.batch_size, epochs=cfg.epochs,
              callbacks=callbacks, val_data=val_data, verbose=False)
    return model


def radiomics_oof_risk_with_val_ids(
    feats: pd.DataFrame,
    df_surv: pd.DataFrame,
    val_ids_per_fold: List[List[str]],
    cox_cfg: CoxCfg,
    seed: int = 0,
) -> Tuple[pd.DataFrame, Dict]:
    df = df_surv.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    
    feats = feats.copy()
    feats["patient_id"] = feats["patient_id"].astype(str)
    
    merged = df.merge(feats, on="patient_id", how="left")
    merged = merged[merged["n_vox"] > 0].reset_index(drop=True).copy()
    
    all_pids = set(merged["patient_id"].tolist())

    ignore_cols = {"patient_id", "time", "event", "retained_mask_ratio", "truncated", "qc_warning"}
    feature_cols = [c for c in merged.columns if c not in ignore_cols]
    
    scaler = StandardScaler()
    X_all = scaler.fit_transform(merged[feature_cols].values.astype(np.float32))
    
    risks = {}
    
    for k, val_ids in enumerate(val_ids_per_fold):
        if not val_ids:
            continue
            
        val_ids_in_merged = [p for p in val_ids if p in all_pids]
        train_ids = [p for p in all_pids if p not in val_ids]
        
        if not val_ids_in_merged:
            continue
        
        val_mask = merged["patient_id"].isin(val_ids_in_merged)
        train_mask = merged["patient_id"].isin(train_ids)
        
        X_tr = X_all[train_mask.values]
        X_va = X_all[val_mask.values]
        d_tr = merged.loc[train_mask, "time"].values.astype(np.float32)
        d_va = merged.loc[val_mask, "time"].values.astype(np.float32)
        e_tr = merged.loc[train_mask, "event"].values.astype(np.int64)
        e_va = merged.loc[val_mask, "event"].values.astype(np.int64)
        
        model = train_coxph(X_tr, d_tr, e_tr, X_va, d_va, e_va, cfg=cox_cfg, seed=seed + k)
        r_va = model.predict(X_va).reshape(-1)
        
        val_pids = merged.loc[val_mask, "patient_id"].tolist()
        for pid, risk in zip(val_pids, r_va):
            risks[pid] = risk

    out_rows = []
    for _, row in merged.iterrows():
        pid = row["patient_id"]
        out_rows.append({
            "patient_id": pid,
            "time": row["time"],
            "event": row["event"],
            "radio_risk_oof": risks.get(pid, np.nan),
        })
    
    out = pd.DataFrame(out_rows)
    
    meta = {"feature_cols": feature_cols, "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}
    return out, meta


def train_radiomics_full_model(
    feats: pd.DataFrame,
    lung1_df: pd.DataFrame,
    cox_cfg: CoxCfg,
    seed: int = 0,
) -> Tuple[CoxPH, StandardScaler, List[str]]:
    df = lung1_df.sort_values("patient_id").reset_index(drop=True).copy()
    feats = feats.copy()
    feats["patient_id"] = feats["patient_id"].astype(str)
    merged = df.merge(feats, on="patient_id", how="left")
    merged = merged[merged["n_vox"] > 0].reset_index(drop=True).copy()

    ignore_cols = {"patient_id", "time", "event", "retained_mask_ratio", "truncated", "qc_warning"}
    feature_cols = [c for c in merged.columns if c not in ignore_cols]
    
    X = merged[feature_cols].values.astype(np.float32)
    d = merged["time"].values.astype(np.float32)
    e = merged["event"].values.astype(np.int64)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    n = len(merged)
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    split = int(n * 0.9)
    tr_idx, va_idx = idx[:split], idx[split:]

    model = train_coxph(Xs[tr_idx], d[tr_idx], e[tr_idx], Xs[va_idx], d[va_idx], e[va_idx], cfg=cox_cfg, seed=seed)
    return model, scaler, feature_cols


def predict_radiomics_risk(
    feats: pd.DataFrame,
    df: pd.DataFrame,
    model: CoxPH,
    scaler: StandardScaler,
    feature_cols: List[str],
) -> pd.DataFrame:
    feats = feats.copy()
    feats["patient_id"] = feats["patient_id"].astype(str)
    merged = df[["patient_id", "time", "event"]].copy()
    merged["patient_id"] = merged["patient_id"].astype(str)
    merged = merged.merge(feats, on="patient_id", how="left")
    merged = merged[merged["n_vox"] > 0].reset_index(drop=True).copy()

    X = merged[feature_cols].values.astype(np.float32)
    Xs = scaler.transform(X).astype(np.float32)
    
    out = merged[["patient_id", "time", "event"]].copy()
    out["radio_risk"] = model.predict(Xs).reshape(-1)
    
    for col in ["retained_mask_ratio", "truncated", "qc_warning"]:
        if col in merged.columns:
            out[col] = merged[col].values
    return out


# ============================================================================
# Fusion
# ============================================================================

def qc_gate(df: pd.DataFrame) -> np.ndarray:
    g = np.ones(len(df), dtype=np.float64)
    
    if "retained_mask_ratio" in df.columns:
        r = pd.to_numeric(df["retained_mask_ratio"], errors="coerce").fillna(1.0).values
        g *= np.clip(r, 0.0, 1.0)
    
    if "truncated" in df.columns:
        g[df["truncated"].fillna(False).astype(bool).values] *= 0.5
    
    if "qc_warning" in df.columns:
        g[df["qc_warning"].fillna(False).astype(bool).values] *= 0.7
    
    return g


def zscore(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu, sd = float(np.mean(x)), float(np.std(x) + 1e-6)
    return (x - mu) / sd, mu, sd


def choose_fusion_weight(
    deep_oof: pd.DataFrame,
    radio_oof: pd.DataFrame,
    w_start: float,
    w_end: float,
    w_step: float,
) -> Dict:
    df = deep_oof.merge(radio_oof[["patient_id", "radio_risk_oof"]], on="patient_id", how="inner")
    df = df.dropna(subset=["deep_risk_oof", "radio_risk_oof"]).reset_index(drop=True)

    deep_z, mu_d, sd_d = zscore(df["deep_risk_oof"].values)
    radio_z, mu_r, sd_r = zscore(df["radio_risk_oof"].values)
    gate = qc_gate(df)

    uno_deep = uno_cindex(train_df=df, test_df=df, risk=deep_z)
    uno_radio = uno_cindex(train_df=df, test_df=df, risk=radio_z * gate)
    
    best = {"w": None, "uno": -1.0}
    grid = np.arange(w_start, w_end + 1e-9, w_step)
    
    for w in grid:
        fused = w * deep_z + (1.0 - w) * (radio_z * gate)
        u = uno_cindex(train_df=df, test_df=df, risk=fused)
        if u > best["uno"]:
            best = {"w": float(w), "uno": float(u)}

    return {
        "grid": {"start": w_start, "end": w_end, "step": w_step},
        "best": best,
        "uno_deep_oof": float(uno_deep),
        "uno_radio_oof": float(uno_radio),
        "z_params": {"deep_mu": mu_d, "deep_sd": sd_d, "radio_mu": mu_r, "radio_sd": sd_r},
    }


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Radiomics Late Fusion v3 (consistent checkpoint + val_pred)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--lung1_crop_log", required=True)
    ap.add_argument("--lung1_clinical", required=True)
    ap.add_argument("--rg_crop_log", required=True)
    ap.add_argument("--rg_clinical", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_tta4", action="store_true")

    ap.add_argument("--w_grid", type=float, nargs=3, default=[0.7, 1.0, 0.02],
                    metavar=("W_START", "W_END", "W_STEP"))

    # Radiomics Cox config
    ap.add_argument("--cox_hidden", type=int, nargs="*", default=[64, 32])
    ap.add_argument("--cox_dropout", type=float, default=0.1)
    ap.add_argument("--cox_lr", type=float, default=1e-3)
    ap.add_argument("--cox_wd", type=float, default=1e-4)
    ap.add_argument("--cox_epochs", type=int, default=256)
    ap.add_argument("--cox_batch", type=int, default=64)
    ap.add_argument("--cox_patience", type=int, default=20)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Phase 5: Radiomics Late Fusion v3")
    print("=" * 80)
    print(f"  Checkpoint: uses winner variant's checkpoint + val_pred")
    print(f"  W grid: [{args.w_grid[0]}, {args.w_grid[1]}] step={args.w_grid[2]}")
    print(f"  Device: {device}")
    print("=" * 80)

    # 1) Load manifests
    lung1 = p4.load_manifest(args.lung1_crop_log, args.lung1_clinical, dataset_label="LUNG1")
    rg = p4.load_manifest(args.rg_crop_log, args.rg_clinical, dataset_label="RG")
    
    print(f"\nLUNG1: n={len(lung1)}, events={lung1['event'].sum()}")
    print(f"RG: n={len(rg)}, events={rg['event'].sum()}")

    # 2) Radiomics features
    feats_lung1_path = os.path.join(args.out_dir, "radiomics_features_lung1.csv")
    feats_rg_path = os.path.join(args.out_dir, "radiomics_features_rg.csv")
    
    print("\n[1/5] Extracting radiomics features ...")
    feats_lung1 = extract_simple_radiomics_df(lung1, cache_csv=feats_lung1_path)
    feats_rg = extract_simple_radiomics_df(rg, cache_csv=feats_rg_path)

    # 3) Deep risk OOF (consistent)
    print("\n[2/5] Computing deep risk (LUNG1 OOF - consistent) ...")
    deep_oof, val_ids_per_fold = deep_risk_oof_consistent(
        lung1_df=lung1,
        ckpt_root=args.ckpt_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        use_tta4=not args.no_tta4,
    )
    deep_oof_path = os.path.join(args.out_dir, "deep_risk_lung1_oof.csv")
    deep_oof.to_csv(deep_oof_path, index=False)
    
    # 验证OOF质量
    valid_oof = deep_oof.dropna(subset=["deep_risk_oof"])
    if len(valid_oof) > 0:
        oof_uno = uno_cindex(train_df=valid_oof, test_df=valid_oof, risk=valid_oof["deep_risk_oof"].values)
        print(f"\n  Overall Deep OOF Uno: {oof_uno:.4f}")
        if oof_uno < 0.55:
            print("  ⚠️ OOF Uno较低，模型可能性能有限")
        else:
            print("  ✓ OOF Uno正常")

    # 4) Radiomics risk OOF
    print("\n[3/5] Training radiomics CoxPH (LUNG1 OOF) ...")
    cox_cfg = CoxCfg(
        hidden=list(args.cox_hidden),
        dropout=args.cox_dropout,
        lr=args.cox_lr,
        weight_decay=args.cox_wd,
        epochs=args.cox_epochs,
        batch_size=args.cox_batch,
        patience=args.cox_patience,
    )

    radio_oof, radio_meta = radiomics_oof_risk_with_val_ids(
        feats=feats_lung1,
        df_surv=lung1[["patient_id", "time", "event"]],
        val_ids_per_fold=val_ids_per_fold,
        cox_cfg=cox_cfg,
        seed=args.seed,
    )
    radio_oof_path = os.path.join(args.out_dir, "radio_risk_lung1_oof.csv")
    radio_oof.to_csv(radio_oof_path, index=False)

    # 5) Choose fusion weight
    print("\n[4/5] Searching fusion weight w on LUNG1 OOF ...")
    w_start, w_end, w_step = args.w_grid
    fusion = choose_fusion_weight(deep_oof, radio_oof, w_start, w_end, w_step)

    fusion_path = os.path.join(args.out_dir, "fusion_weight.json")
    with open(fusion_path, "w") as f:
        json.dump(fusion, f, indent=2)

    print(f"\n  Deep OOF Uno: {fusion['uno_deep_oof']:.4f}")
    print(f"  Radio OOF Uno: {fusion['uno_radio_oof']:.4f}")
    print(f"  Fused OOF best Uno: {fusion['best']['uno']:.4f} @ w={fusion['best']['w']:.3f}")

    # 6) RG prediction
    print("\n[5/5] RG fused evaluation ...")
    deep_rg, deep_meta = deep_risk_rg_ensemble(
        rg_df=rg,
        clin_train_df=lung1,
        ckpt_root=args.ckpt_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        z_source="lung1_val",
        weighted=True,
        use_tta4=not args.no_tta4,
    )

    full_model, full_scaler, feature_cols = train_radiomics_full_model(
        feats_lung1, lung1[["patient_id", "time", "event"]], cox_cfg=cox_cfg, seed=args.seed
    )
    radio_rg = predict_radiomics_risk(feats_rg, rg[["patient_id", "time", "event"]], full_model, full_scaler, feature_cols)

    merged = deep_rg.merge(radio_rg, on=["patient_id", "time", "event"], how="inner")

    zparams = fusion["z_params"]
    deep_z = (merged["deep_risk_ens"].values - zparams["deep_mu"]) / zparams["deep_sd"]
    radio_z = (merged["radio_risk"].values - zparams["radio_mu"]) / zparams["radio_sd"]

    gate = qc_gate(merged)
    w = fusion["best"]["w"]
    fused = w * deep_z + (1.0 - w) * (radio_z * gate)
    merged["fused_risk"] = fused

    # Metrics
    uno_fused = uno_cindex(train_df=lung1[["time", "event"]], test_df=merged, risk=fused)
    har_fused = harrell_cindex(merged, fused)
    uno_deep_only = uno_cindex(train_df=lung1[["time", "event"]], test_df=merged, risk=deep_z)

    print(f"\n{'='*60}")
    print(f"RG RESULTS")
    print(f"{'='*60}")
    print(f"  Deep-only Uno: {uno_deep_only:.4f}")
    print(f"  Fused Uno:     {uno_fused:.4f} (w={w:.2f})")
    print(f"  Fused Harrell: {har_fused:.4f}")
    print(f"  Delta:         {uno_fused - uno_deep_only:+.4f}")
    print(f"{'='*60}")

    # Save
    out_pred = os.path.join(args.out_dir, "rg_fused_predictions.csv")
    merged.to_csv(out_pred, index=False)

    with open(os.path.join(args.out_dir, "deep_ensemble_meta.json"), "w") as f:
        json.dump(deep_meta, f, indent=2)

    summary = {
        "val_ids_source": "winner_val_pred.csv",
        "w_grid": {"start": w_start, "end": w_end, "step": w_step},
        "lung1_oof": {
            "deep_uno": fusion["uno_deep_oof"],
            "radio_uno": fusion["uno_radio_oof"],
            "fused_uno": fusion["best"]["uno"],
            "best_w": fusion["best"]["w"],
        },
        "rg": {
            "deep_only_uno": float(uno_deep_only),
            "fused_uno": float(uno_fused),
            "fused_harrell": float(har_fused),
            "delta": float(uno_fused - uno_deep_only),
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to: {args.out_dir}")


if __name__ == "__main__":
    main()