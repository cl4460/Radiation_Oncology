#!/usr/bin/env python
# phase3_fusion_additive.py
# Image Embedding (fixed) + Clinical-only residual fusion (additive)

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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
    LR: float = 5e-4
    WD: float = 5e-4
    CLIN_DROPOUT: float = 0.3
    MIN_EVENTS_PER_BIN: int = 15


cfg = CFG()


# ================== Clinical utils ==================
def _clean_stage_text(s):
    if pd.isna(s):
        return None
    s = str(s).strip().upper().replace("STAGE", "").replace(" ", "")
    if s in ["I", "IA", "IB", "1"]:
        return "I"
    if s in ["II", "IIA", "IIB", "2"]:
        return "II"
    if s in ["IIIA", "3A"]:
        return "IIIa"
    if s in ["IIIB", "3B"]:
        return "IIIb"
    return None


def _map_gender_to_float(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    if s in ["m", "male", "1"]:
        return 1.0
    if s in ["f", "female", "0"]:
        return 0.0
    return np.nan


def _to_struct(times, events):
    return np.array(
        [(bool(e), float(t)) for t, e in zip(times, events)],
        dtype=[("event", bool), ("time", float)],
    )


def _uno_from_logits(logits, bin_mids, y_tr, y_va, tau):
    """从 logits 计算 Uno C-index（离散 Logistic-Hazard）"""
    haz = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)  # [N, B]
    log_surv = torch.cumsum(torch.log(1.0 - haz), dim=1)  # [N, B]
    surv = torch.exp(log_surv)  # S(t)
    surv_np = surv.cpu().numpy()
    # risk = - ∫ S(t) dt
    risk = -np.trapz(surv_np, x=bin_mids, axis=1)
    return concordance_index_ipcw(y_tr, y_va, risk, tau=tau)[0]


# ================== Data loading ==================
def load_clinical_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(
        columns={
            "PatientID": "patient_id",
            "Survival.time": "time",
            "deadstatus.event": "event",
            "Overall.Stage": "overall_stage",
            "gender": "gender",
            "age": "age",
        }
    )
    df["stage_clean"] = df["overall_stage"].apply(_clean_stage_text)
    stage_map = {"I": 1.0, "II": 2.0, "IIIa": 3.0, "IIIb": 4.0}
    df["stage_numeric_raw"] = df["stage_clean"].map(stage_map)

    df["event"] = df["event"].astype(int)
    df["time"] = df["time"].astype(float)
    df["patient_id"] = df["patient_id"].astype(str)
    return df.set_index("patient_id")


def build_clinical_features(df_indexed, t_pids, v_pids):
    tr = df_indexed.loc[t_pids].copy()
    va = df_indexed.loc[v_pids].copy()

    # Age impute (mean)
    age_mean = tr["age"].mean()
    tr["age"] = tr["age"].fillna(age_mean)
    va["age"] = va["age"].fillna(age_mean)

    # Stage impute (mode)
    st_mode = tr["stage_numeric_raw"].mode()[0]
    tr["stage_numeric_raw"] = tr["stage_numeric_raw"].fillna(st_mode)
    va["stage_numeric_raw"] = va["stage_numeric_raw"].fillna(st_mode)

    # Normalize age & stage
    age_mu, age_std = tr["age"].mean(), tr["age"].std() + 1e-8
    st_mu, st_std = (
        tr["stage_numeric_raw"].mean(),
        tr["stage_numeric_raw"].std() + 1e-8,
    )

    age_tr = (tr["age"].values - age_mu) / age_std
    age_va = (va["age"].values - age_mu) / age_std
    st_tr = (tr["stage_numeric_raw"].values - st_mu) / st_std
    st_va = (va["stage_numeric_raw"].values - st_mu) / st_std

    # Gender to 0/1, impute mode
    g_tr = np.array([_map_gender_to_float(x) for x in tr["gender"]], dtype=float)
    g_va = np.array([_map_gender_to_float(x) for x in va["gender"]], dtype=float)
    g_mode = pd.Series(g_tr).mode()[0]
    g_tr = np.nan_to_num(g_tr, nan=g_mode)
    g_va = np.nan_to_num(g_va, nan=g_mode)

    X_tr = np.stack([age_tr, g_tr, st_tr], axis=1).astype(np.float32)
    X_va = np.stack([age_va, g_va, st_va], axis=1).astype(np.float32)

    return (
        X_tr,
        X_va,
        tr["time"].values,
        tr["event"].values,
        va["time"].values,
        va["event"].values,
    )


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


# ================== Models ==================
class ImageHead(nn.Module):
    """
    与 ResNet10_ImageOnly_GN 中 head 完全一致:
    512 -> 256 -> 128 -> n_bins
    """

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
    """从 best.pt 中提取 head 参数和时间切分信息。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "cuts" not in ckpt:
        raise ValueError(f"{ckpt_path} missing 'cuts' in checkpoint.")

    cuts = np.array(ckpt["cuts"], dtype=float)
    n_bins = ckpt.get("n_time_bins", len(cuts) - 1)

    head = ImageHead(n_bins=n_bins, dropout=dropout)

    state_dict = ckpt.get("state_dict", ckpt)
    head_state = {}
    for k, v in state_dict.items():
        if "head." in k:
            k_clean = k
            if k_clean.startswith("module."):
                k_clean = k_clean[len("module.") :]
            head_state[k_clean] = v

    if not head_state:
        raise ValueError(f"No 'head.' parameters found in {ckpt_path}")

    # 先尝试 strict=True，确保结构完全一致
    try:
        missing, unexpected = head.load_state_dict(head_state, strict=True)
        if missing or unexpected:
            print(f"[WARN] strict=True but got missing={missing}, unexpected={unexpected}")
    except RuntimeError as e:
        print(f"[WARN] strict load failed ({e}), fallback to strict=False")
        missing, unexpected = head.load_state_dict(head_state, strict=False)

    loaded_params = sum(1 for k in head_state if k in head.state_dict())
    print(f"[{ckpt_path.name}] ImageHead loaded params: {loaded_params}")

    head.eval()
    for p in head.parameters():
        p.requires_grad = False

    return head.to(DEVICE), cuts, n_bins


class ClinicalNet(nn.Module):
    def __init__(self, in_dim: int, n_bins: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_bins),
        )
        # Zero-init 最后一层：起点 = image-only baseline
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class AdditiveFusion(nn.Module):
    """
    logits_total = logits_img (frozen) + logits_clin (learned)
    """

    def __init__(self, img_head: nn.Module, clin_net: nn.Module):
        super().__init__()
        self.img_head = img_head
        self.clin_net = clin_net

        # 冻结 image head 参数 & 模式
        self.img_head.eval()
        for p in self.img_head.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        # 只让 clinical 分支进入 train 模式，image head 永远 eval
        super().train(mode)
        self.img_head.eval()
        for p in self.img_head.parameters():
            p.requires_grad = False
        return self

    def forward(self, emb, clin):
        with torch.no_grad():
            logits_img = self.img_head(emb)  # [B, n_bins]
        logits_clin = self.clin_net(clin)   # [B, n_bins]
        return logits_img + logits_clin


# ================== Dataset ==================
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

    print("\n========== PHASE 1: ADDITIVE RESIDUAL FUSION ==========")
    print(f"LR={cfg.LR:.2e}, WD={cfg.WD:.1e}, dropout_clin={cfg.CLIN_DROPOUT:.2f}")

    for fold in range(5):
        print(f"\n=== Fold {fold} ===")
        try:
            t_emb, v_emb, t_pids, v_pids, img_head, cuts, n_bins = load_phase3_artifacts(
                exp_dir, fold, dropout_head=0.35
            )
        except Exception as e:
            print(f"[Fold {fold}] Skip due to error: {e}")
            continue

        # 过滤掉 clinical 中没有的 patient
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

        # LabTrans: 使用原 cuts 中间的切分点
        lab = LabTransDiscreteTime(cuts=cuts[1:-1])
        y_tr = lab.transform(t_tr, e_tr)
        y_va = lab.transform(t_va, e_va)
        y_tr_idx, y_tr_evt = y_tr
        y_va_idx, y_va_evt = y_va

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
        print(f"[Fold {fold}] Image-only baseline Uno = {uno_base:.4f}")

        # Fusion 模型
        clin_net = ClinicalNet(in_dim=3, n_bins=n_bins, dropout=cfg.CLIN_DROPOUT).to(DEVICE)
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
        patience = 25
        no_improve = 0

        for ep in range(1, cfg.EPOCHS + 1):
            model.train()
            total_loss = 0.0
            for emb_b, clin_b, yi_b, ye_b in dl_tr:
                optimizer.zero_grad()
                emb_b = emb_b.to(DEVICE)
                clin_b = clin_b.to(DEVICE)
                yi_b = yi_b.to(DEVICE)
                ye_b = ye_b.to(DEVICE)

                logits = model(emb_b, clin_b)
                loss = loss_fn(logits, yi_b, ye_b)
                loss.backward()
                nn.utils.clip_grad_norm_(clin_net.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * emb_b.size(0)

            avg_loss = total_loss / len(X_tr)

            # 验证
            model.eval()
            with torch.no_grad():
                all_logits = []
                for emb_b, clin_b, _, _ in dl_va:
                    logits = model(emb_b.to(DEVICE), clin_b.to(DEVICE))
                    all_logits.append(logits)
                logits_va = torch.cat(all_logits, dim=0)
                uno_curr = _uno_from_logits(
                    logits_va, bin_mids, y_tr_struct, y_va_struct, tau
                )

            print(
                f"[Fold {fold}] Ep {ep:03d} | loss={avg_loss:.4f} | Uno={uno_curr:.4f}"
            )

            if uno_curr > best_uno + 1e-4:
                best_uno = uno_curr
                best_ep = ep
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"[Fold {fold}] Early stopping at ep {ep}")
                break

        # fallback：如果 best_uno 仍然 <= baseline，当作没提升
        final_uno = max(best_uno, uno_base)
        status = "IMPROVED" if best_uno > uno_base + 1e-3 else "FALLBACK/FLAT"
        print(
            f"[Fold {fold}] Final Uno = {final_uno:.4f} ({status}), Δ={final_uno - uno_base:+.4f}, best_ep={best_ep}"
        )
        fold_results.append(final_uno)

    if fold_results:
        print("\n========== FUSION SUMMARY ==========")
        print(
            f"Mean Uno: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f} "
            f"over {len(fold_results)} folds"
        )


if __name__ == "__main__":
    main()
