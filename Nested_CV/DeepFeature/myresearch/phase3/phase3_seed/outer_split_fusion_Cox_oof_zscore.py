#!/usr/bin/env python
# outer_split_fusion_Cox_oof_zscore.py

"""
Late Fusion (paper/open-source grade, Scheme M+):

1) Image-only risk: use FROZEN image head (best.pt) + embeddings (*.npy)
   -> logits -> discrete-time logistic hazard -> survival -> risk = -∫S(t)dt

2) Clinical-only risk: CoxPHSurvivalAnalysis on tabular features.
   IMPORTANT: for fusion training we use OOF clinical risk on TRAIN (no in-sample leakage).

3) Normalize risks with TRAIN-only Z-score (no leakage, robust vs MinMax).

4) Fuse risks by learning weights on TRAIN via 2-feature Cox:
      Cox on [img_risk_z, clin_risk_z] fitted on TRAIN only (ridge alpha = fuse_alpha)
   Predict fused risk on VAL, evaluate Uno-C.

Notes from scikit-survival docs:
- CoxPHSurvivalAnalysis.predict returns a time-independent "risk score" = X @ coef; higher => higher risk.
- Uno-IPCW requires tau such that censoring survival is positive at tau.
"""

import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored


# -------------------- utils --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic-ish
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_pid_list(path: Path):
    return pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()


def to_struct(times, events):
    return np.array([(bool(e), float(t)) for t, e in zip(times, events)],
                    dtype=[("event", bool), ("time", float)])


def safe_tau(t_tr: np.ndarray, e_tr: np.ndarray, q: float = 0.9) -> float:
    """
    Pick a tau for Uno-IPCW; try to stay inside support.
    If Uno fails (e.g., censoring survival not positive at tau), we fall back to censored C-index.
    """
    t_tr = np.asarray(t_tr, dtype=float)
    e_tr = np.asarray(e_tr, dtype=int)
    ev = t_tr[e_tr == 1]
    base = ev if ev.size >= 5 else t_tr
    tau = float(np.quantile(base, q))
    mx = float(np.max(t_tr))
    if tau >= mx:
        tau = mx - 1e-6
    if tau <= 0:
        tau = mx * 0.5
    return tau


def pick_np(path_a: Path, path_b: Path):
    if path_a.exists():
        return path_a
    if path_b.exists():
        return path_b
    raise FileNotFoundError(f"Missing {path_a} or {path_b}")


def zscore_normalization(train_arr: np.ndarray, test_arr: np.ndarray, eps: float = 1e-8):
    """Z-score using TRAIN mean/std only (no leakage)."""
    tr = np.asarray(train_arr, dtype=np.float64)
    te = np.asarray(test_arr, dtype=np.float64)
    mu = float(np.mean(tr))
    sd = float(np.std(tr))
    if sd < eps:
        sd = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def load_edges_and_head_state(best_pt: Path):
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)

    edges = np.asarray(ckpt["cuts"], dtype=float)
    n_bins = int(ckpt.get("n_time_bins", len(edges) - 1))

    state = ckpt.get("state_dict", ckpt.get("model_state_dict", None))
    if state is None:
        raise KeyError("Checkpoint missing state_dict/model_state_dict")

    # keep only head.*
    head_state = {k.replace("head.", "seq."): v for k, v in state.items() if k.startswith("head.")}

    if not head_state:
        alt = {k: v for k, v in state.items() if "head" in k.lower()}
        raise KeyError(f"Could not find head.* weights. Example keys: {list(alt)[:10]}")
    return edges, n_bins, head_state


# -------------------- frozen head + risk --------------------
class FrozenImageHead(nn.Module):
    """
    Must match the head architecture used in phase3 training.
    512 -> 256 -> 128 -> n_bins with LN/ReLU/Dropout.
    """
    def __init__(self, n_bins: int, dropout: float = 0.35):
        super().__init__()
        self.seq = nn.Sequential(
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

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.seq(emb)


@torch.no_grad()
def logits_to_surv_np(logits: np.ndarray) -> np.ndarray:
    """
    Discrete-time logistic hazard:
      hazard_k = sigmoid(logit_k)
      S_k = Π_{j<=k} (1 - hazard_j)
    """
    logits = np.asarray(logits, dtype=np.float64)
    haz = 1.0 / (1.0 + np.exp(-logits))
    haz = np.clip(haz, 1e-7, 1.0 - 1e-7)
    surv = np.cumprod(1.0 - haz, axis=1)
    return surv.astype(np.float64)


def surv_to_risk_np(surv: np.ndarray, bin_mids: np.ndarray) -> np.ndarray:
    """Risk = - ∫ S(t) dt (trapezoid). Larger risk => worse survival."""
    return -np.trapz(surv, x=bin_mids, axis=1).astype(np.float64)


@torch.no_grad()
def compute_image_risk_from_frozen_head(emb_np: np.ndarray, head: FrozenImageHead, bin_mids: np.ndarray,
                                       device: torch.device, batch_size: int = 512) -> np.ndarray:
    head.eval()
    emb_np = np.asarray(emb_np, dtype=np.float32)
    n = emb_np.shape[0]
    logits_all = []
    for i in range(0, n, batch_size):
        eb = torch.tensor(emb_np[i:i + batch_size], dtype=torch.float32, device=device)
        lg = head(eb).detach().cpu().numpy()
        logits_all.append(lg)
    logits = np.concatenate(logits_all, axis=0)
    surv = logits_to_surv_np(logits)
    risk = surv_to_risk_np(surv, bin_mids)
    return risk.astype(np.float64)


# -------------------- clinical feature builder (unchanged from your version) --------------------
def build_clinical_features_for_fold(df: pd.DataFrame,
                                     tr_pids: list,
                                     va_pids: list,
                                     overall_stage_encoding: str):
    use_cols = [
        "PatientID", "age", "gender",
        "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage",
        "Overall.Stage",
        "Survival.time", "deadstatus.event"
    ]
    d = df[use_cols].copy()
    d["PatientID"] = d["PatientID"].astype(str)

    tr = d[d["PatientID"].isin(tr_pids)].copy()
    va = d[d["PatientID"].isin(va_pids)].copy()

    tr = tr.set_index("PatientID").loc[tr_pids].reset_index()
    va = va.set_index("PatientID").loc[va_pids].reset_index()

    t_tr = tr["Survival.time"].astype(float).to_numpy()
    e_tr = tr["deadstatus.event"].astype(int).to_numpy()
    t_va = va["Survival.time"].astype(float).to_numpy()
    e_va = va["deadstatus.event"].astype(int).to_numpy()

    feats_tr, feats_va, feat_names = [], [], []

    # age z
    age_tr_series = tr["age"].astype(float)
    mu = float(age_tr_series.mean(skipna=True))
    sigma = float(age_tr_series.std(skipna=True))
    sigma = sigma if sigma > 1e-8 else 1.0
    age_tr = age_tr_series.fillna(mu).to_numpy(dtype=np.float32)
    age_va = va["age"].astype(float).fillna(mu).to_numpy(dtype=np.float32)

    feats_tr.append(((age_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((age_va - mu) / sigma).astype(np.float32))
    feat_names.append("age_z")

    # gender 0/1
    gmap = {"male": 1.0, "female": 0.0}
    g_tr = tr["gender"].astype(str).str.strip().str.lower().map(gmap).to_numpy(dtype=np.float32)
    g_va = va["gender"].astype(str).str.strip().str.lower().map(gmap).to_numpy(dtype=np.float32)
    if np.isnan(g_tr).any() or np.isnan(g_va).any():
        raise ValueError("Unexpected gender values found.")
    feats_tr.append(g_tr); feats_va.append(g_va); feat_names.append("gender_01")

    # T z (fill NaN with train mode)
    T_tr_s = tr["clinical.T.Stage"].astype(float)
    T_va_s = va["clinical.T.Stage"].astype(float)
    T_mode = float(T_tr_s.dropna().value_counts().idxmax())
    T_tr = T_tr_s.fillna(T_mode).to_numpy(dtype=np.float32)
    T_va = T_va_s.fillna(T_mode).to_numpy(dtype=np.float32)
    mu = float(np.mean(T_tr))
    sigma = float(np.std(T_tr)) if float(np.std(T_tr)) > 1e-8 else 1.0
    feats_tr.append(((T_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((T_va - mu) / sigma).astype(np.float32))
    feat_names.append("T_z")

    # N z
    N_tr = tr["Clinical.N.Stage"].astype(float).to_numpy(dtype=np.float32)
    N_va = va["Clinical.N.Stage"].astype(float).to_numpy(dtype=np.float32)
    if np.isnan(N_tr).any() or np.isnan(N_va).any():
        raise ValueError("NaN found in Clinical.N.Stage.")
    mu = float(np.mean(N_tr)); sigma = float(np.std(N_tr)) if float(np.std(N_tr)) > 1e-8 else 1.0
    feats_tr.append(((N_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((N_va - mu) / sigma).astype(np.float32))
    feat_names.append("N_z")

    # M z
    M_tr = tr["Clinical.M.Stage"].astype(float).to_numpy(dtype=np.float32)
    M_va = va["Clinical.M.Stage"].astype(float).to_numpy(dtype=np.float32)
    if np.isnan(M_tr).any() or np.isnan(M_va).any():
        raise ValueError("NaN found in Clinical.M.Stage.")
    mu = float(np.mean(M_tr)); sigma = float(np.std(M_tr)) if float(np.std(M_tr)) > 1e-8 else 1.0
    feats_tr.append(((M_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((M_va - mu) / sigma).astype(np.float32))
    feat_names.append("M_z")

    # Overall.Stage fill NaN with train mode
    os_tr = tr["Overall.Stage"].astype("object")
    os_va = va["Overall.Stage"].astype("object")
    os_mode = os_tr.dropna().value_counts().idxmax()
    os_tr = os_tr.fillna(os_mode).astype(str)
    os_va = os_va.fillna(os_mode).astype(str)

    if overall_stage_encoding == "onehot":
        cats = ["I", "II", "IIIa", "IIIb", "IV"]
        for c in cats:
            feats_tr.append((os_tr == c).to_numpy(dtype=np.float32))
            feats_va.append((os_va == c).to_numpy(dtype=np.float32))
            feat_names.append(f"OverallStage_{c}")
    elif overall_stage_encoding == "ordinal":
        mapping = {"I": 1.0, "II": 2.0, "IIIa": 3.0, "IIIb": 4.0, "IV": 5.0}
        os_tr_num = os_tr.map(mapping).to_numpy(dtype=np.float32)
        os_va_num = os_va.map(mapping).to_numpy(dtype=np.float32)
        if np.isnan(os_tr_num).any() or np.isnan(os_va_num).any():
            raise ValueError("Unexpected Overall.Stage values found.")
        mu = float(np.mean(os_tr_num)); sigma = float(np.std(os_tr_num)) if float(np.std(os_tr_num)) > 1e-8 else 1.0
        feats_tr.append(((os_tr_num - mu) / sigma).astype(np.float32))
        feats_va.append(((os_va_num - mu) / sigma).astype(np.float32))
        feat_names.append("OverallStage_ord_z")
    else:
        raise ValueError("overall_stage_encoding must be 'ordinal' or 'onehot'")

    X_tr = np.stack(feats_tr, axis=1)
    X_va = np.stack(feats_va, axis=1)

    print(f"tr_pids[:5] = {tr_pids[:5]}")
    print(f"tr['PatientID'][:5] = {tr['PatientID'].tolist()[:5]}")
    return X_tr, t_tr, e_tr, X_va, t_va, e_va, feat_names


# -------------------- OOF for clinical Cox (key fix) --------------------
def oof_predict_cox(X: np.ndarray, y_st: np.ndarray, e: np.ndarray, alpha: float,
                    n_splits: int, seed: int) -> np.ndarray:
    """
    Produce OOF risk scores on TRAIN only.
    Use StratifiedKFold on event indicator e to reduce 'no-event fold' failures.
    """
    X = np.asarray(X)
    n = X.shape[0]
    oof = np.empty(n, dtype=np.float64)

    if n_splits < 2:
        raise ValueError("oof_folds must be >= 2")
    if n_splits > n:
        raise ValueError(f"oof_folds ({n_splits}) cannot exceed n_train ({n})")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(np.zeros(n), e.astype(int)):
        m = CoxPHSurvivalAnalysis(alpha=float(alpha))
        m.fit(X[tr_idx], y_st[tr_idx])
        oof[va_idx] = m.predict(X[va_idx]).astype(np.float64)

    return oof


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=str, required=True)
    ap.add_argument("--clinical_csv", type=str, required=True)
    ap.add_argument("--out_name", type=str, default="late_fusion_cox2feat_oof_zscore")

    ap.add_argument("--overall_stage_encoding", type=str, default="onehot", choices=["onehot", "ordinal"])
    ap.add_argument("--seed", type=int, default=42)

    # base clinical Cox ridge
    ap.add_argument("--cox_alpha", type=float, default=0.1, help="L2 penalty for clinical Cox")
    # fusion Cox ridge (separate hyper-param)
    ap.add_argument("--fuse_alpha", type=float, default=0.1, help="L2 penalty for 2-feature fusion Cox")

    # OOF settings
    ap.add_argument("--oof_folds", type=int, default=5, help="StratifiedKFold splits inside outer-train")

    ap.add_argument("--img_head_dropout", type=float, default=0.35, help="must match training head")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--img_batch_size", type=int, default=512)

    # (kept only for backward compatibility; not used anymore)
    ap.add_argument("--alpha_step", type=float, default=0.05, help="(unused) legacy grid search step")

    args = ap.parse_args()
    set_seed(args.seed)

    exp_dir = Path(args.exp_dir)
    out_root = exp_dir / args.out_name
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.clinical_csv)
    device = torch.device(args.device)

    rows = []
    per_fold = {}

    for fold in range(5):
        fold_dir = exp_dir / f"fold_{fold}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"Missing fold dir: {fold_dir}")

        best_pt = fold_dir / "best.pt"
        tr_pids = read_pid_list(fold_dir / "train_pids.csv")
        va_pids = read_pid_list(fold_dir / "val_pids.csv")

        tr_emb_path = pick_np(fold_dir / "train_embeddings.npy", fold_dir / "train_embeddings_extracted.npy")
        va_emb_path = pick_np(fold_dir / "val_embeddings.npy", fold_dir / "val_embeddings_extracted.npy")
        tr_emb = np.load(tr_emb_path).astype(np.float32)
        va_emb = np.load(va_emb_path).astype(np.float32)

        assert tr_emb.shape[0] == len(tr_pids), (tr_emb.shape, len(tr_pids), fold_dir)
        assert va_emb.shape[0] == len(va_pids), (va_emb.shape, len(va_pids), fold_dir)

        edges, n_bins, head_state = load_edges_and_head_state(best_pt)
        bin_mids = (edges[:-1] + edges[1:]) / 2.0

        # clinical X/t/e
        X_tr, t_tr, e_tr, X_va, t_va, e_va, feat_names = build_clinical_features_for_fold(
            df=df, tr_pids=tr_pids, va_pids=va_pids, overall_stage_encoding=args.overall_stage_encoding
        )

        # image head
        head = FrozenImageHead(n_bins=n_bins, dropout=args.img_head_dropout)
        head.load_state_dict(head_state, strict=True)
        head.to(device)
        head.eval()

        # image risks (raw)
        img_risk_tr = compute_image_risk_from_frozen_head(tr_emb, head, bin_mids, device, batch_size=args.img_batch_size)
        img_risk_va = compute_image_risk_from_frozen_head(va_emb, head, bin_mids, device, batch_size=args.img_batch_size)

        # survival structs + tau
        y_tr_st = to_struct(t_tr, e_tr)
        y_va_st = to_struct(t_va, e_va)
        tau = safe_tau(t_tr, e_tr)

        # -------- Step 2: OOF clinical risk on TRAIN --------
        clin_risk_tr_oof = oof_predict_cox(
            X=X_tr, y_st=y_tr_st, e=e_tr,
            alpha=float(args.cox_alpha),
            n_splits=int(args.oof_folds),
            seed=int(args.seed + 1337 + fold)  # fold-dependent but reproducible
        )

        # fit clinical Cox on full TRAIN -> predict VAL
        cox_full = CoxPHSurvivalAnalysis(alpha=float(args.cox_alpha))
        cox_full.fit(X_tr, y_tr_st)
        clin_risk_va = cox_full.predict(X_va).astype(np.float64)

        # -------- Step 1: Z-score normalization (train-only) --------
        img_tr_z, img_va_z = zscore_normalization(img_risk_tr, img_risk_va)
        clin_tr_z, clin_va_z = zscore_normalization(clin_risk_tr_oof, clin_risk_va)

        # image-only / clinical-only (on val) (monotonic scaling doesn't change ordering, but keep consistent)
        try:
            uno_img = float(concordance_index_ipcw(y_tr_st, y_va_st, img_va_z, tau=tau)[0])
            uno_clin = float(concordance_index_ipcw(y_tr_st, y_va_st, clin_va_z, tau=tau)[0])
        except Exception:
            uno_img = float(concordance_index_censored(e_va.astype(bool), t_va, img_va_z)[0])
            uno_clin = float(concordance_index_censored(e_va.astype(bool), t_va, clin_va_z)[0])

        # -------- Step 3: 2-feature fusion Cox learned on TRAIN only --------
        # -------- Step 3: 2-feature fusion Cox learned on TRAIN only --------
        X_fuse_tr = np.column_stack([img_tr_z, clin_tr_z]).astype(np.float64)
        X_fuse_va = np.column_stack([img_va_z, clin_va_z]).astype(np.float64)

        cox_fuse = CoxPHSurvivalAnalysis(alpha=float(args.fuse_alpha))
        cox_fuse.fit(X_fuse_tr, y_tr_st)

        # ---- NEW: clamp coefficients to be non-negative (guardrail) ----
        beta_raw = cox_fuse.coef_.astype(np.float64).copy()      # shape (2,)
        beta = np.maximum(beta_raw, 0.0)                         # disallow subtracting a branch

        if beta.sum() < 1e-12:
            # fallback: do not harm ranking; safest is image-only
            beta = np.array([1.0, 0.0], dtype=np.float64)

        # Use clamped linear score as fused risk score (still a valid risk score for C-index)
        fused_risk_va = (X_fuse_va @ beta).astype(np.float64)

        beta_img, beta_clin = float(beta[0]), float(beta[1])
        alpha_rel = beta_img / (beta_img + beta_clin + 1e-12)    # now meaningful because both >= 0

        # Evaluate fused risk
        try:
            uno_fuse = float(concordance_index_ipcw(y_tr_st, y_va_st, fused_risk_va, tau=tau)[0])
        except Exception:
            uno_fuse = float(concordance_index_censored(e_va.astype(bool), t_va, fused_risk_va)[0])

        print(f"[FUSE-COX] beta_raw=({beta_raw[0]:.3f},{beta_raw[1]:.3f})  beta_clamped=({beta_img:.3f},{beta_clin:.3f})  alpha_rel={alpha_rel:.2f}")


        print(f"\n--- Fold {fold} ---")
        print(f"[DEBUG] features ({X_tr.shape[1]}): {feat_names}")
        print(f"Image-only Uno:    {uno_img:.4f}")
        print(f"Clinical-only Uno: {uno_clin:.4f}")
        print(f"Fused Uno (Cox):   {uno_fuse:.4f} (beta_img={beta_img:.3f}, beta_clin={beta_clin:.3f}, alpha_rel={alpha_rel:.2f})")

        rows.append({
            "fold": fold,
            "uno_img": uno_img,
            "uno_clin": uno_clin,
            "uno_fuse": uno_fuse,
            "beta_img": float(beta_img),
            "beta_clin": float(beta_clin),
            "alpha_rel": float(alpha_rel),
            "tau": float(tau),
            "cox_alpha": float(args.cox_alpha),
            "fuse_alpha": float(args.fuse_alpha),
            "oof_folds": int(args.oof_folds),
            "n_feat": int(X_tr.shape[1]),
        })

        per_fold[f"fold_{fold}"] = {
            "feat_names": feat_names,
            "uno_img": uno_img,
            "uno_clin": uno_clin,
            "uno_fuse": uno_fuse,
            "beta_img": float(beta_img),
            "beta_clin": float(beta_clin),
            "alpha_rel": float(alpha_rel),
            "tau": float(tau),
            "cox_alpha": float(args.cox_alpha),
            "fuse_alpha": float(args.fuse_alpha),
            "oof_folds": int(args.oof_folds),
        }

    df_res = pd.DataFrame(rows)
    df_res.to_csv(out_root / "results.csv", index=False)
    with open(out_root / "per_fold_details.json", "w") as f:
        json.dump(per_fold, f, indent=2)

    print("\n========== Summary ==========")
    print(df_res)
    for col in ["uno_img", "uno_clin", "uno_fuse"]:
        m = df_res[col].mean()
        s = df_res[col].std()
        print(f"{col}: {m:.4f} ± {s:.4f}")
    print(f"avg beta_img:  {df_res['beta_img'].mean():.3f}")
    print(f"avg beta_clin: {df_res['beta_clin'].mean():.3f}")
    print(f"avg alpha_rel: {df_res['alpha_rel'].mean():.3f}")


if __name__ == "__main__":
    main()
