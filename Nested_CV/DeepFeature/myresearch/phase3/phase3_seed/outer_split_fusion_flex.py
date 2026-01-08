#!/usr/bin/env python
# outer_split_fusion_flex.py

import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored


# -------------------- utils --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_pid_list(path: Path):
    return pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()


def to_struct(times, events):
    return np.array([(bool(e), float(t)) for t, e in zip(times, events)],
                    dtype=[("event", bool), ("time", float)])


def safe_tau(t_tr: np.ndarray, e_tr: np.ndarray, q: float = 0.9) -> float:
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


def load_edges_and_head_state(best_pt: Path):
    # keep robust to different checkpoint formats
    try:
        ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(best_pt, map_location="cpu")

    edges = np.asarray(ckpt["cuts"], dtype=float)
    n_bins = int(ckpt.get("n_time_bins", len(edges) - 1))

    state = ckpt.get("state_dict", ckpt.get("model_state_dict", None))
    if state is None:
        raise KeyError("Checkpoint missing state_dict/model_state_dict")

    head_state = {k.replace("head.", "seq."): v for k, v in state.items() if k.startswith("head.")}
    if not head_state:
        alt = [k for k in state.keys() if "head" in k.lower()]
        raise KeyError(f"Could not find head.* weights. Example head-like keys: {alt[:10]}")
    return edges, n_bins, head_state


def zscore_by_train(train_arr: np.ndarray, test_arr: np.ndarray, eps: float = 1e-8):
    tr = np.asarray(train_arr, dtype=np.float64)
    te = np.asarray(test_arr, dtype=np.float64)
    mu = float(np.mean(tr))
    sd = float(np.std(tr))
    if sd < eps:
        sd = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum())) + 1e-12
    return float((a*b).sum() / denom)


def residualize_clin_on_img(img_tr: np.ndarray, clin_tr: np.ndarray,
                            img_va: np.ndarray, clin_va: np.ndarray):
    """
    clin_res = clin - k * img, where k is OLS slope learned on TRAIN only.
    Return clin_res_tr, clin_res_va, slope k.
    """
    x = np.asarray(img_tr, dtype=np.float64)
    y = np.asarray(clin_tr, dtype=np.float64)
    vx = float(np.var(x))
    if vx < 1e-12:
        k = 0.0
    else:
        k = float(np.cov(x, y, bias=True)[0, 1] / (vx + 1e-12))
    clin_res_tr = y - k * x
    clin_res_va = np.asarray(clin_va, dtype=np.float64) - k * np.asarray(img_va, dtype=np.float64)
    return clin_res_tr, clin_res_va, k


def cindex_harrell(event, time, risk):
    # concordance_index_censored expects higher risk -> shorter survival
    return float(concordance_index_censored(event.astype(bool), time.astype(float), risk.astype(float))[0])


def eval_uno_or_harrell(y_tr_st, y_va_st, t_va, e_va, risk_va, tau):
    try:
        return float(concordance_index_ipcw(y_tr_st, y_va_st, risk_va, tau=tau)[0])
    except Exception:
        return float(concordance_index_censored(e_va.astype(bool), t_va.astype(float), risk_va.astype(float))[0])


def fit_cox_oof(X_tr: np.ndarray, y_tr_st, alpha: float, oof_folds: int, seed: int):
    """
    OOF prediction for CoxPHSurvivalAnalysis on TRAIN only.
    """
    n = X_tr.shape[0]
    if oof_folds <= 1:
        raise ValueError("oof_folds must be >=2 for OOF.")

    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    folds = np.array_split(idx, oof_folds)
    oof = np.zeros(n, dtype=np.float64)

    for k in range(oof_folds):
        va_idx = folds[k]
        tr_idx = np.concatenate([folds[j] for j in range(oof_folds) if j != k])
        model = CoxPHSurvivalAnalysis(alpha=float(alpha))
        model.fit(X_tr[tr_idx], y_tr_st[tr_idx])
        oof[va_idx] = model.predict(X_tr[va_idx]).astype(np.float64)

    return oof


# -------------------- frozen head --------------------
class FrozenImageHead(nn.Module):
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
    logits = np.asarray(logits, dtype=np.float64)
    haz = 1.0 / (1.0 + np.exp(-logits))
    haz = np.clip(haz, 1e-7, 1.0 - 1e-7)
    surv = np.cumprod(1.0 - haz, axis=1)
    return surv.astype(np.float64)


def surv_to_risk_np(surv: np.ndarray, bin_mids: np.ndarray) -> np.ndarray:
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


# -------------------- CLINICAL FEATURE BUILDER --------------------
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

    # Overall.Stage
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

    return X_tr, t_tr, e_tr, X_va, t_va, e_va, feat_names


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--exp_dir", type=str, required=True)
    ap.add_argument("--clinical_csv", type=str, required=True)
    ap.add_argument("--out_name", type=str, default="late_fusion_flex")

    ap.add_argument("--overall_stage_encoding", type=str, default="onehot", choices=["onehot", "ordinal"])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--cox_alpha", type=float, default=0.1, help="L2 for clinical Cox")
    ap.add_argument("--fuse_model", type=str, default="cox", choices=["cox", "svm"])
    ap.add_argument("--fuse_alpha", type=float, default=0.1, help="regularization for fuse Cox or SVM")
    ap.add_argument("--rank_ratio", type=float, default=1.0, help="FastSurvivalSVM rank_ratio (1=ranking)")

    ap.add_argument("--oof_folds", type=int, default=1, help=">=2 to compute OOF clinical on train")
    ap.add_argument("--use_oof_for_fuse", type=int, default=0, choices=[0, 1],
                    help="if 1, use OOF clinical train preds to learn fusion/normalization")

    # Step switches
    ap.add_argument("--use_residual", type=int, default=1, choices=[0, 1], help="Step B: residualize clinical on image")
    ap.add_argument("--use_gate", type=int, default=0, choices=[0, 1], help="Step D: train-only gate")
    ap.add_argument("--gate_margin", type=float, default=0.00, help="clinical must beat image by this margin on TRAIN to be used")

    # coefficient constraints (mostly for Cox fuse)
    ap.add_argument("--nonneg_fuse", type=int, default=1, choices=[0, 1], help="clamp fusion weights to >=0 (Cox fuse)")
    ap.add_argument("--normalize_beta", type=int, default=1, choices=[0, 1], help="normalize beta to sum=1 after clamping")

    ap.add_argument("--img_head_dropout", type=float, default=0.35)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--img_batch_size", type=int, default=512)

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

        X_tr, t_tr, e_tr, X_va, t_va, e_va, feat_names = build_clinical_features_for_fold(
            df=df,
            tr_pids=tr_pids,
            va_pids=va_pids,
            overall_stage_encoding=args.overall_stage_encoding
        )

        # image head
        head = FrozenImageHead(n_bins=n_bins, dropout=args.img_head_dropout)
        head.load_state_dict(head_state, strict=True)
        head.to(device).eval()

        img_risk_tr = compute_image_risk_from_frozen_head(tr_emb, head, bin_mids, device, batch_size=args.img_batch_size)
        img_risk_va = compute_image_risk_from_frozen_head(va_emb, head, bin_mids, device, batch_size=args.img_batch_size)

        y_tr_st = to_struct(t_tr, e_tr)
        y_va_st = to_struct(t_va, e_va)
        tau = safe_tau(t_tr, e_tr)

        # clinical Cox: fit on full train for VAL prediction
        cox = CoxPHSurvivalAnalysis(alpha=float(args.cox_alpha))
        cox.fit(X_tr, y_tr_st)
        clin_risk_tr_fit = cox.predict(X_tr).astype(np.float64)
        clin_risk_va = cox.predict(X_va).astype(np.float64)

        # optional OOF clinical risk on train (for fairer fusion learning/gating)
        clin_risk_tr_oof = None
        if args.oof_folds >= 2:
            clin_risk_tr_oof = fit_cox_oof(X_tr, y_tr_st, alpha=float(args.cox_alpha),
                                           oof_folds=int(args.oof_folds), seed=int(args.seed + 1000 + fold))

        # choose which train clinical risk to use downstream
        clin_risk_tr_use = clin_risk_tr_fit
        if args.use_oof_for_fuse == 1:
            if clin_risk_tr_oof is None:
                raise ValueError("--use_oof_for_fuse=1 requires --oof_folds>=2")
            clin_risk_tr_use = clin_risk_tr_oof

        # z-score normalize (TRAIN stats only)
        img_tr_z, img_va_z = zscore_by_train(img_risk_tr, img_risk_va)
        clin_tr_z, clin_va_z = zscore_by_train(clin_risk_tr_use, clin_risk_va)

        # Step A: corr print
        corr_tr = pearson_corr(img_tr_z, clin_tr_z)

        # base branch scores on VAL
        uno_img = eval_uno_or_harrell(y_tr_st, y_va_st, t_va, e_va, img_va_z, tau)
        uno_clin = eval_uno_or_harrell(y_tr_st, y_va_st, t_va, e_va, clin_va_z, tau)

        # Step B: residualization (clinical on image), then re-zscore residual for stability
        slope_k = 0.0
        if args.use_residual == 1:
            clin_res_tr, clin_res_va, slope_k = residualize_clin_on_img(img_tr_z, clin_tr_z, img_va_z, clin_va_z)
            clin_res_tr_z, clin_res_va_z = zscore_by_train(clin_res_tr, clin_res_va)
            x2_tr = clin_res_tr_z
            x2_va = clin_res_va_z
        else:
            x2_tr = clin_tr_z
            x2_va = clin_va_z

        # Step D: train-only gate (use OOF clinical if available)
        gate_used = False
        if args.use_gate == 1:
            # image train risk (fixed head) is not overfit; clinical: prefer OOF if available
            clin_for_gate = clin_tr_z
            if clin_risk_tr_oof is not None:
                # recompute zscore using OOF train stats but keep same mapping
                clin_oof_z, _ = zscore_by_train(clin_risk_tr_oof, clin_risk_tr_oof)
                clin_for_gate = clin_oof_z

            c_tr_img = cindex_harrell(e_tr, t_tr, img_tr_z)
            c_tr_clin = cindex_harrell(e_tr, t_tr, clin_for_gate)

            # gate rule: only allow clinical into fusion if it beats image on TRAIN by margin
            if c_tr_clin + float(args.gate_margin) < c_tr_img:
                x2_tr = np.zeros_like(x2_tr)
                x2_va = np.zeros_like(x2_va)
                gate_used = True

        # build fusion design matrices
        X_fuse_tr = np.column_stack([img_tr_z, x2_tr]).astype(np.float64)
        X_fuse_va = np.column_stack([img_va_z, x2_va]).astype(np.float64)

        # Step C (optional): fuse model choice
        beta_img = np.nan
        beta_clin = np.nan
        alpha_rel = np.nan

        if args.fuse_model == "cox":
            fuse = CoxPHSurvivalAnalysis(alpha=float(args.fuse_alpha))
            fuse.fit(X_fuse_tr, y_tr_st)

            beta = fuse.coef_.astype(np.float64).copy()

            # clamp to non-negative (recommended for your “risk + risk” fusion)
            if args.nonneg_fuse == 1:
                beta = np.maximum(beta, 0.0)
                if beta.sum() < 1e-12:
                    beta = np.array([1.0, 0.0], dtype=np.float64)

            if args.normalize_beta == 1:
                s = float(beta.sum())
                if s > 1e-12:
                    beta = beta / s

            fused_risk_va = (X_fuse_va @ beta).astype(np.float64)

            beta_img, beta_clin = float(beta[0]), float(beta[1])
            alpha_rel = beta_img / (beta_img + beta_clin + 1e-12)

        elif args.fuse_model == "svm":
            # rank_ratio=1 => predict() returns risk scores (higher -> higher risk)
            fuse = FastSurvivalSVM(
                alpha=float(args.fuse_alpha),
                rank_ratio=float(args.rank_ratio),
                optimizer="avltree",
                random_state=int(args.seed + 2000 + fold),
                max_iter=50,
            )
            fuse.fit(X_fuse_tr, y_tr_st)
            fused_risk_va = fuse.predict(X_fuse_va).astype(np.float64)

            # report linear weights for interpretability (not forced non-negative)
            if hasattr(fuse, "coef_"):
                beta_img, beta_clin = float(fuse.coef_[0]), float(fuse.coef_[1])
                denom = abs(beta_img) + abs(beta_clin) + 1e-12
                alpha_rel = abs(beta_img) / denom

        else:
            raise ValueError("Unknown fuse_model")

        uno_fuse = eval_uno_or_harrell(y_tr_st, y_va_st, t_va, e_va, fused_risk_va, tau)

        print(f"\n--- Fold {fold} ---")
        print(f"[DEBUG] features ({X_tr.shape[1]}): {feat_names}")
        print(f"[Step A] corr_tr(img_z, clin_z) = {corr_tr:.3f}")
        if args.use_residual == 1:
            print(f"[Step B] residual slope k (clin~img) = {slope_k:.3f}")
        if args.use_gate == 1:
            print(f"[Step D] gate_used={gate_used}")
        print(f"Image-only Uno:    {uno_img:.4f}")
        print(f"Clinical-only Uno: {uno_clin:.4f}")
        print(f"Fused Uno:         {uno_fuse:.4f} | fuse_model={args.fuse_model} "
              f"| beta_img={beta_img:.3f}, beta_clin={beta_clin:.3f}, alpha_rel={alpha_rel:.2f}")

        rows.append({
            "fold": fold,
            "uno_img": uno_img,
            "uno_clin": uno_clin,
            "uno_fuse": uno_fuse,
            "corr_tr": corr_tr,
            "slope_k": slope_k,
            "gate_used": int(gate_used),
            "beta_img": beta_img,
            "beta_clin": beta_clin,
            "alpha_rel": alpha_rel,
            "tau": tau,
            "cox_alpha": float(args.cox_alpha),
            "fuse_alpha": float(args.fuse_alpha),
            "fuse_model": args.fuse_model,
            "use_residual": int(args.use_residual),
            "use_gate": int(args.use_gate),
            "oof_folds": int(args.oof_folds),
            "use_oof_for_fuse": int(args.use_oof_for_fuse),
            "n_feat": int(X_tr.shape[1]),
        })

        per_fold[f"fold_{fold}"] = rows[-1]

    df_res = pd.DataFrame(rows)
    df_res.to_csv(out_root / "results.csv", index=False)
    with open(out_root / "per_fold_details.json", "w") as f:
        json.dump(per_fold, f, indent=2)

    print("\n========== Summary ==========")
    print(df_res[["fold", "uno_img", "uno_clin", "uno_fuse", "corr_tr", "slope_k", "gate_used", "beta_img", "beta_clin", "alpha_rel"]])
    for col in ["uno_img", "uno_clin", "uno_fuse"]:
        m = df_res[col].mean()
        s = df_res[col].std()
        print(f"{col}: {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
