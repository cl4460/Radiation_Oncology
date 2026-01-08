#!/usr/bin/env python
# fusion_baseline.py

"""
Baseline fusion scaffold for clinical features and radiomics engineering.

This script intentionally outputs TWO evaluation modes:

(1) ORACLE (dev upper bound):
    - select alpha on outer-val via grid search
    - report on the SAME outer-val (optimistic; use for sanity check only)

(2) NESTED (paper-defensible):
    - select alpha ONLY inside outer-train via inner CV
    - evaluate once on outer-val (reduces selection bias)

Refs:
- Selection bias & nested CV discussion: Cawley & Talbot (JMLR 2010); scikit-learn nested CV tutorial.
- Uno-IPCW constraints on tau: scikit-survival docs.
"""

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        tau = max(mx * 0.5, 1e-6)
    return tau


def pick_np(path_a: Path, path_b: Path):
    if path_a.exists():
        return path_a
    if path_b.exists():
        return path_b
    raise FileNotFoundError(f"Missing {path_a} or {path_b}")


def minmax_by_train(train_arr: np.ndarray, test_arr: np.ndarray, eps: float = 1e-8):
    tr = np.asarray(train_arr, dtype=np.float64)
    te = np.asarray(test_arr, dtype=np.float64)
    mn = float(np.min(tr))
    mx = float(np.max(tr))
    if mx - mn < eps:
        return np.zeros_like(tr), np.zeros_like(te)
    return (tr - mn) / (mx - mn), (te - mn) / (mx - mn)


def zscore_by_train(train_arr: np.ndarray, test_arr: np.ndarray, eps: float = 1e-8):
    tr = np.asarray(train_arr, dtype=np.float64)
    te = np.asarray(test_arr, dtype=np.float64)
    mu = float(np.mean(tr))
    sd = float(np.std(tr))
    if sd < eps:
        sd = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def uno_or_censored(y_tr_st, y_te_st, e_te, t_te, risk_te, tau):
    """
    Try Uno-IPCW; if it fails, fall back to censored C-index.
    """
    try:
        return float(concordance_index_ipcw(y_tr_st, y_te_st, risk_te, tau=tau)[0])
    except Exception as ex:
        print(f"[WARN] Uno-IPCW failed -> fallback to censored C-index. Reason: {ex}")
        return float(concordance_index_censored(e_te.astype(bool), t_te, risk_te)[0])


def load_edges_and_head_state(best_pt: Path):
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)  # allow numpy objects
    edges = np.asarray(ckpt["cuts"], dtype=float)
    n_bins = int(ckpt.get("n_time_bins", len(edges) - 1))

    state = ckpt.get("state_dict", ckpt.get("model_state_dict", None))
    if state is None:
        raise KeyError("Checkpoint missing state_dict/model_state_dict")

    head_state = {k.replace("head.", "seq."): v for k, v in state.items() if k.startswith("head.")}
    if not head_state:
        alt = [k for k in state.keys() if "head" in k.lower()][:10]
        raise KeyError(f"Could not find head.* weights. Example keys: {alt}")
    return edges, n_bins, head_state


# -------------------- frozen head + risk --------------------
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
    logits = np.clip(logits, -50.0, 50.0) 
    surv = logits_to_surv_np(logits)
    return surv_to_risk_np(surv, bin_mids).astype(np.float64)


# -------------------- clinical feature builder --------------------
def build_clinical_features_for_fold(df: pd.DataFrame, tr_pids: list, va_pids: list, overall_stage_encoding: str):
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

    # hard alignment to pid order
    tr = tr.set_index("PatientID").loc[tr_pids].reset_index()
    va = va.set_index("PatientID").loc[va_pids].reset_index()

    t_tr = tr["Survival.time"].astype(float).to_numpy()
    e_tr = tr["deadstatus.event"].astype(int).to_numpy()
    t_va = va["Survival.time"].astype(float).to_numpy()
    e_va = va["deadstatus.event"].astype(int).to_numpy()

    feats_tr, feats_va, feat_names = [], [], []

    # age z
    age_tr_s = tr["age"].astype(float)
    mu = float(age_tr_s.mean(skipna=True))
    sigma = float(age_tr_s.std(skipna=True))
    sigma = sigma if sigma > 1e-8 else 1.0
    age_tr = age_tr_s.fillna(mu).to_numpy(dtype=np.float32)
    age_va = va["age"].astype(float).fillna(mu).to_numpy(dtype=np.float32)
    feats_tr.append(((age_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((age_va - mu) / sigma).astype(np.float32))
    feat_names.append("age_z")

    # gender 0/1 (defensive)
    gmap = {"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0}
    g_tr = tr["gender"].astype(str).str.strip().str.lower().map(gmap).to_numpy(dtype=np.float32)
    g_va = va["gender"].astype(str).str.strip().str.lower().map(gmap).to_numpy(dtype=np.float32)
    if np.isnan(g_tr).any() or np.isnan(g_va).any():
        bad = sorted(set(tr["gender"].astype(str).str.lower().unique()) | set(va["gender"].astype(str).str.lower().unique()))
        raise ValueError(f"Unexpected gender values found: {bad}")
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
        cats = ["I", "II", "IIIa", "IIIb"]
        for c in cats:
            feats_tr.append((os_tr == c).to_numpy(dtype=np.float32))
            feats_va.append((os_va == c).to_numpy(dtype=np.float32))
            feat_names.append(f"OverallStage_{c}")
    elif overall_stage_encoding == "ordinal":
        mapping = {"I": 1.0, "II": 2.0, "IIIa": 3.0, "IIIb": 4.0}
        os_tr_num = os_tr.map(mapping).to_numpy(dtype=np.float32)
        os_va_num = os_va.map(mapping).to_numpy(dtype=np.float32)
        if np.isnan(os_tr_num).any() or np.isnan(os_va_num).any():
            raise ValueError("Unexpected Overall.Stage values found after normalization.")
        mu = float(np.mean(os_tr_num))
        sigma = float(np.std(os_tr_num)) if float(np.std(os_tr_num)) > 1e-8 else 1.0
        feats_tr.append(((os_tr_num - mu) / sigma).astype(np.float32))
        feats_va.append(((os_va_num - mu) / sigma).astype(np.float32))
        feat_names.append("OverallStage_ord_z")
    else:
        raise ValueError("overall_stage_encoding must be 'ordinal' or 'onehot'")

    X_tr = np.stack(feats_tr, axis=1)
    X_va = np.stack(feats_va, axis=1)
    return X_tr, t_tr, e_tr, X_va, t_va, e_va, feat_names


# -------------------- alpha selection: nested (inner CV on outer-train) --------------------
def _safe_inner_splits(e: np.ndarray, requested: int):
    e = np.asarray(e).astype(int)
    c0 = int((e == 0).sum())
    c1 = int((e == 1).sum())
    max_splits = min(c0, c1)  # StratifiedKFold constraint
    if max_splits < 2:
        return 0
    return int(min(requested, max_splits))


def select_alpha_by_inner_cv(img_risk_tr_raw, X_tr, t_tr, e_tr,
                             alphas, cox_alpha, inner_folds, seed, norm_fn):
    """
    Proper nested selection:
    - for each inner split:
        fit Cox on inner-train
        predict clin risk on inner-train / inner-val
        normalize img + clin risks using inner-train only
        evaluate alpha on inner-val
    """
    t_tr = np.asarray(t_tr, dtype=float)
    e_tr = np.asarray(e_tr, dtype=int)
    y_all = to_struct(t_tr, e_tr)

    inner_k = _safe_inner_splits(e_tr, inner_folds)
    if inner_k < 2:
        return 1.0, [(float(a), float("nan")) for a in alphas]

    skf = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=seed)

    mean_scores = []
    for a in alphas:
        fold_scores = []
        for tr_idx, va_idx in skf.split(np.zeros(len(e_tr)), e_tr):
            # inner train/val labels
            y_tr = y_all[tr_idx]
            y_va = y_all[va_idx]
            t_tr_in, e_tr_in = t_tr[tr_idx], e_tr[tr_idx]
            t_va_in, e_va_in = t_tr[va_idx], e_tr[va_idx]

            # (1) fit Cox on inner-train only
            cox = CoxPHSurvivalAnalysis(alpha=float(cox_alpha))
            cox.fit(X_tr[tr_idx], y_tr)
            clin_tr_in = cox.predict(X_tr[tr_idx]).astype(np.float64)
            clin_va_in = cox.predict(X_tr[va_idx]).astype(np.float64)

            # (2) normalize using inner-train only
            img_tr_in_n, img_va_in_n = norm_fn(img_risk_tr_raw[tr_idx], img_risk_tr_raw[va_idx])
            clin_tr_in_n, clin_va_in_n = norm_fn(clin_tr_in, clin_va_in)

            fused_va = a * img_va_in_n + (1.0 - a) * clin_va_in_n
            tau_in = safe_tau(t_tr_in, e_tr_in)
            s = uno_or_censored(y_tr, y_va, e_va_in, t_va_in, fused_va, tau_in)
            fold_scores.append(float(s))

        mean_scores.append((float(a), float(np.mean(fold_scores))))

    # pick best (ties -> larger a for safety)
    best_a, best_s = 1.0, -1.0
    for a, s in mean_scores:
        if np.isnan(s):
            continue
        if (s > best_s + 1e-12) or (abs(s - best_s) <= 1e-12 and a > best_a):
            best_s, best_a = s, a

    return float(best_a), mean_scores


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=str, required=True)
    ap.add_argument("--clinical_csv", type=str, required=True)
    ap.add_argument("--out_name", type=str, default="late_fusion_baseline")

    ap.add_argument("--overall_stage_encoding", type=str, default="onehot", choices=["onehot", "ordinal"])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--cox_alpha", type=float, default=0.1)
    ap.add_argument("--alpha_step", type=float, default=0.05)

    ap.add_argument("--norm", type=str, default="minmax", choices=["minmax", "zscore"])
    ap.add_argument("--img_head_dropout", type=float, default=0.35)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--img_batch_size", type=int, default=512)

    ap.add_argument("--do_nested", type=int, default=1, choices=[0, 1],
                    help="1=also compute nested (inner-CV) alpha; 0=oracle only")
    ap.add_argument("--inner_folds", type=int, default=5)

    args = ap.parse_args()
    set_seed(args.seed)

    exp_dir = Path(args.exp_dir)
    out_root = exp_dir / args.out_name
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.clinical_csv)
    device = torch.device(args.device)

    alphas = np.round(np.arange(0.0, 1.0 + 1e-9, args.alpha_step), 10)
    norm_fn = minmax_by_train if args.norm == "minmax" else zscore_by_train

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
            df=df, tr_pids=tr_pids, va_pids=va_pids, overall_stage_encoding=args.overall_stage_encoding
        )

        head = FrozenImageHead(n_bins=n_bins, dropout=args.img_head_dropout)
        head.load_state_dict(head_state, strict=True)
        head.to(device)
        head.eval()

        img_risk_tr = compute_image_risk_from_frozen_head(tr_emb, head, bin_mids, device, batch_size=args.img_batch_size)
        img_risk_va = compute_image_risk_from_frozen_head(va_emb, head, bin_mids, device, batch_size=args.img_batch_size)

        y_tr_st = to_struct(t_tr, e_tr)
        y_va_st = to_struct(t_va, e_va)
        tau = safe_tau(t_tr, e_tr)

        # clinical Cox (outer-train fit)
        cox = CoxPHSurvivalAnalysis(alpha=float(args.cox_alpha))
        cox.fit(X_tr, y_tr_st)
        clin_risk_tr = cox.predict(X_tr).astype(np.float64)
        clin_risk_va = cox.predict(X_va).astype(np.float64)

        # normalize (outer-train only)
        img_tr_n, img_va_n = norm_fn(img_risk_tr, img_risk_va)
        clin_tr_n, clin_va_n = norm_fn(clin_risk_tr, clin_risk_va)

        # unimodal on outer-val
        uno_img = uno_or_censored(y_tr_st, y_va_st, e_va, t_va, img_va_n, tau)
        uno_clin = uno_or_censored(y_tr_st, y_va_st, e_va, t_va, clin_va_n, tau)

        # ---------- ORACLE ----------
        best_alpha_oracle, best_uno_oracle = 1.0, -1.0
        alpha_curve_oracle = []
        for a in alphas:
            fused = a * img_va_n + (1.0 - a) * clin_va_n
            uno = uno_or_censored(y_tr_st, y_va_st, e_va, t_va, fused, tau)
            alpha_curve_oracle.append((float(a), float(uno)))
            if uno > best_uno_oracle:
                best_uno_oracle = float(uno)
                best_alpha_oracle = float(a)

        # ---------- NESTED (FIXED: respect --do_nested) ----------
        best_alpha_nested = float("nan")
        best_uno_nested = float("nan")
        alpha_curve_nested = None

        if int(args.do_nested) == 1:
            best_alpha_nested, alpha_curve_nested = select_alpha_by_inner_cv(
                img_risk_tr_raw=img_risk_tr,      # raw
                X_tr=X_tr,
                t_tr=t_tr,
                e_tr=e_tr,
                alphas=alphas,
                cox_alpha=float(args.cox_alpha),
                inner_folds=int(args.inner_folds),
                seed=int(args.seed + 9000 + fold),
                norm_fn=norm_fn,
            )
            fused_va_nested = best_alpha_nested * img_va_n + (1.0 - best_alpha_nested) * clin_va_n
            best_uno_nested = uno_or_censored(y_tr_st, y_va_st, e_va, t_va, fused_va_nested, tau)

        print(f"\n--- Fold {fold} ---")
        print(f"[DEBUG] features ({X_tr.shape[1]}): {feat_names}")
        print(f"Image-only Uno:    {uno_img:.4f}")
        print(f"Clinical-only Uno: {uno_clin:.4f}")
        print(f"Fusion ORACLE:     {best_uno_oracle:.4f} @ alpha={best_alpha_oracle:.2f}  (norm={args.norm})")
        if int(args.do_nested) == 1:
            print(f"Fusion NESTED:     {best_uno_nested:.4f} @ alpha={best_alpha_nested:.2f}  (inner_folds={args.inner_folds})")

        row = {
            "fold": fold,
            "uno_img": float(uno_img),
            "uno_clin": float(uno_clin),
            "best_alpha_oracle": float(best_alpha_oracle),
            "best_uno_oracle": float(best_uno_oracle),
            "tau": float(tau),
            "cox_alpha": float(args.cox_alpha),
            "norm": str(args.norm),
            "n_feat": int(X_tr.shape[1]),
        }
        if int(args.do_nested) == 1:
            row.update({
                "best_alpha_nested": float(best_alpha_nested),
                "best_uno_nested": float(best_uno_nested),
                "inner_folds": int(args.inner_folds),
            })
        rows.append(row)

        per_fold[f"fold_{fold}"] = {
            "feat_names": feat_names,
            "alpha_curve_oracle": alpha_curve_oracle,
            "best_alpha_oracle": float(best_alpha_oracle),
            "best_uno_oracle": float(best_uno_oracle),
            "uno_img": float(uno_img),
            "uno_clin": float(uno_clin),
            "tau": float(tau),
            "norm": str(args.norm),
        }
        if int(args.do_nested) == 1:
            per_fold[f"fold_{fold}"].update({
                "alpha_curve_nested_mean": alpha_curve_nested,
                "best_alpha_nested": float(best_alpha_nested),
                "best_uno_nested": float(best_uno_nested),
                "inner_folds": int(args.inner_folds),
            })

    df_res = pd.DataFrame(rows)
    df_res.to_csv(out_root / "results.csv", index=False)
    with open(out_root / "per_fold_details.json", "w") as f:
        json.dump(per_fold, f, indent=2)

    print("\n========== Summary ==========")
    print(df_res)

    def _summ(col):
        if col not in df_res.columns:
            return
        m = float(df_res[col].mean())
        s = float(df_res[col].std())
        print(f"{col}: {m:.4f} ± {s:.4f}")

    _summ("uno_img")
    _summ("uno_clin")
    _summ("best_uno_oracle")
    if "best_uno_nested" in df_res.columns:
        _summ("best_uno_nested")


if __name__ == "__main__":
    main()


#run command:
#python fusion_baseline.py   
#--exp_dir /home/lichengze/Research/DeepFeature/myresearch/phase3_outputs/lr_6.2e-4  
#--clinical_csv /home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv  
#--out_name late_fusion_final_baseline  
#--overall_stage_encoding onehot  
#--cox_alpha 0.1  
#--alpha_step 0.01  
#--norm minmax  
#--do_nested 1  
#--inner_folds 5  
#--seed 42