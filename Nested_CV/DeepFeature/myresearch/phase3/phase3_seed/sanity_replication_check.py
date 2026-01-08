#!/usr/bin/env python
# sanity_replication_check.py

import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sksurv.metrics import concordance_index_ipcw, concordance_index_censored


# -----------------------
# utils
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_struct(times, events):
    return np.array([(bool(e), float(t)) for t, e in zip(times, events)],
                    dtype=[("event", bool), ("time", float)])


def read_pid_list(path: Path):
    return pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()


def pick_np(path_a: Path, path_b: Path):
    if path_a.exists():
        return path_a
    if path_b.exists():
        return path_b
    raise FileNotFoundError(f"Missing both {path_a} and {path_b}")


def logits_to_surv_np(logits: np.ndarray):
    # logistic hazard -> survival
    haz = 1.0 / (1.0 + np.exp(-logits))
    haz = np.clip(haz, 1e-7, 1.0 - 1e-7)
    log_surv = np.cumsum(np.log(1.0 - haz), axis=1)
    return np.exp(log_surv)


def load_edges_and_head_state(best_pt: Path):
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    edges = np.asarray(ckpt["cuts"], dtype=float)
    n_bins = int(ckpt.get("n_time_bins", len(edges) - 1))
    if len(edges) != n_bins + 1:
        raise ValueError(f"cuts length ({len(edges)}) != n_time_bins+1 ({n_bins+1}).")

    state = ckpt.get("state_dict", ckpt.get("model_state_dict", None))
    if state is None:
        raise ValueError("Checkpoint missing state_dict/model_state_dict.")

    head_state = {k: v for k, v in state.items() if k.startswith("head.")}
    if not head_state:
        raise ValueError("No head.* keys found. Your imageonly model must have self.head = nn.Sequential(...).")

    return edges, n_bins, head_state


# -----------------------
# models
# -----------------------
class FrozenImageHead(nn.Module):
    """
    Must EXACTLY match your phase3_imageonly head:
      Linear(512,256) -> LN -> ReLU -> Dropout ->
      Linear(256,128) -> LN -> ReLU -> Dropout ->
      Linear(128,n_bins)
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

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.head(emb)


class DeltaMLP(nn.Module):
    def __init__(self, in_dim: int, n_bins: int, hidden: int = 64, dropout: float = 0.5, zero_init_last: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, n_bins)

        if zero_init_last:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


class ResidualConcatFusion(nn.Module):
    def __init__(self, frozen_head: FrozenImageHead, clin_dim: int, n_bins: int,
                 delta_hidden: int = 64, delta_dropout: float = 0.5):
        super().__init__()
        self.frozen_head = frozen_head
        for p in self.frozen_head.parameters():
            p.requires_grad = False

        self.delta = DeltaMLP(
            in_dim=512 + clin_dim,
            n_bins=n_bins,
            hidden=delta_hidden,
            dropout=delta_dropout,
            zero_init_last=True,   # IMPORTANT for sanity delta=0 equivalence check
        )

    def forward(self, emb, clin):
        with torch.no_grad():
            img_logits = self.frozen_head(emb)
        delta_logits = self.delta(torch.cat([emb, clin], dim=1))
        return img_logits + delta_logits


# -----------------------
# clinical features (same as your step2)
# -----------------------
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

    # IMPORTANT: enforce same order as pid lists
    tr = tr.set_index("PatientID").loc[tr_pids].reset_index()
    va = va.set_index("PatientID").loc[va_pids].reset_index()

    t_tr = tr["Survival.time"].astype(float).to_numpy()
    e_tr = tr["deadstatus.event"].astype(int).to_numpy()
    t_va = va["Survival.time"].astype(float).to_numpy()
    e_va = va["deadstatus.event"].astype(int).to_numpy()

    feats_tr, feats_va = [], []

    # age z
    age_tr = tr["age"].astype(float)
    age_va = va["age"].astype(float)
    mu = float(age_tr.mean(skipna=True))
    age_tr = age_tr.fillna(mu).to_numpy(dtype=np.float32)
    age_va = age_va.fillna(mu).to_numpy(dtype=np.float32)
    sigma = float(np.std(age_tr)) if float(np.std(age_tr)) > 1e-8 else 1.0
    feats_tr.append(((age_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((age_va - mu) / sigma).astype(np.float32))

    # gender 0/1
    gmap = {"male": 1.0, "female": 0.0}
    g_tr = tr["gender"].astype(str).str.strip().str.lower().map(gmap).to_numpy(dtype=np.float32)
    g_va = va["gender"].astype(str).str.strip().str.lower().map(gmap).to_numpy(dtype=np.float32)
    if np.isnan(g_tr).any() or np.isnan(g_va).any():
        raise ValueError("Unexpected gender values found.")
    feats_tr.append(g_tr); feats_va.append(g_va)

    # T z (fill NaN with train mode)
    T_tr_s = tr["clinical.T.Stage"].astype(float)
    T_va_s = va["clinical.T.Stage"].astype(float)
    T_mode = float(T_tr_s.dropna().value_counts().idxmax())
    T_tr = T_tr_s.fillna(T_mode).to_numpy(dtype=np.float32)
    T_va = T_va_s.fillna(T_mode).to_numpy(dtype=np.float32)
    mu = float(np.mean(T_tr)); sigma = float(np.std(T_tr)) if float(np.std(T_tr)) > 1e-8 else 1.0
    feats_tr.append(((T_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((T_va - mu) / sigma).astype(np.float32))

    # N z
    N_tr = tr["Clinical.N.Stage"].astype(float).to_numpy(dtype=np.float32)
    N_va = va["Clinical.N.Stage"].astype(float).to_numpy(dtype=np.float32)
    mu = float(np.mean(N_tr)); sigma = float(np.std(N_tr)) if float(np.std(N_tr)) > 1e-8 else 1.0
    feats_tr.append(((N_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((N_va - mu) / sigma).astype(np.float32))

    # M z
    M_tr = tr["Clinical.M.Stage"].astype(float).to_numpy(dtype=np.float32)
    M_va = va["Clinical.M.Stage"].astype(float).to_numpy(dtype=np.float32)
    mu = float(np.mean(M_tr)); sigma = float(np.std(M_tr)) if float(np.std(M_tr)) > 1e-8 else 1.0
    feats_tr.append(((M_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((M_va - mu) / sigma).astype(np.float32))

    # Overall.Stage fill NaN with train mode
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
    elif overall_stage_encoding == "ordinal":
        mapping = {"I": 1.0, "II": 2.0, "IIIa": 3.0, "IIIb": 4.0}
        os_tr_num = os_tr.map(mapping).to_numpy(dtype=np.float32)
        os_va_num = os_va.map(mapping).to_numpy(dtype=np.float32)
        if np.isnan(os_tr_num).any() or np.isnan(os_va_num).any():
            raise ValueError("Unexpected Overall.Stage values found.")
        mu = float(np.mean(os_tr_num)); sigma = float(np.std(os_tr_num)) if float(np.std(os_tr_num)) > 1e-8 else 1.0
        feats_tr.append(((os_tr_num - mu) / sigma).astype(np.float32))
        feats_va.append(((os_va_num - mu) / sigma).astype(np.float32))
    else:
        raise ValueError("overall_stage_encoding must be 'ordinal' or 'onehot'")

    X_tr = np.stack(feats_tr, axis=1)
    X_va = np.stack(feats_va, axis=1)
    return X_tr, t_tr, e_tr, X_va, t_va, e_va


# -----------------------
# evaluation
# -----------------------
@torch.no_grad()
def infer_logits(head_or_model, emb_np, clin_np=None, device="cpu", batch_size=1024):
    head_or_model.eval()

    n = emb_np.shape[0]
    logits_list = []
    for i in range(0, n, batch_size):
        emb_b = torch.tensor(emb_np[i:i+batch_size], dtype=torch.float32, device=device)
        if clin_np is None:
            out = head_or_model(emb_b)
        else:
            clin_b = torch.tensor(clin_np[i:i+batch_size], dtype=torch.float32, device=device)
            out = head_or_model(emb_b, clin_b)
        logits_list.append(out.detach().cpu().numpy())
    return np.concatenate(logits_list, axis=0)


def risk_integral_from_logits(logits_np: np.ndarray, edges: np.ndarray):
    bin_mids = (edges[:-1] + edges[1:]) / 2.0
    surv = logits_to_surv_np(logits_np)
    risk_int = -np.trapz(surv, x=bin_mids, axis=1)
    return risk_int.astype(np.float64)


def compute_metrics(risk, t_tr, e_tr, t_va, e_va):
    y_tr_struct = to_struct(t_tr, e_tr)
    y_va_struct = to_struct(t_va, e_va)

    evt_times_tr = t_tr[e_tr == 1]
    tau = np.quantile(evt_times_tr, 0.9) if len(evt_times_tr) > 0 else float(np.max(t_tr))

    uno = concordance_index_ipcw(y_tr_struct, y_va_struct, risk, tau=tau)[0]
    harrell = concordance_index_censored(y_va_struct["event"], y_va_struct["time"], risk)[0]
    return float(uno), float(harrell), float(tau)


# -----------------------
# main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=str, required=True, help="phase3_outputs/<lr_xxx> containing fold_k/")
    ap.add_argument("--clinical_csv", type=str, required=True)

    ap.add_argument("--overall_stage_encoding", type=str, default="ordinal", choices=["ordinal", "onehot"])
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None)

    # optional: evaluate a step2 fusion run by loading delta_state_dict
    ap.add_argument("--fusion_out_dir", type=str, default=None,
                    help="If set, should point to exp_dir/<out_name> (contains fold_k/best.pt with delta_state_dict).")

    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_dir", type=str, default=None, help="Optional directory to dump per-fold predictions.")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_dir = Path(args.exp_dir)
    df = pd.read_csv(args.clinical_csv)

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    base_unos = []
    fusion_unos = []

    print(f"[Info] device={device}")
    print(f"[Info] exp_dir={exp_dir}")
    print(f"[Info] overall_stage_encoding={args.overall_stage_encoding}")
    if args.fusion_out_dir:
        print(f"[Info] fusion_out_dir={args.fusion_out_dir}")

    for fold in range(5):
        if args.fold is not None and fold != args.fold:
            continue

        fold_dir = exp_dir / f"fold_{fold}"
        best_pt = fold_dir / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"Missing {best_pt}")

        tr_emb_path = pick_np(fold_dir / "train_embeddings.npy", fold_dir / "train_embeddings_extracted.npy")
        va_emb_path = pick_np(fold_dir / "val_embeddings.npy", fold_dir / "val_embeddings_extracted.npy")
        tr_pids_path = fold_dir / "train_pids.csv"
        va_pids_path = fold_dir / "val_pids.csv"
        if not tr_pids_path.exists() or not va_pids_path.exists():
            raise FileNotFoundError("Need train_pids.csv and val_pids.csv in each fold dir.")

        tr_pids = read_pid_list(tr_pids_path)
        va_pids = read_pid_list(va_pids_path)

        emb_tr = np.load(tr_emb_path).astype(np.float32)
        emb_va = np.load(va_emb_path).astype(np.float32)
        if emb_tr.shape[1] != 512 or emb_va.shape[1] != 512:
            raise ValueError(f"Expected embeddings dim=512, got train {emb_tr.shape}, val {emb_va.shape}")

        edges, n_bins, head_state = load_edges_and_head_state(best_pt)

        # build clinical + survival arrays (ordered by pid list)
        X_tr, t_tr, e_tr, X_va, t_va, e_va = build_clinical_features_for_fold(
            df, tr_pids, va_pids, args.overall_stage_encoding
        )

        # ---- A) image-only replication ----
        head = FrozenImageHead(n_bins=n_bins, dropout=0.35)
        head.load_state_dict(head_state, strict=True)
        head.eval()
        head.to(device)

        logits_base = infer_logits(head, emb_va, clin_np=None, device=device, batch_size=args.batch_size)
        risk_base = risk_integral_from_logits(logits_base, edges)
        uno_base, har_base, tau = compute_metrics(risk_base, t_tr, e_tr, t_va, e_va)
        base_unos.append(uno_base)

        # ---- B) delta=0 equivalence (concat-residual, but delta output == 0) ----
        fusion0 = ResidualConcatFusion(
            frozen_head=head,
            clin_dim=X_va.shape[1],
            n_bins=n_bins,
            delta_hidden=64,
            delta_dropout=0.5
        ).to(device)
        fusion0.eval()

        logits_delta0 = infer_logits(fusion0, emb_va, clin_np=X_va, device=device, batch_size=args.batch_size)
        risk_delta0 = risk_integral_from_logits(logits_delta0, edges)

        max_abs_logit_diff = float(np.max(np.abs(logits_delta0 - logits_base)))
        max_abs_risk_diff = float(np.max(np.abs(risk_delta0 - risk_base)))

        # ---- C) optional: load step2 delta ckpt and replicate fused metrics ----
        uno_fused = None
        har_fused = None
        ckpt_best_uno = None
        if args.fusion_out_dir:
            fdir = Path(args.fusion_out_dir) / f"fold_{fold}"
            fpt = fdir / "best.pt"
            if not fpt.exists():
                raise FileNotFoundError(f"Missing fusion ckpt: {fpt}")
            fckpt = torch.load(fpt, map_location="cpu", weights_only=False)

            if "delta_state_dict" not in fckpt:
                raise ValueError(f"{fpt} missing delta_state_dict")

            fusion0.delta.load_state_dict(fckpt["delta_state_dict"], strict=True)
            fusion0.eval()

            logits_fused = infer_logits(fusion0, emb_va, clin_np=X_va, device=device, batch_size=args.batch_size)
            risk_fused = risk_integral_from_logits(logits_fused, edges)
            uno_fused, har_fused, _ = compute_metrics(risk_fused, t_tr, e_tr, t_va, e_va)
            fusion_unos.append(uno_fused)

            ckpt_best_uno = float(fckpt.get("best_uno", np.nan))

            if save_dir:
                pd.DataFrame({
                    "patient_id": va_pids,
                    "risk_base": risk_base,
                    "risk_delta0": risk_delta0,
                    "risk_fused": risk_fused,
                    "time": t_va,
                    "event": e_va
                }).to_csv(save_dir / f"fold_{fold}_preds.csv", index=False)

        print(
            f"[Fold {fold}] "
            f"BASE Uno={uno_base:.4f} Harrell={har_base:.4f} tau={tau:.1f} | "
            f"delta0 max|logit|={max_abs_logit_diff:.3e} max|risk|={max_abs_risk_diff:.3e}"
            + (f" | FUSED Uno={uno_fused:.4f} Harrell={har_fused:.4f} (ckpt best_uno={ckpt_best_uno:.4f})"
               if args.fusion_out_dir else "")
        )

        rows.append({
            "fold": fold,
            "uno_base": uno_base,
            "harrell_base": har_base,
            "tau": tau,
            "max_abs_logit_diff_delta0_vs_base": max_abs_logit_diff,
            "max_abs_risk_diff_delta0_vs_base": max_abs_risk_diff,
            "uno_fused": uno_fused,
            "harrell_fused": har_fused,
            "ckpt_best_uno": ckpt_best_uno
        })

    # summary
    print("\n=== Summary ===")
    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))

    print(f"\n[Mean] BASE Uno = {np.mean(base_unos):.4f} ± {np.std(base_unos):.4f}")
    if fusion_unos:
        print(f"[Mean] FUSED Uno = {np.mean(fusion_unos):.4f} ± {np.std(fusion_unos):.4f}")

    if save_dir:
        (save_dir / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
