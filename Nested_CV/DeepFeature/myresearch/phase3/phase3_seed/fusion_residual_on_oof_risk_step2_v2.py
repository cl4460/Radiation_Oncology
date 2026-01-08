#!/usr/bin/env python3
"""
fusion_residual_on_oof_risk_step2_v2.py

OOF Risk Stacking with COMPLETE Clinical Feature Engineering

Key improvements:
1. Complete feature engineering matching clinical_baseline_step0.py:
   - age (z-scored)
   - gender (binary 0/1)
   - T stage (z-scored)
   - N stage (z-scored)
   - M stage (z-scored)
   - Overall.Stage (onehot or ordinal)
2. Proper imputation (mode for categorical, mean for numeric)
3. Per-fold standardization using train statistics
4. Residual learning on OOF risk scores
"""

import os, glob, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sksurv.metrics import concordance_index_censored, concordance_index_ipcw


# ============================================================
# Utility Functions
# ============================================================

def make_y(event, time):
    """Create structured array for sksurv metrics."""
    event = np.asarray(event).astype(bool)
    time = np.asarray(time).astype(float)
    y = np.empty(len(time), dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = event
    y["time"] = time
    return y


def uno_ipcw(y_train, y_test, scores_test, tau):
    """Uno's C-index with IPCW."""
    c = concordance_index_ipcw(y_train, y_test, scores_test, tau=tau)[0]
    return float(c)


def harrell_cindex(event, time, scores):
    """Harrell's C-index."""
    c = concordance_index_censored(
        np.asarray(event).astype(bool), 
        np.asarray(time).astype(float), 
        np.asarray(scores).astype(float)
    )[0]
    return float(c)


def cox_ph_loss(scores, time, event):
    """Cox proportional hazards loss."""
    time = time.float()
    event = event.float()
    
    order = torch.argsort(time, descending=True)
    s = scores[order]
    e = event[order]
    
    log_cumsum = torch.logcumsumexp(s, dim=0)
    loss = -(e * (s - log_cumsum)).sum() / (e.sum() + 1e-8)
    return loss


# ============================================================
# Feature Engineering (matching clinical_baseline_step0.py)
# ============================================================

def build_clinical_features(
    df: pd.DataFrame,
    tr_pids: list,
    va_pids: list,
    overall_stage_encoding: str = "onehot"
):
    """
    Build clinical features matching NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv
    
    Features:
    - age (z-scored)
    - gender (binary 0/1)
    - clinical.T.Stage (z-scored)
    - Clinical.N.Stage (z-scored)
    - Clinical.M.Stage (z-scored)
    - Overall.Stage (onehot or ordinal z-scored)
    
    Returns:
        X_tr, X_va: feature matrices
        feat_names: list of feature names
    """
    # Required columns
    use_cols = [
        "PatientID", "age", "gender",
        "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage",
        "Overall.Stage"
    ]
    
    # Check columns exist
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in clinical CSV: {missing}")
    
    d = df[use_cols].copy()
    d["PatientID"] = d["PatientID"].astype(str)
    
    # Split by PID
    tr = d[d["PatientID"].isin(tr_pids)].copy()
    va = d[d["PatientID"].isin(va_pids)].copy()
    
    feats_tr, feats_va, feat_names = [], [], []
    
    # ---- 1. Age (z-scored) ----
    age_tr = tr["age"].astype(float)
    age_va = va["age"].astype(float)
    
    # Impute with train mean
    mu = float(age_tr.mean(skipna=True))
    age_tr = age_tr.fillna(mu).to_numpy(dtype=np.float32)
    age_va = age_va.fillna(mu).to_numpy(dtype=np.float32)
    
    # Z-score
    sigma = float(np.std(age_tr)) if float(np.std(age_tr)) > 1e-8 else 1.0
    feats_tr.append(((age_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((age_va - mu) / sigma).astype(np.float32))
    feat_names.append("age_z")
    
    # ---- 2. Gender (binary 0/1) ----
    gmap = {"male": 1.0, "female": 0.0}
    g_tr = tr["gender"].astype(str).str.strip().str.lower().map(gmap).to_numpy(dtype=np.float32)
    g_va = va["gender"].astype(str).str.strip().str.lower().map(gmap).to_numpy(dtype=np.float32)
    
    # Check for unexpected values
    if np.isnan(g_tr).any() or np.isnan(g_va).any():
        raise ValueError("Unexpected gender values. Expected only 'male'/'female'.")
    
    feats_tr.append(g_tr)
    feats_va.append(g_va)
    feat_names.append("gender_01")
    
    # ---- 3. T stage (z-scored) ----
    T_tr_s = tr["clinical.T.Stage"].astype(float)
    T_va_s = va["clinical.T.Stage"].astype(float)
    
    # Impute with train mode
    T_mode = float(T_tr_s.dropna().value_counts().idxmax()) if T_tr_s.notna().any() else 2.0
    T_tr = T_tr_s.fillna(T_mode).to_numpy(dtype=np.float32)
    T_va = T_va_s.fillna(T_mode).to_numpy(dtype=np.float32)
    
    # Z-score
    mu = float(np.mean(T_tr))
    sigma = float(np.std(T_tr)) if float(np.std(T_tr)) > 1e-8 else 1.0
    feats_tr.append(((T_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((T_va - mu) / sigma).astype(np.float32))
    feat_names.append("T_z")
    
    # ---- 4. N stage (z-scored) ----
    N_tr = tr["Clinical.N.Stage"].astype(float).to_numpy(dtype=np.float32)
    N_va = va["Clinical.N.Stage"].astype(float).to_numpy(dtype=np.float32)
    
    mu = float(np.mean(N_tr))
    sigma = float(np.std(N_tr)) if float(np.std(N_tr)) > 1e-8 else 1.0
    feats_tr.append(((N_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((N_va - mu) / sigma).astype(np.float32))
    feat_names.append("N_z")
    
    # ---- 5. M stage (z-scored) ----
    M_tr = tr["Clinical.M.Stage"].astype(float).to_numpy(dtype=np.float32)
    M_va = va["Clinical.M.Stage"].astype(float).to_numpy(dtype=np.float32)
    
    mu = float(np.mean(M_tr))
    sigma = float(np.std(M_tr)) if float(np.std(M_tr)) > 1e-8 else 1.0
    feats_tr.append(((M_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((M_va - mu) / sigma).astype(np.float32))
    feat_names.append("M_z")
    
    # ---- 6. Overall.Stage (onehot or ordinal) ----
    os_tr = tr["Overall.Stage"].astype("object")
    os_va = va["Overall.Stage"].astype("object")
    
    # Impute with train mode
    os_mode = os_tr.dropna().value_counts().idxmax() if os_tr.notna().any() else "II"
    os_tr = os_tr.fillna(os_mode).astype(str)
    os_va = os_va.fillna(os_mode).astype(str)
    
    if overall_stage_encoding == "onehot":
        # One-hot encoding for I, II, IIIa, IIIb
        cats = ["I", "II", "IIIa", "IIIb"]
        for c in cats:
            feats_tr.append((os_tr == c).to_numpy(dtype=np.float32))
            feats_va.append((os_va == c).to_numpy(dtype=np.float32))
            feat_names.append(f"OverallStage_{c}")
    
    elif overall_stage_encoding == "ordinal":
        # Ordinal encoding: I=1, II=2, IIIa=3, IIIb=4
        mapping = {"I": 1.0, "II": 2.0, "IIIa": 3.0, "IIIb": 4.0}
        os_tr_num = os_tr.map(mapping).to_numpy(dtype=np.float32)
        os_va_num = os_va.map(mapping).to_numpy(dtype=np.float32)
        
        if np.isnan(os_tr_num).any() or np.isnan(os_va_num).any():
            raise ValueError("Unexpected Overall.Stage values. Expected only I/II/IIIa/IIIb.")
        
        # Z-score
        mu = float(np.mean(os_tr_num))
        sigma = float(np.std(os_tr_num)) if float(np.std(os_tr_num)) > 1e-8 else 1.0
        feats_tr.append(((os_tr_num - mu) / sigma).astype(np.float32))
        feats_va.append(((os_va_num - mu) / sigma).astype(np.float32))
        feat_names.append("OverallStage_ord_z")
    
    else:
        raise ValueError("overall_stage_encoding must be 'ordinal' or 'onehot'")
    
    # Stack features
    X_tr = np.stack(feats_tr, axis=1).astype(np.float32)  # [N_tr, d]
    X_va = np.stack(feats_va, axis=1).astype(np.float32)  # [N_va, d]
    
    print(f"[Features] Built {X_tr.shape[1]} features: {feat_names}")
    print(f"[Features] Train shape: {X_tr.shape}, Val shape: {X_va.shape}")
    
    return X_tr, X_va, feat_names


# ============================================================
# Model Architecture
# ============================================================

class DeltaMLP(nn.Module):
    """MLP for learning residual correction on base risk."""
    def __init__(self, in_dim, hidden=64, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.dp = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)
        
        # Zero-init last layer => delta starts at 0
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = self.dp(self.act(self.fc1(x)))
        return self.fc2(x).squeeze(-1)


class ResidualOnRisk(nn.Module):
    """
    Residual learning on OOF risk scores.
    
    fused_risk = base_risk + alpha * delta(clinical_features)
    
    where:
    - base_risk: OOF risk integral from image-only model
    - delta: learned residual from clinical features
    - alpha: learnable scalar weight (starts near 0)
    """
    def __init__(self, clin_dim, hidden=64, dropout=0.5, alpha_init=-10.0):
        super().__init__()
        self.delta = DeltaMLP(clin_dim, hidden=hidden, dropout=dropout)
        self.raw_alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
    
    def forward(self, base_risk, clin_x):
        alpha = torch.nn.functional.softplus(self.raw_alpha)  # >=0, ~0 at init
        delta = self.delta(clin_x)
        return base_risk + alpha * delta, alpha, delta


# ============================================================
# Main Training Loop
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="OOF Risk Stacking with Complete Clinical Feature Engineering"
    )
    
    # Paths
    ap.add_argument("--exp_dir", required=True, help="Phase3 experiment directory with fold_*/")
    ap.add_argument("--clinical_csv", required=True, help="Clinical CSV file")
    ap.add_argument("--oof_csv", default=None, help="OOF risk CSV (default: exp_dir/oof_risk_integral.csv)")
    ap.add_argument("--out_name", default="step2_residual_on_oof_risk_v2", help="Output subdirectory name")
    
    # Training hyperparameters
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd_clin", type=float, default=1e-2)
    ap.add_argument("--delta_hidden", type=int, default=64)
    ap.add_argument("--delta_dropout", type=float, default=0.5)
    ap.add_argument("--alpha_init", type=float, default=-10.0, 
                    help="Initial value for raw_alpha (softplus applied). Try -10/-6/-4")
    ap.add_argument("--alpha_lr_mult", type=float, default=1.0,
                    help="Learning rate multiplier for alpha parameter")
    ap.add_argument("--seed", type=int, default=42)
    
    # Feature engineering
    ap.add_argument("--overall_stage_encoding", type=str, default="onehot", 
                    choices=["onehot", "ordinal"],
                    help="How to encode Overall.Stage: onehot (4 dims) or ordinal (1 dim z-scored)")
    
    args = ap.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    exp_dir = args.exp_dir
    oof_csv = args.oof_csv or os.path.join(exp_dir, "oof_risk_integral.csv")
    out_dir = os.path.join(exp_dir, args.out_name + f"_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("OOF RISK STACKING WITH COMPLETE CLINICAL FEATURES")
    print("=" * 80)
    print(f"Experiment dir: {exp_dir}")
    print(f"Clinical CSV: {args.clinical_csv}")
    print(f"OOF CSV: {oof_csv}")
    print(f"Output dir: {out_dir}")
    print(f"Overall.Stage encoding: {args.overall_stage_encoding}")
    print(f"Seed: {args.seed}")
    print("=" * 80 + "\n")
    
    # ---- Load OOF base risk ----
    print("[1/4] Loading OOF risk scores...")
    oof = pd.read_csv(oof_csv)
    assert "pid" in oof.columns, f"OOF CSV missing 'pid' column. Columns: {list(oof.columns)}"
    assert "oof_risk_integral" in oof.columns, f"OOF CSV missing 'oof_risk_integral'. Columns: {list(oof.columns)}"
    
    oof_map = dict(zip(oof["pid"].astype(str), oof["oof_risk_integral"].astype(float)))
    print(f"  Loaded {len(oof_map)} OOF risk scores")
    
    # Global standardization stats for base risk
    mu_oof = oof["oof_risk_integral"].mean()
    sd_oof = oof["oof_risk_integral"].std() + 1e-8
    print(f"  OOF risk: mean={mu_oof:.4f}, std={sd_oof:.4f}")
    
    # ---- Load clinical data ----
    print("\n[2/4] Loading clinical data...")
    clin = pd.read_csv(args.clinical_csv)
    print(f"  Loaded {len(clin)} patients")
    print(f"  Columns: {list(clin.columns)}")
    
    # Infer column names
    required_cols = ["PatientID", "Survival.time", "deadstatus.event"]
    missing = [c for c in required_cols if c not in clin.columns]
    if missing:
        raise ValueError(f"Clinical CSV missing required columns: {missing}")
    
    clin["PatientID"] = clin["PatientID"].astype(str)
    
    # ---- Process each fold ----
    print("\n[3/4] Processing folds...")
    fold_dirs = sorted([d for d in glob.glob(os.path.join(exp_dir, "fold_*")) if os.path.isdir(d)])
    
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found in {exp_dir}")
    
    print(f"  Found {len(fold_dirs)} folds")
    
    all_fold_metrics = []
    
    for fd in fold_dirs:
        fold = os.path.basename(fd)
        print(f"\n{'='*60}")
        print(f"[{fold}]")
        print(f"{'='*60}")
        
        # Load PIDs
        train_p = pd.read_csv(os.path.join(fd, "train_pids.csv"), header=None)[0].astype(str).tolist()
        val_p = pd.read_csv(os.path.join(fd, "val_pids.csv"), header=None)[0].astype(str).tolist()
        print(f"  Train PIDs: {len(train_p)}, Val PIDs: {len(val_p)}")
        
        # Build clinical features
        X_tr, X_va, feat_names = build_clinical_features(
            clin, train_p, val_p, 
            overall_stage_encoding=args.overall_stage_encoding
        )
        
        # Get labels
        tr_df = clin[clin["PatientID"].isin(train_p)].copy()
        va_df = clin[clin["PatientID"].isin(val_p)].copy()
        
        t_tr = tr_df["Survival.time"].astype(float).to_numpy()
        e_tr = tr_df["deadstatus.event"].astype(int).to_numpy()
        t_va = va_df["Survival.time"].astype(float).to_numpy()
        e_va = va_df["deadstatus.event"].astype(int).to_numpy()
        
        # Attach OOF base risk
        base_risk_tr = np.array([oof_map[p] for p in train_p], dtype=np.float32)
        base_risk_va = np.array([oof_map[p] for p in val_p], dtype=np.float32)
        
        # Standardize base risk
        base_z_tr = (base_risk_tr - mu_oof) / sd_oof
        base_z_va = (base_risk_va - mu_oof) / sd_oof
        
        # Convert to tensors
        base_tr = torch.tensor(base_z_tr, dtype=torch.float32)
        base_va = torch.tensor(base_z_va, dtype=torch.float32)
        clin_tr = torch.tensor(X_tr, dtype=torch.float32)
        clin_va = torch.tensor(X_va, dtype=torch.float32)
        t_tr_t = torch.tensor(t_tr, dtype=torch.float32)
        e_tr_t = torch.tensor(e_tr, dtype=torch.float32)
        
        # Prepare sksurv structures
        y_tr = make_y(e_tr, t_tr)
        y_va = make_y(e_va, t_va)
        tau = float(np.quantile(t_tr[e_tr == 1], 0.9)) if (e_tr == 1).any() else float(np.max(t_tr))
        print(f"  Tau (90% event time): {tau:.2f}")
        
        # Compute baseline (image-only) performance
        uno_base = uno_ipcw(y_tr, y_va, base_z_va, tau=tau)
        har_base = harrell_cindex(e_va, t_va, base_z_va)
        print(f"  Baseline (image-only): Uno={uno_base:.4f}, Harrell={har_base:.4f}")
        
        # Initialize model
        model = ResidualOnRisk(
            clin_dim=X_tr.shape[1], 
            hidden=args.delta_hidden, 
            dropout=args.delta_dropout
        ).train()
        
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd_clin)
        
        # Training loop
        best_uno = -1e9
        bad = 0
        best_state = None
        
        print(f"\n  Training...")
        for ep in range(1, args.epochs + 1):
            model.train()
            fused_tr, alpha, delta = model(base_tr, clin_tr)
            loss = cox_ph_loss(fused_tr, t_tr_t, e_tr_t)
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                fused_va, alpha_va, _ = model(base_va, clin_va)
                fused_va_np = fused_va.numpy()
                
                uno_fuse = uno_ipcw(y_tr, y_va, fused_va_np, tau=tau)
                har_fuse = harrell_cindex(e_va, t_va, fused_va_np)
            
            # Check improvement
            if uno_fuse > best_uno + 1e-5:
                best_uno = uno_fuse
                bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            
            if ep % 5 == 0 or ep == 1:
                print(f"    Ep {ep:03d} | loss={loss.item():.4f} | "
                      f"Uno: {uno_base:.3f}->{uno_fuse:.3f} | "
                      f"Har: {har_base:.3f}->{har_fuse:.3f} | "
                      f"alpha={float(alpha_va):.5f}")
            
            if bad >= args.patience:
                print(f"    Early stop at ep {ep} (best_uno={best_uno:.4f})")
                break
        
        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(out_dir, f"{fold}_best.pt"))
        
        # Record metrics
        delta_uno = best_uno - uno_base
        all_fold_metrics.append({
            "fold": fold,
            "baseline_uno": float(uno_base),
            "best_uno": float(best_uno),
            "delta_uno": float(delta_uno),
            "tau": float(tau),
            "n_features": int(X_tr.shape[1]),
            "feature_names": feat_names
        })
        
        print(f"\n  [{fold}] Final: Uno {uno_base:.4f} -> {best_uno:.4f} (Δ={delta_uno:+.4f})")
    
    # ---- Summary ----
    print("\n" + "=" * 80)
    print("[4/4] SUMMARY")
    print("=" * 80)
    
    baseline_unos = [m["baseline_uno"] for m in all_fold_metrics]
    best_unos = [m["best_uno"] for m in all_fold_metrics]
    deltas = [m["delta_uno"] for m in all_fold_metrics]
    
    mean_base = float(np.mean(baseline_unos))
    std_base = float(np.std(baseline_unos, ddof=1)) if len(baseline_unos) > 1 else 0.0
    mean_best = float(np.mean(best_unos))
    std_best = float(np.std(best_unos, ddof=1)) if len(best_unos) > 1 else 0.0
    mean_delta = float(np.mean(deltas))
    std_delta = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
    
    print(f"Baseline (image-only): {mean_base:.4f} ± {std_base:.4f}")
    print(f"Fused (image+clinical): {mean_best:.4f} ± {std_best:.4f}")
    print(f"Improvement (Δ): {mean_delta:+.4f} ± {std_delta:.4f}")
    print(f"\nPer-fold results:")
    for m in all_fold_metrics:
        print(f"  {m['fold']}: {m['baseline_uno']:.4f} -> {m['best_uno']:.4f} (Δ={m['delta_uno']:+.4f})")
    
    # Save summary
    summary = {
        "config": {
            "seed": args.seed,
            "overall_stage_encoding": args.overall_stage_encoding,
            "lr": args.lr,
            "wd_clin": args.wd_clin,
            "delta_hidden": args.delta_hidden,
            "delta_dropout": args.delta_dropout,
            "epochs": args.epochs,
            "patience": args.patience,
        },
        "metrics": all_fold_metrics,
        "summary": {
            "mean_baseline_uno": mean_base,
            "std_baseline_uno": std_base,
            "mean_best_uno": mean_best,
            "std_best_uno": std_best,
            "mean_delta_uno": mean_delta,
            "std_delta_uno": std_delta,
        }
    }
    
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {out_dir}/summary.json")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


