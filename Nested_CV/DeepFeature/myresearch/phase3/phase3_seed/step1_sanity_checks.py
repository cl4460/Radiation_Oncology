#!/usr/bin/env python
# step1_sanity_checks.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sksurv.metrics import concordance_index_ipcw, concordance_index_censored

# -----------------------------
# Helpers
# -----------------------------
def to_struct(times, events):
    return np.array([(bool(e), float(t)) for t, e in zip(times, events)],
                    dtype=[("event", bool), ("time", float)])

def read_pid_list(csv_path: Path):
    # supports headerless single-column csv (your val_pids.csv format)
    s = pd.read_csv(csv_path, header=None).iloc[:, 0].astype(str)
    return s.tolist()

def load_clinical_minimal(csv_path: Path):
    """
    Tailored to your file: NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv
    Columns used:
      PatientID, Survival.time, deadstatus.event
    """
    df = pd.read_csv(csv_path)
    needed = ["PatientID", "Survival.time", "deadstatus.event"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"[Clinical CSV] Missing required column: {c}. "
                             f"Found columns: {list(df.columns)}")

    out = df[needed].copy()
    out["PatientID"] = out["PatientID"].astype(str).str.strip()
    out["Survival.time"] = out["Survival.time"].astype(float)
    out["deadstatus.event"] = out["deadstatus.event"].astype(int)
    out = out.dropna(subset=["PatientID", "Survival.time", "deadstatus.event"])
    out = out.set_index("PatientID", drop=False)
    return out

class ImageHead(nn.Module):
    """
    Matches phase3_imageonly.py head:
    Linear(512->256)->LN->ReLU->Dropout->Linear(256->128)->LN->ReLU->Dropout->Linear(128->n_bins)
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

def strip_module_prefix(state_dict):
    # handle DataParallel checkpoints if any
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict

def load_head_and_cuts(best_pt: Path, dropout: float = 0.35):
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"{best_pt} is not a dict checkpoint.")
    if "cuts" not in ckpt:
        raise ValueError(f"{best_pt} missing 'cuts'.")

    cuts = np.asarray(ckpt["cuts"], dtype=float)
    n_bins = int(ckpt.get("n_time_bins", len(cuts) - 1))
    if len(cuts) != n_bins + 1:
        # still proceed, but warn
        print(f"[WARN] cuts length {len(cuts)} != n_bins+1 {n_bins+1} in {best_pt.name}")

    state = ckpt.get("state_dict", ckpt.get("model_state_dict", None))
    if state is None:
        raise ValueError(f"{best_pt} missing 'state_dict'/'model_state_dict'.")
    state = strip_module_prefix(state)

    head_state = {k: v for k, v in state.items() if k.startswith("head.")}
    if not head_state:
        raise ValueError(f"No 'head.*' keys found in {best_pt} state_dict.")

    head = ImageHead(n_bins=n_bins, dropout=dropout)
    head.load_state_dict(head_state, strict=True)
    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head, cuts, n_bins

@torch.no_grad()
def risk_from_head_embeddings(head: nn.Module, emb_np: np.ndarray, cuts: np.ndarray):
    """
    Discrete-time hazard model:
      h_k = sigmoid(logit_k)
      S_k = Π_{m<=k} (1 - h_m)
    Then risk_integral = -∫ S(t) dt approximated by trapz over bin midpoints. :contentReference[oaicite:1]{index=1}
    """
    emb = torch.from_numpy(emb_np).float()
    logits = head(emb).cpu().numpy()  # [N, n_bins]
    haz = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    haz = np.clip(haz, 1e-7, 1 - 1e-7)

    # survival at each bin edge index (aligned with bins): S_k = prod_{j<=k} (1-h_j)
    surv = np.cumprod(1.0 - haz, axis=1)  # [N, n_bins]

    # bin midpoints
    cuts = np.asarray(cuts, dtype=float)
    if cuts.ndim != 1 or len(cuts) < 2:
        raise ValueError("cuts malformed.")
    bin_mids = (cuts[:-1] + cuts[1:]) / 2.0
    nb = min(surv.shape[1], len(bin_mids))
    risk_int = -np.trapz(surv[:, :nb], x=bin_mids[:nb], axis=1)
    return risk_int

def sanity_one_fold(fold_dir: Path, clin: pd.DataFrame):
    print("\n" + "=" * 78)
    print(f"[Fold] {fold_dir.name}  ({fold_dir})")
    print("=" * 78)

    best_pt = fold_dir / "best.pt"
    val_emb = fold_dir / "val_embeddings.npy"
    val_pids_csv = fold_dir / "val_pids.csv"
    tr_pids_csv = fold_dir / "train_pids.csv"
    val_pred_csv = fold_dir / "val_pred.csv"

    for p in [best_pt, val_emb, val_pids_csv, tr_pids_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    tr_pids = read_pid_list(tr_pids_csv)
    va_pids = read_pid_list(val_pids_csv)

    print(f"#train_pids = {len(tr_pids)} | #val_pids = {len(va_pids)}")
    print(f"#clinical_pids = {clin.shape[0]}")

    set_tr, set_va, set_cl = set(tr_pids), set(va_pids), set(clin.index.tolist())
    print(f"|clinical ∩ train| = {len(set_cl & set_tr)} / {len(set_tr)}")
    print(f"|clinical ∩ val|   = {len(set_cl & set_va)} / {len(set_va)}")

    miss_tr = sorted(list(set_tr - set_cl))
    miss_va = sorted(list(set_va - set_cl))
    if miss_tr:
        print(f"[WARN] train_pids missing in clinical: {len(miss_tr)}  (show up to 10): {miss_tr[:10]}")
    if miss_va:
        print(f"[WARN] val_pids missing in clinical: {len(miss_va)}  (show up to 10): {miss_va[:10]}")

    # check duplicates
    if len(tr_pids) != len(set_tr):
        print(f"[WARN] duplicates in train_pids.csv: {len(tr_pids) - len(set_tr)}")
    if len(va_pids) != len(set_va):
        print(f"[WARN] duplicates in val_pids.csv: {len(va_pids) - len(set_va)}")

    # order check against val_pred.csv if present
    if val_pred_csv.exists():
        vp = pd.read_csv(val_pred_csv)
        if "patient_id" in vp.columns:
            vp_ids = vp["patient_id"].astype(str).tolist()
            same_order = (vp_ids == va_pids)
            same_set = (set(vp_ids) == set_va)
            print(f"[val_pred.csv] same PID set as val_pids.csv? {same_set}")
            print(f"[val_pred.csv] same PID ORDER as val_pids.csv? {same_order}")
            if (not same_order) and same_set:
                print("  first10 val_pids.csv:", va_pids[:10])
                print("  first10 val_pred.csv :", vp_ids[:10])
        else:
            print("[WARN] val_pred.csv has no 'patient_id' column (unexpected).")

    # load embeddings
    emb = np.load(val_emb)
    if emb.ndim != 2 or emb.shape[1] != 512:
        print(f"[WARN] val_embeddings shape = {emb.shape} (expected [N, 512])")
    if emb.shape[0] != len(va_pids):
        print(f"[WARN] embeddings rows {emb.shape[0]} != #val_pids {len(va_pids)} (PID alignment likely broken!)")

    # spot-check first 10
    emb_norm = np.linalg.norm(emb, axis=1)
    print("\n[Spot check] first 10 (pid, ||emb||, time, event):")
    for i in range(min(10, len(va_pids), emb.shape[0])):
        pid = va_pids[i]
        if pid in clin.index:
            t = float(clin.loc[pid, "Survival.time"])
            e = int(clin.loc[pid, "deadstatus.event"])
        else:
            t, e = np.nan, -1
        print(f"  {i:02d}  {pid:>10s}  norm={emb_norm[i]:8.3f}  time={t:8.1f}  event={e}")

    # build ordered y for Uno
    # (this is the critical “merge/join order” check: we index by PID list, preserving order)
    tr_df = clin.loc[[p for p in tr_pids if p in clin.index]]
    va_df = clin.loc[[p for p in va_pids if p in clin.index]]

    y_tr = to_struct(tr_df["Survival.time"].values, tr_df["deadstatus.event"].values)
    y_va = to_struct(va_df["Survival.time"].values, va_df["deadstatus.event"].values)

    evt_times_tr = tr_df.loc[tr_df["deadstatus.event"] == 1, "Survival.time"].values
    tau = np.quantile(evt_times_tr, 0.9) if len(evt_times_tr) > 0 else tr_df["Survival.time"].max()

    # load head + cuts, compute “Step2 baseline image-only” risk from head(emb)
    head, cuts, _ = load_head_and_cuts(best_pt, dropout=0.35)
    # IMPORTANT: embeddings must be in the same order as va_df (val_pids.csv). We use va_pids order.
    # If some val_pids were missing in clinical, we must subset embeddings accordingly.
    keep_mask = np.array([pid in clin.index for pid in va_pids], dtype=bool)
    emb_keep = emb[keep_mask]

    risk_head = risk_from_head_embeddings(head, emb_keep, cuts)

    uno_head = concordance_index_ipcw(y_tr, y_va, risk_head, tau=tau)[0]
    har_head = concordance_index_censored(y_va["event"], y_va["time"], risk_head)[0]
    print(f"\n[Image-only via head(emb) — the SAME object Step2 fuses on]")
    print(f"  Uno(IPCW,int) = {uno_head:.3f} | Harrell = {har_head:.3f}   (tau={tau:.1f})")

    # Compare to val_pred.csv risk_integral (likely TTA-based from your phase3_imageonly.py)
    if val_pred_csv.exists():
        vp = pd.read_csv(val_pred_csv)
        if {"patient_id", "risk_integral"}.issubset(vp.columns):
            vp["patient_id"] = vp["patient_id"].astype(str)
            merged = pd.DataFrame({"patient_id": va_df["PatientID"].values})
            merged = merged.merge(vp[["patient_id", "risk_integral"]], on="patient_id", how="left")
            if merged["risk_integral"].isna().any():
                nmiss = int(merged["risk_integral"].isna().sum())
                print(f"[WARN] {nmiss} val patients missing risk_integral in val_pred.csv (PID mismatch).")
            else:
                risk_tta = merged["risk_integral"].values.astype(float)
                uno_tta = concordance_index_ipcw(y_tr, y_va, risk_tta, tau=tau)[0]
                har_tta = concordance_index_censored(y_va["event"], y_va["time"], risk_tta)[0]
                corr = np.corrcoef(risk_head, risk_tta)[0, 1]
                max_abs = float(np.max(np.abs(risk_head - risk_tta)))
                print(f"\n[Image-only via val_pred.csv risk_integral (often TTA-based)]")
                print(f"  Uno(IPCW,int) = {uno_tta:.3f} | Harrell = {har_tta:.3f}")
                print(f"  corr(risk_head, risk_val_pred) = {corr:.3f} | max|diff| = {max_abs:.6f}")
                print("  NOTE: if phase3_imageonly used TTA, risk_val_pred != head(emb) is EXPECTED.")
        else:
            print("[WARN] val_pred.csv missing required columns {patient_id, risk_integral}.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=str, required=True)
    ap.add_argument("--clinical_csv", type=str, required=True)
    ap.add_argument("--fold", type=int, default=None, help="run only one fold (0..4)")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    clin = load_clinical_minimal(Path(args.clinical_csv))

    fold_dirs = sorted([p for p in exp_dir.glob("fold_*") if p.is_dir()])
    if args.fold is not None:
        fold_dirs = [exp_dir / f"fold_{args.fold}"]
    if not fold_dirs:
        raise FileNotFoundError(f"No fold_* directories found under {exp_dir}")

    for fd in fold_dirs:
        sanity_one_fold(fd, clin)

if __name__ == "__main__":
    main()
