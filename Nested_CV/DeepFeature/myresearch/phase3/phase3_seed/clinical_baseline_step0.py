#!/usr/bin/env python
# phase3_clinical_baseline_step0.py
import os, json, math, random, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.loss import NLLLogistiHazardLoss
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored


# ---------------- reproducibility ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_struct(times, events):
    return np.array([(bool(e), float(t)) for t, e in zip(times, events)],
                    dtype=[("event", bool), ("time", float)])


def read_pid_list(path: Path):
    s = pd.read_csv(path, header=None).iloc[:, 0].astype(str)
    return s.tolist()


def load_edges_from_image_bestpt(best_pt: Path):
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    edges = np.asarray(ckpt["cuts"], dtype=float)  # length = n_bins + 1 in your imageonly code
    n_bins = int(ckpt.get("n_time_bins", len(edges) - 1))
    if len(edges) != n_bins + 1:
        raise ValueError(f"cuts length ({len(edges)}) != n_time_bins+1 ({n_bins+1}). "
                         f"Please verify your imageonly checkpoint format.")
    return edges, n_bins


def build_clinical_features_for_fold(df: pd.DataFrame,
                                    tr_pids: list,
                                    va_pids: list,
                                    overall_stage_encoding: str):
    """
    Tailored to NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv

    Uses ONLY:
      - age
      - gender
      - clinical.T.Stage (1 NaN in whole CSV)
      - Clinical.N.Stage (no NaN)
      - Clinical.M.Stage (no NaN)
      - Overall.Stage (1 NaN in whole CSV; categories I/II/IIIa/IIIb)
    """
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

    # labels
    t_tr = tr["Survival.time"].astype(float).to_numpy()
    e_tr = tr["deadstatus.event"].astype(int).to_numpy()
    t_va = va["Survival.time"].astype(float).to_numpy()
    e_va = va["deadstatus.event"].astype(int).to_numpy()

    feats_tr, feats_va, feat_names = [], [], []

    # age: fill NaN with train mean, then z-score
    age_tr = tr["age"].astype(float)
    age_va = va["age"].astype(float)
    mu = float(age_tr.mean(skipna=True))
    age_tr = age_tr.fillna(mu).to_numpy(dtype=np.float32)
    age_va = age_va.fillna(mu).to_numpy(dtype=np.float32)
    sigma = float(np.std(age_tr)) if float(np.std(age_tr)) > 1e-8 else 1.0
    feats_tr.append(((age_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((age_va - mu) / sigma).astype(np.float32))
    feat_names.append("age_z")

    # gender: map to 0/1, no z-score
    gmap = {"male": 1.0, "female": 0.0}
    g_tr = tr["gender"].astype(str).str.strip().str.lower().map(gmap).to_numpy(dtype=np.float32)
    g_va = va["gender"].astype(str).str.strip().str.lower().map(gmap).to_numpy(dtype=np.float32)
    if np.isnan(g_tr).any() or np.isnan(g_va).any():
        raise ValueError("Unexpected gender values found. Expected only 'male'/'female'.")
    feats_tr.append(g_tr)
    feats_va.append(g_va)
    feat_names.append("gender_01")

    # T stage: 1 NaN in whole CSV -> fill with train mode, then z-score
    T_tr_s = tr["clinical.T.Stage"].astype(float)
    T_va_s = va["clinical.T.Stage"].astype(float)
    T_mode = float(T_tr_s.dropna().value_counts().idxmax())
    T_tr = T_tr_s.fillna(T_mode).to_numpy(dtype=np.float32)
    T_va = T_va_s.fillna(T_mode).to_numpy(dtype=np.float32)
    mu = float(np.mean(T_tr)); sigma = float(np.std(T_tr)) if float(np.std(T_tr)) > 1e-8 else 1.0
    feats_tr.append(((T_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((T_va - mu) / sigma).astype(np.float32))
    feat_names.append("T_z")

    # N stage: no NaN -> z-score
    N_tr = tr["Clinical.N.Stage"].astype(float).to_numpy(dtype=np.float32)
    N_va = va["Clinical.N.Stage"].astype(float).to_numpy(dtype=np.float32)
    mu = float(np.mean(N_tr)); sigma = float(np.std(N_tr)) if float(np.std(N_tr)) > 1e-8 else 1.0
    feats_tr.append(((N_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((N_va - mu) / sigma).astype(np.float32))
    feat_names.append("N_z")

    # M stage: no NaN -> z-score
    M_tr = tr["Clinical.M.Stage"].astype(float).to_numpy(dtype=np.float32)
    M_va = va["Clinical.M.Stage"].astype(float).to_numpy(dtype=np.float32)
    mu = float(np.mean(M_tr)); sigma = float(np.std(M_tr)) if float(np.std(M_tr)) > 1e-8 else 1.0
    feats_tr.append(((M_tr - mu) / sigma).astype(np.float32))
    feats_va.append(((M_va - mu) / sigma).astype(np.float32))
    feat_names.append("M_z")

    # Overall.Stage: 1 NaN -> fill with train mode
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
            raise ValueError("Unexpected Overall.Stage values found. Expected only I/II/IIIa/IIIb/NaN.")
        mu = float(np.mean(os_tr_num)); sigma = float(np.std(os_tr_num)) if float(np.std(os_tr_num)) > 1e-8 else 1.0
        feats_tr.append(((os_tr_num - mu) / sigma).astype(np.float32))
        feats_va.append(((os_va_num - mu) / sigma).astype(np.float32))
        feat_names.append("OverallStage_ord_z")
    else:
        raise ValueError("overall_stage_encoding must be 'ordinal' or 'onehot'")

    X_tr = np.stack(feats_tr, axis=1)  # [N, d]
    X_va = np.stack(feats_va, axis=1)

    return X_tr, t_tr, e_tr, X_va, t_va, e_va, feat_names


class ClinicalDS(Dataset):
    def __init__(self, X, y_idx, y_evt):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_idx = torch.tensor(y_idx, dtype=torch.long)
        self.y_evt = torch.tensor(y_evt, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y_idx[i], self.y_evt[i]


class ClinicalNet(nn.Module):
    def __init__(self, in_dim: int, n_bins: int, hidden: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_bins),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def logits_to_surv_np(logits: np.ndarray):
    # logits: [N, n_bins]
    haz = 1.0 / (1.0 + np.exp(-logits))
    haz = np.clip(haz, 1e-7, 1.0 - 1e-7)
    log_surv = np.cumsum(np.log(1.0 - haz), axis=1)
    surv = np.exp(log_surv)
    return surv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=str, required=True, help="phase3_outputs/<lr_xxx> containing fold_k/ with best.pt + pids csv")
    ap.add_argument("--clinical_csv", type=str, required=True)
    ap.add_argument("--out_name", type=str, default="step0_clinical_baseline")
    ap.add_argument("--overall_stage_encoding", type=str, default="ordinal", choices=["ordinal", "onehot"])

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--fold", type=int, default=None, help="0..4 (optional)")

    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = Path(args.exp_dir)
    out_root = exp_dir / args.out_name
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.clinical_csv)

    run_rows = []

    for fold in range(5):
        if args.fold is not None and fold != args.fold:
            continue

        fold_dir = exp_dir / f"fold_{fold}"
        best_pt = fold_dir / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"Missing {best_pt} (need imageonly best.pt for cuts).")

        tr_pids_path = fold_dir / "train_pids.csv"
        va_pids_path = fold_dir / "val_pids.csv"
        if not tr_pids_path.exists() or not va_pids_path.exists():
            raise FileNotFoundError("Need train_pids.csv and val_pids.csv in each fold dir to ensure identical splits.")

        tr_pids = read_pid_list(tr_pids_path)
        va_pids = read_pid_list(va_pids_path)

        edges, n_bins = load_edges_from_image_bestpt(best_pt)
        bin_mids = (edges[:-1] + edges[1:]) / 2.0

        X_tr, t_tr, e_tr, X_va, t_va, e_va, feat_names = build_clinical_features_for_fold(
            df, tr_pids, va_pids, args.overall_stage_encoding
        )

        # label transform exactly like imageonly: use internal boundaries edges[1:-1]
        lab = LabTransDiscreteTime(cuts=edges[1:-1])
        y_tr_idx, y_tr_evt = lab.fit_transform(t_tr, e_tr)
        y_va_idx, y_va_evt = lab.transform(t_va, e_va)

        ds_tr = ClinicalDS(X_tr, y_tr_idx, y_tr_evt)
        ds_va = ClinicalDS(X_va, y_va_idx, y_va_evt)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False)

        model = ClinicalNet(in_dim=X_tr.shape[1], n_bins=n_bins, hidden=args.hidden, dropout=args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = NLLLogistiHazardLoss()

        y_tr_struct = to_struct(t_tr, e_tr)
        y_va_struct = to_struct(t_va, e_va)
        evt_times_tr = t_tr[e_tr == 1]
        tau = np.quantile(evt_times_tr, 0.9) if len(evt_times_tr) > 0 else float(np.max(t_tr))

        best_uno = -1.0
        best_ep = -1
        no_improve = 0

        fold_out = out_root / f"fold_{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)

        for ep in range(args.epochs):
            model.train()
            total = 0.0
            for xb, yb_idx, yb_evt in dl_tr:
                xb = xb.to(device)
                yb_idx = yb_idx.to(device)
                yb_evt = yb_evt.to(device)

                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb_idx, yb_evt)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += float(loss.item()) * xb.size(0)

            tr_loss = total / max(1, len(ds_tr))

            # ---- validate ----
            model.eval()
            logits_list = []
            with torch.no_grad():
                for xb, _, _ in dl_va:
                    logits_list.append(model(xb.to(device)).cpu().numpy())
            logits_va = np.concatenate(logits_list, axis=0)
            surv_va = logits_to_surv_np(logits_va)  # [Nv, n_bins]
            risk_int = -np.trapz(surv_va, x=bin_mids, axis=1)

            uno_int = concordance_index_ipcw(y_tr_struct, y_va_struct, risk_int, tau=tau)[0]
            harrell = concordance_index_censored(y_va_struct["event"], y_va_struct["time"], risk_int)[0]

            print(f"[Fold {fold}] Ep {ep+1:03d} | tr_loss {tr_loss:.4f} | Uno(int) {uno_int:.3f} | Harrell {harrell:.3f}")

            if not np.isnan(uno_int) and uno_int > best_uno + 1e-5:
                best_uno = float(uno_int)
                best_ep = ep
                no_improve = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "feat_names": feat_names,
                        "overall_stage_encoding": args.overall_stage_encoding,
                        "edges": edges.tolist(),
                        "n_bins": n_bins,
                        "best_epoch": best_ep,
                        "best_uno": best_uno,
                        "seed": args.seed,
                    },
                    fold_out / "best.pt",
                )

                # save val preds for later paired analysis
                pd.DataFrame({
                    "patient_id": va_pids,
                    "risk_integral": risk_int,
                    "time": t_va,
                    "event": e_va
                }).to_csv(fold_out / "val_pred.csv", index=False)
            else:
                no_improve += 1

            if no_improve >= args.patience:
                print(f"[Fold {fold}] Early stop at ep {ep+1}")
                break

        run_rows.append({
            "seed": args.seed,
            "fold": fold,
            "best_uno": best_uno,
            "best_epoch": best_ep,
            "in_dim": int(X_tr.shape[1]),
            "overall_stage_encoding": args.overall_stage_encoding,
        })

    # write a small jsonl summary
    with open(out_root / "runs.jsonl", "a") as f:
        for r in run_rows:
            f.write(json.dumps(r) + "\n")

    if run_rows:
        vals = [r["best_uno"] for r in run_rows]
        print(f"\n[Clinical-only] mean Uno(int) = {np.mean(vals):.3f} ± {np.std(vals):.3f}")


if __name__ == "__main__":
    main()
