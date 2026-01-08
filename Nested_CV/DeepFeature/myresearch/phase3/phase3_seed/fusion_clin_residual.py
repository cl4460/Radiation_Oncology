#!/usr/bin/env python
# fusion_clin_residual.py

import os, json, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.loss import NLLLogistiHazardLoss
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored


# ---------------- utils ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_struct(times, events):
    return np.array([(bool(e), float(t)) for t, e in zip(times, events)],
                    dtype=[("event", bool), ("time", float)])

def read_pid_list(path: Path):
    return pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()

def pick_np(path_a: Path, path_b: Path):
    if path_a.exists(): return path_a
    if path_b.exists(): return path_b
    raise FileNotFoundError(f"Missing {path_a} or {path_b}")

def load_edges_and_head_state(best_pt: Path):
    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    edges = np.asarray(ckpt["cuts"], dtype=float)
    n_bins = int(ckpt.get("n_time_bins", len(edges) - 1))
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", None))
    if state is None:
        raise KeyError("Checkpoint missing state_dict/model_state_dict")

    # head.* should match FrozenImageHead keys: head.0.weight, ...
    head_state = {k: v for k, v in state.items() if k.startswith("head.")}
    if len(head_state) == 0:
        raise KeyError("No head.* keys found in checkpoint state dict")
    return edges, n_bins, head_state

def load_embeddings_aligned(fold_dir: Path, split: str, target_pids: list):
    """
    Ensures embeddings rows match target_pids order.
    If *_pids_extracted.csv exists, reorder embeddings by pid mapping.
    """
    assert split in ("train", "val")
    emb_path = pick_np(
        fold_dir / f"{split}_embeddings.npy",
        fold_dir / f"{split}_embeddings_extracted.npy",
    )
    emb = np.load(emb_path).astype(np.float32)

    # optional pid file that indicates embedding row order
    pid_path = fold_dir / f"{split}_pids_extracted.csv"
    if pid_path.exists():
        emb_pids = pd.read_csv(pid_path, header=None).iloc[:, 0].astype(str).tolist()
        if len(emb_pids) != emb.shape[0]:
            raise ValueError(f"{pid_path} length {len(emb_pids)} != emb rows {emb.shape[0]}")

        # reorder to match target_pids
        idx = {pid: i for i, pid in enumerate(emb_pids)}
        missing = [pid for pid in target_pids if pid not in idx]
        if missing:
            raise KeyError(f"Embeddings missing {len(missing)} pids, e.g. {missing[:5]}")
        emb = emb[[idx[pid] for pid in target_pids]]
    else:
        # if no pid file exists, at least enforce same length
        if emb.shape[0] != len(target_pids):
            raise ValueError(
                f"Emb rows {emb.shape[0]} != target_pids {len(target_pids)} "
                f"(no {pid_path.name} to reorder)."
            )
    return emb


# ---------------- model ----------------
class FrozenImageHead(nn.Module):
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
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_bins)
        )
        if zero_init_last:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
    def forward(self, x): return self.net(x)

class ResidualClinFusion(nn.Module):
    def __init__(self, frozen_head: FrozenImageHead, clin_dim: int, n_bins: int,
                 delta_hidden: int = 64, delta_dropout: float = 0.5):
        super().__init__()
        self.frozen_head = frozen_head
        for p in self.frozen_head.parameters():
            p.requires_grad = False
        self.frozen_head.eval()  # keep dropout off

        # delta sees clinical only
        self.delta = DeltaMLP(clin_dim, n_bins, hidden=delta_hidden, dropout=delta_dropout, zero_init_last=True)

    def forward(self, emb, clin):
        with torch.no_grad():
            img_logits = self.frozen_head(emb)
        delta_logits = self.delta(clin)
        return img_logits, delta_logits

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_head.eval()  # force eval regardless
        return self


# ---------------- clinical features ----------------
def _zscore(train_arr, val_arr):
    mu = np.nanmean(train_arr)
    sd = np.nanstd(train_arr)
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    train_z = (np.nan_to_num(train_arr, nan=mu) - mu) / sd
    val_z   = (np.nan_to_num(val_arr,   nan=mu) - mu) / sd
    return train_z.astype(np.float32), val_z.astype(np.float32)

def build_clinical_features_for_fold(df: pd.DataFrame, tr_pids: list, va_pids: list, overall_stage_encoding: str):
    use_cols = [
        "PatientID", "Survival.time", "deadstatus.event",
        "age", "gender",
        "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage",
        "Overall.Stage",
    ]
    d = df[use_cols].copy()
    d["PatientID"] = d["PatientID"].astype(str)
    d = d.set_index("PatientID")

    # hard fail if pids missing (better than silent misalignment)
    missing_tr = [p for p in tr_pids if p not in d.index]
    missing_va = [p for p in va_pids if p not in d.index]
    if missing_tr or missing_va:
        raise KeyError(f"Missing clinical rows: train {len(missing_tr)}, val {len(missing_va)} (e.g., {missing_tr[:3]})")

    tr = d.loc[tr_pids].reset_index()
    va = d.loc[va_pids].reset_index()

    t_tr = tr["Survival.time"].astype(float).to_numpy()
    e_tr = tr["deadstatus.event"].astype(int).to_numpy()
    t_va = va["Survival.time"].astype(float).to_numpy()
    e_va = va["deadstatus.event"].astype(int).to_numpy()

    feats_tr, feats_va, names = [], [], []

    # age z
    age_tr = pd.to_numeric(tr["age"], errors="coerce").to_numpy()
    age_va = pd.to_numeric(va["age"], errors="coerce").to_numpy()
    z_tr, z_va = _zscore(age_tr, age_va)
    feats_tr.append(z_tr); feats_va.append(z_va); names.append("age_z")

    # gender (fit mode on train)
    g_tr = tr["gender"].astype(str).str.lower().str.strip()
    g_va = va["gender"].astype(str).str.lower().str.strip()
    # map common values; unknown -> train mode
    gmap = {"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0}
    gt = g_tr.map(gmap)
    mode_val = float(gt.dropna().mode().iloc[0]) if gt.notna().any() else 0.0
    feats_tr.append(gt.fillna(mode_val).to_numpy(dtype=np.float32))
    feats_va.append(g_va.map(gmap).fillna(mode_val).to_numpy(dtype=np.float32))
    names.append("gender")

    # T/N/M numeric-ish (coerce; zscore using train stats)
    for col, nm in [("clinical.T.Stage", "T"), ("Clinical.N.Stage", "N"), ("Clinical.M.Stage", "M")]:
        a_tr = pd.to_numeric(tr[col], errors="coerce").to_numpy()
        a_va = pd.to_numeric(va[col], errors="coerce").to_numpy()
        z_tr, z_va = _zscore(a_tr, a_va)
        feats_tr.append(z_tr); feats_va.append(z_va); names.append(f"{nm}_z")

    # overall stage
    if overall_stage_encoding == "onehot":
        cats = ["I", "II", "IIIa", "IIIb", "IV"]
        os_tr = tr["Overall.Stage"].astype(str).str.strip()
        os_va = va["Overall.Stage"].astype(str).str.strip()
        # fill missing with train mode
        mode_stage = os_tr.replace("nan", np.nan).dropna().mode().iloc[0] if os_tr.replace("nan", np.nan).dropna().size > 0 else "IIIa"
        os_tr = os_tr.replace("nan", np.nan).fillna(mode_stage)
        os_va = os_va.replace("nan", np.nan).fillna(mode_stage)
        for c in cats:
            feats_tr.append((os_tr == c).to_numpy(dtype=np.float32))
            feats_va.append((os_va == c).to_numpy(dtype=np.float32))
            names.append(f"Stage_{c}")
    elif overall_stage_encoding == "ordinal":
        omap = {"I": 1.0, "II": 2.0, "IIIa": 3.0, "IIIb": 4.0, "IV": 5.0}
        os_tr = tr["Overall.Stage"].astype(str).str.strip().map(omap)
        os_va = va["Overall.Stage"].astype(str).str.strip().map(omap)
        mode_val = float(os_tr.dropna().mode().iloc[0]) if os_tr.notna().any() else 3.0
        feats_tr.append(os_tr.fillna(mode_val).to_numpy(dtype=np.float32))
        feats_va.append(os_va.fillna(mode_val).to_numpy(dtype=np.float32))
        names.append("OverallStage_ord")

    X_tr = np.stack(feats_tr, axis=1)
    X_va = np.stack(feats_va, axis=1)
    return X_tr, t_tr, e_tr, X_va, t_va, e_va, names


# ---------------- eval helpers ----------------
@torch.no_grad()
def logits_to_surv_np(logits: np.ndarray):
    # stable sigmoid
    logits = np.clip(logits, -30.0, 30.0)
    haz = 1.0 / (1.0 + np.exp(-logits))
    haz = np.clip(haz, 1e-7, 1.0 - 1e-7)
    log_surv = np.cumsum(np.log(1.0 - haz), axis=1)
    return np.exp(log_surv)

def safe_tau(t_tr, e_tr):
    t_tr = np.asarray(t_tr, dtype=float)
    e_tr = np.asarray(e_tr, dtype=int)
    ev = t_tr[e_tr == 1]
    base = ev if ev.size >= 5 else t_tr  # fallback if too few events
    tau = float(np.quantile(base, 0.9))
    max_t = float(np.max(t_tr))
    # keep strictly inside range
    tau = min(tau, max_t - 1e-6) if max_t > 1e-6 else max_t
    return max(tau, 1e-6)


# ---------------- main ----------------
class FusionDS(Dataset):
    def __init__(self, emb, clin, y_idx, y_evt):
        self.emb = torch.tensor(emb, dtype=torch.float32)
        self.clin = torch.tensor(clin, dtype=torch.float32)
        self.y_idx = torch.tensor(y_idx, dtype=torch.long)
        self.y_evt = torch.tensor(y_evt, dtype=torch.float32)
    def __len__(self): return self.emb.shape[0]
    def __getitem__(self, i): return self.emb[i], self.clin[i], self.y_idx[i], self.y_evt[i]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=str, required=True)
    ap.add_argument("--clinical_csv", type=str, required=True)
    ap.add_argument("--out_name", type=str, default="step2_clin_residual_only")
    ap.add_argument("--overall_stage_encoding", type=str, default="onehot", choices=["onehot", "ordinal", "none"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--delta_hidden", type=int, default=64)
    ap.add_argument("--delta_dropout", type=float, default=0.5)
    ap.add_argument("--delta_penalty", type=float, default=1e-3)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = Path(args.exp_dir)
    out_root = exp_dir / args.out_name
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.clinical_csv)
    run_rows = []

    for fold in range(5):
        fold_dir = exp_dir / f"fold_{fold}"
        best_pt = fold_dir / "best.pt"

        tr_pids = read_pid_list(fold_dir / "train_pids.csv")
        va_pids = read_pid_list(fold_dir / "val_pids.csv")

        tr_emb = load_embeddings_aligned(fold_dir, "train", tr_pids)
        va_emb = load_embeddings_aligned(fold_dir, "val", va_pids)

        edges, n_bins, head_state = load_edges_and_head_state(best_pt)
        bin_mids = (edges[:-1] + edges[1:]) / 2.0

        X_tr, t_tr, e_tr, X_va, t_va, e_va, feat_names = build_clinical_features_for_fold(
            df, tr_pids, va_pids, args.overall_stage_encoding
        )

        lab = LabTransDiscreteTime(cuts=edges[1:-1])
        # cuts are fixed; fit_transform will warn but is ok. If you want no warning: use transform directly.
        y_tr_idx, y_tr_evt = lab.fit_transform(t_tr, e_tr)
        y_va_idx, y_va_evt = lab.transform(t_va, e_va)

        frozen_head = FrozenImageHead(n_bins=n_bins, dropout=0.35)
        frozen_head.load_state_dict(head_state, strict=True)

        model = ResidualClinFusion(
            frozen_head=frozen_head,
            clin_dim=X_tr.shape[1],
            n_bins=n_bins,
            delta_hidden=args.delta_hidden,
            delta_dropout=args.delta_dropout
        ).to(device)

        optim = torch.optim.AdamW(model.delta.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = NLLLogistiHazardLoss()

        ds_tr = FusionDS(tr_emb, X_tr, y_tr_idx, y_tr_evt)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
        ds_va = FusionDS(va_emb, X_va, y_va_idx, y_va_evt)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False)

        y_tr_st = to_struct(t_tr, e_tr)
        y_va_st = to_struct(t_va, e_va)
        tau = safe_tau(t_tr, e_tr)

        fold_out = out_root / f"fold_{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)

        best_uno = -1.0
        best_ep = -1
        no_improve = 0

        print(f"\n--- Fold {fold} ---")
        print(f"[DEBUG] frozen_head.training = {model.frozen_head.training}")
        print(f"[DEBUG] clin_dim = {X_tr.shape[1]} | features = {feat_names}")

        for ep in range(args.epochs):
            model.train()
            total = 0.0

            img_rms = []
            delta_rms = []

            for eb, cb, yi, ye in dl_tr:
                eb, cb, yi, ye = eb.to(device), cb.to(device), yi.to(device), ye.to(device)
                optim.zero_grad(set_to_none=True)

                img_logits, delta_logits = model(eb, cb)
                final_logits = img_logits + delta_logits

                nll = loss_fn(final_logits, yi, ye)
                penalty = args.delta_penalty * (delta_logits ** 2).mean()
                loss = nll + penalty

                loss.backward()
                nn.utils.clip_grad_norm_(model.delta.parameters(), 1.0)
                optim.step()

                total += loss.item()

                with torch.no_grad():
                    # RMS over time bins
                    img_rms.append(torch.sqrt((img_logits**2).mean(dim=1)).mean().item())
                    delta_rms.append(torch.sqrt((delta_logits**2).mean(dim=1)).mean().item())

            model.eval()
            with torch.no_grad():
                logits_list = []
                for eb, cb, _, _ in dl_va:
                    il, dl = model(eb.to(device), cb.to(device))
                    logits_list.append((il + dl).cpu().numpy())
                val_logits = np.concatenate(logits_list, axis=0)

            surv = logits_to_surv_np(val_logits)
            val_risk = -np.trapz(surv, x=bin_mids, axis=1)

            # UNO IPCW C-index
            val_uno = concordance_index_ipcw(y_tr_st, y_va_st, val_risk, tau=tau)[0]
            # Harrell (censored) C-index
            val_harrell = concordance_index_censored(e_va.astype(bool), t_va, val_risk)[0]

            ratio = float(np.mean(delta_rms) / (np.mean(img_rms) + 1e-8))
            tr_loss = total / max(1, len(dl_tr))

            print(f"Ep {ep+1:03d} | L {tr_loss:.4f} | Uno {val_uno:.3f} | Harrell {val_harrell:.3f} | RMSratio {ratio:.4f}")

            if np.isfinite(val_uno) and val_uno > best_uno + 1e-6:
                best_uno = float(val_uno)
                best_ep = ep + 1
                no_improve = 0
                # save best delta
                torch.save({"delta_state": model.delta.state_dict(),
                            "best_ep": best_ep,
                            "best_uno": best_uno,
                            "tau": tau},
                           fold_out / "best_delta.pt")
                np.save(fold_out / "val_risk_best.npy", val_risk.astype(np.float32))
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    break

        run_rows.append({"fold": fold, "best_uno": best_uno, "best_ep": best_ep})
        print(f"Fold {fold} Best: Uno={best_uno:.4f} @ ep {best_ep}")

    df_res = pd.DataFrame(run_rows)
    df_res.to_csv(out_root / "summary.csv", index=False)
    print("\nResults:")
    print(df_res)
    print(f"Mean: {df_res['best_uno'].mean():.4f} ± {df_res['best_uno'].std():.4f}")

if __name__ == "__main__":
    main()
