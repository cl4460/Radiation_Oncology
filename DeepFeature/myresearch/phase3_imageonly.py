#!/usr/bin/env python
# phase3_imageonly.py
# Image-only survival training: CT + MASK (2-channel), ResNet10-GN, Logistic-Hazard
import os, math, time, random, argparse, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- Paths (edit to your tree) ----------
PHASE2_CROP_LOG = "/home/lichengze/Research/DeepFeature/myresearch/phase2_outputs/crop_log.csv"
CLINICAL_CSV    = "/home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical.csv"
OUTPUT_ROOT     = "/home/lichengze/Research/DeepFeature/myresearch/phase3_outputs"

# ---------- Libs ----------
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    Lambdad, ConcatItemsd, DeleteItemsd, ToTensord,
    RandFlipd, RandAffined, RandGaussianNoised, RandAdjustContrastd
)
from monai.networks.nets import resnet as monai_resnet

from pycox.preprocessing.label_transforms import LabTransDiscreteTime
from pycox.models.loss import NLLLogistiHazardLoss

from sksurv.metrics import concordance_index_ipcw, concordance_index_censored, brier_score

warnings.filterwarnings("ignore")

# ================= Config =================
@dataclass
class CFG:
    N_FOLDS: int = 5
    SEED: int = 42

    # bins
    N_BINS_MIN: int = 12
    N_BINS_MAX: int = 15
    MIN_EVENTS_PER_BIN: int = 15

    # train
    BATCH_SIZE: int = 4
    ACCUM_STEPS: int = 2
    MAX_EPOCHS: int = 100
    EARLY_STOP: int = 25
    LR: float = 7.2e-4
    WD: float = 2e-4
    DROPOUT: float = 0.35
    WARMUP_E: int = 25
    NUM_WORKERS: int = 4
    PREFETCH_FACTOR: int = 2

    # aug
    AUG_P: float = 0.5
    FLIP_P: float = 0.5
    ROT_DEG: Tuple[float,float,float] = (5.0,5.0,10.0)
    SCALE_RANGE: float = 0.12
    NOISE_STD: float = 0.02
    GAMMA_RANGE: Tuple[float,float] = (0.85,1.15)

    # eval
    USE_TTA: bool = True
    EVAL_24M_DAYS: int = 730

cfg = CFG()

# ================= Utils =================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def to_struct(times, events):
    return np.array([(bool(e), float(t)) for t,e in zip(times, events)],
                    dtype=[('event', bool), ('time', float)])

def adaptive_bins(times: np.ndarray, events: np.ndarray) -> int:
    n_events = int(events.sum())
    ideal = n_events // cfg.MIN_EVENTS_PER_BIN
    return int(np.clip(ideal, cfg.N_BINS_MIN, cfg.N_BINS_MAX))

def make_time_cuts(times: np.ndarray, events: np.ndarray, n_bins: int):
    evt_times = times[events==1]
    print("Fold events:", events.sum())
    if len(evt_times) < n_bins:
        return np.linspace(times.min(), times.max(), n_bins+1)
    qs = np.linspace(0, 1, n_bins+1)
    cuts = np.unique(np.quantile(evt_times, qs))
    hist, _ = np.histogram(evt_times, bins=cuts)
    print("Events per bin:", hist.tolist())
    
    if len(cuts) != n_bins+1:
        return np.linspace(times.min(), times.max(), n_bins+1)
    return cuts

# ================= Data =================
class ImageOnlyDS(Dataset):
    def __init__(self, recs: List[dict], y_idx, y_evt, transform: Compose):
        self.recs = recs
        self.y_idx = y_idx
        self.y_evt = y_evt
        self.tfm = transform

    def __len__(self): 
        return len(self.recs)

    def __getitem__(self, i):
        d = self.tfm(self.recs[i])
        d["label_idx"] = torch.tensor(self.y_idx[i], dtype=torch.long)
        d["label_event"] = torch.tensor(self.y_evt[i], dtype=torch.float32)
        d["patient_id"] = self.recs[i]["patient_id"]
        return d

def build_transforms(train: bool) -> Compose:
    tfms = [
        LoadImaged(keys=["ct","mask"]),
        EnsureChannelFirstd(keys=["ct","mask"]),
        EnsureTyped(keys=["ct","mask"], dtype=torch.float32, track_meta=False),
        Lambdad(keys=["mask"], func=lambda x: (x > 0).float()),
    ]
    if train:
        rx, ry, rz = map(lambda d: math.radians(d), cfg.ROT_DEG)
        tfms += [
            RandFlipd(keys=["ct","mask"], prob=cfg.FLIP_P, spatial_axis=1),  # H
            RandFlipd(keys=["ct","mask"], prob=cfg.FLIP_P, spatial_axis=2),  # W
            RandAffined(
                keys=["ct","mask"], 
                prob=cfg.AUG_P,
                rotate_range=(rx,ry,rz),
                scale_range=(cfg.SCALE_RANGE, cfg.SCALE_RANGE, cfg.SCALE_RANGE),
                mode=("bilinear", "nearest"),
                padding_mode="border"
            ),
            RandGaussianNoised(keys=["ct"], prob=cfg.AUG_P, std=cfg.NOISE_STD),
            RandAdjustContrastd(keys=["ct"], prob=cfg.AUG_P, gamma=cfg.GAMMA_RANGE),
        ]
    tfms += [
        ConcatItemsd(keys=["ct","mask"], name="image"),  # [2, D, H, W]
        DeleteItemsd(keys=["ct","mask"]),
        ToTensord(keys=["image"])
    ]
    return Compose(tfms)

# ================= Model =================
class ResNet10_ImageOnly_GN(nn.Module):
    def __init__(self, in_channels: int, n_bins: int, dropout: float=0.35):
        super().__init__()
        # backbone emits 512-dim features when num_classes=512
        self.backbone = monai_resnet.resnet10(
            spatial_dims=3, n_input_channels=in_channels, num_classes=512
        )
        self._bn_to_gn(self.backbone)
        self.head = nn.Sequential(
            nn.Linear(512, 256), 
            nn.LayerNorm(256), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128), 
            nn.LayerNorm(128), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_bins)
        )

    def _bn_to_gn(self, module, max_groups=8):
        for name, child in module.named_children():
            if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                C = child.num_features
                g = min(max_groups, C)
                while C % g != 0 and g > 1: 
                    g //= 2
                setattr(module, name, nn.GroupNorm(g, C))
            else:
                self._bn_to_gn(child, max_groups)

    def forward(self, x):
        feat = self.backbone(x)          # [B, 512]
        logits = self.head(feat)         # [B, n_bins]
        return logits, feat              # return features for saving

@torch.no_grad()
def predict_with_tta4(model, imgs, device, n_bins: int):
    """
    4-view TTA for survival prediction
    
    Views:
    - Original
    - H-flip (Height axis flip)
    - W-flip (Width axis flip)
    - HW-flip (Both axes flip)
    
    Shape verification:
    - img_batch: [B, C, D, H, W] where B=batch_size, C=2, D=128, H=192, W=160
    - dims=[-2] → H axis (192)
    - dims=[-1] → W axis (160)
    """
    # imgs: [B, 2, D, H, W]
    if not cfg.USE_TTA:
        model.eval()
        logits, _ = model(imgs.to(device))
        haz = torch.sigmoid(logits[:, :n_bins])
        haz_clamped = torch.clamp(haz, min=1e-7, max=1-1e-7)  
        log_surv = torch.cumsum(torch.log(1.0 - haz_clamped), dim=1)
        surv = torch.exp(log_surv)
        return surv.cpu().numpy()

    views = [
        imgs,
        torch.flip(imgs, dims=[-2]),       # H
        torch.flip(imgs, dims=[-1]),       # W
        torch.flip(imgs, dims=[-2,-1])     # HW
    ]
    log_survs = []
    model.eval()
    for v in views:
        logits, _ = model(v.to(device))
        haz = torch.sigmoid(logits[:, :n_bins])
        haz_clamped = torch.clamp(haz, min=1e-7, max=1-1e-7)
        log_surv = torch.cumsum(torch.log(1.0 - haz_clamped), dim=1)
        log_survs.append(log_surv)
    
    avg_log_surv = torch.stack(log_survs, 0).mean(0)
    surv = torch.exp(avg_log_surv)
    return surv.cpu().numpy()

# ================= Train one fold =================
def run_fold(fold: int, manifest: pd.DataFrame, idx_tr, idx_va, out_dir: Path) -> float:
    print(f"\n{'='*68}\nFOLD {fold}\n{'='*68}")
    times = manifest["time"].values
    events = manifest["event"].values

    # time bins
    n_bins = adaptive_bins(times[idx_tr], events[idx_tr])
    cuts = make_time_cuts(times[idx_tr], events[idx_tr], n_bins)
    lab = LabTransDiscreteTime(cuts=cuts[1:-1])  # internal boundaries
    y_tr = lab.fit_transform(times[idx_tr], events[idx_tr])
    y_va = lab.transform(times[idx_va], events[idx_va])
    n_time_bins = len(cuts) - 1
    bin_mids = (cuts[:-1] + cuts[1:]) / 2.0

    # datasets & loaders
    recs_tr = [{"patient_id": manifest.iloc[i].patient_id,
                "ct": manifest.iloc[i].ct_path,
                "mask": manifest.iloc[i].mask_path} for i in idx_tr]
    recs_va = [{"patient_id": manifest.iloc[i].patient_id,
                "ct": manifest.iloc[i].ct_path,
                "mask": manifest.iloc[i].mask_path} for i in idx_va]
    ds_tr = ImageOnlyDS(recs_tr, y_tr[0], y_tr[1], build_transforms(True))
    ds_va = ImageOnlyDS(recs_va, y_va[0], y_va[1], build_transforms(False))

    dl_tr = DataLoader(
        ds_tr, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True,
        num_workers=cfg.NUM_WORKERS, 
        pin_memory=True, 
        drop_last=True,
        persistent_workers=True, 
        prefetch_factor=cfg.PREFETCH_FACTOR
    )
    dl_va = DataLoader(
        ds_va, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=False,
        num_workers=cfg.NUM_WORKERS, 
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=cfg.PREFETCH_FACTOR
    )

    # model/opt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet10_ImageOnly_GN(in_channels=2, n_bins=n_time_bins, dropout=cfg.DROPOUT).to(device)

    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.ndim == 1 or 'bias' in n or 'norm' in n.lower():
                nodecay.append(p)
            else:
                decay.append(p)

    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.WD},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=cfg.LR
    )
    def lr_sched(epoch):
        if epoch < cfg.WARMUP_E: return (epoch+1)/cfg.WARMUP_E
        prog = (epoch - cfg.WARMUP_E) / max(1, (cfg.MAX_EPOCHS - cfg.WARMUP_E))
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_sched)
    loss_fn = NLLLogistiHazardLoss()

    # metrics setup
    y_tr_struct = to_struct(times[idx_tr], events[idx_tr])
    y_va_struct = to_struct(times[idx_va], events[idx_va])
    evt_times_tr = times[idx_tr][events[idx_tr]==1]
    tau = np.quantile(evt_times_tr, 0.9) if len(evt_times_tr)>0 else times[idx_tr].max()
    t_idx_24m = int(np.argmin(np.abs(bin_mids - cfg.EVAL_24M_DAYS)))

    # train
    fold_dir = out_dir / f"fold_{fold}"; fold_dir.mkdir(parents=True, exist_ok=True)
    best_uno = -1.0; no_improve = 0

    for ep in range(cfg.MAX_EPOCHS):
        model.train()
        total = 0.0; optim.zero_grad()
        for step, batch in enumerate(dl_tr):
            x = batch["image"].to(device)                  # [B, 2, D, H, W]
            y_idx = batch["label_idx"].to(device)
            y_evt = batch["label_event"].to(device)

            logits, _ = model(x)
            loss = loss_fn(logits, y_idx, y_evt) / cfg.ACCUM_STEPS

            if torch.isnan(loss).item():
                optim.zero_grad() 
                continue

            loss.backward()

            if (step+1) % cfg.ACCUM_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step() 
                optim.zero_grad()
            total += loss.item() * cfg.ACCUM_STEPS * x.size(0)

        if (step+1) % cfg.ACCUM_STEPS != 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step() 
            optim.zero_grad()
        sched.step()
        tr_loss = total / len(idx_tr)

        # ---- validate ----
        model.eval()
        surv_all = []
        feats_all = []
        pids_all = []
        with torch.no_grad():
            for batch in dl_va:
                x = batch["image"]
                surv = predict_with_tta4(model, x, device, n_time_bins)
                surv_all.append(surv)

                # extract features once (use original, no TTA)
                logits, feats = model(x.to(device))
                feats_all.append(feats.cpu().numpy())
                pids_all.extend(batch["patient_id"])

        surv = np.concatenate(surv_all, 0)                    # [Nv, n_bins]
        feats = np.concatenate(feats_all, 0)                  # [Nv, 512]

        # risks
        risk_int = -np.trapz(surv, x=bin_mids, axis=1)
        surv_24m = surv[:, t_idx_24m] if t_idx_24m < surv.shape[1] else surv[:, -1]
        risk_24m = 1.0 - surv_24m

        uno_int = concordance_index_ipcw(y_tr_struct, y_va_struct, risk_int, tau=tau)[0]
        uno_24m = concordance_index_ipcw(y_tr_struct, y_va_struct, risk_24m, tau=tau)[0]
        harrell = concordance_index_censored(y_va_struct["event"],
                                             y_va_struct["time"], risk_int)[0]
        try:
            _, brier = brier_score(y_tr_struct, y_va_struct,
                                   surv_24m.reshape(-1,1), np.array([cfg.EVAL_24M_DAYS]))
            brier_24m = float(brier[0])
        except Exception:
            brier_24m = np.nan

        print(f"Epoch {ep+1:03d} | loss {tr_loss:.4f} | Uno(int) {uno_int:.3f} | "
              f"Uno(24m) {uno_24m:.3f} | Harrell {harrell:.3f} | Brier@24m "
              f"{'%.3f'%brier_24m if not np.isnan(brier_24m) else 'N/A'} "
              f"| TTA={'ON' if cfg.USE_TTA else 'OFF'}")

        # model selection on Uno(int)
        if not np.isnan(uno_int) and uno_int > best_uno + 1e-5:
            best_uno = uno_int 
            no_improve = 0
            ckpt = {
                "state_dict": model.state_dict(),
                "cuts": cuts.tolist(),
                "epoch": ep,
                "uno_integral": uno_int,
                "uno_24m": uno_24m,
                "harrell": harrell,
                "lr": cfg.LR,
                "use_tta": cfg.USE_TTA,
                "n_time_bins": n_time_bins
            }
            torch.save(ckpt, fold_dir/"best.pt")

            # save preds and embeddings
            pd.DataFrame({
                "patient_id": manifest.iloc[idx_va]["patient_id"].values,
                "risk_integral": risk_int,
                "surv_24m": surv_24m,
                "time": times[idx_va],
                "event": events[idx_va]
            }).to_csv(fold_dir/"val_pred.csv", index=False)
            np.save(fold_dir/"val_embeddings.npy", feats)
            pd.Series(pids_all).to_csv(fold_dir/"val_pids.csv", index=False, header=False)
        else:
            no_improve += 1

        if no_improve >= cfg.EARLY_STOP:
            print(f"Early stopping at epoch {ep+1}")
            break

    return float(best_uno)

# ================= Main =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None, help="specific fold 0..4")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--lr", type=float, default=cfg.LR)
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--exp_name", type=str, default=None)
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    cfg.LR = args.lr
    cfg.USE_TTA = not args.no_tta
    set_seed(cfg.SEED)

    # Load crop log (paths) and clinical (labels only: time/event)
    crop = pd.read_csv(PHASE2_CROP_LOG)
    # accept either out_ct/mask or ct_path/mask_path columns
    if "ct_path" not in crop.columns and "out_ct" in crop.columns:
        crop = crop.rename(columns={"out_ct":"ct_path", "out_mask":"mask_path"})
    manifest = crop[["patient_id","ct_path","mask_path"]].copy()

    clin = pd.read_csv(CLINICAL_CSV)
    # normalize common column names
    ren = {}
    for c in clin.columns:
        s = c.lower().strip()
        if "patient" in s or "case" in s: ren[c] = "patient_id"
        elif "survival" in s and "time" in s: ren[c] = "time"
        elif "event" in s or "dead" in s or "status" in s: ren[c] = "event"
    clin = clin.rename(columns=ren)
    clin = clin[["patient_id","time","event"]].dropna(subset=["time","event"])

    manifest = manifest.merge(clin, on="patient_id", how="inner").copy()
    manifest["time"] = manifest["time"].astype(float)
    manifest["event"] = manifest["event"].astype(int)

    print("\n=========== PHASE 3: IMAGE-ONLY (CT+MASK) ===========")
    print(f"Dataset: {len(manifest)} patients | Events: {manifest['event'].sum()} "
          f"({manifest['event'].mean()*100:.1f}%) | Median T: {manifest['time'].median():.0f} days")
    print(f"LR={cfg.LR:.2e} | TTA={'ON' if cfg.USE_TTA else 'OFF'}")

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    exp = args.exp_name or f"imgonly_resnet10_lr{cfg.LR:.0e}" + ("_TTA" if cfg.USE_TTA else "_noTTA")
    out_dir = Path(OUTPUT_ROOT)/exp; out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for k,(tr,va) in enumerate(skf.split(np.zeros(len(manifest)), manifest["event"])):
        if args.fold is not None and k != args.fold: continue
        best = run_fold(k, manifest, tr, va, out_dir)
        results.append({"fold": k, "best_uno": best})

    if len(results):
        df = pd.DataFrame(results)
        if args.fold is not None:
            df.to_csv(out_dir/f"results_fold{args.fold}.csv", index=False)
            print(f"Fold {args.fold}: Uno={df['best_uno'].iloc[0]:.3f}")
        else:
            df.to_csv(out_dir/"results_all.csv", index=False)
            print("\nFinal:")
            print(df)
            print(f"Mean Uno: {df['best_uno'].mean():.3f} ± {df['best_uno'].std():.3f}")

if __name__ == "__main__":
    main()