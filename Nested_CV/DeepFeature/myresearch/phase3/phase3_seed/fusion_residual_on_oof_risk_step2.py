#!/usr/bin/env python3
# fusion_residual_on_oof_risk_step2_fixed.py
import os, glob, argparse, json
from collections import Counter
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

# Optional Uno C-index (IPCW)
_HAS_SKSURV = False
try:
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_ipcw
    _HAS_SKSURV = True
except Exception:
    _HAS_SKSURV = False

# Optional sklearn encoders/scalers
_HAS_SKLEARN = False
try:
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


STAGE_ORDER = ["I", "II", "IIIa", "IIIb"]
STAGE_TO_ORD = {s: i + 1 for i, s in enumerate(STAGE_ORDER)}

def _norm_stage(s: object) -> Optional[str]:
    if not isinstance(s, str):
        return None
    x = s.strip()
    if x == "":
        return None
    x_u = x.upper().replace(" ", "")
    if x_u in ("I", "IA", "IB"):
        return "I"
    if x_u in ("II", "IIA", "IIB"):
        return "II"
    if x_u in ("IIIA", "III A"):
        return "IIIa"
    if x_u in ("IIIB", "III B"):
        return "IIIb"
    if x in STAGE_ORDER:
        return x
    for t in STAGE_ORDER:
        if x.lower() == t.lower():
            return t
    return x  # unknown label


def resolve_col(df: pd.DataFrame, preferred: Optional[str], candidates: List[str]) -> str:
    cols = list(df.columns)
    if preferred:
        if preferred in df.columns:
            return preferred
        low_map = {c.lower(): c for c in cols}
        if preferred.lower() in low_map:
            return low_map[preferred.lower()]
        raise KeyError(f"Column '{preferred}' not found. Available columns: {cols}")

    low_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in low_map:
            return low_map[cand.lower()]
    raise KeyError(f"None of candidates {candidates} found. Available columns: {cols}")


def load_pids(path: str) -> List[str]:
    df = pd.read_csv(path)
    for c in ["pid", "patient_id", "PatientID", "PID"]:
        if c in df.columns:
            return df[c].astype(str).tolist()
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str).tolist()
    raise ValueError(f"Cannot find pid column in {path}, columns={list(df.columns)}")


def harrell_c_index(times: np.ndarray, events: np.ndarray, scores: np.ndarray) -> float:
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    s = np.asarray(scores, dtype=float)

    n = len(t)
    conc = 0.0
    ties = 0.0
    total = 0.0
    for i in range(n):
        if e[i] != 1:
            continue
        for j in range(n):
            if t[j] <= t[i]:
                continue
            total += 1.0
            if s[i] > s[j]:
                conc += 1.0
            elif s[i] == s[j]:
                ties += 1.0
    if total == 0:
        return float("nan")
    return (conc + 0.5 * ties) / total


def uno_c_index_ipcw(train_time, train_event, val_time, val_event, val_scores) -> Optional[float]:
    if not _HAS_SKSURV:
        return None
    y_tr = Surv.from_arrays(event=train_event.astype(bool), time=train_time.astype(float))
    y_va = Surv.from_arrays(event=val_event.astype(bool), time=val_time.astype(float))
    # concordance_index_ipcw returns (cindex, concordant, discordant, tied_risk, tied_time)
    c = concordance_index_ipcw(y_tr, y_va, val_scores.astype(float))[0]
    return float(c)


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(time, descending=True)
    r = risk[order]
    e = event[order]
    log_cumsum = torch.logcumsumexp(r, dim=0)
    idx = (e > 0.5)
    if idx.sum() == 0:
        return torch.zeros((), device=risk.device)
    return -(r[idx] - log_cumsum[idx]).mean()


class DeltaMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.dp = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dp(self.act(self.fc1(x)))
        return self.fc2(x).squeeze(-1)


class ResidualOnRisk(nn.Module):
    def __init__(self, clin_dim: int, hidden: int = 64, dropout: float = 0.5, alpha_init: float = -6.0):
        super().__init__()
        self.delta = DeltaMLP(clin_dim, hidden=hidden, dropout=dropout)
        self.raw_alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))  # softplus -> >=0

    def forward(self, base_risk: torch.Tensor, clin_x: torch.Tensor):
        alpha = torch.nn.functional.softplus(self.raw_alpha)
        delta = self.delta(clin_x)
        fused = base_risk + alpha * delta
        return fused, alpha, delta


def build_clin_features(
    clin: pd.DataFrame,
    pids_train: List[str],
    pids_val: List[str],
    pid_col: str,
    time_col: str,
    event_col: str,
    overall_stage_col: str,
    t_col: str,
    n_col: str,
    m_col: str,
    age_col: str,
    gender_col: str,
    histology_col: Optional[str],
    overall_stage_encoding: str,
    add_histology: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:

    clin = clin.copy()
    clin[pid_col] = clin[pid_col].astype(str)
    clin_idx = clin.set_index(pid_col, drop=False)

    def take(pids: List[str], col: str):
        return clin_idx.loc[pids, col].values

    t_tr = take(pids_train, time_col).astype(float)
    t_va = take(pids_val, time_col).astype(float)
    e_tr = take(pids_train, event_col).astype(float)
    e_va = take(pids_val, event_col).astype(float)

    age_tr = take(pids_train, age_col).astype(float)
    age_va = take(pids_val, age_col).astype(float)
    # Fill missing age with train mean
    age_mean = np.nanmean(age_tr)
    age_tr = np.where(np.isnan(age_tr), age_mean, age_tr)
    age_va = np.where(np.isnan(age_va), age_mean, age_va)

    T_tr = take(pids_train, t_col).astype(float)
    T_va = take(pids_val, t_col).astype(float)
    tr_vals = [float(x) for x in T_tr if not np.isnan(x)]
    mode_T = Counter(tr_vals).most_common(1)[0][0] if len(tr_vals) > 0 else 2.0
    T_tr = np.where(np.isnan(T_tr), mode_T, T_tr)
    T_va = np.where(np.isnan(T_va), mode_T, T_va)

    N_tr = take(pids_train, n_col).astype(float)
    N_va = take(pids_val, n_col).astype(float)
    # Fill missing N with train mode
    tr_n_vals = [float(x) for x in N_tr if not np.isnan(x)]
    mode_N = Counter(tr_n_vals).most_common(1)[0][0] if len(tr_n_vals) > 0 else 0.0
    N_tr = np.where(np.isnan(N_tr), mode_N, N_tr)
    N_va = np.where(np.isnan(N_va), mode_N, N_va)

    M_tr = take(pids_train, m_col).astype(float)
    M_va = take(pids_val, m_col).astype(float)
    # Fill missing M with train mode
    tr_m_vals = [float(x) for x in M_tr if not np.isnan(x)]
    mode_M = Counter(tr_m_vals).most_common(1)[0][0] if len(tr_m_vals) > 0 else 0.0
    M_tr = np.where(np.isnan(M_tr), mode_M, M_tr)
    M_va = np.where(np.isnan(M_va), mode_M, M_va)
    M_tr_bin = (M_tr != 0).astype(np.float32)
    M_va_bin = (M_va != 0).astype(np.float32)

    g_tr_raw = take(pids_train, gender_col)
    g_va_raw = take(pids_val, gender_col)

    def norm_gender(x):
        if not isinstance(x, str):
            return None
        y = x.strip().lower()
        if y in ("m", "male", "man"):
            return 1.0
        if y in ("f", "female", "woman"):
            return 0.0
        return None

    g_tr = np.array([norm_gender(x) for x in g_tr_raw], dtype=float)
    g_va = np.array([norm_gender(x) for x in g_va_raw], dtype=float)
    tr_g_vals = [float(x) for x in g_tr.tolist() if not np.isnan(x)]
    mode_g = Counter(tr_g_vals).most_common(1)[0][0] if len(tr_g_vals) > 0 else 1.0
    g_tr = np.where(np.isnan(g_tr), mode_g, g_tr).astype(np.float32)
    g_va = np.where(np.isnan(g_va), mode_g, g_va).astype(np.float32)

    os_tr_raw = take(pids_train, overall_stage_col)
    os_va_raw = take(pids_val, overall_stage_col)
    os_tr = np.array([_norm_stage(x) for x in os_tr_raw], dtype=object)
    os_va = np.array([_norm_stage(x) for x in os_va_raw], dtype=object)
    tr_stage_vals = [x for x in os_tr.tolist() if x is not None]
    mode_stage = Counter(tr_stage_vals).most_common(1)[0][0] if len(tr_stage_vals) > 0 else "IIIa"
    os_tr = np.array([x if x is not None else mode_stage for x in os_tr], dtype=object)
    os_va = np.array([x if x is not None else mode_stage for x in os_va], dtype=object)

    his_tr = his_va = None
    if add_histology and histology_col is not None:
        his_tr_raw = take(pids_train, histology_col)
        his_va_raw = take(pids_val, histology_col)

        def norm_his(x):
            if not isinstance(x, str):
                return "Unknown"
            y = x.strip()
            return y if y != "" else "Unknown"

        his_tr = np.array([norm_his(x) for x in his_tr_raw], dtype=object)
        his_va = np.array([norm_his(x) for x in his_va_raw], dtype=object)

    num_tr = np.stack([age_tr, T_tr, N_tr, M_tr_bin, g_tr], axis=1).astype(np.float32)
    num_va = np.stack([age_va, T_va, N_va, M_va_bin, g_va], axis=1).astype(np.float32)
    num_names = ["age", "T", "N", "M_bin", "gender_bin"]

    cat_tr = None
    cat_va = None
    cat_names = []

    if overall_stage_encoding == "ordinal":
        def to_ord(x):
            if x in STAGE_TO_ORD:
                return float(STAGE_TO_ORD[x])
            return float(STAGE_TO_ORD.get(mode_stage, 3))
        os_tr_ord = np.array([to_ord(x) for x in os_tr], dtype=np.float32)
        os_va_ord = np.array([to_ord(x) for x in os_va], dtype=np.float32)
        num_tr = np.concatenate([num_tr, os_tr_ord[:, None]], axis=1)
        num_va = np.concatenate([num_va, os_va_ord[:, None]], axis=1)
        num_names.append("OverallStage_ord")

    elif overall_stage_encoding == "onehot":
        if not _HAS_SKLEARN:
            raise RuntimeError("onehot requires scikit-learn; use --overall_stage_encoding ordinal instead.")
        try:
            ohe_stage = OneHotEncoder(categories=[STAGE_ORDER], handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe_stage = OneHotEncoder(categories=[STAGE_ORDER], handle_unknown="ignore", sparse=False)
        cat_tr = ohe_stage.fit_transform(os_tr.reshape(-1, 1))
        cat_va = ohe_stage.transform(os_va.reshape(-1, 1))
        cat_names += [f"OverallStage_{s}" for s in STAGE_ORDER]
    else:
        raise ValueError("overall_stage_encoding must be ordinal or onehot")

    if add_histology and histology_col is not None:
        if not _HAS_SKLEARN:
            raise RuntimeError("histology requires scikit-learn; set --add_histology 0.")
        try:
            ohe_his = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe_his = OneHotEncoder(handle_unknown="ignore", sparse=False)
        h_tr = ohe_his.fit_transform(his_tr.reshape(-1, 1))
        h_va = ohe_his.transform(his_va.reshape(-1, 1))
        if cat_tr is None:
            cat_tr, cat_va = h_tr, h_va
        else:
            cat_tr = np.concatenate([cat_tr, h_tr], axis=1)
            cat_va = np.concatenate([cat_va, h_va], axis=1)
        cat_names += [f"Histology_{c}" for c in ohe_his.categories_[0].tolist()]

    if _HAS_SKLEARN:
        scaler = StandardScaler()
        num_tr_z = scaler.fit_transform(num_tr)
        num_va_z = scaler.transform(num_va)
    else:
        mean = num_tr.mean(axis=0, keepdims=True)
        std = num_tr.std(axis=0, keepdims=True) + 1e-8
        num_tr_z = (num_tr - mean) / std
        num_va_z = (num_va - mean) / std

    if cat_tr is None:
        X_tr = num_tr_z.astype(np.float32)
        X_va = num_va_z.astype(np.float32)
        feat_names = [f"{n}_z" for n in num_names]
    else:
        X_tr = np.concatenate([num_tr_z, cat_tr], axis=1).astype(np.float32)
        X_va = np.concatenate([num_va_z, cat_va], axis=1).astype(np.float32)
        feat_names = [f"{n}_z" for n in num_names] + cat_names

    return X_tr, X_va, t_tr, e_tr, t_va, e_va, feat_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--clinical_csv", required=True)
    ap.add_argument("--oof_csv", default=None)
    ap.add_argument("--out_name", default="step2_residual_on_oof_risk_fixed")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--alpha_lr_mult", type=float, default=5.0)
    ap.add_argument("--wd_clin", type=float, default=1e-2)
    ap.add_argument("--delta_hidden", type=int, default=64)
    ap.add_argument("--delta_dropout", type=float, default=0.5)
    ap.add_argument("--alpha_init", type=float, default=-6.0, help="Try -6/-4/-2")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--pid_col", default=None)
    ap.add_argument("--time_col", default=None)
    ap.add_argument("--event_col", default=None)

    ap.add_argument("--age_col", default="age")
    ap.add_argument("--gender_col", default="gender")
    ap.add_argument("--t_col", default="clinical.T.Stage")
    ap.add_argument("--n_col", default="Clinical.N.Stage")
    ap.add_argument("--m_col", default="Clinical.M.Stage")
    ap.add_argument("--overall_stage_col", default="Overall.Stage")
    ap.add_argument("--histology_col", default="Histology")

    ap.add_argument("--overall_stage_encoding", choices=["ordinal", "onehot"], default="ordinal")
    ap.add_argument("--add_histology", type=int, default=0)
    ap.add_argument("--zscore_base", type=int, default=1)
    ap.add_argument("--flip_risk", type=int, default=0)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_dir = args.exp_dir
    oof_csv = args.oof_csv or os.path.join(exp_dir, "oof_risk_integral.csv")
    out_dir = os.path.join(exp_dir, args.out_name + f"_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    oof = pd.read_csv(oof_csv)
    pid_col_oof = "pid"
    if pid_col_oof not in oof.columns:
        for c in ["pid", "patient_id", "PatientID", "PID"]:
            if c in oof.columns:
                pid_col_oof = c
                break
    assert pid_col_oof in oof.columns
    assert "oof_risk_integral" in oof.columns
    oof_map: Dict[str, float] = dict(zip(oof[pid_col_oof].astype(str), oof["oof_risk_integral"].astype(float)))

    clin = pd.read_csv(args.clinical_csv)
    pid_col = resolve_col(clin, args.pid_col, ["pid", "patient_id", "PatientID", "PID"])
    time_col = resolve_col(clin, args.time_col, ["Survival.time", "survival.time", "time"])
    event_col = resolve_col(clin, args.event_col, ["deadstatus.event", "event", "status"])

    age_col = resolve_col(clin, args.age_col, ["age"])
    gender_col = resolve_col(clin, args.gender_col, ["gender", "sex"])
    t_col = resolve_col(clin, args.t_col, ["clinical.T.Stage", "Clinical.T.Stage", "clinical.t.stage"])
    n_col = resolve_col(clin, args.n_col, ["Clinical.N.Stage", "clinical.N.Stage", "clinical.n.stage"])
    m_col = resolve_col(clin, args.m_col, ["Clinical.M.Stage", "clinical.M.Stage", "clinical.m.stage"])
    overall_stage_col = resolve_col(clin, args.overall_stage_col, ["Overall.Stage", "overall.stage", "overall_stage"])
    histology_col = None
    if args.add_histology:
        try:
            histology_col = resolve_col(clin, args.histology_col, ["Histology", "histology"])
        except Exception:
            histology_col = None

    print("[CLIN] resolved columns:")
    print(f"  pid={pid_col} | time={time_col} | event={event_col}")
    print(f"  age={age_col} | gender={gender_col} | T={t_col} | N={n_col} | M={m_col} | OverallStage={overall_stage_col} | histology={histology_col}")
    print(f"  overall_stage_encoding={args.overall_stage_encoding} | add_histology={bool(args.add_histology)}")
    if args.overall_stage_encoding == "onehot" and not _HAS_SKLEARN:
        raise RuntimeError("onehot requires scikit-learn; use ordinal or install sklearn.")

    fold_dirs = sorted([d for d in glob.glob(os.path.join(exp_dir, "fold_*")) if os.path.isdir(d)])
    assert len(fold_dirs) > 0, f"no fold_* under {exp_dir}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    fold_best = []
    fold_base = []
    fold_details = []

    clin_pid_set = set(clin[pid_col].astype(str).tolist())

    for fd in fold_dirs:
        fold = os.path.basename(fd)

        tr_p_path = os.path.join(fd, "train_pids.csv")
        va_p_path = os.path.join(fd, "val_pids.csv")
        if os.path.exists(tr_p_path) and os.path.exists(va_p_path):
            tr_pids = load_pids(tr_p_path)
            va_pids = load_pids(va_p_path)
        else:
            vp = os.path.join(fd, "val_pred.csv")
            vdf = pd.read_csv(vp)
            pid_in_vp = None
            for c in ["pid", "patient_id", "PatientID", "PID"]:
                if c in vdf.columns:
                    pid_in_vp = c
                    break
            assert pid_in_vp is not None
            va_pids = vdf[pid_in_vp].astype(str).tolist()
            all_pids = clin[pid_col].astype(str).tolist()
            va_set = set(va_pids)
            tr_pids = [p for p in all_pids if p not in va_set]

        tr_pids = [p for p in tr_pids if (p in oof_map and p in clin_pid_set)]
        va_pids = [p for p in va_pids if (p in oof_map and p in clin_pid_set)]

        X_tr, X_va, t_tr, e_tr, t_va, e_va, feat_names = build_clin_features(
            clin=clin,
            pids_train=tr_pids,
            pids_val=va_pids,
            pid_col=pid_col,
            time_col=time_col,
            event_col=event_col,
            overall_stage_col=overall_stage_col,
            t_col=t_col,
            n_col=n_col,
            m_col=m_col,
            age_col=age_col,
            gender_col=gender_col,
            histology_col=histology_col,
            overall_stage_encoding=args.overall_stage_encoding,
            add_histology=bool(args.add_histology),
        )

        base_tr = np.array([oof_map[p] for p in tr_pids], dtype=np.float32)
        base_va = np.array([oof_map[p] for p in va_pids], dtype=np.float32)

        if args.flip_risk:
            base_tr = -base_tr
            base_va = -base_va

        if args.zscore_base:
            mu = float(np.mean(base_tr))
            sd = float(np.std(base_tr) + 1e-8)
            base_tr = (base_tr - mu) / sd
            base_va = (base_va - mu) / sd

        # Check for NaN in features
        if np.isnan(X_tr).any():
            print(f"[{fold}] WARNING: NaN in X_tr. Replacing with 0.")
            X_tr = np.nan_to_num(X_tr, nan=0.0)
        if np.isnan(X_va).any():
            print(f"[{fold}] WARNING: NaN in X_va. Replacing with 0.")
            X_va = np.nan_to_num(X_va, nan=0.0)
        if np.isnan(base_tr).any():
            print(f"[{fold}] WARNING: NaN in base_tr. Replacing with mean.")
            base_tr = np.nan_to_num(base_tr, nan=np.nanmean(base_tr))
        if np.isnan(base_va).any():
            print(f"[{fold}] WARNING: NaN in base_va. Replacing with mean.")
            base_va = np.nan_to_num(base_va, nan=np.nanmean(base_va))

        X_tr_t = torch.from_numpy(X_tr).to(device)
        X_va_t = torch.from_numpy(X_va).to(device)
        base_tr_t = torch.from_numpy(base_tr).to(device)
        base_va_t = torch.from_numpy(base_va).to(device)
        t_tr_t = torch.from_numpy(t_tr.astype(np.float32)).to(device)
        e_tr_t = torch.from_numpy(e_tr.astype(np.float32)).to(device)

        model = ResidualOnRisk(
            clin_dim=X_tr.shape[1],
            hidden=args.delta_hidden,
            dropout=args.delta_dropout,
            alpha_init=args.alpha_init,
        ).to(device)

        opt = optim.Adam(
            [
                {"params": model.delta.parameters(), "lr": args.lr, "weight_decay": args.wd_clin},
                {"params": [model.raw_alpha], "lr": args.lr * args.alpha_lr_mult, "weight_decay": 0.0},
            ]
        )

        base_har = harrell_c_index(t_va, e_va, base_va)
        base_uno = uno_c_index_ipcw(t_tr, e_tr.astype(bool), t_va, e_va.astype(bool), base_va) if _HAS_SKSURV else None

        print(f"[{fold}] clin_dim={X_tr.shape[1]} | n_tr={len(tr_pids)} n_va={len(va_pids)} | base_har={base_har:.3f}" +
              (f" base_uno={base_uno:.3f}" if base_uno is not None else ""))

        best_score = -1e9
        best_ep = -1
        best_state = None
        bad = 0

        for ep in range(1, args.epochs + 1):
            model.train()
            fused_tr, alpha, _ = model(base_tr_t, X_tr_t)
            loss = cox_ph_loss(fused_tr, t_tr_t, e_tr_t)

            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"[{fold}] WARNING: NaN loss at epoch {ep}. Stopping training.")
                break

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            model.eval()
            with torch.no_grad():
                fused_va_t, alpha_t, _ = model(base_va_t, X_va_t)
                fused_va = fused_va_t.detach().cpu().numpy()
                alpha_val = float(alpha_t.detach().cpu().item())

            # Check for NaN in predictions
            if np.isnan(fused_va).any():
                print(f"[{fold}] WARNING: NaN in predictions at epoch {ep}. Using baseline scores.")
                fused_va = base_va.copy()
                har_fuse = base_har
                uno_fuse = base_uno
            else:
                har_fuse = harrell_c_index(t_va, e_va, fused_va)
                uno_fuse = uno_c_index_ipcw(t_tr, e_tr.astype(bool), t_va, e_va.astype(bool), fused_va) if _HAS_SKSURV else None

            if uno_fuse is not None:
                print(f"[{fold}] Ep {ep:03d} | loss {loss.item():.4f} | Uno base {base_uno:.3f} -> fuse {uno_fuse:.3f} | "
                      f"Har base {base_har:.3f} -> fuse {har_fuse:.3f} | alpha {alpha_val:.5f}")
                monitor = uno_fuse
            else:
                print(f"[{fold}] Ep {ep:03d} | loss {loss.item():.4f} | Har base {base_har:.3f} -> fuse {har_fuse:.3f} | alpha {alpha_val:.5f}")
                monitor = har_fuse

            if monitor > best_score + 1e-6:
                best_score = monitor
                best_ep = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"[{fold}] Early stop at ep {ep} (best_ep={best_ep}, best_score={best_score:.4f})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            fused_va_t, alpha_t, _ = model(base_va_t, X_va_t)
            fused_va = fused_va_t.detach().cpu().numpy()
            alpha_final = float(alpha_t.detach().cpu().item())

        har_best = harrell_c_index(t_va, e_va, fused_va)
        uno_best = uno_c_index_ipcw(t_tr, e_tr.astype(bool), t_va, e_va.astype(bool), fused_va) if _HAS_SKSURV else None

        base_score = base_uno if (uno_best is not None) else base_har
        best_score_out = uno_best if (uno_best is not None) else har_best

        fold_best.append(best_score_out)
        fold_base.append(base_score)

        fold_info = {
            "fold": fold,
            "best_ep": best_ep,
            "alpha": alpha_final,
            "base_uno": float(base_uno) if base_uno is not None else None,
            "fuse_uno": float(uno_best) if uno_best is not None else None,
            "base_har": float(base_har),
            "fuse_har": float(har_best),
            "clin_dim": int(X_tr.shape[1]),
            "feature_names": feat_names,
        }
        fold_details.append(fold_info)

        with open(os.path.join(out_dir, f"{fold}_best.json"), "w") as f:
            json.dump(fold_info, f, indent=2)

    arr = np.array(fold_best, dtype=float)
    base_arr = np.array(fold_base, dtype=float)
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    base_mean = float(np.nanmean(base_arr))
    base_std = float(np.nanstd(base_arr))

    print("")
    if _HAS_SKSURV:
        print(f"[Residual-on-OOF-risk] mean Uno(best) = {mean:.3f} ± {std:.3f} | base = {base_mean:.3f} ± {base_std:.3f}")
    else:
        print(f"[Residual-on-OOF-risk] mean Harrell(best) = {mean:.3f} ± {std:.3f} | base = {base_mean:.3f} ± {base_std:.3f}")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(
            {"mean": mean, "std": std, "base_mean": base_mean, "base_std": base_std, "has_sksurv": _HAS_SKSURV, "folds": fold_details},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
