# r5_bagged_ridge.py
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw


def make_structured_y(time_arr, event_arr):
    y = np.zeros(len(time_arr), dtype=[("event", bool), ("time", float)])
    y["event"] = event_arr.astype(bool)
    y["time"] = time_arr.astype(float)
    return y


def compute_tau_from_train(y_train, y_val, q=0.90, eps=1e-6):
    event_times = y_train["time"][y_train["event"]]
    if len(event_times) == 0:
        return float(np.max(y_val["time"]) - eps)
    tau_train = float(np.quantile(event_times, q))
    tau_train = min(tau_train, float(np.max(event_times)) - eps)
    tau_val_cap = float(np.max(y_val["time"]) - eps)
    tau = min(tau_train, tau_val_cap)
    if not np.isfinite(tau) or tau <= 0:
        tau = float(np.max(y_val["time"]) - eps)
    return tau


def eval_fold(y_train, y_val, risk_val, tau):
    harrell = float(concordance_index_censored(y_val["event"], y_val["time"], risk_val)[0])
    uno = float(concordance_index_ipcw(y_train, y_val, risk_val, tau=tau)[0])
    return harrell, uno


def alpha_selection_best(alphas, mean_scores, valid_mask):
    idxs = np.where(valid_mask)[0]
    if idxs.size == 0:
        return None
    return int(idxs[int(np.nanargmax(mean_scores[idxs]))])


def alpha_selection_1se(alphas, mean_scores, se_scores, valid_mask):
    idx_best = alpha_selection_best(alphas, mean_scores, valid_mask)
    if idx_best is None:
        return None
    thresh = mean_scores[idx_best] - se_scores[idx_best]
    ok = np.where((mean_scores >= thresh) & valid_mask)[0]
    if ok.size == 0:
        return int(idx_best)
    # 1SE：选更大的alpha（更强正则）=> 更稳
    return int(ok[int(np.argmax(alphas[ok]))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--id_col", type=str, default="PatientID")
    ap.add_argument("--time_col", type=str, default="Survival.time")
    ap.add_argument("--event_col", type=str, default="deadstatus.event")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--inner_metric", type=str, choices=["harrell", "uno"], default="uno")
    ap.add_argument("--alpha_select", type=str, choices=["best", "1se"], default="1se")

    ap.add_argument("--l1_ratio", type=float, default=0.0, help="0.0=ridge, 0.1~0.3=weak EN")
    ap.add_argument("--n_alphas", type=int, default=80)
    ap.add_argument("--alpha_min_ratio", type=float, default=1e-3)
    ap.add_argument("--max_iter", type=int, default=200000)
    ap.add_argument("--tol", type=float, default=1e-7)

    ap.add_argument("--tau_q", type=float, default=0.90)

    ap.add_argument("--n_bags", type=int, default=32, help="bagging次数，先用32，想更稳用64")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ---- load features ----
    feat_df = pd.read_csv(args.features_csv)
    if args.id_col not in feat_df.columns:
        raise ValueError(f"features_csv missing {args.id_col}")

    feat_df[args.id_col] = feat_df[args.id_col].astype(str)
    feat_cols = [c for c in feat_df.columns if c != args.id_col]

    X_all = feat_df[feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    pids_all = feat_df[args.id_col].to_numpy(dtype=str)

    # ---- load labels ----
    lab_df = pd.read_csv(args.labels_csv).copy()
    lab_df[args.id_col] = lab_df[args.id_col].astype(str)
    lab_df = lab_df[lab_df[args.id_col].isin(set(pids_all))].copy()
    lab_df.sort_values(args.id_col, inplace=True)

    idx_map = {pid: i for i, pid in enumerate(pids_all)}
    idx = np.array([idx_map[pid] for pid in lab_df[args.id_col].to_numpy(dtype=str)], dtype=int)

    X_all = X_all[idx]
    pids = lab_df[args.id_col].to_numpy(dtype=str)

    time_arr = pd.to_numeric(lab_df[args.time_col], errors="coerce").to_numpy(dtype=float)
    event_arr = pd.to_numeric(lab_df[args.event_col], errors="coerce").fillna(0).astype(int).to_numpy()
    y_all = make_structured_y(time_arr, event_arr)

    # ---- splits ----
    splits = json.load(open(args.splits_json, "r"))
    fold_keys = sorted(list(splits.keys()))

    rows = []
    for fk in fold_keys:
        fold_dir = outdir / fk
        fold_dir.mkdir(parents=True, exist_ok=True)

        tr_set = set(map(str, splits[fk]["train"]))
        va_set = set(map(str, splits[fk]["val"]))

        tr_idx = np.array([i for i, pid in enumerate(pids) if pid in tr_set], dtype=int)
        va_idx = np.array([i for i, pid in enumerate(pids) if pid in va_set], dtype=int)
        if tr_idx.size == 0 or va_idx.size == 0:
            raise RuntimeError(f"{fk}: empty train/val after align")

        X_tr_raw = X_all[tr_idx].copy()
        X_va_raw = X_all[va_idx].copy()
        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]

        # ---- impute (train median) ----
        X_tr_raw = np.where(np.isfinite(X_tr_raw), X_tr_raw, np.nan)
        X_va_raw = np.where(np.isfinite(X_va_raw), X_va_raw, np.nan)
        med = np.nanmedian(X_tr_raw, axis=0)
        ok = np.isfinite(med)
        X_tr_raw = X_tr_raw[:, ok]
        X_va_raw = X_va_raw[:, ok]
        med = med[ok]
        X_tr_raw = np.where(np.isfinite(X_tr_raw), X_tr_raw, med)
        X_va_raw = np.where(np.isfinite(X_va_raw), X_va_raw, med)

        # ---- standardize (train only) ----
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)

        # ---- choose alpha by inner CV ----
        base = CoxnetSurvivalAnalysis(
            l1_ratio=float(args.l1_ratio),
            alphas=None,
            n_alphas=int(args.n_alphas),
            alpha_min_ratio=float(args.alpha_min_ratio),
            max_iter=int(args.max_iter),
            tol=float(args.tol),
            fit_baseline_model=False,
        )
        base.fit(X_tr, y_tr)
        alphas = np.array(base.alphas_, dtype=float)

        inner = KFold(n_splits=int(args.inner_folds), shuffle=True, random_state=int(args.seed))
        score_mat = np.full((int(args.inner_folds), len(alphas)), np.nan, dtype=float)

        for j, (itr, iva) in enumerate(inner.split(X_tr)):
            Xi_tr, Xi_va = X_tr[itr], X_tr[iva]
            yi_tr, yi_va = y_tr[itr], y_tr[iva]

            mdl = CoxnetSurvivalAnalysis(
                l1_ratio=float(args.l1_ratio),
                alphas=alphas,
                max_iter=int(args.max_iter),
                tol=float(args.tol),
                fit_baseline_model=False,
            )
            mdl.fit(Xi_tr, yi_tr)

            if args.inner_metric == "uno":
                tau_i = compute_tau_from_train(yi_tr, yi_va, q=float(args.tau_q))
            else:
                tau_i = None

            for k, a in enumerate(alphas):
                r = mdl.predict(Xi_va, alpha=float(a))
                if args.inner_metric == "harrell":
                    score_mat[j, k] = concordance_index_censored(yi_va["event"], yi_va["time"], r)[0]
                else:
                    score_mat[j, k] = concordance_index_ipcw(yi_tr, yi_va, r, tau=tau_i)[0]

        n_valid = np.sum(np.isfinite(score_mat), axis=0).astype(int)
        mean_scores = np.nanmean(score_mat, axis=0)
        std_scores = np.nanstd(score_mat, axis=0, ddof=1)
        se_scores = std_scores / np.sqrt(np.maximum(n_valid, 1))
        valid_mask = np.isfinite(mean_scores) & (n_valid >= 2)

        if args.alpha_select == "best":
            idx_sel = alpha_selection_best(alphas, mean_scores, valid_mask)
        else:
            idx_sel = alpha_selection_1se(alphas, mean_scores, se_scores, valid_mask)

        if idx_sel is None:
            # fallback: choose middle alpha
            idx_sel = int(len(alphas) // 2)
        alpha_sel = float(alphas[idx_sel])

        # ---- bagging fit on outer-train ----
        # 先拟合一个“非bag”模型（可用于对照/排错）
        final = CoxnetSurvivalAnalysis(
            l1_ratio=float(args.l1_ratio),
            alphas=np.array([alpha_sel], dtype=float),
            max_iter=int(args.max_iter),
            tol=float(args.tol),
            fit_baseline_model=False,
        )
        final.fit(X_tr, y_tr)

        # bagging predictions
        preds = np.zeros(len(va_idx), dtype=float)
        for b in range(int(args.n_bags)):
            bs_idx = rng.choice(np.arange(len(tr_idx)), size=len(tr_idx), replace=True)
            mdl_b = CoxnetSurvivalAnalysis(
                l1_ratio=float(args.l1_ratio),
                alphas=np.array([alpha_sel], dtype=float),
                max_iter=int(args.max_iter),
                tol=float(args.tol),
                fit_baseline_model=False,
            )
            mdl_b.fit(X_tr[bs_idx], y_tr[bs_idx])
            preds += mdl_b.predict(X_va, alpha=alpha_sel)

        risk_va = preds / float(args.n_bags)

        tau = compute_tau_from_train(y_tr, y_va, q=float(args.tau_q))
        har, uno = eval_fold(y_tr, y_va, risk_va, tau=tau)

        pd.DataFrame({
            args.id_col: pids[va_idx],
            "risk": risk_va,
        }).to_csv(fold_dir / "predictions.csv", index=False)

        row = {
            "fold": fk,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "l1_ratio": float(args.l1_ratio),
            "alpha": float(alpha_sel),
            "inner_metric": args.inner_metric,
            "alpha_select": args.alpha_select,
            "n_bags": int(args.n_bags),
            "harrell": float(har),
            "uno": float(uno),
        }
        rows.append(row)
        print(f"[{fk}] Harrell={har:.4f} Uno={uno:.4f} | l1={args.l1_ratio} alpha={alpha_sel:.4g} bags={args.n_bags}")

    res = pd.DataFrame(rows)
    res.to_csv(outdir / "summary.csv", index=False)

    print("========================================================================")
    print(f"[DONE] mean Harrell={res.harrell.mean():.4f} ± {res.harrell.std(ddof=1):.4f}")
    print(f"[DONE] mean Uno    ={res.uno.mean():.4f} ± {res.uno.std(ddof=1):.4f}")
    print(f"[DONE] saved to {outdir}")


if __name__ == "__main__":
    main()
