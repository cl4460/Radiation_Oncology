import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw


# -----------------------------
# Utilities
# -----------------------------
def load_splits(splits_json: Path):
    with open(splits_json, "r") as f:
        return json.load(f)


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


def parse_alpha_min_ratio(value):
    # Accept "auto" or float (argparse always passes string unless type=float)
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    return float(value)


def eval_fold(y_train, y_val, risk_val, tau):
    harrell = float(concordance_index_censored(y_val["event"], y_val["time"], risk_val)[0])
    uno = float(concordance_index_ipcw(y_train, y_val, risk_val, tau=tau)[0])
    return harrell, uno


def corr_filter_greedy(X_tr, feat_names, thresh=0.95):
    """
    Greedy correlation filter fitted on TRAIN only.
    Returns kept indices and kept feature names.
    """
    # drop near-constant cols first (avoid NaN corr)
    std = np.nanstd(X_tr, axis=0)
    ok_var = std > 0
    X = X_tr[:, ok_var]
    names = [n for n, ok in zip(feat_names, ok_var) if ok]

    if X.shape[1] <= 1:
        kept = np.arange(X.shape[1], dtype=int)
        return ok_var, kept, names

    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.abs(C)

    kept = []
    for j in range(X.shape[1]):
        if len(kept) == 0:
            kept.append(j)
        else:
            if np.all(C[j, kept] < thresh):
                kept.append(j)

    kept = np.array(kept, dtype=int)

    # map kept back to original feature space
    kept_mask_full = np.zeros(len(feat_names), dtype=bool)
    var_idx = np.where(ok_var)[0]
    kept_mask_full[var_idx[kept]] = True
    kept_names = [feat_names[i] for i in range(len(feat_names)) if kept_mask_full[i]]
    return kept_mask_full, kept, kept_names


def alpha_selection_min(alphas, mean_scores, valid_mask):
    idxs = np.where(valid_mask)[0]
    if idxs.size == 0:
        return None
    best_local = idxs[int(np.nanargmax(mean_scores[idxs]))]
    return int(best_local)


def alpha_selection_1se(alphas, mean_scores, se_scores, valid_mask):
    idx_min = alpha_selection_min(alphas, mean_scores, valid_mask)
    if idx_min is None:
        return None

    thresh = mean_scores[idx_min] - se_scores[idx_min]
    ok = np.where((mean_scores >= thresh) & valid_mask)[0]
    if ok.size == 0:
        return int(idx_min)

    # choose largest alpha among candidates (strongest regularization)
    best_alpha_idx = ok[int(np.argmax(alphas[ok]))]
    return int(best_alpha_idx)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--id_col", type=str, default="PatientID")
    ap.add_argument("--time_col", type=str, default="Survival.time")
    ap.add_argument("--event_col", type=str, default="deadstatus.event")

    ap.add_argument("--mode", type=str, choices=["R0", "R4"], default="R4")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--tau_q", type=float, default=0.90)

    ap.add_argument("--l1_ratios", type=float, nargs="+", default=[0.0, 0.1, 0.5, 0.9])
    ap.add_argument("--n_alphas", type=int, default=400)
    ap.add_argument("--alpha_min_ratio", type=parse_alpha_min_ratio, default=1e-6)
    ap.add_argument("--max_iter", type=int, default=100000)
    ap.add_argument("--tol", type=float, default=1e-7)

    ap.add_argument("--use_corr_filter", action="store_true")
    ap.add_argument("--corr_thresh", type=float, default=0.95)

    ap.add_argument("--alpha_select", type=str, choices=["min", "1se"], default="min")
    ap.add_argument("--min_valid_folds", type=int, default=2)

    ap.add_argument("--drop_nan_frac", type=float, default=0.20)  # drop col if train NaN fraction > this

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # -------- Load features --------
    feat_df = pd.read_csv(args.features_csv)
    feat_df[args.id_col] = feat_df[args.id_col].astype(str)
    feat_df = feat_df.drop_duplicates(subset=[args.id_col]).copy()

    feat_names = [c for c in feat_df.columns if c != args.id_col]

    # Force numeric features (radiomics sometimes comes as strings)
    X_all = feat_df[feat_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    pids_all = feat_df[args.id_col].tolist()

    # -------- Load labels --------
    lab_df = pd.read_csv(args.labels_csv).copy()
    lab_df[args.id_col] = lab_df[args.id_col].astype(str)

    # align to intersection
    idx_map = {pid: i for i, pid in enumerate(pids_all)}
    lab_df = lab_df[lab_df[args.id_col].isin(idx_map)].copy()
    lab_df.sort_values(args.id_col, inplace=True)

    aligned_idx = [idx_map[pid] for pid in lab_df[args.id_col].tolist()]
    X_all = X_all[aligned_idx]
    pids = lab_df[args.id_col].tolist()

    time_arr = pd.to_numeric(lab_df[args.time_col], errors="coerce").to_numpy(dtype=float)
    event_arr = pd.to_numeric(lab_df[args.event_col], errors="coerce").fillna(0).astype(int).to_numpy()
    y_struct = make_structured_y(time_arr, event_arr)

    # -------- Splits --------
    splits = load_splits(Path(args.splits_json))
    fold_keys = sorted(list(splits.keys()))

    # outputs
    oof_rows = []
    summary_rows = []

    for fk in fold_keys:
        fold_dir = out_dir / fk
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_pids = set(map(str, splits[fk]["train"]))
        val_pids = set(map(str, splits[fk]["val"]))

        tr_idx = np.array([i for i, pid in enumerate(pids) if pid in train_pids], dtype=int)
        va_idx = np.array([i for i, pid in enumerate(pids) if pid in val_pids], dtype=int)

        if len(set(tr_idx) & set(va_idx)) > 0:
            raise ValueError(f"{fk}: train/val overlap detected!")

        X_tr_raw = X_all[tr_idx].copy()
        X_va_raw = X_all[va_idx].copy()
        y_tr = y_struct[tr_idx]
        y_va = y_struct[va_idx]

        # R0: permute (time,event) pairs ONLY in outer-train
        if args.mode == "R0":
            perm = rng.permutation(len(y_tr))
            y_tr_used = y_tr[perm]
        else:
            y_tr_used = y_tr

        # ---- Missingness handling (TRAIN-fitted) ----
        nan_frac = np.mean(~np.isfinite(X_tr_raw), axis=0)
        keep_nan = nan_frac <= args.drop_nan_frac
        if np.sum(keep_nan) == 0:
            raise RuntimeError(f"{fk}: all features dropped by NaN filter (drop_nan_frac={args.drop_nan_frac})")

        X_tr_raw = X_tr_raw[:, keep_nan]
        X_va_raw = X_va_raw[:, keep_nan]
        feat_names_fold = [n for n, ok in zip(feat_names, keep_nan) if ok]

        # median impute using TRAIN medians
        med = np.nanmedian(X_tr_raw, axis=0)
        # if a column is all-NaN, nanmedian gives NaN; drop those
        ok_med = np.isfinite(med)
        X_tr_raw = X_tr_raw[:, ok_med]
        X_va_raw = X_va_raw[:, ok_med]
        feat_names_fold = [n for n, ok in zip(feat_names_fold, ok_med) if ok]
        med = med[ok_med]

        X_tr_raw = np.where(np.isfinite(X_tr_raw), X_tr_raw, med)
        X_va_raw = np.where(np.isfinite(X_va_raw), X_va_raw, med)

        # ---- Standardize (TRAIN only) ----
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_va = scaler.transform(X_va_raw)

        # ---- Correlation filter (TRAIN only) ----
        kept_mask = np.ones(X_tr.shape[1], dtype=bool)
        kept_names = feat_names_fold
        removed = []

        if args.use_corr_filter and X_tr.shape[1] > 1:
            kept_mask_full, kept_rel, kept_names = corr_filter_greedy(
                X_tr, feat_names_fold, thresh=args.corr_thresh
            )
            removed = [n for n, ok in zip(feat_names_fold, kept_mask_full) if not ok]
            X_tr = X_tr[:, kept_mask_full]
            X_va = X_va[:, kept_mask_full]
            kept_mask = kept_mask_full

        # ---- Feature sanity ----
        if X_tr.shape[1] == 0:
            raise RuntimeError(f"{fk}: No features left after preprocessing!")
        if X_tr.shape[1] < 5:
            print(f"[WARN] {fk}: Only {X_tr.shape[1]} features after preprocessing")

        # fixed tau per OUTER fold
        tau = compute_tau_from_train(y_tr_used, y_va, q=args.tau_q)

        # --------------------
        # Inner CV: choose l1_ratio + alpha
        # Choose best l1_ratio by lambda.min score, then apply alpha_select rule within it.
        # --------------------
        inner = KFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed)

        best_l1_choice = None  # store dict including alphas, curves, idx_min, idx_1se

        for l1 in args.l1_ratios:
            base = CoxnetSurvivalAnalysis(
                l1_ratio=float(l1),
                alphas=None,
                n_alphas=args.n_alphas,
                alpha_min_ratio=args.alpha_min_ratio,
                max_iter=args.max_iter,
                tol=args.tol,
                fit_baseline_model=False,
            )
            try:
                base.fit(X_tr, y_tr_used)
            except Exception as e:
                # skip totally failing l1_ratio
                (fold_dir / f"inner_fail_l1_{l1}.txt").write_text(str(e))
                continue

            alphas = np.array(base.alphas_, dtype=float)  # learned alpha grid :contentReference[oaicite:2]{index=2}
            nA = len(alphas)

            score_mat = np.full((args.inner_folds, nA), np.nan, dtype=float)

            for j, (itr, iva) in enumerate(inner.split(X_tr)):
                Xi_tr, Xi_va = X_tr[itr], X_tr[iva]
                yi_tr, yi_va = y_tr_used[itr], y_tr_used[iva]

                mdl = CoxnetSurvivalAnalysis(
                    l1_ratio=float(l1),
                    alphas=alphas,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    fit_baseline_model=False,
                )
                try:
                    mdl.fit(Xi_tr, yi_tr)
                    # score each alpha
                    for k, a in enumerate(alphas):
                        r = mdl.predict(Xi_va, alpha=float(a))  # predict supports alpha :contentReference[oaicite:3]{index=3}
                        score_mat[j, k] = concordance_index_censored(
                            yi_va["event"], yi_va["time"], r
                        )[0]
                except Exception:
                    # leave NaNs
                    continue

            n_valid = np.sum(np.isfinite(score_mat), axis=0).astype(int)

            mean_scores = np.full(nA, np.nan, dtype=float)
            std_scores = np.full(nA, np.nan, dtype=float)
            se_scores = np.full(nA, np.nan, dtype=float)

            for k in range(nA):
                if n_valid[k] >= 1:
                    mean_scores[k] = float(np.nanmean(score_mat[:, k]))
                if n_valid[k] >= 2:
                    std_scores[k] = float(np.nanstd(score_mat[:, k], ddof=1))
                    se_scores[k] = std_scores[k] / np.sqrt(n_valid[k])

            valid_mask = (n_valid >= args.min_valid_folds) & np.isfinite(mean_scores)

            idx_min = alpha_selection_min(alphas, mean_scores, valid_mask)
            idx_1se = alpha_selection_1se(alphas, mean_scores, se_scores, valid_mask)

            # If nothing valid, skip this l1_ratio
            if idx_min is None or idx_1se is None:
                detail = {
                    "l1_ratio": float(l1),
                    "reason": "no valid alpha (all NaN or too few valid folds)",
                    "min_valid_folds": int(args.min_valid_folds),
                    "n_valid": n_valid.tolist(),
                }
                (fold_dir / f"inner_invalid_l1_{l1}.json").write_text(json.dumps(detail, indent=2))
                continue

            # save curve for audit
            curve_df = pd.DataFrame({
                "alpha": alphas,
                "mean": mean_scores,
                "se": se_scores,
                "n_valid": n_valid,
                "valid": valid_mask.astype(int),
            })
            curve_df.to_csv(fold_dir / f"inner_curve_l1_{l1}.csv", index=False)

            choice = {
                "l1_ratio": float(l1),
                "alphas": alphas,
                "mean_scores": mean_scores,
                "se_scores": se_scores,
                "n_valid": n_valid,
                "idx_min": int(idx_min),
                "idx_1se": int(idx_1se),
                "best_mean_min": float(mean_scores[idx_min]),
            }

            if (best_l1_choice is None) or (choice["best_mean_min"] > best_l1_choice["best_mean_min"]):
                best_l1_choice = choice

        if best_l1_choice is None:
            raise RuntimeError(f"{fk}: all l1_ratios failed in inner CV. Check NaNs / preprocessing / alpha grid.")

        l1 = best_l1_choice["l1_ratio"]
        alphas = best_l1_choice["alphas"]
        idx_min = best_l1_choice["idx_min"]
        idx_1se = best_l1_choice["idx_1se"]

        if args.alpha_select == "min":
            idx_sel = idx_min
        else:
            idx_sel = idx_1se

        a_sel = float(alphas[idx_sel])

        # --------------------
        # Final fit on OUTER-TRAIN and eval on OUTER-VAL
        # --------------------
        final = CoxnetSurvivalAnalysis(
            l1_ratio=float(l1),
            alphas=alphas,
            max_iter=args.max_iter,
            tol=args.tol,
            fit_baseline_model=False,
        )
        final.fit(X_tr, y_tr_used)

        risk_va = final.predict(X_va, alpha=a_sel)

        har, uno = eval_fold(y_tr_used, y_va, risk_va, tau=tau)

        # coef + nnz
        coefs = final.coef_  # (p, n_alphas)
        k_sel = int(np.argmin(np.abs(alphas - a_sel)))
        coef_sel = coefs[:, k_sel]
        nnz = int(np.sum(coef_sel != 0.0))

        # write artifacts
        metrics = {
            "mode": args.mode,
            "fold": fk,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "tau": float(tau),
            "tau_q": float(args.tau_q),
            "alpha_select": args.alpha_select,
            "l1_ratio": float(l1),
            "alpha": float(a_sel),
            "harrell": float(har),
            "uno": float(uno),
            "nnz": nnz,
            "n_features_after_preproc": int(X_tr.shape[1]),
            "use_corr_filter": bool(args.use_corr_filter),
            "corr_thresh": float(args.corr_thresh),
            "drop_nan_frac": float(args.drop_nan_frac),
        }
        (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        pred_df = pd.DataFrame({
            args.id_col: [pids[i] for i in va_idx],
            "risk": risk_va.astype(float),
        })
        pred_df.to_csv(fold_dir / "predictions.csv", index=False)

        def pack_coef(vec, names):
            nz = np.where(vec != 0.0)[0]
            return {
                "idx": nz.tolist(),
                "feature": [names[i] for i in nz],
                "coef": vec[nz].astype(float).tolist(),
            }

        coef_pack = {
            "alpha": float(a_sel),
            "coef": pack_coef(coef_sel, kept_names),
            "feature_names": kept_names,
            "removed_by_corr_filter": removed,
        }
        (fold_dir / "coef.json").write_text(json.dumps(coef_pack, indent=2))

        for pid, rv in zip(pred_df[args.id_col].tolist(), pred_df["risk"].tolist()):
            oof_rows.append((pid, float(rv), fk))

        summary_rows.append(metrics)
        print(f"[{fk}] har={har:.4f} uno={uno:.4f} | l1={l1} alpha={a_sel:.6g} nnz={nnz} feats={X_tr.shape[1]}")

    # overall summary
    summ_df = pd.DataFrame(summary_rows)
    summ_df.to_csv(out_dir / "summary_by_fold.csv", index=False)

    oof_df = pd.DataFrame(oof_rows, columns=[args.id_col, "risk", "fold"])
    oof_df.to_csv(out_dir / "oof_pred.csv", index=False)

    overall = {
        "mode": args.mode,
        "alpha_select": args.alpha_select,
        "har_mean": float(summ_df["harrell"].mean()),
        "har_std": float(summ_df["harrell"].std(ddof=1)),
        "uno_mean": float(summ_df["uno"].mean()),
        "uno_std": float(summ_df["uno"].std(ddof=1)),
        "seed": int(args.seed),
        "n_folds": int(len(fold_keys)),
    }
    (out_dir / "overall.json").write_text(json.dumps(overall, indent=2))
    print("[OVERALL]", json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()

#run command:
#python run_R4_coxnet.py \
#  --features_csv /home/lichengze/Research/DeepFeature/myresearch/main/R4/Tb_bw25_iso111_clip_full/Tb_features.csv \
#  --labels_csv   /home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
#  --splits_json  /home/lichengze/Research/DeepFeature/myresearch/main/R0/splits.json \
#  --out_dir      /home/lichengze/Research/DeepFeature/myresearch/main/R4/R4_solid_v2_alpha400 \
#  --mode R4 --seed 42 \
#  --alpha_min_ratio 1e-6 --n_alphas 400 \
#  --l1_ratios 0.0 0.1 0.5 0.9 \
#  --use_corr_filter --corr_thresh 0.95 \
#  --alpha_select min \
#  --time_col Survival.time --event_col deadstatus.event