import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw


def load_splits(splits_json: Path):
    with open(splits_json, "r") as f:
        splits = json.load(f)
    # expected: { "fold_0": {"train":[...], "val":[...]}, ... }
    return splits


def make_structured_y(time_arr, event_arr):
    # sksurv expects dtype with ('event', bool) and ('time', float)
    y = np.zeros(len(time_arr), dtype=[("event", bool), ("time", float)])
    y["event"] = event_arr.astype(bool)
    y["time"] = time_arr.astype(float)
    return y


def compute_tau_from_train(y_train, y_val, q=0.90, eps=1e-6):
    """
    Tau for Uno C should be fixed per OUTER fold, derived from OUTER-TRAIN only,
    and capped to avoid numerical issues when tau exceeds val max time.
    """
    event_times = y_train["time"][y_train["event"]]
    if len(event_times) == 0:
        # degenerate case: no events in train
        return float(np.max(y_val["time"]) - eps)

    tau_train = float(np.quantile(event_times, q))
    tau_train = min(tau_train, float(np.max(event_times)) - eps)

    tau_val_cap = float(np.max(y_val["time"]) - eps)
    tau = min(tau_train, tau_val_cap)

    # ensure positive
    if not np.isfinite(tau) or tau <= 0:
        tau = float(np.max(y_val["time"]) - eps)
    return tau


def alpha_selection_min(mean_scores):
    try:
        return int(np.nanargmax(mean_scores))
    except ValueError:
        # fallback: if all NaN, select middle alpha
        return len(mean_scores) // 2


def alpha_selection_1se(alphas, mean_scores, se_scores):
    """
    1-SE rule (maximize metric): choose the LARGEST alpha (strongest regularization)
    whose mean score is within 1 *standard error* of the best mean score.
    """
    # 1. find the index of the highest mean score
    best_idx = int(np.nanargmax(mean_scores))
    
    # 2. calculate the threshold: best score - 1 standard error (note here we use se_scores)
    # calculate the threshold: best score - 1 standard error (note here we use se_scores)
    thresh = mean_scores[best_idx] - se_scores[best_idx]
    
    # 3. find the indices of all alphas that satisfy score >= thresh
    # find the indices of all alphas that satisfy score >= thresh
    ok = np.where(mean_scores >= thresh)[0]
    
    if ok.size == 0:
        return best_idx
    
    # 4. choose the largest alpha in the candidates (strongest regularization)
    # (assume alphas is descending order, we want to find the largest alpha that satisfies the condition)
    candidate_alphas = alphas[ok]
    best_candidate_idx_in_subset = np.argmax(candidate_alphas) 
    final_idx = ok[best_candidate_idx_in_subset]
    
    return int(final_idx)


def parse_alpha_min_ratio(value):
    """Convert alpha_min_ratio to float if numeric, keep 'auto' as string."""
    if value.lower() == "auto":
        return "auto"
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"alpha_min_ratio must be 'auto' or a numeric value, got: {value}")


def eval_fold(y_train, y_val, risk_val, tau):
    harrell = float(concordance_index_censored(y_val["event"], y_val["time"], risk_val)[0])
    uno = float(concordance_index_ipcw(y_train, y_val, risk_val, tau=tau)[0])
    return harrell, uno


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--embeddings_npy", type=str, required=True)
    ap.add_argument("--pids_csv", type=str, required=True, help="CSV with at least column PatientID")
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--id_col", type=str, default="PatientID")
    ap.add_argument("--time_col", type=str, default="Survival.time")
    ap.add_argument("--event_col", type=str, default="deadstatus.event")

    ap.add_argument("--mode", type=str, choices=["R0", "R3"], default="R3")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--l1_ratios", type=float, nargs="+", default=[0.1, 0.5, 0.7, 0.9])
    ap.add_argument("--n_alphas", type=int, default=100)
    ap.add_argument("--alpha_min_ratio", type=parse_alpha_min_ratio, default="auto")
    ap.add_argument("--max_iter", type=int, default=100000)
    ap.add_argument("--tol", type=float, default=1e-7)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.embeddings_npy)
    pids = pd.read_csv(args.pids_csv)[args.id_col].astype(str).tolist()

    if X.shape[0] != len(pids):
        raise ValueError(f"Mismatch: embeddings rows={X.shape[0]} vs pids={len(pids)}")

    if X.shape[1] <= 8:
        raise RuntimeError(f"Embedding dim too small: {X.shape}. "
                           f"Likely logits/head, not deep features. Re-extract from avgpool.")

    y_df = pd.read_csv(args.labels_csv).copy()
    y_df[args.id_col] = y_df[args.id_col].astype(str)

    # Align
    idx_map = {pid: i for i, pid in enumerate(pids)}
    y_df = y_df[y_df[args.id_col].isin(idx_map)].copy()
    y_df.sort_values(args.id_col, inplace=True)

    aligned_idx = [idx_map[pid] for pid in y_df[args.id_col].tolist()]
    X = X[aligned_idx]
    pids = y_df[args.id_col].tolist()

    time_arr = y_df[args.time_col].to_numpy()
    event_arr = y_df[args.event_col].to_numpy().astype(int)

    y_struct = make_structured_y(time_arr, event_arr)

    splits = load_splits(Path(args.splits_json))
    fold_keys = sorted(list(splits.keys()))

    rng = np.random.default_rng(args.seed)

    # outputs
    oof_min = []
    oof_1se = []
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

        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y_struct[tr_idx], y_struct[va_idx]

        # R0: shuffle labels (permute (time,event) pairs) ONLY within training pool, then apply model
        if args.mode == "R0":
            perm = rng.permutation(len(y_tr))
            y_tr_used = y_tr[perm]
        else:
            y_tr_used = y_tr

        # Standardize using OUTER-TRAIN only (no outer-val leakage)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        # fixed tau per OUTER fold
        tau = compute_tau_from_train(y_tr_used, y_va, q=0.90)

        # --------------------
        # Inner CV to pick (l1_ratio, alpha_min, alpha_1se)
        # --------------------
        inner = KFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed)

        best_choice = None  # dict
        for l1 in args.l1_ratios:
            # alpha path generated on OUTER-TRAIN (pragmatic + stable grid)
            base = CoxnetSurvivalAnalysis(
                l1_ratio=l1,
                alphas=None,
                n_alphas=args.n_alphas,
                alpha_min_ratio=args.alpha_min_ratio,
                max_iter=args.max_iter,
                tol=args.tol,
                fit_baseline_model=False,
            )
            base.fit(X_tr_s, y_tr_used)
            alphas = base.alphas_

            # score each alpha in inner CV
            score_mat = np.full((args.inner_folds, len(alphas)), np.nan, dtype=float)

            for fold_j, (itr, iva) in enumerate(inner.split(X_tr_s)):
                Xi_tr, Xi_va = X_tr_s[itr], X_tr_s[iva]
                yi_tr, yi_va = y_tr_used[itr], y_tr_used[iva]

                mdl = CoxnetSurvivalAnalysis(
                    l1_ratio=l1,
                    alphas=alphas,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    fit_baseline_model=False,
                )
                try:
                    mdl.fit(Xi_tr, yi_tr)
                    # predict risk for each alpha -> need loop because sksurv predicts one alpha at a time
                    for k, a in enumerate(alphas):
                        r = mdl.predict(Xi_va, alpha=a)
                        score_mat[fold_j, k] = concordance_index_censored(
                            yi_va["event"], yi_va["time"], r
                        )[0]
                except Exception:
                    # leave NaNs, this alpha path / fold is unstable
                    pass

            mean_scores = np.nanmean(score_mat, axis=0)
            n_valid = np.sum(np.isfinite(score_mat), axis=0)
            std_scores = np.nanstd(score_mat, axis=0, ddof=1)

            mean_scores[n_valid < 2] = np.nan
            std_scores[n_valid < 2] = np.nan

            se_scores = std_scores / np.sqrt(n_valid)
            idx_min = alpha_selection_min(mean_scores)
            idx_1se = alpha_selection_1se(alphas, mean_scores, se_scores)

            candidate = {
                "l1_ratio": float(l1),
                "alphas": alphas,
                "mean_scores": mean_scores,
                "std_scores": std_scores,
                "idx_min": int(idx_min),
                "idx_1se": int(idx_1se),
                "best_mean": float(mean_scores[idx_min]),
            }

            if (best_choice is None) or (candidate["best_mean"] > best_choice["best_mean"]):
                best_choice = candidate

        # --------------------
        # Train final on OUTER-TRAIN and eval on OUTER-VAL for both lambda.min and lambda.1se
        # --------------------
        l1 = best_choice["l1_ratio"]
        alphas = best_choice["alphas"]
        a_min = float(alphas[best_choice["idx_min"]])
        a_1se = float(alphas[best_choice["idx_1se"]])

        final = CoxnetSurvivalAnalysis(
            l1_ratio=l1,
            alphas=alphas,
            max_iter=args.max_iter,
            tol=args.tol,
            fit_baseline_model=False,
        )
        final.fit(X_tr_s, y_tr_used)

        r_min = final.predict(X_va_s, alpha=a_min)
        r_1se = final.predict(X_va_s, alpha=a_1se)

        har_min, uno_min = eval_fold(y_tr_used, y_va, r_min, tau=tau)
        har_1se, uno_1se = eval_fold(y_tr_used, y_va, r_1se, tau=tau)

        # coefficients + nnz
        coefs = final.coef_  # (p, n_alphas)
        k_min = int(np.argmin(np.abs(alphas - a_min)))
        k_1se = int(np.argmin(np.abs(alphas - a_1se)))

        coef_min = coefs[:, k_min]
        coef_1se = coefs[:, k_1se]

        nnz_min = int(np.sum(coef_min != 0.0))
        nnz_1se = int(np.sum(coef_1se != 0.0))

        # save fold artifacts
        metrics = {
            "mode": args.mode,
            "fold": fk,
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "tau": float(tau),
            "l1_ratio": float(l1),
            "alpha_min": float(a_min),
            "alpha_1se": float(a_1se),
            "harrell_min": float(har_min),
            "uno_min": float(uno_min),
            "harrell_1se": float(har_1se),
            "uno_1se": float(uno_1se),
            "nnz_min": nnz_min,
            "nnz_1se": nnz_1se,
            "embed_dim": int(X.shape[1]),
        }
        (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        # preds
        pred_df = pd.DataFrame({
            args.id_col: [pids[i] for i in va_idx],
            "risk_min": r_min,
            "risk_1se": r_1se,
        })
        pred_df.to_csv(fold_dir / "predictions.csv", index=False)

        # coefs (store only non-zeros to keep file small)
        def pack_coef(vec):
            nz = np.where(vec != 0.0)[0]
            return {"idx": nz.tolist(), "coef": vec[nz].astype(float).tolist()}

        coef_pack = {
            "alpha_min": float(a_min),
            "alpha_1se": float(a_1se),
            "coef_min": pack_coef(coef_min),
            "coef_1se": pack_coef(coef_1se),
        }
        (fold_dir / "coef.json").write_text(json.dumps(coef_pack, indent=2))

        # oof
        for pid, rv in zip(pred_df[args.id_col].tolist(), pred_df["risk_min"].tolist()):
            oof_min.append((pid, float(rv), fk))
        for pid, rv in zip(pred_df[args.id_col].tolist(), pred_df["risk_1se"].tolist()):
            oof_1se.append((pid, float(rv), fk))

        summary_rows.append(metrics)
        print(f"[{fk}] har(min)={har_min:.4f} uno(min)={uno_min:.4f} | "
              f"har(1se)={har_1se:.4f} uno(1se)={uno_1se:.4f} | nnz1se={nnz_1se}")

    # write OOF + summary
    pd.DataFrame(oof_min, columns=[args.id_col, "risk", "fold"]).to_csv(out_dir / "oof_pred_lambda_min.csv", index=False)
    pd.DataFrame(oof_1se, columns=[args.id_col, "risk", "fold"]).to_csv(out_dir / "oof_pred_lambda_1se.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_by_fold.csv", index=False)

    print(f"[DONE] mode={args.mode} saved to {out_dir}")


if __name__ == "__main__":
    main()


#run command:
#python run_R3_coxnet.py \
#  --embeddings_npy /home/lichengze/Research/DeepFeature/myresearch/main/R3/Df_embeddings.npy \
#  --pids_csv        /home/lichengze/Research/DeepFeature/myresearch/main/R3/Df_pids.csv \
#  --labels_csv      /home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
#  --splits_json     /home/lichengze/Research/DeepFeature/myresearch/main/R0/splits.json \
#  --out_dir         /home/lichengze/Research/DeepFeature/myresearch/main/R3/R0_shuffle \
#  --mode R0 \
#  --time_col Survival.time \
#  --event_col deadstatus.event


#python run_R3_coxnet.py \
#  --embeddings_npy /home/lichengze/Research/DeepFeature/myresearch/main/R3/Df_embeddings.npy \
#  --pids_csv        /home/lichengze/Research/DeepFeature/myresearch/main/R3/Df_pids.csv \
#  --labels_csv      /home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
#  --splits_json     /home/lichengze/Research/DeepFeature/myresearch/main/R0/splits.json \
#  --out_dir         /home/lichengze/Research/DeepFeature/myresearch/main/R3/R3_coxnet_fix1se \
#  --mode R3 \
#  --alpha_min_ratio 0.0001 \
#  --n_alphas 200 \
#  --time_col Survival.time \
#  --event_col deadstatus.event
