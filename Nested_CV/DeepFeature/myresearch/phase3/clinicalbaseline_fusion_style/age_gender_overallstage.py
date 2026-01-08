#!/usr/bin/env python
# clinical_featureset_cv.py

"""
Clinical-only CV for ablation + (optional) nested feature-set selection.

Two modes:
1) --mode sweep
   Evaluate EACH candidate feature-set with the fixed OUTER 5 folds.
   Report per-fold score, mean/std, and max-fold (descriptive only).

2) --mode nested_select
   Proper nested CV to avoid selection bias when choosing the best feature-set:
   - Outer: fixed 5 folds from exp_dir/fold_k/(train_pids.csv, val_pids.csv)
   - Inner: StratifiedKFold on outer-train to select the best feature-set
   - Outer-val: evaluate ONCE using the feature-set selected by inner CV

Important:
- Any preprocessing/encoding statistics (mean/std/mode) MUST be computed
  using the *training split of the current loop* (inner-train or outer-train),
  otherwise you leak info and re-introduce bias.

Metric:
- Prefer Uno IPCW C-index; fallback to censored C-index if Uno fails.
"""

import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored


# -------------------- utils --------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_pid_list(path: Path):
    return pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()


def to_struct(times, events):
    return np.array([(bool(e), float(t)) for t, e in zip(times, events)],
                    dtype=[("event", bool), ("time", float)])


def safe_tau(t_tr: np.ndarray, e_tr: np.ndarray, q: float = 0.9) -> float:
    t_tr = np.asarray(t_tr, dtype=float)
    e_tr = np.asarray(e_tr, dtype=int)
    ev = t_tr[e_tr == 1]
    base = ev if ev.size >= 5 else t_tr
    tau = float(np.quantile(base, q))
    mx = float(np.max(t_tr))
    if tau >= mx:
        tau = mx - 1e-6
    if tau <= 0:
        tau = max(mx * 0.5, 1e-6)
    return tau


def minmax_by_train(train_arr: np.ndarray, test_arr: np.ndarray, eps: float = 1e-8):
    tr = np.asarray(train_arr, dtype=np.float64)
    te = np.asarray(test_arr, dtype=np.float64)
    mn = float(np.min(tr))
    mx = float(np.max(tr))
    if mx - mn < eps:
        return np.zeros_like(tr), np.zeros_like(te)
    return (tr - mn) / (mx - mn), (te - mn) / (mx - mn)


def zscore_by_train(train_arr: np.ndarray, test_arr: np.ndarray, eps: float = 1e-8):
    tr = np.asarray(train_arr, dtype=np.float64)
    te = np.asarray(test_arr, dtype=np.float64)
    mu = float(np.mean(tr))
    sd = float(np.std(tr))
    if sd < eps:
        sd = 1.0
    return (tr - mu) / sd, (te - mu) / sd


def uno_or_censored(y_tr_st, y_te_st, e_te, t_te, risk_te, tau):
    try:
        return float(concordance_index_ipcw(y_tr_st, y_te_st, risk_te, tau=tau)[0])
    except Exception as ex:
        print(f"[WARN] Uno-IPCW failed -> fallback to censored C-index. Reason: {ex}")
        return float(concordance_index_censored(e_te.astype(bool), t_te, risk_te)[0])


def _safe_inner_splits(e: np.ndarray, requested: int):
    e = np.asarray(e).astype(int)
    c0 = int((e == 0).sum())
    c1 = int((e == 1).sum())
    max_splits = min(c0, c1)  # StratifiedKFold constraint
    if max_splits < 2:
        return 0
    return int(min(requested, max_splits))


# -------------------- clinical feature builder (matches fusion_baseline.py rules) --------------------
def build_clinical_features(df: pd.DataFrame,
                            tr_pids: list,
                            va_pids: list,
                            feature_names: list,
                            overall_stage_encoding: str):
    """
    feature_names subset of: ["age", "gender", "T", "N", "M", "stage"]
    Preprocessing rules match your fusion_baseline.py.

    Returns:
      X_tr, t_tr, e_tr, X_va, t_va, e_va, feat_cols_out
    """
    use_cols = ["PatientID", "Survival.time", "deadstatus.event"]
    if "age" in feature_names:
        use_cols += ["age"]
    if "gender" in feature_names:
        use_cols += ["gender"]
    if "T" in feature_names:
        use_cols += ["clinical.T.Stage"]
    if "N" in feature_names:
        use_cols += ["Clinical.N.Stage"]
    if "M" in feature_names:
        use_cols += ["Clinical.M.Stage"]
    if "stage" in feature_names:
        use_cols += ["Overall.Stage"]

    d = df[use_cols].copy()
    d["PatientID"] = d["PatientID"].astype(str)

    tr = d[d["PatientID"].isin(tr_pids)].copy()
    va = d[d["PatientID"].isin(va_pids)].copy()

    # hard alignment to pid order (same style as your fusion code)
    tr = tr.set_index("PatientID").loc[tr_pids].reset_index()
    va = va.set_index("PatientID").loc[va_pids].reset_index()

    t_tr = tr["Survival.time"].astype(float).to_numpy()
    e_tr = tr["deadstatus.event"].astype(int).to_numpy()
    t_va = va["Survival.time"].astype(float).to_numpy()
    e_va = va["deadstatus.event"].astype(int).to_numpy()

    feats_tr, feats_va, feat_cols_out = [], [], []

    # age z (fill NaN with train mean)
    if "age" in feature_names:
        age_tr_s = tr["age"].astype(float)
        mu = float(age_tr_s.mean(skipna=True))
        sigma = float(age_tr_s.std(skipna=True))
        sigma = sigma if sigma > 1e-8 else 1.0
        age_tr = age_tr_s.fillna(mu).to_numpy(dtype=np.float32)
        age_va = va["age"].astype(float).fillna(mu).to_numpy(dtype=np.float32)
        feats_tr.append(((age_tr - mu) / sigma).astype(np.float32))
        feats_va.append(((age_va - mu) / sigma).astype(np.float32))
        feat_cols_out.append("age_z")

    # gender 0/1 (defensive map)
    if "gender" in feature_names:
        gmap = {"male": 1.0, "m": 1.0, "female": 0.0, "f": 0.0}
        g_tr = tr["gender"].astype(str).str.strip().str.lower().map(gmap).to_numpy(dtype=np.float32)
        g_va = va["gender"].astype(str).str.strip().str.lower().map(gmap).to_numpy(dtype=np.float32)
        if np.isnan(g_tr).any() or np.isnan(g_va).any():
            bad = sorted(set(tr["gender"].astype(str).str.lower().unique()) |
                         set(va["gender"].astype(str).str.lower().unique()))
            raise ValueError(f"Unexpected gender values found: {bad}")
        feats_tr.append(g_tr); feats_va.append(g_va); feat_cols_out.append("gender_01")

    # T z (fill NaN with train mode)
    if "T" in feature_names:
        T_tr_s = tr["clinical.T.Stage"].astype(float)
        T_va_s = va["clinical.T.Stage"].astype(float)
        T_mode = float(T_tr_s.dropna().value_counts().idxmax())
        T_tr = T_tr_s.fillna(T_mode).to_numpy(dtype=np.float32)
        T_va = T_va_s.fillna(T_mode).to_numpy(dtype=np.float32)
        mu = float(np.mean(T_tr))
        sigma = float(np.std(T_tr)) if float(np.std(T_tr)) > 1e-8 else 1.0
        feats_tr.append(((T_tr - mu) / sigma).astype(np.float32))
        feats_va.append(((T_va - mu) / sigma).astype(np.float32))
        feat_cols_out.append("T_z")

    # N z (no NaN allowed, same as your fusion)
    if "N" in feature_names:
        N_tr = tr["Clinical.N.Stage"].astype(float).to_numpy(dtype=np.float32)
        N_va = va["Clinical.N.Stage"].astype(float).to_numpy(dtype=np.float32)
        if np.isnan(N_tr).any() or np.isnan(N_va).any():
            raise ValueError("NaN found in Clinical.N.Stage.")
        mu = float(np.mean(N_tr))
        sigma = float(np.std(N_tr)) if float(np.std(N_tr)) > 1e-8 else 1.0
        feats_tr.append(((N_tr - mu) / sigma).astype(np.float32))
        feats_va.append(((N_va - mu) / sigma).astype(np.float32))
        feat_cols_out.append("N_z")

    # M z (no NaN allowed)
    if "M" in feature_names:
        M_tr = tr["Clinical.M.Stage"].astype(float).to_numpy(dtype=np.float32)
        M_va = va["Clinical.M.Stage"].astype(float).to_numpy(dtype=np.float32)
        if np.isnan(M_tr).any() or np.isnan(M_va).any():
            raise ValueError("NaN found in Clinical.M.Stage.")
        mu = float(np.mean(M_tr))
        sigma = float(np.std(M_tr)) if float(np.std(M_tr)) > 1e-8 else 1.0
        feats_tr.append(((M_tr - mu) / sigma).astype(np.float32))
        feats_va.append(((M_va - mu) / sigma).astype(np.float32))
        feat_cols_out.append("M_z")

    # Overall.Stage
    if "stage" in feature_names:
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
                feat_cols_out.append(f"OverallStage_{c}")
        elif overall_stage_encoding == "ordinal":
            mapping = {"I": 1.0, "II": 2.0, "IIIa": 3.0, "IIIb": 4.0}
            os_tr_num = os_tr.map(mapping).to_numpy(dtype=np.float32)
            os_va_num = os_va.map(mapping).to_numpy(dtype=np.float32)
            if np.isnan(os_tr_num).any() or np.isnan(os_va_num).any():
                raise ValueError("Unexpected Overall.Stage values found after normalization.")
            mu = float(np.mean(os_tr_num))
            sigma = float(np.std(os_tr_num)) if float(np.std(os_tr_num)) > 1e-8 else 1.0
            feats_tr.append(((os_tr_num - mu) / sigma).astype(np.float32))
            feats_va.append(((os_va_num - mu) / sigma).astype(np.float32))
            feat_cols_out.append("OverallStage_ord_z")
        else:
            raise ValueError("overall_stage_encoding must be 'ordinal' or 'onehot'")

    if len(feats_tr) == 0:
        raise ValueError("Empty feature set: no features were selected.")

    X_tr = np.stack(feats_tr, axis=1)
    X_va = np.stack(feats_va, axis=1)
    return X_tr, t_tr, e_tr, X_va, t_va, e_va, feat_cols_out


# -------------------- evaluate a single feature-set on ONE outer fold --------------------
def eval_one_fold(df, tr_pids, va_pids, feature_set, stage_enc, cox_alpha, norm_fn):
    X_tr, t_tr, e_tr, X_va, t_va, e_va, feat_cols_out = build_clinical_features(
        df=df,
        tr_pids=tr_pids,
        va_pids=va_pids,
        feature_names=feature_set,
        overall_stage_encoding=stage_enc,
    )
    y_tr_st = to_struct(t_tr, e_tr)
    y_va_st = to_struct(t_va, e_va)
    tau = safe_tau(t_tr, e_tr)

    cox = CoxPHSurvivalAnalysis(alpha=float(cox_alpha))
    cox.fit(X_tr, y_tr_st)

    risk_tr = cox.predict(X_tr).astype(np.float64)
    risk_va = cox.predict(X_va).astype(np.float64)

    # Keep normalization for pipeline consistency (doesn't change ranking in theory)
    _, risk_va_n = norm_fn(risk_tr, risk_va)

    score = uno_or_censored(y_tr_st, y_va_st, e_va, t_va, risk_va_n, tau)
    return float(score), float(tau), feat_cols_out


# -------------------- inner CV selection of feature-set --------------------
def select_featureset_by_inner_cv(df, outer_tr_pids, feature_set_specs, inner_folds, seed, cox_alpha, norm_fn):
    """
    feature_set_specs: list of dicts, each dict:
      {"name": "...", "features": [...], "stage_enc": "onehot"/"ordinal"}
    """
    # get events for stratification in outer-train
    tmp = df[["PatientID", "deadstatus.event"]].copy()
    tmp["PatientID"] = tmp["PatientID"].astype(str)
    tmp = tmp.set_index("PatientID")
    e_outer = tmp.loc[outer_tr_pids]["deadstatus.event"].astype(int).to_numpy()

    k = _safe_inner_splits(e_outer, inner_folds)
    if k < 2:
        # cannot do proper inner CV; fall back to first spec
        return 0, [(spec["name"], float("nan")) for spec in feature_set_specs]

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    mean_scores = []
    # pre-create pid array for indexing
    pids_arr = np.array(outer_tr_pids, dtype=object)

    for j, spec in enumerate(feature_set_specs):
        fold_scores = []
        for tr_idx, va_idx in skf.split(np.zeros(len(e_outer)), e_outer):
            inner_tr_pids = pids_arr[tr_idx].tolist()
            inner_va_pids = pids_arr[va_idx].tolist()

            s, _tau, _cols = eval_one_fold(
                df=df,
                tr_pids=inner_tr_pids,
                va_pids=inner_va_pids,
                feature_set=spec["features"],
                stage_enc=spec["stage_enc"],
                cox_alpha=cox_alpha,
                norm_fn=norm_fn,
            )
            fold_scores.append(float(s))

        mean_scores.append((spec["name"], float(np.mean(fold_scores))))

    # pick best (ties -> simpler: fewer output cols, then fewer raw features)
    best_idx = 0
    best_score = -1e18
    for j, spec in enumerate(feature_set_specs):
        s = mean_scores[j][1]
        if np.isnan(s):
            continue
        if s > best_score + 1e-12:
            best_score = s
            best_idx = j
        elif abs(s - best_score) <= 1e-12:
            # tie-break: prefer fewer raw features
            if len(spec["features"]) < len(feature_set_specs[best_idx]["features"]):
                best_idx = j

    return int(best_idx), mean_scores


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", type=str, required=True)
    ap.add_argument("--clinical_csv", type=str, required=True)

    ap.add_argument("--out_name", type=str, default="clinical_featureset_cv")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--cox_alpha", type=float, default=1e-4)
    ap.add_argument("--norm", type=str, default="zscore", choices=["minmax", "zscore"])

    ap.add_argument("--mode", type=str, default="sweep", choices=["sweep", "nested_select"])
    ap.add_argument("--inner_folds", type=int, default=5)

    # For your current request: only evaluate age+gender+stage (default candidate list)
    ap.add_argument("--stage_enc", type=str, default="onehot", choices=["onehot", "ordinal"])

    args = ap.parse_args()
    set_seed(args.seed)

    exp_dir = Path(args.exp_dir)
    out_root = exp_dir / args.out_name
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.clinical_csv)
    norm_fn = minmax_by_train if args.norm == "minmax" else zscore_by_train

    # --- candidates: start with ONLY (age, gender, stage) as you requested ---
    # Later you can add more specs here without changing any methodology.
    feature_set_specs = [
        {"name": f"age+gender+stage({args.stage_enc})",
         "features": ["age", "gender", "stage"],
         "stage_enc": args.stage_enc},
    ]

    # helper to load fixed outer folds from exp_dir
    outer_folds = []
    for fold in range(5):
        fold_dir = exp_dir / f"fold_{fold}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"Missing fold dir: {fold_dir}")
        outer_folds.append({
            "fold": fold,
            "tr_pids": read_pid_list(fold_dir / "train_pids.csv"),
            "va_pids": read_pid_list(fold_dir / "val_pids.csv"),
        })

    if args.mode == "sweep":
        # Evaluate EACH candidate independently on outer folds (good for ablation tables).
        all_rows = []
        details = {}

        for spec in feature_set_specs:
            per_fold_scores = []
            per_fold = []
            for of in outer_folds:
                score, tau, feat_cols_out = eval_one_fold(
                    df=df,
                    tr_pids=of["tr_pids"],
                    va_pids=of["va_pids"],
                    feature_set=spec["features"],
                    stage_enc=spec["stage_enc"],
                    cox_alpha=args.cox_alpha,
                    norm_fn=norm_fn,
                )
                per_fold_scores.append(score)
                per_fold.append({
                    "fold": of["fold"],
                    "score": float(score),
                    "tau": float(tau),
                    "n_feat": int(len(feat_cols_out)),
                    "feat_cols": feat_cols_out,
                })
                all_rows.append({
                    "spec": spec["name"],
                    "fold": of["fold"],
                    "score": float(score),
                    "tau": float(tau),
                    "cox_alpha": float(args.cox_alpha),
                    "norm": str(args.norm),
                    "n_feat": int(len(feat_cols_out)),
                })

            m = float(np.mean(per_fold_scores))
            sd = float(np.std(per_fold_scores))
            mx = float(np.max(per_fold_scores))
            mn = float(np.min(per_fold_scores))

            details[spec["name"]] = {
                "per_fold": per_fold,
                "mean": m,
                "std": sd,
                "max_fold_score": mx,
                "min_fold_score": mn,
            }

            print(f"\n=== SPEC: {spec['name']} ===")
            print("per-fold:", [round(x, 4) for x in per_fold_scores])
            print(f"mean±std: {m:.4f} ± {sd:.4f}")
            print(f"max fold (descriptive): {mx:.4f} | min fold: {mn:.4f}")

        df_res = pd.DataFrame(all_rows)
        df_res.to_csv(out_root / "sweep_results.csv", index=False)
        with open(out_root / "sweep_details.json", "w") as f:
            json.dump(details, f, indent=2)

        print(f"\n[Saved] {out_root/'sweep_results.csv'}")
        print(f"[Saved] {out_root/'sweep_details.json'}")

    else:
        # Proper nested selection of feature-set (unbiased evaluation of the selection procedure).
        outer_rows = []
        details = {"feature_set_specs": feature_set_specs, "outer": {}}

        for of in outer_folds:
            fold = of["fold"]
            outer_tr_pids = of["tr_pids"]
            outer_va_pids = of["va_pids"]

            best_idx, inner_mean_scores = select_featureset_by_inner_cv(
                df=df,
                outer_tr_pids=outer_tr_pids,
                feature_set_specs=feature_set_specs,
                inner_folds=args.inner_folds,
                seed=args.seed + 10000 + fold,
                cox_alpha=args.cox_alpha,
                norm_fn=norm_fn,
            )
            best_spec = feature_set_specs[best_idx]

            outer_score, outer_tau, feat_cols_out = eval_one_fold(
                df=df,
                tr_pids=outer_tr_pids,
                va_pids=outer_va_pids,
                feature_set=best_spec["features"],
                stage_enc=best_spec["stage_enc"],
                cox_alpha=args.cox_alpha,
                norm_fn=norm_fn,
            )

            outer_rows.append({
                "fold": fold,
                "selected_spec": best_spec["name"],
                "outer_score": float(outer_score),
                "outer_tau": float(outer_tau),
                "cox_alpha": float(args.cox_alpha),
                "norm": str(args.norm),
                "n_feat": int(len(feat_cols_out)),
            })

            details["outer"][f"fold_{fold}"] = {
                "selected_spec": best_spec["name"],
                "inner_mean_scores": inner_mean_scores,
                "outer_score": float(outer_score),
                "outer_tau": float(outer_tau),
                "feat_cols": feat_cols_out,
            }

            print(f"\n--- Outer fold {fold} ---")
            print("inner mean scores:", inner_mean_scores)
            print(f"selected: {best_spec['name']}")
            print(f"outer score: {outer_score:.4f} (tau={outer_tau:.4f})")

        df_out = pd.DataFrame(outer_rows)
        df_out.to_csv(out_root / "nested_outer_results.csv", index=False)
        with open(out_root / "nested_details.json", "w") as f:
            json.dump(details, f, indent=2)

        m = float(df_out["outer_score"].mean())
        sd = float(df_out["outer_score"].std())
        mx = float(df_out["outer_score"].max())
        print("\n========== Nested Summary ==========")
        print(df_out)
        print(f"outer_score mean±std: {m:.4f} ± {sd:.4f}")
        print(f"outer_score max fold (descriptive): {mx:.4f}")

        print(f"\n[Saved] {out_root/'nested_outer_results.csv'}")
        print(f"[Saved] {out_root/'nested_details.json'}")


if __name__ == "__main__":
    main()
