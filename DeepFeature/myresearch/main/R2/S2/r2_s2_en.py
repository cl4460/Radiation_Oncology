#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw


# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def warn_or_raise_missing_cols(name: str, provided: List[str], missing: List[str], strict: bool) -> None:
    if not missing:
        return
    msg = f"[{name}] These columns do not exist in CSV and would be dropped: {missing}"
    if strict:
        raise ValueError(msg)
    print(f"[WARNING] {msg}")
    
def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def mean_std(series: pd.Series) -> Tuple[float, float]:
    s = series.dropna().astype(float)
    if len(s) == 0:
        return float("nan"), float("nan")
    return float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0

def make_surv_struct(event: np.ndarray, time_arr: np.ndarray) -> np.ndarray:
    e = event.astype(bool)
    t = time_arr.astype(float)
    y = np.empty(len(t), dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = e
    y["time"] = t
    return y

def compute_tau_from_train(y_train: np.ndarray, quantile: float = 0.9) -> float:
    times = y_train["time"].astype(float)
    events = y_train["event"].astype(bool)
    ev_times = times[events]
    if ev_times.shape[0] >= 5:
        tau = float(np.quantile(ev_times, quantile))
    else:
        tau = float(np.quantile(times, quantile))
    max_time = float(np.max(times))
    if tau >= max_time:
        tau = max_time - 1e-6
    return tau

def safe_uno_cindex(y_train: np.ndarray, y_val: np.ndarray, risk_val: np.ndarray, tau: float) -> float:
    try:
        return float(concordance_index_ipcw(y_train, y_val, risk_val, tau=tau)[0])
    except Exception:
        return float("nan")

def get_feature_names(preprocessor: ColumnTransformer) -> Optional[List[str]]:
    try:
        if hasattr(preprocessor, "get_feature_names_out"):
            names = preprocessor.get_feature_names_out()
            return [str(x) for x in names]
        if hasattr(preprocessor, "get_feature_names"):
            names = preprocessor.get_feature_names()
            return [str(x) for x in names]
        return None
    except Exception:
        return None

def compute_split_overlap(train_ids: List[str], val_ids: List[str]) -> List[str]:
    return sorted(list(set(map(str, train_ids)).intersection(set(map(str, val_ids)))))

def random_risk_baseline(y_val: np.ndarray, n_trials: int = 200, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    cs = []
    for _ in range(n_trials):
        rr = rng.random(len(y_val))
        c = float(concordance_index_censored(y_val["event"], y_val["time"], rr)[0])
        cs.append(c)
    return {
        "random_risk_mean": float(np.mean(cs)),
        "random_risk_std": float(np.std(cs, ddof=1)) if len(cs) > 1 else 0.0,
        "n_trials": int(n_trials),
    }

@dataclass
class FoldData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray


# ---------------------------
# Data loading / splitting
# ---------------------------

def load_and_merge(features_csv: Path,
                   labels_csv: Path,
                   id_col: str,
                   time_col: str,
                   event_col: str) -> pd.DataFrame:
    X_df = pd.read_csv(features_csv).copy()
    y_df = pd.read_csv(labels_csv).copy()

    X_df[id_col] = X_df[id_col].astype(str)
    y_df[id_col] = y_df[id_col].astype(str)

    if time_col not in y_df.columns or event_col not in y_df.columns:
        raise ValueError(f"labels_csv missing columns: need [{time_col}, {event_col}]")

    # avoid duplication on merge
    if time_col in X_df.columns:
        X_df = X_df.drop(columns=[time_col])
    if event_col in X_df.columns:
        X_df = X_df.drop(columns=[event_col])

    y_keep = y_df[[id_col, time_col, event_col]].copy()
    df = X_df.merge(y_keep, on=id_col, how="inner")

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df.dropna(subset=[time_col, event_col]).copy()
    df = df[df[time_col] > 0].copy()

    df[event_col] = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int)
    df[event_col] = (df[event_col] > 0).astype(int)
    return df

def select_feature_cols(df: pd.DataFrame,
                        id_col: str,
                        time_col: str,
                        event_col: str,
                        feature_cols_arg: Optional[List[str]],
                        strict_cols: bool = False) -> List[str]:
    default_r1 = ["age", "gender", "Overall.Stage"]
    if feature_cols_arg and len(feature_cols_arg) > 0:
        cols = [c for c in feature_cols_arg if c in df.columns]
        missing = [c for c in feature_cols_arg if c not in df.columns]
        if len(missing) > 0:
            warn_or_raise_missing_cols("feature_cols", feature_cols_arg, missing, strict=strict_cols)
        if len(cols) == 0:
            raise ValueError(f"--feature_cols provided but none exist in CSV: {feature_cols_arg}")
        return cols
    if all(c in df.columns for c in default_r1):
        return default_r1
    excluded = {id_col, time_col, event_col}
    return [c for c in df.columns if c not in excluded]

def split_fold(df: pd.DataFrame,
               id_col: str,
               time_col: str,
               event_col: str,
               feature_cols: List[str],
               train_ids: List[str],
               val_ids: List[str]) -> FoldData:
    train_ids = [str(x) for x in train_ids]
    val_ids = [str(x) for x in val_ids]
    train_df = df[df[id_col].isin(train_ids)].copy()
    val_df = df[df[id_col].isin(val_ids)].copy()
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(f"Empty split: train={len(train_df)}, val={len(val_df)}. Check IDs / merge.")
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    y_train = make_surv_struct(train_df[event_col].to_numpy(), train_df[time_col].to_numpy())
    y_val = make_surv_struct(val_df[event_col].to_numpy(), val_df[time_col].to_numpy())
    return FoldData(train_df, val_df, X_train, X_val, y_train, y_val)


# ---------------------------
# Preprocessing
# ---------------------------

def make_onehot_encoder(drop_mode: str,
                        fixed_categories: Optional[List[np.ndarray]] = None) -> OneHotEncoder:
    """
    Version-safe OneHotEncoder.
    drop_mode: 'none' or 'first'
    fixed_categories: list aligned with cat_cols, e.g. [np.array([...]), ...]
    """
    drop = None if drop_mode == "none" else "first"
    kwargs: Dict[str, Any] = dict(handle_unknown="ignore", drop=drop)
    if fixed_categories is not None:
        kwargs["categories"] = fixed_categories

    # dense output across versions
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)

def infer_fixed_categories(X_train: pd.DataFrame, cat_cols: List[str]) -> List[np.ndarray]:
    """
    Fix category sets from the OUTER-train data to keep feature dimensionality stable in inner-CV.
    """
    cats = []
    for c in cat_cols:
        s = X_train[c]
        # keep original types; OneHotEncoder supports numeric categories too.
        uniq = pd.Series(s.dropna().unique())
        # stable ordering
        try:
            uniq_sorted = np.array(sorted(list(uniq)))
        except Exception:
            uniq_sorted = uniq.astype(object).to_numpy()
        cats.append(uniq_sorted)
    return cats

def make_column_transformer(num_cols: List[str],
                            cat_cols: List[str],
                            drop_mode: str,
                            fixed_categories: Optional[List[np.ndarray]]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder(drop_mode=drop_mode, fixed_categories=fixed_categories)),
    ])
    try:
        ct = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )
    except TypeError:
        ct = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
        )
    return ct


# ---------------------------
# Inner-CV selection for Coxnet
# ---------------------------

def alpha_selection_1se(alphas: np.ndarray, mean_scores: np.ndarray, std_scores: np.ndarray) -> float:
    """
    1-SE rule: pick the *largest alpha* whose mean score >= (best_mean - best_std).
    Larger alpha => stronger regularization => sparser/more stable.
    """
    best_idx = int(np.nanargmax(mean_scores))
    thresh = float(mean_scores[best_idx] - std_scores[best_idx])
    # alphas is typically descending from alpha_max to alpha_min (but we won't assume)
    # We want the largest alpha satisfying condition.
    valid = np.where(mean_scores >= thresh)[0]
    if len(valid) == 0:
        return float(alphas[best_idx])
    # choose index corresponding to max alpha value
    idx = valid[int(np.argmax(alphas[valid]))]
    return float(alphas[idx])

def inner_cv_pick_alpha(
    X: pd.DataFrame,
    y: np.ndarray,
    num_cols: List[str],
    cat_cols: List[str],
    fixed_categories: Optional[List[np.ndarray]],
    drop_mode: str,
    l1_ratio: float,
    alphas: np.ndarray,
    inner_folds: int,
    seed: int,
    tol: float,
    max_iter: int,
) -> Dict[str, Any]:
    """
    Returns:
      - mean_scores per alpha
      - std_scores per alpha
      - alpha_best, alpha_1se
    """
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    strat = y["event"].astype(int)

    scores = np.full((inner_folds, len(alphas)), np.nan, dtype=float)

    for k, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(strat)), strat)):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        prep = make_column_transformer(num_cols, cat_cols, drop_mode=drop_mode, fixed_categories=fixed_categories)
        X_tr_t = prep.fit_transform(X_tr)
        X_va_t = prep.transform(X_va)

        model = CoxnetSurvivalAnalysis(
            l1_ratio=float(l1_ratio),
            alphas=np.asarray(alphas, dtype=float),
            alpha_min_ratio="auto",  # ignored because alphas provided
            n_alphas=len(alphas),
            tol=float(tol),
            max_iter=int(max_iter),
            normalize=False,
        )
        model.fit(X_tr_t, y_tr)
        if hasattr(model, "n_iter_") and hasattr(model, "max_iter"):
            try:
                if np.max(np.asarray(model.n_iter_)) >= model.max_iter:
                    print(f"[WARNING] Coxnet may not have converged: max(n_iter_)={np.max(model.n_iter_)} >= max_iter={model.max_iter}")
            except Exception:
                pass
        # evaluate each alpha on inner-val
        for j, a in enumerate(alphas):
            try:
                risk = model.predict(X_va_t, alpha=float(a))
                c = float(concordance_index_censored(y_va["event"], y_va["time"], risk)[0])
                scores[k, j] = c
            except Exception:
                scores[k, j] = np.nan

    mean_scores = np.nanmean(scores, axis=0)
    std_scores = np.nanstd(scores, axis=0, ddof=1)

    best_idx = int(np.nanargmax(mean_scores))
    alpha_best = float(alphas[best_idx])
    alpha_1se = alpha_selection_1se(alphas, mean_scores, std_scores)

    return {
        "scores_matrix": scores,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "alpha_best": alpha_best,
        "alpha_1se": alpha_1se,
        "best_mean": float(mean_scores[best_idx]),
        "best_std": float(std_scores[best_idx]),
    }

def fit_full_get_alphas(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    num_cols: List[str],
    cat_cols: List[str],
    fixed_categories: Optional[List[np.ndarray]],
    drop_mode: str,
    l1_ratio: float,
    n_alphas: int,
    alpha_min_ratio: str,
    tol: float,
    max_iter: int,
) -> Tuple[ColumnTransformer, np.ndarray]:
    """
    Fit preprocess on full outer-train, then fit Coxnet once to get alphas_ path.
    """
    prep = make_column_transformer(num_cols, cat_cols, drop_mode=drop_mode, fixed_categories=fixed_categories)
    X_t = prep.fit_transform(X_train)

    model = CoxnetSurvivalAnalysis(
        l1_ratio=float(l1_ratio),
        n_alphas=int(n_alphas),
        alphas=None,
        alpha_min_ratio=alpha_min_ratio,
        tol=float(tol),
        max_iter=int(max_iter),
        normalize=False,
    )
    model.fit(X_t, y_train)
    return prep, np.asarray(model.alphas_, dtype=float)


# ---------------------------
# Main R2 S2-EN
# ---------------------------

def run_r2_s2_en(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    np.random.seed(args.seed)

    df = load_and_merge(Path(args.features_csv), Path(args.labels_csv),
                        args.id_col, args.time_col, args.event_col)

    feature_cols = select_feature_cols(df, args.id_col, args.time_col, args.event_col, args.feature_cols, strict_cols=args.strict_cols)

    # explicit categorical columns
    if args.cat_cols and len(args.cat_cols) > 0:
        missing_cat = [c for c in args.cat_cols if c not in df.columns]
        warn_or_raise_missing_cols("cat_cols", args.cat_cols, missing_cat, strict=args.strict_cols)

        ignored_cat = [c for c in args.cat_cols if c in df.columns and c not in feature_cols]
        if len(ignored_cat) > 0:
            msg = f"[WARNING] The following cat_cols exist in CSV but are NOT in feature_cols (so they will be IGNORED): {ignored_cat}. Did you forget to add them to --feature_cols?"
            print(msg)
        
        cat_cols = [c for c in args.cat_cols if c in feature_cols]
        num_cols = [c for c in feature_cols if c not in cat_cols]
    else:
        cat_cols = [c for c in feature_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
        num_cols = [c for c in feature_cols if c not in cat_cols]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns selected.")
    if len(num_cols) + len(cat_cols) != len(feature_cols):
        raise RuntimeError("num/cat split inconsistent.")

    splits = read_json(Path(args.splits_json))
    fold_keys = sorted(list(splits.keys()))

    all_fold_metrics = []
    oof_rows = []
    random_baseline_rows = []

    for fold_name in fold_keys:
        fold_start = time.time()

        train_ids = splits[fold_name]["train"]
        val_ids = splits[fold_name]["val"]

        overlap = compute_split_overlap(train_ids, val_ids)
        if len(overlap) > 0:
            raise ValueError(f"[R2S2EN][{fold_name}] train/val overlap: {overlap[:10]} (n={len(overlap)})")
        print(f"[R2S2EN][{fold_name}] split ok: train={len(train_ids)}, val={len(val_ids)}")

        try:
            fold_idx = int(str(fold_name).split("_")[-1])
        except Exception:
            fold_idx = fold_keys.index(fold_name)

        fd = split_fold(df, args.id_col, args.time_col, args.event_col, feature_cols, train_ids, val_ids)

        # Fix category sets from OUTER-train to stabilize inner-CV feature dimension
        fixed_categories = infer_fixed_categories(fd.X_train, cat_cols) if len(cat_cols) > 0 else None

        # --- inner-CV search over l1_ratio + alpha ---
        best_combo = None

        inner_summary_rows = []

        for l1 in args.l1_ratios:
            # 1) get alpha path on full outer-train for this l1_ratio
            prep_full, alpha_path = fit_full_get_alphas(
                fd.X_train, fd.y_train,
                num_cols=num_cols,
                cat_cols=cat_cols,
                fixed_categories=fixed_categories,
                drop_mode=args.onehot_drop,
                l1_ratio=float(l1),
                n_alphas=args.n_alphas,
                alpha_min_ratio=args.alpha_min_ratio,
                tol=args.tol,
                max_iter=args.max_iter,
            )

            # 2) inner-CV evaluate each alpha on that path
            cv = inner_cv_pick_alpha(
                X=fd.X_train,
                y=fd.y_train,
                num_cols=num_cols,
                cat_cols=cat_cols,
                fixed_categories=fixed_categories,
                drop_mode=args.onehot_drop,
                l1_ratio=float(l1),
                alphas=alpha_path,
                inner_folds=args.inner_folds,
                seed=args.seed + 10 * fold_idx,
                tol=args.tol,
                max_iter=args.max_iter,
            )

            alpha_best = cv["alpha_best"]
            alpha_1se = cv["alpha_1se"]
            best_mean = cv["best_mean"]

            inner_summary_rows.append({
                "fold": int(fold_idx),
                "l1_ratio": float(l1),
                "alpha_best": float(alpha_best),
                "alpha_1se": float(alpha_1se),
                "best_mean_cindex": float(best_mean),
                "n_alphas": int(len(alpha_path)),
            })

            if best_combo is None or best_mean > best_combo["best_mean_cindex"]:
                best_combo = {
                    "l1_ratio": float(l1),
                    "alpha_path": alpha_path,
                    "prep_full": prep_full,
                    "cv": cv,
                    "best_mean_cindex": float(best_mean),
                    "alpha_best": float(alpha_best),
                    "alpha_1se": float(alpha_1se),
                }

        assert best_combo is not None

        # choose alpha per rule (best vs 1se)
        alpha_selected = best_combo["alpha_best"] if args.alpha_select == "best" else best_combo["alpha_1se"]
        l1_selected = best_combo["l1_ratio"]
        alpha_path = best_combo["alpha_path"]

        # --- fit final model on full outer-train with selected hyperparams ---
        prep_final = make_column_transformer(num_cols, cat_cols, drop_mode=args.onehot_drop, fixed_categories=fixed_categories)
        X_train_t = prep_final.fit_transform(fd.X_train)
        X_val_t = prep_final.transform(fd.X_val)

        n_features_after = int(X_train_t.shape[1])
        print(f"[R2S2EN][{fold_name}] features: raw={len(feature_cols)} -> after_preprocess={n_features_after}")
        if n_features_after == 0:
            raise ValueError(f"[R2S2EN][{fold_name}] No features after preprocessing!")
        if n_features_after > args.warn_feature_dim_gt:
            print(f"[WARNING][R2S2EN][{fold_name}] unexpectedly many features: {n_features_after}")

        model_final = CoxnetSurvivalAnalysis(
            l1_ratio=float(l1_selected),
            alphas=np.asarray(alpha_path, dtype=float),
            n_alphas=len(alpha_path),
            alpha_min_ratio="auto",
            tol=float(args.tol),
            max_iter=int(args.max_iter),
            normalize=False,
        )
        model_final.fit(X_train_t, fd.y_train)

        # Predict on outer-val using selected alpha
        risk_val = model_final.predict(X_val_t, alpha=float(alpha_selected))

        # Metrics
        tau = compute_tau_from_train(fd.y_train, quantile=args.tau_q)
        harrell_c = float(concordance_index_censored(fd.y_val["event"], fd.y_val["time"], risk_val)[0])
        uno_c = safe_uno_cindex(fd.y_train, fd.y_val, risk_val, tau=tau)

        train_stats = {
            "n_events": int(fd.y_train["event"].sum()),
            "n_censored": int((~fd.y_train["event"]).sum()),
            "event_rate": round(float(fd.y_train["event"].mean()), 6),
            "median_time": round(float(np.median(fd.y_train["time"])), 6),
        }
        val_stats = {
            "n_events": int(fd.y_val["event"].sum()),
            "n_censored": int((~fd.y_val["event"]).sum()),
            "event_rate": round(float(fd.y_val["event"].mean()), 6),
            "median_time": round(float(np.median(fd.y_val["time"])), 6),
        }

        fold_time = time.time() - fold_start

        exp_id = getattr(args, "exp_id", "R2_S2_EN")
        metrics = {
            "exp_id": str(exp_id),
            "fold": int(fold_idx),
            "fold_name": str(fold_name),
            "n_train": int(len(fd.train_df)),
            "n_val": int(len(fd.val_df)),
            "train_stats": train_stats,
            "val_stats": val_stats,
            "harrell_c": round(float(harrell_c), 6),
            "uno_c": None if np.isnan(uno_c) else round(float(uno_c), 6),
            "tau": round(float(tau), 6),
            "tau_quantile": float(args.tau_q),
            "n_features_raw": int(len(feature_cols)),
            "n_features_after_preprocess": int(n_features_after),
            "feature_cols_raw": feature_cols,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "onehot_drop": args.onehot_drop,
            "inner_folds": int(args.inner_folds),
            "l1_ratios_tested": [float(x) for x in args.l1_ratios],
            "l1_ratio_selected": float(l1_selected),
            "alpha_selected": float(alpha_selected),
            "alpha_select_rule": str(args.alpha_select),
            "runtime_seconds": round(float(fold_time), 2),
        }

        fold_dir = outdir / f"fold_{fold_idx}"
        ensure_dir(fold_dir)

        # Save predictions
        pred_df = pd.DataFrame({
            args.id_col: fd.val_df[args.id_col].astype(str).to_numpy(),
            "time": fd.y_val["time"].astype(float),
            "event": fd.y_val["event"].astype(bool),
            "risk_pred": risk_val.astype(float),
        })
        pred_df.to_csv(fold_dir / "predictions.csv", index=False)

        # Save model pack (preprocessor + coxnet + alpha)
        model_pack = {
            "preprocessor": prep_final,
            "coxnet": model_final,
            "alpha_selected": float(alpha_selected),
            "l1_ratio_selected": float(l1_selected),
            "feature_cols_raw": feature_cols,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "onehot_drop": args.onehot_drop,
        }
        joblib.dump(model_pack, fold_dir / "model_pack.joblib")

        # Save inner-CV summary (per l1_ratio)
        pd.DataFrame(inner_summary_rows).to_csv(fold_dir / "inner_cv_summary.csv", index=False)

        # Save full alpha-path performance for selected l1_ratio (mean/std per alpha)
        cv = best_combo["cv"]
        alpha_path_df = pd.DataFrame({
            "alpha": alpha_path,
            "mean_cindex": cv["mean_scores"],
            "std_cindex": cv["std_scores"],
        })
        alpha_path_df.to_csv(fold_dir / "alpha_path_scores.csv", index=False)

        # Save coefficients at selected alpha
        try:
            # find index in model_final.alphas_
            alphas_used = np.asarray(model_final.alphas_, dtype=float)
            idx = int(np.argmin(np.abs(alphas_used - float(alpha_selected))))
            coef_vec = np.asarray(model_final.coef_[:, idx], dtype=float).reshape(-1)

            feat_names = get_feature_names(prep_final)
            if feat_names is not None and len(feat_names) == len(coef_vec):
                coef_dict = {name: float(c) for name, c in zip(feat_names, coef_vec)}
                nonzero = [name for name, c in coef_dict.items() if abs(c) > 1e-12]
            else:
                coef_dict = {f"feature_{i}": float(c) for i, c in enumerate(coef_vec)}
                nonzero = [k for k, v in coef_dict.items() if abs(v) > 1e-12]

            write_json(fold_dir / "coefficients.json", {
                "alpha_selected": float(alpha_selected),
                "l1_ratio_selected": float(l1_selected),
                "n_features": int(len(coef_vec)),
                "feature_names_available": bool(feat_names is not None),
                "n_nonzero": int(len(nonzero)),
                "nonzero_features": nonzero,
                "coefficients": coef_dict,
            })
        except Exception as e:
            print(f"[WARNING][R2S2EN][{fold_name}] could not extract coefficients: {e}")

        # Save metrics/params
        write_json(fold_dir / "metrics.json", metrics)
        write_json(fold_dir / "params.json", {
            "features_csv": str(args.features_csv),
            "labels_csv": str(args.labels_csv),
            "splits_json": str(args.splits_json),
            "id_col": args.id_col,
            "time_col": args.time_col,
            "event_col": args.event_col,
            "feature_cols_raw": feature_cols,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "onehot_drop": args.onehot_drop,
            "l1_ratios": [float(x) for x in args.l1_ratios],
            "inner_folds": int(args.inner_folds),
            "n_alphas": int(args.n_alphas),
            "alpha_min_ratio": str(args.alpha_min_ratio),
            "alpha_select": str(args.alpha_select),
            "tau_q": float(args.tau_q),
            "seed": int(args.seed),
            "tol": float(args.tol),
            "max_iter": int(args.max_iter),
        })

        # Collect OOF rows
        for r in pred_df.itertuples(index=False):
            oof_rows.append({
                args.id_col: getattr(r, args.id_col),
                "fold": int(fold_idx),
                "time": float(r.time),
                "event": bool(r.event),
                "risk_pred": float(r.risk_pred),
            })

        # Random-risk sanity check
        rb = random_risk_baseline(fd.y_val, n_trials=args.random_risk_trials, seed=args.seed + 1000 + fold_idx)
        random_baseline_rows.append({"fold": int(fold_idx), **rb})

        all_fold_metrics.append(metrics)

    # Save OOF predictions
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(outdir / "oof_predictions.csv", index=False)

    # Save per-fold metrics table
    met_df = pd.DataFrame(all_fold_metrics)
    met_df.to_csv(outdir / "per_fold_metrics.csv", index=False)

    # Save random baseline
    pd.DataFrame(random_baseline_rows).to_csv(outdir / "random_risk_baseline.csv", index=False)

    # Per-fold averages
    har_mean, har_std = mean_std(met_df["harrell_c"])
    uno_mean, uno_std = mean_std(met_df["uno_c"])

    # Overall OOF Harrell/Uno (Uno here uses OOF as both train/test; ok as a summary, but not the main number)
    harrell_c_oof = float("nan")
    uno_c_oof = float("nan")
    if len(oof_rows) > 0:
        try:
            events_oof = oof_df["event"].to_numpy().astype(bool)
            times_oof = oof_df["time"].to_numpy().astype(float)
            risks_oof = oof_df["risk_pred"].to_numpy().astype(float)

            harrell_c_oof = float(concordance_index_censored(events_oof, times_oof, risks_oof)[0])

            y_oof_struct = make_surv_struct(events_oof, times_oof)
            tau_oof = compute_tau_from_train(y_oof_struct, quantile=args.tau_q)
            uno_c_oof = float(concordance_index_ipcw(y_oof_struct, y_oof_struct, risks_oof, tau=tau_oof)[0])
        except Exception as e:
            print(f"[WARNING] Failed to compute OOF metrics: {e}")

    exp_id = getattr(args, "exp_id", "R2_S2_EN")
    summary = {
        "exp_id": str(exp_id),
        "n_folds": int(len(met_df)),
        "harrell_c_mean": round(float(har_mean), 6),
        "harrell_c_std": round(float(har_std), 6),
        "uno_c_mean": None if np.isnan(uno_mean) else round(float(uno_mean), 6),
        "uno_c_std": None if np.isnan(uno_std) else round(float(uno_std), 6),
        "harrell_c_oof": None if np.isnan(harrell_c_oof) else round(float(harrell_c_oof), 6),
        "uno_c_oof": None if np.isnan(uno_c_oof) else round(float(uno_c_oof), 6),
        "note": "R2 S2-EN: Coxnet (Elastic-Net) with inner-CV on outer-train; evaluate on outer-val.",
    }
    write_json(outdir / "summary.json", summary)

    print("\n[R2_S2_EN] Done.")
    print(f"[R2_S2_EN] Per-fold averages:")
    print(f"     Harrell C: mean={summary['harrell_c_mean']} std={summary['harrell_c_std']}")
    print(f"     Uno C:     mean={summary['uno_c_mean']} std={summary['uno_c_std']}")
    print(f"[R2_S2_EN] Overall OOF (combined):")
    print(f"     Harrell C (OOF): {summary['harrell_c_oof']}")
    print(f"     Uno C (OOF):     {summary['uno_c_oof']}")
    print(f"[R2_S2_EN] Outputs written to: {outdir.resolve()}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Track R - R2 S2-EN: Coxnet (Elastic-Net) + inner-CV on outer-train")
    ap.add_argument("--features_csv", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--id_col", type=str, default="PatientID")
    ap.add_argument("--time_col", type=str, default="Survival.time")
    ap.add_argument("--event_col", type=str, default="deadstatus.event")

    ap.add_argument("--feature_cols", type=str, nargs="*", default=None)
    ap.add_argument("--cat_cols", type=str, nargs="*", default=None)

    ap.add_argument("--onehot_drop", type=str, choices=["none", "first"], default="none",
                    help="OneHotEncoder drop mode. For penalized models, 'none' avoids bias across categories.")
    ap.add_argument("--l1_ratios", type=float, nargs="*", default=[0.1, 0.5, 0.9],
                    help="Elastic-net mixing values to search.")
    ap.add_argument("--inner_folds", type=int, default=5)

    ap.add_argument("--n_alphas", type=int, default=100)
    ap.add_argument("--alpha_min_ratio", type=str, default="auto",
                    help="Coxnet alpha_min_ratio when generating alpha path (if alphas=None).")
    ap.add_argument("--alpha_select", type=str, choices=["best", "1se"], default="best",
                    help="Pick alpha by best mean inner-CV c-index or 1-SE rule.")

    ap.add_argument("--tau_q", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--max_iter", type=int, default=100000)

    ap.add_argument("--warn_feature_dim_gt", type=int, default=200)
    ap.add_argument("--random_risk_trials", type=int, default=200)
    
    ap.add_argument("--exp_id", type=str, default="R2_S2_EN",
                    help="Experiment ID for output naming.")
    ap.add_argument("--strict_cols", action="store_true",
                    help="If set, error out when any provided feature_cols/cat_cols do not exist in CSV.")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_r2_s2_en(args)


#run command:
#python r2_s2_en.py \
#  --features_csv /home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
#  --labels_csv /home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
#  --splits_json /home/lichengze/Research/DeepFeature/myresearch/main/R0/splits.json \
#  --outdir /home/lichengze/Research/DeepFeature/myresearch/main/R2/R2_EN_no_M \
#  --exp_id R2_EN_no_M \
#  --id_col PatientID \
#  --time_col "Survival.time" \
#  --event_col "deadstatus.event" \
#  --feature_cols age gender Overall.Stage clinical.T.Stage Clinical.N.Stage \
#  --cat_cols gender Overall.Stage clinical.T.Stage Clinical.N.Stage \
#  --onehot_drop none \
#  --l1_ratios 0.01 0.05 0.1 0.3 0.5 0.7 0.9 1.0 \
#  --inner_folds 5 \
#  --alpha_select best \
#  --strict_cols \
#  --tau_q 0.9 \
#  --seed 42
