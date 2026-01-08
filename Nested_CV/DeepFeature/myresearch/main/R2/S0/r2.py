#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sksurv.linear_model import CoxPHSurvivalAnalysis
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

def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def mean_std(series: pd.Series) -> Tuple[float, float]:
    s = series.dropna().astype(float)
    if len(s) == 0:
        return float("nan"), float("nan")
    return float(s.mean()), float(s.std(ddof=1)) if len(s) > 1 else 0.0

def make_surv_struct(event: np.ndarray, time_arr: np.ndarray) -> np.ndarray:
    """Return structured array y with dtype [('event', bool), ('time', float)]."""
    e = event.astype(bool)
    t = time_arr.astype(float)
    y = np.empty(len(t), dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = e
    y["time"] = t
    return y

def compute_tau_from_train(y_train: np.ndarray, quantile: float = 0.9) -> float:
    """tau = quantile of EVENT times in training; fallback to all times if too few events."""
    times = y_train["time"].astype(float)
    events = y_train["event"].astype(bool)
    ev_times = times[events]
    if ev_times.shape[0] >= 5:
        tau = float(np.quantile(ev_times, quantile))
    else:
        tau = float(np.quantile(times, quantile))
    # ensure tau < max_time
    max_time = float(np.max(times))
    if tau >= max_time:
        tau = max_time - 1e-6
    return tau

def safe_uno_cindex(y_train: np.ndarray, y_val: np.ndarray, risk_val: np.ndarray, tau: float) -> float:
    """Uno C-index (IPCW) with defensive exception handling."""
    try:
        # returns (cindex, concordant, discordant, tied_risk, tied_time)
        return float(concordance_index_ipcw(y_train, y_val, risk_val, tau=tau)[0])
    except Exception:
        return float("nan")

def make_onehot_encoder() -> OneHotEncoder:
    """Version-safe OneHotEncoder: sparse_output (new) vs sparse (old)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop='first')
    except TypeError:
        # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False, drop='first')

def make_column_transformer(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Version-safe ColumnTransformer for feature-name output."""
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder()),
    ])

    # verbose_feature_names_out exists in newer sklearn; guard for old versions.
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

def get_feature_names(preprocessor: ColumnTransformer) -> Optional[List[str]]:
    """Robust feature name extraction across sklearn versions."""
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
    s1, s2 = set(train_ids), set(val_ids)
    return sorted(list(s1.intersection(s2)))

def random_risk_baseline(y_val: np.ndarray, n_trials: int = 200, seed: int = 42) -> Dict[str, float]:
    """Pure random risk baseline: should be ~0.5 (not exactly, but close)."""
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
# Core R2
# ---------------------------

def load_and_merge(features_csv: Path,
                   labels_csv: Path,
                   id_col: str,
                   time_col: str,
                   event_col: str) -> pd.DataFrame:
    X_df = pd.read_csv(features_csv).copy()
    y_df = pd.read_csv(labels_csv).copy()

    # normalize id to str
    X_df[id_col] = X_df[id_col].astype(str)
    y_df[id_col] = y_df[id_col].astype(str)

    # keep only needed label columns
    if time_col not in y_df.columns or event_col not in y_df.columns:
        raise ValueError(f"labels_csv missing columns: need [{time_col}, {event_col}]")

    # If features_csv and labels_csv are the same file, or if X_df already has time/event columns,
    # drop them from X_df to avoid conflicts during merge
    if time_col in X_df.columns:
        X_df = X_df.drop(columns=[time_col])
    if event_col in X_df.columns:
        X_df = X_df.drop(columns=[event_col])

    y_keep = y_df[[id_col, time_col, event_col]].copy()
    df = X_df.merge(y_keep, on=id_col, how="inner")
    
    # 1) 空白字符串 -> NaN（比 df.replace('', np.nan) 更强）
    df = df.replace(r'^\s*$', np.nan, regex=True)
    
    # 2) 对"应该是数值"的特征列做强制数值化（脏值 -> NaN）
    #    这里可以写死你R2会用到的列名集合，或在run_r2里对num_cols做同样处理
    maybe_numeric = [
        "age",
        "clinical_T_Stage",
        "Clinical_N_Stage",
        "Clinical_M_Stage",
        "Clinical_M_binary",
        "Overall_Stage_I", "Overall_Stage_II", "Overall_Stage_IIIa", "Overall_Stage_IIIb", "Overall_Stage_Unknown",
    ]
    for c in maybe_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # basic cleaning: time positive
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")
    df = df.dropna(subset=[time_col, event_col]).copy()
    df = df[df[time_col] > 0].copy()

    # event to bool (0/1)
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int)
    df[event_col] = (df[event_col] > 0).astype(int)
    return df

def select_feature_cols(df: pd.DataFrame,
                        id_col: str,
                        time_col: str,
                        event_col: str,
                        feature_cols_arg: Optional[List[str]]) -> List[str]:
    default_r2 = ["age", "gender", "Overall.Stage"]

    if feature_cols_arg and len(feature_cols_arg) > 0:
        cols = [c for c in feature_cols_arg if c in df.columns]
        if len(cols) == 0:
            raise ValueError(f"--feature_cols provided but none exist in CSV: {feature_cols_arg}")
        return cols

    # if defaults present, use them
    if all(c in df.columns for c in default_r2):
        return default_r2

    # fallback: everything except id/time/event
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

def build_pipeline(num_cols: List[str], cat_cols: List[str], cox_alpha: float) -> Pipeline:
    prep = make_column_transformer(num_cols=num_cols, cat_cols=cat_cols)
    coxph = CoxPHSurvivalAnalysis(alpha=cox_alpha)
    pipe = Pipeline(steps=[
        ("prep", prep),
        ("coxph", coxph),
    ])
    return pipe

def run_r2(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # global seed (repro / any components using global RNG)
    np.random.seed(args.seed)

    df = load_and_merge(
        features_csv=Path(args.features_csv),
        labels_csv=Path(args.labels_csv),
        id_col=args.id_col,
        time_col=args.time_col,
        event_col=args.event_col,
    )

    feature_cols = select_feature_cols(df, args.id_col, args.time_col, args.event_col, args.feature_cols)

    # explicit categorical columns
    if args.cat_cols and len(args.cat_cols) > 0:
        cat_cols = [c for c in args.cat_cols if c in feature_cols]
        num_cols = [c for c in feature_cols if c not in cat_cols]
    else:
        # fallback: auto-detect
        cat_cols = [c for c in feature_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
        num_cols = [c for c in feature_cols if c not in cat_cols]

    if len(feature_cols) == 0:
        raise ValueError("No feature columns selected. Check --feature_cols / CSV columns.")
    if len(num_cols) + len(cat_cols) != len(feature_cols):
        raise RuntimeError("Internal error: num/cat split inconsistent.")

    # 强制数值化：对所有 num_cols 统一转换，防止 CSV dtype 为 object 导致后续处理失败
    # 这一步在进入 ColumnTransformer 之前执行，确保无论 CSV 来源如何，数值列都能被正确处理
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    splits = read_json(Path(args.splits_json))
    fold_keys = sorted(list(splits.keys()))

    all_fold_metrics = []
    oof_rows = []

    # a global random-risk baseline per fold (optional but useful)
    random_baseline_rows = []

    for fold_name in fold_keys:
        fold_start = time.time()

        train_ids = splits[fold_name]["train"]
        val_ids = splits[fold_name]["val"]

        overlap = compute_split_overlap(train_ids, val_ids)
        if len(overlap) > 0:
            raise ValueError(f"[R2][{fold_name}] train/val overlap detected: {overlap[:10]} (n={len(overlap)})")
        print(f"[R2][{fold_name}] train/val split verified: no overlap, train={len(train_ids)}, val={len(val_ids)}")

        try:
            fold_idx = int(str(fold_name).split("_")[-1])
        except Exception:
            fold_idx = fold_keys.index(fold_name)

        fd = split_fold(df, args.id_col, args.time_col, args.event_col, feature_cols, train_ids, val_ids)

        # Build + fit
        pipe = build_pipeline(num_cols=num_cols, cat_cols=cat_cols, cox_alpha=args.cox_alpha)
        pipe.fit(fd.X_train, fd.y_train)

        # Feature-dim sanity check (after preprocess)
        prep = pipe.named_steps["prep"]
        X_train_t = prep.transform(fd.X_train)
        n_features_after = int(X_train_t.shape[1])
        print(f"[R2][{fold_name}] Features: raw={len(feature_cols)} -> after_preprocess={n_features_after}")
        if n_features_after == 0:
            raise ValueError(f"[R2][{fold_name}] No features after preprocessing!")
        if n_features_after > args.warn_feature_dim_gt:
            print(f"[WARNING][R2][{fold_name}] Unexpectedly many features after preprocess: {n_features_after}")

        # Predict on val
        risk_val = pipe.predict(fd.X_val)

        # Metrics
        tau = compute_tau_from_train(fd.y_train, quantile=args.tau_q)
        harrell_c = float(concordance_index_censored(fd.y_val["event"], fd.y_val["time"], risk_val)[0])
        uno_c = safe_uno_cindex(fd.y_train, fd.y_val, risk_val, tau=tau)

        # Train/val stats (for paper/debug)
        train_stats = {
            "n_events": int(fd.y_train["event"].sum()),
            "n_censored": int((~fd.y_train["event"]).sum()),
            "event_rate": round(float(fd.y_train["event"].mean()), 6),
            "median_time": round(float(np.median(fd.y_train["time"])), 6),
            "median_event_time": (
                round(float(np.median(fd.y_train["time"][fd.y_train["event"]])), 6)
                if int(fd.y_train["event"].sum()) > 0 else None
            ),
        }
        val_stats = {
            "n_events": int(fd.y_val["event"].sum()),
            "n_censored": int((~fd.y_val["event"]).sum()),
            "event_rate": round(float(fd.y_val["event"].mean()), 6),
            "median_time": round(float(np.median(fd.y_val["time"])), 6),
        }

        fold_time = time.time() - fold_start

        metrics = {
            "exp_id": "R2",
            "fold": int(fold_idx),
            "fold_name": str(fold_name),
            "n_train": int(len(fd.train_df)),
            "n_val": int(len(fd.val_df)),
            "train_stats": train_stats,
            "val_stats": val_stats,
            "harrell_c": round(harrell_c, 6),
            "uno_c": None if np.isnan(uno_c) else round(float(uno_c), 6),
            "tau": round(float(tau), 6),
            "tau_quantile": float(args.tau_q),
            "n_features_raw": int(len(feature_cols)),
            "n_features_after_preprocess": int(n_features_after),
            "feature_cols_raw": feature_cols,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "cox_alpha": float(args.cox_alpha),
            "runtime_seconds": round(float(fold_time), 2),
        }

        fold_dir = outdir / f"fold_{fold_idx}"
        ensure_dir(fold_dir)

        # Save predictions (per fold)
        pred_df = pd.DataFrame({
            args.id_col: fd.val_df[args.id_col].astype(str).to_numpy(),
            "time": fd.y_val["time"].astype(float),
            "event": fd.y_val["event"].astype(bool),
            "risk_pred": risk_val.astype(float),
        })
        pred_df.to_csv(fold_dir / "predictions.csv", index=False)

        # Save model
        joblib.dump(pipe, fold_dir / "model.joblib")

        # Save coefficients (CoxPH) + feature names
        try:
            coxph_model = pipe.named_steps["coxph"]
            coef = np.asarray(coxph_model.coef_, dtype=float).reshape(-1)

            feat_names = get_feature_names(prep)
            if feat_names is not None and len(feat_names) == len(coef):
                coef_dict = {name: float(c) for name, c in zip(feat_names, coef)}
            else:
                coef_dict = {f"feature_{i}": float(c) for i, c in enumerate(coef)}

            write_json(fold_dir / "coefficients.json", {
                "n_features": int(len(coef)),
                "feature_names_available": bool(feat_names is not None),
                "coefficients": coef_dict,
            })
        except Exception as e:
            print(f"[WARNING][R2][{fold_name}] Could not extract coefficients: {e}")

        # Save metrics + params
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
            "cox_alpha": float(args.cox_alpha),
            "tau_q": float(args.tau_q),
            "seed": int(args.seed),
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

        # Random-risk baseline per fold (optional, but very good to keep)
        rb = random_risk_baseline(fd.y_val, n_trials=args.random_risk_trials, seed=args.seed + 1000 + fold_idx)
        rb_row = {"fold": int(fold_idx), **rb}
        random_baseline_rows.append(rb_row)

        all_fold_metrics.append(metrics)

    # Save OOF predictions
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(outdir / "oof_predictions.csv", index=False)

    # Save per-fold metrics table
    met_df = pd.DataFrame(all_fold_metrics)
    met_df.to_csv(outdir / "per_fold_metrics.csv", index=False)

    # Save random baseline
    pd.DataFrame(random_baseline_rows).to_csv(outdir / "random_risk_baseline.csv", index=False)

    # Summary - per-fold averages
    har_mean, har_std = mean_std(met_df["harrell_c"])
    uno_mean, uno_std = mean_std(met_df["uno_c"])

    # Compute overall OOF metrics (all folds combined)
    harrell_c_oof = float("nan")
    uno_c_oof = float("nan")
    
    if len(oof_rows) > 0:
        try:
            # Extract all OOF predictions
            events_oof = oof_df["event"].to_numpy().astype(bool)
            times_oof = oof_df["time"].to_numpy().astype(float)
            risks_oof = oof_df["risk_pred"].to_numpy().astype(float)
            
            # Compute Harrell C-index on all OOF predictions
            harrell_c_oof = float(concordance_index_censored(
                events_oof, times_oof, risks_oof
            )[0])
            
            # Compute Uno C-index on all OOF predictions
            # Note: Uno C requires a training set for IPCW estimation
            # Here we use all OOF data as both train and test (reasonable for OOF evaluation)
            y_oof_struct = make_surv_struct(events_oof, times_oof)
            
            # Compute tau from all OOF event times
            event_times_oof = times_oof[events_oof]
            if len(event_times_oof) >= 5:
                tau_oof = float(np.quantile(event_times_oof, args.tau_q))
            else:
                tau_oof = float(np.quantile(times_oof, args.tau_q))
            
            # Ensure tau < max time
            max_time_oof = float(np.max(times_oof))
            if tau_oof >= max_time_oof:
                tau_oof = max_time_oof - 1e-6
            
            # Compute Uno C-index
            uno_c_oof = float(concordance_index_ipcw(
                y_oof_struct, y_oof_struct, risks_oof, tau=tau_oof
            )[0])
            
        except Exception as e:
            print(f"[WARNING] Failed to compute OOF metrics: {e}")
            harrell_c_oof = float("nan")
            uno_c_oof = float("nan")

    summary = {
        "exp_id": "R2",
        "n_folds": int(len(met_df)),
        "harrell_c_mean": round(float(har_mean), 6),
        "harrell_c_std": round(float(har_std), 6),
        "uno_c_mean": None if np.isnan(uno_mean) else round(float(uno_mean), 6),
        "uno_c_std": None if np.isnan(uno_std) else round(float(uno_std), 6),
        "harrell_c_oof": None if np.isnan(harrell_c_oof) else round(float(harrell_c_oof), 6),
        "uno_c_oof": None if np.isnan(uno_c_oof) else round(float(uno_c_oof), 6),
        "note": "R2 = Clinical features with CoxPH, S0, evaluate on outer-val folds.",
        "note_oof": "harrell_c_oof and uno_c_oof are computed on all OOF predictions combined (not averaged across folds)",
    }
    write_json(outdir / "summary.json", summary)

    print("\n[R2] Done.")
    print(f"[R2] Per-fold averages:")
    print(f"     Harrell C: mean={summary['harrell_c_mean']} std={summary['harrell_c_std']}")
    print(f"     Uno C:     mean={summary['uno_c_mean']} std={summary['uno_c_std']}")
    print(f"[R2] Overall OOF metrics (all folds combined):")
    print(f"     Harrell C (OOF): {summary['harrell_c_oof']}")
    print(f"     Uno C (OOF):     {summary['uno_c_oof']}")
    print(f"[R2] Outputs written to: {outdir.resolve()}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Track R - R2: Clinical features with CoxPHSurvivalAnalysis")
    ap.add_argument("--features_csv", type=str, required=True, help="CSV with features (must include id_col).")
    ap.add_argument("--labels_csv", type=str, required=True, help="CSV with labels (must include id_col, time_col, event_col).")
    ap.add_argument("--splits_json", type=str, required=True, help="splits.json with fold_{k}:{train:[...],val:[...]}")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory for R2.")

    ap.add_argument("--id_col", type=str, default="PatientID")
    ap.add_argument("--time_col", type=str, default="Survival.time")
    ap.add_argument("--event_col", type=str, default="deadstatus.event")

    ap.add_argument("--feature_cols", type=str, nargs="*", default=None,
                    help="Explicit raw feature cols for R2. If omitted, use [age, gender, Overall.Stage] if present; else fallback to all non-label cols.")
    ap.add_argument("--cat_cols", type=str, nargs="*", default=["gender", "Overall.Stage"],
                    help="Explicit categorical columns (one-hot). Remaining features treated as numeric.")
    ap.add_argument("--cox_alpha", type=float, default=1e-4,
                    help="Ridge penalty alpha for CoxPHSurvivalAnalysis (small >0 improves numerical stability).")
    ap.add_argument("--tau_q", type=float, default=0.9,
                    help="Quantile for tau computed from training event times (for Uno C-index).")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warn_feature_dim_gt", type=int, default=50,
                    help="Warn if features after preprocess exceed this number (R2 should be small).")
    ap.add_argument("--random_risk_trials", type=int, default=200,
                    help="How many random-risk trials per fold to sanity-check metric behavior.")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_r2(args)

#run command: 
#python r2.py \
#  --features_csv /home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
#  --labels_csv /home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
#  --splits_json /home/lichengze/Research/DeepFeature/myresearch/main/R0/splits.json \
#  --outdir /home/lichengze/Research/DeepFeature/myresearch/main/R2/R2_TNM_all \
#  --id_col PatientID \
#  --time_col "Survival.time" \
#  --event_col "deadstatus.event" \
#  --feature_cols age gender Overall.Stage clinical.T.Stage Clinical.N.Stage Clinical.M.Stage \
#  --cat_cols gender Overall.Stage clinical.T.Stage Clinical.N.Stage Clinical.M.Stage \
#  --cox_alpha 1e-4 \
#  --tau_q 0.9 \
#  --seed 42
