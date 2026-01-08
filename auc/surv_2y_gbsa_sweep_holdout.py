#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Holdout (non-CV) hyperparameter sweep for GBSA 2-year survival probability.

Why this script exists
----------------------
You already have a *scientifically valid* outer split (train/test) given by split_json.
To improve probability quality (AUC@2y, Brier@2y) without tuning on the test set, we:

1) Keep the outer test set **frozen**.
2) Split the outer training set into an **inner train / inner val** (single holdout).
3) Sweep a small set of GBSA hyperparameters around your V3-regularized baseline.
4) For each candidate, fit Platt calibration using **OOF predictions on inner train**
   (calibration folds are only for calibration, not for model selection CV).
5) Select the best candidate by **lowest calibrated Brier@2y on inner val**
   (tie-breaker: higher AUC@2y).
6) Retrain the final model on the full outer train and evaluate once on the frozen outer test.

This is the cleanest way to try to beat V3 numerically *without overfitting*.

Prerequisites
-------------
Place this file in the same folder as `surv_2y_system_v3.py` so we can import it.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import surv_2y_system_v3 as v3


def _ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _make_inner_split(
    df_train: pd.DataFrame,
    horizon: float,
    val_ratio: float,
    val_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Single holdout inside outer-train.

    Stratify by "death before horizon" so early-death rate is similar in inner splits.
    """

    t = df_train["Survival.time"].astype(float).values
    e = df_train["deadstatus.event"].astype(int).values
    y_strat = ((e == 1) & (t < float(horizon))).astype(int)

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(val_ratio),
        random_state=int(val_seed),
    )
    idx = np.arange(len(df_train))
    tr_idx, va_idx = next(sss.split(idx, y_strat))
    df_inner_tr = df_train.iloc[tr_idx].copy()
    df_inner_va = df_train.iloc[va_idx].copy()
    return df_inner_tr, df_inner_va


def _default_param_grid(base: Dict[str, Any]) -> List[Dict[str, Any]]:
    """A small, targeted grid around the V3-regularized params.

    Design intent:
    - move along the (learning_rate, n_estimators) trade-off
    - test depth 2 vs 3
    - test stronger leaf regularization
    - small change to subsample and dropout
    """

    lr_n_pairs = [
        (0.05, 300),
        (0.05, 500),
        (0.10, 200),
        (0.10, 400),
        (0.20, 100),
        (0.20, 200),
    ]
    depths = [2, 3]
    leaves = [10, 20]
    subs = [0.5, 0.7]
    drops = [0.0, 0.1]

    out: List[Dict[str, Any]] = []
    for lr, n_est in lr_n_pairs:
        for d in depths:
            for leaf in leaves:
                for sub in subs:
                    for dr in drops:
                        p = dict(base)
                        p.update(
                            {
                                "learning_rate": float(lr),
                                "n_estimators": int(n_est),
                                "max_depth": int(d),
                                "min_samples_leaf": int(leaf),
                                "subsample": float(sub),
                                "dropout_rate": float(dr),
                            }
                        )
                        out.append(p)
    return out


def _score_candidate_on_val(
    *,
    df_inner_tr: pd.DataFrame,
    df_inner_va: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
    params: Dict[str, Any],
    horizon: float,
    age_impute: str,
    scale: str,
    seed: int,
    calibrate: str,
    calibrate_oof_folds: int,
    platt_C: float,
) -> Dict[str, Any]:
    """Train on inner_tr, calibrate using OOF(inner_tr), evaluate on inner_va."""

    # preprocess fit on inner_tr only
    prep = v3.fit_preprocess_bundle(
        df_inner_tr,
        numeric_cols,
        cat_cols,
        age_impute=age_impute,
        scale=scale,
    )
    Xtr = prep.transform(df_inner_tr)
    Xva = prep.transform(df_inner_va)
    ytr = v3.y_from_df(df_inner_tr)
    yva = v3.y_from_df(df_inner_va)

    # model
    model = v3.fit_gbsa(Xtr, ytr, params=params, seed=seed)
    risk_tr = model.predict(Xtr)
    risk_va = model.predict(Xva)
    p2_va_raw = v3.predict_survival_at(model, Xva, horizon)

    # calibration (trained on inner_tr only)
    if calibrate == "none":
        p2_va = v3.clip_probability(p2_va_raw)
        calib_obj = None
    else:
        # OOF raw probabilities on inner_tr
        p_oof = v3.oof_pred_p2_raw(
            df_inner_tr,
            numeric_cols,
            cat_cols,
            params=params,
            horizon=horizon,
            n_folds=int(calibrate_oof_folds),
            age_impute=age_impute,
            scale=scale,
            seed=seed,
        )
        calib_obj = v3.fit_calibrator(
            calibrate="platt",
            time=ytr["time"],
            event=ytr["event"],
            p_raw=p_oof,
            horizon=horizon,
            seed=seed,
            platt_C=float(platt_C),
            pmin=1e-6,
            pmax=1.0 - 1e-6,
        )
        p2_va = v3.clip_probability(calib_obj.predict(v3.clip_probability(p2_va_raw)))

    # metrics
    m_raw = v3.compute_core_metrics(ytr, yva, risk_tr, risk_va, v3.clip_probability(p2_va_raw), horizon)
    m_cal = v3.compute_core_metrics(ytr, yva, risk_tr, risk_va, p2_va, horizon)

    out = {
        "brier_raw": float(m_raw["brier2y"]),
        "auc_raw": float(m_raw["auc2y"]),
        "cindex_raw": float(m_raw["cindex_test"]),
        "brier_cal": float(m_cal["brier2y"]),
        "auc_cal": float(m_cal["auc2y"]),
        "cindex_cal": float(m_cal["cindex_test"]),
        "train_cindex": float(m_cal["cindex_train"]),
        "train_test_gap": float(m_cal["cindex_train"] - m_cal["cindex_test"]),
    }
    return out


def cmd_sweep(args: argparse.Namespace) -> None:
    _ensure_dir(args.out_dir)

    base_params = _read_json(args.gbsa_base_params_json)
    if args.param_grid_json:
        grid_list = _read_json(args.param_grid_json)
        if not isinstance(grid_list, list):
            raise ValueError("param_grid_json must be a JSON list of dicts")
        candidates: List[Dict[str, Any]] = [dict(base_params, **d) for d in grid_list]
    else:
        candidates = _default_param_grid(base_params)

    if args.max_trials > 0:
        candidates = candidates[: int(args.max_trials)]

    selected_r = v3.read_txt_list(args.selected_radiomics_txt)
    split = v3.read_split_json(args.split_json)
    df_all = v3.load_and_merge(
        args.clinical_csv,
        args.radiomics_csv,
        selected_radiomics=selected_r,
        use_n_stage=bool(args.use_n_stage),
        use_m_stage=bool(args.use_m_stage),
    )
    df_tr, df_te = v3.make_train_test(df_all, split)

    horizon = float(args.horizon)
    df_inner_tr, df_inner_va = _make_inner_split(
        df_tr,
        horizon=horizon,
        val_ratio=float(args.val_ratio),
        val_seed=int(args.val_seed),
    )

    numeric_cols, cat_cols = v3.build_feature_lists(
        df_all,
        selected_r,
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        use_histology=bool(args.use_histology),
        use_t_stage=bool(args.use_t_stage),
        use_stage=bool(args.use_stage),
        use_n_stage=bool(args.use_n_stage),
        use_m_stage=bool(args.use_m_stage),
    )

    print("=== HOLDOUT SWEEP (no test tuning) ===")
    print(f"Outer split: train={len(df_tr)} test={len(df_te)}")
    print(f"Inner split: inner_train={len(df_inner_tr)} inner_val={len(df_inner_va)}")
    print(f"Candidates: {len(candidates)}")
    print(f"Select by: min(brier_cal@{int(horizon)}d), tie-break max(auc_cal)")

    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(candidates, start=1):
        # hard guardrail: prevent obviously overfit configs
        if float(p.get("learning_rate", 0.1)) > 0.3 and int(p.get("max_depth", 3)) >= 4:
            continue

        m = _score_candidate_on_val(
            df_inner_tr=df_inner_tr,
            df_inner_va=df_inner_va,
            numeric_cols=numeric_cols,
            cat_cols=cat_cols,
            params=p,
            horizon=horizon,
            age_impute=args.age_impute,
            scale=args.scale,
            seed=int(args.seed),
            calibrate=args.calibrate,
            calibrate_oof_folds=int(args.calibrate_oof_folds),
            platt_C=float(args.platt_C),
        )

        row = {
            "idx": i,
            **{k: p.get(k) for k in sorted(set(p.keys()))},
            **m,
        }
        rows.append(row)

        if i % 5 == 0 or i == len(candidates):
            print(
                f"[{i:>3}/{len(candidates)}] "
                f"brier_cal={m['brier_cal']:.4f} auc_cal={m['auc_cal']:.4f} "
                f"cindex_cal={m['cindex_cal']:.4f} gap={m['train_test_gap']:.3f}"
            )

    if not rows:
        raise RuntimeError("No candidates were evaluated. Check your grid / guardrails.")

    df_res = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "sweep_results.csv")
    df_res.to_csv(out_csv, index=False)

    # select best: lowest brier_cal, then highest auc_cal
    df_sorted = df_res.sort_values(
        by=["brier_cal", "auc_cal", "cindex_cal"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    best = df_sorted.iloc[0].to_dict()
    best_params = {k: best[k] for k in base_params.keys()}
    # ensure types
    for k in [
        "n_estimators",
        "max_depth",
        "max_leaf_nodes",
        "min_samples_leaf",
        "min_samples_split",
    ]:
        if k in best_params and pd.notna(best_params[k]):
            best_params[k] = int(best_params[k])
    for k in [
        "learning_rate",
        "subsample",
        "max_features",
        "min_impurity_decrease",
        "min_weight_fraction_leaf",
        "dropout_rate",
    ]:
        if k in best_params and pd.notna(best_params[k]):
            best_params[k] = float(best_params[k])

    best_params_path = os.path.join(args.out_dir, "best_params.json")
    _write_json(best_params, best_params_path)

    best_summary = {
        "inner_val": {
            "brier_cal": float(best["brier_cal"]),
            "auc_cal": float(best["auc_cal"]),
            "cindex_cal": float(best["cindex_cal"]),
            "gap": float(best["train_test_gap"]),
        },
        "best_params": best_params,
        "sweep_results_csv": out_csv,
    }
    _write_json(best_summary, os.path.join(args.out_dir, "best_summary.json"))

    print("\n=== BEST ON INNER-VAL ===")
    print(json.dumps(best_summary["inner_val"], indent=2))
    print("Best params saved:", best_params_path)
    print("Results saved:", out_csv)

    # ------------------------------------------------------------------
    # Retrain on full outer-train and evaluate once on outer-test
    # ------------------------------------------------------------------
    print("\n=== RETRAIN BEST ON OUTER TRAIN, EVAL ON FROZEN OUTER TEST ===")

    prep = v3.fit_preprocess_bundle(
        df_tr,
        numeric_cols,
        cat_cols,
        age_impute=args.age_impute,
        scale=args.scale,
    )
    Xtr = prep.transform(df_tr)
    Xte = prep.transform(df_te)
    ytr = v3.y_from_df(df_tr)
    yte = v3.y_from_df(df_te)

    model = v3.fit_gbsa(Xtr, ytr, params=best_params, seed=int(args.seed))
    risk_tr = model.predict(Xtr)
    risk_te = model.predict(Xte)
    p2_te_raw = v3.predict_survival_at(model, Xte, horizon)

    if args.calibrate == "none":
        calibrator = None
        p2_te = v3.clip_probability(p2_te_raw)
    else:
        p_oof = v3.oof_pred_p2_raw(
            df_tr,
            numeric_cols,
            cat_cols,
            params=best_params,
            horizon=horizon,
            n_folds=int(args.calibrate_oof_folds),
            age_impute=args.age_impute,
            scale=args.scale,
            seed=int(args.seed),
        )
        calibrator = v3.fit_calibrator(
            calibrate="platt",
            time=ytr["time"],
            event=ytr["event"],
            p_raw=p_oof,
            horizon=horizon,
            seed=int(args.seed),
            platt_C=float(args.platt_C),
            pmin=1e-6,
            pmax=1.0 - 1e-6,
        )
        p2_te = v3.clip_probability(calibrator.predict(v3.clip_probability(p2_te_raw)))

    m_final = v3.compute_core_metrics(ytr, yte, risk_tr, risk_te, p2_te, horizon)
    print(
        f"[OUTER TEST] cindex={m_final['cindex_test']:.4f} "
        f"auc2y={m_final['auc2y']:.4f} brier2y={m_final['brier2y']:.4f}"
    )

    # save final bundle for deployment
    bundle = {
        "model": model,
        "prep": prep,
        "calibrator": calibrator,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "horizon": horizon,
        "gbsa_params": best_params,
        "scale": args.scale,
        "age_impute": args.age_impute,
        "selected_radiomics": selected_r,
        "clinical_flags": {
            "use_age": bool(args.use_age),
            "use_gender": bool(args.use_gender),
            "use_histology": bool(args.use_histology),
            "use_t_stage": bool(args.use_t_stage),
            "use_stage": bool(args.use_stage),
            "use_n_stage": bool(args.use_n_stage),
            "use_m_stage": bool(args.use_m_stage),
        },
    }
    model_path = os.path.join(args.out_dir, "gbsa_2y_model_best.joblib")
    v3.joblib.dump(bundle, model_path)
    _write_json(m_final, os.path.join(args.out_dir, "best_outer_test_report.json"))

    # save predictions on outer test (for audit)
    ids_te = df_te["PatientID"].astype(str).values
    out_pred = pd.DataFrame(
        {
            "PatientID": ids_te,
            "risk": risk_te,
            "p2y": p2_te,
            "time": df_te["Survival.time"].astype(float).values,
            "event": df_te["deadstatus.event"].astype(int).values,
        }
    )
    out_pred.to_csv(os.path.join(args.out_dir, "outer_test_pred_2y.csv"), index=False)

    print("[OK] Saved:")
    print(" -", out_csv)
    print(" -", best_params_path)
    print(" -", model_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Holdout hyperparameter sweep for GBSA 2y survival probability (no test tuning)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="run holdout sweep and retrain best on frozen outer test")
    s.add_argument("--clinical_csv", required=True)
    s.add_argument("--radiomics_csv", required=True)
    s.add_argument("--split_json", required=True)
    s.add_argument("--selected_radiomics_txt", required=True)
    s.add_argument("--gbsa_base_params_json", required=True)
    s.add_argument("--out_dir", required=True)
    s.add_argument("--seed", type=int, default=91)

    s.add_argument("--horizon", type=float, default=730.0)
    s.add_argument("--scale", choices=["minmax", "zscore", "none"], default="minmax")
    s.add_argument("--age_impute", choices=["mean", "median"], default="mean")

    s.add_argument("--use_age", type=int, default=1)
    s.add_argument("--use_gender", type=int, default=1)
    s.add_argument("--use_histology", type=int, default=1)
    s.add_argument("--use_t_stage", type=int, default=1)
    s.add_argument("--use_stage", type=int, default=1)
    s.add_argument("--use_n_stage", type=int, default=0)
    s.add_argument("--use_m_stage", type=int, default=0)

    s.add_argument("--calibrate", choices=["none", "platt"], default="platt")
    s.add_argument("--calibrate_oof_folds", type=int, default=5)
    s.add_argument("--platt_C", type=float, default=10.0)

    s.add_argument("--val_ratio", type=float, default=0.2)
    s.add_argument("--val_seed", type=int, default=2026)

    s.add_argument(
        "--param_grid_json",
        default="",
        help="Optional JSON list of param overrides (list[dict]). If omitted, use default small grid.",
    )
    s.add_argument("--max_trials", type=int, default=0, help="Limit number of candidates (0 = no limit)")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "sweep":
        cmd_sweep(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
