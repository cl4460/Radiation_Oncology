#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
surv_horizon_gbsa_rsf_fusion.py

Goal:
- Try a strictly leakage-safe improvement over GBSA bagging by adding a diverse model family:
  RandomSurvivalForest (RSF), then fuse GBSA/RSF using ONLY OOF predictions on OUTER-TRAIN.

Key safety points:
- Outer test is NEVER used for choosing fusion weight.
- Fusion weight w is selected by minimizing IPCW Brier score on OOF predictions of outer-train.
- Final Platt calibrator is fit on OOF fused predictions of outer-train (same idea as your OOF Platt).
- If fusion does not improve OOF Brier by >= eps, we fall back to w=1.0 (pure GBSA).

Outputs:
- fused_model.joblib (GBSA ensemble + RSF + preprocess + fusion weight + calibrator)
- fused_test_pred.csv
- fused_report.json (including bootstrap CIs on outer-test)
"""

import argparse
import json
import os
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import brier_score, cumulative_dynamic_auc

# Reuse your proven preprocessing / IO / calibrator / bootstrap code
import surv_2y_system_v3 as v3


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _brier_ipcw(y_train, y_test, p_surv, horizon: float) -> float:
    """IPCW Brier score at a single horizon."""
    p = np.asarray(p_surv, dtype=float).reshape(-1, 1)
    t = np.asarray([horizon], dtype=float)
    _, bs = brier_score(y_train, y_test, p, t)
    return float(bs[0])


def _auc_dynamic(y_train, y_test, risk, horizon: float) -> float:
    """Cumulative dynamic AUC at a single horizon."""
    t = np.asarray([horizon], dtype=float)
    auc, _ = cumulative_dynamic_auc(y_train, y_test, np.asarray(risk, dtype=float), t)
    return float(auc[0])


def _oof_pred_rsf_raw(
    df: pd.DataFrame,
    numeric_cols,
    cat_cols,
    horizon: float,
    scale: str,
    age_impute: str,
    n_folds: int,
    seed: int,
    rsf_params: dict,
) -> np.ndarray:
    """
    OOF survival probability S(horizon) for RSF.
    IMPORTANT: preprocessing is fitted inside each fold (no leakage).
    """
    from sklearn.model_selection import KFold

    n = len(df)
    oof = np.full(n, np.nan, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for tr_idx, va_idx in kf.split(np.arange(n)):
        df_tr = df.iloc[tr_idx].copy()
        df_va = df.iloc[va_idx].copy()

        prep = v3.fit_preprocess_bundle(df_tr, numeric_cols, cat_cols, age_impute, scale)
        X_tr = prep.transform(df_tr)
        X_va = prep.transform(df_va)

        y_tr = v3.y_from_df(df_tr)

        rsf = RandomSurvivalForest(
            random_state=seed,
            **rsf_params,
        )
        rsf.fit(X_tr, y_tr)

        oof[va_idx] = v3.predict_survival_at(rsf, X_va, horizon)

    assert np.isfinite(oof).all(), "OOF RSF predictions contain NaN/inf."
    return oof


def _train_gbsa_ensemble(X, y, params: dict, seed0: int, n_models: int):
    models = []
    seeds = list(range(seed0, seed0 + n_models))
    for s in seeds:
        m = v3.fit_gbsa(X, y, params=params, seed=s)
        models.append(m)
    return models, seeds


def _predict_gbsa_ensemble_raw(models, X, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
    - p_surv_raw: averaged survival prob S(horizon)
    - risk_raw: averaged GBSA risk score (model.predict)
    """
    p_list = []
    r_list = []
    for m in models:
        p_list.append(v3.predict_survival_at(m, X, horizon))
        r_list.append(m.predict(X))
    p = np.mean(np.vstack(p_list), axis=0)
    r = np.mean(np.vstack(r_list), axis=0)
    return p, r


def train_eval(args: argparse.Namespace) -> None:
    _ensure_dir(args.out_dir)

    split = v3.read_split_json(args.split_json)
    gbsa_params = json.load(open(args.gbsa_params_json, "r"))
    selected_r = v3.read_txt_list(args.selected_radiomics_txt)

    # 1) Load + merge
    df = v3.load_and_merge(
        args.clinical_csv,
        args.radiomics_csv,
        selected_r,
        use_n_stage=bool(args.use_n_stage),
        use_m_stage=bool(args.use_m_stage),
    )
    df_tr, df_te = v3.make_train_test(df, split)

    # 2) Build feature lists (must match your system choice)
    numeric_cols, cat_cols = v3.build_feature_lists(
        df,
        selected_r,
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        use_histology=bool(args.use_histology),
        use_t_stage=bool(args.use_t_stage),
        use_stage=bool(args.use_stage),
        use_n_stage=bool(args.use_n_stage),
        use_m_stage=bool(args.use_m_stage),
    )

    # 3) Fit preprocess on OUTER-TRAIN
    prep = v3.fit_preprocess_bundle(df_tr, numeric_cols, cat_cols, args.age_impute, args.scale)
    Xtr = prep.transform(df_tr)
    Xte = prep.transform(df_te)
    ytr = v3.y_from_df(df_tr)
    yte = v3.y_from_df(df_te)

    print("============================================================")
    print(f"[FUSION] Outer split: train={len(df_tr)} test={len(df_te)}")
    print(f"Horizon: {args.horizon} days | scale={args.scale} | OOF folds={args.oof_folds}")
    print(f"GBSA ensemble: n_models={args.n_gbsa_models} seeds={args.seed}..{args.seed+args.n_gbsa_models-1}")
    print("RSF params:", json.dumps({
        "n_estimators": args.rsf_n_estimators,
        "min_samples_leaf": args.rsf_min_samples_leaf,
        "min_samples_split": args.rsf_min_samples_split,
        "max_features": args.rsf_max_features,
    }, indent=2))
    print("============================================================")

    # 4) OOF raw predictions on OUTER-TRAIN (leakage-safe)
    # 4.1 GBSA ensemble OOF: average across seeds
    print("[OOF] GBSA ensemble raw S(t) on outer-train ...")
    p_gbsa_oof_list = []
    for s in range(args.seed, args.seed + args.n_gbsa_models):
        p_s = v3.oof_pred_p2_raw(
            df_tr,
            numeric_cols,
            cat_cols,
            gbsa_params,
            args.horizon,
            args.scale,
            args.age_impute,
            n_folds=args.oof_folds,
            seed=s,
        )
        p_gbsa_oof_list.append(p_s)
    p_gbsa_oof = np.mean(np.vstack(p_gbsa_oof_list), axis=0)

    # 4.2 RSF OOF raw
    print("[OOF] RSF raw S(t) on outer-train ...")
    rsf_params = dict(
        n_estimators=int(args.rsf_n_estimators),
        min_samples_leaf=int(args.rsf_min_samples_leaf),
        min_samples_split=int(args.rsf_min_samples_split),
        max_features=args.rsf_max_features,
        n_jobs=args.n_jobs,
    )
    p_rsf_oof = _oof_pred_rsf_raw(
        df_tr, numeric_cols, cat_cols,
        horizon=args.horizon,
        scale=args.scale,
        age_impute=args.age_impute,
        n_folds=args.oof_folds,
        seed=args.seed + 777,
        rsf_params=rsf_params,
    )

    # 5) Select fusion weight ONLY using OOF on outer-train
    print("[FUSION] Selecting weight w by minimizing OOF IPCW-Brier ...")
    base_brier = _brier_ipcw(ytr, ytr, p_gbsa_oof, args.horizon)
    base_auc = _auc_dynamic(ytr, ytr, risk=-p_gbsa_oof, horizon=args.horizon)

    best = None
    for w in np.linspace(0.0, 1.0, int(1.0 / args.w_step) + 1):
        p_fused = w * p_gbsa_oof + (1.0 - w) * p_rsf_oof
        b = _brier_ipcw(ytr, ytr, p_fused, args.horizon)
        a = _auc_dynamic(ytr, ytr, risk=-p_fused, horizon=args.horizon)
        cand = (b, -a, w)  # minimize b, then maximize a
        if best is None or cand < best:
            best = cand

    best_brier, neg_best_auc, w_star = best
    best_auc = -neg_best_auc

    # Safety fallback: require meaningful OOF Brier improvement
    if (base_brier - best_brier) < args.fusion_eps:
        w_star = 1.0
        best_brier = base_brier
        best_auc = base_auc
        print(f"[FUSION] No OOF improvement >= {args.fusion_eps:.4g}. Fallback to w=1.0 (pure GBSA).")
    else:
        print(f"[FUSION] Selected w={w_star:.3f} | OOF brier {base_brier:.4f} -> {best_brier:.4f} | "
              f"OOF auc {base_auc:.4f} -> {best_auc:.4f}")

    p_fused_oof = w_star * p_gbsa_oof + (1.0 - w_star) * p_rsf_oof

    # 6) Fit calibrator on OOF fused predictions (leakage-safe calibration)
    calibrator = v3.fit_calibrator(
        args.calibrate,
        df_tr["Survival.time"].to_numpy(),
        df_tr["deadstatus.event"].to_numpy(),
        p_fused_oof,
        args.horizon,
        seed=args.seed,
        platt_C=args.platt_C,
        pmin=args.clip_lo,
        pmax=args.clip_hi,
    )
    p_fused_oof_cal = calibrator.predict(p_fused_oof)

    # 7) Train final models on full outer-train, evaluate once on frozen outer-test
    print("[TRAIN] Fitting final GBSA ensemble on outer-train ...")
    gbsa_models, gbsa_seeds = _train_gbsa_ensemble(Xtr, ytr, gbsa_params, args.seed, args.n_gbsa_models)
    p_gbsa_te_raw, _ = _predict_gbsa_ensemble_raw(gbsa_models, Xte, args.horizon)

    print("[TRAIN] Fitting final RSF on outer-train ...")
    rsf_final = RandomSurvivalForest(random_state=args.seed + 777, **rsf_params)
    rsf_final.fit(Xtr, ytr)
    p_rsf_te_raw = v3.predict_survival_at(rsf_final, Xte, args.horizon)

    p_fused_te_raw = w_star * p_gbsa_te_raw + (1.0 - w_star) * p_rsf_te_raw
    p_fused_te = calibrator.predict(p_fused_te_raw)

    # Use risk proxy for AUC/C-index: lower survival prob => higher risk
    risk_te = -p_fused_te
    risk_tr = -p_fused_oof_cal

    metrics = v3.compute_core_metrics(
        ytr=ytr,
        yte=yte,
        risk_tr=risk_tr,
        risk_te=risk_te,
        p2_te=p_fused_te,
        horizon=args.horizon,
    )
    
    # Compute Brier skill score using KM baseline
    from sksurv.metrics import brier_score
    s0 = v3.km_survival_at(df_tr["Survival.time"].values, df_tr["deadstatus.event"].values, args.horizon)
    _, bs0 = brier_score(ytr, yte, np.c_[np.full(len(df_te), s0)], np.array([args.horizon]))
    metrics["km_surv2y_train"] = float(s0)
    metrics["brier2y_km"] = float(bs0[0])
    metrics["brier2y_skill"] = float(1.0 - metrics["brier2y"] / max(metrics["brier2y_km"], 1e-12))

    ci = v3.bootstrap_ci(
        ytr=ytr,
        yte=yte,
        risk_te=risk_te,
        p2_te=p_fused_te,
        horizon=args.horizon,
        n_boot=int(args.bootstrap),
        seed=args.seed,
    )

    report = {
        "model": "GBSA_ensemble + RSF fusion",
        "horizon": float(args.horizon),
        "fusion_weight_w_gbsa": float(w_star),
        "oof_selection": {
            "metric": "min_ipcw_brier",
            "baseline_gbsa_brier": float(base_brier),
            "baseline_gbsa_auc": float(base_auc),
            "best_brier": float(best_brier),
            "best_auc": float(best_auc),
            "fusion_eps": float(args.fusion_eps),
        },
        "test_metrics": metrics,
        "ci95": ci,
        "gbsa_params": gbsa_params,
        "rsf_params": rsf_params,
        "gbsa_seeds": gbsa_seeds,
    }

    # 8) Save artifacts
    bundle = {
        "kind": "gbsa_rsf_fusion",
        "horizon": float(args.horizon),
        "prep": prep,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
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
        "gbsa_params": gbsa_params,
        "gbsa_models": gbsa_models,
        "rsf_model": rsf_final,
        "fusion_weight_w_gbsa": float(w_star),
        "calibrator": calibrator,
        "clip": (float(args.clip_lo), float(args.clip_hi)),
    }
    joblib.dump(bundle, os.path.join(args.out_dir, "fused_model.joblib"))

    pred_df = pd.DataFrame({
        "PatientID": df_te["PatientID"].astype(str).values,
        "pred_surv": p_fused_te,
        "pred_surv_raw": p_fused_te_raw,
        "event": df_te["deadstatus.event"].astype(int).values,
        "time": df_te["Survival.time"].astype(float).values,
    })
    pred_df.to_csv(os.path.join(args.out_dir, "fused_test_pred.csv"), index=False)

    with open(os.path.join(args.out_dir, "fused_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("============================================================")
    print("[DONE] Outer-test metrics (single evaluation)")
    print(f"  AUC@{args.horizon:.0f}d   : {metrics['auc2y']:.4f}")
    print(f"  Brier@{args.horizon:.0f}d : {metrics['brier2y']:.4f} | skill={metrics['brier2y_skill']:.4f}")
    print(f"  C-index(test) : {metrics['cindex_test']:.4f}")
    print("============================================================")
    print("[OK] Saved:")
    print(" -", os.path.join(args.out_dir, "fused_model.joblib"))
    print(" -", os.path.join(args.out_dir, "fused_test_pred.csv"))
    print(" -", os.path.join(args.out_dir, "fused_report.json"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train_eval")
    p.add_argument("--clinical_csv", required=True)
    p.add_argument("--radiomics_csv", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--selected_radiomics_txt", required=True)
    p.add_argument("--gbsa_params_json", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--seed", type=int, default=91)
    p.add_argument("--horizon", type=float, default=730.0)
    p.add_argument("--scale", default="minmax", choices=["minmax", "zscore", "none"])
    p.add_argument("--age_impute", default="mean", choices=["mean", "median"])

    # clinical feature toggles (match your v3 settings)
    p.add_argument("--use_age", type=int, default=1)
    p.add_argument("--use_gender", type=int, default=1)
    p.add_argument("--use_histology", type=int, default=1)
    p.add_argument("--use_t_stage", type=int, default=1)
    p.add_argument("--use_stage", type=int, default=1)
    p.add_argument("--use_n_stage", type=int, default=0)
    p.add_argument("--use_m_stage", type=int, default=0)

    # GBSA ensemble
    p.add_argument("--n_gbsa_models", type=int, default=50)

    # OOF + calibration
    p.add_argument("--oof_folds", type=int, default=5)
    p.add_argument("--calibrate", default="platt", choices=["none", "platt", "isotonic"])
    p.add_argument("--platt_C", type=float, default=10.0)
    p.add_argument("--clip_lo", type=float, default=0.01)
    p.add_argument("--clip_hi", type=float, default=0.99)

    # RSF params
    p.add_argument("--rsf_n_estimators", type=int, default=500)
    p.add_argument("--rsf_min_samples_leaf", type=int, default=10)
    p.add_argument("--rsf_min_samples_split", type=int, default=20)
    p.add_argument("--rsf_max_features", default="sqrt")  # can also be float like 0.5
    p.add_argument("--n_jobs", type=int, default=-1)

    # fusion selection
    p.add_argument("--w_step", type=float, default=0.02)      # grid step
    p.add_argument("--fusion_eps", type=float, default=0.001) # require OOF brier improvement at least this much

    # reporting
    p.add_argument("--bootstrap", type=int, default=200)

    args = ap.parse_args()
    if args.cmd == "train_eval":
        train_eval(args)


if __name__ == "__main__":
    main()
