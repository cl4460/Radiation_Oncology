#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
surv_horizon_gbsa_ensemble.py

GBSA 生存模型的 Seed-Averaging Ensemble（严格不碰 test 调参）：
- 使用固定 outer split (split_json) 的 train/test
- 在 outer-train 上训练 N 个 GBSA（超参相同、seed 不同）
- 用 outer-train 的 OOF 预测（每个 seed 自己的 KFold）得到 ensemble 的 OOF S(t)raw
- 在 OOF 上拟合一个 Platt 校准器（logit(p_raw) -> P(T>t | x)）
- 在 outer-test 上：raw 先做 ensemble 平均，再做一次校准，然后计算 AUC(t) / Brier(t) / C-index
- 输出 joblib 模型包 + test 预测 CSV + JSON 报告（含 bootstrap CI）

依赖：
- 复用你已有的 surv_2y_system_v3.py（同目录即可 import）
- sklearn / scikit-survival / pandas / numpy / joblib
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib

# 需要与该脚本放在同一目录，或在 PYTHONPATH 中
import surv_2y_system_v3 as v3


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _merge_params(base_params: Dict[str, Any], override_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    params = dict(base_params)
    if override_params:
        params.update(override_params)
    return params


def _stack_mean(list_of_1d: List[np.ndarray]) -> np.ndarray:
    """Mean over a list of 1D arrays with the same length."""
    return np.mean(np.stack(list_of_1d, axis=0), axis=0)


def _known_status_mask(time: np.ndarray, event: np.ndarray, horizon: float) -> np.ndarray:
    """
    known@horizon 的定义：
      - 若 time >= horizon：在 horizon 时刻仍有随访（无论之后是否发生事件），其 0/1 标签可判定为“存活到 horizon”
      - 若 event==1 且 time < horizon：在 horizon 之前发生死亡事件，其标签可判定为“未存活到 horizon”
      - 若 event==0 且 time < horizon：在 horizon 前删失，标签未知 -> 排除
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    return (time >= horizon) | (event & (time < horizon))


def train_eval(args: argparse.Namespace) -> None:
    # -------------------------
    # 0) 准备：读取配置、数据
    # -------------------------
    selected_radiomics = v3.read_txt_list(args.selected_radiomics_txt)
    split = v3.read_split_json(args.split_json)

    df = v3.load_and_merge(
        args.clinical_csv,
        args.radiomics_csv,
        selected_radiomics,
        use_n_stage=bool(args.use_n_stage),
        use_m_stage=bool(args.use_m_stage),
    )
    df_tr, df_te = v3.make_train_test(df, split)

    numeric_cols, cat_cols = v3.build_feature_lists(
        df,
        selected_radiomics,
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        use_histology=bool(args.use_histology),
        use_t_stage=bool(args.use_t_stage),
        use_stage=bool(args.use_stage),
        use_n_stage=bool(args.use_n_stage),
        use_m_stage=bool(args.use_m_stage),
    )

    # 仅在 outer-train 上 fit 预处理器（不碰 test）
    prep = v3.fit_preprocess_bundle(df_tr, numeric_cols, cat_cols, args.age_impute, args.scale)
    Xtr = prep.transform(df_tr)
    Xte = prep.transform(df_te)

    ytr = v3.y_from_df(df_tr)
    yte = v3.y_from_df(df_te)

    base_params = _load_json(args.gbsa_params_json)
    override_params = _load_json(args.gbsa_params_override_json) if args.gbsa_params_override_json else None
    params = _merge_params(base_params, override_params)

    # -------------------------
    # 1) 训练 N 个 GBSA（full train）
    # -------------------------
    seeds = [args.seed + i for i in range(args.n_models)]
    models = []
    risk_tr_list = []
    risk_te_list = []
    p_te_raw_list = []

    print("============================================================")
    print(f"[ENSEMBLE] Training {args.n_models} GBSA models on outer-train (n={len(df_tr)})")
    print(f"Seeds: {seeds[0]}..{seeds[-1]}")
    print(f"Horizon: {args.horizon} days | Calibrate: {args.calibrate} (OOF folds={args.calibrate_oof_folds}, C={args.platt_C})")
    print("============================================================")

    for i, s in enumerate(seeds, 1):
        m = v3.fit_gbsa(Xtr, ytr, params=params, seed=int(s))
        models.append(m)
        risk_tr_list.append(m.predict(Xtr))
        risk_te_list.append(m.predict(Xte))
        p_te_raw_list.append(v3.predict_survival_at(m, Xte, args.horizon))
        if (i % max(1, args.n_models // 5) == 0) or (i == args.n_models):
            print(f"  done {i:>2}/{args.n_models} (seed={s})")

    risk_tr_ens = _stack_mean(risk_tr_list)
    risk_te_ens = _stack_mean(risk_te_list)
    p_te_raw_ens = _stack_mean(p_te_raw_list)

    # -------------------------
    # 2) OOF 预测用于校准（严格在 outer-train 内完成）
    #    注意：这里每个 seed 都做一次 OOF，然后再平均 => ensemble 的 OOF
    # -------------------------
    p_oof_raw_list = []
    print("============================================================")
    print(f"[CAL] Computing OOF raw S(t) on outer-train for calibration ({args.n_models} seeds × {args.calibrate_oof_folds} folds)")
    print("============================================================")
    for i, s in enumerate(seeds, 1):
        p_oof = v3.oof_pred_p2_raw(
            df_tr,
            numeric_cols,
            cat_cols,
            params,
            args.horizon,
            args.scale,
            args.age_impute,
            n_folds=int(args.calibrate_oof_folds),
            seed=int(s),
        )
        p_oof_raw_list.append(np.asarray(p_oof, dtype=float))
        if (i % max(1, args.n_models // 5) == 0) or (i == args.n_models):
            print(f"  OOF done {i:>2}/{args.n_models} (seed={s})")

    p_oof_raw_ens = _stack_mean(p_oof_raw_list)

    calibrator = v3.fit_calibrator(
        args.calibrate,
        df_tr["Survival.time"].to_numpy(),
        df_tr["deadstatus.event"].to_numpy(),
        p_oof_raw_ens,
        args.horizon,
        args.seed,
        args.platt_C,
        args.clip_min,
        args.clip_max,
    )

    p_te_cal = calibrator.predict(p_te_raw_ens)

    # -------------------------
    # 3) 指标（outer-test 一次性评估）
    # -------------------------
    metrics = v3.compute_core_metrics(ytr, yte, risk_tr_ens, risk_te_ens, p_te_cal, args.horizon)
    
    # Compute Brier skill score using KM baseline
    from sksurv.metrics import brier_score
    s0 = v3.km_survival_at(df_tr["Survival.time"].values, df_tr["deadstatus.event"].values, args.horizon)
    _, bs0 = brier_score(ytr, yte, np.c_[np.full(len(df_te), s0)], np.array([args.horizon]))
    metrics["km_surv2y_train"] = float(s0)
    metrics["brier2y_km"] = float(bs0[0])
    metrics["brier2y_skill"] = float(1.0 - metrics["brier2y"] / max(metrics["brier2y_km"], 1e-12))
    
    if int(args.bootstrap) > 0:
        ci = v3.bootstrap_ci(ytr, yte, risk_te_ens, p_te_cal, args.horizon, n_boot=int(args.bootstrap), seed=args.seed)
    else:
        ci = {}

    # known-status 统计（解释为什么不同 horizon AUC 会差异很大）
    te_time = df_te["Survival.time"].to_numpy(dtype=float)
    te_event = df_te["deadstatus.event"].to_numpy(dtype=int)
    known = _known_status_mask(te_time, te_event, args.horizon)
    n_known = int(known.sum())
    n_early_death = int(((te_event == 1) & (te_time < args.horizon)).sum())

    report: Dict[str, Any] = dict(metrics)
    report.update(
        {
            "horizon_days": float(args.horizon),
            "n_models": int(args.n_models),
            "seeds": seeds,
            "outer_train_n": int(len(df_tr)),
            "outer_test_n": int(len(df_te)),
            "outer_test_known_n": n_known,
            "outer_test_death_before_horizon_n": n_early_death,
            "params_base_json": args.gbsa_params_json,
            "params_override_json": args.gbsa_params_override_json,
            "calibrate": args.calibrate,
            "calibrate_oof_folds": int(args.calibrate_oof_folds),
            "platt_C": float(args.platt_C),
            "clip": [float(args.clip_min), float(args.clip_max)],
            "bootstrap_ci": ci,
        }
    )

    # -------------------------
    # 4) 保存
    # -------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    bundle: Dict[str, Any] = {
        "version": "gbsa_ensemble_v1",
        "models": models,
        "prep": prep,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "selected_radiomics": selected_radiomics,
        "clinical_flags": {
            "use_age": bool(args.use_age),
            "use_gender": bool(args.use_gender),
            "use_histology": bool(args.use_histology),
            "use_t_stage": bool(args.use_t_stage),
            "use_stage": bool(args.use_stage),
            "use_n_stage": bool(args.use_n_stage),
            "use_m_stage": bool(args.use_m_stage),
        },
        "age_impute": args.age_impute,
        "scale": args.scale,
        "horizon": float(args.horizon),
        "calibrate": args.calibrate,
        "calibrate_oof_folds": int(args.calibrate_oof_folds),
        "platt_C": float(args.platt_C),
        "pclip_min": float(args.clip_min),
        "pclip_max": float(args.clip_max),
        "calibrator": calibrator,
        "train_risk_mean": float(np.mean(risk_tr_ens)),
        "train_risk_std": float(np.std(risk_tr_ens)),
        "report": report,
    }

    model_path = os.path.join(args.out_dir, "gbsa_horizon_ensemble.joblib")
    joblib.dump(bundle, model_path)

    pred = df_te[["PatientID", "Survival.time", "deadstatus.event"]].copy()
    pred["risk_ens"] = risk_te_ens
    pred["surv_prob_raw_ens"] = p_te_raw_ens
    pred["surv_prob_cal_ens"] = p_te_cal
    pred_path = os.path.join(args.out_dir, "ensemble_test_pred.csv")
    pred.to_csv(pred_path, index=False)

    report_path = os.path.join(args.out_dir, "ensemble_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("============================================================")
    print("[DONE] Outer-test metrics (single evaluation)")
    print(f"  C-index (test) : {metrics['cindex_test']:.4f}")
    print(f"  AUC@{args.horizon:.0f}d     : {metrics['auc2y']:.4f}")
    print(f"  Brier@{args.horizon:.0f}d   : {metrics['brier2y']:.4f} | skill={metrics['brier2y_skill']:.4f}")
    print(f"  known@{args.horizon:.0f}d on test: {n_known}/{len(df_te)} | deaths before horizon: {n_early_death}")
    if ci:
        print(f"  CI95 (AUC)   : {ci['auc2y']['ci95_lo']:.4f}-{ci['auc2y']['ci95_hi']:.4f}")
        print(f"  CI95 (Brier) : {ci['brier2y']['ci95_lo']:.4f}-{ci['brier2y']['ci95_hi']:.4f}")
        print(f"  CI95 (Cidx)  : {ci['cindex_test']['ci95_lo']:.4f}-{ci['cindex_test']['ci95_hi']:.4f}")
    print("============================================================")
    print(f"[OK] Saved model : {model_path}")
    print(f"[OK] Saved pred  : {pred_path}")
    print(f"[OK] Saved report: {report_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="surv_horizon_gbsa_ensemble.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train_eval", help="Train ensemble on outer-train and evaluate once on frozen outer-test")
    t.add_argument("--clinical_csv", required=True)
    t.add_argument("--radiomics_csv", required=True)
    t.add_argument("--split_json", required=True)
    t.add_argument("--selected_radiomics_txt", required=True)

    t.add_argument("--gbsa_params_json", required=True, help="Base GBSA params JSON (e.g., gbsa91_regularized_params.json)")
    t.add_argument("--gbsa_params_override_json", default=None, help="Optional override JSON (e.g., best_params.json)")

    t.add_argument("--out_dir", required=True)

    t.add_argument("--seed", type=int, default=91)
    t.add_argument("--n_models", type=int, default=20)

    t.add_argument("--horizon", type=float, default=730.0)
    t.add_argument("--scale", default="minmax", choices=["none", "standard", "minmax"])
    t.add_argument("--age_impute", default="mean", choices=["mean", "median"])

    t.add_argument("--use_age", type=int, default=1)
    t.add_argument("--use_gender", type=int, default=1)
    t.add_argument("--use_histology", type=int, default=1)
    t.add_argument("--use_t_stage", type=int, default=1)
    t.add_argument("--use_stage", type=int, default=1)
    t.add_argument("--use_n_stage", type=int, default=0)
    t.add_argument("--use_m_stage", type=int, default=0)

    t.add_argument("--calibrate", default="platt", choices=["none", "platt"])
    t.add_argument("--calibrate_oof_folds", type=int, default=5)
    t.add_argument("--platt_C", type=float, default=10.0)
    t.add_argument("--clip_min", type=float, default=0.01)
    t.add_argument("--clip_max", type=float, default=0.99)

    t.add_argument("--bootstrap", type=int, default=200)

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.cmd == "train_eval":
        train_eval(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()

#run command:
#python surv_horizon_gbsa_ensemble.py train_eval   
#--clinical_csv /home/lichengze/Research/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
#--radiomics_csv /home/lichengze/Research/com/Tb_features_1688_from_yaml.csv \
#--split_json /home/lichengze/Research/com/out_zip_audit_splits/split_ClinicplusRadiomics_6cf7bc2c.json \
#--selected_radiomics_txt /home/lichengze/Research/com/gbsa91_selected_radiomics.txt \
#--gbsa_params_json /home/lichengze/Research/auc/gbsa91_regularized_params.json  
#--out_dir /home/lichengze/Research/auc/gbsa_2y_ensemble_regularized \
#--seed 91 \
#--n_models 20 \
#--horizon 730 \
#--scale minmax \
#--calibrate platt \
#--calibrate_oof_folds 5 \
#--platt_C 10 \
#--bootstrap 200