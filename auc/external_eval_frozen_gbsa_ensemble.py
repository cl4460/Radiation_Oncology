#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
external_eval_frozen_gbsa_ensemble.py

External validation for a frozen GBSA ensemble model (joblib) on an external cohort.

Design goals:
- NO refit, NO parameter search, NO recalibration on external data
- Reuse EXACT preprocessing + calibration stored in the model bundle
- Report time-dependent AUC and IPCW-Brier at a fixed horizon
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import joblib

from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc, brier_score, concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator

import surv_2y_system_v3 as v3


def _sha16_ids(ids):
    import hashlib
    s = ",".join(sorted(map(str, ids)))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _get_from_bundle(bundle, keys, default=None):
    for k in keys:
        if isinstance(bundle, dict) and k in bundle:
            return bundle[k]
        if hasattr(bundle, k):
            return getattr(bundle, k)
    return default


def _extract_models(bundle):
    """
    Accept a few possible bundle formats:
    - {"models": [model, ...]}
    - {"ensemble_models": [{"model": m, ...}, ...]}
    """
    models = _get_from_bundle(bundle, ["models", "gbsa_models", "ensemble_models"], None)
    if models is None:
        raise KeyError("Cannot find models list in joblib bundle. Keys: %s" %
                       (list(bundle.keys()) if isinstance(bundle, dict) else type(bundle)))

    # normalize
    if len(models) > 0 and isinstance(models[0], dict) and "model" in models[0]:
        models = [d["model"] for d in models]
    return models


def _predict_surv_at(models, X, horizon):
    """
    Average raw survival probability S(horizon) across ensemble members.
    Uses v3.predict_survival_at for consistency.
    """
    preds = []
    for m in models:
        preds.append(v3.predict_survival_at(m, X, horizon))
    return np.mean(np.vstack(preds), axis=0)


def _apply_calibrator(calibrator, p_raw):
    if calibrator is None:
        return p_raw
    # v3 calibrators expose .predict(p)
    if hasattr(calibrator, "predict"):
        try:
            return np.asarray(calibrator.predict(p_raw), dtype=float)
        except TypeError:
            # some sklearn-like calibrators expect 2D
            return np.asarray(calibrator.predict(p_raw.reshape(-1, 1)), dtype=float)
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]
    raise TypeError(f"Unknown calibrator type: {type(calibrator)}")


def _km_survival_at(y, t):
    # y is structured array from Surv
    times, surv = kaplan_meier_estimator(y["event"], y["time"])
    # survival is step function; take last value with time <= t
    idx = np.searchsorted(times, t, side="right") - 1
    if idx < 0:
        return 1.0
    return float(surv[idx])


def eval_external(args):
    os.makedirs(args.out_dir, exist_ok=True)

    bundle = joblib.load(args.model_joblib)
    horizon = float(args.horizon) if args.horizon is not None else float(_get_from_bundle(bundle, ["horizon", "horizon_days"], 730.0))

    selected_radiomics = _get_from_bundle(bundle, ["selected_radiomics"], None)
    if selected_radiomics is None:
        # fallback: use provided txt
        if args.selected_radiomics_txt is None:
            raise ValueError("Bundle has no selected_radiomics; please provide --selected_radiomics_txt")
        selected_radiomics = v3.read_txt_list(args.selected_radiomics_txt)

    use_n_stage = bool(_get_from_bundle(bundle, ["use_n_stage"], args.use_n_stage))
    use_m_stage = bool(_get_from_bundle(bundle, ["use_m_stage"], args.use_m_stage))

    models = _extract_models(bundle)

    prep = _get_from_bundle(bundle, ["prep", "preprocess", "preprocessor"], None)
    if prep is None:
        raise KeyError("Cannot find preprocess bundle in joblib. Expected key: prep/preprocess/preprocessor")

    calibrator = _get_from_bundle(bundle, ["calibrator", "cal", "calib"], None)

    # ---- load external data ----
    # Normalize column names (handle patient_id vs PatientID, histology vs Histology, etc.)
    clin_ext = pd.read_csv(args.external_clinical_csv)
    rad_ext = pd.read_csv(args.external_radiomics_csv)
    
    # Standardize column names to match load_and_merge expectations
    # PatientID
    if "patient_id" in clin_ext.columns and "PatientID" not in clin_ext.columns:
        clin_ext.rename(columns={"patient_id": "PatientID"}, inplace=True)
    if "patient_id" in rad_ext.columns and "PatientID" not in rad_ext.columns:
        rad_ext.rename(columns={"patient_id": "PatientID"}, inplace=True)
    
    # Clinical columns (capitalize first letter to match LUNG1 format)
    rename_map = {
        "histology": "Histology",
        "overall_stage": "Overall.Stage",
        "time_days": "Survival.time",
        "event": "deadstatus.event",
    }
    for old, new in rename_map.items():
        if old in clin_ext.columns and new not in clin_ext.columns:
            clin_ext.rename(columns={old: new}, inplace=True)
    
    # T/N/M stages: map to expected format
    if "T" in clin_ext.columns and "clinical.T.Stage" not in clin_ext.columns:
        clin_ext.rename(columns={"T": "clinical.T.Stage"}, inplace=True)
    if "N" in clin_ext.columns and "Clinical.N.Stage" not in clin_ext.columns:
        clin_ext.rename(columns={"N": "Clinical.N.Stage"}, inplace=True)
    if "M" in clin_ext.columns and "Clinical.M.Stage" not in clin_ext.columns:
        clin_ext.rename(columns={"M": "Clinical.M.Stage"}, inplace=True)
    
    # Save temporarily with standardized names (load_and_merge expects file paths)
    import tempfile
    temp_clin = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    clin_ext.to_csv(temp_clin.name, index=False)
    temp_clin.close()
    
    temp_rad = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    rad_ext.to_csv(temp_rad.name, index=False)
    temp_rad.close()
    
    try:
        df_ext = v3.load_and_merge(
            temp_clin.name,
            temp_rad.name,
            selected_radiomics,
            use_n_stage=use_n_stage,
            use_m_stage=use_m_stage
        )
    finally:
        # Clean up temp files
        if os.path.exists(temp_clin.name):
            os.remove(temp_clin.name)
        if os.path.exists(temp_rad.name):
            os.remove(temp_rad.name)

    # basic checks
    df_ext = df_ext.copy()
    df_ext["PatientID"] = df_ext["PatientID"].astype(str).str.strip()
    df_ext = df_ext.drop_duplicates(subset=["PatientID"])

    # transform
    Xext = prep.transform(df_ext)

    # outcomes
    t = df_ext[args.time_col].astype(float).values
    e = df_ext[args.event_col].astype(int).values.astype(bool)
    y_ext = Surv.from_arrays(event=e, time=t)

    # predictions
    p_raw = _predict_surv_at(models, Xext, horizon=horizon)
    p_cal = _apply_calibrator(calibrator, p_raw)

    # risk for c-index / dynamic auc (higher risk = worse)
    risk = -p_cal

    # ---- metrics (IPCW based) ----
    times = np.asarray([horizon], dtype=float)
    aucs, mean_auc = cumulative_dynamic_auc(y_ext, y_ext, risk, times)
    auc_t = float(aucs[0])

    _, brier = brier_score(y_ext, y_ext, p_cal.reshape(-1, 1), times)
    brier_t = float(brier[0])

    # brier skill vs null (KM survival at horizon)
    s0 = _km_survival_at(y_ext, horizon)
    p_null = np.full(len(df_ext), s0, dtype=float)
    _, b0 = brier_score(y_ext, y_ext, p_null.reshape(-1, 1), times)
    b0 = float(b0[0])
    brier_skill = float(1.0 - (brier_t / b0)) if b0 > 0 else float("nan")

    # c-index
    cidx = concordance_index_censored(e, t, risk)[0]
    cidx = float(cidx)

    # known/unknown at horizon (important interpretability)
    known = (t >= horizon) | ((t < horizon) & e)
    n_known = int(known.sum())
    n_unknown = int((~known).sum())
    deaths_before = int(((t < horizon) & e).sum())

    report = {
        "external_n": int(len(df_ext)),
        "horizon_days": float(horizon),
        "known_at_horizon": n_known,
        "unknown_censored_before_horizon": n_unknown,
        "deaths_before_horizon": deaths_before,
        "auc_dynamic": auc_t,
        "brier_ipcw": brier_t,
        "brier_null_ipcw": b0,
        "brier_skill": brier_skill,
        "cindex": cidx,
        "test_ids_sha16": _sha16_ids(df_ext["PatientID"].tolist()),
        "notes": "Metrics computed WITHOUT refit/recalibration on external data. IPCW weights estimated on external cohort."
    }

    # optional bootstrap CIs on external
    if args.bootstrap and args.bootstrap > 0:
        rng = np.random.default_rng(args.seed)
        auc_bs, brier_bs, cidx_bs = [], [], []
        n = len(df_ext)
        for _ in range(int(args.bootstrap)):
            idx = rng.integers(0, n, size=n)
            yb = y_ext[idx]
            pb = p_cal[idx]
            rb = risk[idx]
            try:
                aucs_b, _ = cumulative_dynamic_auc(yb, yb, rb, times)
                _, b_b = brier_score(yb, yb, pb.reshape(-1, 1), times)
                c_b = concordance_index_censored(yb["event"], yb["time"], rb)[0]
                auc_bs.append(float(aucs_b[0]))
                brier_bs.append(float(b_b[0]))
                cidx_bs.append(float(c_b))
            except Exception:
                continue

        def ci95(arr):
            if len(arr) < 20:
                return None
            lo, hi = np.percentile(arr, [2.5, 97.5])
            return [float(lo), float(hi), int(len(arr))]

        report["ci95_bootstrap"] = {
            "auc": ci95(auc_bs),
            "brier": ci95(brier_bs),
            "cindex": ci95(cidx_bs),
        }

    # save
    out_pred = pd.DataFrame({
        "PatientID": df_ext["PatientID"].values,
        args.time_col: t,
        args.event_col: e.astype(int),
        f"S_raw_{int(horizon)}d": p_raw,
        f"S_cal_{int(horizon)}d": p_cal,
        "risk": risk,
    })
    out_pred.to_csv(os.path.join(args.out_dir, "external_pred.csv"), index=False)

    with open(os.path.join(args.out_dir, "external_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("============================================================")
    print("[EXTERNAL] Done")
    print(f"n={report['external_n']} | horizon={report['horizon_days']}d")
    print(f"known@horizon: {report['known_at_horizon']} | unknown(censored<h): {report['unknown_censored_before_horizon']}")
    print(f"AUC@horizon  : {report['auc_dynamic']:.4f}")
    print(f"Brier@horizon: {report['brier_ipcw']:.4f} | skill={report['brier_skill']:.4f}")
    print(f"C-index      : {report['cindex']:.4f}")
    if "ci95_bootstrap" in report:
        print("CI95 bootstrap:", report["ci95_bootstrap"])
    print("[OK] Saved:", args.out_dir)
    print("============================================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_joblib", required=True)
    ap.add_argument("--external_clinical_csv", required=True)
    ap.add_argument("--external_radiomics_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--time_col", default="Survival.time")
    ap.add_argument("--event_col", default="deadstatus.event")

    ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--selected_radiomics_txt", default=None)  # only if bundle lacks it
    ap.add_argument("--use_n_stage", type=int, default=0)
    ap.add_argument("--use_m_stage", type=int, default=0)

    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)

    args = ap.parse_args()
    eval_external(args)


if __name__ == "__main__":
    main()
