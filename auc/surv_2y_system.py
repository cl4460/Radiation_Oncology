#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.isotonic import IsotonicRegression

# scikit-survival
from sksurv.util import Surv
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator


# -----------------------------
# Utilities
# -----------------------------
def read_split_json(split_json: str) -> Tuple[List[str], List[str]]:
    """Expect split_json like your exported one: {"train_ids":[...], "test_ids":[...]} or similar."""
    with open(split_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # Be permissive:
    for k_tr, k_te in [
        ("train_ids", "test_ids"),
        ("train", "test"),
        ("train_id", "test_id"),
        ("train_patients", "test_patients"),
    ]:
        if k_tr in obj and k_te in obj:
            return list(map(str, obj[k_tr])), list(map(str, obj[k_te]))
    raise ValueError(f"Unrecognized split_json keys in {split_json}. Keys={list(obj.keys())}")


def safe_mode(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return None
    return series.value_counts().index[0]


def km_survival_at_t(time: np.ndarray, event: np.ndarray, t: float) -> float:
    """Kaplan-Meier survival estimate at time t (supports censoring)."""
    # event: True = event occurred
    x, y = kaplan_meier_estimator(event.astype(bool), time.astype(float))
    # x is time grid, y is survival probs
    if t <= x[0]:
        return float(y[0])
    if t >= x[-1]:
        return float(y[-1])
    # find rightmost x <= t
    idx = np.searchsorted(x, t, side="right") - 1
    return float(y[idx])


def build_y(df: pd.DataFrame, time_col: str, event_col: str):
    t = df[time_col].astype(float).to_numpy()
    e = df[event_col].astype(int).to_numpy().astype(bool)
    return Surv.from_arrays(event=e, time=t)


def infer_time_unit_hint(df: pd.DataFrame, time_col: str) -> str:
    """Just a heuristic hint for user; not used in training."""
    t = df[time_col].astype(float)
    p50 = float(t.median())
    # If median > 100, likely days; if ~10-60, could be months.
    if p50 > 100:
        return f"median({time_col})≈{p50:.1f} (looks like DAYS). 2y=730."
    return f"median({time_col})≈{p50:.1f} (could be MONTHS). If months, 2y=24."


def load_merged(
    clinical_csv: str,
    radiomics_csv: str,
    id_col: str,
) -> pd.DataFrame:
    clin = pd.read_csv(clinical_csv)
    rad = pd.read_csv(radiomics_csv)
    clin[id_col] = clin[id_col].astype(str)
    rad[id_col] = rad[id_col].astype(str)
    df = clin.merge(rad, on=id_col, how="inner")
    return df


def select_features(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    event_col: str,
    clinical_cols: List[str],
    selected_radiomics: List[str],
) -> pd.DataFrame:
    keep = [id_col, time_col, event_col] + clinical_cols + selected_radiomics
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in merged dataframe: {missing[:20]} (total {len(missing)})")
    return df[keep].copy()


def read_feature_list(txt_path: str) -> List[str]:
    feats = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            feats.append(s)
    return feats


@dataclass
class ModelBundle:
    id_col: str
    time_col: str
    event_col: str
    horizon: float
    clinical_cols: List[str]
    radiomics_cols: List[str]
    preprocessor: ColumnTransformer
    model: GradientBoostingSurvivalAnalysis
    calibrator: Optional[IsotonicRegression]
    ref_values: Dict[str, object]  # medians/modes on raw feature space
    train_risk: np.ndarray         # risk scores distribution on train (for percentile)


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scale: str,
    age_impute: str,
) -> ColumnTransformer:
    if scale not in ("minmax", "zscore", "none"):
        raise ValueError("scale must be one of: minmax, zscore, none")

    scaler = None
    if scale == "minmax":
        scaler = MinMaxScaler()
    elif scale == "zscore":
        scaler = StandardScaler()
    else:
        scaler = "passthrough"

    # NOTE:
    # - numeric: impute median by default, but allow age mean if you insist
    # - categorical: most_frequent + onehot(handle_unknown='ignore')
    # For strict reproducibility, keep this fixed once you publish.
    num_imputer = SimpleImputer(strategy="median")
    # Optional: make age use mean (as you ran before)
    if age_impute == "mean":
        # We'll still use median for other numeric; simplest is keep median for all
        # If you *must* do age-only mean, do it outside and pass fully filled age.
        pass

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", num_imputer),
            ("scaler", scaler),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def fit_calibrator_2y(
    df_train_raw: pd.DataFrame,
    p_surv2y_train: np.ndarray,
    time_col: str,
    event_col: str,
    horizon: float,
) -> Optional[IsotonicRegression]:
    """
    Simple practical calibration for S(2y):
    - Keep samples whose status at 2y is known:
        * event happened before 2y  -> label survive2y = 0
        * time >= 2y (regardless event) -> label survive2y = 1 if no event before 2y
      (censored before 2y is dropped)
    - Fit isotonic regression mapping p_raw -> observed survive2y
    """
    t = df_train_raw[time_col].astype(float).to_numpy()
    e = df_train_raw[event_col].astype(int).to_numpy().astype(bool)

    known = (t >= horizon) | (e & (t < horizon))
    if known.sum() < 30:
        # Too few to calibrate reliably; skip
        return None

    y2 = np.zeros_like(t, dtype=int)
    # survive at 2y means no event before 2y and follow-up >=2y
    y2[(t >= horizon) & (~(e & (t < horizon)))] = 1

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p_surv2y_train[known], y2[known].astype(float))
    return iso


def predict_survival_at(
    model: GradientBoostingSurvivalAnalysis,
    X: np.ndarray,
    t: float,
) -> np.ndarray:
    surv_fns = model.predict_survival_function(X)
    return np.asarray([float(fn(t)) for fn in surv_fns], dtype=float)


def risk_percentile(train_risk: np.ndarray, r: float) -> float:
    # percentile in [0,1]
    return float((train_risk < r).mean())


# -----------------------------
# Commands
# -----------------------------
def cmd_train(args):
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_merged(args.clinical_csv, args.radiomics_csv, args.id_col)
    print("[INFO]", infer_time_unit_hint(df, args.time_col))
    train_ids, test_ids = read_split_json(args.split_json)

    df_train = df[df[args.id_col].astype(str).isin(train_ids)].copy()
    df_test  = df[df[args.id_col].astype(str).isin(test_ids)].copy()

    # Clinical columns
    clinical_cols = []
    if args.use_age:
        clinical_cols.append(args.age_col)
    if args.use_gender:
        clinical_cols.append(args.gender_col)
    if args.use_histology:
        clinical_cols.append(args.histology_col)
    if args.use_t_stage:
        clinical_cols.append(args.t_stage_col)
    if args.use_stage:
        clinical_cols.append(args.stage_col)

    # Radiomics selected list
    if not args.selected_radiomics_txt:
        raise ValueError("You must provide --selected_radiomics_txt to freeze the final signature (recommended).")
    radiomics_cols = read_feature_list(args.selected_radiomics_txt)

    df_train = select_features(df_train, args.id_col, args.time_col, args.event_col, clinical_cols, radiomics_cols)
    df_test  = select_features(df_test,  args.id_col, args.time_col, args.event_col, clinical_cols, radiomics_cols)

    # Build raw X
    X_train_raw = df_train[clinical_cols + radiomics_cols].copy()
    X_test_raw  = df_test[clinical_cols + radiomics_cols].copy()

    # Define numeric vs categorical
    categorical_cols = []
    numeric_cols = []
    for c in clinical_cols:
        if c in (args.gender_col, args.histology_col, args.t_stage_col, args.stage_col):
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)
    # Radiomics are numeric
    numeric_cols += radiomics_cols

    pre = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale=args.scale,
        age_impute=args.age_impute,
    )

    # Fit preprocess on train
    Xtr = pre.fit_transform(X_train_raw)
    Xte = pre.transform(X_test_raw)

    ytr = build_y(df_train, args.time_col, args.event_col)
    yte = build_y(df_test,  args.time_col, args.event_col)

    # GBSA params
    if args.gbsa_params_json:
        with open(args.gbsa_params_json, "r", encoding="utf-8") as f:
            params = json.load(f)
    else:
        # If you don't pass, we still provide a reasonable default,
        # but for reproducibility you should pass your best_params json.
        params = dict(
            loss="coxph",
            learning_rate=0.5710308095913009,
            n_estimators=100,
            criterion="friedman_mse",
            max_depth=7,
            max_features=0.34046888947419796,
            max_leaf_nodes=12,
            min_impurity_decrease=0.03,
            min_samples_leaf=3,
            min_samples_split=8,
            min_weight_fraction_leaf=0.2,
            subsample=0.2,
            dropout_rate=0.30000000000000004,
            random_state=args.seed,
        )

    # Make sure random_state exists
    if "random_state" not in params:
        params["random_state"] = args.seed

    model = GradientBoostingSurvivalAnalysis(**params)
    model.fit(Xtr, ytr)

    # Internal sanity metrics (still keep for logging)
    risk_tr = model.predict(Xtr)
    risk_te = model.predict(Xte)
    c_tr = concordance_index_censored(ytr["event"], ytr["time"], risk_tr)[0]
    c_te = concordance_index_censored(yte["event"], yte["time"], risk_te)[0]
    print(f"[TRAIN] cindex={c_tr:.4f}  [TEST] cindex={c_te:.4f}")

    # 2-year survival probability
    p2_tr_raw = predict_survival_at(model, Xtr, args.horizon)
    p2_te_raw = predict_survival_at(model, Xte, args.horizon)

    calibrator = None
    if args.calibrate == "isotonic":
        calibrator = fit_calibrator_2y(df_train, p2_tr_raw, args.time_col, args.event_col, args.horizon)
        if calibrator is not None:
            p2_te = calibrator.predict(p2_te_raw)
            p2_tr = calibrator.predict(p2_tr_raw)
            print("[CAL] isotonic fitted on known-status@2y subset")
        else:
            p2_te = p2_te_raw
            p2_tr = p2_tr_raw
            print("[CAL] skipped (too few known-status@2y)")
    else:
        p2_te = p2_te_raw
        p2_tr = p2_tr_raw

    # 2y metrics: AUC(t) and Brier(t)
    times = np.asarray([args.horizon], dtype=float)
    _, aucs = cumulative_dynamic_auc(ytr, yte, risk_te, times)
    # Handle both scalar and array returns (scikit-survival version dependent)
    auc2y = float(aucs[0] if isinstance(aucs, np.ndarray) and aucs.ndim > 0 else aucs)

    # brier_score expects survival probabilities
    _, bs = brier_score(ytr, yte, np.c_[p2_te], times)
    # Handle both scalar and array returns (scikit-survival version dependent)
    brier2y = float(bs[0] if isinstance(bs, np.ndarray) and bs.ndim > 0 else bs)

    # Calibration-by-bins (KM observed at 2y)
    bins = pd.qcut(p2_te, q=min(args.calib_bins, len(p2_te)), duplicates="drop")
    calib_rows = []
    for b in bins.categories:
        idx = (bins == b)  # Already returns numpy array
        obs = km_survival_at_t(
            df_test.loc[idx, args.time_col].to_numpy(),
            df_test.loc[idx, args.event_col].to_numpy(),
            args.horizon,
        )
        pred = float(np.mean(p2_te[idx]))
        calib_rows.append({"bin": str(b), "n": int(idx.sum()), "pred_surv2y": pred, "obs_surv2y_km": obs})

    # Reference values for explanations (from raw train)
    ref_values = {}
    for c in numeric_cols:
        ref_values[c] = float(np.nanmedian(X_train_raw[c].astype(float).to_numpy()))
    for c in categorical_cols:
        ref_values[c] = safe_mode(X_train_raw[c])

    bundle = ModelBundle(
        id_col=args.id_col,
        time_col=args.time_col,
        event_col=args.event_col,
        horizon=float(args.horizon),
        clinical_cols=clinical_cols,
        radiomics_cols=radiomics_cols,
        preprocessor=pre,
        model=model,
        calibrator=calibrator,
        ref_values=ref_values,
        train_risk=risk_tr.astype(float),
    )

    joblib_path = os.path.join(args.out_dir, "gbsa_2y_model.joblib")
    joblib.dump(bundle, joblib_path)

    # Save metrics
    out = {
        "seed": args.seed,
        "horizon": float(args.horizon),
        "internal_test": {
            "cindex": c_te,
            "auc2y": auc2y,
            "brier2y": brier2y,
        },
        "calibration_bins": calib_rows,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "n_raw_features": int(len(clinical_cols) + len(radiomics_cols)),
        "n_final_features_after_encoding": int(Xtr.shape[1]),
        "selected_radiomics_txt": args.selected_radiomics_txt,
        "gbsa_params_json": args.gbsa_params_json,
    }
    with open(os.path.join(args.out_dir, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Save internal test predictions (for later UI/tests)
    pred_df = df_test[[args.id_col, args.time_col, args.event_col]].copy()
    pred_df["risk"] = risk_te
    pred_df["surv2y_raw"] = p2_te_raw
    pred_df["surv2y"] = p2_te
    pred_df.to_csv(os.path.join(args.out_dir, "internal_test_pred_2y.csv"), index=False)
    print("[OK] Saved:", joblib_path)
    print("[OK] Saved:", os.path.join(args.out_dir, "train_report.json"))
    print("[OK] Saved:", os.path.join(args.out_dir, "internal_test_pred_2y.csv"))


def cmd_eval(args):
    bundle: ModelBundle = joblib.load(args.model_joblib)

    df = load_merged(args.clinical_csv, args.radiomics_csv, bundle.id_col)
    train_ids, test_ids = read_split_json(args.split_json)

    df_train = df[df[bundle.id_col].astype(str).isin(train_ids)].copy()
    df_test  = df[df[bundle.id_col].astype(str).isin(test_ids)].copy()

    df_train = select_features(df_train, bundle.id_col, bundle.time_col, bundle.event_col,
                               bundle.clinical_cols, bundle.radiomics_cols)
    df_test  = select_features(df_test,  bundle.id_col, bundle.time_col, bundle.event_col,
                               bundle.clinical_cols, bundle.radiomics_cols)

    Xtr_raw = df_train[bundle.clinical_cols + bundle.radiomics_cols].copy()
    Xte_raw = df_test[bundle.clinical_cols + bundle.radiomics_cols].copy()

    Xtr = bundle.preprocessor.transform(Xtr_raw)
    Xte = bundle.preprocessor.transform(Xte_raw)

    ytr = build_y(df_train, bundle.time_col, bundle.event_col)
    yte = build_y(df_test,  bundle.time_col, bundle.event_col)

    model = bundle.model
    risk_te = model.predict(Xte)
    c_te = concordance_index_censored(yte["event"], yte["time"], risk_te)[0]

    p2_raw = predict_survival_at(model, Xte, bundle.horizon)
    p2 = bundle.calibrator.predict(p2_raw) if bundle.calibrator is not None else p2_raw

    times = np.asarray([bundle.horizon], dtype=float)
    _, aucs = cumulative_dynamic_auc(ytr, yte, risk_te, times)
    # Handle both scalar and array returns (scikit-survival version dependent)
    auc2y = float(aucs[0] if isinstance(aucs, np.ndarray) and aucs.ndim > 0 else aucs)
    _, bs = brier_score(ytr, yte, np.c_[p2], times)
    # Handle both scalar and array returns (scikit-survival version dependent)
    brier2y = float(bs[0] if isinstance(bs, np.ndarray) and bs.ndim > 0 else bs)

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "horizon": float(bundle.horizon),
        "internal_test": {
            "cindex": float(c_te),
            "auc2y": float(auc2y),
            "brier2y": float(brier2y),
        },
        "n_test": int(len(df_test)),
    }
    with open(os.path.join(args.out_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    pred_df = df_test[[bundle.id_col, bundle.time_col, bundle.event_col]].copy()
    pred_df["risk"] = risk_te
    pred_df["surv2y_raw"] = p2_raw
    pred_df["surv2y"] = p2
    pred_df.to_csv(os.path.join(args.out_dir, "eval_pred_2y.csv"), index=False)

    print("[EVAL] cindex={:.4f} auc2y={:.4f} brier2y={:.4f}".format(c_te, auc2y, brier2y))
    print("[OK] Saved:", os.path.join(args.out_dir, "eval_report.json"))
    print("[OK] Saved:", os.path.join(args.out_dir, "eval_pred_2y.csv"))


def cmd_predict(args):
    bundle: ModelBundle = joblib.load(args.model_joblib)

    # Patient clinical: accept either csv (one row) or json
    if args.patient_clinical_json:
        with open(args.patient_clinical_json, "r", encoding="utf-8") as f:
            clin_obj = json.load(f)
        clin_row = {k: clin_obj.get(k, None) for k in bundle.clinical_cols}
        pid = str(clin_obj.get(bundle.id_col, args.patient_id))
    else:
        # CSV with at least id + clinical cols
        dfc = pd.read_csv(args.patient_clinical_csv)
        if args.patient_id:
            row = dfc[dfc[bundle.id_col].astype(str) == str(args.patient_id)].iloc[0]
        else:
            row = dfc.iloc[0]
        clin_row = {k: row.get(k, None) for k in bundle.clinical_cols}
        pid = str(row.get(bundle.id_col, args.patient_id))

    # Patient radiomics: CSV with id_col + 1688 cols; we only pick selected radiomics
    dfr = pd.read_csv(args.patient_radiomics_csv)
    if args.patient_id:
        rr = dfr[dfr[bundle.id_col].astype(str) == str(args.patient_id)].iloc[0]
    else:
        rr = dfr[dfr[bundle.id_col].astype(str) == str(pid)].iloc[0] if (bundle.id_col in dfr.columns) else dfr.iloc[0]
    rad_row = {k: rr.get(k, None) for k in bundle.radiomics_cols}

    # Build one-row raw df
    x_raw = {**clin_row, **rad_row}
    X_raw = pd.DataFrame([x_raw], columns=bundle.clinical_cols + bundle.radiomics_cols)

    X = bundle.preprocessor.transform(X_raw)
    model = bundle.model

    risk = float(model.predict(X)[0])
    p2_raw = float(predict_survival_at(model, X, bundle.horizon)[0])
    p2 = float(bundle.calibrator.predict([p2_raw])[0]) if bundle.calibrator is not None else p2_raw
    pct = risk_percentile(bundle.train_risk, risk)

    # Local explanation: 1-feature-at-a-time counterfactual to reference values
    rows = []
    for feat, ref in bundle.ref_values.items():
        X_cf_raw = X_raw.copy()
        X_cf_raw.loc[0, feat] = ref
        X_cf = bundle.preprocessor.transform(X_cf_raw)
        p2_cf_raw = float(predict_survival_at(model, X_cf, bundle.horizon)[0])
        p2_cf = float(bundle.calibrator.predict([p2_cf_raw])[0]) if bundle.calibrator is not None else p2_cf_raw
        delta = p2_cf - p2  # positive means "setting to ref increases survival"
        rows.append({
            "feature": feat,
            "patient_value": None if pd.isna(X_raw.loc[0, feat]) else X_raw.loc[0, feat],
            "ref_value": ref,
            "delta_surv2y_if_set_to_ref": float(delta),
        })
    expl = pd.DataFrame(rows)
    expl["abs_delta"] = expl["delta_surv2y_if_set_to_ref"].abs()
    expl = expl.sort_values("abs_delta", ascending=False).drop(columns=["abs_delta"]).head(args.explain_topk)

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = {
        "patient_id": pid,
        "horizon": float(bundle.horizon),
        "pred_survival_2y": p2,
        "pred_survival_2y_raw": p2_raw,
        "pred_risk": risk,
        "risk_percentile_vs_train": pct,
    }
    with open(os.path.join(args.out_dir, f"{pid}_2y_prediction.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    expl.to_csv(os.path.join(args.out_dir, f"{pid}_explanation_top{args.explain_topk}.csv"), index=False)

    print("============================================================")
    print(f"Patient: {pid}")
    print(f"2-year survival probability S(2y): {p2:.3f}  ({p2*100:.1f}%)")
    print(f"Risk score: {risk:.4f}  (percentile vs train: {pct*100:.1f}%)")
    print("Top explanation (delta = S2y_if_ref - S2y):")
    print(expl.to_string(index=False))
    print("============================================================")
    print("[OK] Saved:", os.path.join(args.out_dir, f"{pid}_2y_prediction.json"))
    print("[OK] Saved:", os.path.join(args.out_dir, f"{pid}_explanation_top{args.explain_topk}.csv"))


def build_parser():
    p = argparse.ArgumentParser("surv_2y_system.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    tr = sub.add_parser("train")
    tr.add_argument("--clinical_csv", required=True)
    tr.add_argument("--radiomics_csv", required=True)
    tr.add_argument("--split_json", required=True)
    tr.add_argument("--out_dir", required=True)
    tr.add_argument("--seed", type=int, default=91)

    tr.add_argument("--id_col", default="PatientID")
    tr.add_argument("--time_col", default="Survival.time")
    tr.add_argument("--event_col", default="deadstatus.event")

    tr.add_argument("--age_col", default="age")
    tr.add_argument("--gender_col", default="gender")
    tr.add_argument("--histology_col", default="Histology")
    tr.add_argument("--t_stage_col", default="clinical.T.Stage")
    tr.add_argument("--stage_col", default="Overall.Stage")

    tr.add_argument("--use_age", type=int, default=1)
    tr.add_argument("--use_gender", type=int, default=1)
    tr.add_argument("--use_histology", type=int, default=1)
    tr.add_argument("--use_t_stage", type=int, default=1)
    tr.add_argument("--use_stage", type=int, default=1)

    tr.add_argument("--selected_radiomics_txt", required=True,
                    help="TXT list of selected radiomics features (freeze signature).")
    tr.add_argument("--gbsa_params_json", default=None,
                    help="Best params json (recommended).")

    tr.add_argument("--scale", choices=["minmax", "zscore", "none"], default="minmax")
    tr.add_argument("--age_impute", choices=["mean", "median"], default="mean")

    tr.add_argument("--horizon", type=float, default=730.0, help="2-year horizon in SAME unit as time_col.")
    tr.add_argument("--calibrate", choices=["none", "isotonic"], default="isotonic")
    tr.add_argument("--calib_bins", type=int, default=10)

    # eval
    ev = sub.add_parser("eval")
    ev.add_argument("--model_joblib", required=True)
    ev.add_argument("--clinical_csv", required=True)
    ev.add_argument("--radiomics_csv", required=True)
    ev.add_argument("--split_json", required=True)
    ev.add_argument("--out_dir", required=True)

    # predict
    pr = sub.add_parser("predict")
    pr.add_argument("--model_joblib", required=True)
    pr.add_argument("--patient_id", default=None)
    pr.add_argument("--patient_clinical_json", default=None)
    pr.add_argument("--patient_clinical_csv", default=None)
    pr.add_argument("--patient_radiomics_csv", required=True)
    pr.add_argument("--out_dir", required=True)
    pr.add_argument("--explain_topk", type=int, default=10)

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    else:
        raise RuntimeError("Unknown cmd")


if __name__ == "__main__":
    main()
