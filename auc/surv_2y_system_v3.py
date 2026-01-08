#!/usr/bin/env python3
"""surv_2y_system_v3.py

Goal
----
Train / evaluate / deploy a 2-year survival probability model on NSCLC-Radiomics (LUNG1)
using the *same split* and feature subset you already validated for the GBSA baseline.

This v3 focuses on **probability reliability** (AUC@2y + Brier@2y + calibration) and fixes
remaining issues seen in v1/v2:

1) Isotonic collapse to 0/1  -> recommend Platt, always clip to [pmin, pmax].
2) In-sample calibration bias -> optional **OOF calibration** on training set.
3) Calibration table crash    -> robust bin handling (Series vs Categorical).
4) "no valid feature names" warnings -> always pass DataFrames with stable column names.

Dependencies (same as your current env):
  - numpy, pandas, joblib
  - scikit-learn
  - scikit-survival (sksurv)

Typical usage (mirrors your v2 commands)
----------------------------------------
Train (with OOF Platt calibration):

python surv_2y_system_v3.py train \
  --clinical_csv ...Lung1.clinical-version3-Oct-2019.csv \
  --radiomics_csv ...Tb_features_1688_from_yaml.csv \
  --split_json ...split_ClinicplusRadiomics_6cf7bc2c.json \
  --out_dir ./gbsa_2y_export_v3 \
  --seed 91 \
  --selected_radiomics_txt gbsa91_selected_radiomics.txt \
  --gbsa_params_json gbsa91_best_params.json \
  --scale minmax \
  --horizon 730 \
  --calibrate platt \
  --calibrate_oof_folds 5 \
  --platt_C 1e6

Evaluate:

python surv_2y_system_v3.py eval \
  --model_joblib ./gbsa_2y_export_v3/gbsa_2y_model_v3.joblib \
  --clinical_csv ... \
  --radiomics_csv ... \
  --split_json ... \
  --out_dir ./gbsa_2y_eval_v3 \
  --bootstrap 200 --seed 42

Predict single patient:

python surv_2y_system_v3.py predict \
  --model_joblib ./gbsa_2y_export_v3/gbsa_2y_model_v3.joblib \
  --patient_id LUNG1-006 \
  --clinical_json ./LUNG1-006_clinical.json \
  --radiomics_row_csv ./LUNG1-006_radiomics.csv \
  --out_dir ./predictions_v3
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv


# ---------------------------
#  Utilities / conventions
# ---------------------------


def clip_probability(p: np.ndarray, pmin: float = 0.01, pmax: float = 0.99) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.clip(p, pmin, pmax)


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def inv_logit(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_txt_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def read_split_json(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # Accept either {train_ids, test_ids} or {train, test}
    for a, b in [("train_ids", "test_ids"), ("train", "test")]:
        if a in obj and b in obj:
            return {"train": obj[a], "test": obj[b]}
    raise ValueError(f"Unrecognized split_json format: keys={list(obj.keys())}")


def km_survival_at(time: np.ndarray, event: np.ndarray, t: float) -> float:
    """Kaplan–Meier estimate S(t) for a cohort."""
    event_bool = np.asarray(event).astype(bool)
    time = np.asarray(time, dtype=float)
    km_t, km_s = kaplan_meier_estimator(event_bool, time)
    # Step function: take last survival prob where time <= t
    if len(km_t) == 0:
        return 1.0
    idx = np.searchsorted(km_t, t, side="right") - 1
    if idx < 0:
        return 1.0
    return float(km_s[idx])


def predict_survival_at(model: GradientBoostingSurvivalAnalysis, X: pd.DataFrame, t: float) -> np.ndarray:
    """Return S(t|x) for each row."""
    sf_list = model.predict_survival_function(X)
    out = np.empty(len(sf_list), dtype=float)
    for i, sf in enumerate(sf_list):
        out[i] = float(sf(t))
    return out


def filter_params_for_estimator(params: Dict, estimator) -> Dict:
    """Drop unknown keys so JSONs from other versions don't crash."""
    valid = set(estimator.get_params(deep=False).keys())
    return {k: v for k, v in params.items() if k in valid}


# ---------------------------
#  Clinical mappings
# ---------------------------


GENDER_MAP = {
    1: "male",
    2: "female",
    "1": "male",
    "2": "female",
    "M": "male",
    "F": "female",
    "male": "male",
    "female": "female",
}


HISTOLOGY_MAP = {
    1: "adenocarcinoma",
    2: "squamous cell carcinoma",
    3: "large cell carcinoma",
    4: "not otherwise specified",
    "1": "adenocarcinoma",
    "2": "squamous cell carcinoma",
    "3": "large cell carcinoma",
    "4": "not otherwise specified",
}


def map_value(v, mapping: Dict, default: str = "unknown") -> str:
    if pd.isna(v):
        return default
    return mapping.get(v, mapping.get(str(v), default))


def normalize_stage(v) -> str:
    if pd.isna(v):
        return "unknown"
    s = str(v).strip()
    # Normalize common variants
    s = s.replace("Stage", "").replace("stage", "").strip()
    s = s.replace(" ", "")
    return s if s else "unknown"


# ---------------------------
#  Preprocessing bundle
# ---------------------------


@dataclass
class PreprocessBundle:
    numeric_cols: List[str]
    cat_cols: List[str]
    age_impute: str
    scale: str

    num_fill: Dict[str, float]
    cat_fill: Dict[str, str]
    scaler: object
    encoder: OneHotEncoder
    feature_names: List[str]

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        # Fill
        for c in self.numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(self.num_fill[c])
        for c in self.cat_cols:
            df[c] = df[c].astype("object").where(~df[c].isna(), self.cat_fill[c])

        X_num = df[self.numeric_cols].to_numpy(dtype=float)
        if self.scale == "none":
            X_num_s = X_num
        else:
            X_num_s = self.scaler.transform(X_num)

        X_cat = df[self.cat_cols].astype("object")
        X_cat_e = self.encoder.transform(X_cat)

        X = np.hstack([X_num_s, X_cat_e])
        return pd.DataFrame(X, columns=self.feature_names, index=df.index)


def fit_preprocess_bundle(
    df_train: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
    age_impute: str = "mean",
    scale: str = "minmax",
) -> PreprocessBundle:
    df = df_train.copy()
    # Numeric fill
    num_fill: Dict[str, float] = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if c == "age":
            if age_impute == "median":
                num_fill[c] = float(np.nanmedian(s.values))
            else:
                num_fill[c] = float(np.nanmean(s.values))
        else:
            num_fill[c] = float(np.nanmedian(s.values))
        df[c] = s.fillna(num_fill[c])

    # Categorical fill
    cat_fill: Dict[str, str] = {}
    for c in cat_cols:
        s = df[c].astype("object")
        if s.isna().all():
            cat_fill[c] = "unknown"
        else:
            cat_fill[c] = str(s.dropna().mode().iloc[0])
        df[c] = s.where(~s.isna(), cat_fill[c])

    X_num = df[numeric_cols].to_numpy(dtype=float)
    if scale == "none":
        scaler = None
        X_num_s = X_num
    elif scale == "zscore":
        scaler = StandardScaler()
        X_num_s = scaler.fit_transform(X_num)
    else:
        scaler = MinMaxScaler()
        X_num_s = scaler.fit_transform(X_num)

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = df[cat_cols].astype("object")
    X_cat_e = encoder.fit_transform(X_cat)

    feat_names = list(numeric_cols) + list(encoder.get_feature_names_out(cat_cols))
    X = np.hstack([X_num_s, X_cat_e])
    # Create DF only to validate shapes; we return names.
    _ = pd.DataFrame(X, columns=feat_names)

    return PreprocessBundle(
        numeric_cols=numeric_cols,
        cat_cols=cat_cols,
        age_impute=age_impute,
        scale=scale,
        num_fill=num_fill,
        cat_fill=cat_fill,
        scaler=scaler,
        encoder=encoder,
        feature_names=feat_names,
    )


# ---------------------------
#  Calibration
# ---------------------------


@dataclass
class PlattCalibrator:
    lr: LogisticRegression
    pmin: float = 0.01
    pmax: float = 0.99

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        z = logit(p_raw)
        p = self.lr.predict_proba(z.reshape(-1, 1))[:, 1]
        return clip_probability(p, self.pmin, self.pmax)


@dataclass
class IdentityCalibrator:
    pmin: float = 0.01
    pmax: float = 0.99

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        return clip_probability(p_raw, self.pmin, self.pmax)


def build_2y_labels(time: np.ndarray, event: np.ndarray, horizon: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mask_known, y_survive2y) for calibration and simple sanity checks.

    known if:
      - event happened before horizon (then survive2y=0)
      - OR time >= horizon (then survive2y=1)
    unknown if censored before horizon.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event).astype(int)
    known = (time >= horizon) | ((time < horizon) & (event == 1))
    y = (time >= horizon).astype(int)
    return known, y


def fit_platt_calibrator(
    time: np.ndarray,
    event: np.ndarray,
    p_raw: np.ndarray,
    horizon: float,
    C: float = 1e6,
    seed: int = 0,
    pmin: float = 0.01,
    pmax: float = 0.99,
) -> PlattCalibrator:
    known, y = build_2y_labels(time, event, horizon)
    z = logit(p_raw[known])
    yk = y[known]
    # Use very weak regularization by default (C large) to avoid over-compressing.
    lr = LogisticRegression(C=float(C), solver="lbfgs", random_state=int(seed), max_iter=2000)
    lr.fit(z.reshape(-1, 1), yk)
    return PlattCalibrator(lr=lr, pmin=pmin, pmax=pmax)


def fit_calibrator(
    calibrate: str,
    time: np.ndarray,
    event: np.ndarray,
    p_raw: np.ndarray,
    horizon: float,
    seed: int,
    platt_C: float,
    pmin: float,
    pmax: float,
):
    calibrate = (calibrate or "none").lower()
    if calibrate in ("none", "identity"):
        return IdentityCalibrator(pmin=pmin, pmax=pmax)
    if calibrate == "platt":
        return fit_platt_calibrator(time, event, p_raw, horizon, C=platt_C, seed=seed, pmin=pmin, pmax=pmax)
    raise ValueError(f"Unsupported calibrator: {calibrate}. Supported: none|platt")


# ---------------------------
#  Core pipeline
# ---------------------------


def load_and_merge(
    clinical_csv: str,
    radiomics_csv: str,
    selected_radiomics: Optional[List[str]],
    use_n_stage: bool,
    use_m_stage: bool,
) -> pd.DataFrame:
    clin = pd.read_csv(clinical_csv)
    rad = pd.read_csv(radiomics_csv)

    if "PatientID" not in clin.columns or "PatientID" not in rad.columns:
        raise ValueError("Both clinical and radiomics CSV must contain 'PatientID' column")

    # Normalize clinical categorical columns for consistency.
    clin = clin.copy()
    clin["gender"] = clin["gender"].apply(lambda v: map_value(v, GENDER_MAP, "unknown"))
    clin["Histology"] = clin["Histology"].apply(lambda v: map_value(v, HISTOLOGY_MAP, "unknown"))
    clin["clinical.T.Stage"] = clin["clinical.T.Stage"].apply(normalize_stage)
    clin["Overall.Stage"] = clin["Overall.Stage"].apply(normalize_stage)

    if use_n_stage and "Clinical.N.Stage" in clin.columns:
        clin["Clinical.N.Stage"] = clin["Clinical.N.Stage"].apply(normalize_stage)
    if use_m_stage and "Clinical.M.Stage" in clin.columns:
        clin["Clinical.M.Stage"] = clin["Clinical.M.Stage"].apply(normalize_stage)

    # Keep only selected radiomics to avoid accidental leakage and speed issues.
    if selected_radiomics is not None:
        missing = [c for c in selected_radiomics if c not in rad.columns]
        if missing:
            raise ValueError(f"Selected radiomics not found in radiomics CSV: {missing[:10]} ... (n={len(missing)})")
        keep_cols = ["PatientID"] + selected_radiomics
        rad = rad[keep_cols]

    df = clin.merge(rad, on="PatientID", how="inner")
    # Sanity columns
    if "Survival.time" not in df.columns or "deadstatus.event" not in df.columns:
        raise ValueError("clinical CSV must contain 'Survival.time' and 'deadstatus.event'")
    return df


def make_train_test(df: pd.DataFrame, split: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr_ids = set(split["train"])
    te_ids = set(split["test"])
    df_tr = df[df["PatientID"].isin(tr_ids)].copy()
    df_te = df[df["PatientID"].isin(te_ids)].copy()
    # Keep deterministic order (split json already stable).
    df_tr["__order"] = df_tr["PatientID"].apply(lambda x: split["train"].index(x) if x in tr_ids else 10**9)
    df_te["__order"] = df_te["PatientID"].apply(lambda x: split["test"].index(x) if x in te_ids else 10**9)
    df_tr = df_tr.sort_values("__order").drop(columns=["__order"])
    df_te = df_te.sort_values("__order").drop(columns=["__order"])
    return df_tr, df_te


def build_feature_lists(
    df: pd.DataFrame,
    selected_radiomics: List[str],
    use_age: bool,
    use_gender: bool,
    use_histology: bool,
    use_t_stage: bool,
    use_stage: bool,
    use_n_stage: bool,
    use_m_stage: bool,
) -> Tuple[List[str], List[str]]:
    numeric_cols: List[str] = []
    cat_cols: List[str] = []
    if use_age:
        numeric_cols.append("age")
    # radiomics are numeric
    numeric_cols.extend(selected_radiomics)

    if use_gender:
        cat_cols.append("gender")
    if use_histology:
        cat_cols.append("Histology")
    if use_t_stage:
        cat_cols.append("clinical.T.Stage")
    if use_stage:
        cat_cols.append("Overall.Stage")
    if use_n_stage:
        cat_cols.append("Clinical.N.Stage")
    if use_m_stage:
        cat_cols.append("Clinical.M.Stage")

    # Verify existence
    for c in numeric_cols + cat_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required feature column in merged DF: {c}")
    return numeric_cols, cat_cols


def y_from_df(df: pd.DataFrame):
    tmp = pd.DataFrame({"time": df["Survival.time"].astype(float), "event": df["deadstatus.event"].astype(int)})
    return Surv.from_dataframe("event", "time", tmp)


def fit_gbsa(
    Xtr: pd.DataFrame,
    ytr,
    params: Dict,
    seed: int,
) -> GradientBoostingSurvivalAnalysis:
    base = GradientBoostingSurvivalAnalysis(random_state=int(seed))
    params_f = filter_params_for_estimator(params, base)
    params_f["random_state"] = int(seed)
    model = GradientBoostingSurvivalAnalysis(**params_f)
    model.fit(Xtr, ytr)
    return model


def compute_core_metrics(
    ytr,
    yte,
    risk_tr: np.ndarray,
    risk_te: np.ndarray,
    p2_te: np.ndarray,
    horizon: float,
) -> Dict[str, float]:
    c_tr = float(concordance_index_censored(ytr["event"], ytr["time"], risk_tr)[0])
    c_te = float(concordance_index_censored(yte["event"], yte["time"], risk_te)[0])

    times = np.array([float(horizon)])
    auc, _ = cumulative_dynamic_auc(ytr, yte, risk_te, times)
    
    # 【修改处】：把 bs, _ 改为 _, bs
    _, bs = brier_score(ytr, yte, np.c_[p2_te], times)

    return {
        "cindex_train": c_tr,
        "cindex_test": c_te,
        "auc2y": float(auc[0]),
        "brier2y": float(bs[0]),
    }


def calibration_table_km(
    time: np.ndarray,
    event: np.ndarray,
    p: np.ndarray,
    horizon: float,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin by predicted survival prob and compute KM observed S(horizon) per bin."""
    p = np.asarray(p, dtype=float)
    # qcut can return Categorical or Series depending on input
    bins = pd.qcut(p, q=min(int(n_bins), len(p)), duplicates="drop")
    if isinstance(bins, pd.Series):
        cats = bins.cat.categories
        bin_codes = bins
    else:
        # pd.Categorical
        cats = bins.categories
        bin_codes = pd.Series(bins)

    rows = []
    for c in cats:
        m = bin_codes == c
        if int(m.sum()) == 0:
            continue
        p_mean = float(np.mean(p[m]))
        s_obs = km_survival_at(time[m], event[m], horizon)
        rows.append({
            "bin": str(c),
            "n": int(m.sum()),
            "p_mean": p_mean,
            "km_surv2y": float(s_obs),
            "abs_error": float(abs(p_mean - s_obs)),
        })

    out = pd.DataFrame(rows)
    if len(out) > 0:
        out["weight"] = out["n"] / out["n"].sum()
        out["ece"] = float((out["weight"] * out["abs_error"]).sum())
    return out


def bootstrap_ci(
    ytr,
    yte,
    risk_te: np.ndarray,
    p2_te: np.ndarray,
    horizon: float,
    n_boot: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(int(seed))
    n = len(risk_te)
    times = np.array([float(horizon)])
    aucs = []
    briers = []
    cidxs = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        yb = yte[idx]
        rb = risk_te[idx]
        pb = p2_te[idx]
        cidxs.append(float(concordance_index_censored(yb["event"], yb["time"], rb)[0]))
        auc, _ = cumulative_dynamic_auc(ytr, yb, rb, times)
        
        # 【修改处】：把 bs, _ 改为 _, bs
        _, bs = brier_score(ytr, yb, np.c_[pb], times)
        
        aucs.append(float(auc[0]))
        briers.append(float(bs[0]))

    def summar(x: List[float]) -> Dict[str, float]:
        x = np.asarray(x, dtype=float)
        lo, hi = np.percentile(x, [2.5, 97.5])
        return {
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)),
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),
        }

    return {
        "cindex_test": summar(cidxs),
        "auc2y": summar(aucs),
        "brier2y": summar(briers),
    }


def oof_pred_p2_raw(
    df_train: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
    params: Dict,
    horizon: float,
    scale: str,
    age_impute: str,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """OOF S(horizon) for training set, used to fit calibrator without in-sample bias."""
    kf = KFold(n_splits=int(n_folds), shuffle=True, random_state=int(seed))
    p_oof = np.full(len(df_train), np.nan, dtype=float)
    for tr_idx, va_idx in kf.split(df_train):
        df_tr = df_train.iloc[tr_idx]
        df_va = df_train.iloc[va_idx]
        prep = fit_preprocess_bundle(df_tr, numeric_cols, cat_cols, age_impute=age_impute, scale=scale)
        Xtr = prep.transform(df_tr)
        Xva = prep.transform(df_va)
        ytr = y_from_df(df_tr)
        model = fit_gbsa(Xtr, ytr, params=params, seed=seed)
        p_oof[va_idx] = predict_survival_at(model, Xva, horizon)

    if np.isnan(p_oof).any():
        raise RuntimeError("OOF prediction produced NaNs (unexpected)")
    return p_oof


# ---------------------------
#  CLI commands
# ---------------------------


def cmd_train(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)

    selected_r = read_txt_list(args.selected_radiomics_txt)
    split = read_split_json(args.split_json)
    df = load_and_merge(
        args.clinical_csv,
        args.radiomics_csv,
        selected_radiomics=selected_r,
        use_n_stage=bool(args.use_n_stage),
        use_m_stage=bool(args.use_m_stage),
    )
    df_tr, df_te = make_train_test(df, split)

    horizon = float(args.horizon)
    med = float(np.median(df["Survival.time"].values))
    print(f"[INFO] median(Survival.time)≈{med:.1f} (looks like DAYS). 2y={horizon}.")

    numeric_cols, cat_cols = build_feature_lists(
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

    prep = fit_preprocess_bundle(df_tr, numeric_cols, cat_cols, age_impute=args.age_impute, scale=args.scale)
    Xtr = prep.transform(df_tr)
    Xte = prep.transform(df_te)
    ytr = y_from_df(df_tr)
    yte = y_from_df(df_te)

    base_params = json.load(open(args.gbsa_params_json, "r", encoding="utf-8"))
    model = fit_gbsa(Xtr, ytr, params=base_params, seed=int(args.seed))

    risk_tr = model.predict(Xtr)
    risk_te = model.predict(Xte)
    p2_tr_raw = predict_survival_at(model, Xtr, horizon)
    p2_te_raw = predict_survival_at(model, Xte, horizon)

    # Calibration (optionally OOF)
    if args.calibrate_oof_folds and int(args.calibrate_oof_folds) > 1 and args.calibrate.lower() != "none":
        print(f"[CAL] Using OOF calibration with {int(args.calibrate_oof_folds)} folds")
        p2_oof = oof_pred_p2_raw(
            df_tr,
            numeric_cols=numeric_cols,
            cat_cols=cat_cols,
            params=base_params,
            horizon=horizon,
            scale=args.scale,
            age_impute=args.age_impute,
            n_folds=int(args.calibrate_oof_folds),
            seed=int(args.seed),
        )
        calibrator = fit_calibrator(
            args.calibrate,
            time=df_tr["Survival.time"].values,
            event=df_tr["deadstatus.event"].values,
            p_raw=p2_oof,
            horizon=horizon,
            seed=int(args.seed),
            platt_C=float(args.platt_C),
            pmin=float(args.pmin),
            pmax=float(args.pmax),
        )
    else:
        calibrator = fit_calibrator(
            args.calibrate,
            time=df_tr["Survival.time"].values,
            event=df_tr["deadstatus.event"].values,
            p_raw=p2_tr_raw,
            horizon=horizon,
            seed=int(args.seed),
            platt_C=float(args.platt_C),
            pmin=float(args.pmin),
            pmax=float(args.pmax),
        )

    p2_te = calibrator.predict(p2_te_raw)

    # Metrics
    metrics = compute_core_metrics(ytr, yte, risk_tr, risk_te, p2_te, horizon)
    print(f"[TRAIN] cindex={metrics['cindex_train']:.4f}  [TEST] cindex={metrics['cindex_test']:.4f}")

    # Baseline KM for Brier skill
    s0 = km_survival_at(df_tr["Survival.time"].values, df_tr["deadstatus.event"].values, horizon)
    _, bs0 = brier_score(ytr, yte, np.c_[np.full(len(df_te), s0)], np.array([horizon]))
    metrics["km_surv2y_train"] = float(s0)
    metrics["brier2y_km"] = float(bs0[0])
    metrics["brier2y_skill"] = float(1.0 - metrics["brier2y"] / max(metrics["brier2y_km"], 1e-12))

    # Calibration table on test
    calib_df = calibration_table_km(
        time=df_te["Survival.time"].values,
        event=df_te["deadstatus.event"].values,
        p=p2_te,
        horizon=horizon,
        n_bins=int(args.calib_bins),
    )
    calib_path = os.path.join(args.out_dir, "calibration_table_test.csv")
    calib_df.to_csv(calib_path, index=False)

    # Save predictions
    out_pred = pd.DataFrame({
        "PatientID": df_te["PatientID"].values,
        "time": df_te["Survival.time"].values,
        "event": df_te["deadstatus.event"].values,
        "risk": risk_te,
        "surv2y_raw": p2_te_raw,
        "surv2y_cal": p2_te,
    })
    pred_path = os.path.join(args.out_dir, "internal_test_pred_2y.csv")
    out_pred.to_csv(pred_path, index=False)

    bundle = {
        "seed": int(args.seed),
        "horizon": float(horizon),
        "selected_radiomics": selected_r,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "preprocess": prep,
        "model": model,
        "calibrator": calibrator,
        "train_risk": np.asarray(risk_tr, dtype=float),
        "metrics": metrics,
    }
    model_path = os.path.join(args.out_dir, "gbsa_2y_model_v3.joblib")
    joblib.dump(bundle, model_path)

    report = {
        "metrics": metrics,
        "n_train": int(len(df_tr)),
        "n_test": int(len(df_te)),
        "n_features": int(len(prep.feature_names)),
        "calibrate": args.calibrate,
        "calibrate_oof_folds": int(args.calibrate_oof_folds or 0),
        "platt_C": float(args.platt_C),
        "pmin": float(args.pmin),
        "pmax": float(args.pmax),
        "calibration_table": "calibration_table_test.csv",
    }
    with open(os.path.join(args.out_dir, "train_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Saved: {model_path}")
    print(f"[OK] Saved: {os.path.join(args.out_dir, 'train_report.json')}")
    print(f"[OK] Saved: {pred_path}")
    print(f"[OK] Saved: {calib_path}")


def cmd_eval(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    bundle = joblib.load(args.model_joblib)
    horizon = float(bundle["horizon"])

    split = read_split_json(args.split_json)
    df = load_and_merge(
        args.clinical_csv,
        args.radiomics_csv,
        selected_radiomics=bundle["selected_radiomics"],
        use_n_stage=("Clinical.N.Stage" in bundle["cat_cols"]),
        use_m_stage=("Clinical.M.Stage" in bundle["cat_cols"]),
    )
    df_tr, df_te = make_train_test(df, split)

    prep: PreprocessBundle = bundle["preprocess"]
    model: GradientBoostingSurvivalAnalysis = bundle["model"]
    calibrator = bundle["calibrator"]

    Xtr = prep.transform(df_tr)
    Xte = prep.transform(df_te)
    ytr = y_from_df(df_tr)
    yte = y_from_df(df_te)

    risk_te = model.predict(Xte)
    p2_te_raw = predict_survival_at(model, Xte, horizon)
    p2_te = calibrator.predict(p2_te_raw)

    metrics = compute_core_metrics(ytr, yte, model.predict(Xtr), risk_te, p2_te, horizon)

    # Baseline KM
    s0 = km_survival_at(df_tr["Survival.time"].values, df_tr["deadstatus.event"].values, horizon)
    _, bs0 = brier_score(ytr, yte, np.c_[np.full(len(df_te), s0)], np.array([horizon]))
    metrics["km_surv2y_train"] = float(s0)
    metrics["brier2y_km"] = float(bs0[0])
    metrics["brier2y_skill"] = float(1.0 - metrics["brier2y"] / max(metrics["brier2y_km"], 1e-12))

    # Calibration table on test
    calib_df = calibration_table_km(
        time=df_te["Survival.time"].values,
        event=df_te["deadstatus.event"].values,
        p=p2_te,
        horizon=horizon,
        n_bins=int(args.calib_bins),
    )
    calib_path = os.path.join(args.out_dir, "calibration_table_test.csv")
    calib_df.to_csv(calib_path, index=False)

    # Bootstrap CI
    ci = None
    if args.bootstrap and int(args.bootstrap) > 0:
        ci = bootstrap_ci(
            ytr=ytr,
            yte=yte,
            risk_te=np.asarray(risk_te, dtype=float),
            p2_te=np.asarray(p2_te, dtype=float),
            horizon=horizon,
            n_boot=int(args.bootstrap),
            seed=int(args.seed),
        )

    # Save preds
    pred = pd.DataFrame({
        "PatientID": df_te["PatientID"].values,
        "time": df_te["Survival.time"].values,
        "event": df_te["deadstatus.event"].values,
        "risk": risk_te,
        "surv2y_raw": p2_te_raw,
        "surv2y_cal": p2_te,
    })
    pred_path = os.path.join(args.out_dir, "eval_pred_2y.csv")
    pred.to_csv(pred_path, index=False)

    report = {"metrics": metrics, "bootstrap": ci}
    rep_path = os.path.join(args.out_dir, "eval_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[EVAL] cindex={metrics['cindex_test']:.4f} auc2y={metrics['auc2y']:.4f} brier2y={metrics['brier2y']:.4f} (skill={metrics['brier2y_skill']:.3f})")
    if ci is not None:
        print("[CI95] cindex={ci0[ci95_lo]:.4f}-{ci0[ci95_hi]:.4f} | auc2y={ci1[ci95_lo]:.4f}-{ci1[ci95_hi]:.4f} | brier2y={ci2[ci95_lo]:.4f}-{ci2[ci95_hi]:.4f}".format(
            ci0=ci["cindex_test"], ci1=ci["auc2y"], ci2=ci["brier2y"],
        ))

    print(f"[OK] Saved: {rep_path}")
    print(f"[OK] Saved: {pred_path}")
    print(f"[OK] Saved: {calib_path}")


def cmd_predict(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    bundle = joblib.load(args.model_joblib)
    horizon = float(bundle["horizon"])
    prep: PreprocessBundle = bundle["preprocess"]
    model: GradientBoostingSurvivalAnalysis = bundle["model"]
    calibrator = bundle["calibrator"]
    train_risk = np.asarray(bundle.get("train_risk", []), dtype=float)

    clin = json.load(open(args.clinical_json, "r", encoding="utf-8"))
    rad = pd.read_csv(args.radiomics_row_csv)
    if len(rad) != 1:
        raise ValueError("radiomics_row_csv must contain exactly one row")

    row = {"PatientID": args.patient_id}
    # Expected clinical keys
    row["age"] = clin.get("age")
    row["gender"] = map_value(clin.get("gender"), GENDER_MAP, "unknown")
    row["Histology"] = map_value(clin.get("Histology"), HISTOLOGY_MAP, "unknown")
    row["clinical.T.Stage"] = normalize_stage(clin.get("clinical.T.Stage"))
    row["Overall.Stage"] = normalize_stage(clin.get("Overall.Stage"))
    if "Clinical.N.Stage" in prep.cat_cols:
        row["Clinical.N.Stage"] = normalize_stage(clin.get("Clinical.N.Stage"))
    if "Clinical.M.Stage" in prep.cat_cols:
        row["Clinical.M.Stage"] = normalize_stage(clin.get("Clinical.M.Stage"))

    # Radiomics values (must include all selected)
    for feat in bundle["selected_radiomics"]:
        if feat not in rad.columns:
            raise ValueError(f"Missing radiomics feature in row csv: {feat}")
        row[feat] = float(rad.loc[0, feat])

    df1 = pd.DataFrame([row])
    X = prep.transform(df1)
    risk = float(model.predict(X)[0])
    p_raw = float(predict_survival_at(model, X, horizon)[0])
    p_cal = float(calibrator.predict(np.array([p_raw]))[0])

    # Percentile (higher risk = worse)
    pct = None
    if train_risk.size > 0:
        pct = float((train_risk < risk).mean() * 100.0)

    out = {
        "PatientID": args.patient_id,
        "horizon_days": horizon,
        "survival_prob_2y": p_cal,
        "survival_prob_2y_raw": p_raw,
        "risk_score": risk,
        "risk_percentile_vs_train": pct,
    }

    out_path = os.path.join(args.out_dir, f"{args.patient_id}_2y_prediction_v3.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("=" * 70)
    print(f"Patient: {args.patient_id}")
    print(f"2-year survival probability: {p_cal:.3f} ({p_cal*100:.1f}%)")
    print(f"Raw (uncalibrated)        : {p_raw:.3f}")
    print(f"Risk score                : {risk:.4f}")
    if pct is not None:
        print(f"Risk percentile vs train  : {pct:.1f}%")
    print("=" * 70)
    print(f"[OK] Saved: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="2-year survival probability system (GBSA) - v3")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common feature flags (default matches your pipeline)
    def add_feature_flags(sp):
        sp.add_argument("--use_age", type=int, default=1)
        sp.add_argument("--use_gender", type=int, default=1)
        sp.add_argument("--use_histology", type=int, default=1)
        sp.add_argument("--use_t_stage", type=int, default=1)
        sp.add_argument("--use_stage", type=int, default=1)
        sp.add_argument("--use_n_stage", type=int, default=0)
        sp.add_argument("--use_m_stage", type=int, default=0)

    # Train
    t = sub.add_parser("train", help="Train GBSA + calibration for 2y survival probability")
    t.add_argument("--clinical_csv", required=True)
    t.add_argument("--radiomics_csv", required=True)
    t.add_argument("--split_json", required=True)
    t.add_argument("--out_dir", required=True)
    t.add_argument("--seed", type=int, default=91)
    t.add_argument("--selected_radiomics_txt", required=True)
    t.add_argument("--gbsa_params_json", required=True)
    t.add_argument("--scale", choices=["minmax", "zscore", "none"], default="minmax")
    t.add_argument("--age_impute", choices=["mean", "median"], default="mean")
    t.add_argument("--horizon", type=float, default=730.0)
    t.add_argument("--calibrate", choices=["none", "platt"], default="platt")
    t.add_argument("--calibrate_oof_folds", type=int, default=5)
    t.add_argument("--platt_C", type=float, default=1e6)
    t.add_argument("--pmin", type=float, default=0.01)
    t.add_argument("--pmax", type=float, default=0.99)
    t.add_argument("--calib_bins", type=int, default=10)
    add_feature_flags(t)

    # Eval
    e = sub.add_parser("eval", help="Evaluate a saved model on internal test split")
    e.add_argument("--model_joblib", required=True)
    e.add_argument("--clinical_csv", required=True)
    e.add_argument("--radiomics_csv", required=True)
    e.add_argument("--split_json", required=True)
    e.add_argument("--out_dir", required=True)
    e.add_argument("--bootstrap", type=int, default=200)
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--calib_bins", type=int, default=10)

    # Predict
    pr = sub.add_parser("predict", help="Predict 2y survival for a single patient")
    pr.add_argument("--model_joblib", required=True)
    pr.add_argument("--patient_id", required=True)
    pr.add_argument("--clinical_json", required=True)
    pr.add_argument("--radiomics_row_csv", required=True)
    pr.add_argument("--out_dir", required=True)

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
