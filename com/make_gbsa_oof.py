#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

try:
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
except Exception as e:
    raise ImportError("Need scikit-survival (sksurv). Please install it in your env.") from e


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_split(split_json: str) -> Dict[str, List[str]]:
    obj = json.load(open(split_json, "r"))
    # allow train/test or train/val/test
    if "train" in obj and "test" in obj:
        out = {"train": [str(x) for x in obj["train"]], "test": [str(x) for x in obj["test"]]}
        if "val" in obj:
            out["val"] = [str(x) for x in obj["val"]]
        return out
    if "train_ids" in obj and "test_ids" in obj:
        out = {"train": [str(x) for x in obj["train_ids"]], "test": [str(x) for x in obj["test_ids"]]}
        if "val_ids" in obj:
            out["val"] = [str(x) for x in obj["val_ids"]]
        return out
    if "splits" in obj and "train" in obj["splits"] and "test" in obj["splits"]:
        sp = obj["splits"]
        out = {"train": [str(x) for x in sp["train"]], "test": [str(x) for x in sp["test"]]}
        if "val" in sp:
            out["val"] = [str(x) for x in sp["val"]]
        return out
    raise ValueError(f"Unrecognized split json format: {split_json}")


def cindex(y_struct, risk: np.ndarray) -> float:
    ev = y_struct["event"]
    tm = y_struct["time"]
    return float(concordance_index_censored(ev, tm, risk)[0])


def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # try case-insensitive match
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def build_Xy(
    clinical_csv: str,
    radiomics_csv: str,
    selected_rad: List[str],
    use_age: int,
    use_gender: int,
    use_histology: int,
    use_t_stage: int,
    use_stage: int,
    use_n_stage: int,
    use_m_stage: int,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    clin = pd.read_csv(clinical_csv)
    rad = pd.read_csv(radiomics_csv)

    pid_c = pick_col(clin, ["PatientID", "patient_id", "PatientId"])
    pid_r = pick_col(rad,  ["PatientID", "patient_id", "PatientId"])
    if pid_c is None or pid_r is None:
        raise ValueError("Both clinical and radiomics must contain PatientID column (case-insensitive allowed).")

    clin[pid_c] = clin[pid_c].astype(str)
    rad[pid_r] = rad[pid_r].astype(str)
    clin = clin.set_index(pid_c)
    rad  = rad.set_index(pid_r)

    # survival columns
    tcol = pick_col(clin, ["Survival.time", "survival_time", "time"])
    ecol = pick_col(clin, ["deadstatus.event", "event", "status"])
    if tcol is None or ecol is None:
        raise ValueError("Cannot find survival time/event columns in clinical csv.")

    time = pd.to_numeric(clin[tcol], errors="coerce").astype(float)
    event = pd.to_numeric(clin[ecol], errors="coerce").fillna(0).astype(int).astype(bool)

    # selected radiomics - filter to only those that exist in radiomics CSV
    # (selected_features may contain clinical features too, which are handled separately)
    rad_sel_cols = [c for c in selected_rad if c in rad.columns]
    missing_rad = [c for c in selected_rad if c not in rad.columns]
    if missing_rad:
        print(f"[INFO] {len(missing_rad)} selected features are not in radiomics CSV (likely clinical features): {missing_rad[:5]}...")
    if not rad_sel_cols:
        raise ValueError("No radiomics features found in selected_features list that exist in radiomics CSV.")
    
    rad_sel = rad[rad_sel_cols].copy()
    for c in rad_sel.columns:
        rad_sel[c] = pd.to_numeric(rad_sel[c], errors="coerce")
    rad_sel = rad_sel.add_prefix("rad__")

    X_parts = []
    cat_cols = []
    num_cols = []

    if use_age:
        age_col = pick_col(clin, ["age", "Age"])
        if age_col is None:
            raise ValueError("use_age=1 but cannot find 'age' column.")
        age = pd.to_numeric(clin[age_col], errors="coerce")
        X_parts.append(pd.DataFrame({"age": age}, index=clin.index))
        num_cols.append("age")

    def add_cat(name: str, col_candidates: List[str]):
        col = pick_col(clin, col_candidates)
        if col is None:
            raise ValueError(f"use_{name}=1 but cannot find column among {col_candidates}")
        s = clin[col].astype("object")
        X_parts.append(pd.DataFrame({name: s}, index=clin.index))
        cat_cols.append(name)

    if use_gender:
        add_cat("gender", ["gender", "Gender"])
    if use_histology:
        add_cat("Histology", ["Histology", "histology"])
    if use_t_stage:
        add_cat("clinical.T.Stage", ["clinical.T.Stage", "Clinical.T.Stage", "T.Stage", "T_stage"])
    if use_stage:
        add_cat("Overall.Stage", ["Overall.Stage", "overall_stage", "Stage", "stage"])
    if use_n_stage:
        add_cat("Clinical.N.Stage", ["Clinical.N.Stage", "clinical.N.Stage", "N.Stage", "N_stage"])
    if use_m_stage:
        add_cat("Clinical.M.Stage", ["Clinical.M.Stage", "clinical.M.Stage", "M.Stage", "M_stage"])

    X_parts.append(rad_sel)
    num_cols.extend(list(rad_sel.columns))

    X = pd.concat(X_parts, axis=1)

    # drop rows without survival time
    ok = time.notna()
    X = X.loc[ok]
    time = time.loc[ok]
    event = event.loc[ok]

    y = Surv.from_arrays(event=event.values, time=time.values)
    return X, y, cat_cols, num_cols


def build_gbsa(params: Dict, seed: int):
    gb = GradientBoostingSurvivalAnalysis(random_state=seed)
    allowed = set(gb.get_params().keys())
    filt = {k: v for k, v in params.items() if k in allowed}
    # always enforce these two if present
    if "random_state" in allowed:
        filt["random_state"] = seed
    gb.set_params(**filt)
    return gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_csv", required=True)
    ap.add_argument("--radiomics_csv", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--n_folds", type=int, default=5)

    ap.add_argument("--selected_features_txt", required=True,
                    help="One radiomics feature per line (the zip/recipe mRMR-selected list).")
    ap.add_argument("--gbsa_params_json", required=True,
                    help="GBSA params json dumped from recipe_summary (best_params).")

    ap.add_argument("--scale", choices=["minmax", "zscore", "none"], default="minmax")
    ap.add_argument("--age_impute", choices=["mean", "median"], default="mean")

    ap.add_argument("--use_age", type=int, default=1)
    ap.add_argument("--use_gender", type=int, default=1)
    ap.add_argument("--use_histology", type=int, default=1)
    ap.add_argument("--use_t_stage", type=int, default=1)
    ap.add_argument("--use_stage", type=int, default=1)
    ap.add_argument("--use_n_stage", type=int, default=0)
    ap.add_argument("--use_m_stage", type=int, default=0)

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    selected = [l.strip() for l in open(args.selected_features_txt, "r") if l.strip()]
    params = json.load(open(args.gbsa_params_json, "r"))

    X, y, cat_cols, num_cols = build_Xy(
        args.clinical_csv, args.radiomics_csv, selected,
        args.use_age, args.use_gender, args.use_histology,
        args.use_t_stage, args.use_stage, args.use_n_stage, args.use_m_stage
    )

    split = load_split(args.split_json)
    train_ids = [pid for pid in split["train"] if pid in X.index]
    test_ids  = [pid for pid in split["test"]  if pid in X.index]

    Xtr = X.loc[train_ids].copy()
    ytr = y[np.array([X.index.get_loc(pid) for pid in train_ids])]
    Xte = X.loc[test_ids].copy()
    yte = y[np.array([X.index.get_loc(pid) for pid in test_ids])]

    # preprocess
    num_imputer = SimpleImputer(strategy=args.age_impute if "age" in num_cols else "median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    if args.scale == "minmax":
        scaler = MinMaxScaler()
    elif args.scale == "zscore":
        scaler = StandardScaler()
    else:
        scaler = "passthrough"

    # sklearn >=1.2 uses sparse_output
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", num_imputer), ("sc", scaler)]), num_cols),
            ("cat", Pipeline([("imp", cat_imputer), ("ohe", ohe)]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    gbsa = build_gbsa(params, seed=args.seed)

    # OOF
    strat = ytr["event"].astype(int)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    oof = np.zeros(len(train_ids), dtype=float)
    for fold, (i_tr, i_va) in enumerate(skf.split(Xtr, strat), 1):
        pipe = Pipeline([("prep", preprocess), ("gbsa", build_gbsa(params, seed=args.seed))])
        pipe.fit(Xtr.iloc[i_tr], ytr[i_tr])
        oof[i_va] = pipe.predict(Xtr.iloc[i_va])

    oof_ci = cindex(ytr, oof)

    # refit and test
    pipe_full = Pipeline([("prep", preprocess), ("gbsa", gbsa)])
    pipe_full.fit(Xtr, ytr)
    test_pred = pipe_full.predict(Xte)
    test_ci = cindex(yte, test_pred)

    print(f"[GBSA OOF FIXED] train_oof_cindex = {oof_ci:.6f}")
    print(f"[GBSA refit FIXED] test_cindex(refit) = {test_ci:.6f}")
    print(f"[INFO] n_train={len(train_ids)} n_test={len(test_ids)} | rad_selected={len(selected)}")

    pd.DataFrame({"PatientID": train_ids, "risk": oof}).to_csv(os.path.join(args.out_dir, "base_train_oof.csv"), index=False)
    pd.DataFrame({"PatientID": test_ids,  "risk": test_pred}).to_csv(os.path.join(args.out_dir, "base_test_refit.csv"), index=False)

    json.dump({
        "seed": args.seed,
        "n_folds": args.n_folds,
        "train_oof_cindex": float(oof_ci),
        "test_refit_cindex": float(test_ci),
        "n_rad_selected": int(len(selected)),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "gbsa_params_used": gbsa.get_params(),
    }, open(os.path.join(args.out_dir, "gbsa_oof_info.json"), "w"), indent=2)

    print(f"[OK] Saved: {args.out_dir}/base_train_oof.csv")
    print(f"[OK] Saved: {args.out_dir}/base_test_refit.csv")
    print(f"[OK] Saved: {args.out_dir}/gbsa_oof_info.json")


if __name__ == "__main__":
    main()
