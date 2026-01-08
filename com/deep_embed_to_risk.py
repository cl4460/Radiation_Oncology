#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deep_embed_to_risk.py

Train a light survival model on pre-extracted deep embeddings (PatientID, d0..d511)
and export train/test risk predictions aligned to a provided split JSON.

Key design goal: make fusion honest & stable.
- We select Coxnet hyperparams on an inner validation split of the TRAIN set.
- We export TRAIN predictions as out-of-fold (OOF) predictions (k-fold on train),
  so that downstream fusion weight selection does not see in-sample deep risk.

Outputs:
  out_dir/
    deep_train_preds.csv   columns: pid, time, event, pred_risk   (OOF)
    deep_test_preds.csv    columns: pid, time, event, pred_risk   (model trained on full train)
    deep_model_info.json   chosen hyperparams + c-index stats
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

try:
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored
    from sksurv.linear_model import CoxnetSurvivalAnalysis
except Exception as e:
    raise ImportError(
        "This script requires scikit-survival (sksurv). "
        "Install: `pip install scikit-survival` following your OS instructions."
    ) from e


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_split_ids(split_json: str) -> Dict[str, List[str]]:
    with open(split_json, "r") as f:
        obj = json.load(f)

    if "train" in obj and "test" in obj:
        out = {"train": obj["train"], "test": obj["test"]}
        if "val" in obj:
            out["val"] = obj["val"]
        return {k: [str(x) for x in v] for k, v in out.items()}

    if "train_ids" in obj and "test_ids" in obj:
        out = {"train": obj["train_ids"], "test": obj["test_ids"]}
        if "val_ids" in obj:
            out["val"] = obj["val_ids"]
        return {k: [str(x) for x in v] for k, v in out.items()}

    if "splits" in obj and isinstance(obj["splits"], dict) and "train" in obj["splits"] and "test" in obj["splits"]:
        sp = obj["splits"]
        out = {"train": sp["train"], "test": sp["test"]}
        if "val" in sp:
            out["val"] = sp["val"]
        return {k: [str(x) for x in v] for k, v in out.items()}

    raise ValueError(f"Unrecognized split JSON format: {split_json}")


def cindex(y_struct, risk: np.ndarray) -> float:
    return float(concordance_index_censored(y_struct["event"], y_struct["time"], risk)[0])


def read_clinical_targets(clinical_csv: str, time_col: str, event_col: str) -> pd.DataFrame:
    df = pd.read_csv(clinical_csv)
    if "PatientID" not in df.columns:
        raise ValueError("Clinical CSV must contain PatientID column.")
    df["PatientID"] = df["PatientID"].astype(str)
    df = df.set_index("PatientID")

    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not found in clinical CSV columns.")
    if event_col not in df.columns:
        raise ValueError(f"event_col '{event_col}' not found in clinical CSV columns.")

    time = pd.to_numeric(df[time_col], errors="coerce").astype(float)
    event = pd.to_numeric(df[event_col], errors="coerce").fillna(0).astype(int).astype(bool)
    out = pd.DataFrame({"time": time, "event": event}, index=df.index)
    return out


def read_embeddings(embed_csv: str) -> pd.DataFrame:
    emb = pd.read_csv(embed_csv)
    if "PatientID" not in emb.columns:
        raise ValueError("Embedding CSV must contain PatientID column.")
    emb["PatientID"] = emb["PatientID"].astype(str)
    emb = emb.set_index("PatientID")
    for c in emb.columns:
        emb[c] = pd.to_numeric(emb[c], errors="coerce")
    return emb


def preprocess_embeddings(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    scale: str = "zscore",
    pca_dim: int = 0,
    pca_whiten: int = 0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Fit preprocessing on train only, apply to train/test.
    """
    info: Dict = {}

    # median impute (train)
    med = X_train_df.median(axis=0)
    X_train = X_train_df.fillna(med).values.astype(np.float32)
    X_test = X_test_df.fillna(med).values.astype(np.float32)
    info["impute"] = "median(train)"

    if scale == "none":
        info["scale"] = "none"
    elif scale == "zscore":
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        info["scale"] = "StandardScaler(train)"
    else:
        raise ValueError(f"Unknown scale mode: {scale}")

    if pca_dim and pca_dim > 0:
        pca = PCA(n_components=int(pca_dim), random_state=int(seed), whiten=bool(pca_whiten))
        pca.fit(X_train)
        X_train = pca.transform(X_train).astype(np.float32)
        X_test = pca.transform(X_test).astype(np.float32)
        info["pca_dim"] = int(pca_dim)
        info["pca_whiten"] = int(bool(pca_whiten))
        info["pca_explained_var_sum"] = float(np.sum(pca.explained_variance_ratio_))
    else:
        info["pca_dim"] = 0

    return X_train, X_test, info


def select_coxnet_alpha(
    X_train: np.ndarray,
    y_train_df: pd.DataFrame,
    seed: int = 42,
    val_ratio: float = 0.2,
    l1_ratios: List[float] = [0.05, 0.1, 0.3, 0.5, 0.8],
    n_alphas: int = 120,
    alpha_min_ratio: float = 1e-4,
    max_iter: int = 200000,
) -> Dict:
    """
    Pick (l1_ratio, alpha) by maximizing C-index on an inner validation split of TRAIN.
    """
    strat = y_train_df["event"].astype(int).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=float(val_ratio), random_state=int(seed))
    idx = np.arange(len(strat))
    tr_idx, va_idx = next(sss.split(idx, strat))

    X_in, X_va = X_train[tr_idx], X_train[va_idx]
    y_in_df, y_va_df = y_train_df.iloc[tr_idx], y_train_df.iloc[va_idx]
    y_in = Surv.from_arrays(event=y_in_df["event"].values, time=y_in_df["time"].values)
    y_va = Surv.from_arrays(event=y_va_df["event"].values, time=y_va_df["time"].values)

    best = {
        "inner_val_cindex": -np.inf,
        "l1_ratio": None,
        "alpha": None,
        "alpha_index": None,
    }

    for l1 in l1_ratios:
        model = CoxnetSurvivalAnalysis(
            l1_ratio=float(l1),
            alphas=None,
            n_alphas=int(n_alphas),
            alpha_min_ratio=float(alpha_min_ratio),
            max_iter=int(max_iter),
        )
        model.fit(X_in, y_in)

        coef_path = model.coef_.T  # (n_alphas, n_features)
        for ai, alpha in enumerate(model.alphas_):
            beta = coef_path[ai]
            risk_va = X_va @ beta
            ci = cindex(y_va, risk_va)
            if ci > best["inner_val_cindex"]:
                best.update({"inner_val_cindex": float(ci), "l1_ratio": float(l1), "alpha": float(alpha), "alpha_index": int(ai)})

    best.update({
        "inner_val_ratio": float(val_ratio),
        "n_alphas": int(n_alphas),
        "alpha_min_ratio": float(alpha_min_ratio),
        "max_iter": int(max_iter),
    })
    return best


def fit_coxnet(X: np.ndarray, y_df: pd.DataFrame, l1_ratio: float, alpha: float, max_iter: int) -> CoxnetSurvivalAnalysis:
    y = Surv.from_arrays(event=y_df["event"].values, time=y_df["time"].values)
    model = CoxnetSurvivalAnalysis(l1_ratio=float(l1_ratio), alphas=[float(alpha)], max_iter=int(max_iter))
    model.fit(X, y)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_csv", required=True)
    ap.add_argument("--embed_csv", required=True)
    ap.add_argument("--split_json", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--time_col", default="Survival.time")
    ap.add_argument("--event_col", default="deadstatus.event")

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--scale", default="zscore", choices=["zscore", "none"])
    ap.add_argument("--pca_dim", type=int, default=0)
    ap.add_argument("--pca_whiten", type=int, default=0)

    ap.add_argument("--inner_val_ratio", type=float, default=0.2)
    ap.add_argument("--l1_ratios", default="0.05,0.1,0.3,0.5,0.8")
    ap.add_argument("--n_alphas", type=int, default=120)
    ap.add_argument("--alpha_min_ratio", type=float, default=1e-4)
    ap.add_argument("--max_iter", type=int, default=200000)

    ap.add_argument("--oof_folds", type=int, default=5, help="Generate OOF predictions on TRAIN with StratifiedKFold.")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    split = load_split_ids(args.split_json)

    y_all = read_clinical_targets(args.clinical_csv, args.time_col, args.event_col)
    emb = read_embeddings(args.embed_csv)

    common = y_all.index.intersection(emb.index)
    y_all = y_all.loc[common]
    emb = emb.loc[common]

    train_ids = [pid for pid in split["train"] if pid in common]
    test_ids = [pid for pid in split["test"] if pid in common]

    if len(train_ids) < 50 or len(test_ids) < 20:
        raise RuntimeError(f"Too few samples after intersection: train={len(train_ids)}, test={len(test_ids)}")

    Xtr_df = emb.loc[train_ids]
    Xte_df = emb.loc[test_ids]
    ytr_df = y_all.loc[train_ids]
    yte_df = y_all.loc[test_ids]

    # preprocess
    Xtr, Xte, pp_info = preprocess_embeddings(
        Xtr_df, Xte_df,
        scale=args.scale,
        pca_dim=args.pca_dim,
        pca_whiten=args.pca_whiten,
        seed=args.seed,
    )

    # hyperparam selection on inner val
    l1_list = [float(x) for x in args.l1_ratios.split(",") if x.strip() != ""]
    best = select_coxnet_alpha(
        Xtr, ytr_df,
        seed=args.seed,
        val_ratio=args.inner_val_ratio,
        l1_ratios=l1_list,
        n_alphas=args.n_alphas,
        alpha_min_ratio=args.alpha_min_ratio,
        max_iter=args.max_iter,
    )

    # OOF predictions on train (with chosen alpha)
    oof = np.zeros(len(train_ids), dtype=np.float32)
    skf = StratifiedKFold(n_splits=int(args.oof_folds), shuffle=True, random_state=int(args.seed))
    strat = ytr_df["event"].astype(int).values
    for tr_idx, va_idx in skf.split(np.arange(len(train_ids)), strat):
        m = fit_coxnet(Xtr[tr_idx], ytr_df.iloc[tr_idx], best["l1_ratio"], best["alpha"], best["max_iter"])
        oof[va_idx] = m.predict(Xtr[va_idx]).astype(np.float32)

    # Fit full model for test predictions
    full_model = fit_coxnet(Xtr, ytr_df, best["l1_ratio"], best["alpha"], best["max_iter"])
    r_test = full_model.predict(Xte).astype(np.float32)

    # decide sign by OOF c-index (more honest than in-sample)
    ytr = Surv.from_arrays(event=ytr_df["event"].values, time=ytr_df["time"].values)
    yte = Surv.from_arrays(event=yte_df["event"].values, time=yte_df["time"].values)

    ci_oof_pos = cindex(ytr, oof)
    ci_oof_neg = cindex(ytr, -oof)
    flipped = ci_oof_neg > ci_oof_pos
    if flipped:
        oof = -oof
        r_test = -r_test
    train_cindex_oof = cindex(ytr, oof)
    test_cindex = cindex(yte, r_test)

    # export preds
    df_tr = pd.DataFrame({"pid": train_ids, "time": ytr_df["time"].values, "event": ytr_df["event"].values.astype(int), "pred_risk": oof})
    df_te = pd.DataFrame({"pid": test_ids, "time": yte_df["time"].values, "event": yte_df["event"].values.astype(int), "pred_risk": r_test})

    df_tr.to_csv(os.path.join(args.out_dir, "deep_train_preds.csv"), index=False)
    df_te.to_csv(os.path.join(args.out_dir, "deep_test_preds.csv"), index=False)

    info = {
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
        "n_features_in": int(emb.shape[1]),
        "X_dim_after_preproc": int(Xtr.shape[1]),
        "train_cindex_oof": float(train_cindex_oof),
        "test_cindex": float(test_cindex),
        "flipped_sign": bool(flipped),
        "best_hparams": best,
        "preprocess": pp_info,
        "embed_csv": os.path.abspath(args.embed_csv),
        "split_json": os.path.abspath(args.split_json),
        "clinical_csv": os.path.abspath(args.clinical_csv),
        "time_col": args.time_col,
        "event_col": args.event_col,
        "oof_folds": int(args.oof_folds),
    }

    with open(os.path.join(args.out_dir, "deep_model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print("=== Deep-Embedding Survival (Coxnet) ===")
    print(f"TRAIN OOF c-index: {train_cindex_oof:.6f}")
    print(f"TEST c-index     : {test_cindex:.6f}")
    print(f"Saved: {args.out_dir}/deep_train_preds.csv")
    print(f"Saved: {args.out_dir}/deep_test_preds.csv")
    print(f"Saved: {args.out_dir}/deep_model_info.json")


if __name__ == "__main__":
    main()
