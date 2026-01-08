#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os
import numpy as np
import pandas as pd

from sklearn.preprocessing import QuantileTransformer

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def harrell_cindex(time, event, risk):
    return float(concordance_index_censored(event.astype(bool), time.astype(float), risk.astype(float))[0])

def read_pred(path):
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # find id
    id_candidates = ["PatientID", "patient_id", "pid", "case_id"]
    id_col = None
    for c in df.columns:
        if c.lower() in [x.lower() for x in id_candidates]:
            id_col = c
            break
    if id_col is None:
        raise ValueError(f"Cannot find PatientID column in {path}. Columns={list(df.columns)}")

    # find risk column
    risk_candidates = ["risk", "pred", "prediction", "y_pred", "score"]
    risk_col = None
    for c in df.columns:
        if c.lower() in [x.lower() for x in risk_candidates]:
            risk_col = c
            break
    if risk_col is None:
        # fallback: last numeric col
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError(f"No numeric column found in {path}")
        risk_col = num_cols[-1]

    out = df[[id_col, risk_col]].copy()
    out.columns = ["PatientID", "risk"]
    out["PatientID"] = out["PatientID"].astype(str)
    out["risk"] = pd.to_numeric(out["risk"], errors="coerce")
    return out

def read_clinical(path):
    df = pd.read_csv(path)
    # columns
    def find_col(cands):
        lower = {c.lower(): c for c in df.columns}
        for x in cands:
            if x.lower() in lower:
                return lower[x.lower()]
        return None
    id_col = find_col(["PatientID", "patient_id", "pid", "case_id"])
    t_col  = find_col(["Survival.time", "time", "OS.time", "os_time", "survival_time"])
    e_col  = find_col(["deadstatus.event", "event", "OS.event", "os_event", "status"])
    if id_col is None or t_col is None or e_col is None:
        raise ValueError(f"Clinical CSV missing columns: id={id_col}, time={t_col}, event={e_col}")
    df = df.set_index(id_col)
    df.index = df.index.astype(str)
    df[t_col] = pd.to_numeric(df[t_col], errors="coerce")
    df[e_col] = pd.to_numeric(df[e_col], errors="coerce")
    return df, t_col, e_col

def read_split_json(path):
    with open(path, "r") as f:
        obj = json.load(f)
    for k_train, k_test in [("train","test"),("train_ids","test_ids"),("train_pids","test_pids"),("train_index","test_index")]:
        if k_train in obj and k_test in obj:
            train_ids = [str(x) for x in obj[k_train]]
            test_ids  = [str(x) for x in obj[k_test]]
            return train_ids, test_ids
    raise ValueError(f"Cannot find train/test keys in {path}. Keys={list(obj.keys())[:30]}")

def rankgauss_fit(train_risk, seed=42):
    qt = QuantileTransformer(
        n_quantiles=min(1000, len(train_risk)),
        output_distribution="normal",
        random_state=seed
    )
    train_t = qt.fit_transform(train_risk.reshape(-1,1)).ravel()
    def transform(x):
        return qt.transform(x.reshape(-1,1)).ravel()
    return train_t, transform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_csv", required=True)
    ap.add_argument("--split_json", required=True)

    ap.add_argument("--base_train_oof", required=True)
    ap.add_argument("--base_test", required=True)  # zip xlsx or csv

    ap.add_argument("--deep_train", required=True, help="csv paths, comma-separated")
    ap.add_argument("--deep_test", required=True, help="csv paths, comma-separated")

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--space", choices=["raw", "rankgauss"], default="rankgauss")
    ap.add_argument("--grid_step", type=float, default=0.01)
    ap.add_argument("--safety_eps", type=float, default=0.001)

    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args()
    ensure_dir(args.out_dir)

    clin_df, t_col, e_col = read_clinical(args.clinical_csv)
    train_ids, test_ids = read_split_json(args.split_json)

    # Load preds
    base_tr = read_pred(args.base_train_oof).set_index("PatientID").loc[train_ids]
    base_te = read_pred(args.base_test).set_index("PatientID").loc[test_ids]

    deep_train_paths = [p.strip() for p in args.deep_train.split(",") if p.strip()]
    deep_test_paths  = [p.strip() for p in args.deep_test.split(",") if p.strip()]
    if len(deep_train_paths) != len(deep_test_paths):
        raise ValueError("deep_train and deep_test must have same number of files.")

    deep_tr_list = []
    deep_te_list = []
    for p_tr, p_te in zip(deep_train_paths, deep_test_paths):
        dtr = read_pred(p_tr).set_index("PatientID").loc[train_ids]
        dte = read_pred(p_te).set_index("PatientID").loc[test_ids]
        deep_tr_list.append(dtr["risk"].values)
        deep_te_list.append(dte["risk"].values)

    deep_tr = np.vstack(deep_tr_list).T  # [n_train, n_deep]
    deep_te = np.vstack(deep_te_list).T  # [n_test, n_deep]

    # survival arrays
    y_tr_time = clin_df.loc[train_ids, t_col].values
    y_tr_event = clin_df.loc[train_ids, e_col].astype(int).astype(bool).values
    y_te_time = clin_df.loc[test_ids, t_col].values
    y_te_event = clin_df.loc[test_ids, e_col].astype(int).astype(bool).values

    # baseline c-index (OOF on train, truth on test)
    c_base_oof = harrell_cindex(y_tr_time, y_tr_event, base_tr["risk"].values)
    c_base_test = harrell_cindex(y_te_time, y_te_event, base_te["risk"].values)

    print(f"[baseline] train_oof_cindex={c_base_oof:.6f} | test_cindex(zip)={c_base_test:.6f}")

    # build deep ensemble = avg across deep models (in chosen space)
    if args.space == "rankgauss":
        # transform baseline
        base_tr_t, base_tf = rankgauss_fit(base_tr["risk"].values.astype(float), seed=args.seed)
        base_te_t = base_tf(base_te["risk"].values.astype(float))

        # transform each deep predictor using its train OOF distribution
        deep_tr_t_list = []
        deep_te_t_list = []
        for j in range(deep_tr.shape[1]):
            trj = deep_tr[:, j].astype(float)
            tej = deep_te[:, j].astype(float)
            trj_t, tf = rankgauss_fit(trj, seed=args.seed + 17 * (j+1))
            deep_tr_t_list.append(trj_t)
            deep_te_t_list.append(tf(tej))
        deep_tr_t = np.vstack(deep_tr_t_list).T
        deep_te_t = np.vstack(deep_te_t_list).T
    else:
        base_tr_t = base_tr["risk"].values.astype(float)
        base_te_t = base_te["risk"].values.astype(float)
        deep_tr_t = deep_tr.astype(float)
        deep_te_t = deep_te.astype(float)

    deep_tr_ens = deep_tr_t.mean(axis=1)
    deep_te_ens = deep_te_t.mean(axis=1)

    c_deep_oof = harrell_cindex(y_tr_time, y_tr_event, deep_tr_ens)
    c_deep_test = harrell_cindex(y_te_time, y_te_event, deep_te_ens)
    print(f"[deep-ens] train_oof_cindex={c_deep_oof:.6f} | test_cindex={c_deep_test:.6f}")

    # grid search w in [0,1]
    grid = np.arange(0.0, 1.0 + 1e-12, args.grid_step)
    best = (-1.0, 1.0)
    for w in grid:
        fused_tr = w * base_tr_t + (1.0 - w) * deep_tr_ens
        ci = harrell_cindex(y_tr_time, y_tr_event, fused_tr)
        if ci > best[0]:
            best = (ci, w)

    best_oof, best_w = best
    accepted = (best_oof >= c_base_oof + args.safety_eps)
    w_final = best_w if accepted else 1.0

    # test
    fused_te = w_final * base_te_t + (1.0 - w_final) * deep_te_ens
    c_fused_test = harrell_cindex(y_te_time, y_te_event, fused_te)

    # bootstrap on test
    rng = np.random.default_rng(args.seed)
    boot = []
    n = len(test_ids)
    for _ in range(args.bootstrap):
        idx = rng.integers(0, n, size=n)
        boot.append(harrell_cindex(y_te_time[idx], y_te_event[idx], fused_te[idx]))
    boot = np.array(boot, dtype=float)
    boot_mean = float(boot.mean())
    boot_std = float(boot.std(ddof=1))
    ci95 = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    print("=== OOF Safe Fusion Summary ===")
    print(f"base_oof_cindex={c_base_oof:.6f}")
    print(f"best_oof_cindex={best_oof:.6f} at w={best_w:.2f} | accepted={accepted} | w_final={w_final:.2f}")
    print(f"test_cindex_fused={c_fused_test:.6f} (baseline_zip={c_base_test:.6f})")
    print(f"boot_mean±std={boot_mean:.6f}±{boot_std:.6f} CI95={ci95}")

    # save
    pd.DataFrame({"PatientID": test_ids, "risk_fused": fused_te}).to_csv(
        os.path.join(args.out_dir, "fused_test_preds.csv"), index=False
    )
    summary = {
        "base_train_oof_cindex": c_base_oof,
        "base_test_cindex_zip": c_base_test,
        "deep_train_oof_cindex": c_deep_oof,
        "deep_test_cindex": c_deep_test,
        "best_train_oof_cindex": best_oof,
        "best_w": float(best_w),
        "accepted": bool(accepted),
        "w_final": float(w_final),
        "test_cindex_fused": c_fused_test,
        "bootstrap": {
            "n": args.bootstrap,
            "mean": boot_mean,
            "std": boot_std,
            "ci95_pct": ci95
        },
        "space": args.space,
        "grid_step": args.grid_step,
        "safety_eps": args.safety_eps,
        "deep_files": {"train": deep_train_paths, "test": deep_test_paths},
        "base_files": {"train_oof": args.base_train_oof, "test": args.base_test},
    }
    with open(os.path.join(args.out_dir, "fusion_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Saved: {args.out_dir}/fusion_summary.json")


if __name__ == "__main__":
    main()
