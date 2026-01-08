#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
import pandas as pd

def pick_col(df, candidates):
    low = {c.lower(): c for c in df.columns}
    for key in candidates:
        key = key.lower()
        for c in df.columns:
            if key in c.lower():
                return c
    raise ValueError(f"Cannot find any of columns containing {candidates}. Available={list(df.columns)}")

def summarize_runs(run_glob: str, out_csv: str):
    files = sorted(glob.glob(run_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {run_glob}")

    rows = []
    for f in files:
        df = pd.read_csv(f)
        run_dir = os.path.basename(os.path.dirname(f))

        # Try to infer columns by substring (robust to your column naming)
        col_har_min = pick_col(df, ["har(min)", "har_min", "harrell_min", "harrell(min)"])
        col_uno_min = pick_col(df, ["uno(min)", "uno_min"])
        col_har_1se = pick_col(df, ["har(1se)", "har_1se", "harrell_1se"])
        col_uno_1se = pick_col(df, ["uno(1se)", "uno_1se"])
        # nnz might be absent in some summaries
        col_nnz_1se = None
        for cand in ["nnz1se", "nnz_1se", "nonzero_1se"]:
            try:
                col_nnz_1se = pick_col(df, [cand])
                break
            except Exception:
                pass

        rec = {
            "run": run_dir,
            "n_folds": len(df),
            "har_min_mean": float(df[col_har_min].mean()),
            "uno_min_mean": float(df[col_uno_min].mean()),
            "har_1se_mean": float(df[col_har_1se].mean()),
            "uno_1se_mean": float(df[col_uno_1se].mean()),
        }
        if col_nnz_1se is not None:
            rec["nnz_1se_mean"] = float(df[col_nnz_1se].mean())
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("run")
    out.to_csv(out_csv, index=False)
    print(f"[OK] wrote {out_csv} with {len(out)} runs")

    # Print quick stats
    for k in ["har_min_mean", "uno_min_mean", "har_1se_mean", "uno_1se_mean"]:
        v = out[k].to_numpy()
        print(f"{k}: mean={v.mean():.4f} std={v.std(ddof=1):.4f} "
              f"p10={np.percentile(v,10):.4f} p50={np.percentile(v,50):.4f} p90={np.percentile(v,90):.4f}")

    return out

def empirical_pvalue(perm_vals, obs):
    perm_vals = np.asarray(perm_vals, dtype=float)
    # one-sided: perm >= obs
    return (1.0 + np.sum(perm_vals >= obs)) / (1.0 + len(perm_vals))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm_root", required=True, help="e.g. /.../R3 where folders R0_perm_* live")
    ap.add_argument("--perm_pattern", default="R0_perm_series/R0_perm_*/summary_by_fold.csv")
    ap.add_argument("--obs_summary", default="", help="optional: path to R3_coxnet/summary_by_fold.csv")
    ap.add_argument("--out_csv", default="perm_summary.csv")
    args = ap.parse_args()

    run_glob = os.path.join(args.perm_root, args.perm_pattern)
    out = summarize_runs(run_glob, os.path.join(args.perm_root, args.out_csv))

    if args.obs_summary:
        obs_df = pd.read_csv(args.obs_summary)
        col_har_min = pick_col(obs_df, ["har(min)", "har_min", "harrell_min", "harrell(min)"])
        obs = float(obs_df[col_har_min].mean())
        p = empirical_pvalue(out["har_min_mean"].to_numpy(), obs)
        print(f"[Permutation p-value] obs_har_min_mean={obs:.4f}, p(one-sided)={p:.4g}")

if __name__ == "__main__":
    main()
