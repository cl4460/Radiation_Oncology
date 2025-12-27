#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_overall_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


@dataclass
class PermStats:
    metric: str
    obs: float
    null_mean: float
    null_std: float
    null_min: float
    null_max: float
    p_one_sided: float
    z_effect: float
    n_perm: int


def empirical_pvalue_one_sided(null_vals: np.ndarray, obs: float) -> float:
    """
    One-sided p = P(null >= obs), with +1 correction to avoid p=0.
    Reference idea: permutation p-values should never be 0.
    """
    null_vals = null_vals[np.isfinite(null_vals)]
    m = len(null_vals)
    if m == 0:
        return float("nan")
    b = int(np.sum(null_vals >= obs))
    return (b + 1.0) / (m + 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs_overall", required=True,
                    help="Path to observed run overall.json, e.g. R4_solid_v2_alpha400/overall.json")
    ap.add_argument("--perm_glob", required=True,
                    help='Glob to permutation overall.json files, e.g. "/path/R0_tb_perm_v2_*/overall.json"')
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--metrics", nargs="+", default=["har_mean", "uno_mean"],
                    help="Which metrics in overall.json to evaluate.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    obs = read_overall_json(args.obs_overall)

    # Try direct glob first
    perm_paths = sorted(glob.glob(args.perm_glob))
    
    # If no matches, try searching in common subdirectory patterns
    if len(perm_paths) == 0:
        # Extract base directory from glob pattern
        glob_dir = os.path.dirname(args.perm_glob) if os.path.dirname(args.perm_glob) else "."
        glob_pattern = os.path.basename(args.perm_glob)
        
        # Try subdirectory: R0_tb_perm_v2/R0_tb_perm_v2_*/overall.json
        alt_patterns = [
            os.path.join(glob_dir, "R0_tb_perm_v2", "R0_tb_perm_v2_*", "overall.json"),
            os.path.join(glob_dir, "*", "R0_tb_perm_v2_*", "overall.json"),  # recursive search
        ]
        
        for alt in alt_patterns:
            found = sorted(glob.glob(alt))
            if len(found) > 0:
                perm_paths = found
                print(f"[INFO] Found {len(perm_paths)} files using pattern: {alt}")
                break
    
    if len(perm_paths) == 0:
        raise SystemExit(f"No permutation overall.json matched glob: {args.perm_glob}\n"
                        f"Please check the path. Expected pattern like: /path/to/R0_tb_perm_v2/R0_tb_perm_v2_*/overall.json")

    rows = []
    for p in perm_paths:
        try:
            d = read_overall_json(p)
            row = {"path": p}
            for k in args.metrics:
                row[k] = safe_float(d.get(k, np.nan))
            rows.append(row)
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "perm_values.csv"), index=False)

    report: Dict[str, Any] = {
        "obs_overall": args.obs_overall,
        "perm_glob": args.perm_glob,
        "n_perm_total": int(len(perm_paths)),
        "n_perm_loaded": int(len(df)),
        "results": {}
    }

    for metric in args.metrics:
        null_vals = df[metric].to_numpy(dtype=float)
        obs_val = safe_float(obs.get(metric, np.nan))

        ps = PermStats(
            metric=metric,
            obs=obs_val,
            null_mean=float(np.nanmean(null_vals)),
            null_std=float(np.nanstd(null_vals, ddof=1)),
            null_min=float(np.nanmin(null_vals)),
            null_max=float(np.nanmax(null_vals)),
            p_one_sided=float(empirical_pvalue_one_sided(null_vals, obs_val)),
            z_effect=float((obs_val - np.nanmean(null_vals)) / (np.nanstd(null_vals, ddof=1) + 1e-12)),
            n_perm=int(np.sum(np.isfinite(null_vals))),
        )

        report["results"][metric] = ps.__dict__

        # Plot
        finite = null_vals[np.isfinite(null_vals)]
        plt.figure()
        plt.hist(finite, bins=30)
        plt.axvline(obs_val, linewidth=2)
        plt.title(f"Permutation null for {metric} (n={len(finite)})")
        plt.xlabel(metric)
        plt.ylabel("count")
        out_png = os.path.join(args.out_dir, f"perm_hist_{metric}.png")
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()

    out_json = os.path.join(args.out_dir, "perm_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] wrote {out_json}")
    print(json.dumps(report["results"], indent=2))


if __name__ == "__main__":
    main()
