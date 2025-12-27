#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, glob, json, os
from collections import defaultdict
import numpy as np
import pandas as pd

def read_json(p):
    with open(p, "r") as f:
        return json.load(f)

def extract_fold_coefs(coef_json_path: str, which: str = "coef_min"):
    """
    Returns list of (feature_name, coef_value).
    Handles two possible coef.json structures:
    1. New format: {"alpha": ..., "coef": {"idx": [...], "feature": [...], "coef": [...]}}
    2. Old format: {"feature_names": [...], "coef_min": {"idx": [...], "coef": [...]}}
    """
    d = read_json(coef_json_path)
    out = []
    
    # Try new format first (coef directly in root)
    if "coef" in d and isinstance(d["coef"], dict):
        coef_block = d["coef"]
        if "feature" in coef_block and "coef" in coef_block:
            # New format: feature names are directly in coef.feature
            features = coef_block.get("feature", [])
            coefs = coef_block.get("coef", [])
            for feat, c in zip(features, coefs):
                out.append((str(feat), float(c)))
            return out
    
    # Try old format (coef_min or coef_1se)
    feat_names = d.get("feature_names", [])
    block = d.get(which, {})
    if not block:
        # If which is not found, try "coef" as fallback
        block = d.get("coef", {})
    
    idx = block.get("idx", [])
    coef = block.get("coef", [])
    for i, c in zip(idx, coef):
        try:
            name = feat_names[int(i)]
        except (IndexError, KeyError, ValueError, TypeError):
            name = f"feat_{i}"
        out.append((name, float(c)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_glob", required=True,
                    help='Glob to R4 run dirs, e.g. "/path/R4_seed_sweep_v2_*" or a single run dir')
    ap.add_argument("--which", default="coef_min", choices=["coef_min", "coef_1se"])
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    run_dirs = sorted(glob.glob(args.runs_glob))
    if not run_dirs:
        raise SystemExit("No run dirs matched.")

    # feature -> list of coefs across all (run, fold) where selected
    coef_pool = defaultdict(list)
    # feature -> count selected
    sel_count = defaultdict(int)
    # total fold slots
    total_slots = 0

    for rd in run_dirs:
        fold_dirs = sorted(glob.glob(os.path.join(rd, "fold_*")))
        for fd in fold_dirs:
            total_slots += 1
            cj = os.path.join(fd, "coef.json")
            if not os.path.exists(cj):
                continue
            pairs = extract_fold_coefs(cj, which=args.which)
            for name, c in pairs:
                sel_count[name] += 1
                coef_pool[name].append(c)

    rows = []
    for name, cnt in sel_count.items():
        vals = np.array(coef_pool[name], dtype=float)
        pos_frac = float(np.mean(vals > 0)) if len(vals) else float("nan")
        rows.append({
            "feature": name,
            "selected_count": int(cnt),
            "selected_frac": float(cnt / max(total_slots, 1)),
            "coef_mean": float(np.mean(vals)) if len(vals) else float("nan"),
            "coef_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "pos_frac": pos_frac,
        })

    if len(rows) == 0:
        raise SystemExit(f"No features found in any coef.json files. Check that files exist and have correct format.")

    df = pd.DataFrame(rows)
    if "selected_count" in df.columns:
        df = df.sort_values(
            ["selected_count", "selected_frac"], ascending=False
        )
    else:
        print("[WARN] selected_count column not found, skipping sort")
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] total_slots={total_slots} | wrote {args.out_csv}")
    print(df.head(25).to_string(index=False))

if __name__ == "__main__":
    main()
