#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re, glob
import pandas as pd

def pick(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="/home/lichengze/Research/auc/horizon_sweep_v3")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = []
    paths = glob.glob(os.path.join(args.root, "h*", "eval", "eval_report.json"))
    if not paths:
        raise SystemExit(f"No eval_report.json found under {args.root}/h*/eval/")

    for p in sorted(paths):
        with open(p, "r") as f:
            rep = json.load(f)

        m = re.search(r"/h(\d+)/eval/eval_report\.json$", p)
        horizon = float(m.group(1)) if m else rep.get("horizon", None)

        # Check nested structure: metrics.* or top-level
        metrics = rep.get("metrics", rep)
        
        row = {
            "horizon_days": horizon,
            "cindex": pick(metrics, ["cindex_test", "cindex", "cindex_internal", "cindex@t"]),
            "auc": pick(metrics, ["auc2y", "auc", "auc_dynamic", "auc@t"]),
            "brier": pick(metrics, ["brier2y", "brier", "brier_cal", "brier@t"]),
            "brier_skill": pick(metrics, ["brier2y_skill", "brier_skill", "skill", "brier_skill_score"]),
            "path": p,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("horizon_days")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
