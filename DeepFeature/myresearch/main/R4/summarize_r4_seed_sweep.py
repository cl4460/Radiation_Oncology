#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, glob, json
import numpy as np
import pandas as pd

def read_json(p):
    with open(p, "r") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob_overall", required=True,
                    help='Glob like "/path/R4_seed_sweep_v2_*/overall.json"')
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob_overall))
    if not paths:
        raise SystemExit("No overall.json matched.")

    rows = []
    for p in paths:
        d = read_json(p)
        rows.append({
            "run_dir": p.rsplit("/", 1)[0],
            "har_mean": float(d.get("har_mean", np.nan)),
            "uno_mean": float(d.get("uno_mean", np.nan)),
            "har_std": float(d.get("har_std", np.nan)),
            "uno_std": float(d.get("uno_std", np.nan)),
            "seed": int(d.get("seed", -1)) if "seed" in d else -1,
        })

    df = pd.DataFrame(rows).sort_values(["har_mean"], ascending=False)
    df.to_csv(args.out_csv, index=False)

    def summ(x):
        x = x[np.isfinite(x)]
        return {
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
            "min": float(np.min(x)) if len(x) else float("nan"),
            "p25": float(np.quantile(x, 0.25)) if len(x) else float("nan"),
            "median": float(np.median(x)) if len(x) else float("nan"),
            "p75": float(np.quantile(x, 0.75)) if len(x) else float("nan"),
            "max": float(np.max(x)) if len(x) else float("nan"),
        }

    print("HAR:", summ(df["har_mean"].to_numpy(float)))
    print("UNO:", summ(df["uno_mean"].to_numpy(float)))
    print(f"[OK] wrote {args.out_csv}")

if __name__ == "__main__":
    main()
