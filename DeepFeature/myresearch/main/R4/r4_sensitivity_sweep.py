#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List

def read_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())

def run_one(cmd: List[str], cwd: Path) -> int:
    print("\n" + "=" * 120)
    print("[RUN]", " ".join(cmd))
    print("=" * 120)
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r4_script", required=True, help="Path to run_R4_coxnet_solid_v2.py")
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--splits_json", required=True)
    ap.add_argument("--base_out_dir", required=True, help="Where to create sweep subfolders")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--n_alphas", type=int, default=400)
    ap.add_argument("--alpha_min_ratio", default="1e-6")
    ap.add_argument("--l1_ratios", nargs="+", type=float, default=[0.0, 0.1, 0.5, 0.9])
    ap.add_argument("--time_col", default="Survival.time")
    ap.add_argument("--event_col", default="deadstatus.event")
    ap.add_argument("--use_corr_filter", action="store_true", default=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    r4_script = Path(args.r4_script).resolve()
    base_out = Path(args.base_out_dir).resolve()
    base_out.mkdir(parents=True, exist_ok=True)

    # --- Sensitivity grid (small but meaningful) ---
    alpha_select_grid = ["min", "1se"]
    corr_thresh_grid = [0.90, 0.95]
    tau_q_grid = [0.80, 0.90, 0.95]

    rows = []
    configs = list(itertools.product(alpha_select_grid, corr_thresh_grid, tau_q_grid))
    print(f"[INFO] total runs = {len(configs)}")

    for alpha_select, corr_thresh, tau_q in configs:
        tag = f"alpha_{alpha_select}__corr_{corr_thresh:.2f}__tauq_{tau_q:.2f}__seed_{args.seed}"
        out_dir = base_out / tag

        cmd = [
            "python", str(r4_script),
            "--features_csv", args.features_csv,
            "--labels_csv", args.labels_csv,
            "--splits_json", args.splits_json,
            "--out_dir", str(out_dir),
            "--mode", "R4",
            "--seed", str(args.seed),
            "--alpha_min_ratio", str(args.alpha_min_ratio),
            "--n_alphas", str(args.n_alphas),
            "--alpha_select", alpha_select,
            "--inner_folds", str(args.inner_folds),
            "--tau_q", str(tau_q),
            "--time_col", args.time_col,
            "--event_col", args.event_col,
            "--corr_thresh", str(corr_thresh),
        ]

        # l1 ratios
        cmd += ["--l1_ratios"] + [str(x) for x in args.l1_ratios]

        # corr filter
        if args.use_corr_filter:
            cmd += ["--use_corr_filter"]

        if args.dry_run:
            print("[DRY RUN]", " ".join(cmd))
            continue

        rc = run_one(cmd, cwd=Path.cwd())
        if rc != 0:
            print(f"[WARN] run failed: {tag} (rc={rc})")
            rows.append({
                "tag": tag,
                "alpha_select": alpha_select,
                "corr_thresh": corr_thresh,
                "tau_q": tau_q,
                "seed": args.seed,
                "status": "FAIL",
            })
            continue

        overall_path = out_dir / "overall.json"
        if not overall_path.exists():
            print(f"[WARN] missing overall.json for {tag}")
            rows.append({
                "tag": tag,
                "alpha_select": alpha_select,
                "corr_thresh": corr_thresh,
                "tau_q": tau_q,
                "seed": args.seed,
                "status": "MISSING_OVERALL",
            })
            continue

        overall = read_json(overall_path)

        # These keys might differ depending on your script; adjust if needed.
        har = overall.get("har_mean", None)
        uno = overall.get("uno_mean", None)
        har_std = overall.get("har_std", None)
        uno_std = overall.get("uno_std", None)

        rows.append({
            "tag": tag,
            "alpha_select": alpha_select,
            "corr_thresh": corr_thresh,
            "tau_q": tau_q,
            "seed": args.seed,
            "status": "OK",
            "har_mean": har,
            "har_std": har_std,
            "uno_mean": uno,
            "uno_std": uno_std,
        })

    # Write summary CSV (no pandas dependency)
    out_csv = base_out / "r4_sensitivity_sweep_summary.csv"
    cols = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    print(f"\n[OK] wrote: {out_csv}")

    # Quick console summary
    ok = [r for r in rows if r.get("status") == "OK" and r.get("har_mean") is not None]
    if ok:
        best = max(ok, key=lambda r: float(r["har_mean"]))
        worst = min(ok, key=lambda r: float(r["har_mean"]))
        print(f"[SUMMARY] OK runs = {len(ok)}/{len(rows)}")
        print(f"[BEST]  har={best['har_mean']}  ({best['tag']})")
        print(f"[WORST] har={worst['har_mean']} ({worst['tag']})")
    else:
        print("[SUMMARY] no successful runs parsed")

if __name__ == "__main__":
    main()
