#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_prob_kpi_ablation.py

Run the 7-way ablation over 3 modalities (Tb, Cp, Df) for the *probability-KPI*
stacking evaluation script (r6_stack_prob_kpi.py).

What it does
------------
Given OOF risk predictions for multiple modalities, this wrapper runs
`r6_stack_prob_kpi.py` on every non-empty subset of modalities:

- 1-modality: Tb, Cp, Df
- 2-modality: Tb+Cp, Tb+Df, Cp+Df
- 3-modality: Tb+Cp+Df

It writes each run into a subfolder and creates a `ablation_summary.csv` that
you can drop into your MICCAI tables.

Usage (example)
--------------
python run_prob_kpi_ablation.py \
  --labels_csv /home/.../NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
  --splits_json /home/.../splits.json \
  --outdir /home/.../prob_kpi_ablation \
  --tau 730 \
  --inner_folds 3 \
  --inner_metric brier \
  --n_calib_bins 5 \
  --auto_sign 1 \
  --oof Tb:/home/.../Tb_oof.csv \
  --oof Cp:/home/.../Cp_oof.csv \
  --oof Df:/home/.../Df_oof.csv

Notes
-----
- This wrapper is intentionally boring: it just automates repeat runs and
  standardizes output.
- It assumes `r6_stack_prob_kpi.py` is available (default: same directory as
  this wrapper). You can override with --base_script.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_oof_arg(s: str) -> Tuple[str, Path]:
    if ":" not in s:
        raise ValueError(f"--oof must be TAG:/abs/path/file.csv, got: {s}")
    tag, path = s.split(":", 1)
    tag = tag.strip()
    path = Path(path).expanduser()
    if not tag:
        raise ValueError(f"Empty TAG in --oof {s}")
    return tag, path


def powerset_nonempty(tags: List[str]):
    for r in range(1, len(tags) + 1):
        for comb in itertools.combinations(tags, r):
            yield list(comb)


def combo_name(tags: List[str]) -> str:
    return "+".join(tags)


def run_one(
    python_exe: str,
    base_script: Path,
    labels_csv: Path,
    splits_json: Path,
    outdir: Path,
    oof_map: Dict[str, Path],
    tags: List[str],
    tau: float,
    inner_folds: int,
    inner_metric: str,
    alpha_grid: List[float],
    n_calib_bins: int,
    auto_sign: int,
    extra_args: List[str],
    dry_run: bool,
) -> int:
    cmd = [
        python_exe,
        str(base_script),
        "--labels_csv",
        str(labels_csv),
        "--splits_json",
        str(splits_json),
        "--outdir",
        str(outdir),
        "--tau",
        str(tau),
        "--inner_folds",
        str(inner_folds),
        "--inner_metric",
        str(inner_metric),
        "--n_calib_bins",
        str(n_calib_bins),
        "--auto_sign",
        str(auto_sign),
        "--alpha_grid",
        *[str(a) for a in alpha_grid],
    ]

    for t in tags:
        cmd += ["--oof", f"{t}:{oof_map[t]}"]

    cmd += extra_args

    print("\n" + "=" * 80)
    print(f"[RUN] {combo_name(tags)}")
    print(" " + " ".join(cmd))
    print("=" * 80)

    if dry_run:
        return 0

    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def read_summary(run_dir: Path) -> Dict:
    p = run_dir / "summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True, type=str)
    ap.add_argument("--splits_json", required=True, type=str)
    ap.add_argument("--outdir", required=True, type=str)

    ap.add_argument("--oof", action="append", required=True, help="TAG:/path/oof.csv")

    ap.add_argument("--tau", type=float, default=730.0)
    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--inner_metric", choices=["brier", "ibs"], default="brier")
    ap.add_argument("--alpha_grid", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    ap.add_argument("--n_calib_bins", type=int, default=5)
    ap.add_argument("--auto_sign", type=int, default=1)

    ap.add_argument("--base_script", type=str, default="",
                    help="Path to r6_stack_prob_kpi.py (default: sibling of this wrapper)")
    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable")

    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Extra args passed through to r6_stack_prob_kpi.py")

    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stop_on_error", action="store_true")

    args = ap.parse_args()

    labels_csv = Path(args.labels_csv)
    splits_json = Path(args.splits_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.base_script:
        base_script = Path(args.base_script)
    else:
        base_script = Path(__file__).resolve().with_name("r6_stack_prob_kpi.py")

    if not base_script.exists():
        raise FileNotFoundError(
            f"Cannot find base_script={base_script}. Either place r6_stack_prob_kpi.py next to this wrapper, "
            "or pass --base_script /abs/path/r6_stack_prob_kpi.py"
        )

    # Parse OOF map
    oof_map: Dict[str, Path] = {}
    for s in args.oof:
        tag, path = parse_oof_arg(s)
        if tag in oof_map:
            raise ValueError(f"Duplicate --oof tag: {tag}")
        oof_map[tag] = path

    tags_all = sorted(oof_map.keys())
    if len(tags_all) < 2:
        raise ValueError("Need at least 2 modalities for ablation; got: " + ",".join(tags_all))

    # If user provides exactly Tb/Cp/Df we will run 7-way; if more, we run all subsets.
    combos = list(powerset_nonempty(tags_all))

    run_manifest = {
        "labels_csv": str(labels_csv),
        "splits_json": str(splits_json),
        "tau": float(args.tau),
        "inner_folds": int(args.inner_folds),
        "inner_metric": str(args.inner_metric),
        "alpha_grid": [float(x) for x in args.alpha_grid],
        "n_calib_bins": int(args.n_calib_bins),
        "auto_sign": int(args.auto_sign),
        "oof": {k: str(v) for k, v in oof_map.items()},
        "base_script": str(base_script),
        "python": str(args.python),
        "extra": args.extra,
        "runs": [],
    }

    rows = []

    for comb in combos:
        name = combo_name(comb)
        run_dir = outdir / f"ablation_{name}"
        run_dir.mkdir(parents=True, exist_ok=True)

        rc = run_one(
            python_exe=args.python,
            base_script=base_script,
            labels_csv=labels_csv,
            splits_json=splits_json,
            outdir=run_dir,
            oof_map=oof_map,
            tags=comb,
            tau=args.tau,
            inner_folds=args.inner_folds,
            inner_metric=args.inner_metric,
            alpha_grid=args.alpha_grid,
            n_calib_bins=args.n_calib_bins,
            auto_sign=args.auto_sign,
            extra_args=args.extra,
            dry_run=args.dry_run,
        )

        run_manifest["runs"].append({"combo": comb, "outdir": str(run_dir), "returncode": rc})

        if rc != 0:
            print(f"[ERROR] {name} failed with return code {rc}")
            if args.stop_on_error:
                break
            continue

        summ = read_summary(run_dir)
        if not summ:
            print(f"[WARN] Missing summary.json for {name} (did r6_stack_prob_kpi.py write output?)")
            continue

        m = summ.get("metrics_mean", {})
        s = summ.get("metrics_std", {})

        row = {
            "combo": name,
            "models": "+".join(summ.get("models", comb)),
            "tau": summ.get("tau", args.tau),
            "inner_metric": summ.get("inner_metric", args.inner_metric),
            "brier_tau_mean": m.get("brier_tau"),
            "brier_tau_std": s.get("brier_tau"),
            "ibs_0_tau_mean": m.get("ibs_0_tau"),
            "ibs_0_tau_std": s.get("ibs_0_tau"),
            "auc_tau_mean": m.get("auc_tau"),
            "auc_tau_std": s.get("auc_tau"),
            "uno_c_mean": m.get("uno_c"),
            "uno_c_std": s.get("uno_c"),
            "harrell_c_mean": m.get("harrell_c"),
            "harrell_c_std": s.get("harrell_c"),
            "cal_in_the_large_mean": m.get("cal_in_the_large"),
            "outdir": str(run_dir),
        }
        rows.append(row)

    # Write manifest + summary table
    (outdir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(["brier_tau_mean", "ibs_0_tau_mean"], ascending=True, na_position="last")
        df.to_csv(outdir / "ablation_summary.csv", index=False)
        print("\n[OK] Wrote:")
        print(f"  - {outdir / 'ablation_summary.csv'}")
        print(f"  - {outdir / 'run_manifest.json'}")
        print("\nTop by Brier@tau (lower is better):")
        print(df[["combo", "brier_tau_mean", "ibs_0_tau_mean", "auc_tau_mean", "uno_c_mean"]].head(10).to_string(index=False))
    else:
        print("[WARN] No successful runs produced a summary.")


if __name__ == "__main__":
    main()
