#!/usr/bin/env python3
import argparse, json, os, glob
from pathlib import Path

import numpy as np
import pandas as pd


def _read_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r4_out_dir", required=True, help="R4 run output dir containing fold_*/ and overall.json")
    ap.add_argument("--perm_report", default=None, help="perm_report.json (optional)")
    ap.add_argument("--seed_sweep_csv", default=None, help="r4_seed_sweep_summary.csv (optional)")
    ap.add_argument("--feature_stability_csv", default=None, help="feature_stability_seed_sweep.csv (optional)")
    ap.add_argument("--out_md", default="R4_modelcard.md")
    ap.add_argument("--out_qc", default="R4_qc.json")
    args = ap.parse_args()

    r4_dir = Path(args.r4_out_dir)
    assert r4_dir.exists(), f"Not found: {r4_dir}"

    # -------- Load overall --------
    overall_path = r4_dir / "overall.json"
    overall = _read_json(overall_path) if overall_path.exists() else {}

    # -------- Load folds --------
    fold_rows = []
    coef_rows = []
    flags = []

    fold_dirs = sorted([Path(p) for p in glob.glob(str(r4_dir / "fold_*")) if Path(p).is_dir()])
    if not fold_dirs:
        flags.append(f"[FATAL] No fold_* dirs found under {r4_dir}")

    for fd in fold_dirs:
        mpath = fd / "metrics.json"
        cpath = fd / "coef.json"
        if not mpath.exists():
            flags.append(f"[WARN] Missing metrics.json: {mpath}")
            continue
        m = _read_json(mpath)

        # core sanity checks
        nnz = m.get("nnz_min", m.get("nnz", None))
        har = m.get("harrell_min", m.get("har", None))
        uno = m.get("uno_min", m.get("uno", None))

        # collapse detection
        if nnz == 0 or _safe_float(har) == 0.5:
            flags.append(f"[RED FLAG] {fd.name}: possible collapse (nnz={nnz}, har={har})")

        fold_rows.append({
            "fold": fd.name,
            "n_train": m.get("n_train"),
            "n_val": m.get("n_val"),
            "tau": m.get("tau"),
            "l1_ratio": m.get("l1_ratio"),
            "alpha_min": m.get("alpha_min"),
            "alpha_1se": m.get("alpha_1se"),
            "har_min": har,
            "uno_min": uno,
            "har_1se": m.get("harrell_1se"),
            "uno_1se": m.get("uno_1se"),
            "nnz_min": m.get("nnz_min"),
            "nnz_1se": m.get("nnz_1se"),
            "n_features_after_preproc": m.get("n_features_after_preproc"),
            "use_corr_filter": m.get("use_corr_filter"),
            "corr_thresh": m.get("corr_thresh"),
        })

        if cpath.exists():
            c = _read_json(cpath)
            feat_names = c.get("feature_names", [])
            coef_min = c.get("coef_min", {})
            idx = coef_min.get("idx", [])
            val = coef_min.get("coef", [])
            for i, v in zip(idx, val):
                fname = feat_names[i] if (0 <= i < len(feat_names)) else f"idx_{i}"
                coef_rows.append({
                    "fold": fd.name,
                    "feature": fname,
                    "coef": float(v),
                })

    folds_df = pd.DataFrame(fold_rows)
    coefs_df = pd.DataFrame(coef_rows)

    # -------- Permutation report --------
    perm = None
    if args.perm_report:
        perm_path = Path(args.perm_report)
        if perm_path.exists():
            perm = _read_json(perm_path)
        else:
            flags.append(f"[WARN] perm_report not found: {perm_path}")

    # -------- Seed sweep --------
    seed_sweep = None
    if args.seed_sweep_csv:
        p = Path(args.seed_sweep_csv)
        if p.exists():
            ss = pd.read_csv(p)
            seed_sweep = {
                "n_runs": int(len(ss)),
                "har_mean_avg": float(ss["har_mean"].mean()),
                "har_mean_std_across_seeds": float(ss["har_mean"].std(ddof=1)),
                "uno_mean_avg": float(ss["uno_mean"].mean()),
                "uno_mean_std_across_seeds": float(ss["uno_mean"].std(ddof=1)),
                "har_mean_min": float(ss["har_mean"].min()),
                "har_mean_max": float(ss["har_mean"].max()),
                "uno_mean_min": float(ss["uno_mean"].min()),
                "uno_mean_max": float(ss["uno_mean"].max()),
            }
        else:
            flags.append(f"[WARN] seed_sweep_csv not found: {p}")

    # -------- Feature stability --------
    feat_stab_top = None
    if args.feature_stability_csv:
        p = Path(args.feature_stability_csv)
        if p.exists():
            fs = pd.read_csv(p).sort_values("selected_frac", ascending=False)
            feat_stab_top = fs.head(15).copy()
            feat_stab_top["selected_pct"] = (feat_stab_top["selected_frac"] * 100).round(1)
            feat_stab_top = feat_stab_top[["feature", "selected_pct", "pos_frac", "coef_mean", "coef_std", "selected_count"]]
        else:
            flags.append(f"[WARN] feature_stability_csv not found: {p}")

    # -------- Compose markdown --------
    lines = []
    lines.append("# R4 Model Card / QC Report\n")

    lines.append("## 1) Overall\n")
    if overall:
        lines.append("```json\n" + json.dumps(overall, indent=2) + "\n```\n")
    else:
        lines.append("- (overall.json not found)\n")

    if not folds_df.empty:
        lines.append("## 2) Fold metrics\n")
        lines.append(folds_df.to_markdown(index=False))
        lines.append("\n")

        lines.append("### Quick checks\n")
        lines.append(f"- Folds: {len(folds_df)}\n")
        lines.append(f"- Any nnz==0? {bool((folds_df['nnz_min'] == 0).any())}\n")
        lines.append(f"- Any har_min==0.5? {bool((folds_df['har_min'] == 0.5).any())}\n")
        lines.append("\n")

    if perm is not None:
        lines.append("## 3) Permutation sanity check\n")
        lines.append("```json\n" + json.dumps(perm, indent=2) + "\n```\n")

    if seed_sweep is not None:
        lines.append("## 4) Seed sweep stability\n")
        lines.append("```json\n" + json.dumps(seed_sweep, indent=2) + "\n```\n")

    if feat_stab_top is not None:
        lines.append("## 5) Top stable features (from seed sweep)\n")
        lines.append(feat_stab_top.to_markdown(index=False))
        lines.append("\n")

    if not coefs_df.empty:
        lines.append("## 6) Coef aggregation (min) across folds\n")
        agg = coefs_df.groupby("feature")["coef"].agg(["count", "mean", "std"]).sort_values("count", ascending=False).head(25)
        lines.append(agg.to_markdown())
        lines.append("\n")

    lines.append("## 7) Flags\n")
    if flags:
        for f in flags:
            lines.append(f"- {f}")
    else:
        lines.append("- None")
    lines.append("\n")

    Path(args.out_md).write_text("\n".join(lines))
    qc = {
        "r4_out_dir": str(r4_dir),
        "n_folds_found": int(len(fold_dirs)),
        "flags": flags,
        "overall": overall,
        "fold_summary": folds_df.to_dict(orient="records") if not folds_df.empty else [],
        "perm_report": perm,
        "seed_sweep": seed_sweep,
    }
    Path(args.out_qc).write_text(json.dumps(qc, indent=2))
    print(f"[OK] wrote {args.out_md} and {args.out_qc}")


if __name__ == "__main__":
    main()
