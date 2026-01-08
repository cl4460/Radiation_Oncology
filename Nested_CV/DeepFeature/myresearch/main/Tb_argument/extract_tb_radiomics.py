#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract expanded radiomics (Tb) with PyRadiomics.

Design goals:
- No label usage -> no leakage
- Conservative config aligned with common LUNG1 practice:
  * 1x1x1 mm resampling
  * ROI resegmentation [-600, 400]
  * Original: fixed bin width (25 HU)
  * Filtered (LoG + Wavelet): fixed bin count (32)
- Robust logging + optional parallelism

Outputs (in --outdir):
- Tb_features.csv            (PatientID + radiomics features)
- Tb_diagnostics.csv         (PatientID + diagnostics_* fields)
- Tb_errors.csv              (PatientID + error message)
- extraction_metadata.json   (configs, versions, counts)
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

try:
    from radiomics import featureextractor
    import radiomics
    HAS_PYRADIOMICS = True
except Exception as e:
    HAS_PYRADIOMICS = False
    _PYR_ERR = str(e)


def _load_manifest(path: Path, id_col: str, image_col: str, mask_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in [id_col, image_col, mask_col]:
        if c not in df.columns:
            raise ValueError(f"Manifest missing required column '{c}'. "
                             f"Found columns: {list(df.columns)}")
    df = df[[id_col, image_col, mask_col]].copy()
    df[id_col] = df[id_col].astype(str)

    # Drop duplicates (keep first) to avoid repeated extraction
    if df[id_col].duplicated().any():
        df = df.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)
    return df


def _extract_one_case(
    pid: str,
    image_path: str,
    mask_path: str,
    yaml_original: Optional[str],
    yaml_filtered: Optional[str],
    label: int,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns: (pid, features_dict, diagnostics_dict, error_str)
    """
    try:
        img_p = Path(image_path)
        msk_p = Path(mask_path)
        if not img_p.exists():
            raise FileNotFoundError(f"Image not found: {img_p}")
        if not msk_p.exists():
            raise FileNotFoundError(f"Mask not found: {msk_p}")

        features: Dict[str, Any] = {}
        diagnostics: Dict[str, Any] = {}

        def run_yaml(yaml_path: str) -> Dict[str, Any]:
            ext = featureextractor.RadiomicsFeatureExtractor(yaml_path)
            # force label from args (even if yaml has it)
            ext.settings["label"] = int(label)
            return ext.execute(str(img_p), str(msk_p))

        # Original (binWidth=25)
        if yaml_original is not None:
            res_o = run_yaml(yaml_original)
            for k, v in res_o.items():
                if str(k).startswith("diagnostics_"):
                    diagnostics[k] = v
                else:
                    features[k] = v

        # Filtered (binCount=32)
        if yaml_filtered is not None:
            res_f = run_yaml(yaml_filtered)
            for k, v in res_f.items():
                if str(k).startswith("diagnostics_"):
                    # keep original diagnostics if already present
                    diagnostics.setdefault(k, v)
                else:
                    features[k] = v

        if len(features) == 0:
            raise RuntimeError("No features extracted (empty result). Check yaml / mask label / inputs.")

        return pid, features, diagnostics, None

    except Exception as e:
        return pid, None, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description="Extract Tb radiomics with PyRadiomics (Original + LoG + Wavelet).")
    ap.add_argument("--manifest_csv", type=str, required=True,
                    help="CSV with PatientID, image_path, mask_path.")
    ap.add_argument("--id_col", type=str, default="PatientID")
    ap.add_argument("--image_col", type=str, default="image_path")
    ap.add_argument("--mask_col", type=str, default="mask_path")

    ap.add_argument("--yaml_original", type=str, default=None,
                    help="PyRadiomics yaml for Original image (recommended: binWidth=25).")
    ap.add_argument("--yaml_filtered", type=str, default=None,
                    help="PyRadiomics yaml for filtered images (LoG/Wavelet; recommended: binCount=32).")

    ap.add_argument("--label", type=int, default=1, help="Mask label value for ROI.")
    ap.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs. Requires joblib. Use 1 for sequential.")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory.")
    ap.add_argument("--out_csv", type=str, default="Tb_features.csv", help="Output features CSV filename.")
    args = ap.parse_args()

    if not HAS_PYRADIOMICS:
        raise RuntimeError(
            "PyRadiomics not available. Install with: pip install pyradiomics SimpleITK\n"
            f"Import error: {_PYR_ERR}"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(Path(args.manifest_csv), args.id_col, args.image_col, args.mask_col)
    print(f"[Tb-EXTRACT] n_cases={len(manifest)} | n_jobs={args.n_jobs} | label={args.label}")
    print(f"[Tb-EXTRACT] yaml_original={args.yaml_original}")
    print(f"[Tb-EXTRACT] yaml_filtered={args.yaml_filtered}")

    rows = manifest.to_dict(orient="records")

    # Run extraction
    if args.n_jobs > 1:
        if not HAS_JOBLIB:
            print("[WARNING] joblib not found. Falling back to sequential.")
            args.n_jobs = 1

    if args.n_jobs > 1:
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(_extract_one_case)(
                pid=str(r[args.id_col]),
                image_path=str(r[args.image_col]),
                mask_path=str(r[args.mask_col]),
                yaml_original=args.yaml_original,
                yaml_filtered=args.yaml_filtered,
                label=int(args.label),
            ) for r in rows
        )
    else:
        results = [
            _extract_one_case(
                pid=str(r[args.id_col]),
                image_path=str(r[args.image_col]),
                mask_path=str(r[args.mask_col]),
                yaml_original=args.yaml_original,
                yaml_filtered=args.yaml_filtered,
                label=int(args.label),
            ) for r in rows
        ]

    pids: List[str] = []
    feat_records: List[Dict[str, Any]] = []
    diag_records: List[Dict[str, Any]] = []
    err_records: List[Dict[str, Any]] = []

    for pid, feat, diag, err in results:
        pids.append(pid)
        if err is not None:
            err_records.append({args.id_col: pid, "error": err})
            feat_records.append({})
            diag_records.append({})
        else:
            feat_records.append(feat or {})
            diag_records.append(diag or {})

    # Build DataFrames
    feat_df = pd.DataFrame.from_records(feat_records)
    feat_df.insert(0, args.id_col, pids)

    diag_df = pd.DataFrame.from_records(diag_records)
    diag_df.insert(0, args.id_col, pids)

    err_df = pd.DataFrame(err_records, columns=[args.id_col, "error"])

    # Drop all-NaN feature columns
    non_id_cols = [c for c in feat_df.columns if c != args.id_col]
    if non_id_cols:
        keep_cols = [args.id_col] + [c for c in non_id_cols if not feat_df[c].isna().all()]
        feat_df = feat_df[keep_cols]

    out_features = outdir / args.out_csv
    out_diag = outdir / "Tb_diagnostics.csv"
    out_err = outdir / "Tb_errors.csv"
    out_meta = outdir / "extraction_metadata.json"

    feat_df.to_csv(out_features, index=False)
    diag_df.to_csv(out_diag, index=False)
    err_df.to_csv(out_err, index=False)

    # Metadata for reproducibility
    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_manifest": int(len(manifest)),
        "n_extracted": int(len(feat_df)),
        "n_error": int(len(err_df)),
        "n_features": int(len([c for c in feat_df.columns if c != args.id_col])),
        "yaml_original": args.yaml_original,
        "yaml_filtered": args.yaml_filtered,
        "label": int(args.label),
        "pyradiomics_version": getattr(radiomics, "__version__", "unknown"),
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("========================================================================")
    print("[Tb-EXTRACT] Done.")
    print(f"[Tb-EXTRACT] features_csv: {out_features}")
    print(f"[Tb-EXTRACT] diagnostics_csv: {out_diag}")
    print(f"[Tb-EXTRACT] errors_csv: {out_err}")
    print(f"[Tb-EXTRACT] metadata: {out_meta}")
    print("========================================================================")


if __name__ == "__main__":
    main()

#run command:
#python extract_tb_radiomics.py \
#  --manifest_csv /home/lichengze/Research/DeepFeature/myresearch/main/Tb_argument/images.csv \
#  --id_col PatientID \
#  --image_col image_path \
#  --mask_col mask_path \
#  --yaml_original /home/lichengze/Research/DeepFeature/myresearch/main/Tb_argument/tb_original_bw25.yaml \
#  --yaml_filtered /home/lichengze/Research/DeepFeature/myresearch/main/Tb_argument/tb_filtered_bc32.yaml \
#  --label 1 \
#  --n_jobs 8 \
#  --outdir /home/lichengze/Research/DeepFeature/myresearch/main/Tb_argument \
#  --out_csv Tb_features.csv