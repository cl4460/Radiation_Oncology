#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import SimpleITK as sitk
from radiomics import featureextractor


FEATURE_CLASSES_DEFAULT = [
    "firstorder",
    "shape",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
]


def read_csv(images_csv: str) -> pd.DataFrame:
    df = pd.read_csv(images_csv)
    required = {"PatientID", "image_path", "mask_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"images_csv missing columns: {sorted(list(missing))}")
    df["PatientID"] = df["PatientID"].astype(str)
    return df


def _clip_hu(img: sitk.Image, hu_min: float, hu_max: float) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = np.clip(arr, hu_min, hu_max)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def _mask_has_label(msk: sitk.Image, label: int) -> bool:
    # Use view to avoid extra copies; still OK for 422
    arr = sitk.GetArrayViewFromImage(msk)
    return bool(np.any(arr == label))


def build_extractor(
    bin_width: float,
    do_resample: bool,
    spacing_xyz,
    interpolator: str,
    label: int,
    geometry_tol: float,
    correct_mask: bool,
    feature_classes,
) -> featureextractor.RadiomicsFeatureExtractor:
    """
    Use PyRadiomics parameter dict so everything is reproducible.
    """
    settings = {
        "binWidth": float(bin_width),
        "normalize": False,
        "geometryTolerance": float(geometry_tol),
        "correctMask": bool(correct_mask),
    }

    if do_resample:
        # Enable PyRadiomics internal resampling
        settings["resampledPixelSpacing"] = [float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])]
        settings["interpolator"] = str(interpolator)  # e.g. "sitkBSpline"

    params = {
        "setting": settings,
        "imageType": {
            "Original": {}
        },
        "featureClass": {cls: [] for cls in feature_classes},
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(params)

    # Explicit label passed at execute time; keep here only for documentation consistency
    # (PyRadiomics supports label argument in execute)
    return extractor


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--images_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--label", type=int, default=1)
    ap.add_argument("--fallback_all_positive", action="store_true",
                    help="If label not found, use all mask>0 as ROI (with warning).")

    ap.add_argument("--bin_width", type=float, default=25.0)

    ap.add_argument("--resample", action="store_true")
    ap.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--interpolator", type=str, default="sitkBSpline",
                    choices=["sitkNearestNeighbor", "sitkLinear", "sitkBSpline", "sitkGaussian"],
                    help="PyRadiomics expects SimpleITK interpolator string, default sitkBSpline.")

    ap.add_argument("--geometry_tol", type=float, default=1e-3)
    ap.add_argument("--correct_mask", action="store_true",
                    help="Let PyRadiomics resample mask to image if geometry mismatch is small/valid.")

    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--clip_min", type=float, default=-1000.0)
    ap.add_argument("--clip_max", type=float, default=400.0)

    ap.add_argument("--feature_classes", type=str, nargs="+", default=FEATURE_CLASSES_DEFAULT)

    ap.add_argument("--limit", type=int, default=0, help="Dry-run N patients (0=all).")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv(args.images_csv)
    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].copy()

    extractor = build_extractor(
        bin_width=args.bin_width,
        do_resample=args.resample,
        spacing_xyz=args.spacing,
        interpolator=args.interpolator,
        label=args.label,
        geometry_tol=args.geometry_tol,
        correct_mask=args.correct_mask,
        feature_classes=args.feature_classes,
    )

    # Save params for reproducibility
    run_params = {
        "images_csv": str(Path(args.images_csv).resolve()),
        "out_dir": str(out_dir.resolve()),
        "label": int(args.label),
        "fallback_all_positive": bool(args.fallback_all_positive),
        "bin_width": float(args.bin_width),
        "resample": bool(args.resample),
        "spacing": [float(x) for x in args.spacing],
        "interpolator": str(args.interpolator),
        "geometry_tol": float(args.geometry_tol),
        "correct_mask": bool(args.correct_mask),
        "clip": bool(args.clip),
        "clip_min": float(args.clip_min),
        "clip_max": float(args.clip_max),
        "feature_classes": list(args.feature_classes),
        "limit": int(args.limit),
    }
    (out_dir / "Tb_params.json").write_text(json.dumps(run_params, indent=2))

    rows = []
    failures = []

    t0 = time.time()
    for i, r in df.iterrows():
        pid = str(r["PatientID"])
        img_path = str(r["image_path"])
        msk_path = str(r["mask_path"])

        try:
            img = sitk.ReadImage(img_path)
            msk = sitk.ReadImage(msk_path)

            if args.clip:
                img = _clip_hu(img, args.clip_min, args.clip_max)

            used_label = args.label
            if not _mask_has_label(msk, args.label):
                if args.fallback_all_positive:
                    arr = sitk.GetArrayViewFromImage(msk)
                    uniq_pos = np.unique(arr[arr > 0])
                    print(f"[WARN] {pid}: label={args.label} not found. positive labels={uniq_pos}. "
                          f"Using mask>0 as ROI.")
                    msk = sitk.Cast(msk > 0, sitk.sitkUInt8)
                    used_label = 1
                else:
                    raise ValueError(f"Label {args.label} not present in mask (set --fallback_all_positive to override).")

            # Execute extraction (returns dict with diagnostics + features)
            result = extractor.execute(img, msk, label=int(used_label))

            feat = {k: v for k, v in result.items() if not str(k).startswith("diagnostics")}
            feat["PatientID"] = pid
            rows.append(feat)

            if (len(rows) % 20) == 0:
                print(f"[OK] processed {len(rows)}/{len(df)}")

        except Exception as e:
            failures.append({"PatientID": pid, "image_path": img_path, "mask_path": msk_path, "error": str(e)})
            print(f"[FAIL] {pid}: {e}")

    # Build dataframe
    if len(rows) == 0:
        raise RuntimeError("No features extracted. Check failures.csv and input paths.")

    feat_df = pd.DataFrame(rows)

    # Ensure PatientID first
    cols = ["PatientID"] + sorted([c for c in feat_df.columns if c != "PatientID"])
    feat_df = feat_df[cols]

    feat_df.to_csv(out_dir / "Tb_features.csv", index=False)
    pd.DataFrame(failures).to_csv(out_dir / "Tb_failures.csv", index=False)

    # ---- Sanity stats ----
    X = feat_df.drop(columns=["PatientID"])
    n_features = int(X.shape[1])
    n_nan_total = int(X.isna().sum().sum())
    n_pat_with_nan = int(X.isna().any(axis=1).sum())

    sanity = {
        "n_patients_attempted": int(len(df)),
        "n_patients_ok": int(len(feat_df)),
        "n_patients_failed": int(len(failures)),
        "n_features": n_features,
        "n_nan_total": n_nan_total,
        "n_patients_with_any_nan": n_pat_with_nan,
        "elapsed_sec": float(time.time() - t0),
    }
    (out_dir / "Tb_sanity.json").write_text(json.dumps(sanity, indent=2))

    print("[Sanity]", json.dumps(sanity, indent=2))
    if n_features < 50:
        print(f"[WARN] Very few features ({n_features}). Check feature classes / extraction settings.")
    if n_pat_with_nan > 0:
        print(f"[WARN] {n_pat_with_nan} patients have NaN(s). You must handle NaNs before Coxnet.")

    print(f"[DONE] wrote: {out_dir/'Tb_features.csv'}")


if __name__ == "__main__":
    main()

#run command:
#python extract_Tb_pyradiomics.py \
#  --images_csv /home/lichengze/Research/DeepFeature/myresearch/main/R3/images.csv \
#  --out_dir    /home/lichengze/Research/DeepFeature/myresearch/main/R4/Tb_bw25_iso111_clip_dryrun10 \
#  --label 1 \
#  --fallback_all_positive \
#  --clip --clip_min -1000 --clip_max 400 \
#  --bin_width 25 \
#  --resample --spacing 1 1 1 --interpolator sitkBSpline \
#  --correct_mask --geometry_tol 1e-3 \
#  --limit 10