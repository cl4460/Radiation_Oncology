#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import numpy as np

import SimpleITK as sitk
from radiomics import featureextractor


TEXTURE_CLASSES = ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")


def _resample_to_ref(mask: sitk.Image, ref: sitk.Image) -> sitk.Image:
    """Resample mask to reference image geometry (nearest neighbor)."""
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetTransform(sitk.Transform())
    rs.SetDefaultPixelValue(0)
    out = rs.Execute(mask)
    return sitk.Cast(out, sitk.sitkUInt8)


def make_ring(mask_bin: sitk.Image, inner_mm: float, outer_mm: float) -> sitk.Image:
    """
    Create peritumoral ring mask using signed distance map in physical space (mm).
    mask_bin must be 0/1 (uint8), where 1 is tumor.
    ring is outside tumor: inner_mm < dist <= outer_mm.
    """
    dist = sitk.SignedMaurerDistanceMap(
        mask_bin,
        insideIsPositive=False,   # inside tumor -> negative, outside -> positive
        squaredDistance=False,
        useImageSpacing=True
    )
    eps = 1e-6
    ring = sitk.BinaryThreshold(
        dist,
        lowerThreshold=float(inner_mm) + eps,
        upperThreshold=float(outer_mm),
        insideValue=1,
        outsideValue=0
    )
    ring = sitk.Cast(ring, sitk.sitkUInt8)
    ring.CopyInformation(mask_bin)
    return ring


def build_extractor(bin_width: float = 25.0, force2d: bool = True, force2d_dim: int = 0):
    """
    Build a PyRadiomics extractor for CT HU:
    - Fixed binWidth (25 HU is common in lung CT radiomics literature)
    - force2D for anisotropic voxels (z ~ 3mm)
    - Only Original image type
    - No shape features (ring shapes are derived from tumor geometry; mostly redundant)
    """
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()

    for cls in TEXTURE_CLASSES:
        extractor.enableFeatureClassByName(cls)

    # Settings
    extractor.settings["binWidth"] = float(bin_width)
    extractor.settings["force2D"] = bool(force2d)
    extractor.settings["force2Ddimension"] = int(force2d_dim)

    # Keep intensity as HU; do NOT normalize by default
    extractor.settings["normalize"] = False
    extractor.settings["removeOutliers"] = None

    # Speed: crop to bounding box of mask
    extractor.settings["preCrop"] = True

    return extractor


def extract_one_case(extractor, pid: str, ct_path: str, mask_path: str,
                     rings, min_voxels: int):
    ct = sitk.ReadImage(ct_path)
    m = sitk.ReadImage(mask_path)

    # Ensure mask matches CT geometry
    if (m.GetSize() != ct.GetSize() or
        m.GetSpacing() != ct.GetSpacing() or
        m.GetOrigin() != ct.GetOrigin() or
        m.GetDirection() != ct.GetDirection()):
        m = _resample_to_ref(m, ct)

    # Binary tumor
    m_bin = sitk.Cast(m > 0, sitk.sitkUInt8)

    row = {"PatientID": pid}

    for (inner_mm, outer_mm) in rings:
        ring = make_ring(m_bin, inner_mm, outer_mm)

        arr = sitk.GetArrayViewFromImage(ring)
        nvox = int(arr.sum())
        row[f"PR_{inner_mm:g}_{outer_mm:g}mm_nvox"] = nvox

        if nvox < min_voxels:
            # Too small => texture unstable; mark as NaN
            for cls in TEXTURE_CLASSES:
                # We don't know exact keys beforehand; skip and just warn via nvox
                pass
            continue

        # Extract (label=1)
        feats = extractor.execute(ct, ring, label=1)

        # Filter diagnostics
        feats = {k: v for k, v in feats.items() if not k.startswith("diagnostics_")}

        # Prefix
        prefix = f"PR_{inner_mm:g}_{outer_mm:g}mm_"
        for k, v in feats.items():
            row[prefix + k] = v

    return row


def parse_rings(ring_str: str):
    """
    ring_str: "0-3,3-6" -> [(0,3),(3,6)]
    """
    rings = []
    for part in ring_str.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("-")
        rings.append((float(a), float(b)))
    return rings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_csv", required=True,
                    help="CSV with columns: PatientID, image_path (or ct_path), mask_path")
    ap.add_argument("--out_csv", required=True,
                    help="Output CSV path for peritumoral ring features")
    ap.add_argument("--rings", default="0-3,3-6",
                    help='Ring shells in mm, e.g. "0-3,3-6"')
    ap.add_argument("--bin_width", type=float, default=25.0)
    ap.add_argument("--force2d", action="store_true", default=True)
    ap.add_argument("--force2d_dim", type=int, default=0,
                    help="0 means compute textures per axial slice in PyRadiomics")
    ap.add_argument("--min_voxels", type=int, default=64,
                    help="Skip texture extraction if ring voxels < this threshold")
    ap.add_argument("--fail_fast", action="store_true",
                    help="Raise on first error (default: skip case)")
    args = ap.parse_args()

    df = pd.read_csv(args.images_csv)
    need = {"PatientID", "mask_path"}
    if not need.issubset(df.columns):
        raise ValueError(f"images_csv must contain columns {need}, got {list(df.columns)}")
    
    # Support both image_path and ct_path
    if "image_path" in df.columns:
        image_col = "image_path"
    elif "ct_path" in df.columns:
        image_col = "ct_path"
    else:
        raise ValueError(f"images_csv must contain either 'image_path' or 'ct_path', got {list(df.columns)}")

    rings = parse_rings(args.rings)
    extractor = build_extractor(bin_width=args.bin_width,
                                force2d=args.force2d,
                                force2d_dim=args.force2d_dim)

    rows = []
    n_ok, n_fail = 0, 0

    for r in df.itertuples(index=False):
        pid = getattr(r, "PatientID")
        ct_path = getattr(r, image_col)
        mask_path = getattr(r, "mask_path")

        try:
            if not os.path.exists(ct_path):
                raise FileNotFoundError(ct_path)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(mask_path)

            row = extract_one_case(extractor, pid, ct_path, mask_path, rings, args.min_voxels)
            rows.append(row)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            msg = f"[FAIL] {pid}: {type(e).__name__}: {e}"
            if args.fail_fast:
                raise RuntimeError(msg) from e
            else:
                print(msg)

    out_df = pd.DataFrame(rows)

    # Drop columns that are entirely NaN (happens if some features never computed)
    out_df = out_df.dropna(axis=1, how="all")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"[DONE] ok={n_ok} fail={n_fail} -> {args.out_csv}")
    print(f"[INFO] shape={out_df.shape}")


if __name__ == "__main__":
    main()
