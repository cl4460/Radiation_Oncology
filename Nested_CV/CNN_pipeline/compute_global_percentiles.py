#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute global HU percentiles (p05 and p99.5) over TRAIN CT volumes.

- Supports DICOM series (directory) and NIfTI (.nii/.nii.gz) files.
- Builds a global histogram to avoid loading everything in memory at once.

Usage examples:
  python compute_global_percentiles.py --phase1_qc /path/to/phase1_qc.csv --split train
  python compute_global_percentiles.py --path_list /path/to/train_ct_paths.txt
"""

import os, sys, argparse, math
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# IO helpers
# -----------------------------
def guess_is_dicom_dir(p: Path) -> bool:
    if p.is_dir():
        # Heuristic: directory with many files and not a single NIfTI
        for f in p.iterdir():
            if f.suffix.lower() in [".nii", ".gz", ".mha", ".mhd", ".nrrd"]:
                return False
        return True
    return False

def read_ct_as_hu(path: Path) -> np.ndarray:
    """
    Read CT volume as HU using SimpleITK.
    For DICOM series: uses ImageSeriesReader (ITK applies RescaleSlope/Intercept).
    For NIfTI: assumes already in HU (your Phase-1 outputs usually are).
    Returns a float32 numpy array in (Z,Y,X).
    """
    if guess_is_dicom_dir(path):
        reader = sitk.ImageSeriesReader()
        series = reader.GetGDCMSeriesFileNames(str(path))
        if len(series) == 0:
            raise RuntimeError(f"No DICOM series found in: {path}")
        reader.SetFileNames(series)
        img = reader.Execute()  # HU scaling applied by ITK/GDCM
    else:
        img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    return arr

# -----------------------------
# Global histogram percentiles
# -----------------------------
def percentile_from_histogram(counts: np.ndarray, bin_edges: np.ndarray, q: float) -> float:
    """
    q in [0,100]. Linear interpolation within the crossing bin.
    """
    assert 0.0 <= q <= 100.0
    cdf = np.cumsum(counts, dtype=np.float64)
    total = cdf[-1]
    if total == 0:
        return float("nan")
    target = (q / 100.0) * total
    idx = np.searchsorted(cdf, target)
    idx = min(max(int(idx), 0), len(counts) - 1)
    # Bin edges → choose left/right; refine linearly within the bin
    left_count = cdf[idx - 1] if idx > 0 else 0.0
    in_bin = counts[idx]
    if in_bin <= 0:
        return float(bin_edges[idx])  # degenerate
    frac = (target - left_count) / in_bin
    # linear within [edge_i, edge_{i+1}]
    return float(bin_edges[idx] + frac * (bin_edges[idx+1] - bin_edges[idx]))

def accumulate_hist_for_volume(arr: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Accumulate histogram for a single volume, skipping non-finite.
    """
    a = arr[np.isfinite(arr)]
    # For stability, clip extreme outliers to histogram range
    a = np.clip(a, bin_edges[0], bin_edges[-1])
    hist, _ = np.histogram(a, bins=bin_edges)
    return hist.astype(np.int64)

# -----------------------------
# Main compute
# -----------------------------
def compute_global_percentiles(
    ct_paths,
    p_low=5.0,
    p_high=99.5,
    hu_min=-1500.0,
    hu_max=1500.0,
    bin_width=1.0
):
    bin_edges = np.arange(hu_min, hu_max + bin_width, bin_width, dtype=np.float32)
    global_counts = np.zeros(len(bin_edges) - 1, dtype=np.int64)
    n_vols = 0
    n_voxels = 0

    print(f"Processing {len(ct_paths)} CT volumes...", file=sys.stderr)
    for p in tqdm(ct_paths, desc="Computing histograms", unit="vol"):
        p = Path(p)
        if not (p.exists()):
            print(f"[WARN] path not found, skip: {p}", file=sys.stderr)
            continue
        try:
            vol = read_ct_as_hu(p)
            global_counts += accumulate_hist_for_volume(vol, bin_edges)
            n_vols += 1
            n_voxels += vol.size
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}", file=sys.stderr)
            continue

    if n_vols == 0:
        raise RuntimeError("No valid CT volumes processed. Please check your paths.")

    p05  = percentile_from_histogram(global_counts, bin_edges, p_low)
    p995 = percentile_from_histogram(global_counts, bin_edges, p_high)
    return p05, p995, n_vols, n_voxels

# -----------------------------
# Utilities to collect paths
# -----------------------------
def ct_paths_from_phase1_qc(qc_csv: Path, split: str = "train", path_col: str = "ct_path"):
    df = pd.read_csv(qc_csv)
    if "split" in df.columns and split is not None:
        df = df[df["split"].astype(str).str.lower() == split.lower()]
    paths = df[path_col].dropna().astype(str).tolist()
    # optional: fix your two common roots
    fixed = []
    for p in paths:
        if os.path.exists(p):
            fixed.append(p); continue
        p1 = p.replace("/home/lichengze/CNN_pipeline/", "/home/lichengze/Research/CNN_pipeline/")
        p2 = p.replace("/home/lichengze/Research/CNN_pipeline/", "/home/lichengze/CNN_pipeline/")
        if os.path.exists(p1): fixed.append(p1)
        elif os.path.exists(p2): fixed.append(p2)
        else: fixed.append(p)  # keep original; reader will warn if missing
    return fixed

def ct_paths_from_txt(list_txt: Path):
    return [l.strip() for l in open(list_txt, "r").read().splitlines() if l.strip()]

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1_qc", type=str, default=None, help="phase1_qc.csv path")
    ap.add_argument("--split", type=str, default="train", help="split name in qc csv (e.g., train/val)")
    ap.add_argument("--path_list", type=str, default=None, help="txt file with one CT path per line")
    ap.add_argument("--hu_min", type=float, default=-1500.0)
    ap.add_argument("--hu_max", type=float, default=1500.0)
    ap.add_argument("--bin_width", type=float, default=1.0)
    args = ap.parse_args()

    if args.path_list:
        ct_paths = ct_paths_from_txt(Path(args.path_list))
    elif args.phase1_qc:
        ct_paths = ct_paths_from_phase1_qc(Path(args.phase1_qc), split=args.split)
    else:
        print("Please provide --phase1_qc or --path_list", file=sys.stderr)
        sys.exit(1)

    p05, p995, n_vols, n_vox = compute_global_percentiles(
        ct_paths,
        p_low=5.0,
        p_high=99.5,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        bin_width=args.bin_width
    )
    print("="*60)
    print(f"Volumes processed : {n_vols}")
    print(f"Total voxels      : {n_vox:,}")
    print(f"Global clip thresholds: p05={p05:.1f} HU, p99.5={p995:.1f} HU")
    print("="*60)

if __name__ == "__main__":
    main()
