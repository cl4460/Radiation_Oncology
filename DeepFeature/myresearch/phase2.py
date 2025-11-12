#!/usr/bin/env python
# phase2_preprocessing_med3d.py
# Phase 2: Resample -> Med3D intensity norm (0.5/99.5 + per-volume z-score) -> ROI crop with margin -> k-divisible pad
# Outputs: CT patch (float32, z-scored), MASK patch (uint8) with correct spatial metadata
# References:
#   Med3D Intensity Normalization: truncate to [0.5, 99.5] percentiles per volume, then z-score per volume (mean/std). (Chen et al., Med3D) 

import os, json, warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# ==============================
# Config (adjust paths if needed)
# ==============================
PHASE1_QC = "/home/lichengze/Research/DeepFeature/myresearch/phase1_outputs/phase1_qc.csv"
OUT_DIR   = "/home/lichengze/Research/DeepFeature/myresearch/phase2_outputs"

# Spatial settings
TARGET_SP   = (1.0, 1.0, 3.0)         # (X, Y, Z) in mm
FIXED_PATCH = (128, 192, 160)         # (Z, Y, X) in voxels, network input tile
MARGIN_MM   = 20                      # physical margin around tumor (mm)
K_DIV       = (32, 32, 32)            # ResNet10 total downsample factor per axis

EPS = 1e-8

# ==============================
# Helpers
# ==============================
def bbox_zyx(mask_arr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (min_zyx, max_zyx) for mask > 0; None if empty."""
    idx = np.argwhere(mask_arr > 0)
    if idx.size == 0:
        return None
    return idx.min(0), idx.max(0)

def pad_to_kdiv(arr: np.ndarray, k=(32,32,32)) -> np.ndarray:
    """Right-pad array in Z,Y,X so that each dim is divisible by k."""
    z, y, x = arr.shape
    pz = (k[0] - z % k[0]) % k[0]
    py = (k[1] - y % k[1]) % k[1]
    px = (k[2] - x % k[2]) % k[2]
    if arr.dtype.kind == 'f':
        # z-score 后 mean≈0，用 0 填充最稳妥
        return np.pad(arr, ((0,pz), (0,py), (0,px)), mode="constant", constant_values=0.0)
    else:
        return np.pad(arr, ((0,pz), (0,py), (0,px)), mode="constant", constant_values=0)

def sitk_resample(image: sitk.Image, new_spacing: tuple, is_label: bool=False) -> sitk.Image:
    """Resample to new_spacing while preserving direction/origin; size computed by spacing ratio."""
    original_spacing = image.GetSpacing()
    original_size    = image.GetSize()
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
        for i in range(3)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0 if is_label else -1000)
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resample.Execute(image)

def med3d_intensity_norm_per_volume(ct_resampled_arr: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    """
    Med3D normalization (per-volume):
      1) clip to [p0.5, p99.5] of the resampled volume
      2) z-score with this volume's mean/std (after clipping)
    Returns: (ct_norm, p05, p995, mu, sigma)
    """
    # Percentiles on the WHOLE resampled volume (not the cropped patch)
    p05  = float(np.percentile(ct_resampled_arr, 0.5))
    p995 = float(np.percentile(ct_resampled_arr, 99.5))
    ct_clip = np.clip(ct_resampled_arr, p05, p995)
    mu  = float(ct_clip.mean())
    sd  = float(ct_clip.std()) + EPS
    ct_norm = (ct_clip - mu) / sd
    return ct_norm, p05, p995, mu, sd

def create_output_image(arr: np.ndarray, ref_img: sitk.Image, start_zyx: np.ndarray, spacing_xyz: tuple, is_label=False) -> sitk.Image:
    """
    Create a SimpleITK image from array, preserving direction and spacing,
    and updating origin by physical offset computed from START index (ZYX).
    """
    out = sitk.GetImageFromArray(arr.astype(np.uint8 if is_label else np.float32))
    D = np.array(ref_img.GetDirection()).reshape(3,3)
    # start_zyx index -> offset in XYZ (note spacing is given in XYZ)
    offset_index_xyz = np.array([start_zyx[2], start_zyx[1], start_zyx[0]], dtype=float)
    offset_phys_xyz  = offset_index_xyz * np.array(spacing_xyz, dtype=float)
    new_origin_xyz   = np.array(ref_img.GetOrigin()) + D.dot(offset_phys_xyz)

    out.SetOrigin(new_origin_xyz.tolist())
    out.SetSpacing(spacing_xyz)
    out.SetDirection(ref_img.GetDirection())
    return out

# ==============================
# Core per-case processing
# ==============================
def process_case(rec: dict, out_dir: Path) -> Dict:
    pid = rec["patient_id"]
    od  = out_dir / pid
    od.mkdir(parents=True, exist_ok=True)

    # 1) Load & resample CT and MASK
    ct_img = sitk.ReadImage(rec["ct_path"])
    ms_img = sitk.ReadImage(rec["mask_path"])
    ct_rs  = sitk_resample(ct_img, TARGET_SP, is_label=False)
    ms_rs  = sitk_resample(ms_img, TARGET_SP, is_label=True)

    # Numpy volumes (ZYX)
    ct_rs_np = sitk.GetArrayFromImage(ct_rs).astype(np.float32)
    ms_rs_np = sitk.GetArrayFromImage(ms_rs).astype(np.uint8)

    # 2) Med3D per-volume intensity normalization on the FULL resampled volume
    ct_norm_full, p05, p995, mu, sd = med3d_intensity_norm_per_volume(ct_rs_np)

    # 3) Compute crop window with margin (in ZYX index space)
    bb = bbox_zyx(ms_rs_np)
    truncated = False
    retained_ratio = 1.0

    if bb is None:
        # No mask -> center crop fallback
        center = np.array(ct_norm_full.shape) // 2
        half   = np.array(FIXED_PATCH) // 2
        start  = np.maximum(center - half, 0)
    else:
        mn, mx   = bb
        roi_size = mx - mn + 1  # (Z,Y,X)

        # Convert 20 mm margin to voxels in Z,Y,X (spacing order Z,Y,X = 3.0,1.0,1.0)
        spacing_zyx = np.array([TARGET_SP[2], TARGET_SP[1], TARGET_SP[0]], dtype=float)
        margin_vox  = np.ceil(MARGIN_MM / spacing_zyx).astype(int)  # (Z,Y,X) ~ (7,20,20)

        wanted      = roi_size + 2 * margin_vox           # desired crop in Z,Y,X
        center      = (mn + mx) // 2
        start       = np.zeros(3, dtype=int)

        for ax in range(3):  # 0:Z, 1:Y, 2:X
            if wanted[ax] <= FIXED_PATCH[ax]:
                st = center[ax] - wanted[ax] // 2
                st = max(0, min(st, ct_norm_full.shape[ax] - FIXED_PATCH[ax]))
            else:
                # Can't fit with full margin; center on ROI and record truncation
                st = center[ax] - FIXED_PATCH[ax] // 2
                st = max(0, min(st, ct_norm_full.shape[ax] - FIXED_PATCH[ax]))
                truncated = True
            start[ax] = st

    end = start + np.array(FIXED_PATCH)
    end = np.minimum(end, np.array(ct_norm_full.shape))

    # 4) Crop
    ct_crop = ct_norm_full[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    ms_crop = ms_rs_np   [start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    # Mask retention for QC
    if bb is not None:
        total_mask = float((ms_rs_np > 0).sum())
        kept_mask  = float((ms_crop  > 0).sum())
        retained_ratio = kept_mask / total_mask if total_mask > 0 else 0.0

    # 5) Pad to FIXED_PATCH then to K_DIV
    # First pad to FIXED_PATCH (right/bottom/back only, origin offset already handled by 'start')
    pad_z = max(0, FIXED_PATCH[0] - ct_crop.shape[0])
    pad_y = max(0, FIXED_PATCH[1] - ct_crop.shape[1])
    pad_x = max(0, FIXED_PATCH[2] - ct_crop.shape[2])
    if pad_z or pad_y or pad_x:
        ct_crop = np.pad(ct_crop, ((0,pad_z),(0,pad_y),(0,pad_x)), mode="constant", constant_values=0.0)
        ms_crop = np.pad(ms_crop, ((0,pad_z),(0,pad_y),(0,pad_x)), mode="constant", constant_values=0)

    # Then ensure k-divisible
    ct_crop = pad_to_kdiv(ct_crop, K_DIV)
    ms_crop = pad_to_kdiv(ms_crop, K_DIV)

    # 6) Save as NIfTI with correct spatial metadata
    out_ct   = od / f"{pid}_ct_patch.nii.gz"
    out_mask = od / f"{pid}_mask_patch.nii.gz"

    sitk.WriteImage(create_output_image(ct_crop,  ct_rs, start, TARGET_SP, is_label=False), str(out_ct))
    sitk.WriteImage(create_output_image(ms_crop,  ms_rs, start, TARGET_SP, is_label=True),  str(out_mask))

    # QC warn
    if (0.0 < retained_ratio < 0.9):
        warnings.warn(f"[{pid}] tumor retained {retained_ratio:.1%} after cropping")

    return {
        "patient_id": pid,
        "spacing_xyz": list(TARGET_SP),
        "patch_zyx": list(ct_crop.shape),
        "crop_start_zyx": list(start.astype(int)),
        "truncated": bool(truncated),
        "retained_mask_ratio": float(retained_ratio),
        # Med3D norm stats (useful for auditing)
        "p0.5":  float(p05),
        "p99.5": float(p995),
        "mu":    float(mu),
        "sigma": float(sd),
        "out_ct":   str(out_ct),
        "out_mask": str(out_mask),
    }

# ==============================
# Entry
# ==============================
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("Phase-2 Preprocessing (Med3D normalization; CT+Mask only)")
    print("="*70)
    print(f"Target spacing (XYZ): {TARGET_SP}")
    print(f"Fixed patch (ZYX):    {FIXED_PATCH}")
    print(f"Margin (mm):          {MARGIN_MM}")
    print(f"K-DIV:                {K_DIV}")
    print("="*70)

    # Basic sanity: FIXED_PATCH must be divisible by K_DIV
    assert all(FIXED_PATCH[i] % K_DIV[i] == 0 for i in range(3)), \
        f"FIXED_PATCH {FIXED_PATCH} must be divisible by K_DIV {K_DIV}"

    df = pd.read_csv(PHASE1_QC)
    records = [r for r in df.to_dict("records") if r.get("ct_path") and r.get("mask_path")]

    logs = []
    n_trunc = 0
    n_low   = 0

    for r in tqdm(records, desc="Processing cases"):
        try:
            log = process_case(r, out_dir)
            logs.append(log)
            if log["truncated"]:
                n_trunc += 1
            if 0.0 < log["retained_mask_ratio"] < 0.9:
                n_low += 1
        except Exception as e:
            logs.append({"patient_id": r.get("patient_id","NA"), "error": str(e)})
            print(f"\n[ERROR] {r.get('patient_id','NA')}: {e}")

    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(out_dir / "crop_log.csv", index=False)

    print("\n" + "="*70)
    print("QC Summary")
    print(f"  Total cases:           {len(records)}")
    success_count = len(logs_df) - int(logs_df["error"].notna().sum()) if "error" in logs_df.columns else len(logs_df)
    print(f"  Success (no error):    {success_count}")
    print(f"  Truncated cases:       {n_trunc} ({(n_trunc/max(1,len(records)))*100:.1f}%)")
    print(f"  Low retention (<90%):  {n_low}")
    print(f"  Logs saved to:         {out_dir / 'crop_log.csv'}")
    print("="*70)

if __name__ == "__main__":
    main()