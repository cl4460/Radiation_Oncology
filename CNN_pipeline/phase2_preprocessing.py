# phase2_preprocessing_final.py

PHASE1_QC = "/home/lichengze/CNN_pipeline/phase1_outputs/phase1_qc.csv"  
OUT_DIR = "/home/lichengze/CNN_pipeline/phase2_outputs"

# Uniform anisotropic spacing (not isotropic)
TARGET_SP = (1.0, 1.0, 3.0)  # Matches median Z-spacing in your dataset
FIXED_PATCH = (160, 192, 128)  # Slightly larger for context
MARGIN_MM = 20  # Physical margin around tumor

NORM_MODE = "window"
HU_WIN = (-1000, 400)
K_DIV = (8, 8, 8)

import os, json, warnings
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

def bbox_zyx(mask_arr: np.ndarray):
    idx = np.argwhere(mask_arr > 0)
    if idx.size == 0:
        return None
    return idx.min(0), idx.max(0)

def pad_kdiv(arr: np.ndarray, k=(8,8,8)) -> np.ndarray:
    z, y, x = arr.shape
    pz = (k[0] - z % k[0]) % k[0]
    py = (k[1] - y % k[1]) % k[1]
    px = (k[2] - x % k[2]) % k[2]
    return np.pad(arr, ((0,pz), (0,py), (0,px)), mode='constant')

def edt01(mask_arr: np.ndarray) -> np.ndarray:
    m = (mask_arr > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=np.float32)
    d = distance_transform_edt(m).astype(np.float32)
    return d / (d.max() + 1e-6)

def scale_window01(ct: np.ndarray, a_min: int, a_max: int):
    ct = np.clip(ct, a_min, a_max)
    return (ct - a_min) / (a_max - a_min + 1e-6)

def sitk_resample(image: sitk.Image, new_spacing: tuple, is_label: bool = False) -> sitk.Image:
    """Resample to uniform spacing while preserving metadata"""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
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

def process_case(r: dict, out_dir: Path) -> Dict:
    """Process single case with proper margin handling and spatial metadata"""
    pid = r["patient_id"]
    od = out_dir / pid
    od.mkdir(parents=True, exist_ok=True)
    
    # Load and resample to uniform spacing
    ct_img = sitk.ReadImage(r["ct_path"])
    m_img = sitk.ReadImage(r["mask_path"])
    
    ct_resampled = sitk_resample(ct_img, TARGET_SP, is_label=False)
    m_resampled = sitk_resample(m_img, TARGET_SP, is_label=True)
    
    ct = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)
    m = sitk.GetArrayFromImage(m_resampled).astype(np.uint8)
    
    # Normalize intensity
    ct = scale_window01(ct, *HU_WIN)
    
    # Calculate crop with proper margin handling
    bb = bbox_zyx(m)
    truncated = False
    retained_ratio = 1.0
    
    if bb is None:
        # No mask found - center crop fallback
        center = np.array(ct.shape) // 2
        half_patch = np.array(FIXED_PATCH) // 2
        start = np.maximum(center - half_patch, 0)
        retained_ratio = 0.0
    else:
        mn, mx = bb
        roi_size = mx - mn + 1
        
        # Convert margin from mm to voxels (per-axis)
        margin_vox = np.ceil(MARGIN_MM / np.array(TARGET_SP)).astype(int)
        
        # Check if ROI + margin fits in patch
        wanted_size = roi_size + 2 * margin_vox
        center = (mn + mx) // 2
        
        start = np.zeros(3, dtype=int)
        for ax in range(3):
            if wanted_size[ax] <= FIXED_PATCH[ax]:
                # ROI + margin fits - center it with margin
                st = center[ax] - wanted_size[ax] // 2
                st = max(0, min(st, ct.shape[ax] - FIXED_PATCH[ax]))
            else:
                # Doesn't fit - center on ROI and mark truncation
                st = center[ax] - FIXED_PATCH[ax] // 2
                st = max(0, min(st, ct.shape[ax] - FIXED_PATCH[ax]))
                truncated = True
            start[ax] = st
    
    # Perform crop
    end = start + np.array(FIXED_PATCH)
    end = np.minimum(end, ct.shape)
    
    c_ct = ct[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    c_m = m[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    
    # Calculate retained mask ratio for QC
    if bb is not None:
        total_mask = float((m > 0).sum())
        kept_mask = float((c_m > 0).sum())
        retained_ratio = kept_mask / total_mask if total_mask > 0 else 0.0
    
    # Pad if necessary (edge cases)
    if c_ct.shape != tuple(FIXED_PATCH):
        pad_width = [(0, max(0, FIXED_PATCH[i] - c_ct.shape[i])) for i in range(3)]
        c_ct = np.pad(c_ct, pad_width, constant_values=0)
        c_m = np.pad(c_m, pad_width, constant_values=0)
    
    # Generate EDT
    edt = edt01(c_m)
    
    # Ensure k-divisible for network architecture
    c_ct = pad_kdiv(c_ct, K_DIV)
    c_m = pad_kdiv(c_m, K_DIV)
    edt = pad_kdiv(edt, K_DIV)
    
    # Create output images with proper spatial metadata
    def create_output_image(arr, ref_img, is_label=False):
        """Create SimpleITK image preserving spatial metadata with direction-aware origin"""
        out = sitk.GetImageFromArray(arr.astype(np.uint8 if is_label else np.float32))
        
        # Direction-aware origin update (critical for spatial consistency)
        D = np.array(ref_img.GetDirection()).reshape(3, 3)
        offset_index_xyz = np.array([start[2], start[1], start[0]], dtype=float)
        offset_phys_xyz = offset_index_xyz * np.array(TARGET_SP, dtype=float)
        new_origin = np.array(ref_img.GetOrigin()) + D.dot(offset_phys_xyz)
        
        out.SetOrigin(new_origin.tolist())
        out.SetSpacing(TARGET_SP)
        out.SetDirection(ref_img.GetDirection())
        return out
    
    # Save outputs
    out_ct = od / f"{pid}_ct_patch.nii.gz"
    out_mask = od / f"{pid}_mask_patch.nii.gz"
    out_edt = od / f"{pid}_edt_patch.nii.gz"
    
    sitk.WriteImage(create_output_image(c_ct, ct_resampled), str(out_ct))
    sitk.WriteImage(create_output_image(c_m, m_resampled, True), str(out_mask))
    sitk.WriteImage(create_output_image(edt, ct_resampled), str(out_edt))
    
    # QC warning for significant truncation
    if retained_ratio < 0.9 and retained_ratio > 0:
        warnings.warn(f"Patient {pid}: Only {retained_ratio:.1%} of tumor retained after cropping")
    
    return {
        "patient_id": pid,
        "spacing": list(TARGET_SP),
        "norm": NORM_MODE,
        "out_ct": str(out_ct),
        "out_mask": str(out_mask),
        "out_edt": str(out_edt),
        "patch_shape": list(c_ct.shape),
        "crop_start": list(start),
        "truncated": bool(truncated),
        "retained_mask_ratio": float(retained_ratio),
        "qc_warning": retained_ratio < 0.9 if retained_ratio > 0 else False
    }

def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Phase-2 Preprocessing (Final Version)")
    print("="*70)
    print(f"Target spacing: {TARGET_SP} mm (uniform anisotropic)")
    print(f"Fixed patch size: {FIXED_PATCH}")
    print(f"Physical margin: {MARGIN_MM} mm")
    print("="*70)
    
    # Load data
    data = pd.read_csv(PHASE1_QC).to_dict('records')
    data = [r for r in data if r.get("ct_path") and r.get("mask_path")]
    
    logs = []
    truncated_count = 0
    low_retention_count = 0
    
    for r in tqdm(data, desc="Processing cases"):
        try:
            log = process_case(r, out_dir)
            logs.append(log)
            
            if log["truncated"]:
                truncated_count += 1
            if log.get("qc_warning", False):
                low_retention_count += 1
                
        except Exception as e:
            logs.append({"patient_id": r["patient_id"], "error": str(e)})
            print(f"\n[ERROR] {r['patient_id']}: {e}")
    
    # Save logs
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(out_dir / "crop_log.csv", index=False)
    
    # QC summary
    print("\n" + "="*70)
    print("QC Summary:")
    print(f"  Total cases: {len(data)}")
    print(f"  Successfully processed: {len([l for l in logs if 'error' not in l])}")
    print(f"  Truncated cases: {truncated_count} ({truncated_count/len(data)*100:.1f}%)")
    print(f"  Low retention (<90%): {low_retention_count} cases")
    
    if truncated_count > len(data) * 0.2:
        print("\n⚠️  Warning: >20% of cases truncated. Consider increasing FIXED_PATCH size.")
    
    print(f"\nLogs saved to: {out_dir / 'crop_log.csv'}")
    print("="*70)

if __name__ == "__main__":
    main()