# phase2_preprocessing_radiogenomics.py
"""
Phase 2 preprocessing for NSCLC-Radiogenomics dataset
Handles cases with and without tumor segmentations
"""

# ========== CONFIGURATION ==========
PHASE1_QC = "/home/lichengze/Research/Radiogenomics/phase1_outputs_fixed/phase1_qc.csv"
OUT_DIR = "/home/lichengze/Research/Radiogenomics/phase2_outputs"

# Processing mode: 'with_seg_only' or 'all'
PROCESSING_MODE = "with_seg_only"  # Change to 'all' if you want to process all patients

# Uniform anisotropic spacing (not isotropic)
TARGET_SP = (1.0, 1.0, 3.0)  # Matches median Z-spacing
FIXED_PATCH = (160, 192, 128)  # Slightly larger for context
MARGIN_MM = 20  # Physical margin around tumor (only for cases with segmentation)

# Normalization
NORM_MODE = "window"
HU_WIN = (-1000, 400)
K_DIV = (32, 32, 32)

import os, json, warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

def bbox_zyx(mask_arr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Calculate bounding box of mask"""
    idx = np.argwhere(mask_arr > 0)
    if idx.size == 0:
        return None
    return idx.min(0), idx.max(0)

def pad_kdiv(arr: np.ndarray, k=(8,8,8)) -> np.ndarray:
    """Pad array to be divisible by k"""
    z, y, x = arr.shape
    pz = (k[0] - z % k[0]) % k[0]
    py = (k[1] - y % k[1]) % k[1]
    px = (k[2] - x % k[2]) % k[2]
    return np.pad(arr, ((0,pz), (0,py), (0,px)), mode='constant')

def edt01(mask_arr: np.ndarray) -> np.ndarray:
    """Normalized Euclidean distance transform"""
    m = (mask_arr > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=np.float32)
    d = distance_transform_edt(m).astype(np.float32)
    return d / (d.max() + 1e-6)

def scale_window01(ct: np.ndarray, a_min: int, a_max: int) -> np.ndarray:
    """Window and normalize CT to [0,1]"""
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

def process_case_with_segmentation(r: dict, out_dir: Path) -> Dict:
    """Process case that has tumor segmentation"""
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
    
    # Calculate crop with margin
    bb = bbox_zyx(m)
    truncated = False
    retained_ratio = 1.0
    
    if bb is None:
        raise ValueError(f"Mask is empty for {pid} despite having segmentation file")
    
    mn, mx = bb
    roi_size = mx - mn + 1
    
    # Convert margin from mm to voxels (z,y,x order)
    spacing_zyx = np.array([TARGET_SP[2], TARGET_SP[1], TARGET_SP[0]])  # (3.0, 1.0, 1.0)
    margin_vox = np.ceil(MARGIN_MM / spacing_zyx).astype(int)  # (7, 20, 20)
    
    # Check if ROI + margin fits in patch
    wanted_size = roi_size + 2 * margin_vox
    center = (mn + mx) // 2
    
    start = np.zeros(3, dtype=int)
    for ax in range(3):
        if wanted_size[ax] <= FIXED_PATCH[ax]:
            # ROI + margin fits - center it
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
    
    # Calculate retained mask ratio
    total_mask = float((m > 0).sum())
    kept_mask = float((c_m > 0).sum())
    retained_ratio = kept_mask / total_mask if total_mask > 0 else 0.0
    
    # Pad if necessary
    if c_ct.shape != tuple(FIXED_PATCH):
        pad_width = [(0, max(0, FIXED_PATCH[i] - c_ct.shape[i])) for i in range(3)]
        c_ct = np.pad(c_ct, pad_width, constant_values=0)
        c_m = np.pad(c_m, pad_width, constant_values=0)
    
    # Generate EDT
    edt = edt01(c_m)
    
    # Ensure k-divisible
    c_ct = pad_kdiv(c_ct, K_DIV)
    c_m = pad_kdiv(c_m, K_DIV)
    edt = pad_kdiv(edt, K_DIV)
    
    # Create output images with spatial metadata
    def create_output_image(arr, ref_img, is_label=False):
        out = sitk.GetImageFromArray(arr.astype(np.uint8 if is_label else np.float32))
        
        # Direction-aware origin update
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
    
    # QC warning
    if retained_ratio < 0.9:
        warnings.warn(f"Patient {pid}: Only {retained_ratio:.1%} of tumor retained")
    
    return {
        "patient_id": pid,
        "has_segmentation": True,
        "spacing": list(TARGET_SP),
        "norm": NORM_MODE,
        "out_ct": str(out_ct),
        "out_mask": str(out_mask),
        "out_edt": str(out_edt),
        "patch_shape": list(c_ct.shape),
        "crop_start": list(start),
        "truncated": bool(truncated),
        "retained_mask_ratio": float(retained_ratio),
        "qc_warning": retained_ratio < 0.9
    }

def process_case_without_segmentation(r: dict, out_dir: Path) -> Dict:
    """Process case without tumor segmentation - center crop"""
    pid = r["patient_id"]
    od = out_dir / pid
    od.mkdir(parents=True, exist_ok=True)
    
    # Load and resample
    ct_img = sitk.ReadImage(r["ct_path"])
    ct_resampled = sitk_resample(ct_img, TARGET_SP, is_label=False)
    ct = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)
    
    # Normalize
    ct = scale_window01(ct, *HU_WIN)
    
    # Center crop
    center = np.array(ct.shape) // 2
    half_patch = np.array(FIXED_PATCH) // 2
    start = np.maximum(center - half_patch, 0)
    start = np.minimum(start, np.array(ct.shape) - np.array(FIXED_PATCH))
    
    end = start + np.array(FIXED_PATCH)
    c_ct = ct[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    
    # Pad if needed
    if c_ct.shape != tuple(FIXED_PATCH):
        pad_width = [(0, max(0, FIXED_PATCH[i] - c_ct.shape[i])) for i in range(3)]
        c_ct = np.pad(c_ct, pad_width, constant_values=0)
    
    # Create empty mask and EDT
    c_m = np.zeros_like(c_ct, dtype=np.uint8)
    edt = np.zeros_like(c_ct, dtype=np.float32)
    
    # Ensure k-divisible
    c_ct = pad_kdiv(c_ct, K_DIV)
    c_m = pad_kdiv(c_m, K_DIV)
    edt = pad_kdiv(edt, K_DIV)
    
    # Create output images
    def create_output_image(arr, ref_img, is_label=False):
        out = sitk.GetImageFromArray(arr.astype(np.uint8 if is_label else np.float32))
        
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
    sitk.WriteImage(create_output_image(c_m, ct_resampled, True), str(out_mask))
    sitk.WriteImage(create_output_image(edt, ct_resampled), str(out_edt))
    
    return {
        "patient_id": pid,
        "has_segmentation": False,
        "spacing": list(TARGET_SP),
        "norm": NORM_MODE,
        "out_ct": str(out_ct),
        "out_mask": str(out_mask),
        "out_edt": str(out_edt),
        "patch_shape": list(c_ct.shape),
        "crop_start": list(start),
        "truncated": False,
        "retained_mask_ratio": 0.0,
        "qc_warning": False,
        "note": "No segmentation - center crop"
    }

def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Phase-2 Preprocessing for NSCLC-Radiogenomics")
    print("="*70)
    print(f"Target spacing: {TARGET_SP} mm")
    print(f"Fixed patch size: {FIXED_PATCH}")
    print(f"Physical margin: {MARGIN_MM} mm (for cases with segmentation)")
    print(f"Processing mode: {PROCESSING_MODE}")
    print("="*70)
    
    # Load phase1 QC data
    df = pd.read_csv(PHASE1_QC)
    
    # Check for mask_voxels column
    if 'mask_voxels' not in df.columns:
        print("ERROR: phase1_qc.csv missing 'mask_voxels' column")
        print("Please run the updated phase1_preprocessing_radiogenomics.py first")
        return
    
    # Separate cases with/without segmentations
    with_seg = df[df['mask_voxels'] > 0].to_dict('records')
    without_seg = df[df['mask_voxels'] == 0].to_dict('records')
    
    print(f"\nDataset overview:")
    print(f"  Total patients: {len(df)}")
    print(f"  With segmentation: {len(with_seg)}")
    print(f"  Without segmentation: {len(without_seg)}")
    
    # Filter based on processing mode
    if PROCESSING_MODE == "with_seg_only":
        data = with_seg
        print(f"\nProcessing only {len(data)} patients WITH segmentation")
        print("Change PROCESSING_MODE to 'all' to process all patients")
    elif PROCESSING_MODE == "all":
        data = df.to_dict('records')
        print(f"\nProcessing ALL {len(data)} patients")
    else:
        print(f"ERROR: Invalid PROCESSING_MODE '{PROCESSING_MODE}'")
        print("Valid options: 'with_seg_only' or 'all'")
        return
    
    print("\nStarting processing...")
    
    logs = []
    truncated_count = 0
    low_retention_count = 0
    no_seg_processed = 0
    
    for r in tqdm(data, desc="Processing"):
        try:
            # Check if patient has segmentation
            has_seg = r.get('mask_voxels', 0) > 0
            
            if has_seg:
                log = process_case_with_segmentation(r, out_dir)
                if log["truncated"]:
                    truncated_count += 1
                if log.get("qc_warning", False):
                    low_retention_count += 1
            else:
                log = process_case_without_segmentation(r, out_dir)
                no_seg_processed += 1
            
            logs.append(log)
            
        except Exception as e:
            logs.append({
                "patient_id": r["patient_id"],
                "has_segmentation": r.get('mask_voxels', 0) > 0,
                "error": str(e)
            })
            print(f"\n[ERROR] {r['patient_id']}: {e}")
    
    # Save logs
    logs_df = pd.DataFrame(logs)
    logs_df.to_csv(out_dir / "phase2_crop_log.csv", index=False)
    
    # Save separate lists for downstream use
    with_seg_ids = logs_df[logs_df['has_segmentation'] == True]['patient_id'].tolist()
    without_seg_ids = logs_df[logs_df['has_segmentation'] == False]['patient_id'].tolist()
    
    with open(out_dir / "patients_with_segmentation.txt", 'w') as f:
        f.write('\n'.join(with_seg_ids))
    
    if without_seg_ids:
        with open(out_dir / "patients_without_segmentation.txt", 'w') as f:
            f.write('\n'.join(without_seg_ids))
    
    # Summary
    print("\n" + "="*70)
    print("Phase-2 Preprocessing Complete!")
    print("="*70)
    print(f"Total processed: {len(logs)}")
    print(f"  With segmentation: {len(with_seg_ids)}")
    print(f"  Without segmentation: {no_seg_processed}")
    print(f"  Failed: {len([l for l in logs if 'error' in l])}")
    
    if with_seg_ids:
        print(f"\nSegmentation QC (for {len(with_seg_ids)} cases with masks):")
        print(f"  Truncated cases: {truncated_count} ({truncated_count/len(with_seg_ids)*100:.1f}%)")
        print(f"  Low retention (<90%): {low_retention_count}")
        
        if truncated_count > len(with_seg_ids) * 0.2:
            print("\n  ⚠ Warning: >20% truncated. Consider increasing FIXED_PATCH.")
    
    print(f"\nOutputs:")
    print(f"  Cropped images: {out_dir}/*/")
    print(f"  Processing log: {out_dir}/phase2_crop_log.csv")
    print(f"  Patient lists: {out_dir}/patients_*.txt")
    
    if PROCESSING_MODE == "all" and no_seg_processed > 0:
        print(f"\n⚠ Note: {no_seg_processed} patients processed without segmentation")
        print("  These used center crop instead of tumor-centered crop")
        print("  Consider downloading external annotations or auto-segmenting")
        print("  See HOW_TO_ACCESS_SEGMENTATIONS.txt for options")
    
    print("="*70)

if __name__ == "__main__":
    main()