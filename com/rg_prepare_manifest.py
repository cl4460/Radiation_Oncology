#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd

def scan_phase1_outputs(ct_dir: Path):
    """Return dict pid -> (ct_path, mask_path) for files named:
       {pid}_ct_raw_LPS.nii.gz and {pid}_mask_raw_LPS.nii.gz
    """
    ct_files = sorted(ct_dir.glob("*_ct_raw_LPS.nii.gz"))
    mapping = {}
    for ct in ct_files:
        pid = ct.name.replace("_ct_raw_LPS.nii.gz", "")
        mask = ct_dir / f"{pid}_mask_raw_LPS.nii.gz"
        if mask.exists():
            mapping[pid] = (ct, mask)
    return mapping

def maybe_mask_voxels_ok(mask_path: Path, label: int, min_voxels: int):
    if min_voxels <= 0:
        return True, None
    try:
        import SimpleITK as sitk
        import numpy as np
    except Exception as e:
        raise RuntimeError(
            "min_mask_voxels>0 需要 SimpleITK + numpy。请先安装：pip install SimpleITK numpy"
        ) from e

    m = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(m)
    vox = int((arr == label).sum())
    return vox >= min_voxels, vox

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rg_root", required=True, help="解压后的 Radiogenomics 根目录（包含 phase1_outputs_fixed/ 等）")
    ap.add_argument("--out_manifest", required=True, help="输出 manifest.csv")
    ap.add_argument("--only_has_seg", type=int, default=1, help="1=只保留 seg_check.has_seg==True 的病例")
    ap.add_argument("--min_time_days", type=float, default=1.0, help="过滤 time_days < min_time_days")
    ap.add_argument("--drop_qc_time_nonpositive", type=int, default=1, help="1=丢掉 qc_time_nonpositive==True 的病例（若该列存在）")
    ap.add_argument("--min_mask_voxels", type=int, default=10, help="过滤 mask 体素数过小（<=0 可关闭）")
    ap.add_argument("--mask_label", type=int, default=255, help="mask 标签值（与你的 radiomic_setting.yaml 保持一致，默认 255）")
    args = ap.parse_args()

    rg_root = Path(args.rg_root)
    ct_dir = rg_root / "phase1_outputs_fixed"
    clin_csv = rg_root / "Radiogenomics_clinical_aligned.csv"
    seg_check_csv = rg_root / "radiogenomics_segmentation_check.csv"

    if not ct_dir.exists():
        raise FileNotFoundError(f"Not found: {ct_dir}")
    if not clin_csv.exists():
        raise FileNotFoundError(f"Not found: {clin_csv}")
    if args.only_has_seg and not seg_check_csv.exists():
        raise FileNotFoundError(f"only_has_seg=1 但找不到: {seg_check_csv}")

    clin = pd.read_csv(clin_csv)
    required_cols = {"patient_id", "time_days", "event"}
    missing = required_cols - set(clin.columns)
    if missing:
        raise ValueError(f"{clin_csv} 缺少列: {missing}")

    eligible = set(clin["patient_id"].astype(str))

    # seg filter
    if args.only_has_seg:
        seg = pd.read_csv(seg_check_csv)
        if "patient_id" not in seg.columns or "has_seg" not in seg.columns:
            raise ValueError(f"{seg_check_csv} 需要包含 patient_id 与 has_seg 列")
        seg_yes = set(seg.loc[seg["has_seg"].astype(bool), "patient_id"].astype(str))
        eligible &= seg_yes

    # time filters
    clin2 = clin.copy()
    clin2["patient_id"] = clin2["patient_id"].astype(str)
    clin2 = clin2[clin2["patient_id"].isin(eligible)]
    clin2 = clin2[clin2["time_days"].astype(float) >= float(args.min_time_days)]
    if args.drop_qc_time_nonpositive and "qc_time_nonpositive" in clin2.columns:
        clin2 = clin2[~clin2["qc_time_nonpositive"].astype(bool)]

    # paths
    mapping = scan_phase1_outputs(ct_dir)
    rows = []
    missing_paths = 0
    filtered_empty = 0

    for _, r in clin2.iterrows():
        pid = str(r["patient_id"])
        if pid not in mapping:
            missing_paths += 1
            continue
        ct_path, mask_path = mapping[pid]

        ok, vox = maybe_mask_voxels_ok(mask_path, label=args.mask_label, min_voxels=args.min_mask_voxels)
        if not ok:
            filtered_empty += 1
            continue

        rows.append({
            "patient_id": pid,
            "image_path": str(ct_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "time_days": float(r["time_days"]),
            "event": int(r["event"]),
            "mask_voxels": vox if vox is not None else "",
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_manifest, index=False)

    print("=== RG manifest summary ===")
    print(f"RG clinical rows (after filters): {len(clin2)}")
    print(f"With CT/mask paths             : {len(out)}")
    print(f"Missing CT/mask paths          : {missing_paths}")
    print(f"Filtered by min_mask_voxels    : {filtered_empty}")
    print(f"Saved: {args.out_manifest}")

if __name__ == "__main__":
    main()
