#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from pydicom.dataset import Dataset


def safe_get(ds: Dataset, attr: str, default=None):
    return getattr(ds, attr, default)


def find_first_dicom_file(folder: Path) -> Optional[Path]:
    for p in folder.rglob("*.dcm"):
        return p
    return None


def read_modality_and_uids(dcm_path: Path) -> Tuple[str, str, str, str]:
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
    modality = str(safe_get(ds, "Modality", ""))
    series_uid = str(safe_get(ds, "SeriesInstanceUID", ""))
    sop_uid = str(safe_get(ds, "SOPInstanceUID", ""))
    series_desc = str(safe_get(ds, "SeriesDescription", ""))
    return modality, series_uid, sop_uid, series_desc


def collect_series(patient_dir: Path) -> Dict[str, Dict]:
    """
    Traverse leaf folders and group by SeriesInstanceUID.
    Returns dict: series_uid -> {modality, desc, files(list)}
    """
    series: Dict[str, Dict] = {}
    # Radiogenomics 目录通常是 patient/Study/Series/...dcm
    # 我们粗暴遍历所有 dcm，按 SeriesInstanceUID 聚合（足够稳健，但会慢一些）
    for dcm_path in patient_dir.rglob("*.dcm"):
        try:
            modality, series_uid, sop_uid, desc = read_modality_and_uids(dcm_path)
        except Exception:
            continue
        if not series_uid:
            continue
        if series_uid not in series:
            series[series_uid] = {
                "modality": modality,
                "desc": desc,
                "files": [],
            }
        series[series_uid]["files"].append(str(dcm_path))
    # 补充 slice 数（对 CT/PT 有意义）
    for uid, info in series.items():
        info["n_files"] = len(info["files"])
    return series


def extract_seg_segment_metadata(seg_ds: Dataset) -> List[Dict]:
    out = []
    seg_seq = safe_get(seg_ds, "SegmentSequence", None)
    if not seg_seq:
        return out

    for i, seg in enumerate(seg_seq, start=1):
        # SegmentLabel / Description
        seg_label = str(safe_get(seg, "SegmentLabel", ""))
        seg_desc = str(safe_get(seg, "SegmentDescription", ""))

        # Code sequences (may not exist)
        def code_meaning(seq_name: str) -> str:
            seq = safe_get(seg, seq_name, None)
            if seq and len(seq) > 0:
                return str(safe_get(seq[0], "CodeMeaning", ""))
            return ""

        prop_type = code_meaning("SegmentedPropertyTypeCodeSequence")
        prop_cat  = code_meaning("SegmentedPropertyCategoryCodeSequence")
        anat_reg  = code_meaning("AnatomicRegionSequence")

        out.append({
            "segment_id": i,
            "SegmentLabel": seg_label,
            "SegmentDescription": seg_desc,
            "PropertyTypeCodeMeaning": prop_type,
            "PropertyCategoryCodeMeaning": prop_cat,
            "AnatomicRegionCodeMeaning": anat_reg,
        })
    return out


def seg_referenced_series_uids(seg_ds: Dataset) -> List[str]:
    """
    Robust-ish extraction:
    Walk all sequences; whenever we see a Dataset containing SeriesInstanceUID together
    with a 'ReferencedSeriesSequence' context, we collect it.

    Note: DICOM SEG referencing is messy; this covers the common patterns.
    """
    uids: Set[str] = set()

    def walk(ds: Dataset):
        for elem in ds:
            if elem.VR == "SQ":
                for item in elem.value:
                    if isinstance(item, Dataset):
                        if hasattr(item, "SeriesInstanceUID"):
                            uids.add(str(item.SeriesInstanceUID))
                        walk(item)

    walk(seg_ds)
    return sorted(list(uids))


def read_ct_as_sitk(ct_files: List[str]) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ct_files)
    return reader.Execute()


def seg_to_mask_sitk(seg_path: Path) -> Tuple[sitk.Image, Dict]:
    """
    Use SimpleITK ReadImage on SEG is NOT correct.
    Here we do a pragmatic fallback:
    - We read SEG via pydicom, then try to decode with SimpleITK's ImageFileReader may fail.
    - In你的 pipeline里你用的是 pydicom_seg.SegmentReader，这里也建议用它。
    但为了不给你引入新的依赖差异，这个函数只返回 placeholder。
    你现在已有 rg_debug_seg_ct_mismatch.py 能正确算 raw/mapped voxels，
    所以我更推荐你在你那份脚本基础上把 metadata 写出来。

    这里我先抛出 NotImplementedError，避免你误以为它能直接跑完 voxel 映射。
    """
    raise NotImplementedError(
        "请在你现有的 rg_debug_seg_ct_mismatch.py 的 voxel 计算逻辑基础上，"
        "把 extract_seg_segment_metadata() 和 seg_referenced_series_uids() 的输出写入 CSV。"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rg_root", type=str, required=True,
                    help="Radiogenomics 根目录（包含 R01-xxx 文件夹那层）")
    ap.add_argument("--patient_ids_csv", type=str, default="",
                    help="可选：只检查指定 patient_id 列表（csv 需包含 patient_id 列）")
    ap.add_argument("--out_csv", type=str, default="rg_seg_ct_debug_v2_meta_only.csv")
    args = ap.parse_args()

    rg_root = Path(args.rg_root)
    assert rg_root.exists(), f"Not found: {rg_root}"

    restrict_ids: Optional[Set[str]] = None
    if args.patient_ids_csv:
        df_ids = pd.read_csv(args.patient_ids_csv)
        restrict_ids = set(df_ids["patient_id"].astype(str).tolist())

    rows = []
    for patient_dir in sorted([p for p in rg_root.iterdir() if p.is_dir()]):
        pid = patient_dir.name
        if restrict_ids is not None and pid not in restrict_ids:
            continue

        series = collect_series(patient_dir)
        ct_series = {uid: info for uid, info in series.items() if info["modality"] == "CT"}
        seg_series = {uid: info for uid, info in series.items() if info["modality"] == "SEG"}

        # pick one SEG file (if multiple, you应该逐个检查)
        seg_file = None
        seg_uid = None
        if seg_series:
            seg_uid = sorted(seg_series.keys())[0]
            seg_file = Path(seg_series[seg_uid]["files"][0])

        if not seg_file:
            continue

        try:
            seg_ds = pydicom.dcmread(str(seg_file), stop_before_pixels=True, force=True)
        except Exception as e:
            rows.append({"patient_id": pid, "seg_file": str(seg_file), "error": f"read_seg_failed: {e}"})
            continue

        seg_meta = extract_seg_segment_metadata(seg_ds)
        ref_uids = seg_referenced_series_uids(seg_ds)

        rows.append({
            "patient_id": pid,
            "seg_file": str(seg_file),
            "seg_series_uid": seg_uid,
            "n_ct_series": len(ct_series),
            "n_seg_series": len(seg_series),
            "seg_segments": json.dumps(seg_meta, ensure_ascii=False),
            "seg_ref_series_uids": json.dumps(ref_uids, ensure_ascii=False),
            # 这里先只输出 metadata；voxel 映射建议你把这段逻辑合并进你已有的脚本里
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}  (rows={len(out)})")


if __name__ == "__main__":
    main()
