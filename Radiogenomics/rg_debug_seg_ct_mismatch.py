#!/usr/bin/env python3
"""
rg_debug_seg_ct_mismatch.py
Diagnose SEG->CT mapping failures caused by wrong CT series selection.
"""

import os, argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import pydicom
import SimpleITK as sitk
from pydicom_seg import SegmentReader


def list_patients(rg_root: Path, patient_ids_file: Optional[str]) -> List[str]:
    if patient_ids_file is None:
        return sorted([p.name for p in rg_root.iterdir() if p.is_dir()])
    df = pd.read_csv(patient_ids_file)
    for col in ["patient_id", "PatientID", "case_id", "id"]:
        if col in df.columns:
            return df[col].astype(str).tolist()
    return df.iloc[:, 0].astype(str).tolist()


def fast_dcmread(path: str):
    return pydicom.dcmread(path, stop_before_pixels=True, force=True)


def find_series(patient_dir: Path) -> Dict[str, dict]:
    series = {}
    for root, _, files in os.walk(patient_dir):
        for fn in files:
            if not fn.lower().endswith(".dcm"):
                continue
            fpath = os.path.join(root, fn)
            try:
                ds = fast_dcmread(fpath)
            except Exception:
                continue

            suid = getattr(ds, "SeriesInstanceUID", None)
            if not suid:
                continue
            modality = getattr(ds, "Modality", "")
            desc = getattr(ds, "SeriesDescription", "")
            frame = getattr(ds, "FrameOfReferenceUID", "")

            if suid not in series:
                series[suid] = {
                    "series_uid": str(suid),
                    "modality": str(modality),
                    "n_files": 0,
                    "sample_path": fpath,
                    "series_desc": str(desc) if desc is not None else "",
                    "frame_uid": str(frame) if frame is not None else "",
                    "dir": str(Path(fpath).parent),
                }
            series[suid]["n_files"] += 1
    return series


def find_seg_files(patient_dir: Path) -> List[str]:
    segs = []
    for root, _, files in os.walk(patient_dir):
        for fn in files:
            if not fn.lower().endswith(".dcm"):
                continue
            fpath = os.path.join(root, fn)
            try:
                ds = fast_dcmread(fpath)
            except Exception:
                continue
            if getattr(ds, "Modality", "") == "SEG":
                segs.append(fpath)
    return sorted(segs)


def get_referenced_series_uids(seg_ds) -> List[str]:
    uids = set()
    if hasattr(seg_ds, "ReferencedSeriesSequence"):
        for rs in seg_ds.ReferencedSeriesSequence:
            if hasattr(rs, "SeriesInstanceUID"):
                uids.add(str(rs.SeriesInstanceUID))
    if hasattr(seg_ds, "ReferencedFrameOfReferenceSequence"):
        for for_item in seg_ds.ReferencedFrameOfReferenceSequence:
            if hasattr(for_item, "RTReferencedStudySequence"):
                for st in for_item.RTReferencedStudySequence:
                    if hasattr(st, "RTReferencedSeriesSequence"):
                        for rs in st.RTReferencedSeriesSequence:
                            if hasattr(rs, "SeriesInstanceUID"):
                                uids.add(str(rs.SeriesInstanceUID))
    return sorted(uids)


def load_ct_volume_from_dir(ct_dir: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(ct_dir)
    if len(files) == 0:
        raise RuntimeError(f"No DICOM files found in: {ct_dir}")
    reader.SetFileNames(files)
    return reader.Execute()


def resample_mask_to_ref(mask_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ref_img)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetTransform(sitk.Transform())
    rs.SetDefaultPixelValue(0)
    rs.SetOutputPixelType(sitk.sitkUInt8)
    return rs.Execute(mask_img)


def compute_lung_overlap(ct_img: sitk.Image, mask_img: sitk.Image) -> float:
    ct = sitk.GetArrayFromImage(ct_img).astype(np.float32)
    mask = sitk.GetArrayFromImage(mask_img) > 0
    vox = int(mask.sum())
    if vox == 0:
        return 0.0
    lung = ct < -320.0
    return float((lung & mask).sum() / (vox + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rg_root", required=True)
    ap.add_argument("--patient_ids", default=None)
    ap.add_argument("--out_csv", default="rg_seg_ct_debug.csv")
    args = ap.parse_args()

    rg_root = Path(args.rg_root)
    pids = list_patients(rg_root, args.patient_ids)

    rows = []
    for pid in pids:
        pdir = rg_root / pid
        if not pdir.exists():
            continue

        series = find_series(pdir)
        seg_files = find_seg_files(pdir)

        ct_series = [s for s in series.values() if s["modality"] == "CT"]
        ct_series_sorted = sorted(ct_series, key=lambda x: x["n_files"], reverse=True)
        heuristic_ct = ct_series_sorted[0] if ct_series_sorted else None

        if len(seg_files) == 0:
            rows.append({
                "patient_id": pid,
                "has_seg_file": False,
                "n_ct_series": len(ct_series_sorted),
                "heuristic_ct_uid": heuristic_ct["series_uid"] if heuristic_ct else "",
                "heuristic_ct_desc": heuristic_ct["series_desc"] if heuristic_ct else "",
                "heuristic_ct_n": heuristic_ct["n_files"] if heuristic_ct else 0,
            })
            continue

        for seg_path in seg_files:
            try:
                seg_ds_full = pydicom.dcmread(seg_path, force=True)
            except Exception as e:
                rows.append({"patient_id": pid, "seg_path": seg_path, "error": f"SEG read fail: {e}"})
                continue

            ref_series_uids = get_referenced_series_uids(seg_ds_full)

            raw_voxels = {}
            seg_labels = {}
            seg_read_error = ""
            try:
                reader = SegmentReader()
                seg_obj = reader.read(seg_ds_full)
                for sid in seg_obj.available_segments:
                    info = seg_obj.segment_infos[sid]
                    seg_labels[int(sid)] = getattr(info, "segment_label", "")
                    img = seg_obj.segment_image(sid)
                    arr = sitk.GetArrayFromImage(img)
                    raw_voxels[int(sid)] = int((arr > 0).sum())
            except Exception as e:
                seg_read_error = str(e)

            best_sid = max(raw_voxels.keys(), key=lambda k: raw_voxels[k]) if raw_voxels else None
            mask_img = None
            if best_sid is not None and raw_voxels.get(best_sid, 0) > 0:
                seg_obj = SegmentReader().read(seg_ds_full)
                mask_img = seg_obj.segment_image(best_sid)

            ref_ct = None
            for uid in ref_series_uids:
                if uid in series and series[uid]["modality"] == "CT":
                    ref_ct = series[uid]
                    break

            mapped_heuristic = ""
            mapped_ref = ""
            overlap_ref = ""

            if heuristic_ct and mask_img is not None:
                try:
                    ct_img = load_ct_volume_from_dir(heuristic_ct["dir"])
                    mask_m = resample_mask_to_ref(mask_img, ct_img)
                    mapped_heuristic = int((sitk.GetArrayFromImage(mask_m) > 0).sum())
                except Exception as e:
                    mapped_heuristic = f"ERR:{e}"

            if ref_ct and mask_img is not None:
                try:
                    ct_img = load_ct_volume_from_dir(ref_ct["dir"])
                    mask_m = resample_mask_to_ref(mask_img, ct_img)
                    mapped_ref = int((sitk.GetArrayFromImage(mask_m) > 0).sum())
                    overlap_ref = compute_lung_overlap(ct_img, mask_m)
                except Exception as e:
                    mapped_ref = f"ERR:{e}"

            rows.append({
                "patient_id": pid,
                "seg_path": seg_path,
                "ref_series_uids": ";".join(ref_series_uids),
                "n_ct_series": len(ct_series_sorted),
                "heuristic_ct_uid": heuristic_ct["series_uid"] if heuristic_ct else "",
                "heuristic_ct_desc": heuristic_ct["series_desc"] if heuristic_ct else "",
                "heuristic_ct_n": heuristic_ct["n_files"] if heuristic_ct else 0,
                "ref_ct_uid_found": ref_ct["series_uid"] if ref_ct else "",
                "ref_ct_desc": ref_ct["series_desc"] if ref_ct else "",
                "ref_ct_n": ref_ct["n_files"] if ref_ct else 0,
                "best_segment_id": best_sid if best_sid is not None else "",
                "best_segment_label": seg_labels.get(best_sid, "") if best_sid is not None else "",
                "raw_voxels_best": raw_voxels.get(best_sid, "") if best_sid is not None else "",
                "mapped_voxels_on_heuristic_ct": mapped_heuristic,
                "mapped_voxels_on_ref_ct": mapped_ref,
                "lung_overlap_ratio_ref_ct": overlap_ref,
                "seg_read_error": seg_read_error,
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
