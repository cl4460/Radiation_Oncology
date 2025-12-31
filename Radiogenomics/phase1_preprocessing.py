#!/usr/bin/env python3
"""
phase1_preprocessing_rg_fixed.py
================================
FIXED VERSION for NSCLC-Radiogenomics dataset

关键修复：CT Series选择策略
---------------------------
原问题：代码选择slice最多的CT，但SEG文件可能引用的是另一个CT series
       导致mask和CT不匹配，最终mask为空

修复：优先使用SEG引用的CT series
     CT选择优先级: SEG引用 > RTSTRUCT引用 > slice最多(fallback)

预期效果：133例 → 143例 (救回~10例)
"""

RAW_BASE = "/data0/lichengze/NSCLC_Radiogenomics/NSCLC Radiogenomics"  
OUT_DIR = "/home/lichengze/Research/Radiogenomics/phase1_outputs_fixed"  # 使用新目录避免覆盖
ORIENTATION = "LPS"

import os
import re
import glob
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import pydicom_seg
from rt_utils import RTStructBuilder

# ---------- Logger ----------
def get_logger():
    lg = logging.getLogger("phase1_fixed")
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg

logger = get_logger()


# ---------- ROI Selection Helpers ----------
_GTV1_RE = re.compile(r"(^|\b)gtv\s*[-_ ]*0*1(\b|[^0-9])", re.I)
_GTVPT_RE = re.compile(r"\bgtv\s*[-_ ]*(p|t)\b", re.I)


def _is_primary_like(s: str) -> bool:
    """Check if ROI name suggests primary tumor."""
    s = s.lower()
    good = ("gtv", "gross", "tumor", "tumour", "primary", "lesion", "mass")
    bad = ("ptv", "itv", "ctv", "heart", "aorta", "mediast", "esoph", "cord", "spine",
           "vertebra", "liver", "kidney", "spleen", "ribs", "body")
    return any(k in s for k in good) and not any(k in s for k in bad)


# ==========================================================================
# 关键修复：获取SEG文件引用的CT Series UID
# ==========================================================================
def get_seg_referenced_series_uids(seg_path: str) -> Set[str]:
    """
    从SEG文件中提取它引用的CT Series UID列表
    
    SEG文件通过 ReferencedSeriesSequence 记录它是基于哪个CT series创建的
    这是修复的关键 - 我们需要使用SEG引用的CT，而不是slice最多的CT
    """
    uids = set()
    try:
        ds = pydicom.dcmread(seg_path, stop_before_pixels=True)
        
        # 方法1: 直接从 ReferencedSeriesSequence 获取
        if hasattr(ds, 'ReferencedSeriesSequence'):
            for ref_series in ds.ReferencedSeriesSequence:
                if hasattr(ref_series, 'SeriesInstanceUID'):
                    uids.add(str(ref_series.SeriesInstanceUID))
        
        # 方法2: 从 SharedFunctionalGroupsSequence 获取
        if hasattr(ds, 'SharedFunctionalGroupsSequence'):
            for sfg in ds.SharedFunctionalGroupsSequence:
                if hasattr(sfg, 'DerivationImageSequence'):
                    for deriv in sfg.DerivationImageSequence:
                        if hasattr(deriv, 'SourceImageSequence'):
                            for src in deriv.SourceImageSequence:
                                if hasattr(src, 'ReferencedSeriesSequence'):
                                    for rs in src.ReferencedSeriesSequence:
                                        if hasattr(rs, 'SeriesInstanceUID'):
                                            uids.add(str(rs.SeriesInstanceUID))
        
        # 方法3: 从 PerFrameFunctionalGroupsSequence 获取 (逐帧引用)
        if hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
            for pffg in ds.PerFrameFunctionalGroupsSequence:
                if hasattr(pffg, 'DerivationImageSequence'):
                    for deriv in pffg.DerivationImageSequence:
                        if hasattr(deriv, 'SourceImageSequence'):
                            for src in deriv.SourceImageSequence:
                                # 有时候是 ReferencedSOPInstanceUID，需要通过它找series
                                pass  # 这种情况较复杂，先跳过
        
        # 方法4: 从 ReferencedImageEvidenceSequence 获取
        if hasattr(ds, 'ReferencedImageEvidenceSequence'):
            for ref in ds.ReferencedImageEvidenceSequence:
                if hasattr(ref, 'ReferencedSeriesSequence'):
                    for rs in ref.ReferencedSeriesSequence:
                        if hasattr(rs, 'SeriesInstanceUID'):
                            uids.add(str(rs.SeriesInstanceUID))
                            
    except Exception as e:
        logger.warning(f"Failed to extract SEG referenced UIDs: {e}")
    
    return uids


def get_rtstruct_referenced_series_uid(rtstruct_path: str) -> Optional[str]:
    """从RTSTRUCT文件获取引用的CT Series UID"""
    if not rtstruct_path or not os.path.exists(rtstruct_path):
        return None
    try:
        ds = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
        for frame in getattr(ds, "ReferencedFrameOfReferenceSequence", []):
            for study in getattr(frame, "RTReferencedStudySequence", []):
                for series in getattr(study, "RTReferencedSeriesSequence", []):
                    return str(series.SeriesInstanceUID)
    except Exception:
        pass
    return None


def _first_foruid_in_ct_dir(ct_dir: Optional[str]) -> Optional[str]:
    """获取CT目录中的FrameOfReferenceUID"""
    if not ct_dir:
        return None
    for f in sorted(Path(ct_dir).glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            return str(ds.FrameOfReferenceUID)
        except Exception:
            continue
    for f in sorted(Path(ct_dir).iterdir()):
        if f.is_file() and not f.name.startswith('.'):
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                if hasattr(ds, 'FrameOfReferenceUID'):
                    return str(ds.FrameOfReferenceUID)
            except Exception:
                continue
    return None


# ==========================================================================
# 修复后的 classify_series 函数
# ==========================================================================
def classify_series(patient_dir: str) -> Dict:
    """
    分类患者目录中的所有DICOM series
    
    修复点：CT选择优先级
    1. SEG引用的CT (最重要！因为mask是基于这个CT创建的)
    2. RTSTRUCT引用的CT
    3. slice最多的CT (fallback)
    """
    series = {}
    
    # 扫描所有DICOM文件
    for dcm in Path(patient_dir).rglob("*"):
        if not dcm.is_file() or dcm.name.startswith('.'):
            continue
        
        try:
            ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
            if not hasattr(ds, 'SeriesInstanceUID'):
                continue
            
            uid = str(ds.SeriesInstanceUID)
            mod = str(getattr(ds, "Modality", "")).upper()
            
            series.setdefault(uid, {
                "modality": mod,
                "files": [],
                "dir": str(Path(dcm).parent)
            })
            series[uid]["files"].append(str(dcm))
        except Exception:
            continue
    
    # 找到RTSTRUCT和SEG文件
    rtstruct_file = next((info["files"][0] for uid, info in series.items()
                          if info["modality"] == "RTSTRUCT"), None)
    seg_file = next((info["files"][0] for uid, info in series.items()
                     if info["modality"] in ("SEG", "RTSEG")), None)
    
    # ==========================================================================
    # 修复的核心逻辑：CT选择
    # ==========================================================================
    ct_dir = None
    ct_selection_method = None
    
    # 收集所有CT series
    ct_series = {uid: info for uid, info in series.items() if info["modality"] == "CT"}
    
    if not ct_series:
        logger.warning(f"No CT series found in {patient_dir}")
        return dict(
            ct_dir=None,
            rtstruct_file=rtstruct_file,
            seg_file=seg_file,
            ct_foruid=None,
            rt_foruid=None,
            foruid_match=None,
            ct_selection_method="no_ct_found"
        )
    
    # 优先级1: SEG引用的CT
    if seg_file:
        seg_ref_uids = get_seg_referenced_series_uids(seg_file)
        logger.info(f"SEG references CT UIDs: {seg_ref_uids}")
        
        for ref_uid in seg_ref_uids:
            if ref_uid in ct_series:
                ct_dir = ct_series[ref_uid]["dir"]
                ct_selection_method = "seg_referenced"
                logger.info(f"Using SEG-referenced CT: {ref_uid}")
                break
    
    # 优先级2: RTSTRUCT引用的CT
    if ct_dir is None and rtstruct_file:
        rt_ref_uid = get_rtstruct_referenced_series_uid(rtstruct_file)
        if rt_ref_uid and rt_ref_uid in ct_series:
            ct_dir = ct_series[rt_ref_uid]["dir"]
            ct_selection_method = "rtstruct_referenced"
            logger.info(f"Using RTSTRUCT-referenced CT: {rt_ref_uid}")
    
    # 优先级3: Fallback - slice最多的CT
    if ct_dir is None:
        max_slices = 0
        for uid, info in ct_series.items():
            if len(info["files"]) > max_slices:
                max_slices = len(info["files"])
                ct_dir = info["dir"]
        ct_selection_method = "max_slices_fallback"
        logger.warning(f"Using fallback (max slices) CT selection")
    
    # 获取FrameOfReferenceUID用于验证
    ct_foruid = _first_foruid_in_ct_dir(ct_dir)
    rt_foruid = None
    if rtstruct_file:
        try:
            ds = pydicom.dcmread(rtstruct_file, stop_before_pixels=True)
            seq = getattr(ds, "ReferencedFrameOfReferenceSequence", [])
            if seq:
                rt_foruid = str(seq[0].FrameOfReferenceUID)
        except Exception:
            pass
    
    return dict(
        ct_dir=ct_dir,
        rtstruct_file=rtstruct_file,
        seg_file=seg_file,
        ct_foruid=ct_foruid,
        rt_foruid=rt_foruid,
        foruid_match=(ct_foruid == rt_foruid) if (ct_foruid and rt_foruid) else None,
        ct_selection_method=ct_selection_method  # 新增：记录CT选择方法
    )


def _load_ct_series(ct_dir: str) -> sitk.Image:
    """Load CT series from directory"""
    files = []
    
    dcm_files = list(Path(ct_dir).glob("*.dcm"))
    if not dcm_files:
        dcm_files = [f for f in Path(ct_dir).iterdir()
                     if f.is_file() and not f.name.startswith('.')]
    
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            if hasattr(ds, 'ImagePositionPatient'):
                z = float(ds.ImagePositionPatient[2])
                files.append((str(f), z))
        except Exception:
            continue
    
    if not files:
        raise RuntimeError(f"No valid DICOM CT files found in {ct_dir}")
    
    files.sort(key=lambda x: x[1])
    rdr = sitk.ImageSeriesReader()
    rdr.SetFileNames([p for p, _ in files])
    return rdr.Execute()


def extract_mask(ct_dir: str, rtstruct_file: Optional[str],
                 seg_file: Optional[str], ref_ct: sitk.Image) -> Tuple[sitk.Image, dict]:
    """Extract tumor mask from RTSTRUCT, SEG, or create empty"""
    
    # 1) Try RTSTRUCT first
    if rtstruct_file and os.path.exists(rtstruct_file):
        try:
            rt = RTStructBuilder.create_from(
                dicom_series_path=ct_dir,
                rt_struct_path=rtstruct_file
            )
            roi_names = rt.get_roi_names()
            
            def _pick():
                for n in roi_names:
                    if _GTV1_RE.search(n):
                        return n, "exact_gtv1"
                for n in roi_names:
                    if _GTVPT_RE.search(n):
                        return n, "gtv_pt"
                ct_arr = sitk.GetArrayFromImage(ref_ct)
                lung = (ct_arr < -320).astype(np.uint8)
                best, score = None, (-1, -1, -1)
                
                for n in roi_names:
                    if not _is_primary_like(n):
                        continue
                    try:
                        m_yxz = rt.get_roi_mask_by_name(n).astype(np.uint8)
                        m = np.moveaxis(m_yxz, -1, 0)
                        if m.shape != ct_arr.shape:
                            continue
                        vox = int(m.sum())
                        if vox == 0:
                            continue
                        lung_ratio = float((lung & (m > 0)).sum() / max(1, vox))
                        
                        lab = sitk.ConnectedComponent(
                            sitk.GetImageFromArray((m > 0).astype(np.uint8))
                        )
                        stat = sitk.LabelShapeStatisticsImageFilter()
                        stat.Execute(lab)
                        if not stat.GetLabels():
                            continue
                        sizes = [stat.GetNumberOfPixels(l) for l in stat.GetLabels()]
                        lcc = max(sizes) / max(1, vox)
                        
                        sc = (vox, lung_ratio, lcc)
                        if sc > score:
                            best, score = n, sc
                    except Exception:
                        continue
                
                return best, "fallback_scored" if best else None
            
            name, reason = _pick()
            if name:
                m_yxz = rt.get_roi_mask_by_name(name).astype(np.uint8)
                m = np.moveaxis(m_yxz, -1, 0)
                mask = sitk.GetImageFromArray(m.astype(np.uint8))
                mask.CopyInformation(ref_ct)
                return mask, dict(method="RTSTRUCT", roi=name, select_reason=reason)
        except Exception as e:
            logger.warning(f"RTSTRUCT parse failed: {e}")
    
    # 2) Try DICOM-SEG
    if seg_file and os.path.exists(seg_file):
        try:
            ds = pydicom.dcmread(seg_file)
            reader = pydicom_seg.SegmentReader()
            res = reader.read(ds)
            
            seg_id, label = None, None
            
            # If only 1 segment, use it directly
            if len(res.available_segments) == 1:
                seg_id = list(res.available_segments)[0]
                info = res.segment_infos[seg_id]
                lab = str(info.get("SegmentLabel", ""))
                desc = str(info.get("SegmentDescription", ""))
                label = lab or desc or f"SEG-{seg_id}"
                logger.info(f"Using single segment: {label}")
            else:
                # Multiple segments - find tumor-like one
                for sid in res.available_segments:
                    info = res.segment_infos[sid]
                    lab = str(info.get("SegmentLabel", ""))
                    desc = str(info.get("SegmentDescription", ""))
                    if (_GTV1_RE.search(lab) or _GTV1_RE.search(desc) or
                        "primary" in lab.lower() or "primary" in desc.lower() or
                        "tumor" in lab.lower() or "tumor" in desc.lower()):
                        seg_id, label = sid, (lab or desc or f"SEG-{sid}")
                        break
                
                # Fallback: use first segment
                if seg_id is None:
                    seg_id = list(res.available_segments)[0]
                    info = res.segment_infos[seg_id]
                    lab = str(info.get("SegmentLabel", ""))
                    label = lab or f"SEG-{seg_id}"
                    logger.warning(f"No tumor-like segment found, using first: {label}")
            
            if seg_id is not None:
                mask_img = sitk.Cast(res.segment_image(seg_id), sitk.sitkUInt8)
                
                # Resample if needed
                need = (mask_img.GetSize() != ref_ct.GetSize() or
                        mask_img.GetSpacing() != ref_ct.GetSpacing())
                if need:
                    rs = sitk.ResampleImageFilter()
                    rs.SetReferenceImage(ref_ct)
                    rs.SetInterpolator(sitk.sitkNearestNeighbor)
                    rs.SetDefaultPixelValue(0)
                    rs.SetOutputPixelType(sitk.sitkUInt8)
                    mask_img = rs.Execute(mask_img)
                
                return mask_img, dict(
                    method="DICOM-SEG",
                    roi=f"{seg_id}:{label}",
                    select_reason="single_segment" if len(res.available_segments) == 1 else "multi_segment_matched"
                )
        except Exception as e:
            logger.warning(f"SEG parse failed: {e}")
    
    # 3) No segmentation found - create empty mask
    empty = sitk.Image(ref_ct.GetSize(), sitk.sitkUInt8)
    empty.CopyInformation(ref_ct)
    return empty, dict(method="empty", roi=None, select_reason="no_segmentation")


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pats_amc = sorted(glob.glob(os.path.join(RAW_BASE, "AMC-*")))
    pats_r01 = sorted(glob.glob(os.path.join(RAW_BASE, "R01-*")))
    pats = pats_amc + pats_r01
    
    logger.info(f"Found {len(pats_amc)} AMC-* and {len(pats_r01)} R01-* patient directories")
    logger.info(f"Total patients to process: {len(pats)}")
    
    qc_rows, ok, fail, empty_masks = [], [], [], []
    ct_selection_stats = {"seg_referenced": 0, "rtstruct_referenced": 0, "max_slices_fallback": 0}
    
    for idx, pdir in enumerate(pats, 1):
        pid = Path(pdir).name
        
        try:
            cls = classify_series(pdir)
            
            if not cls["ct_dir"]:
                raise RuntimeError("No CT series found")
            
            # 记录CT选择方法统计
            method = cls.get("ct_selection_method", "unknown")
            if method in ct_selection_stats:
                ct_selection_stats[method] += 1
            
            ct_img = _load_ct_series(cls["ct_dir"])
            
            mask_img, info = extract_mask(
                cls["ct_dir"],
                cls["rtstruct_file"],
                cls["seg_file"],
                ct_img
            )
            
            ct_lps = sitk.DICOMOrient(ct_img, ORIENTATION)
            mask_lps = sitk.DICOMOrient(mask_img, ORIENTATION)
            
            ct_path = out_dir / f"{pid}_ct_raw_{ORIENTATION}.nii.gz"
            msk_path = out_dir / f"{pid}_mask_raw_{ORIENTATION}.nii.gz"
            sitk.WriteImage(ct_lps, str(ct_path))
            sitk.WriteImage(mask_lps, str(msk_path))
            
            arr = sitk.GetArrayFromImage(mask_lps)
            vox = int((arr > 0).sum())
            vol_cm3 = vox * float(np.prod(mask_lps.GetSpacing())) / 1000.0
            
            if vox == 0:
                empty_masks.append(pid)
            
            qc_rows.append({
                "patient_id": pid,
                "ct_dir": cls["ct_dir"],
                "rtstruct_file": cls["rtstruct_file"],
                "seg_file": cls["seg_file"],
                "ct_foruid": cls["ct_foruid"],
                "rt_foruid": cls["rt_foruid"],
                "foruid_match": cls["foruid_match"],
                "ct_selection_method": cls.get("ct_selection_method"),  # 新增列
                "ct_path": str(ct_path),
                "mask_path": str(msk_path),
                "mask_voxels": vox,
                "mask_volume_cm3": vol_cm3,
                "roi": info.get("roi"),
                "roi_method": info.get("method"),
                "select_reason": info.get("select_reason"),
                "processing_status": "success"
            })
            
            ok.append(pid)
            
            if idx % 20 == 0:
                logger.info(f"Progress: {idx}/{len(pats)} ({idx / len(pats) * 100:.1f}%)")
        
        except Exception as e:
            fail.append(pid)
            logger.error(f"[{pid}] failed: {e}")
            qc_rows.append({
                "patient_id": pid,
                "processing_status": "failed",
                "error": str(e)
            })
    
    pd.DataFrame(qc_rows).to_csv(out_dir / "phase1_qc.csv", index=False)
    
    with open(out_dir / "phase1_metadata.json", "w") as f:
        json.dump({
            "total": len(pats),
            "success": len(ok),
            "failed": len(fail),
            "empty_masks": len(empty_masks),
            "with_segmentation": len(ok) - len(empty_masks),
            "ct_selection_stats": ct_selection_stats,  # 新增统计
            "out_dir": str(out_dir)
        }, f, indent=2)
    
    logger.info("=" * 70)
    logger.info(f"Phase-1 Complete: {len(ok)}/{len(pats)} processed successfully")
    logger.info(f"  With segmentations: {len(ok) - len(empty_masks)}")
    logger.info(f"  Empty masks: {len(empty_masks)}")
    logger.info(f"  Failed: {len(fail)}")
    logger.info(f"  CT Selection Stats: {ct_selection_stats}")
    logger.info(f"QC report: {out_dir / 'phase1_qc.csv'}")
    
    if empty_masks:
        logger.warning(f"\n{len(empty_masks)} patients have NO segmentations:")
        pd.DataFrame({'patient_id': empty_masks}).to_csv(
            out_dir / "patients_without_segmentations.csv",
            index=False
        )
    
    logger.info("=" * 70)


if __name__ == "__main__":
    main()