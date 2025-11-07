# phase1_preprocessing_hardpaths.py
RAW_BASE = "/home/lichengze/NSCLC_Radiomics_dataset/NSCLC-Radiomics"  
OUT_DIR  = "/home/lichengze/CNN_pipeline/phase1_outputs"                
ORIENTATION = "LPS"  

import os, re, glob, json, logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import pydicom_seg
from rt_utils import RTStructBuilder

# ---------- logger ----------
def get_logger():
    lg = logging.getLogger("phase1"); 
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        lg.addHandler(h); lg.setLevel(logging.INFO)
    return lg
logger = get_logger()

# ---------- helpers ----------
_GTV1_RE  = re.compile(r"(^|\b)gtv\s*[-_ ]*0*1(\b|[^0-9])", re.I)
_GTVPT_RE = re.compile(r"\bgtv\s*[-_ ]*(p|t)\b", re.I)

def _is_primary_like(s: str) -> bool:
    s = s.lower()
    good = ("gtv","gross","tumor","tumour","primary","lesion","mass")
    bad  = ("ptv","itv","ctv","heart","aorta","mediast","esoph","cord","spine","vertebra","liver","kidney","spleen","ribs","body")
    return any(k in s for k in good) and not any(k in s for k in bad)

def _first_foruid_in_ct_dir(ct_dir: Optional[str]) -> Optional[str]:
    if not ct_dir: 
        return None
    for f in sorted(Path(ct_dir).glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            return str(ds.FrameOfReferenceUID)
        except Exception:
            continue
    return None

def _rtstruct_ref_ct_uid(rtstruct_path: Optional[str]) -> Optional[str]:
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

def _load_ct_series(ct_dir: str) -> sitk.Image:
    files = []
    for f in Path(ct_dir).glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            z = float(ds.ImagePositionPatient[2])
            files.append((str(f), z))
        except Exception:
            continue
    files.sort(key=lambda x: x[1])
    rdr = sitk.ImageSeriesReader()
    rdr.SetFileNames([p for p,_ in files])
    return rdr.Execute()

def classify_series(patient_dir: str) -> Dict[str, Optional[str]]:
    series = {}
    for dcm in Path(patient_dir).rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
            uid = str(ds.SeriesInstanceUID); mod = str(getattr(ds, "Modality", "")).upper()
            series.setdefault(uid, {"modality": mod, "files": [], "dir": str(Path(dcm).parent)})
            series[uid]["files"].append(str(dcm))
        except Exception:
            continue
    rtstruct_file = next((info["files"][0] for uid,info in series.items() if info["modality"]=="RTSTRUCT"), None)
    seg_file      = next((info["files"][0] for uid,info in series.items() if info["modality"] in ("SEG","RTSEG")), None)
    rt_ref_ct_uid = _rtstruct_ref_ct_uid(rtstruct_file) if rtstruct_file else None

    # RTSTRUCT first, CT with most slices otherwise
    ct_dir = None
    if rt_ref_ct_uid and rt_ref_ct_uid in series and series[rt_ref_ct_uid]["modality"]=="CT":
        ct_dir = series[rt_ref_ct_uid]["dir"]
    else:
        max_s = 0
        for uid,info in series.items():
            if info["modality"]=="CT" and len(info["files"])>max_s:
                max_s = len(info["files"])
                ct_dir = info["dir"]

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

    return dict(ct_dir=ct_dir, rtstruct_file=rtstruct_file, seg_file=seg_file,
                ct_foruid=ct_foruid, rt_foruid=rt_foruid,
                foruid_match=(ct_foruid==rt_foruid) if (ct_foruid and rt_foruid) else None)

def extract_mask(ct_dir: str, rtstruct_file: Optional[str], seg_file: Optional[str], ref_ct: sitk.Image) -> Tuple[sitk.Image, dict]:
    # 1) RTSTRUCT
    if rtstruct_file and os.path.exists(rtstruct_file):
        try:
            rt = RTStructBuilder.create_from(dicom_series_path=ct_dir, rt_struct_path=rtstruct_file)
            roi_names = rt.get_roi_names()

            def _pick():
                for n in roi_names:
                    if _GTV1_RE.search(n): 
                        return n, "exact_gtv1"
                for n in roi_names:
                    if _GTVPT_RE.search(n): 
                        return n, "gtv_pt"
                # fallback: volume/lung overlap/largest connected component
                ct_arr = sitk.GetArrayFromImage(ref_ct)
                lung = (ct_arr < -320).astype(np.uint8)
                best, score = None, (-1,-1,-1)
                for n in roi_names:
                    if not _is_primary_like(n): 
                        continue
                    try:
                        m_yxz = rt.get_roi_mask_by_name(n).astype(np.uint8)
                        m = np.moveaxis(m_yxz, -1, 0)  # (Z,Y,X)
                        if m.shape != ct_arr.shape: 
                            logger.warning(f"[fallback] ROI {n} shape{m.shape}!={ct_arr.shape}, skip")
                            continue
                        vox = int(m.sum())
                        if vox==0: 
                            continue
                        lung_ratio = float((lung & (m>0)).sum() / max(1, vox))
                        lab = sitk.ConnectedComponent(sitk.GetImageFromArray((m>0).astype(np.uint8)))
                        stat = sitk.LabelShapeStatisticsImageFilter() 
                        stat.Execute(lab)
                        if not stat.GetLabels(): 
                            continue
                        sizes = [stat.GetNumberOfPixels(l) for l in stat.GetLabels()]
                        lcc = max(sizes)/max(1,vox)
                        sc = (vox, lung_ratio, lcc)
                        if sc > score: 
                            best, score = n, sc
                    except Exception as e:
                        logger.warning(f"[fallback] ROI {n} scoring failed: {e}")
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

    # 2) DICOM-SEG
    if seg_file and os.path.exists(seg_file):
        try:
            ds = pydicom.dcmread(seg_file)
            reader = pydicom_seg.SegmentReader(); res = reader.read(ds)
            seg_id, label = None, None
            for sid in res.available_segments:
                info = res.segment_infos[sid]
                lab  = str(info.get("SegmentLabel","")) 
                desc = str(info.get("SegmentDescription",""))
                if _GTV1_RE.search(lab) or _GTV1_RE.search(desc) or "primary" in lab.lower() or "primary" in desc.lower():
                    seg_id, label = sid, (lab or desc or f"SEG-{sid}") 
                    break
            if seg_id is not None:
                mask_img = sitk.Cast(res.segment_image(seg_id), sitk.sitkUInt8)
                need = (mask_img.GetSize()!=ref_ct.GetSize() or mask_img.GetSpacing()!=ref_ct.GetSpacing())
                if need:
                    rs = sitk.ResampleImageFilter() 
                    rs.SetReferenceImage(ref_ct)
                    rs.SetInterpolator(sitk.sitkNearestNeighbor) 
                    rs.SetDefaultPixelValue(0)
                    rs.SetOutputPixelType(sitk.sitkUInt8) 
                    mask_img = rs.Execute(mask_img)
                return mask_img, dict(method="DICOM-SEG", roi=f"{seg_id}:{label}", select_reason="seg_gtv1_like")
        except Exception as e:
            logger.warning(f"SEG parse failed: {e}")

    # 3) empty mask
    empty = sitk.Image(ref_ct.GetSize(), sitk.sitkUInt8) 
    empty.CopyInformation(ref_ct)
    return empty, dict(method="empty", roi=None, select_reason="no_roi")

def main():
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    pats = sorted(glob.glob(os.path.join(RAW_BASE, "LUNG1-*")))
    qc_rows, ok, fail = [], [], []

    for pdir in pats:
        pid = Path(pdir).name
        try:
            cls = classify_series(pdir)
            if not cls["ct_dir"]: raise RuntimeError("No CT series found")
            ct_img = _load_ct_series(cls["ct_dir"])
            mask_img, info = extract_mask(cls["ct_dir"], cls["rtstruct_file"], cls["seg_file"], ct_img)

            # LPS alignment (once, no resampling/normalization)
            ct_lps   = sitk.DICOMOrient(ct_img, ORIENTATION)
            mask_lps = sitk.DICOMOrient(mask_img, ORIENTATION)

            ct_path  = out_dir / f"{pid}_ct_raw_{ORIENTATION}.nii.gz"
            msk_path = out_dir / f"{pid}_mask_raw_{ORIENTATION}.nii.gz"
            sitk.WriteImage(ct_lps,  str(ct_path))
            sitk.WriteImage(mask_lps, str(msk_path))

            arr = sitk.GetArrayFromImage(mask_lps); vox = int((arr>0).sum())
            vol_cm3 = vox * float(np.prod(mask_lps.GetSpacing()))/1000.0

            qc_rows.append({
                "patient_id": pid,
                "ct_dir": cls["ct_dir"], "rtstruct_file": cls["rtstruct_file"], "seg_file": cls["seg_file"],
                "ct_foruid": cls["ct_foruid"], "rt_foruid": cls["rt_foruid"], "foruid_match": cls["foruid_match"],
                "ct_path": str(ct_path), "mask_path": str(msk_path),
                "mask_voxels": vox, "mask_volume_cm3": vol_cm3,
                "roi": info.get("roi"), "roi_method": info.get("method"), "select_reason": info.get("select_reason")
            })
            ok.append(pid); logger.info(f"[{pid}] ok (vox={vox}, vol={vol_cm3:.2f} cm3, roi={info.get('roi')})")
        except Exception as e:
            fail.append(pid); logger.error(f"[{pid}] failed: {e}")
            qc_rows.append({"patient_id": pid, "processing_status":"failed", "error": str(e)})

    pd.DataFrame(qc_rows).to_csv(out_dir/"phase1_qc.csv", index=False)
    with open(out_dir/"phase1_metadata.json","w") as f:
        json.dump(dict(total=len(pats), success=len(ok), failed=len(fail), out_dir=str(out_dir)), f, indent=2)

    empty = [r["patient_id"] for r in qc_rows if r.get("mask_voxels",1)==0]
    if empty: 
        logger.warning(f"Empty masks: {len(empty)} cases -> {empty[:6]}{'...' if len(empty)>6 else ''}")
    logger.info(f"Phase-1 done: {len(ok)}/{len(pats)} success. QC: {out_dir/'phase1_qc.csv'}")

if __name__ == "__main__":
    main()
