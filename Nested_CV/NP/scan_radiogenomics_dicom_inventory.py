#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pydicom
    from pydicom.misc import is_dicom
except ImportError as e:
    raise SystemExit(
        "This script needs pydicom. Install with:\n"
        "  pip install pydicom\n"
        "Then rerun."
    ) from e


@dataclass
class InstanceInfo:
    patient_id: str
    study_uid: str
    series_uid: str
    modality: str
    rows: Optional[int]
    cols: Optional[int]
    pixel_spacing_r: Optional[float]
    pixel_spacing_c: Optional[float]
    slice_thickness: Optional[float]
    image_position_z: Optional[float]
    instance_number: Optional[int]
    manufacturer: str
    convolution_kernel: str


@dataclass
class SeriesSummary:
    patient_id: str
    study_uid: str
    series_uid: str
    modality: str
    n_instances: int
    rows: Optional[int]
    cols: Optional[int]
    pixel_spacing_r: Optional[float]
    pixel_spacing_c: Optional[float]
    slice_thickness: Optional[float]
    slice_spacing_median: Optional[float]
    z_min: Optional[float]
    z_max: Optional[float]
    manufacturer: str
    convolution_kernel: str


def safe_get(ds, key, default=None):
    try:
        return getattr(ds, key, default)
    except Exception:
        return default


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def read_instance_header(fp: Path) -> Optional[InstanceInfo]:
    try:
        ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
    except Exception:
        return None

    modality = str(safe_get(ds, "Modality", "")).strip()
    patient_id = str(safe_get(ds, "PatientID", "")).strip()
    study_uid = str(safe_get(ds, "StudyInstanceUID", "")).strip()
    series_uid = str(safe_get(ds, "SeriesInstanceUID", "")).strip()

    rows = safe_int(safe_get(ds, "Rows", None))
    cols = safe_int(safe_get(ds, "Columns", None))

    ps = safe_get(ds, "PixelSpacing", None)
    ps_r = safe_float(ps[0]) if isinstance(ps, (list, tuple)) and len(ps) >= 2 else None
    ps_c = safe_float(ps[1]) if isinstance(ps, (list, tuple)) and len(ps) >= 2 else None

    thick = safe_float(safe_get(ds, "SliceThickness", None))

    ipp = safe_get(ds, "ImagePositionPatient", None)
    z = safe_float(ipp[2]) if isinstance(ipp, (list, tuple)) and len(ipp) >= 3 else None

    inst_no = safe_int(safe_get(ds, "InstanceNumber", None))

    manufacturer = str(safe_get(ds, "Manufacturer", "")).strip()
    kernel = str(safe_get(ds, "ConvolutionKernel", "")).strip()

    return InstanceInfo(
        patient_id=patient_id,
        study_uid=study_uid,
        series_uid=series_uid,
        modality=modality,
        rows=rows,
        cols=cols,
        pixel_spacing_r=ps_r,
        pixel_spacing_c=ps_c,
        slice_thickness=thick,
        image_position_z=z,
        instance_number=inst_no,
        manufacturer=manufacturer,
        convolution_kernel=kernel,
    )


def summarize_series(instances: List[InstanceInfo]) -> SeriesSummary:
    first = instances[0]
    zs = [ins.image_position_z for ins in instances if ins.image_position_z is not None]
    zs_sorted = sorted(set(zs))

    slice_spacing = None
    z_min = min(zs_sorted) if zs_sorted else None
    z_max = max(zs_sorted) if zs_sorted else None
    if len(zs_sorted) >= 2:
        diffs = [abs(zs_sorted[i+1] - zs_sorted[i]) for i in range(len(zs_sorted)-1)]
        diffs = [d for d in diffs if d > 1e-6]
        if diffs:
            slice_spacing = float(median(diffs))

    return SeriesSummary(
        patient_id=first.patient_id,
        study_uid=first.study_uid,
        series_uid=first.series_uid,
        modality=first.modality,
        n_instances=len(instances),
        rows=first.rows,
        cols=first.cols,
        pixel_spacing_r=first.pixel_spacing_r,
        pixel_spacing_c=first.pixel_spacing_c,
        slice_thickness=first.slice_thickness,
        slice_spacing_median=slice_spacing,
        z_min=z_min,
        z_max=z_max,
        manufacturer=first.manufacturer,
        convolution_kernel=first.convolution_kernel,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom_root", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    ap.add_argument("--max_files", type=int, default=0, help="0 means no limit")
    args = ap.parse_args()

    root: Path = args.dicom_root
    if not root.exists():
        raise SystemExit(f"dicom_root not found: {root}")

    # collect candidate files
    candidates: List[Path] = []
    for fp in root.rglob("*"):
        if fp.is_file():
            try:
                if is_dicom(str(fp)):
                    candidates.append(fp)
            except Exception:
                continue
        if args.max_files and len(candidates) >= args.max_files:
            break

    print(f"Found DICOM candidates: {len(candidates)}")

    series_map: Dict[Tuple[str, str, str], List[InstanceInfo]] = defaultdict(list)

    for i, fp in enumerate(candidates, 1):
        info = read_instance_header(fp)
        if info is None:
            continue
        key = (info.patient_id, info.study_uid, info.series_uid)
        series_map[key].append(info)

        if i % 2000 == 0:
            print(f"Processed {i}/{len(candidates)}")

    # summarize
    summaries: List[SeriesSummary] = []
    for key, insts in series_map.items():
        # skip weird entries with missing uids
        if not key[0] or not key[1] or not key[2]:
            continue
        summaries.append(summarize_series(insts))

    summaries.sort(key=lambda x: (x.patient_id, x.modality, x.n_instances), reverse=False)

    # write csv
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()) if summaries else [])
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))

    print(f"Saved series inventory: {args.out_csv}")

    # quick stats
    ct = [s for s in summaries if s.modality == "CT"]
    pt = [s for s in summaries if s.modality == "PT"]
    seg = [s for s in summaries if s.modality == "SEG"]

    print(f"Series counts: CT={len(ct)} PT={len(pt)} SEG={len(seg)}")
    if ct:
        spacings = [s.slice_spacing_median for s in ct if s.slice_spacing_median is not None]
        if spacings:
            print(f"CT slice spacing median (mm): median={float(np.median(spacings)):.3f}, "
                  f"p10={float(np.quantile(spacings,0.1)):.3f}, p90={float(np.quantile(spacings,0.9)):.3f}")
        inplane = [(s.pixel_spacing_r, s.pixel_spacing_c) for s in ct if s.pixel_spacing_r and s.pixel_spacing_c]
        if inplane:
            rs = [x[0] for x in inplane]; cs = [x[1] for x in inplane]
            print(f"CT in-plane spacing (mm): r median={float(np.median(rs)):.3f}, c median={float(np.median(cs)):.3f}")

if __name__ == "__main__":
    main()
