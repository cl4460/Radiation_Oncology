#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a PyRadiomics path CSV (PatientID,image_path,mask_path) from your Radiogenomics crop_log.

Example:
python make_rg_pathcsv.py \
  --crop_log /home/lichengze/Research/Radiogenomics/phase2_outputs/phase2_crop_log.csv \
  --out_csv  /home/lichengze/Research/Radiogenomics/rg_paths_for_radiomics1688.csv
"""

import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop_log", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.crop_log)
    # expected columns in crop_log: patient_id, out_ct, out_mask
    need = ["patient_id", "out_ct", "out_mask"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"crop_log missing column: {c}. columns={df.columns.tolist()}")

    out = df[need].rename(columns={"patient_id":"PatientID","out_ct":"image_path","out_mask":"mask_path"})
    out.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote {args.out_csv}  rows={len(out)}")

if __name__ == "__main__":
    main()
