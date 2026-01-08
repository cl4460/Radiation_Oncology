
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_deep_csv_from_npy.py

Convert deep embeddings saved as .npy into a CSV with PatientID as key.

Typical usage:
python make_deep_csv_from_npy.py \
  --emb_npy path/to/embeddings.npy \
  --ids_txt path/to/patient_ids.txt \
  --out_csv deep_resnet10_512.csv

Where:
- embeddings.npy: shape (N, D) float
- patient_ids.txt: N lines, each is a PatientID (string) in the same order as embeddings.npy

If you don't have ids_txt but you have clinical_csv and you know the embedding row order matches
clinical_csv row order, you can do:
python make_deep_csv_from_npy.py --emb_npy embeddings.npy --clinical_csv clinical.csv --out_csv deep.csv

(Only use this fallback if you are 100% sure the order matches.)
"""
import argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npy", required=True, help="Path to embeddings.npy of shape (N,D)")
    ap.add_argument("--ids_txt", default=None, help="Text file with N lines of PatientID")
    ap.add_argument("--ids_npy", default=None, help="Optional npy with PatientID array of shape (N,)")
    ap.add_argument("--clinical_csv", default=None, help="Fallback: use PatientID order from this CSV")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    emb = np.load(args.emb_npy)
    if emb.ndim != 2:
        raise ValueError(f"Expected emb_npy to be 2D (N,D). Got shape={emb.shape}")
    N, D = emb.shape
a a
    ids = None
    if args.ids_npy:
        ids = np.load(args.ids_npy)
        ids = [str(x) for x in ids.tolist()]
    elif args.ids_txt:
        with open(args.ids_txt, "r") as f:
            ids = [line.strip() for line in f if line.strip() != ""]
    elif args.clinical_csv:
        clin = pd.read_csv(args.clinical_csv)
        if "PatientID" not in clin.columns:
            raise ValueError("clinical_csv must contain PatientID column")
        ids = clin["PatientID"].astype(str).tolist()
    else:
        raise ValueError("Provide one of --ids_txt / --ids_npy / --clinical_csv to map rows to PatientID.")

    if len(ids) != N:
        raise ValueError(f"IDs length {len(ids)} does not match embeddings N={N}.")

    cols = ["PatientID"] + [f"d{i}" for i in range(D)]
    out = pd.DataFrame(np.column_stack([np.array(ids, dtype=object), emb]), columns=cols)
    # ensure numeric for embedding columns
    for c in cols[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}  (N={N}, D={D})")

if __name__ == "__main__":
    main()
