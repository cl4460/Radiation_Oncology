#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd

def _extract_one(row_dict, yaml_path: str, label: int, include_diagnostics: int):
    # 这里在子进程里 import，避免 joblib pickling extractor 的问题
    from radiomics import featureextractor

    extractor = featureextractor.RadiomicsFeatureExtractor(yaml_path)
    # 强制 label 一致（防止 yaml 里不一致）
    extractor.settings["label"] = int(label)

    result = extractor.execute(row_dict["image_path"], row_dict["mask_path"])
    feat = {}
    for k, v in result.items():
        if (not include_diagnostics) and str(k).startswith("diagnostics"):
            continue
        feat[str(k)] = v
    feat["patient_id"] = row_dict["patient_id"]
    return feat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--yaml", required=True, help="radiomic_setting.yaml")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--mask_label", type=int, default=255)
    ap.add_argument("--include_diagnostics", type=int, default=0)
    args = ap.parse_args()

    man = pd.read_csv(args.manifest_csv)
    need = {"patient_id", "image_path", "mask_path"}
    if not need.issubset(man.columns):
        raise ValueError(f"manifest 需要列: {need}")

    rows = man.to_dict(orient="records")

    if args.n_jobs == 1:
        feats = []
        for r in rows:
            feats.append(_extract_one(r, args.yaml, args.mask_label, args.include_diagnostics))
    else:
        from joblib import Parallel, delayed
        feats = Parallel(n_jobs=args.n_jobs, prefer="processes")(
            delayed(_extract_one)(r, args.yaml, args.mask_label, args.include_diagnostics) for r in rows
        )

    df = pd.DataFrame(feats)
    # 把 patient_id 放第一列
    cols = ["patient_id"] + [c for c in df.columns if c != "patient_id"]
    df = df[cols]
    df.to_csv(args.out_csv, index=False)

    print("=== Radiomics extraction done ===")
    print(f"n_cases: {len(df)}")
    print(f"n_cols : {df.shape[1]}")
    print(f"Saved : {args.out_csv}")

if __name__ == "__main__":
    main()
