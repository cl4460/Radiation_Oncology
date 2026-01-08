import argparse
import os
import pandas as pd

GROUPS = {
    "shape": ["original_shape_"],
    "firstorder": ["original_firstorder_"],
    "texture": [
        "original_glcm_",
        "original_gldm_",
        "original_glrlm_",
        "original_glszm_",
        "original_ngtdm_",
    ],
    # 你也可以加 "all": None 之类的
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--id_col", default="PatientID")
    ap.add_argument("--group", choices=sorted(GROUPS.keys()), required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    assert args.id_col in df.columns, f"Missing id_col={args.id_col}"

    prefixes = GROUPS[args.group]
    feat_cols = [c for c in df.columns if c != args.id_col]

    keep = []
    for c in feat_cols:
        if any(c.startswith(p) for p in prefixes):
            keep.append(c)

    if len(keep) == 0:
        raise RuntimeError(f"No features matched group={args.group}")

    out = df[[args.id_col] + keep].copy()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"Tb_features_{args.group}.csv")
    out.to_csv(out_path, index=False)

    print(f"[OK] group={args.group} kept {len(keep)} / {len(feat_cols)} features")
    print(f"[OK] wrote {out_path}")

if __name__ == "__main__":
    main()
