#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lock a fixed Train/Val/Test split for NSCLC-Radiomics (Lung1) clinical CSV.

Design goals:
- No cherry-picking: one deterministic seed -> one split.
- Stratify on (event x stage) to keep event rate + stage distribution consistent.
- Val set not too small: default Train/Val/Test = 0.65/0.15/0.20.
- Save both JSON (IDs + metadata) and CSV (id->split) for full reproducibility.

Example:
  python step1_lock_split_lung1.py \
    --clinical_csv "/path/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv" \
    --out_dir "./splits" \
    --seed 42 \
    --test_size 0.20 \
    --val_size 0.15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def canonicalize_stage(x) -> str:
    """Map various stage strings to a canonical set: I, II, IIIA, IIIB, IV, 0, UNK."""
    if pd.isna(x):
        return "UNK"
    s = str(x).strip()
    if s == "":
        return "UNK"
    s = s.upper()
    s = s.replace("STAGE", "")
    s = s.replace(" ", "")
    s = re.sub(r"[^A-Z0-9]", "", s)

    if s in {"TIS", "IS", "0"}:
        return "0"
    if s in {"I", "II", "III", "IV", "IIIA", "IIIB"}:
        return s
    if s in {"1", "2", "3", "4"}:
        return {"1": "I", "2": "II", "3": "III", "4": "IV"}[s]

    m = re.match(r"^(I{1,3}|IV)(A|B)?$", s)
    if m:
        base = m.group(1)
        suf = m.group(2) or ""
        return base + suf

    return s  # fallback: keep as-is (rare)


def sha16_of_ids(ids: List[str]) -> str:
    s = ",".join(sorted(map(str, ids)))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def summarize_split(df_part: pd.DataFrame, time_col: str, event_col: str, stage_clean_col: str) -> Dict:
    return {
        "n": int(len(df_part)),
        "events": int(df_part[event_col].astype(int).sum()),
        "event_rate": float(df_part[event_col].astype(int).mean()),
        "median_time": float(df_part[time_col].median()),
        "stage_counts": df_part[stage_clean_col].value_counts(dropna=False).to_dict(),
    }


def make_one_split(
    df: pd.DataFrame,
    seed: int,
    test_size: float,
    val_size: float,
    id_col: str,
    time_col: str,
    event_col: str,
    stage_col: str,
) -> Dict:
    df = df.copy()
    df[id_col] = df[id_col].astype(str).str.strip()

    # Canonical stage (keep UNK for true missing)
    df["stage_clean"] = df[stage_col].apply(canonicalize_stage)

    # For stratification only: avoid singleton class "UNK" breaking StratifiedShuffleSplit
    mode_stage = df.loc[df["stage_clean"] != "UNK", "stage_clean"].mode().iloc[0]
    df["stage_strat"] = df["stage_clean"].replace({"UNK": mode_stage})

    df["event_int"] = df[event_col].astype(int)

    # Stratify only on event × stage (no histology)
    strata = df["event_int"].astype(str) + "_" + df["stage_strat"].astype(str)
    df["strata"] = strata

    idx_all = np.arange(len(df))

    # 1) TrainVal vs Test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(idx_all, df["strata"]))
    df_trainval = df.iloc[trainval_idx].copy()
    df_test = df.iloc[test_idx].copy()

    # 2) Train vs Val inside TrainVal
    val_frac_within_trainval = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac_within_trainval, random_state=seed + 1)
    idx_tv = np.arange(len(df_trainval))
    train_idx2, val_idx2 = next(sss2.split(idx_tv, df_trainval["strata"]))
    df_train = df_trainval.iloc[train_idx2].copy()
    df_val = df_trainval.iloc[val_idx2].copy()

    out = {
        "seed": int(seed),
        "test_size": float(test_size),
        "val_size": float(val_size),
        "id_col": id_col,
        "time_col": time_col,
        "event_col": event_col,
        "stage_col": stage_col,
        "mode_stage_for_stratify": mode_stage,
        "splits": {
            "train": df_train[id_col].tolist(),
            "val": df_val[id_col].tolist(),
            "test": df_test[id_col].tolist(),
        },
        "fingerprint": {
            "train": sha16_of_ids(df_train[id_col].tolist()),
            "val": sha16_of_ids(df_val[id_col].tolist()),
            "test": sha16_of_ids(df_test[id_col].tolist()),
        },
        "summary": {
            "train": summarize_split(df_train, time_col, event_col, "stage_clean"),
            "val": summarize_split(df_val, time_col, event_col, "stage_clean"),
            "test": summarize_split(df_test, time_col, event_col, "stage_clean"),
        },
    }
    return out


def save_outputs(split_obj: Dict, df: pd.DataFrame, out_dir: str, id_col: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    seed = split_obj["seed"]
    json_path = os.path.join(out_dir, f"fixed_split_seed{seed}.json")
    csv_path = os.path.join(out_dir, f"fixed_split_seed{seed}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(split_obj, f, ensure_ascii=False, indent=2)

    # Build id->split CSV
    rows = []
    for split_name, ids in split_obj["splits"].items():
        for pid in ids:
            rows.append({id_col: pid, "split": split_name})
    df_map = pd.DataFrame(rows).drop_duplicates(subset=[id_col])
    df_map.to_csv(csv_path, index=False, encoding="utf-8")

    return json_path, csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra_seeds", type=int, nargs="*", default=[],
                    help="Optional additional seeds for robustness check (NOT for picking best).")

    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--val_size", type=float, default=0.15)

    ap.add_argument("--id_col", type=str, default="PatientID")
    ap.add_argument("--time_col", type=str, default="Survival.time")
    ap.add_argument("--event_col", type=str, default="deadstatus.event")
    ap.add_argument("--stage_col", type=str, default="Overall.Stage")

    args = ap.parse_args()

    df = pd.read_csv(args.clinical_csv)

    seeds = [args.seed] + list(args.extra_seeds)

    manifest = {"clinical_csv": args.clinical_csv, "out_dir": args.out_dir, "splits": []}

    for s in seeds:
        split_obj = make_one_split(
            df=df,
            seed=s,
            test_size=args.test_size,
            val_size=args.val_size,
            id_col=args.id_col,
            time_col=args.time_col,
            event_col=args.event_col,
            stage_col=args.stage_col,
        )
        json_path, csv_path = save_outputs(split_obj, df, args.out_dir, args.id_col)

        print("\n" + "=" * 80)
        print(f"[Split seed={s}]  Train/Val/Test = "
              f"{split_obj['summary']['train']['n']}/"
              f"{split_obj['summary']['val']['n']}/"
              f"{split_obj['summary']['test']['n']}")
        print(f"Fingerprints: {split_obj['fingerprint']}")
        for part in ["train", "val", "test"]:
            summ = split_obj["summary"][part]
            print(f"  - {part:5s}: n={summ['n']}, events={summ['events']} "
                  f"(rate={summ['event_rate']:.3f}), median_time={summ['median_time']:.1f}")
            print(f"           stage_counts={summ['stage_counts']}")

        manifest["splits"].append({
            "seed": s,
            "json": json_path,
            "csv": csv_path,
            "fingerprint": split_obj["fingerprint"],
            "summary": split_obj["summary"],
        })

    manifest_path = os.path.join(args.out_dir, "splits_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("\nSaved manifest:", manifest_path)


if __name__ == "__main__":
    main()
