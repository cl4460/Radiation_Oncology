#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_csv_as_str(csv_path: Path) -> pd.DataFrame:
    """
    Read everything as string first so we can detect dirty values reliably.
    Then normalize blank/whitespace-only strings to NaN.
    """
    df = pd.read_csv(csv_path, dtype=str)
    # Turn "" or "   " into NaN (robust against whitespace-only cells)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df


def coerce_numeric(df_str: pd.DataFrame, cols: list):
    """
    Force numeric conversion. Anything non-numeric becomes NaN (so imputer can handle it).
    Return:
      - df_num: converted dataframe
      - bad_counts: how many non-empty values became NaN due to conversion
    """
    df_num = df_str.copy()
    bad_counts = {}
    for c in cols:
        if c not in df_num.columns:
            continue
        before = df_str[c]
        after = pd.to_numeric(before, errors="coerce")
        newly_bad = before.notna() & after.isna()
        bad_counts[c] = int(newly_bad.sum())
        df_num[c] = after
    return df_num, bad_counts


def expect_in_set(series: pd.Series, allowed: set, colname: str, errors: list, warnings: list, allow_nan=True):
    s = series.copy()
    if allow_nan:
        s = s.dropna()
    uniq = set(s.unique().tolist())
    bad = uniq - allowed
    if bad:
        errors.append(f"[ERROR] {colname} has values outside {sorted(list(allowed))}: {sorted(list(bad))}")


def check_range(series: pd.Series, lo: float, hi: float, colname: str, errors: list, warnings: list, allow_nan=True):
    s = series.copy()
    if allow_nan:
        s = s.dropna()
    if len(s) == 0:
        warnings.append(f"[WARN] {colname}: all missing.")
        return
    mn, mx = float(np.min(s)), float(np.max(s))
    if mn < lo or mx > hi:
        errors.append(f"[ERROR] {colname} out of range [{lo}, {hi}]. min={mn}, max={mx}")


def check_integer_like(series: pd.Series, colname: str, errors: list, warnings: list, allow_nan=True, tol=1e-6):
    s = series.copy()
    if allow_nan:
        s = s.dropna()
    if len(s) == 0:
        warnings.append(f"[WARN] {colname}: all missing.")
        return
    frac = np.abs(s - np.round(s))
    bad = int((frac > tol).sum())
    if bad > 0:
        warnings.append(f"[WARN] {colname}: {bad} values are not integer-like (tol={tol}).")


def load_splits(splits_json: Path) -> dict:
    with open(splits_json, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to clinical_features_new.csv (or equivalent).")
    ap.add_argument("--id_col", type=str, default="PatientID")

    ap.add_argument("--time_col", type=str, default="Survival.time")
    ap.add_argument("--event_col", type=str, default="deadstatus.event")

    ap.add_argument("--age_col", type=str, default="age")
    ap.add_argument("--gender_col", type=str, default="gender")
    ap.add_argument("--t_col", type=str, default="clinical_T_Stage")
    ap.add_argument("--n_col", type=str, default="Clinical_N_Stage")
    ap.add_argument("--m_col", type=str, default="Clinical_M_binary")

    ap.add_argument("--stage_prefix", type=str, default="Overall_Stage_")

    ap.add_argument("--splits_json", type=str, default=None,
                    help="Optional: path to splits.json to check fold-wise distributions.")
    ap.add_argument("--write_clean_csv", type=str, default=None,
                    help="Optional: write a cleaned numeric CSV to this path.")

    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[FATAL] CSV not found: {csv_path}")
        sys.exit(2)

    errors, warnings, notes = [], [], []

    # 1) Load as string + normalize blanks
    df_str = load_csv_as_str(csv_path)

    # Basic column existence
    required_cols = [args.id_col, args.age_col, args.gender_col, args.t_col, args.n_col, args.m_col]
    missing_req = [c for c in required_cols if c not in df_str.columns]
    if missing_req:
        errors.append(f"[ERROR] Missing required columns: {missing_req}")

    # Stage cols (allow: can be 0 columns if user doesn't use stage)
    stage_cols = [c for c in df_str.columns if c.startswith(args.stage_prefix)]
    if len(stage_cols) == 0:
        warnings.append(f"[WARN] No stage columns found with prefix '{args.stage_prefix}'.")

    # Survival cols (optional but strongly recommended)
    has_time = args.time_col in df_str.columns
    has_event = args.event_col in df_str.columns
    if not has_time or not has_event:
        warnings.append(f"[WARN] Survival columns not both present: "
                        f"time_col={has_time}, event_col={has_event}. "
                        f"(Your R1 script can still merge from labels_csv.)")

    # 2) ID checks
    if args.id_col in df_str.columns:
        id_series = df_str[args.id_col]
        n_total = len(id_series)
        n_missing_id = int(id_series.isna().sum())
        if n_missing_id > 0:
            errors.append(f"[ERROR] {args.id_col} has {n_missing_id} missing values.")
        # normalize to str
        ids = id_series.astype(str)
        n_unique = int(ids.nunique(dropna=False))
        if n_unique != n_total:
            dup = ids[ids.duplicated()].head(10).tolist()
            errors.append(f"[ERROR] {args.id_col} not unique: {n_total-n_unique} duplicates. Examples: {dup}")
        notes.append(f"[OK] IDs: n={n_total}, unique={n_unique}, missing={n_missing_id}")

    # 3) Force numeric conversion on numeric-like columns
    numeric_cols = []
    for c in [args.age_col, args.gender_col, args.t_col, args.n_col, args.m_col]:
        if c in df_str.columns:
            numeric_cols.append(c)
    for c in stage_cols:
        numeric_cols.append(c)
    if has_time:
        numeric_cols.append(args.time_col)
    if has_event:
        numeric_cols.append(args.event_col)

    df_num, bad_counts = coerce_numeric(df_str, numeric_cols)

    # If any column has dirty values (non-empty -> NaN after numeric coercion), flag it
    for c, bad in bad_counts.items():
        if bad > 0:
            errors.append(f"[ERROR] Column '{c}' has {bad} dirty non-numeric values (non-empty but not parseable).")

    # 4) Check missingness summary
    miss = df_num[numeric_cols].isna().sum().sort_values(ascending=False)
    notes.append("[INFO] Missing counts (top): " + ", ".join([f"{k}={int(v)}" for k, v in miss.head(8).items()]))

    # 5) Value legality checks
    if args.gender_col in df_num.columns:
        expect_in_set(df_num[args.gender_col], {0.0, 1.0}, args.gender_col, errors, warnings, allow_nan=True)

    if args.m_col in df_num.columns:
        expect_in_set(df_num[args.m_col], {0.0, 1.0}, args.m_col, errors, warnings, allow_nan=True)

    # Stage columns should be 0/1
    for c in stage_cols:
        expect_in_set(df_num[c], {0.0, 1.0}, c, errors, warnings, allow_nan=True)

    # age range (be generous)
    if args.age_col in df_num.columns:
        check_range(df_num[args.age_col], 0.0, 120.0, args.age_col, errors, warnings, allow_nan=True)

    # T/N stage ranges + integer-like check
    if args.t_col in df_num.columns:
        check_integer_like(df_num[args.t_col], args.t_col, errors, warnings, allow_nan=True)
        # typical allowed set {1..5} for your dataset (warn if outside)
        s = df_num[args.t_col].dropna().unique()
        if len(s) > 0:
            bad = [x for x in sorted(map(float, s)) if x < 1 or x > 5]
            if bad:
                warnings.append(f"[WARN] {args.t_col} has values outside [1,5]: {bad}")

    if args.n_col in df_num.columns:
        check_integer_like(df_num[args.n_col], args.n_col, errors, warnings, allow_nan=True)
        s = df_num[args.n_col].dropna().unique()
        if len(s) > 0:
            bad = [x for x in sorted(map(float, s)) if x < 0 or x > 4]
            if bad:
                warnings.append(f"[WARN] {args.n_col} has values outside [0,4]: {bad}")

    # Survival checks
    if has_time:
        t = df_num[args.time_col]
        if (t.dropna() <= 0).any():
            errors.append(f"[ERROR] {args.time_col} has non-positive values.")
    if has_event:
        expect_in_set(df_num[args.event_col], {0.0, 1.0}, args.event_col, errors, warnings, allow_nan=True)

    # 6) Stage mutual exclusivity check (allow all-zeros to represent the reference category, e.g., Stage I)
    if len(stage_cols) > 0:
        stage_sum = df_num[stage_cols].fillna(0).sum(axis=1)
        if (stage_sum > 1.0 + 1e-6).any():
            bad_rows = df_num.loc[stage_sum > 1.0 + 1e-6, [args.id_col] + stage_cols].head(10)
            errors.append("[ERROR] Stage one-hot not mutually exclusive (sum > 1). Examples:\n" + bad_rows.to_string(index=False))
        notes.append(f"[OK] Stage one-hot sums distribution: {stage_sum.value_counts().head(10).to_dict()}")

    # 7) Constant/near-constant columns (can hurt interpretability; model won't crash but you may want to know)
    const_cols = []
    for c in numeric_cols:
        s = df_num[c].dropna()
        if len(s) == 0:
            continue
        if float(np.nanstd(s)) == 0.0:
            const_cols.append(c)
    if const_cols:
        warnings.append(f"[WARN] Constant columns (std=0 after dropping NaN): {const_cols}")

    # 8) Optional: fold-wise checks using splits.json
    if args.splits_json is not None:
        splits_path = Path(args.splits_json)
        if not splits_path.exists():
            errors.append(f"[ERROR] splits_json not found: {splits_path}")
        else:
            splits = load_splits(splits_path)
            all_ids = set(df_num[args.id_col].astype(str).tolist())
            for fk in sorted(splits.keys()):
                tr = [str(x) for x in splits[fk]["train"]]
                va = [str(x) for x in splits[fk]["val"]]
                miss_tr = [x for x in tr if x not in all_ids]
                miss_va = [x for x in va if x not in all_ids]
                if miss_tr or miss_va:
                    errors.append(f"[ERROR] Fold {fk}: IDs missing in CSV. train_missing={len(miss_tr)}, val_missing={len(miss_va)}")
                    continue

                tr_df = df_num[df_num[args.id_col].astype(str).isin(tr)]
                va_df = df_num[df_num[args.id_col].astype(str).isin(va)]

                # M_binary distribution (critical for stability)
                if args.m_col in df_num.columns:
                    tr_pos = int((tr_df[args.m_col].fillna(0) > 0).sum())
                    va_pos = int((va_df[args.m_col].fillna(0) > 0).sum())
                    if tr_pos == 0 and va_pos > 0:
                        warnings.append(f"[WARN] Fold {fk}: {args.m_col} has 0 positives in TRAIN but {va_pos} positives in VAL "
                                        f"(model cannot learn this effect in that fold).")

            notes.append("[OK] Fold-wise checks done (see WARN/ERROR if any).")

    # 9) Optionally write a cleaned numeric CSV
    if args.write_clean_csv is not None:
        out_path = Path(args.write_clean_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # For cleanliness: keep original column order, but numeric columns are numeric now
        df_clean = df_str.copy()
        for c in numeric_cols:
            if c in df_clean.columns:
                df_clean[c] = df_num[c]
        df_clean.to_csv(out_path, index=False)
        notes.append(f"[OK] Wrote cleaned CSV: {out_path}")

    # 10) Print report
    print("\n" + "=" * 88)
    print(f"[REPORT] CSV: {csv_path}")
    print("=" * 88)

    for n in notes:
        print(n)
    if warnings:
        print("\n--- WARNINGS ---")
        for w in warnings:
            print(w)
    if errors:
        print("\n--- ERRORS ---")
        for e in errors:
            print(e)

    print("\n" + "=" * 88)
    if errors:
        print(f"[RESULT] FAIL: {len(errors)} error(s), {len(warnings)} warning(s).")
        sys.exit(1)
    else:
        print(f"[RESULT] PASS: 0 error(s), {len(warnings)} warning(s).")
        sys.exit(0)


if __name__ == "__main__":
    main()

#run command:
#python clinical_sanity_check.py \
#  --csv /home/lichengze/Research/DeepFeature/myresearch/main/R0/r0_feature_csvs_update/clinical_features.csv \
#  --splits_json /home/lichengze/Research/DeepFeature/myresearch/main/R0/splits.json