#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prepare_radiogenomics_timezero.py

Generate a *time-zero aligned* survival table for TCIA **NSCLC-Radiogenomics**.

Background
----------
Radiogenomics provides:
- CT acquisition date (CT Date)
- Surgery delay (Days between CT and surgery)
- Time to death (days) measured from *surgery*
- Date of last known alive / date of death

To support your MICCAI-ready evaluation, we create two aligned endpoints:

(A) Surgery time-zero (surgery start):
    time_surg0_days:
        - if Dead: use "Time to Death (days)" when available
        - else: last_alive_date - surgery_date

(B) CT time-zero (imaging time-zero, often most meaningful for imaging models):
    time_ct0_days:
        - if Dead: prefer (death_date - ct_date)
                  fallback: days_ct_to_surgery + time_to_death
        - else: (last_alive_date - ct_date)

We also output CT→Surgery delay (days) so you can do sensitivity analyses
(e.g., excluding >180 days).

Usage
-----
python prepare_radiogenomics_timezero.py \
  --input_csv /path/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv \
  --output_csv /path/radiogenomics_timezero_aligned.csv \
  --max_ct_to_surgery_days 180 \
  --output_filtered_csv /path/radiogenomics_timezero_aligned_ct2surg_le180.csv

Notes
-----
- This script is intentionally *conservative*: if required dates are missing,
  the corresponding time will be NaN instead of silently fabricating values.
- It is normal that external calibration fails between Lung1 (RT/CRT) and
  Radiogenomics (surgery) due to baseline risk differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _to_date(series: pd.Series) -> pd.Series:
    """Parse date strings to pandas datetime (NaT on failure)."""
    return pd.to_datetime(series, errors="coerce")


def _to_num(series: pd.Series) -> pd.Series:
    """Parse numeric to float (NaN on failure)."""
    return pd.to_numeric(series, errors="coerce")


def _safe_days(delta: pd.Series) -> pd.Series:
    """Convert timedeltas to days (float), NaN if missing."""
    out = delta.dt.total_seconds() / 86400.0
    out = out.astype(float)
    return out


def build_aligned_table(df: pd.DataFrame, max_ct_to_surgery_days: int = 180) -> pd.DataFrame:
    required = [
        "Case ID",
        "Survival Status",
        "CT Date",
        "Days between CT and surgery",
        "Date of Last Known Alive",
        "Date of Death",
        "Time to Death (days)",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    out = pd.DataFrame()
    out["Case_ID"] = df["Case ID"].astype(str).str.strip()

    ct_date = _to_date(df["CT Date"])
    days_ct2surg = _to_num(df["Days between CT and surgery"]).astype(float)
    surg_date = ct_date + pd.to_timedelta(days_ct2surg, unit="D")

    status = df["Survival Status"].astype(str).str.strip()
    status_lower = status.str.lower()

    death_date = _to_date(df["Date of Death"])
    last_alive_date = _to_date(df["Date of Last Known Alive"])
    ttd = _to_num(df["Time to Death (days)"]).astype(float)

    # Event definition: treat as event if status says dead OR death date OR time-to-death is present
    event = (status_lower == "dead") | death_date.notna() | ttd.notna()
    event = event.astype(int)

    # Time from surgery
    time_surg0 = pd.Series(np.nan, index=df.index, dtype=float)
    # dead: prefer time-to-death from surgery
    dead_mask = event == 1
    use_ttd = dead_mask & ttd.notna()
    time_surg0.loc[use_ttd] = ttd.loc[use_ttd]

    # dead but missing ttd: try death_date - surgery_date
    use_death_minus_surg = dead_mask & time_surg0.isna() & death_date.notna() & surg_date.notna()
    time_surg0.loc[use_death_minus_surg] = _safe_days(death_date.loc[use_death_minus_surg] - surg_date.loc[use_death_minus_surg])

    # alive: last_alive - surgery
    alive_mask = event == 0
    use_alive = alive_mask & last_alive_date.notna() & surg_date.notna()
    time_surg0.loc[use_alive] = _safe_days(last_alive_date.loc[use_alive] - surg_date.loc[use_alive])

    # Time from CT
    time_ct0 = pd.Series(np.nan, index=df.index, dtype=float)
    time_ct0_source = pd.Series("", index=df.index, dtype=str)

    # dead: prefer death_date - ct_date
    use_death_minus_ct = dead_mask & death_date.notna() & ct_date.notna()
    time_ct0.loc[use_death_minus_ct] = _safe_days(death_date.loc[use_death_minus_ct] - ct_date.loc[use_death_minus_ct])
    time_ct0_source.loc[use_death_minus_ct] = "death_date_minus_ct_date"

    # dead fallback: days_ct2surg + ttd
    use_sum = dead_mask & time_ct0.isna() & days_ct2surg.notna() & ttd.notna()
    time_ct0.loc[use_sum] = days_ct2surg.loc[use_sum] + ttd.loc[use_sum]
    time_ct0_source.loc[use_sum] = "days_ct2surg_plus_ttd"

    # alive: last_alive - ct_date
    use_alive_ct = alive_mask & last_alive_date.notna() & ct_date.notna()
    time_ct0.loc[use_alive_ct] = _safe_days(last_alive_date.loc[use_alive_ct] - ct_date.loc[use_alive_ct])
    time_ct0_source.loc[use_alive_ct] = "last_alive_minus_ct_date"

    # Sanity: negative times -> NaN
    for col in [time_surg0, time_ct0]:
        neg = col < 0
        if neg.any():
            col.loc[neg] = np.nan

    out["CT_Date"] = ct_date.dt.strftime("%Y-%m-%d")
    out["Days_CT_to_Surgery"] = days_ct2surg
    out["Surgery_Date"] = surg_date.dt.strftime("%Y-%m-%d")
    out["Survival_Status"] = status
    out["Death_Date"] = death_date.dt.strftime("%Y-%m-%d")
    out["LastAlive_Date"] = last_alive_date.dt.strftime("%Y-%m-%d")
    out["event"] = event
    out["time_surg0_days"] = time_surg0
    out["time_ct0_days"] = time_ct0
    out["time_ct0_source"] = time_ct0_source
    out[f"ct_to_surgery_gt_{max_ct_to_surgery_days}"] = (days_ct2surg > float(max_ct_to_surgery_days)).astype(int)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, type=str, help="Raw Radiogenomics labels CSV")
    ap.add_argument("--output_csv", required=True, type=str, help="Where to write aligned CSV")
    ap.add_argument("--max_ct_to_surgery_days", type=int, default=180,
                    help="Threshold for sensitivity analysis flag/filtered output")
    ap.add_argument("--output_filtered_csv", type=str, default="",
                    help="Optional path: write an additional CSV filtered to ct_to_surgery <= threshold")
    args = ap.parse_args()

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)

    df = pd.read_csv(in_path)
    aligned = build_aligned_table(df, max_ct_to_surgery_days=int(args.max_ct_to_surgery_days))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(out_path, index=False)

    print(f"[OK] Wrote aligned table: {out_path} (n={len(aligned)})")
    # quick audit
    print(f"      events: {aligned['event'].sum()} / {len(aligned)}")
    print(f"      missing time_surg0_days: {aligned['time_surg0_days'].isna().sum()}")
    print(f"      missing time_ct0_days:  {aligned['time_ct0_days'].isna().sum()}")

    if args.output_filtered_csv:
        filt_path = Path(args.output_filtered_csv)
        filt = aligned[aligned[f"ct_to_surgery_gt_{int(args.max_ct_to_surgery_days)}"] == 0].copy()
        filt_path.parent.mkdir(parents=True, exist_ok=True)
        filt.to_csv(filt_path, index=False)
        print(f"[OK] Wrote filtered table (ct_to_surgery <= {args.max_ct_to_surgery_days}): {filt_path} (n={len(filt)})")


if __name__ == "__main__":
    main()
