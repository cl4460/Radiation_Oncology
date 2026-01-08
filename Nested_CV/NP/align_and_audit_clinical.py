#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
NA_STRINGS = {
    "", "na", "n/a", "nan", "none", "null", "not collected", "unknown", "unk"
}

def _norm_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def _to_nan_if_na_string(x):
    s = _norm_str(x).lower()
    if s in NA_STRINGS:
        return np.nan
    return _norm_str(x)

def parse_mdy_date(s: str) -> Optional[datetime]:
    s = _norm_str(s)
    if not s:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None

def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def map_sex_to_mf(x) -> str:
    s = _norm_str(x).lower()
    if s in ("m", "male"):
        return "M"
    if s in ("f", "female"):
        return "F"
    return ""

def map_histology(x) -> str:
    s = _norm_str(x).lower()
    if not s:
        return ""
    # canonical buckets
    if "adeno" in s:
        return "adenocarcinoma"
    if "squamous" in s:
        return "squamous"
    if "large" in s:
        return "large_cell"
    if "nos" in s:
        return "nos"
    # keep original if not matched
    return s

def parse_tnm_numeric_from_string(x, prefix: str) -> float:
    """
    Parse TNM strings like 'T2', 'T2A', 'N0', 'M1', 'TX', 'Not Collected'
    Return numeric ordinal (Tis->0, T1->1, ...; NX/MX -> NaN)
    """
    x = _to_nan_if_na_string(x)
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()

    # handle special
    if s in (f"{prefix}X", "X"):
        return np.nan
    if prefix == "T" and "TIS" in s:
        return 0.0

    m = re.search(rf"{prefix}\s*([0-4])", s)
    if m:
        return float(m.group(1))
    return np.nan


# -----------------------------
# Load + harmonize Lung1
# -----------------------------
def load_lung1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # rename to standard
    out = pd.DataFrame()
    out["patient_id"] = df["PatientID"].astype(str).str.strip()
    out["dataset"] = "LUNG1"

    out["age"] = pd.to_numeric(df["age"], errors="coerce")
    out["sex"] = df["gender"].map(map_sex_to_mf)

    out["histology"] = df["Histology"].map(map_histology)

    # keep raw TNM as numeric (already encoded)
    out["t_stage_raw"] = df["clinical.T.Stage"]
    out["n_stage_raw"] = df["Clinical.N.Stage"]
    out["m_stage_raw"] = df["Clinical.M.Stage"]

    out["t_stage_num"] = pd.to_numeric(df["clinical.T.Stage"], errors="coerce")
    out["n_stage_num"] = pd.to_numeric(df["Clinical.N.Stage"], errors="coerce")
    out["m_stage_num"] = pd.to_numeric(df["Clinical.M.Stage"], errors="coerce")

    # overall stage (only in Lung1)
    out["overall_stage_raw"] = df["Overall.Stage"].astype(str).str.strip()
    # optional ordinal mapping (I=1, II=2, IIIA=3, IIIB=4)
    st = out["overall_stage_raw"].str.upper()
    stage_map = {"I": 1, "II": 2, "IIIA": 3, "IIIB": 4}
    out["overall_stage_ord"] = st.map(stage_map).astype("float")

    out["time_days"] = pd.to_numeric(df["Survival.time"], errors="coerce")
    out["event"] = pd.to_numeric(df["deadstatus.event"], errors="coerce").fillna(0).astype(int)

    return out


# -----------------------------
# Load + harmonize Radiogenomics
# -----------------------------
def compute_rg_time_event(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Radiogenomics: OS baseline effectively at surgery date.
    Dead: Time to Death (days) is provided.
    Alive: use (LastKnownAlive - (CTDate + DaysBetweenCTAndSurgery)).
    """
    status = df["Survival Status"].astype(str).str.strip().str.lower()
    event = (status == "dead").astype(int).values

    # parse dates
    ct_date = df["CT Date"].map(parse_mdy_date)
    days_between = df["Days between CT and surgery"].map(safe_float)
    surg_date = []
    for ct, d in zip(ct_date, days_between):
        if ct is None or np.isnan(d):
            surg_date.append(None)
        else:
            surg_date.append(ct + pd.to_timedelta(float(d), unit="D"))

    last_alive = df["Date of Last Known Alive"].map(parse_mdy_date)

    ttd = df["Time to Death (days)"].map(safe_float).values

    time = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        if event[i] == 1:
            time[i] = ttd[i]
        else:
            if surg_date[i] is None or last_alive[i] is None:
                time[i] = np.nan
            else:
                time[i] = float((last_alive[i] - surg_date[i]).days)

    return time, event

def load_radiogenomics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    out = pd.DataFrame()
    out["patient_id"] = df["Case ID"].astype(str).str.strip()
    out["dataset"] = "RADIOGENOMICS"

    out["age"] = pd.to_numeric(df["Age at Histological Diagnosis"], errors="coerce")
    out["sex"] = df["Gender"].map(map_sex_to_mf)
    out["histology"] = df["Histology "].map(map_histology)

    # TNM from strings (pathological)
    out["t_stage_raw"] = df["Pathological T stage"].map(_to_nan_if_na_string)
    out["n_stage_raw"] = df["Pathological N stage"].map(_to_nan_if_na_string)
    out["m_stage_raw"] = df["Pathological M stage"].map(_to_nan_if_na_string)

    out["t_stage_num"] = df["Pathological T stage"].map(lambda x: parse_tnm_numeric_from_string(x, "T"))
    out["n_stage_num"] = df["Pathological N stage"].map(lambda x: parse_tnm_numeric_from_string(x, "N"))
    out["m_stage_num"] = df["Pathological M stage"].map(lambda x: parse_tnm_numeric_from_string(x, "M"))

    # overall stage not available in this table
    out["overall_stage_raw"] = ""
    out["overall_stage_ord"] = np.nan

    time, event = compute_rg_time_event(df)
    out["time_days"] = time
    out["event"] = event

    # keep some provenance columns for debugging if you want
    out["_rg_ct_date"] = df["CT Date"].astype(str)
    out["_rg_days_ct_to_surg"] = df["Days between CT and surgery"]
    out["_rg_last_alive"] = df["Date of Last Known Alive"].astype(str)
    out["_rg_survival_status"] = df["Survival Status"].astype(str)
    out["_rg_time_to_death_days"] = df["Time to Death (days)"]

    return out


# -----------------------------
# Audit / compare
# -----------------------------
def audit_df(df: pd.DataFrame, name: str) -> dict:
    rep = {}
    rep["name"] = name
    rep["n_rows"] = int(df.shape[0])
    rep["n_unique_patients"] = int(df["patient_id"].nunique())
    rep["n_events"] = int(df["event"].sum())
    rep["event_rate"] = float(df["event"].mean())

    t = df["time_days"].copy()
    rep["time_days_missing"] = int(t.isna().sum())
    rep["time_days_min"] = float(np.nanmin(t.values)) if t.notna().any() else np.nan
    rep["time_days_median"] = float(np.nanmedian(t.values)) if t.notna().any() else np.nan
    rep["time_days_max"] = float(np.nanmax(t.values)) if t.notna().any() else np.nan

    rep["missingness"] = {c: float(df[c].isna().mean()) for c in ["age","sex","histology","t_stage_num","n_stage_num","m_stage_num","overall_stage_ord","time_days","event"]}
    rep["histology_counts_top"] = df["histology"].value_counts(dropna=False).head(10).to_dict()
    rep["sex_counts"] = df["sex"].value_counts(dropna=False).to_dict()
    rep["t_stage_counts_top"] = df["t_stage_raw"].value_counts(dropna=False).head(10).to_dict()
    rep["n_stage_counts_top"] = df["n_stage_raw"].value_counts(dropna=False).head(10).to_dict()
    rep["m_stage_counts_top"] = df["m_stage_raw"].value_counts(dropna=False).head(10).to_dict()
    rep["overall_stage_counts"] = df["overall_stage_raw"].value_counts(dropna=False).head(10).to_dict()
    return rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lung1_csv", required=True, type=Path)
    ap.add_argument("--rg_csv", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    lung1 = load_lung1(args.lung1_csv)
    rg = load_radiogenomics(args.rg_csv)

    # basic sanity checks
    for df, name in [(lung1, "LUNG1"), (rg, "RADIOGENOMICS")]:
        dup = df["patient_id"].duplicated().sum()
        if dup > 0:
            print(f"[WARN] {name}: duplicated patient_id rows = {dup}")
        bad_time = (df["time_days"].notna() & (df["time_days"] <= 0)).sum()
        if bad_time > 0:
            print(f"[WARN] {name}: non-positive time_days rows = {bad_time}")

    # write harmonized tables
    lung1_out = args.outdir / "lung1_harmonized.csv"
    rg_out = args.outdir / "radiogenomics_harmonized.csv"
    lung1.to_csv(lung1_out, index=False)
    rg.to_csv(rg_out, index=False)

    # report
    report = {
        "lung1": audit_df(lung1, "LUNG1"),
        "radiogenomics": audit_df(rg, "RADIOGENOMICS"),
        "common_columns_in_harmonized": list(lung1.columns.intersection(rg.columns)),
    }
    (args.outdir / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("Saved:")
    print(" -", lung1_out)
    print(" -", rg_out)
    print(" -", args.outdir / "audit_report.json")

    # quick on-screen highlights
    print("\n=== QUICK HIGHLIGHTS ===")
    print("LUNG1 Overall.Stage counts:", lung1["overall_stage_raw"].value_counts(dropna=False).to_dict())
    print("RADIOGENOMICS Survival Status counts:", rg["_rg_survival_status"].value_counts(dropna=False).to_dict())
    print("RADIOGENOMICS Path T missing rate:", float(rg["t_stage_raw"].isna().mean()))

if __name__ == "__main__":
    main()
