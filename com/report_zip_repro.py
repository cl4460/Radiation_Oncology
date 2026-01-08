#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_zip_repro.py

Purpose
-------
Reproduce (and *verify*) the experiment-report baseline strictly from the shipped
artifacts inside `实验结果.zip`, treating `test_preds_*.xlsx` as the ground truth.

This script focuses on **Goal A**:
    Align the C-index computed from `test_preds_*.xlsx`
because those are the only outputs that are directly verifiable.

What you get
------------
1) A manifest of (task, seed, model) -> file paths found in the zip.
2) A baseline table computed from `test_preds_*.xlsx` with bootstrap CI.
3) A "recipe dump" that exposes what the zip actually did:
   - imputation strategy + fill values for clinical columns
   - numeric scaler type + categorical encoder type
   - selected features list per model
   - best_params per model (GBSA hyperparams etc.)
4) Export test split IDs (and optionally train IDs) as JSON, to make your re-runs fair.

Important
---------
- This script does NOT try to "guess" missing details from the DOCX.
- If the DOCX numbers disagree with the C-index computed from `test_preds`,
  the DOCX is treated as unreliable; the zip predictions are the truth.

Usage
-----
# 1) Score baseline from test_preds (default truth check)
python report_zip_repro.py score --zip_path /path/to/实验结果.zip --out_dir out --bootstrap 200 --seed 42

# 2) Dump a compact recipe summary
python report_zip_repro.py dump-recipe --zip_path /path/to/实验结果.zip --out_dir out

# 3) Export split IDs
python report_zip_repro.py export-splits --zip_path /path/to/实验结果.zip --out_dir out \
    --clinical_csv /path/to/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv

Outputs
-------
out/manifest.json
out/baseline.csv
out/baseline.json
out/recipe_summary.json
out/splits_index.json
out/split_*.json
"""

from __future__ import annotations

import argparse
import io
import json
import re
import os
import zipfile
import hashlib
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import joblib
import yaml


TASK_PAT = re.compile(
    r"^(Clinic\+Radiomics|Clinic|Radiomics)/seed_(\d+)/(.*?)/test_preds_.*\.xlsx$"
)


def to_builtin(obj):
    """Recursively convert numpy / pandas scalars to plain Python types for JSON."""
    import numpy as _np
    import pandas as _pd

    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (_np.generic,)):
        return obj.item()
    if isinstance(obj, (_pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]
    return str(obj)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def read_text_from_zip(zf: zipfile.ZipFile, name: str) -> str:
    with zf.open(name) as f:
        return f.read().decode("utf-8", errors="replace")


def read_xlsx_from_zip(zf: zipfile.ZipFile, name: str, sheet_name=0) -> pd.DataFrame:
    with zf.open(name) as f:
        data = f.read()
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet_name)


def load_joblib_from_zip(zf: zipfile.ZipFile, name: str) -> Any:
    with zf.open(name) as f:
        data = f.read()
    return joblib.load(io.BytesIO(data))


def harrell_cindex(event, time, risk) -> float:
    """
    Harrell's C-index for right-censored survival.
    risk: higher -> worse survival.
    O(n^2) implementation is OK for n_test~90.
    """
    event = np.asarray(event).astype(bool)
    time = np.asarray(time).astype(float)
    risk = np.asarray(risk).astype(float)

    n = len(time)
    conc = 0.0
    ties = 0.0
    perm = 0.0

    for i in range(n):
        if not event[i]:
            continue
        ti = time[i]
        ri = risk[i]
        for j in range(n):
            if time[j] <= ti:
                continue
            perm += 1
            rj = risk[j]
            if ri > rj:
                conc += 1
            elif ri == rj:
                ties += 1

    if perm == 0:
        return float("nan")
    return float((conc + 0.5 * ties) / perm)


def bootstrap_cindex(event, time, risk, n_boot: int = 200, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(time)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ci = harrell_cindex(event[idx], time[idx], risk[idx])
        if np.isfinite(ci):
            vals.append(ci)

    if len(vals) == 0:
        return {"boot_mean": np.nan, "boot_std": np.nan, "ci95": [np.nan, np.nan], "n_eff": 0}

    vals = np.asarray(vals, dtype=float)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"boot_mean": mean, "boot_std": std, "ci95": [float(lo), float(hi)], "n_eff": int(len(vals))}


def scan_zip(zip_path: str) -> List[Dict[str, Any]]:
    zf = zipfile.ZipFile(zip_path)
    entries = []
    for name in zf.namelist():
        m = TASK_PAT.match(name)
        if not m:
            continue
        task, seed, model_dir = m.group(1), int(m.group(2)), m.group(3)
        entries.append({"task": task, "seed": seed, "model_dir": model_dir, "test_preds": name})
    entries.sort(key=lambda d: (d["task"], d["model_dir"], d["seed"]))
    return entries


def related_paths(zf: zipfile.ZipFile, task: str, seed: int, model_dir: str) -> Dict[str, Optional[str]]:
    base = f"{task}/seed_{seed}/{model_dir}/"
    seed_base = f"{task}/seed_{seed}/"

    def first_match(prefix: str, suffix: str) -> Optional[str]:
        cands = [p for p in zf.namelist() if p.startswith(base + prefix) and p.endswith(suffix)]
        return cands[0] if cands else None

    best_params = first_match("best_params", ".yaml")
    selected_features = first_match("selected_features", ".xlsx")
    model_scores = first_match("test_scores", ".xlsx")

    train_fill_values = seed_base + "train_fill_values.joblib" if (seed_base + "train_fill_values.joblib") in zf.namelist() else None
    normalizer = seed_base + "normalizer.joblib" if (seed_base + "normalizer.joblib") in zf.namelist() else None
    exp_specs = seed_base + "exp_specs.yaml" if (seed_base + "exp_specs.yaml") in zf.namelist() else None
    create_split_specs = seed_base + "create_datasplits_specs.yaml" if (seed_base + "create_datasplits_specs.yaml") in zf.namelist() else None

    return {
        "best_params": best_params,
        "selected_features": selected_features,
        "test_scores": model_scores,
        "train_fill_values": train_fill_values,
        "normalizer": normalizer,
        "exp_specs": exp_specs,
        "create_datasplits_specs": create_split_specs,
    }


def read_yaml_optional(zf: zipfile.ZipFile, name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    txt = read_text_from_zip(zf, name)
    return yaml.safe_load(txt)


def read_selected_features(zf: zipfile.ZipFile, name: Optional[str]) -> Optional[List[str]]:
    if not name:
        return None
    df = read_xlsx_from_zip(zf, name)
    feats: List[str] = []
    for col in df.columns:
        feats.extend([str(x) for x in df[col].dropna().tolist()])
    out: List[str] = []
    seen = set()
    for f in feats:
        if f not in seen:
            out.append(f)
            seen.add(f)
    return out


def _get_list(d: Dict[str, Any], *keys: str) -> List[Any]:
    for k in keys:
        if k in d:
            return list(d[k])
    return []


def summarize_preprocess(zf: zipfile.ZipFile, fill_path: Optional[str], norm_path: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    if fill_path:
        fill = load_joblib_from_zip(zf, fill_path)
        out["impute_strategy"] = fill.get("strategy")

        fv = fill.get("fill_values", {})
        clin_keys = [k for k in ["age", "gender", "Histology", "clinical.T.Stage", "Clinical.N.Stage", "Clinical.M.Stage", "Overall.Stage"] if k in fv]
        out["clinical_fill_values"] = {k: fv[k] for k in clin_keys}
        out["n_fill_values"] = int(len(fv))

        h = hashlib.sha1()
        for k in sorted(fv.keys()):
            h.update(str(k).encode())
            h.update(b"=")
            h.update(str(fv[k]).encode())
            h.update(b";")
        out["fill_values_sha1"] = h.hexdigest()

    if norm_path:
        norm = load_joblib_from_zip(zf, norm_path)
        normals = norm.get("normalizers", {})
        out["numeric_scaler"] = type(normals.get("numeric")).__name__ if "numeric" in normals else None
        out["categorical_encoder"] = type(normals.get("categorical")).__name__ if "categorical" in normals else None
        out["numeric_columns"] = _get_list(norm, "numeric_columns", "numeric_cols", "numeric_colnames")
        out["category_columns"] = _get_list(norm, "category_columns", "categorical_columns", "category_cols")

    return out


def cmd_score(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)

    entries = scan_zip(args.zip_path)

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    zf = zipfile.ZipFile(args.zip_path)
    rows = []
    for ent in entries:
        df = read_xlsx_from_zip(zf, ent["test_preds"])

        if "pred_risk" not in df.columns:
            for cand in ["risk", "pred", "prediction"]:
                if cand in df.columns:
                    df = df.rename(columns={cand: "pred_risk"})
                    break
        if "deadstatus.event" not in df.columns:
            for cand in ["event", "status", "E"]:
                if cand in df.columns:
                    df = df.rename(columns={cand: "deadstatus.event"})
                    break
        if "time" not in df.columns:
            for cand in ["Survival.time", "T"]:
                if cand in df.columns:
                    df = df.rename(columns={cand: "time"})
                    break

        event = df["deadstatus.event"].values.astype(bool)
        time = df["time"].values.astype(float)
        risk = df["pred_risk"].values.astype(float)

        ci = harrell_cindex(event, time, risk)
        boot = bootstrap_cindex(event, time, risk, n_boot=args.bootstrap, seed=args.seed)

        rows.append(
            {
                "task": ent["task"],
                "model": ent["model_dir"],
                "seed": ent["seed"],
                "n_test": int(len(df)),
                "test_cindex": float(ci),
                **boot,
                "test_preds_path": ent["test_preds"],
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["task", "model", "seed"]).reset_index(drop=True)

    csv_path = os.path.join(args.out_dir, "baseline.csv")
    out_df.to_csv(csv_path, index=False)

    json_path = os.path.join(args.out_dir, "baseline.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(out_df.to_dict(orient="records")), f, indent=2, ensure_ascii=False)

    print("=== Baseline (computed from test_preds_*.xlsx; truth) ===")
    print(out_df[["task", "model", "seed", "n_test", "test_cindex", "boot_mean", "boot_std"]].to_string(index=False))
    print(f"\nSaved: {manifest_path}\nSaved: {csv_path}\nSaved: {json_path}")


def cmd_dump_recipe(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    entries = scan_zip(args.zip_path)
    zf = zipfile.ZipFile(args.zip_path)

    out: Dict[str, Any] = {"zip_path": args.zip_path, "models": []}

    for ent in entries:
        task, seed, model_dir = ent["task"], int(ent["seed"]), ent["model_dir"]
        rel = related_paths(zf, task, seed, model_dir)

        best_params = read_yaml_optional(zf, rel["best_params"])
        exp_specs = read_yaml_optional(zf, rel["exp_specs"])
        split_specs = read_yaml_optional(zf, rel["create_datasplits_specs"])
        feats = read_selected_features(zf, rel["selected_features"])
        prep = summarize_preprocess(zf, rel["train_fill_values"], rel["normalizer"])

        df_preds = read_xlsx_from_zip(zf, ent["test_preds"])
        test_ids = [str(x) for x in df_preds["pid"].tolist()] if "pid" in df_preds.columns else None

        out["models"].append(
            {
                "task": task,
                "seed_dir": f"seed_{seed}",
                "model_dir": model_dir,
                "paths": {**ent, **rel},
                "n_test": int(len(df_preds)),
                "test_ids_sha1": hashlib.sha1(("\n".join(test_ids or [])).encode()).hexdigest() if test_ids else None,
                "selected_features": feats,
                "n_selected_features": len(feats) if feats else None,
                "best_params": best_params,
                "exp_specs": exp_specs,
                "split_specs": split_specs,
                "preprocess_summary": prep,
            }
        )

    out_path = os.path.join(args.out_dir, "recipe_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(out), f, indent=2, ensure_ascii=False)

    print("Saved:", out_path)
    print("\nHuman checklist (what to open first):")
    print("1) preprocess_summary.impute_strategy + clinical_fill_values -> proves categorical missing values were imputed before one-hot.")
    print("2) preprocess_summary.numeric_scaler / categorical_encoder  -> typically MinMaxScaler + OneHotEncoder.")
    print("3) selected_features length                                -> radiomics usually ~16 for mRMR ratio 0.01 on 1688.")
    print("4) best_params for GBSA                                     -> should match your screenshot.")


def cmd_export_splits(args: argparse.Namespace) -> None:
    """
    Export the exact test split(s) used in the zip as JSON.

    Why:
    - If you want to compare your own re-implementation / deep-feature fusion fairly,
      you MUST reuse the same test IDs. Otherwise, you are comparing on different splits.

    Notes:
    - In this zip, Radiomics and Clinic+Radiomics share the same split (same 90 test IDs).
    - Clinic-only appears to have multiple different splits across models.

    Output:
    - out/splits_index.json (overview)
    - out/split_{task}_{sha1[:8]}.json
    """
    ensure_dir(args.out_dir)

    zf = zipfile.ZipFile(args.zip_path)
    entries = scan_zip(args.zip_path)

    all_ids = None
    if args.clinical_csv:
        dfc = pd.read_csv(args.clinical_csv)
        id_col = None
        for cand in ["PatientID", "patient_id", "pid", "ID", "Id"]:
            if cand in dfc.columns:
                id_col = cand
                break
        if not id_col:
            raise ValueError(f"Cannot find patient id column in clinical_csv. Available columns: {list(dfc.columns)[:20]}")
        all_ids = set(dfc[id_col].astype(str).tolist())

    splits: Dict[str, Dict[str, Any]] = {}
    for ent in entries:
        df = read_xlsx_from_zip(zf, ent["test_preds"])
        test_ids = [str(x) for x in df["pid"].astype(str).tolist()]
        sha = hashlib.sha1(("\n".join(sorted(test_ids))).encode()).hexdigest()
        key = f"{ent['task']}|{sha}"
        if key not in splits:
            splits[key] = {
                "task": ent["task"],
                "test_ids_sha1": sha,
                "n_test": int(len(test_ids)),
                "test_ids": sorted(test_ids),
                "models": [],
            }
        splits[key]["models"].append({"seed": ent["seed"], "model_dir": ent["model_dir"], "test_preds": ent["test_preds"]})

    for _, v in splits.items():
        if all_ids is not None:
            test_set = set(v["test_ids"])
            v["train_ids"] = sorted(list(all_ids - test_set))
            v["n_train"] = int(len(v["train_ids"]))

    index = {}
    for _, v in splits.items():
        task = v["task"]
        sha8 = v["test_ids_sha1"][:8]
        out_name = f"split_{task.replace('+', 'plus')}_{sha8}.json"
        out_path = os.path.join(args.out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(to_builtin(v), f, indent=2, ensure_ascii=False)
        index.setdefault(task, []).append({"sha1": v["test_ids_sha1"], "file": out_name, "n_test": v["n_test"], "n_train": v.get("n_train")})

    index_path = os.path.join(args.out_dir, "splits_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(index), f, indent=2, ensure_ascii=False)

    print("Saved:", index_path)
    print("Generated split files:")
    for task, items in index.items():
        for it in items:
            print(f"  - {task}: {it['file']} (n_test={it['n_test']}, n_train={it.get('n_train')})")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("score", help="Compute Harrell C-index from test_preds_*.xlsx (truth baseline).")
    sp.add_argument("--zip_path", required=True)
    sp.add_argument("--out_dir", required=True)
    sp.add_argument("--bootstrap", type=int, default=200)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("dump-recipe", help="Dump a recipe summary (preprocess + selected features + best params) from the zip.")
    sp.add_argument("--zip_path", required=True)
    sp.add_argument("--out_dir", required=True)
    sp.set_defaults(func=cmd_dump_recipe)

    sp = sub.add_parser("export-splits", help="Export the exact test split IDs used in the zip as JSON (for fair re-runs).")
    sp.add_argument("--zip_path", required=True)
    sp.add_argument("--out_dir", required=True)
    sp.add_argument("--clinical_csv", default=None, help="Optional: provide clinical csv to also output train_ids = all_ids - test_ids.")
    sp.set_defaults(func=cmd_export_splits)

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
