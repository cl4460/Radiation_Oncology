# diagnose_splits.py
import json
from pathlib import Path
import pandas as pd
import numpy as np

def _safe_col(df, col):
    return col in df.columns and df[col].notna().any()

def main(
    labels_csv: str,
    splits_json: str,
    id_col: str = "PatientID",
    time_col: str = "Survival.time",
    event_col: str = "deadstatus.event",
    stage_col: str | None = None,
):
    lab = pd.read_csv(labels_csv).copy()
    lab[id_col] = lab[id_col].astype(str)
    lab[time_col] = pd.to_numeric(lab[time_col], errors="coerce")
    lab[event_col] = pd.to_numeric(lab[event_col], errors="coerce").fillna(0).astype(int)

    if stage_col is not None and stage_col not in lab.columns:
        print(f"[WARN] stage_col={stage_col} not in labels_csv columns, skip stage stats.")
        stage_col = None

    with open(splits_json, "r") as f:
        splits = json.load(f)

    rows = []
    for fk in sorted(splits.keys()):
        val_ids = set(map(str, splits[fk]["val"]))
        dfv = lab[lab[id_col].isin(val_ids)].copy()

        n = len(dfv)
        n_event = int(dfv[event_col].sum())
        event_rate = n_event / max(n, 1)

        # follow-up / time stats
        t = dfv[time_col].dropna().values
        med_t = float(np.median(t)) if len(t) else np.nan
        q25 = float(np.quantile(t, 0.25)) if len(t) else np.nan
        q75 = float(np.quantile(t, 0.75)) if len(t) else np.nan

        row = {
            "fold": fk,
            "n_val": n,
            "n_event": n_event,
            "event_rate": event_rate,
            "time_median": med_t,
            "time_q25": q25,
            "time_q75": q75,
        }

        if stage_col is not None:
            vc = dfv[stage_col].astype(str).value_counts(dropna=False)
            # 只保留前若干类，避免输出太长
            top = vc.head(10).to_dict()
            row["stage_top_counts"] = top

        rows.append(row)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    # 给一个非常直接的“异常折提示”
    print("\n[Heuristic flags]")
    # event_rate 异常
    er = out["event_rate"].values
    for i, fk in enumerate(out["fold"].tolist()):
        if np.isfinite(er[i]) and (er[i] < np.quantile(er, 0.2) or er[i] > np.quantile(er, 0.8)):
            print(f" - {fk}: event_rate={er[i]:.3f} looks imbalanced vs other folds.")
    # time 异常
    tm = out["time_median"].values
    for i, fk in enumerate(out["fold"].tolist()):
        if np.isfinite(tm[i]) and (tm[i] < np.quantile(tm, 0.2) or tm[i] > np.quantile(tm, 0.8)):
            print(f" - {fk}: median follow-up/time={tm[i]:.1f} looks shifted vs other folds.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--splits_json", required=True)
    ap.add_argument("--id_col", default="PatientID")
    ap.add_argument("--time_col", default="Survival.time")
    ap.add_argument("--event_col", default="deadstatus.event")
    ap.add_argument("--stage_col", default=None)
    args = ap.parse_args()
    main(
        labels_csv=args.labels_csv,
        splits_json=args.splits_json,
        id_col=args.id_col,
        time_col=args.time_col,
        event_col=args.event_col,
        stage_col=args.stage_col,
    )
