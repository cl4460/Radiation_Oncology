# make_stratified_splits_surv.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def make_time_bins(time, event, n_bins=3):
    # 用 event=1 的时间做分箱更合理（避免大量删失时间干扰）
    t_event = time[event.astype(bool)]
    if len(t_event) < n_bins:
        # fallback：用全部时间
        t_event = time
    qs = np.quantile(t_event, np.linspace(0, 1, n_bins + 1))
    # 去重避免边界重复
    qs = np.unique(qs)
    # bin index: 0..(len(qs)-2)
    bins = np.digitize(time, qs[1:-1], right=True)
    return bins

def main(labels_csv, out_json, seed=42,
         id_col="PatientID", time_col="Survival.time", event_col="deadstatus.event",
         stage_col=None, n_splits=5, n_repeats=10, n_time_bins=3):
    lab = pd.read_csv(labels_csv).copy()
    lab[id_col] = lab[id_col].astype(str)
    time = pd.to_numeric(lab[time_col], errors="coerce").to_numpy()
    event = pd.to_numeric(lab[event_col], errors="coerce").fillna(0).astype(int).to_numpy()

    # stage 可选：没有就不加
    if stage_col is not None and stage_col in lab.columns:
        stage = lab[stage_col].astype(str).fillna("NA").to_numpy()
    else:
        stage = np.array(["NA"] * len(lab), dtype=object)

    # time bins
    time_bins = make_time_bins(time, event, n_bins=n_time_bins)

    # 分层标签 = event + time_bin + stage
    strata = np.array([f"{e}_T{tb}_S{st}" for e, tb, st in zip(event, time_bins, stage)], dtype=object)

    pids = lab[id_col].to_numpy()

    all_splits = {}
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for k, (tr, va) in enumerate(skf.split(np.zeros(len(pids)), strata)):
            key = f"rep{r}_fold{k}"
            all_splits[key] = {
                "train": pids[tr].astype(str).tolist(),
                "val": pids[va].astype(str).tolist(),
            }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(all_splits, indent=2))
    print(f"[OK] wrote {len(all_splits)} folds -> {out_json}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage_col", default="Overall_Stage")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--n_repeats", type=int, default=10)
    ap.add_argument("--n_time_bins", type=int, default=3)
    args = ap.parse_args()
    main(args.labels_csv, args.out_json, seed=args.seed,
         stage_col=args.stage_col, n_splits=args.n_splits,
         n_repeats=args.n_repeats, n_time_bins=args.n_time_bins)
