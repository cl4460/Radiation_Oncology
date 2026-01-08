# audit_r4_folds.py
import argparse, json
import numpy as np
import pandas as pd

def load_splits(path):
    s = json.load(open(path, "r"))
    # support:
    # fold_k: {"train":[...], "val":[...]} or {"train_pids":[...], "val_pids":[...]}
    folds = []
    for k in sorted(s.keys()):
        d = s[k]
        tr = d.get("train", d.get("train_pids", d.get("train_ids")))
        va = d.get("val", d.get("val_pids", d.get("val_ids")))
        if tr is None or va is None:
            raise ValueError(f"Unrecognized split format in {k}: keys={list(d.keys())}")
        folds.append((k, list(map(str, tr)), list(map(str, va))))
    return folds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--splits_json", required=True)
    ap.add_argument("--id_col", default="PatientID")
    ap.add_argument("--time_col", required=True)
    ap.add_argument("--event_col", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.labels_csv)
    df[args.id_col] = df[args.id_col].astype(str)
    time = pd.to_numeric(df[args.time_col], errors="coerce")
    event = pd.to_numeric(df[args.event_col], errors="coerce").fillna(0).astype(int)
    df["_time_"] = time
    df["_event_"] = event

    folds = load_splits(args.splits_json)

    print("=== Fold Audit ===")
    for fk, tr_ids, va_ids in folds:
        dtr = df[df[args.id_col].isin(tr_ids)].copy()
        dva = df[df[args.id_col].isin(va_ids)].copy()

        # basic sanity
        ntr, nva = len(dtr), len(dva)
        etr, eva = int(dtr["_event_"].sum()), int(dva["_event_"].sum())

        # time stats
        ttr = dtr["_time_"].dropna().to_numpy()
        tva = dva["_time_"].dropna().to_numpy()

        def stats(x):
            if len(x) == 0:
                return {"min": None, "p50": None, "p90": None, "max": None}
            return {
                "min": float(np.min(x)),
                "p50": float(np.median(x)),
                "p90": float(np.quantile(x, 0.90)),
                "max": float(np.max(x)),
            }

        print(f"\n[{fk}] n_train={ntr} events_train={etr} ({etr/max(ntr,1):.3f}) | "
              f"n_val={nva} events_val={eva} ({eva/max(nva,1):.3f})")
        print(f"  train_time: {stats(ttr)}")
        print(f"  val_time:   {stats(tva)}")

        if fk == "fold_3":
            # highlight potential risk
            if etr < 20:
                print("  [WARN] fold_3 train events is very small -> Cox may be unstable.")
            if eva < 5:
                print("  [WARN] fold_3 val events is very small -> C-index variance large.")

if __name__ == "__main__":
    main()
