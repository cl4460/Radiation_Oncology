#!/usr/bin/env python3
import os, glob, argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--oof_csv", default=None)
    ap.add_argument("--oof_pid_col", default="pid", help="PID column name in oof_csv")
    ap.add_argument("--valpred_pid_col", default="patient_id", help="PID column name in val_pred.csv")
    ap.add_argument("--risk_col", default="risk_integral")
    args = ap.parse_args()

    exp_dir = args.exp_dir
    oof_csv = args.oof_csv or os.path.join(exp_dir, "oof_risk_integral.csv")

    oof = pd.read_csv(oof_csv)
    assert args.oof_pid_col in oof.columns, f"oof_csv missing {args.oof_pid_col}, cols={list(oof.columns)[:20]}"
    assert "oof_risk_integral" in oof.columns, f"oof_csv missing oof_risk_integral, cols={list(oof.columns)[:20]}"
    oof_map = dict(zip(oof[args.oof_pid_col].astype(str), oof["oof_risk_integral"].astype(float)))

    fold_dirs = sorted([d for d in glob.glob(os.path.join(exp_dir, "fold_*")) if os.path.isdir(d)])
    assert len(fold_dirs) > 0, f"no fold_* under {exp_dir}"

    print(f"[OOF] {oof_csv}")
    print(f"  unique pids={len(oof_map)}")

    for fd in fold_dirs:
        fold = os.path.basename(fd)
        vp = os.path.join(fd, "val_pred.csv")
        assert os.path.exists(vp), f"missing {vp}"
        df = pd.read_csv(vp)
        assert args.valpred_pid_col in df.columns, f"{vp} missing {args.valpred_pid_col}"
        assert args.risk_col in df.columns, f"{vp} missing {args.risk_col}"

        pids = df[args.valpred_pid_col].astype(str).tolist()
        v = df[args.risk_col].astype(float).to_numpy()
        o = np.array([oof_map[p] for p in pids], dtype=float)

        maxdiff = np.max(np.abs(v - o))
        same = np.allclose(v, o, rtol=0.0, atol=1e-10)
        print(f"[{fold}] val_pred vs oof_map: allclose={same} | max|diff|={maxdiff:.12g} | n={len(pids)}")

    print("[DONE] If any fold shows maxdiff>1e-10, STOP and fix mapping before training.")

if __name__ == "__main__":
    main()
