# build_oof_risk_integral.py
import os, glob
import pandas as pd
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--out_csv", default="oof_risk_integral.csv")
    args = ap.parse_args()

    rows = []
    fold_dirs = sorted([d for d in glob.glob(os.path.join(args.exp_dir, "fold_*")) if os.path.isdir(d)])
    assert len(fold_dirs) > 0, f"No fold_* found under {args.exp_dir}"

    for fd in fold_dirs:
        vp = os.path.join(fd, "val_pred.csv")
        if not os.path.exists(vp):
            raise FileNotFoundError(vp)
        df = pd.read_csv(vp)

        # 兼容列名：支持 patient_id, pid, PID
        if "patient_id" in df.columns:
            pid_col = "patient_id"
        elif "pid" in df.columns:
            pid_col = "pid"
        elif "PID" in df.columns:
            pid_col = "PID"
        else:
            raise ValueError(f"{vp} has no patient_id/pid/PID column. columns={list(df.columns)}")

        if "risk_integral" not in df.columns:
            raise ValueError(f"{vp} has no risk_integral column. columns={list(df.columns)}")

        tmp = df[[pid_col, "risk_integral"]].copy()
        tmp.columns = ["pid", "oof_risk_integral"]
        tmp["fold_src"] = os.path.basename(fd)
        rows.append(tmp)

    oof = pd.concat(rows, axis=0, ignore_index=True)

    # 每个 pid 理论上只出现一次（每人只在某一折当 val）
    dup = oof["pid"].duplicated(keep=False)
    if dup.any():
        print("[WARN] duplicated pids in concatenated val_pred.csv. Showing first 20:")
        print(oof.loc[dup].sort_values("pid").head(20))

        # 先保守处理：取平均（你也可改成 assert 直接报错）
        oof = oof.groupby("pid", as_index=False)["oof_risk_integral"].mean()

    oof = oof.sort_values("pid").reset_index(drop=True)
    oof.to_csv(args.out_csv, index=False)

    print(f"[OK] wrote {args.out_csv}")
    print(f"  unique pids = {oof['pid'].nunique()} | rows = {len(oof)}")
    print("  head:")
    print(oof.head(10))

if __name__ == "__main__":
    main()
