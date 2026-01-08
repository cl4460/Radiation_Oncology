"""
R6: Late fusion of 3 risk sources (C/D/T) with nested CV weight tuning.

Inputs (typically produced by R1/R3/R4):
  - risk(C): clinical risk predictions (OOF by outer fold)
  - risk(D): deep/embedding risk predictions (OOF by outer fold)
  - risk(T): traditional radiomics risk predictions (OOF by outer fold)

For each OUTER fold:
  1) Use OUTER-train to tune fusion weights w = (wC,wD,wT), w>=0, sum=1, via inner-CV.
  2) Fit normalization on OUTER-train only; evaluate on OUTER-val using tuned weights.
  3) Save per-fold artifacts + overall summary.

Key fixes vs older version:
  - Normalization is applied for ALL methods, strictly "fit on train, transform on val" in inner-CV and outer-eval.
  - Add single-modality baselines (C-only/D-only/T-only).
  - Add checks: NaN/Inf, near-constant risk, weight degeneration warnings.
  - overall.json includes weight stats + fusion gain vs best single modality.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from statistics import NormalDist

from sksurv.metrics import concordance_index_censored, concordance_index_ipcw


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(p: Path, obj: dict) -> None:
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_structured_y(time_arr: np.ndarray, event_arr: np.ndarray) -> np.ndarray:
    y = np.zeros(len(time_arr), dtype=[("event", bool), ("time", float)])
    y["event"] = event_arr.astype(bool)
    y["time"] = time_arr.astype(float)
    return y


def compute_tau_from_train(
    y_train: np.ndarray,
    y_val: np.ndarray,
    q: float = 0.90,
    eps: float = 1e-6
) -> float:
    """
    Compute tau using TRAIN only (Uno C-index truncation time), capped by VAL max time.
    """
    event_times = y_train["time"][y_train["event"]]
    if len(event_times) == 0:
        return float(np.max(y_val["time"]) - eps)

    tau_train = float(np.quantile(event_times, q))
    tau_train = min(tau_train, float(np.max(event_times)) - eps)

    tau_val_cap = float(np.max(y_val["time"]) - eps)
    tau = min(tau_train, tau_val_cap)

    if (not np.isfinite(tau)) or tau <= 0:
        tau = float(np.max(y_val["time"]) - eps)
    return float(tau)


def eval_metrics(
    y_train: np.ndarray,
    y_val: np.ndarray,
    risk_val: np.ndarray,
    tau: float
) -> Tuple[float, float]:
    har = float(concordance_index_censored(y_val["event"], y_val["time"], risk_val)[0])
    uno = float(concordance_index_ipcw(y_train, y_val, risk_val, tau=tau)[0])
    return har, uno


def enumerate_simplex_weights(step: float) -> np.ndarray:
    """
    Enumerate weights on 3-simplex: wC,wD,wT >= 0, sum=1 with grid resolution 'step'.
    """
    if step <= 0 or step > 1:
        raise ValueError("grid_step must be in (0, 1].")
    n = int(round(1.0 / step))
    weights = []
    for i in range(n + 1):
        wC = i * step
        for j in range(n + 1 - i):
            wD = j * step
            wT = 1.0 - wC - wD
            # numerical tolerance
            if wT < -1e-12:
                continue
            wT = max(0.0, wT)
            s = wC + wD + wT
            if abs(s - 1.0) > 1e-8:
                continue
            weights.append([wC, wD, wT])
    if not weights:
        raise RuntimeError("No simplex weights enumerated. Try a larger grid_step (e.g., 0.1).")
    return np.asarray(weights, dtype=float)


def tie_break(w: np.ndarray) -> float:
    """
    Balanced tie-break: prefer more balanced weights when inner-CV mean ties.
    For sum(w)=1, sum(w^2) is minimized at uniform weights.
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    return float(np.sum(w ** 2))


# -----------------------------
# Risk loading
# -----------------------------
def _pick_risk_col(cols: List[str]) -> str:
    # common variants across your pipeline
    candidates = ["risk", "risk_fused", "risk_min", "risk_pred", "pred", "oof_pred", "risk_integral"]
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Cannot find a risk column in: {cols}")


def read_risk_csv(path: Path, id_col: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if id_col not in df.columns:
        raise ValueError(f"{path}: missing id_col='{id_col}'")
    df[id_col] = df[id_col].astype(str)
    risk_col = _pick_risk_col(df.columns.tolist())
    out = df[[id_col, risk_col]].copy()
    out.rename(columns={risk_col: "risk"}, inplace=True)
    out["risk"] = pd.to_numeric(out["risk"], errors="coerce")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out = out.dropna(subset=["risk"]).copy()
    # de-dup defensively
    out = out.drop_duplicates(subset=[id_col], keep="first").copy()
    return out


# -----------------------------
# Normalization (fit on train only)
# -----------------------------
class ZScoreNorm:
    def __init__(self, x_train: np.ndarray):
        x_train = np.asarray(x_train, dtype=float)
        mu = float(np.mean(x_train))
        sd = float(np.std(x_train, ddof=1)) if len(x_train) >= 2 else 1.0
        if (not np.isfinite(sd)) or sd == 0.0:
            sd = 1.0
        self.mu = mu
        self.sd = sd

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = (x - self.mu) / self.sd
        z = np.where(np.isfinite(z), z, 0.0)
        return z


class RankGaussTrainNorm:
    """
    Non-transductive Rank-Gauss (Gaussian rank):
      - Fit: store sorted TRAIN values
      - Transform: map each x to TRAIN empirical percentile, then apply inverse normal CDF
    This aligns modalities to ~N(0,1) while avoiding using VAL distribution.
    """
    def __init__(self, x_train: np.ndarray):
        x_train = np.asarray(x_train, dtype=float)
        self.sorted_tr = np.sort(x_train, kind="mergesort")
        self.n = int(len(self.sorted_tr))
        self._nd = NormalDist()

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.n <= 1:
            return np.zeros_like(x, dtype=float)
        # k in [1, n]
        k = np.searchsorted(self.sorted_tr, x, side="right").astype(float)
        k = np.clip(k, 1.0, float(self.n))
        # Blom-like plotting position -> strictly in (0,1)
        p = (k - 0.5) / float(self.n)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        z = np.array([self._nd.inv_cdf(float(pi)) for pi in p], dtype=float)
        z = np.where(np.isfinite(z), z, 0.0)
        return z


class QuantileNormalNorm:
    """
    Fit QuantileTransformer on TRAIN, transform any set to approx normal.
    Note: sklearn warns this can be prone to overfitting on very small datasets.
    """
    def __init__(self, x_train: np.ndarray, seed: int = 0):
        x_train = np.asarray(x_train, dtype=float).reshape(-1, 1)
        n = len(x_train)
        n_q = int(min(1000, max(10, n)))
        self.qt = QuantileTransformer(
            n_quantiles=n_q,
            output_distribution="normal",
            subsample=int(1e9),
            random_state=seed,
            copy=True,
        )
        self.qt.fit(x_train)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        z = self.qt.transform(x).reshape(-1)
        z = np.where(np.isfinite(z), z, 0.0)
        return z


def build_norm(method: str, x_train: np.ndarray, seed: int) -> Any:
    if method == "zscore":
        return ZScoreNorm(x_train)
    if method == "rank_gauss":
        return RankGaussTrainNorm(x_train)
    if method == "quantile_normal":
        return QuantileNormalNorm(x_train, seed=seed)
    raise ValueError(f"Unknown norm method: {method}")


def normalize_train_val(
    method: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    """
    Normalize per-modality, fit only on X_train, apply to both X_train and X_val.
    Returns (X_train_n, X_val_n, norm_stats_for_json_or_None).
    """
    X_train = np.asarray(X_train, dtype=float)
    X_val = np.asarray(X_val, dtype=float)

    if method == "none":
        return X_train, X_val, None

    Xtr_n = np.zeros_like(X_train, dtype=float)
    Xva_n = np.zeros_like(X_val, dtype=float)

    stats: Dict[str, Any] = {"method": method, "per_modality": {}}

    for m, name in enumerate(["risk_c", "risk_d", "risk_t"]):
        norm = build_norm(method, X_train[:, m], seed=seed + 1000 * m)
        Xtr_n[:, m] = norm.transform(X_train[:, m])
        Xva_n[:, m] = norm.transform(X_val[:, m])

        # light stats for audit
        if isinstance(norm, ZScoreNorm):
            stats["per_modality"][name] = {"mean": float(norm.mu), "std": float(norm.sd)}
        elif isinstance(norm, RankGaussTrainNorm):
            stats["per_modality"][name] = {"n_train": int(norm.n)}
        else:
            stats["per_modality"][name] = {"type": norm.__class__.__name__}

    return Xtr_n, Xva_n, stats


def warn_risk_quality(fk: str, X: np.ndarray) -> None:
    X = np.asarray(X, dtype=float)
    for idx, nm in enumerate(["risk_c", "risk_d", "risk_t"]):
        col = X[:, idx]
        n_bad = int(np.sum(~np.isfinite(col)))
        sd = float(np.std(col)) if len(col) >= 2 else 0.0
        if n_bad > 0:
            print(f"[WARN] {fk}: {nm} has {n_bad} non-finite values in train.")
        if sd < 1e-10:
            print(f"[WARN] {fk}: {nm} is nearly constant in train (std={sd:.2e}).")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--risk_c_csv", type=str, required=True)
    ap.add_argument("--risk_d_csv", type=str, required=True)
    ap.add_argument("--risk_t_csv", type=str, required=True)
    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--id_col", type=str, default="PatientID")
    ap.add_argument("--time_col", type=str, default="Survival.time")
    ap.add_argument("--event_col", type=str, default="deadstatus.event")

    ap.add_argument("--metric", type=str, choices=["harrell", "uno"], default="uno",
                    help="Which metric to optimize in inner-CV weight selection.")
    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--tau_q", type=float, default=0.90)
    ap.add_argument("--grid_step", type=float, default=0.05)

    ap.add_argument("--normalize", type=str,
                    choices=["none", "zscore", "rank_gauss", "quantile_normal"],
                    default="rank_gauss")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dominance_thresh", type=float, default=0.95)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Load risks
    rC = read_risk_csv(Path(args.risk_c_csv), args.id_col).rename(columns={"risk": "risk_c"})
    rD = read_risk_csv(Path(args.risk_d_csv), args.id_col).rename(columns={"risk": "risk_d"})
    rT = read_risk_csv(Path(args.risk_t_csv), args.id_col).rename(columns={"risk": "risk_t"})

    # Load labels
    lab = pd.read_csv(args.labels_csv).copy()
    if args.id_col not in lab.columns:
        raise ValueError(f"labels_csv missing id_col='{args.id_col}'")
    lab[args.id_col] = lab[args.id_col].astype(str)
    if args.time_col not in lab.columns or args.event_col not in lab.columns:
        raise ValueError(f"labels_csv missing [{args.time_col}, {args.event_col}]")
    lab[args.time_col] = pd.to_numeric(lab[args.time_col], errors="coerce")
    lab[args.event_col] = pd.to_numeric(lab[args.event_col], errors="coerce").fillna(0).astype(int)

    # Basic label cleaning
    lab = lab.dropna(subset=[args.time_col]).copy()
    lab = lab[lab[args.time_col] > 0].copy()
    lab[args.event_col] = (lab[args.event_col] > 0).astype(int)

    # Merge (inner join on ids present in all three risks + labels)
    df = lab[[args.id_col, args.time_col, args.event_col]].merge(rC, on=args.id_col, how="inner")
    df = df.merge(rD, on=args.id_col, how="inner")
    df = df.merge(rT, on=args.id_col, how="inner")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["risk_c", "risk_d", "risk_t", args.time_col, args.event_col]).copy()

    # Splits
    splits = read_json(Path(args.splits_json))
    fold_keys = sorted(list(splits.keys()))
    if not fold_keys:
        raise ValueError("splits_json has no folds.")

    weight_grid = enumerate_simplex_weights(args.grid_step)

    all_fold_rows: List[Dict[str, Any]] = []
    oof_rows: List[Tuple[str, float, str]] = []

    for fk in fold_keys:
        fold_dir = out_dir / fk
        ensure_dir(fold_dir)

        train_ids = set(map(str, splits[fk]["train"]))
        val_ids = set(map(str, splits[fk]["val"]))

        tr_df = df[df[args.id_col].isin(train_ids)].copy()
        va_df = df[df[args.id_col].isin(val_ids)].copy()
        if len(tr_df) == 0 or len(va_df) == 0:
            raise ValueError(
                f"{fk}: empty train/val after merging risks+labels. "
                f"train={len(tr_df)} val={len(va_df)}"
            )

        # survival arrays
        y_tr = make_structured_y(tr_df[args.time_col].to_numpy(float),
                                 tr_df[args.event_col].to_numpy(int))
        y_va = make_structured_y(va_df[args.time_col].to_numpy(float),
                                 va_df[args.event_col].to_numpy(int))

        X_tr = tr_df[["risk_c", "risk_d", "risk_t"]].to_numpy(float)
        X_va = va_df[["risk_c", "risk_d", "risk_t"]].to_numpy(float)

        warn_risk_quality(fk, X_tr)

        # inner-CV setup (stratify by event)
        strat = y_tr["event"].astype(int)
        skf = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed)

        # Evaluate each weight triple by inner-CV
        cand_scores = np.full((len(weight_grid), args.inner_folds), np.nan, dtype=float)

        for f_i, (itr, iva) in enumerate(skf.split(np.zeros(len(strat)), strat)):
            Xi_tr = X_tr[itr]
            Xi_va = X_tr[iva]
            yi_tr = y_tr[itr]
            yi_va = y_tr[iva]

            # normalize strictly on INNER-train
            Xi_tr_n, Xi_va_n, _ = normalize_train_val(args.normalize, Xi_tr, Xi_va, seed=args.seed + 17 * f_i)

            # tau for Uno computed from INNER-train only
            tau_i = compute_tau_from_train(yi_tr, yi_va, q=args.tau_q)

            for w_idx, w in enumerate(weight_grid):
                r_va = Xi_va_n @ w
                if args.metric == "harrell":
                    score = float(concordance_index_censored(yi_va["event"], yi_va["time"], r_va)[0])
                else:
                    score = float(concordance_index_ipcw(yi_tr, yi_va, r_va, tau=tau_i)[0])
                cand_scores[w_idx, f_i] = score

        mean_scores = np.nanmean(cand_scores, axis=1)
        std_scores = np.nanstd(cand_scores, axis=1, ddof=1)

        best_mean = np.nanmax(mean_scores)
        best_idxs = np.where(np.isfinite(mean_scores) & (np.abs(mean_scores - best_mean) < 1e-12))[0]
        if best_idxs.size == 0:
            raise RuntimeError(f"{fk}: no valid weight candidates (all NaN). Check risks / events / tau.")
        if best_idxs.size == 1:
            best_idx = int(best_idxs[0])
        else:
            tieb = np.array([tie_break(weight_grid[i]) for i in best_idxs], dtype=float)
            best_idx = int(best_idxs[np.argmin(tieb)])

        w_best = weight_grid[best_idx].astype(float)

        # Save inner-CV surface
        inner_df = pd.DataFrame({
            "wC": weight_grid[:, 0],
            "wD": weight_grid[:, 1],
            "wT": weight_grid[:, 2],
            "mean_score": mean_scores,
            "std_score": std_scores,
        })
        inner_df.to_csv(fold_dir / "inner_cv_weights.csv", index=False)

        # Outer eval: fit normalization on FULL outer-train and evaluate on outer-val
        X_tr_n, X_va_n, norm_stats = normalize_train_val(args.normalize, X_tr, X_va, seed=args.seed + 999)

        tau = compute_tau_from_train(y_tr, y_va, q=args.tau_q)
        risk_va = X_va_n @ w_best
        har, uno = eval_metrics(y_tr, y_va, risk_va, tau=tau)

        # Single-modality baselines (on the SAME normalized space)
        baselines: Dict[str, float] = {}
        for m_idx, m_name in enumerate(["C", "D", "T"]):
            risk_va_single = X_va_n[:, m_idx]
            har_s, uno_s = eval_metrics(y_tr, y_va, risk_va_single, tau=tau)
            baselines[f"har_{m_name}_only"] = float(har_s)
            baselines[f"uno_{m_name}_only"] = float(uno_s)

        # Weight degeneration warning
        max_w = float(np.max(w_best))
        dominant_idx = int(np.argmax(w_best))
        dominant_name = ["C", "D", "T"][dominant_idx]
        if max_w >= float(args.dominance_thresh):
            print(f"[WARN] {fk}: weights degenerated to single modality ({dominant_name}={max_w:.2f})")

        # Save per-fold outputs
        weights_pack: Dict[str, Any] = {
            "fold": fk,
            "metric_optimized": args.metric,
            "normalize": args.normalize,
            "grid_step": float(args.grid_step),
            "inner_folds": int(args.inner_folds),
            "seed": int(args.seed),
            "weights": {"wC": float(w_best[0]), "wD": float(w_best[1]), "wT": float(w_best[2])},
            "best_inner_mean": float(best_mean),
        }
        if norm_stats is not None:
            weights_pack["norm_train_stats"] = norm_stats
        write_json(fold_dir / "weights.json", weights_pack)

        metrics: Dict[str, Any] = {
            "fold": fk,
            "n_train": int(len(tr_df)),
            "n_val": int(len(va_df)),
            "tau_q": float(args.tau_q),
            "tau": float(tau),
            "harrell": float(har),
            "uno": float(uno),
            "wC": float(w_best[0]),
            "wD": float(w_best[1]),
            "wT": float(w_best[2]),
            "dominant_modality": dominant_name,
            "dominant_weight": float(max_w),
            **baselines,
        }
        write_json(fold_dir / "metrics.json", metrics)

        pred_df = va_df[[args.id_col]].copy()
        pred_df["risk_fused"] = risk_va.astype(float)
        pred_df.to_csv(fold_dir / "predictions.csv", index=False)

        # collect OOF
        for pid, rv in zip(pred_df[args.id_col].tolist(), pred_df["risk_fused"].tolist()):
            oof_rows.append((str(pid), float(rv), fk))

        all_fold_rows.append(metrics)
        print(f"[{fk}] har={har:.4f} uno={uno:.4f} | w=({w_best[0]:.3f},{w_best[1]:.3f},{w_best[2]:.3f})")

    # Overall summary
    met_df = pd.DataFrame(all_fold_rows)
    met_df.to_csv(out_dir / "summary_by_fold.csv", index=False)

    oof_df = pd.DataFrame(oof_rows, columns=[args.id_col, "risk_fused", "fold"])
    oof_df.to_csv(out_dir / "oof_pred.csv", index=False)

    # weight stats
    overall: Dict[str, Any] = {
        "metric_optimized": args.metric,
        "normalize": args.normalize,
        "grid_step": float(args.grid_step),
        "tau_q": float(args.tau_q),
        "seed": int(args.seed),
        "n_folds": int(len(fold_keys)),
        "har_mean": float(met_df["harrell"].mean()),
        "har_std": float(met_df["harrell"].std(ddof=1)),
        "uno_mean": float(met_df["uno"].mean()),
        "uno_std": float(met_df["uno"].std(ddof=1)),
        "weight_c_mean": float(met_df["wC"].mean()),
        "weight_c_std": float(met_df["wC"].std(ddof=1)),
        "weight_d_mean": float(met_df["wD"].mean()),
        "weight_d_std": float(met_df["wD"].std(ddof=1)),
        "weight_t_mean": float(met_df["wT"].mean()),
        "weight_t_std": float(met_df["wT"].std(ddof=1)),
    }

    # baseline means
    for m in ["C", "D", "T"]:
        overall[f"har_{m}_only_mean"] = float(met_df[f"har_{m}_only"].mean())
        overall[f"uno_{m}_only_mean"] = float(met_df[f"uno_{m}_only"].mean())

    # fusion gain vs best single (by Harrell and by Uno)
    best_single_har = float(max(overall["har_C_only_mean"], overall["har_D_only_mean"], overall["har_T_only_mean"]))
    best_single_uno = float(max(overall["uno_C_only_mean"], overall["uno_D_only_mean"], overall["uno_T_only_mean"]))
    overall["best_single_har_mean"] = best_single_har
    overall["best_single_uno_mean"] = best_single_uno
    overall["fusion_gain_har"] = float(overall["har_mean"] - best_single_har)
    overall["fusion_gain_uno"] = float(overall["uno_mean"] - best_single_uno)

    best_single_mod_har = ["C", "D", "T"][int(np.argmax([overall["har_C_only_mean"], overall["har_D_only_mean"], overall["har_T_only_mean"]]))]
    best_single_mod_uno = ["C", "D", "T"][int(np.argmax([overall["uno_C_only_mean"], overall["uno_D_only_mean"], overall["uno_T_only_mean"]]))]
    overall["best_single_modality_har"] = best_single_mod_har
    overall["best_single_modality_uno"] = best_single_mod_uno

    write_json(out_dir / "overall.json", overall)
    print("[OVERALL]", json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
