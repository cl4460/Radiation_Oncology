
import argparse
import json
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw


# ============================================================
# IO utils
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)

def _json_default(o):
    # numpy / pathlib safe serialization
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def mean_std(arr: List[float]) -> Tuple[float, float]:
    x = np.asarray([v for v in arr if v is not None and np.isfinite(v)], dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        return float(x.mean()), 0.0
    return float(x.mean()), float(x.std(ddof=1))


# ============================================================
# Version / params dump
# ============================================================

def args_to_dict(args: argparse.Namespace) -> dict:
    d = vars(args).copy()
    # convert Path-like to str for JSON
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
    return d

def get_versions() -> dict:
    # Keep this lightweight and robust
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import sklearn
        out["sklearn"] = getattr(sklearn, "__version__", "unknown")
    except Exception:
        out["sklearn"] = "unknown"
    try:
        import sksurv
        out["sksurv"] = getattr(sksurv, "__version__", "unknown")
    except Exception:
        out["sksurv"] = "unknown"
    try:
        out["numpy"] = np.__version__
    except Exception:
        out["numpy"] = "unknown"
    try:
        out["pandas"] = pd.__version__
    except Exception:
        out["pandas"] = "unknown"
    return out

def dump_params(outdir: Path, args: argparse.Namespace) -> None:
    payload = {
        "exp_id": str(getattr(args, "exp_id", "R5")),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "versions": get_versions(),
        "args": args_to_dict(args),
    }
    write_json(outdir / "params.json", payload)


# ============================================================
# Survival utils
# ============================================================

def make_surv_struct(event: np.ndarray, time_arr: np.ndarray) -> np.ndarray:
    e = event.astype(bool)
    t = time_arr.astype(float)
    y = np.empty(len(t), dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = e
    y["time"] = t
    return y

def compute_tau_from_train(y_train: np.ndarray, quantile: float = 0.9) -> float:
    times = y_train["time"].astype(float)
    events = y_train["event"].astype(bool)
    ev_times = times[events]
    if ev_times.shape[0] >= 5:
        tau = float(np.quantile(ev_times, quantile))
    else:
        tau = float(np.quantile(times, quantile))
    max_time = float(np.max(times))
    if tau >= max_time:
        tau = max_time - 1e-6
    return tau

def safe_uno_cindex(y_train: np.ndarray, y_val: np.ndarray, risk_val: np.ndarray, tau: float) -> float:
    try:
        return float(concordance_index_ipcw(y_train, y_val, risk_val, tau=tau)[0])
    except Exception:
        return float("nan")

def harrell_cindex(y: np.ndarray, risk: np.ndarray) -> float:
    return float(concordance_index_censored(y["event"], y["time"], risk)[0])

def score_metric(metric: str, y_tr: np.ndarray, y_va: np.ndarray, risk_va: np.ndarray, tau_q: float) -> float:
    if metric == "uno":
        tau = compute_tau_from_train(y_tr, quantile=tau_q)
        return safe_uno_cindex(y_tr, y_va, risk_va, tau=tau)
    return harrell_cindex(y_va, risk_va)


# ============================================================
# OOF risk normalization (optional, for pooled C-index)
# ============================================================

def normalize_risk_for_oof(risk: np.ndarray, mode: str) -> np.ndarray:
    """
    IMPORTANT: pooled OOF C-index mixes predictions from different fold models.
    Risk scales may not be comparable across folds (especially for Cox linear predictors),
    so this provides an optional fold-wise normalization for the pooled OOF report only.

    Per-fold metrics are always computed on raw risk.
    """
    r = np.asarray(risk, dtype=float)
    mode = str(mode or "none").lower()
    if mode == "none":
        return r
    if mode == "zscore":
        mu = float(np.nanmean(r))
        sd = float(np.nanstd(r))
        if not np.isfinite(sd) or sd <= 1e-12:
            return r - mu
        return (r - mu) / sd
    if mode == "rank":
        # map to [0, 1]; ties get average rank
        tmp = pd.Series(r)
        return tmp.rank(method="average", na_option="keep", pct=True).to_numpy(dtype=float)
    raise ValueError(f"Unknown --oof_norm={mode}. Choose from: none, zscore, rank")


# ============================================================
# Preprocess utils
# ============================================================

def make_onehot_encoder(drop: str = "none") -> OneHotEncoder:
    # sklearn versions differ on sparse_output arg name
    drop_arg = None if drop == "none" else drop
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=drop_arg)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, drop=drop_arg)

def build_preprocessor(num_cols: List[str], cat_cols: List[str], onehot_drop: str = "none") -> ColumnTransformer:
    transformers = []

    if len(num_cols) > 0:
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, num_cols))

    if len(cat_cols) > 0:
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder(drop=onehot_drop)),
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    if len(transformers) == 0:
        raise ValueError("No num_cols/cat_cols to preprocess.")

    try:
        return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)
    except TypeError:
        return ColumnTransformer(transformers=transformers, remainder="drop")

def get_feature_names(prep: ColumnTransformer) -> Optional[List[str]]:
    try:
        if hasattr(prep, "get_feature_names_out"):
            return [str(x) for x in prep.get_feature_names_out()]
        if hasattr(prep, "get_feature_names"):
            return [str(x) for x in prep.get_feature_names()]
        return None
    except Exception:
        return None


# ============================================================
# Data loading & typing
# ============================================================

def read_features_only(features_csv: Path,
                       id_col: str,
                       time_col: str,
                       event_col: str) -> pd.DataFrame:
    """
    Read feature CSV and drop label columns if present.
    """
    X_df = pd.read_csv(features_csv).copy()
    if id_col not in X_df.columns:
        raise ValueError(f"features_csv missing id_col={id_col}: {features_csv}")
    X_df[id_col] = X_df[id_col].astype(str)

    # drop any accidental label columns from the feature CSV
    for c in [time_col, event_col]:
        if c in X_df.columns:
            X_df = X_df.drop(columns=[c])

    X_df = X_df.replace(r"^\s*$", np.nan, regex=True)
    return X_df

def merge_with_labels(features_df: pd.DataFrame,
                      labels_csv: Path,
                      id_col: str,
                      time_col: str,
                      event_col: str) -> pd.DataFrame:
    y_df = pd.read_csv(labels_csv).copy()
    if id_col not in y_df.columns:
        raise ValueError(f"labels_csv missing id_col={id_col}: {labels_csv}")
    if time_col not in y_df.columns or event_col not in y_df.columns:
        raise ValueError(f"labels_csv missing columns: need [{time_col}, {event_col}]")

    y_df[id_col] = y_df[id_col].astype(str)

    X_df = features_df.copy()
    X_df[id_col] = X_df[id_col].astype(str)

    # ensure labels not duplicated
    for c in [time_col, event_col]:
        if c in X_df.columns:
            X_df = X_df.drop(columns=[c])

    df = X_df.merge(y_df[[id_col, time_col, event_col]], on=id_col, how="inner")
    df = df.replace(r"^\s*$", np.nan, regex=True)

    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")

    df = df.dropna(subset=[time_col, event_col]).copy()
    df = df[df[time_col] > 0].copy()
    df[event_col] = (df[event_col] > 0).astype(int)
    return df

def validate_unique_ids(df: pd.DataFrame, id_col: str, name: str) -> None:
    if id_col not in df.columns:
        raise ValueError(f"[{name}] missing id_col={id_col}")
    dup = df[id_col][df[id_col].astype(str).duplicated()]
    if len(dup) > 0:
        bad = dup.astype(str).value_counts().head(10).to_dict()
        raise ValueError(f"[{name}] duplicated IDs found (showing up to 10): {bad}. "
                         f"Please deduplicate before merging.")

def prefix_features(df: pd.DataFrame, id_col: str, tag: str) -> pd.DataFrame:
    """
    Prefix all columns except id_col with '{tag}__' to avoid collisions and to track modality.
    """
    if tag is None:
        tag = ""
    tag = str(tag)
    prefix = f"{tag}__" if tag != "" else ""
    rename = {c: f"{prefix}{c}" for c in df.columns if c != id_col}
    return df.rename(columns=rename)

def load_features_dataframe(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Returns:
      features_df: contains id_col + feature columns only (NO labels)
      origin_map: feature_name -> tag (Cp/Ce/Df/Tb/UNK)
    """
    id_col = str(args.id_col)
    time_col = str(args.time_col)
    event_col = str(args.event_col)

    features_csv = getattr(args, "features_csv", None)
    deep_csv = getattr(args, "deep_features_csv", None)
    rad_csv = getattr(args, "radiomics_features_csv", None)
    clin_csv = getattr(args, "clinical_features_csv", None)

    tag_deep = str(getattr(args, "tag_deep", "Df"))
    tag_rad = str(getattr(args, "tag_radiomics", "Tb"))
    tag_clin = str(getattr(args, "tag_clinical", "Cp"))
    no_prefix = bool(getattr(args, "no_feature_prefix", False))

    # mutually exclusive
    if features_csv and (deep_csv or rad_csv or clin_csv):
        raise ValueError("Provide either --features_csv OR (--deep_features_csv/--radiomics_features_csv/--clinical_features_csv), not both.")

    if features_csv:
        X_df = read_features_only(Path(features_csv), id_col=id_col, time_col=time_col, event_col=event_col)
        validate_unique_ids(X_df, id_col=id_col, name="features_csv")

        # origin map inferred by tags in column names (if any)
        origin_map: Dict[str, str] = {}
        tags = [tag_clin, tag_deep, tag_rad]
        for c in X_df.columns:
            if c == id_col:
                continue
            origin_map[c] = infer_feature_origin_from_name(c, tags=tags, fallback="UNK")
        return X_df, origin_map

    # else: merge separate sources
    sources: List[Tuple[str, str, str]] = []
    # (name, path, tag)
    if deep_csv:
        sources.append(("deep", str(deep_csv), tag_deep))
    if rad_csv:
        sources.append(("radiomics", str(rad_csv), tag_rad))
    if clin_csv:
        sources.append(("clinical", str(clin_csv), tag_clin))

    if len(sources) == 0:
        raise ValueError("No features provided. Set --features_csv or at least one of --deep_features_csv/--radiomics_features_csv/--clinical_features_csv.")

    dfs = []
    origin_map: Dict[str, str] = {}
    for name, path, tag in sources:
        df0 = read_features_only(Path(path), id_col=id_col, time_col=time_col, event_col=event_col)
        validate_unique_ids(df0, id_col=id_col, name=name)
        if not no_prefix:
            df0 = prefix_features(df0, id_col=id_col, tag=tag)
        dfs.append(df0)

    merge_how = str(getattr(args, "merge_how", "inner")).lower()
    if merge_how not in {"inner", "outer", "left", "right"}:
        raise ValueError(f"Invalid --merge_how={merge_how}. Choose from inner/outer/left/right.")
    X_df = dfs[0]
    for df1 in dfs[1:]:
        X_df = X_df.merge(df1, on=id_col, how=merge_how)

    validate_unique_ids(X_df, id_col=id_col, name="merged_features")

    # build origin map
    tags = [tag_clin, tag_deep, tag_rad]
    for c in X_df.columns:
        if c == id_col:
            continue
        origin_map[c] = infer_feature_origin_from_name(c, tags=tags, fallback="UNK")

    # optionally dump merged features CSV
    if bool(getattr(args, "dump_merged_features_csv", False)):
        outdir = Path(str(args.outdir))
        ensure_dir(outdir)
        fname = str(getattr(args, "merged_features_filename", "merged_features.csv"))
        if not fname.endswith(".csv"):
            fname += ".csv"
        out_path = outdir / fname
        X_df.to_csv(out_path, index=False)
        print(f"[R5] merged features dumped to: {out_path}")

    return X_df, origin_map

def select_feature_cols(df: pd.DataFrame,
                        id_col: str,
                        time_col: str,
                        event_col: str,
                        feature_cols_arg: Optional[List[str]]) -> List[str]:
    if feature_cols_arg and len(feature_cols_arg) > 0:
        cols = [c for c in feature_cols_arg if c in df.columns]
        if len(cols) == 0:
            raise ValueError(f"--feature_cols provided but none exist in CSV: {feature_cols_arg}")
        return cols
    excluded = {id_col, time_col, event_col}
    return [c for c in df.columns if c not in excluded]

def infer_num_cat_cols(df: pd.DataFrame,
                       feature_cols: List[str],
                       explicit_cat_cols: Optional[List[str]],
                       numeric_frac_thresh: float = 0.9) -> Tuple[List[str], List[str]]:
    # If user passed --cat_cols (even empty), we honor it.
    if explicit_cat_cols is not None:
        cat_cols = [c for c in explicit_cat_cols if c in feature_cols]
        num_cols = [c for c in feature_cols if c not in cat_cols]
        return num_cols, cat_cols

    num_cols, cat_cols = [], []
    for c in feature_cols:
        s_num = pd.to_numeric(df[c], errors="coerce")
        frac = float(s_num.notna().mean())
        if frac >= numeric_frac_thresh:
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols

def enforce_types_inplace(df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> None:
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat_cols:
        s = df[c].astype("string").str.strip()
        df[c] = s.replace({"": pd.NA})


# ============================================================
# Inner CV selection (supports Coxnet + optional Ridge-CoxPH)
# ============================================================

@dataclass
class BestParams:
    kind: str              # "coxnet" or "coxph_ridge"
    l1_ratio: float        # 0.0 for ridge branch; >0 for coxnet
    alpha: float
    inner_score_mean: float
    inner_score_se: float
    alpha_select: str
    total_conv_warnings: int = 0

def _warn_count(wlist) -> int:
    return int(sum(1 for ww in wlist if issubclass(ww.category, ConvergenceWarning)))

def fit_alpha_path_coxnet(X: np.ndarray, y: np.ndarray,
                          l1_ratio: float, n_alphas: int,
                          alpha_min_ratio: Union[str, float],
                          tol: float, max_iter: int) -> Tuple[np.ndarray, int]:
    model = CoxnetSurvivalAnalysis(
        n_alphas=int(n_alphas),
        alphas=None,
        alpha_min_ratio=alpha_min_ratio,
        l1_ratio=float(l1_ratio),
        tol=float(tol),
        max_iter=int(max_iter),
        normalize=False,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.fit(X, y)
        n_warn = _warn_count(w)

    alphas = np.asarray(model.alphas_, dtype=float)
    alphas = alphas[np.isfinite(alphas)]
    if alphas.size > 0:
        alphas = np.unique(alphas)
        alphas = np.sort(alphas)[::-1]  # descending: strong -> weak regularization
    return alphas, n_warn

def _mean_se(x: List[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    m = float(arr.mean())
    if arr.size <= 1:
        return m, 0.0
    se = float(arr.std(ddof=1) / np.sqrt(arr.size))
    return m, se

def inner_select_coxnet(X_all: np.ndarray, y_all: np.ndarray,
                        l1_ratio: float, inner_folds: int, seed: int,
                        n_alphas: int, alpha_min_ratio: Union[str, float],
                        tol: float, max_iter: int,
                        metric: str, tau_q: float,
                        alpha_select: str) -> Optional[BestParams]:
    if not (float(l1_ratio) > 0.0):
        return None

    skf = StratifiedKFold(n_splits=int(inner_folds), shuffle=True, random_state=int(seed))
    y_event = y_all["event"].astype(int)

    alphas, warn0 = fit_alpha_path_coxnet(X_all, y_all, l1_ratio, n_alphas, alpha_min_ratio, tol, max_iter)
    if alphas.size == 0:
        return None

    scores: List[List[float]] = [[] for _ in range(len(alphas))]
    total_warn = int(warn0)

    for tr_idx, va_idx in skf.split(X_all, y_event):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        model = CoxnetSurvivalAnalysis(
            n_alphas=len(alphas),
            alphas=alphas,
            l1_ratio=float(l1_ratio),
            tol=float(tol),
            max_iter=int(max_iter),
            normalize=False,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X_tr, y_tr)
            total_warn += _warn_count(w)

        for i, a in enumerate(alphas):
            try:
                risk_va = model.predict(X_va, alpha=float(a)).astype(float)
                sc = score_metric(metric, y_tr, y_va, risk_va, tau_q)
                if np.isfinite(sc):
                    scores[i].append(float(sc))
            except Exception:
                continue

    mean_sc = np.array([_mean_se(s)[0] for s in scores], dtype=float)
    se_sc = np.array([_mean_se(s)[1] for s in scores], dtype=float)

    if not np.isfinite(mean_sc).any():
        return None

    best_idx = int(np.nanargmax(mean_sc))
    chosen_idx = best_idx
    alpha_select = str(alpha_select).lower()
    if alpha_select == "1se":
        # choose the *largest alpha* (strongest penalty) within 1 standard error of the best mean
        threshold = float(mean_sc[best_idx] - se_sc[best_idx])
        ok = np.where(mean_sc >= threshold)[0]
        if ok.size > 0:
            chosen_idx = int(ok[np.argmax(alphas[ok])])  # pick largest alpha (strongest penalty), order-independent
    elif alpha_select != "best":
        raise ValueError(f"Unknown alpha_select={alpha_select}. Choose from best/1se.")

    return BestParams(
        kind="coxnet",
        l1_ratio=float(l1_ratio),
        alpha=float(alphas[chosen_idx]),
        inner_score_mean=float(mean_sc[chosen_idx]),
        inner_score_se=float(se_sc[chosen_idx]),
        alpha_select=str(alpha_select),
        total_conv_warnings=int(total_warn),
    )

def inner_select_ridge_coxph(X_all: np.ndarray, y_all: np.ndarray,
                             inner_folds: int, seed: int,
                             alpha_grid: np.ndarray,
                             coxph_tol: float, coxph_max_iter: int, ties: str,
                             metric: str, tau_q: float,
                             alpha_select: str) -> Optional[BestParams]:
    skf = StratifiedKFold(n_splits=int(inner_folds), shuffle=True, random_state=int(seed))
    y_event = y_all["event"].astype(int)

    scores: List[List[float]] = [[] for _ in range(len(alpha_grid))]
    total_warn = 0

    for tr_idx, va_idx in skf.split(X_all, y_event):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]

        for j, a in enumerate(alpha_grid):
            try:
                est = CoxPHSurvivalAnalysis(alpha=float(a),
                                           ties=str(ties),
                                           tol=float(coxph_tol),
                                           n_iter=int(coxph_max_iter))
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    est.fit(X_tr, y_tr)
                    total_warn += _warn_count(w)

                risk_va = est.predict(X_va).astype(float)
                sc = score_metric(metric, y_tr, y_va, risk_va, tau_q)
                if np.isfinite(sc):
                    scores[j].append(float(sc))
            except Exception:
                continue

    mean_sc = np.array([_mean_se(s)[0] for s in scores], dtype=float)
    se_sc = np.array([_mean_se(s)[1] for s in scores], dtype=float)
    if not np.isfinite(mean_sc).any():
        return None

    best_idx = int(np.nanargmax(mean_sc))
    chosen_idx = best_idx
    alpha_select = str(alpha_select).lower()
    if alpha_select == "1se":
        threshold = float(mean_sc[best_idx] - se_sc[best_idx])
        ok = np.where(mean_sc >= threshold)[0]
        if ok.size > 0:
            # choose the *largest alpha* (strongest penalty) within 1-SE
            chosen_idx = int(ok[np.argmax(alpha_grid[ok])])
    elif alpha_select != "best":
        raise ValueError(f"Unknown alpha_select={alpha_select}. Choose from best/1se.")

    return BestParams(
        kind="coxph_ridge",
        l1_ratio=0.0,
        alpha=float(alpha_grid[chosen_idx]),
        inner_score_mean=float(mean_sc[chosen_idx]),
        inner_score_se=float(se_sc[chosen_idx]),
        alpha_select=str(alpha_select),
        total_conv_warnings=int(total_warn),
    )


def ridge_alpha_grid(ridge_alpha_min: float, ridge_alpha_max: float, ridge_n_alphas: int) -> np.ndarray:
    """Create a log-spaced ridge alpha grid (ascending).

    Notes:
    - Larger alpha => stronger L2 penalty.
    - np.logspace(log10(min), log10(max), n) is ascending when min < max.
    """
    a_min = float(ridge_alpha_min)
    a_max = float(ridge_alpha_max)
    n = int(ridge_n_alphas)

    if n <= 0:
        raise ValueError("ridge_n_alphas must be > 0")
    if a_min <= 0 or a_max <= 0:
        raise ValueError("ridge_alpha_min/max must be > 0")
    if a_min > a_max:
        a_min, a_max = a_max, a_min

    return np.logspace(np.log10(a_min), np.log10(a_max), n).astype(float)


def select_best_params_inner_cv(X_all: np.ndarray, y_all: np.ndarray,
                                l1_ratios: List[float],
                                inner_folds: int, seed: int,
                                n_alphas: int, alpha_min_ratio: Union[str, float],
                                tol: float, max_iter: int,
                                ridge_alpha_min: float, ridge_alpha_max: float, ridge_n_alphas: int,
                                coxph_tol: float, coxph_max_iter: int, ties: str,
                                metric: str, tau_q: float,
                                enable_ridge: bool,
                                alpha_select: str) -> BestParams:
    best: Optional[BestParams] = None

    if enable_ridge:
        ridge_grid = ridge_alpha_grid(ridge_alpha_min, ridge_alpha_max, ridge_n_alphas)
    else:
        ridge_grid = np.asarray([], dtype=float)

    for l1 in l1_ratios:
        l1 = float(l1)
        cand: Optional[BestParams] = None

        if l1 == 0.0:
            if not enable_ridge:
                continue
            cand = inner_select_ridge_coxph(
                X_all, y_all,
                inner_folds=inner_folds, seed=seed,
                alpha_grid=ridge_grid,
                coxph_tol=coxph_tol, coxph_max_iter=coxph_max_iter, ties=ties,
                metric=metric, tau_q=tau_q,
                alpha_select=alpha_select,
            )
        else:
            cand = inner_select_coxnet(
                X_all, y_all,
                l1_ratio=l1,
                inner_folds=inner_folds, seed=seed,
                n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio,
                tol=tol, max_iter=max_iter,
                metric=metric, tau_q=tau_q,
                alpha_select=alpha_select,
            )

        if cand is None:
            continue
        if best is None or cand.inner_score_mean > best.inner_score_mean:
            best = cand

    if best is None:
        raise RuntimeError("Inner selection failed for all candidates.")
    return best


# ============================================================
# Coxnet fallback for all-zero solution
# ============================================================

def count_nonzero_coef(coef_vec: np.ndarray, eps: float = 1e-10) -> int:
    v = np.asarray(coef_vec, dtype=float).reshape(-1)
    return int(np.sum(np.abs(v) > eps))

def refit_coxnet_fixed_alpha(X_tr: np.ndarray, y_tr: np.ndarray,
                             l1_ratio: float, alpha: float,
                             tol: float, max_iter: int) -> Tuple[CoxnetSurvivalAnalysis, int]:
    model = CoxnetSurvivalAnalysis(
        n_alphas=1,
        alphas=np.asarray([float(alpha)], dtype=float),
        l1_ratio=float(l1_ratio),
        tol=float(tol),
        max_iter=int(max_iter),
        normalize=False,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model.fit(X_tr, y_tr)
        n_warn = _warn_count(w)
    return model, int(n_warn)

def get_coxnet_coef_vec(model: CoxnetSurvivalAnalysis) -> np.ndarray:
    coef = np.asarray(model.coef_, dtype=float)
    if coef.ndim == 1:
        return coef.reshape(-1)
    # expected: (n_features, n_alphas)
    return coef[:, 0].reshape(-1)

def fallback_alpha_if_all_zero_coxnet(X_tr: np.ndarray, y_tr: np.ndarray,
                                     best_l1: float, best_alpha: float,
                                     n_alphas: int, alpha_min_ratio: Union[str, float],
                                     tol: float, max_iter: int,
                                     min_nonzero: int, max_steps: int,
                                     eps: float = 1e-10) -> Tuple[float, int, int]:
    alphas, warn0 = fit_alpha_path_coxnet(X_tr, y_tr, best_l1, n_alphas, alpha_min_ratio, tol, max_iter)
    if alphas.size == 0:
        return float(best_alpha), 0, int(warn0)

    i0 = int(np.argmin(np.abs(alphas - float(best_alpha))))
    warn_total = int(warn0)

    # walk towards smaller alpha => less regularization => more non-zeros
    for k in range(i0, min(len(alphas), i0 + max_steps + 1)):
        a = float(alphas[k])
        model_k, wn = refit_coxnet_fixed_alpha(X_tr, y_tr, best_l1, a, tol, max_iter)
        warn_total += wn
        nnz = count_nonzero_coef(get_coxnet_coef_vec(model_k), eps=eps)
        if nnz >= int(min_nonzero):
            return a, int(k - i0), int(warn_total)

    a_last = float(alphas[min(len(alphas) - 1, i0 + max_steps)])
    return a_last, int(min(max_steps, len(alphas) - 1 - i0)), int(warn_total)


# ============================================================
# Feature origin inference (for selected_features.json)
# ============================================================

def infer_feature_origin_from_name(feature_name: str, tags: List[str], fallback: str = "UNK") -> str:
    """
    Try to infer modality tag by checking whether '{tag}__' substring appears in the feature name.
    Works with ColumnTransformer names like 'num__Df__emb_000'.
    """
    s = str(feature_name)
    for tag in tags:
        if tag and (f"{tag}__" in s):
            return str(tag)
    return str(fallback)

def summarize_selected_features(feature_names: List[str],
                               coef_vec: np.ndarray,
                               origin_tags: List[str],
                               eps: float = 1e-10,
                               topk: Optional[int] = None) -> List[dict]:
    v = np.asarray(coef_vec, dtype=float).reshape(-1)
    out = []
    for n, c, tag in zip(feature_names, v, origin_tags):
        if abs(float(c)) > float(eps):
            out.append({
                "feature": str(n),
                "coef": float(c),
                "abs_coef": float(abs(float(c))),
                "origin": str(tag),
            })
    out.sort(key=lambda d: d["abs_coef"], reverse=True)
    if topk is not None and int(topk) > 0:
        out = out[:int(topk)]
    return out


# ============================================================
# Main outer CV
# ============================================================

def validate_args(args: argparse.Namespace) -> None:
    # Input routing
    if getattr(args, "features_csv", None) is None:
        if not (getattr(args, "deep_features_csv", None) or getattr(args, "radiomics_features_csv", None) or getattr(args, "clinical_features_csv", None)):
            raise ValueError("You must provide --features_csv OR at least one of --deep_features_csv/--radiomics_features_csv/--clinical_features_csv.")

    # Mode enforcement
    mode = str(getattr(args, "mode", "coxnet_only")).lower()
    if mode not in {"coxnet_only", "allow_ridge"}:
        raise ValueError("--mode must be one of: coxnet_only, allow_ridge")

    l1s = [float(x) for x in getattr(args, "l1_ratios", [])]
    if any((x < 0.0 or x > 1.0) for x in l1s):
        raise ValueError(f"--l1_ratios must be within [0, 1]. Got: {l1s}")

    if mode == "coxnet_only":
        # enforce strict: no 0.0 allowed
        if any(abs(x - 0.0) < 1e-12 for x in l1s):
            raise ValueError("In --mode coxnet_only, l1_ratio=0.0 is not allowed. "
                             "Remove 0.0 from --l1_ratios, or rerun with --mode allow_ridge if you intentionally want ridge-CoxPH.")
        # disable ridge regardless of flags
        args.enable_ridge = False

    if mode == "allow_ridge":
        # If user includes 0.0, ridge must be enabled (otherwise it's a silent no-op)
        if any(abs(x - 0.0) < 1e-12 for x in l1s) and not bool(getattr(args, "enable_ridge", False)):
            raise ValueError("You included l1_ratio=0.0 in --l1_ratios but ridge is disabled. "
                             "Either remove 0.0 or add --enable_ridge.")

    # alpha_select validation
    alpha_select = str(getattr(args, "alpha_select", "best")).lower()
    if alpha_select not in {"best", "1se"}:
        raise ValueError("--alpha_select must be 'best' or '1se'")

    # oof_norm validation
    _ = normalize_risk_for_oof(np.asarray([0.0, 1.0]), mode=str(getattr(args, "oof_norm", "none")).lower())

def run_r5(args: argparse.Namespace) -> None:
    validate_args(args)

    outdir = Path(str(args.outdir))
    ensure_dir(outdir)
    np.random.seed(int(args.seed))

    # dump full params early (before any heavy compute)
    dump_params(outdir, args)

    # load features (single CSV or merge multi-CSV), then attach labels
    X_df, origin_map_raw = load_features_dataframe(args)
    df = merge_with_labels(X_df, Path(str(args.labels_csv)),
                           id_col=str(args.id_col),
                           time_col=str(args.time_col),
                           event_col=str(args.event_col))

    feature_cols = select_feature_cols(df, str(args.id_col), str(args.time_col), str(args.event_col), args.feature_cols)
    num_cols, cat_cols = infer_num_cat_cols(df, feature_cols, args.cat_cols, numeric_frac_thresh=args.numeric_frac_thresh)
    enforce_types_inplace(df, num_cols=num_cols, cat_cols=cat_cols)

    print(f"[R5] exp_id={getattr(args, 'exp_id', 'R5')} | mode={getattr(args, 'mode', 'coxnet_only')}")
    print(f"[R5] n={len(df)} | raw_features={len(feature_cols)} | num={len(num_cols)} cat={len(cat_cols)}")
    if len(cat_cols) > 0:
        print(f"[R5] cat_cols={cat_cols[:10]}{' ...' if len(cat_cols) > 10 else ''}")
    print(f"[R5] enable_ridge={args.enable_ridge} | inner_metric={args.inner_metric} | alpha_select={args.alpha_select}")

    splits = read_json(Path(str(args.splits_json)))
    fold_keys = sorted(list(splits.keys()))

    all_fold_metrics = []
    oof_rows = []

    # selected feature aggregation
    per_fold_selected: Dict[str, dict] = {}
    # for aggregate frequency
    selected_freq: Dict[str, int] = {}
    selected_coef_sum: Dict[str, float] = {}
    selected_abscoef_sum: Dict[str, float] = {}
    selected_origin: Dict[str, str] = {}

    # tags for origin inference
    tags = [str(getattr(args, "tag_clinical", "Cp")),
            str(getattr(args, "tag_deep", "Df")),
            str(getattr(args, "tag_radiomics", "Tb"))]

    for fold_name in fold_keys:
        t0 = time.time()
        train_ids = [str(x) for x in splits[fold_name]["train"]]
        val_ids = [str(x) for x in splits[fold_name]["val"]]

        df_tr = df[df[str(args.id_col)].isin(train_ids)].copy()
        df_va = df[df[str(args.id_col)].isin(val_ids)].copy()
        if df_tr.shape[0] == 0 or df_va.shape[0] == 0:
            raise ValueError(f"[R5][{fold_name}] empty split: train={df_tr.shape[0]}, val={df_va.shape[0]}")

        X_tr_raw = df_tr[feature_cols].copy()
        X_va_raw = df_va[feature_cols].copy()
        y_tr = make_surv_struct(df_tr[str(args.event_col)].to_numpy(), df_tr[str(args.time_col)].to_numpy())
        y_va = make_surv_struct(df_va[str(args.event_col)].to_numpy(), df_va[str(args.time_col)].to_numpy())

        prep = build_preprocessor(num_cols, cat_cols, onehot_drop=str(args.onehot_drop))
        X_tr = prep.fit_transform(X_tr_raw)
        X_va = prep.transform(X_va_raw)

        n_feat = int(X_tr.shape[1])
        fold_idx = int(str(fold_name).split("_")[-1]) if "fold_" in str(fold_name) else fold_keys.index(fold_name)
        print(f"[R5][{fold_name}] train={len(df_tr)} val={len(df_va)} | raw={len(feature_cols)} -> preprocessed={n_feat}")

        best = select_best_params_inner_cv(
            X_all=X_tr, y_all=y_tr,
            l1_ratios=args.l1_ratios,
            inner_folds=args.inner_folds,
            seed=int(args.seed) + 1000 + int(fold_idx),
            n_alphas=args.n_alphas,
            alpha_min_ratio=args.alpha_min_ratio,
            tol=args.tol,
            max_iter=args.max_iter,
            ridge_alpha_min=args.ridge_alpha_min,
            ridge_alpha_max=args.ridge_alpha_max,
            ridge_n_alphas=args.ridge_n_alphas,
            coxph_tol=args.coxph_tol,
            coxph_max_iter=args.coxph_max_iter,
            ties=args.coxph_ties,
            metric=args.inner_metric,
            tau_q=args.tau_q,
            enable_ridge=bool(args.enable_ridge),
            alpha_select=str(args.alpha_select),
        )

        alpha_used = float(best.alpha)
        fallback_steps = 0
        conv_warn_fallback = 0
        conv_warn_final = 0
        nnz = None

        if best.kind == "coxnet":
            if args.fallback_to_nonzero_alpha:
                tmp_model, wn0 = refit_coxnet_fixed_alpha(X_tr, y_tr, best.l1_ratio, alpha_used, args.tol, args.max_iter)
                coef_tmp = get_coxnet_coef_vec(tmp_model)
                nnz0 = count_nonzero_coef(coef_tmp, eps=float(args.selected_features_eps))
                if nnz0 < args.fallback_min_nonzero:
                    alpha_used, fallback_steps, conv_warn_fallback = fallback_alpha_if_all_zero_coxnet(
                        X_tr, y_tr,
                        best_l1=best.l1_ratio,
                        best_alpha=best.alpha,
                        n_alphas=args.n_alphas,
                        alpha_min_ratio=args.alpha_min_ratio,
                        tol=args.tol,
                        max_iter=args.max_iter,
                        min_nonzero=args.fallback_min_nonzero,
                        max_steps=args.fallback_max_steps,
                        eps=float(args.selected_features_eps),
                    )
                    print(f"[WARNING][R5][{fold_name}] all-zero at alpha={best.alpha:.6g}. "
                          f"Fallback -> alpha_used={alpha_used:.6g} (steps={fallback_steps})")

            model, conv_warn_final = refit_coxnet_fixed_alpha(X_tr, y_tr, best.l1_ratio, alpha_used, args.tol, args.max_iter)
            coef_vec = get_coxnet_coef_vec(model)
            nnz = count_nonzero_coef(coef_vec, eps=float(args.selected_features_eps))
            risk_pred_raw = model.predict(X_va, alpha=float(alpha_used)).astype(float)
            kind = "coxnet"

        else:
            # ridge CoxPH
            est = CoxPHSurvivalAnalysis(alpha=float(alpha_used), ties=args.coxph_ties,
                                        tol=args.coxph_tol, n_iter=int(args.coxph_max_iter))
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                est.fit(X_tr, y_tr)
                conv_warn_final = _warn_count(w)
            model = est
            coef_vec = np.asarray(model.coef_, dtype=float).reshape(-1)
            nnz = count_nonzero_coef(coef_vec, eps=float(args.selected_features_eps))
            risk_pred_raw = model.predict(X_va).astype(float)
            kind = "coxph_ridge"

        # per-fold metrics computed on raw risk
        har_final = harrell_cindex(y_va, risk_pred_raw)
        tau = compute_tau_from_train(y_tr, quantile=args.tau_q)
        uno_final = safe_uno_cindex(y_tr, y_va, risk_pred_raw, tau=tau)

        # pooled OOF uses optional per-fold normalization
        risk_pred_oof = normalize_risk_for_oof(risk_pred_raw, mode=str(args.oof_norm))

        fold_dir = outdir / f"fold_{fold_idx}"
        ensure_dir(fold_dir)

        pred_df = pd.DataFrame({
            str(args.id_col): df_va[str(args.id_col)].astype(str).to_numpy(),
            "time": y_va["time"].astype(float),
            "event": y_va["event"].astype(bool),
            "risk_pred_raw": risk_pred_raw.astype(float),
            "risk_pred_oof": risk_pred_oof.astype(float),
        })
        pred_df.to_csv(fold_dir / "predictions.csv", index=False)

        if args.save_model:
            joblib.dump({"prep": prep, "model": model, "kind": kind, "alpha_used": alpha_used}, fold_dir / "model.joblib")

        # coefficients + selected features
        feat_names = get_feature_names(prep)
        if feat_names is None or len(feat_names) != len(coef_vec):
            feat_names = [f"feature_{i}" for i in range(len(coef_vec))]

        # origin tags for each preprocessed feature name
        origin_tags = [infer_feature_origin_from_name(n, tags=tags, fallback="UNK") for n in feat_names]

        coef_dict = {n: float(v) for n, v in zip(feat_names, coef_vec)}

        write_json(fold_dir / "coefficients.json", {
            "kind": kind,
            "selected_l1_ratio": float(best.l1_ratio),
            "selected_alpha": float(best.alpha),
            "alpha_used_final": float(alpha_used),
            "alpha_select": str(best.alpha_select),
            "inner_score_mean": float(best.inner_score_mean),
            "inner_score_se": float(best.inner_score_se),
            "fallback_steps": int(fallback_steps),
            "n_features": int(len(coef_vec)),
            "n_nonzero_coef": int(nnz),
            "sparsity_ratio": round(1.0 - nnz / max(1, len(coef_vec)), 6),
            "feature_names_available": True,
            "coefficients": coef_dict,
        })

        selected_list = summarize_selected_features(
            feature_names=feat_names,
            coef_vec=coef_vec,
            origin_tags=origin_tags,
            eps=float(args.selected_features_eps),
            topk=(int(args.selected_features_topk) if args.selected_features_topk is not None else None),
        )
        # per-fold selected_features.json (compact)
        write_json(fold_dir / "selected_features.json", {
            "fold": int(fold_idx),
            "kind": kind,
            "selected_l1_ratio": float(best.l1_ratio),
            "alpha_used_final": float(alpha_used),
            "eps_nonzero": float(args.selected_features_eps),
            "n_selected": int(len(selected_list)),
            "selected": selected_list,
        })

        # update aggregate selection stats
        for d in selected_list:
            fname = d["feature"]
            selected_freq[fname] = int(selected_freq.get(fname, 0) + 1)
            selected_coef_sum[fname] = float(selected_coef_sum.get(fname, 0.0) + float(d["coef"]))
            selected_abscoef_sum[fname] = float(selected_abscoef_sum.get(fname, 0.0) + float(d["abs_coef"]))
            selected_origin[fname] = str(d.get("origin", "UNK"))

        # per-fold params.json (for reproducibility)
        write_json(fold_dir / "params.json", {
            "exp_id": str(getattr(args, "exp_id", "R5")),
            "fold": int(fold_idx),
            "fold_name": str(fold_name),
            "mode": str(getattr(args, "mode", "coxnet_only")),
            "selected": asdict(best),
            "alpha_used_final": float(alpha_used),
            "tau": float(tau),
            "tau_q": float(args.tau_q),
        })

        metrics = {
            "exp_id": str(getattr(args, "exp_id", "R5")),
            "fold": int(fold_idx),
            "fold_name": str(fold_name),
            "mode": str(getattr(args, "mode", "coxnet_only")),
            "kind": kind,
            "n_train": int(df_tr.shape[0]),
            "n_val": int(df_va.shape[0]),
            "n_features_raw": int(len(feature_cols)),
            "n_features_preprocessed": int(n_feat),
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "selected_l1_ratio": float(best.l1_ratio),
            "selected_alpha": float(best.alpha),
            "alpha_used_final": float(alpha_used),
            "alpha_select": str(best.alpha_select),
            "inner_best_score_mean": float(best.inner_score_mean),
            "inner_best_score_se": float(best.inner_score_se),
            "inner_metric": str(args.inner_metric),
            "conv_warnings_inner_total": int(best.total_conv_warnings),
            "conv_warnings_fallback": int(conv_warn_fallback),
            "conv_warnings_final_fit": int(conv_warn_final),
            "fallback_enabled": bool(args.fallback_to_nonzero_alpha),
            "fallback_steps": int(fallback_steps),
            "fallback_min_nonzero": int(args.fallback_min_nonzero),
            "n_nonzero_coef_final": int(nnz),
            "harrell_c_final": round(float(har_final), 6),
            "uno_c_final": None if np.isnan(uno_final) else round(float(uno_final), 6),
            "tau": round(float(tau), 6),
            "tau_q": float(args.tau_q),
            "oof_norm": str(args.oof_norm),
            "runtime_seconds": round(float(time.time() - t0), 2),
        }
        write_json(fold_dir / "metrics.json", metrics)
        all_fold_metrics.append(metrics)

        for idx in range(len(pred_df)):
            row = pred_df.iloc[idx]
            oof_rows.append({
                str(args.id_col): str(row[str(args.id_col)]),
                "fold": int(fold_idx),
                "time": float(row["time"]),
                "event": bool(row["event"]),
                "risk_pred_raw": float(row["risk_pred_raw"]),
                "risk_pred_oof": float(row["risk_pred_oof"]),
            })

        print(f"[R5][{fold_name}] HarrellC={har_final:.4f} UnoC={uno_final:.4f} | "
              f"kind={kind} l1={best.l1_ratio} alpha={best.alpha:.6g} (sel={best.alpha_select}) nnz={nnz}")

    # ---- write OOF + fold metrics
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(outdir / "oof_predictions.csv", index=False)

    met_df = pd.DataFrame(all_fold_metrics)
    met_df.to_csv(outdir / "per_fold_metrics.csv", index=False)

    # ---- fold-average (raw, per-fold)
    har_vals = met_df["harrell_c_final"].astype(float).tolist()
    uno_vals = [float(x) if x is not None else np.nan for x in met_df["uno_c_final"].tolist()]
    har_mean, har_std = mean_std(har_vals)
    uno_mean, uno_std = mean_std(uno_vals)

    # ---- pooled OOF: raw and normalized
    har_oof_raw = float(concordance_index_censored(
        oof_df["event"].to_numpy().astype(bool),
        oof_df["time"].to_numpy().astype(float),
        oof_df["risk_pred_raw"].to_numpy().astype(float)
    )[0])

    har_oof_oofnorm = float(concordance_index_censored(
        oof_df["event"].to_numpy().astype(bool),
        oof_df["time"].to_numpy().astype(float),
        oof_df["risk_pred_oof"].to_numpy().astype(float)
    )[0])

    y_oof = make_surv_struct(oof_df["event"].to_numpy().astype(int), oof_df["time"].to_numpy().astype(float))
    tau_oof = compute_tau_from_train(y_oof, quantile=args.tau_q)
    uno_oof_raw = safe_uno_cindex(y_oof, y_oof, oof_df["risk_pred_raw"].to_numpy().astype(float), tau=tau_oof)
    uno_oof_oofnorm = safe_uno_cindex(y_oof, y_oof, oof_df["risk_pred_oof"].to_numpy().astype(float), tau=tau_oof)

    # ---- selected features aggregate
    agg_list = []
    for fname, freq in selected_freq.items():
        k = int(freq)
        agg_list.append({
            "feature": str(fname),
            "origin": str(selected_origin.get(fname, "UNK")),
            "freq": int(k),
            "freq_ratio": float(k / max(1, len(fold_keys))),
            "mean_coef": float(selected_coef_sum.get(fname, 0.0) / max(1, k)),
            "mean_abs_coef": float(selected_abscoef_sum.get(fname, 0.0) / max(1, k)),
        })
    agg_list.sort(key=lambda d: (d["freq"], d["mean_abs_coef"]), reverse=True)

    # modality counts
    mod_cnt: Dict[str, int] = {}
    for d in agg_list:
        mod = str(d["origin"])
        mod_cnt[mod] = int(mod_cnt.get(mod, 0) + 1)

    selected_features_payload = {
        "exp_id": str(getattr(args, "exp_id", "R5")),
        "mode": str(getattr(args, "mode", "coxnet_only")),
        "eps_nonzero": float(args.selected_features_eps),
        "topk_per_fold": (int(args.selected_features_topk) if args.selected_features_topk is not None else None),
        "tags": {
            "clinical": str(getattr(args, "tag_clinical", "Cp")),
            "deep": str(getattr(args, "tag_deep", "Df")),
            "radiomics": str(getattr(args, "tag_radiomics", "Tb")),
        },
        "aggregate": {
            "n_unique_selected": int(len(agg_list)),
            "by_origin_unique_count": mod_cnt,
            "features": agg_list,
        },
    }
    write_json(outdir / "selected_features.json", selected_features_payload)

    # ---- summary / metrics at root
    summary = {
        "exp_id": str(getattr(args, "exp_id", "R5")),
        "mode": str(getattr(args, "mode", "coxnet_only")),
        "n_folds": int(len(met_df)),
        "harrell_c_mean": round(float(har_mean), 6),
        "harrell_c_std": round(float(har_std), 6),
        "uno_c_mean": None if np.isnan(uno_mean) else round(float(uno_mean), 6),
        "uno_c_std": None if np.isnan(uno_std) else round(float(uno_std), 6),
        "harrell_c_oof_raw": round(float(har_oof_raw), 6),
        "uno_c_oof_raw": None if np.isnan(uno_oof_raw) else round(float(uno_oof_raw), 6),
        "harrell_c_oof_oofnorm": round(float(har_oof_oofnorm), 6),
        "uno_c_oof_oofnorm": None if np.isnan(uno_oof_oofnorm) else round(float(uno_oof_oofnorm), 6),
        "oof_norm": str(args.oof_norm),
        "l1_ratios_candidates": [float(x) for x in args.l1_ratios],
        "inner_folds": int(args.inner_folds),
        "inner_metric": str(args.inner_metric),
        "alpha_select": str(args.alpha_select),
        "enable_ridge": bool(args.enable_ridge),
        "note": "Nested-CV selects (l1_ratio, alpha) using inner-CV only. Per-fold metrics use raw risk. "
                "Pooled OOF is reported both as raw and fold-normalized (oof_norm) to avoid scale artifacts.",
    }
    write_json(outdir / "summary.json", summary)
    write_json(outdir / "metrics.json", {"summary": summary, "per_fold": all_fold_metrics})

    print("\n" + "=" * 72)
    print("[R5] Done.")
    print(f"[R5] Per-fold avg (raw): HarrellC={summary['harrell_c_mean']:.4f}±{summary['harrell_c_std']:.4f} | "
          f"UnoC={summary['uno_c_mean']}±{summary['uno_c_std']}")
    print(f"[R5] Overall OOF raw: HarrellC={summary['harrell_c_oof_raw']:.4f} | UnoC={summary['uno_c_oof_raw']}")
    print(f"[R5] Overall OOF ({summary['oof_norm']}): HarrellC={summary['harrell_c_oof_oofnorm']:.4f} | UnoC={summary['uno_c_oof_oofnorm']}")
    print(f"[R5] Outputs: {outdir.resolve()}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("R5: Early fusion (Coxnet) with nested CV + optional ridge-CoxPH")

    # ---- experiment id / mode
    ap.add_argument("--exp_id", type=str, default="R5",
                    help="Experiment ID to store in outputs (does not change folder name).")
    ap.add_argument("--mode", type=str, default="coxnet_only",
                    choices=["coxnet_only", "allow_ridge"],
                    help="coxnet_only: forbid l1_ratio=0 and disable ridge branch. "
                         "allow_ridge: allow l1_ratio=0 to trigger ridge-CoxPH branch.")

    # ---- inputs
    ap.add_argument("--features_csv", type=str, default=None,
                    help="Single feature CSV (already merged). Mutually exclusive with *_features_csv.")
    ap.add_argument("--deep_features_csv", type=str, default=None,
                    help="Deep feature CSV (Df). If provided with others, will be merged by id_col.")
    ap.add_argument("--radiomics_features_csv", type=str, default=None,
                    help="Radiomics feature CSV (Tb). If provided with others, will be merged by id_col.")
    ap.add_argument("--clinical_features_csv", type=str, default=None,
                    help="Clinical feature CSV (Cp/Ce). If provided with others, will be merged by id_col.")
    ap.add_argument("--merge_how", type=str, default="inner",
                    choices=["inner", "outer", "left", "right"],
                    help="How to merge multiple feature CSVs when more than one is provided.")
    ap.add_argument("--no_feature_prefix", action="store_true",
                    help="If set, do NOT prefix columns by modality tag when merging multiple feature CSVs. "
                         "Not recommended (may cause column name collisions and lose origin tracking).")
    ap.add_argument("--tag_deep", type=str, default="Df",
                    help="Tag for deep features (prefix '{tag}__' when merging multi-CSV).")
    ap.add_argument("--tag_radiomics", type=str, default="Tb",
                    help="Tag for radiomics features (prefix '{tag}__' when merging multi-CSV).")
    ap.add_argument("--tag_clinical", type=str, default="Cp",
                    help="Tag for clinical features (prefix '{tag}__' when merging multi-CSV). "
                         "Use 'Ce' if you merge extended clinical.")
    ap.add_argument("--dump_merged_features_csv", action="store_true",
                    help="If set and multi-CSV merge is used, dump the merged feature table (without labels) to outdir.")
    ap.add_argument("--merged_features_filename", type=str, default="merged_features.csv",
                    help="Filename for dumped merged features (only if --dump_merged_features_csv).")

    ap.add_argument("--labels_csv", type=str, required=True)
    ap.add_argument("--splits_json", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--id_col", type=str, default="PatientID")
    ap.add_argument("--time_col", type=str, default="Survival.time")
    ap.add_argument("--event_col", type=str, default="deadstatus.event")

    # ---- feature selection / preprocessing config
    ap.add_argument("--feature_cols", type=str, nargs="*", default=None)
    ap.add_argument("--cat_cols", type=str, nargs="*", default=None,
                    help="Categorical columns. Omit flag => heuristic; pass flag alone => no categoricals.")
    ap.add_argument("--numeric_frac_thresh", type=float, default=0.9)
    ap.add_argument("--onehot_drop", type=str, choices=["none", "first"], default="none",
                    help="OneHotEncoder drop mode. For penalized models, 'none' avoids bias across categories.")

    # ---- model selection config
    ap.add_argument("--l1_ratios", type=float, nargs="+",
                    default=[0.01, 0.05, 0.1, 0.2, 0.5],
                    help="Candidates. 0.0 triggers ridge CoxPH branch only when --mode allow_ridge.")
    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--inner_metric", type=str, default="harrell", choices=["harrell", "uno"])
    ap.add_argument("--alpha_select", type=str, default="best", choices=["best", "1se"],
                    help="Pick alpha by best mean inner-CV score or 1-SE rule (more regularized).")

    # Coxnet path settings
    ap.add_argument("--n_alphas", type=int, default=100)
    ap.add_argument("--alpha_min_ratio", type=str, default="auto")
    ap.add_argument("--tol", type=float, default=1e-7)
    ap.add_argument("--max_iter", type=int, default=100000)

    # ridge CoxPH settings (only if allow_ridge and l1_ratio==0.0 is used)
    ap.add_argument("--enable_ridge", dest="enable_ridge", action="store_true",
                    help="Enable ridge CoxPH branch when l1_ratio==0.0 is included (only effective in allow_ridge mode).")
    ap.add_argument("--disable_ridge", dest="enable_ridge", action="store_false",
                    help="Disable ridge branch (skip l1_ratio==0.0).")
    ap.set_defaults(enable_ridge=True)

    ap.add_argument("--ridge_alpha_min", type=float, default=1e-4)
    ap.add_argument("--ridge_alpha_max", type=float, default=1e4)
    ap.add_argument("--ridge_n_alphas", type=int, default=60)

    ap.add_argument("--coxph_tol", type=float, default=1e-9)
    ap.add_argument("--coxph_max_iter", type=int, default=100)
    ap.add_argument("--coxph_ties", type=str, default="efron", choices=["efron", "breslow"])

    # ---- evaluation config
    ap.add_argument("--tau_q", type=float, default=0.9)
    ap.add_argument("--oof_norm", type=str, default="zscore", choices=["none", "zscore", "rank"],
                    help="How to normalize risk within each fold for pooled OOF C-index report. "
                         "Per-fold metrics always use raw risk.")

    # ---- run config
    ap.add_argument("--seed", type=int, default=42)

    # ---- Coxnet all-zero fallback
    ap.add_argument("--fallback_to_nonzero_alpha", dest="fallback_to_nonzero_alpha", action="store_true",
                    help="Enable fallback if Coxnet coefficients are all-zero at selected alpha.")
    ap.add_argument("--no_fallback_to_nonzero_alpha", dest="fallback_to_nonzero_alpha", action="store_false",
                    help="Disable fallback mechanism.")
    ap.set_defaults(fallback_to_nonzero_alpha=True)

    ap.add_argument("--fallback_min_nonzero", type=int, default=1)
    ap.add_argument("--fallback_max_steps", type=int, default=10)

    # ---- selected feature dump config
    ap.add_argument("--selected_features_eps", type=float, default=1e-10,
                    help="Non-zero threshold for coefficients when dumping selected_features.json.")
    ap.add_argument("--selected_features_topk", type=int, default=None,
                    help="If set, keep only top-K selected features per fold by |coef| in selected_features.json. "
                         "Does NOT affect training; only affects output size.")

    ap.add_argument("--save_model", action="store_true")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    try:
        if args.alpha_min_ratio != "auto":
            args.alpha_min_ratio = float(args.alpha_min_ratio)
    except Exception:
        pass
    run_r5(args)





#run command:
#clinical
#python r5.py \
#  --exp_id R5_Cp_coxnet \
#  --mode coxnet_only \
#  --features_csv /home/lichengze/Research/DeepFeature/myresearch/main/R0/r0_feature_csvs_update/clinical_features.csv \
#  --labels_csv /home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
#  --splits_json /home/lichengze/Research/DeepFeature/myresearch/main/R0/splits.json \
#  --outdir /home/lichengze/Research/DeepFeature/myresearch/main/R5/clinical_enet_update_2 \
#  --feature_cols age gender Overall_Stage_II Overall_Stage_IIIa Overall_Stage_IIIb Overall_Stage_Unknown clinical_T_Stage Clinical_N_Stage Clinical_M_binary \
#  --cat_cols \
#  --l1_ratios 0.01 0.05 0.1 0.2 0.5 \
#  --inner_folds 5 --inner_metric harrell \
#  --n_alphas 100 --alpha_min_ratio auto \
#  --alpha_select best \
#  --tau_q 0.9 --seed 42 \
#  --oof_norm zscore


#radiomics
#python r5.py \
#  --exp_id R5_Tb \
#  --mode coxnet_only \
#  --features_csv /home/lichengze/Research/DeepFeature/myresearch/main/R4/Tb_bw25_iso111_clip_full/Tb_features.csv \
#  --labels_csv /home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
#  --splits_json /home/lichengze/Research/DeepFeature/myresearch/main/R0/splits.json \
#  --outdir /home/lichengze/Research/DeepFeature/myresearch/main/R5/radiomics_enet_update_3 \
#  --l1_ratios 0.5 0.8 1.0 \
#  --inner_folds 3 \
#  --n_alphas 50 \
#  --max_iter 50000 \
#  --tau_q 0.9 \
#  --oof_norm zscore \
#  --seed 42


#deep feature
#python r5.py   
#  --mode coxnet_only \   
#  --features_csv /home/lichengze/Research/DeepFeature/myresearch/main/R0/r0_feature_csvs_update/deep_features.csv \
#  --labels_csv /home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv \
#  --splits_json /home/lichengze/Research/DeepFeature/myresearch/main/R0/splits.json \
#  --outdir /home/lichengze/Research/DeepFeature/myresearch/main/R5/deep_enet_update_2 \
#  --l1_ratios 0.01 0.05 0.1 0.2 0.5 \
#  --inner_folds 5 --inner_metric harrell \
#  --n_alphas 100 --alpha_min_ratio auto \
#  --alpha_select best \
#  --tau_q 0.9 \
#  --seed 42 \
#  --save_model