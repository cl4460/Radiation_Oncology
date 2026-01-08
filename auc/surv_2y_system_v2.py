#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
surv_2y_system_v2.py - 修复版2年生存概率预测系统

修复的问题：
1. 过拟合 - 添加正则化参数选项，降低模型复杂度
2. Isotonic校准器截断 - 改用Platt scaling + 概率裁剪[0.01, 0.99]
3. 解释失效 - 用原始概率计算delta，确保解释有意义

主要改进：
- 添加 --regularize 选项自动降低模型复杂度
- 改用 Platt scaling (sigmoid) 校准，避免阶梯状截断
- 概率裁剪到 [0.01, 0.99]，永不输出0%或100%
- 解释基于原始概率计算，不受校准影响
- 添加更详细的诊断输出

用法与原版相同：
  python surv_2y_system_v2.py train ...
  python surv_2y_system_v2.py eval ...
  python surv_2y_system_v2.py predict ...
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# scikit-survival
from sksurv.util import Surv
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# =============================================================================
# 常量定义
# =============================================================================
PROB_MIN = 0.01  # 最小概率（永不输出0%）
PROB_MAX = 0.99  # 最大概率（永不输出100%）


# =============================================================================
# 工具函数
# =============================================================================
def read_split_json(split_json: str) -> Tuple[List[str], List[str]]:
    """读取训练/测试划分JSON文件"""
    with open(split_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    for k_tr, k_te in [
        ("train_ids", "test_ids"),
        ("train", "test"),
        ("train_id", "test_id"),
        ("train_patients", "test_patients"),
    ]:
        if k_tr in obj and k_te in obj:
            return list(map(str, obj[k_tr])), list(map(str, obj[k_te]))
    raise ValueError(f"无法识别的split_json格式: {list(obj.keys())}")


def safe_mode(series: pd.Series):
    """安全获取众数"""
    series = series.dropna()
    if series.empty:
        return None
    return series.value_counts().index[0]


def km_survival_at_t(time: np.ndarray, event: np.ndarray, t: float) -> float:
    """计算Kaplan-Meier在时间t的生存概率"""
    if len(time) == 0:
        return float("nan")
    x, y = kaplan_meier_estimator(event.astype(bool), time.astype(float))
    if t <= x[0]:
        return float(y[0])
    if t >= x[-1]:
        return float(y[-1])
    idx = np.searchsorted(x, t, side="right") - 1
    return float(y[idx])


def build_y(df: pd.DataFrame, time_col: str, event_col: str):
    """构建生存分析的y标签"""
    t = df[time_col].astype(float).to_numpy()
    e = df[event_col].astype(int).to_numpy().astype(bool)
    return Surv.from_arrays(event=e, time=t)


def infer_time_unit_hint(df: pd.DataFrame, time_col: str) -> str:
    """推断时间单位（天/月）"""
    t = df[time_col].astype(float)
    p50 = float(t.median())
    if p50 > 100:
        return f"median({time_col})≈{p50:.1f} (看起来是天). 2年=730天."
    return f"median({time_col})≈{p50:.1f} (可能是月). 如果是月, 2年=24."


def load_merged(clinical_csv: str, radiomics_csv: str, id_col: str) -> pd.DataFrame:
    """加载并合并临床和radiomics数据"""
    clin = pd.read_csv(clinical_csv)
    rad = pd.read_csv(radiomics_csv)
    clin[id_col] = clin[id_col].astype(str)
    rad[id_col] = rad[id_col].astype(str)
    df = clin.merge(rad, on=id_col, how="inner")
    return df


def select_features(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    event_col: str,
    clinical_cols: List[str],
    selected_radiomics: List[str],
) -> pd.DataFrame:
    """选择指定的特征列"""
    keep = [id_col, time_col, event_col] + clinical_cols + selected_radiomics
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise KeyError(f"缺失列: {missing[:20]} (共{len(missing)}个)")
    return df[keep].copy()


def read_feature_list(txt_path: str) -> List[str]:
    """从txt文件读取特征列表"""
    feats = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            feats.append(s)
    return feats


def clip_probability(p: np.ndarray) -> np.ndarray:
    """
    裁剪概率到 [PROB_MIN, PROB_MAX] 范围
    
    原因：概率永远不应该精确等于0或1
    - 0% 意味着"绝对不可能生存"，这在医学上不合理
    - 100% 意味着"绝对会生存"，同样不合理
    """
    return np.clip(p, PROB_MIN, PROB_MAX)


# =============================================================================
# 校准器类
# =============================================================================
class PlattCalibrator:
    """
    Platt校准器（Sigmoid校准）
    
    相比Isotonic的优势：
    1. 平滑的映射函数，不会产生阶梯状截断
    2. 在极端概率区域表现更稳定
    3. 参数更少，不容易过拟合
    """
    
    def __init__(self):
        self.lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.fitted = False
    
    def fit(self, p_raw: np.ndarray, y_binary: np.ndarray):
        """
        拟合校准器
        
        Args:
            p_raw: 原始预测概率
            y_binary: 二值标签 (1=在horizon内生存, 0=在horizon内死亡)
        """
        # Logistic回归需要2D输入
        X = p_raw.reshape(-1, 1)
        self.lr.fit(X, y_binary)
        self.fitted = True
        return self
    
    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        """
        校准概率
        """
        if not self.fitted:
            return clip_probability(p_raw)
        X = np.asarray(p_raw).reshape(-1, 1)
        p_cal = self.lr.predict_proba(X)[:, 1]
        return clip_probability(p_cal)


class IsotonicCalibratorSafe:
    """
    安全的Isotonic校准器
    
    改进：
    1. 预测后强制裁剪到 [PROB_MIN, PROB_MAX]
    2. 样本不足时不拟合
    """
    
    def __init__(self, min_samples: int = 30):
        self.iso = IsotonicRegression(y_min=PROB_MIN, y_max=PROB_MAX, out_of_bounds="clip")
        self.fitted = False
        self.min_samples = min_samples
    
    def fit(self, p_raw: np.ndarray, y_binary: np.ndarray):
        if len(p_raw) < self.min_samples:
            print(f"[CAL] 样本不足({len(p_raw)}<{self.min_samples})，跳过Isotonic校准")
            return self
        self.iso.fit(p_raw, y_binary.astype(float))
        self.fitted = True
        return self
    
    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return clip_probability(p_raw)
        p_cal = self.iso.predict(np.asarray(p_raw))
        return clip_probability(p_cal)


# =============================================================================
# 模型包数据类
# =============================================================================
@dataclass
class ModelBundle:
    """模型包，包含推理所需的全部组件"""
    id_col: str
    time_col: str
    event_col: str
    horizon: float
    clinical_cols: List[str]
    radiomics_cols: List[str]
    preprocessor: ColumnTransformer
    model: GradientBoostingSurvivalAnalysis
    calibrator: object  # PlattCalibrator 或 IsotonicCalibratorSafe
    ref_values: Dict[str, object]  # 参考值（用于解释）
    train_risk: np.ndarray  # 训练集风险分布（用于计算百分位）
    version: str = "2.0"  # 版本号


# =============================================================================
# 预处理器构建
# =============================================================================
def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scale: str,
    age_impute: str,
) -> ColumnTransformer:
    """构建预处理流水线"""
    if scale not in ("minmax", "zscore", "none"):
        raise ValueError("scale必须是: minmax, zscore, none之一")

    if scale == "minmax":
        scaler = MinMaxScaler()
    elif scale == "zscore":
        scaler = StandardScaler()
    else:
        scaler = "passthrough"

    num_imputer = SimpleImputer(strategy="median" if age_impute != "mean" else "mean")

    numeric_pipe = Pipeline([
        ("imputer", num_imputer),
        ("scaler", scaler),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


# =============================================================================
# 核心功能函数
# =============================================================================
def predict_survival_at(
    model: GradientBoostingSurvivalAnalysis,
    X: np.ndarray,
    t: float,
) -> np.ndarray:
    """预测在时间t的生存概率"""
    surv_fns = model.predict_survival_function(X)
    return np.asarray([float(fn(t)) for fn in surv_fns], dtype=float)


def risk_percentile(train_risk: np.ndarray, r: float) -> float:
    """计算风险值在训练集中的百分位"""
    return float((train_risk < r).mean())


def fit_calibrator_2y(
    df_train_raw: pd.DataFrame,
    p_surv2y_train: np.ndarray,
    time_col: str,
    event_col: str,
    horizon: float,
    method: str = "platt",
) -> object:
    """
    拟合2年校准器
    
    改进说明：
    - 只使用"已知状态"的样本：
      * 在horizon前发生事件 -> 标签=0（未生存到2年）
      * 随访时间>=horizon -> 标签=1（生存到2年）
      * 在horizon前删失 -> 排除（状态未知）
    """
    t = df_train_raw[time_col].astype(float).to_numpy()
    e = df_train_raw[event_col].astype(int).to_numpy().astype(bool)

    # 已知状态的样本
    known = (t >= horizon) | (e & (t < horizon))
    
    if known.sum() < 30:
        print(f"[CAL] 已知状态样本不足({known.sum()}<30)，跳过校准")
        return None

    # 构建二值标签
    # y2=1: 生存到2年（t>=horizon且在此之前无事件，或t>=horizon）
    # y2=0: 未生存到2年（在horizon前发生事件）
    y2 = np.zeros(len(t), dtype=int)
    y2[t >= horizon] = 1  # 生存到2年
    y2[e & (t < horizon)] = 0  # 在2年内死亡

    # 选择校准方法
    if method == "platt":
        cal = PlattCalibrator()
        cal.fit(p_surv2y_train[known], y2[known])
        print(f"[CAL] Platt校准器已拟合 (n={known.sum()})")
    elif method == "isotonic":
        cal = IsotonicCalibratorSafe()
        cal.fit(p_surv2y_train[known], y2[known])
        if cal.fitted:
            print(f"[CAL] Isotonic校准器已拟合 (n={known.sum()})")
    else:
        cal = None
        print(f"[CAL] 不使用校准")
    
    return cal


def get_regularized_params(base_params: dict, level: str = "moderate") -> dict:
    """
    获取正则化后的GBSA参数
    
    过拟合修复说明：
    - 原始参数: learning_rate=0.57, max_depth=7, min_samples_leaf=3
    - 这些参数太激进，导致训练集C-index=0.95但测试集只有0.66
    
    正则化策略：
    - 降低learning_rate: 让模型学习更慢，减少震荡
    - 减少max_depth: 限制树的深度，防止过度拟合
    - 增加min_samples_leaf: 叶节点需要更多样本，提高泛化
    - 减少n_estimators: 更少的树
    """
    params = base_params.copy()
    
    if level == "none":
        return params
    
    elif level == "light":
        # 轻度正则化
        params["learning_rate"] = min(params.get("learning_rate", 0.1), 0.3)
        params["max_depth"] = min(params.get("max_depth", 5), 5)
        params["min_samples_leaf"] = max(params.get("min_samples_leaf", 5), 5)
        
    elif level == "moderate":
        # 中度正则化（推荐）
        params["learning_rate"] = min(params.get("learning_rate", 0.1), 0.1)
        params["max_depth"] = min(params.get("max_depth", 3), 3)
        params["min_samples_leaf"] = max(params.get("min_samples_leaf", 10), 10)
        params["n_estimators"] = min(params.get("n_estimators", 100), 100)
        params["subsample"] = min(params.get("subsample", 0.8), 0.8)
        
    elif level == "strong":
        # 强正则化
        params["learning_rate"] = 0.05
        params["max_depth"] = 2
        params["min_samples_leaf"] = 15
        params["n_estimators"] = 50
        params["subsample"] = 0.7
    
    return params


# =============================================================================
# 命令：训练
# =============================================================================
def cmd_train(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # 加载数据
    df = load_merged(args.clinical_csv, args.radiomics_csv, args.id_col)
    print("[INFO]", infer_time_unit_hint(df, args.time_col))
    train_ids, test_ids = read_split_json(args.split_json)

    df_train = df[df[args.id_col].astype(str).isin(train_ids)].copy()
    df_test = df[df[args.id_col].astype(str).isin(test_ids)].copy()
    
    print(f"[INFO] 训练集: {len(df_train)} 样本, 测试集: {len(df_test)} 样本")

    # 临床特征列
    clinical_cols = []
    if args.use_age:
        clinical_cols.append(args.age_col)
    if args.use_gender:
        clinical_cols.append(args.gender_col)
    if args.use_histology:
        clinical_cols.append(args.histology_col)
    if args.use_t_stage:
        clinical_cols.append(args.t_stage_col)
    if args.use_stage:
        clinical_cols.append(args.stage_col)

    # Radiomics特征
    if not args.selected_radiomics_txt:
        raise ValueError("必须提供 --selected_radiomics_txt")
    radiomics_cols = read_feature_list(args.selected_radiomics_txt)
    print(f"[INFO] 使用 {len(clinical_cols)} 个临床特征 + {len(radiomics_cols)} 个radiomics特征")

    df_train = select_features(df_train, args.id_col, args.time_col, args.event_col, clinical_cols, radiomics_cols)
    df_test = select_features(df_test, args.id_col, args.time_col, args.event_col, clinical_cols, radiomics_cols)

    # 构建原始特征矩阵
    X_train_raw = df_train[clinical_cols + radiomics_cols].copy()
    X_test_raw = df_test[clinical_cols + radiomics_cols].copy()

    # 区分数值和分类特征
    categorical_cols = []
    numeric_cols = []
    for c in clinical_cols:
        if c in (args.gender_col, args.histology_col, args.t_stage_col, args.stage_col):
            categorical_cols.append(c)
        else:
            numeric_cols.append(c)
    numeric_cols += radiomics_cols

    # 预处理
    pre = build_preprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scale=args.scale,
        age_impute=args.age_impute,
    )

    Xtr = pre.fit_transform(X_train_raw)
    Xte = pre.transform(X_test_raw)

    ytr = build_y(df_train, args.time_col, args.event_col)
    yte = build_y(df_test, args.time_col, args.event_col)

    # GBSA参数
    if args.gbsa_params_json:
        with open(args.gbsa_params_json, "r", encoding="utf-8") as f:
            base_params = json.load(f)
    else:
        base_params = dict(
            loss="coxph",
            learning_rate=0.1,
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=10,
            random_state=args.seed,
        )

    # 应用正则化（修复过拟合）
    params = get_regularized_params(base_params, level=args.regularize)
    if "random_state" not in params:
        params["random_state"] = args.seed
    
    print(f"[INFO] 正则化级别: {args.regularize}")
    print(f"[INFO] GBSA关键参数: lr={params.get('learning_rate')}, "
          f"max_depth={params.get('max_depth')}, "
          f"min_samples_leaf={params.get('min_samples_leaf')}")

    # 训练模型
    model = GradientBoostingSurvivalAnalysis(**params)
    model.fit(Xtr, ytr)

    # 评估C-index
    risk_tr = model.predict(Xtr)
    risk_te = model.predict(Xte)
    c_tr = concordance_index_censored(ytr["event"], ytr["time"], risk_tr)[0]
    c_te = concordance_index_censored(yte["event"], yte["time"], risk_te)[0]
    
    # 过拟合诊断
    overfit_gap = c_tr - c_te
    if overfit_gap > 0.15:
        print(f"[警告] 过拟合风险: Train C={c_tr:.4f}, Test C={c_te:.4f}, Gap={overfit_gap:.4f}")
        print(f"[建议] 尝试 --regularize strong")
    else:
        print(f"[TRAIN] C-index={c_tr:.4f}  [TEST] C-index={c_te:.4f}  Gap={overfit_gap:.4f}")

    # 2年生存概率（原始）
    p2_tr_raw = predict_survival_at(model, Xtr, args.horizon)
    p2_te_raw = predict_survival_at(model, Xte, args.horizon)

    # 校准（使用改进的校准器）
    calibrator = fit_calibrator_2y(
        df_train, p2_tr_raw, args.time_col, args.event_col, args.horizon,
        method=args.calibrate
    )
    
    if calibrator is not None:
        p2_te = calibrator.predict(p2_te_raw)
        p2_tr = calibrator.predict(p2_tr_raw)
    else:
        p2_te = clip_probability(p2_te_raw)
        p2_tr = clip_probability(p2_tr_raw)

    # 概率分布诊断
    print(f"[INFO] 校准后概率分布: min={p2_te.min():.3f}, max={p2_te.max():.3f}, "
          f"mean={p2_te.mean():.3f}, std={p2_te.std():.3f}")

    # 2年指标
    times = np.asarray([args.horizon], dtype=float)
    try:
        _, aucs = cumulative_dynamic_auc(ytr, yte, risk_te, times)
        # Handle both scalar and array returns (scikit-survival version dependent)
        auc2y = float(aucs[0] if isinstance(aucs, np.ndarray) and aucs.ndim > 0 else aucs)
    except Exception as e:
        print(f"[警告] AUC计算失败: {e}")
        auc2y = float("nan")

    try:
        _, bs = brier_score(ytr, yte, np.c_[p2_te], times)
        # Handle both scalar and array returns (scikit-survival version dependent)
        brier2y = float(bs[0] if isinstance(bs, np.ndarray) and bs.ndim > 0 else bs)
    except Exception as e:
        print(f"[警告] Brier计算失败: {e}")
        brier2y = float("nan")

    print(f"[METRICS] AUC@2y={auc2y:.4f}, Brier@2y={brier2y:.4f}")

    # 校准表
    try:
        bins = pd.qcut(p2_te, q=min(args.calib_bins, len(p2_te)), duplicates="drop")
        calib_rows = []
        for b in bins.categories:  # bins is already Categorical, no need for .cat
            idx = (bins == b)  # Already returns numpy array
            obs = km_survival_at_t(
                df_test.loc[df_test.index[idx], args.time_col].to_numpy(),
                df_test.loc[df_test.index[idx], args.event_col].to_numpy(),
                args.horizon,
            )
            pred = float(np.mean(p2_te[idx]))
            calib_rows.append({"bin": str(b), "n": int(idx.sum()), "pred_surv2y": pred, "obs_surv2y_km": obs})
    except Exception as e:
        print(f"[警告] 校准表生成失败: {e}")
        calib_rows = []

    # 参考值（用于解释）
    ref_values = {}
    for c in numeric_cols:
        ref_values[c] = float(np.nanmedian(X_train_raw[c].astype(float).to_numpy()))
    for c in categorical_cols:
        ref_values[c] = safe_mode(X_train_raw[c])

    # 打包模型
    bundle = ModelBundle(
        id_col=args.id_col,
        time_col=args.time_col,
        event_col=args.event_col,
        horizon=float(args.horizon),
        clinical_cols=clinical_cols,
        radiomics_cols=radiomics_cols,
        preprocessor=pre,
        model=model,
        calibrator=calibrator,
        ref_values=ref_values,
        train_risk=risk_tr.astype(float),
        version="2.0",
    )

    joblib_path = os.path.join(args.out_dir, "gbsa_2y_model_v2.joblib")
    joblib.dump(bundle, joblib_path)

    # 保存报告
    out = {
        "version": "2.0",
        "seed": args.seed,
        "horizon": float(args.horizon),
        "regularize": args.regularize,
        "calibrate": args.calibrate,
        "internal_train": {
            "n": int(len(df_train)),
            "cindex": float(c_tr),
        },
        "internal_test": {
            "n": int(len(df_test)),
            "cindex": float(c_te),
            "auc2y": auc2y,
            "brier2y": brier2y,
        },
        "overfit_gap": float(overfit_gap),
        "calibration_bins": calib_rows,
        "gbsa_params_used": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) 
                            for k, v in params.items()},
        "prob_range": [float(p2_te.min()), float(p2_te.max())],
    }
    
    report_path = os.path.join(args.out_dir, "train_report_v2.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # 保存测试集预测
    pred_df = df_test[[args.id_col, args.time_col, args.event_col]].copy()
    pred_df["risk"] = risk_te
    pred_df["surv2y_raw"] = p2_te_raw
    pred_df["surv2y"] = p2_te
    pred_path = os.path.join(args.out_dir, "internal_test_pred_2y_v2.csv")
    pred_df.to_csv(pred_path, index=False)

    print(f"[OK] 模型已保存: {joblib_path}")
    print(f"[OK] 报告已保存: {report_path}")
    print(f"[OK] 预测已保存: {pred_path}")


# =============================================================================
# 命令：评估
# =============================================================================
def cmd_eval(args):
    bundle: ModelBundle = joblib.load(args.model_joblib)

    df = load_merged(args.clinical_csv, args.radiomics_csv, bundle.id_col)
    train_ids, test_ids = read_split_json(args.split_json)

    df_train = df[df[bundle.id_col].astype(str).isin(train_ids)].copy()
    df_test = df[df[bundle.id_col].astype(str).isin(test_ids)].copy()

    df_train = select_features(df_train, bundle.id_col, bundle.time_col, bundle.event_col,
                               bundle.clinical_cols, bundle.radiomics_cols)
    df_test = select_features(df_test, bundle.id_col, bundle.time_col, bundle.event_col,
                              bundle.clinical_cols, bundle.radiomics_cols)

    Xtr_raw = df_train[bundle.clinical_cols + bundle.radiomics_cols].copy()
    Xte_raw = df_test[bundle.clinical_cols + bundle.radiomics_cols].copy()

    Xtr = bundle.preprocessor.transform(Xtr_raw)
    Xte = bundle.preprocessor.transform(Xte_raw)

    ytr = build_y(df_train, bundle.time_col, bundle.event_col)
    yte = build_y(df_test, bundle.time_col, bundle.event_col)

    model = bundle.model
    risk_te = model.predict(Xte)
    c_te = concordance_index_censored(yte["event"], yte["time"], risk_te)[0]

    p2_raw = predict_survival_at(model, Xte, bundle.horizon)
    if bundle.calibrator is not None:
        p2 = bundle.calibrator.predict(p2_raw)
    else:
        p2 = clip_probability(p2_raw)

    times = np.asarray([bundle.horizon], dtype=float)
    try:
        _, aucs = cumulative_dynamic_auc(ytr, yte, risk_te, times)
        # Handle both scalar and array returns (scikit-survival version dependent)
        auc2y = float(aucs[0] if isinstance(aucs, np.ndarray) and aucs.ndim > 0 else aucs)
    except:
        auc2y = float("nan")

    try:
        _, bs = brier_score(ytr, yte, np.c_[p2], times)
        # Handle both scalar and array returns (scikit-survival version dependent)
        brier2y = float(bs[0] if isinstance(bs, np.ndarray) and bs.ndim > 0 else bs)
    except:
        brier2y = float("nan")

    os.makedirs(args.out_dir, exist_ok=True)
    out = {
        "horizon": float(bundle.horizon),
        "internal_test": {
            "cindex": float(c_te),
            "auc2y": float(auc2y),
            "brier2y": float(brier2y),
        },
        "n_test": int(len(df_test)),
        "prob_range": [float(p2.min()), float(p2.max())],
    }
    
    eval_report_path = os.path.join(args.out_dir, "eval_report_v2.json")
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    pred_df = df_test[[bundle.id_col, bundle.time_col, bundle.event_col]].copy()
    pred_df["risk"] = risk_te
    pred_df["surv2y_raw"] = p2_raw
    pred_df["surv2y"] = p2
    pred_path = os.path.join(args.out_dir, "eval_pred_2y_v2.csv")
    pred_df.to_csv(pred_path, index=False)

    print(f"[EVAL] C-index={c_te:.4f}, AUC@2y={auc2y:.4f}, Brier@2y={brier2y:.4f}")
    print(f"[INFO] 概率范围: [{p2.min():.3f}, {p2.max():.3f}]")
    print(f"[OK] 已保存: {eval_report_path}")
    print(f"[OK] 已保存: {pred_path}")


# =============================================================================
# 命令：预测
# =============================================================================
def cmd_predict(args):
    bundle: ModelBundle = joblib.load(args.model_joblib)

    # 加载病人临床数据
    if args.patient_clinical_json:
        with open(args.patient_clinical_json, "r", encoding="utf-8") as f:
            clin_obj = json.load(f)
        clin_row = {k: clin_obj.get(k, None) for k in bundle.clinical_cols}
        pid = str(clin_obj.get(bundle.id_col, args.patient_id))
    else:
        dfc = pd.read_csv(args.patient_clinical_csv)
        if args.patient_id:
            row = dfc[dfc[bundle.id_col].astype(str) == str(args.patient_id)].iloc[0]
        else:
            row = dfc.iloc[0]
        clin_row = {k: row.get(k, None) for k in bundle.clinical_cols}
        pid = str(row.get(bundle.id_col, args.patient_id))

    # 加载病人radiomics数据
    dfr = pd.read_csv(args.patient_radiomics_csv)
    if args.patient_id:
        rr = dfr[dfr[bundle.id_col].astype(str) == str(args.patient_id)].iloc[0]
    else:
        rr = dfr[dfr[bundle.id_col].astype(str) == str(pid)].iloc[0] if (bundle.id_col in dfr.columns) else dfr.iloc[0]
    rad_row = {k: rr.get(k, None) for k in bundle.radiomics_cols}

    # 构建单行原始特征
    x_raw = {**clin_row, **rad_row}
    X_raw = pd.DataFrame([x_raw], columns=bundle.clinical_cols + bundle.radiomics_cols)

    X = bundle.preprocessor.transform(X_raw)
    model = bundle.model

    # 预测
    risk = float(model.predict(X)[0])
    p2_raw = float(predict_survival_at(model, X, bundle.horizon)[0])
    
    # 校准（使用改进的校准器）
    if bundle.calibrator is not None:
        p2 = float(bundle.calibrator.predict(np.array([p2_raw]))[0])
    else:
        p2 = float(clip_probability(np.array([p2_raw]))[0])
    
    pct = risk_percentile(bundle.train_risk, risk)

    # ========================================
    # 关键修复：用原始概率计算解释
    # ========================================
    # 原因：校准后的概率可能被截断（如都变成0），导致delta全为0
    # 解决：用原始概率计算delta，这样解释始终有意义
    
    rows = []
    for feat, ref in bundle.ref_values.items():
        X_cf_raw = X_raw.copy()
        
        # 安全地设置反事实值
        try:
            X_cf_raw.loc[0, feat] = ref
        except Exception:
            X_cf_raw[feat] = X_cf_raw[feat].astype(object)
            X_cf_raw.loc[0, feat] = ref
        
        X_cf = bundle.preprocessor.transform(X_cf_raw)
        
        # 用原始概率计算delta（关键修复！）
        p2_cf_raw = float(predict_survival_at(model, X_cf, bundle.horizon)[0])
        delta = p2_cf_raw - p2_raw  # 基于原始概率
        
        # 获取当前值
        current_val = X_raw.loc[0, feat]
        if pd.isna(current_val):
            current_val = None
        elif isinstance(current_val, (np.floating, np.integer)):
            current_val = float(current_val) if isinstance(current_val, np.floating) else int(current_val)
        
        rows.append({
            "feature": feat,
            "patient_value": current_val,
            "ref_value": ref,
            "delta_surv2y_if_set_to_ref": float(delta),
        })
    
    expl = pd.DataFrame(rows)
    expl["abs_delta"] = expl["delta_surv2y_if_set_to_ref"].abs()
    expl = expl.sort_values("abs_delta", ascending=False).drop(columns=["abs_delta"]).head(args.explain_topk)

    # 保存结果
    os.makedirs(args.out_dir, exist_ok=True)
    
    out_json = {
        "patient_id": pid,
        "horizon": float(bundle.horizon),
        "pred_survival_2y": float(p2),
        "pred_survival_2y_raw": float(p2_raw),
        "pred_survival_2y_pct": f"{p2*100:.1f}%",
        "pred_risk": float(risk),
        "risk_percentile_vs_train": float(pct),
        "risk_percentile_pct": f"{pct*100:.1f}%",
        "interpretation": _generate_interpretation(p2, pct),
        "version": "2.0",
    }
    
    json_path = os.path.join(args.out_dir, f"{pid}_2y_prediction_v2.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    expl_path = os.path.join(args.out_dir, f"{pid}_explanation_top{args.explain_topk}_v2.csv")
    expl.to_csv(expl_path, index=False)

    # 打印结果
    print("=" * 70)
    print(f"患者: {pid}")
    print(f"2年生存概率: {p2:.3f} ({p2*100:.1f}%)")
    print(f"原始概率(未校准): {p2_raw:.3f} ({p2_raw*100:.1f}%)")
    print(f"风险评分: {risk:.4f} (高于训练集{pct*100:.1f}%的患者)")
    print("-" * 70)
    print("解释 (Top特征贡献):")
    print("  delta > 0: 改为参考值会提高生存概率（该特征当前不利）")
    print("  delta < 0: 改为参考值会降低生存概率（该特征当前有利）")
    print("-" * 70)
    print(expl.to_string(index=False))
    print("=" * 70)
    print(f"[OK] 已保存: {json_path}")
    print(f"[OK] 已保存: {expl_path}")


def _generate_interpretation(p2: float, pct: float) -> str:
    """生成自然语言解释"""
    if p2 >= 0.7:
        risk_level = "较低"
        outlook = "预后较好"
    elif p2 >= 0.4:
        risk_level = "中等"
        outlook = "预后一般"
    else:
        risk_level = "较高"
        outlook = "预后较差"
    
    return f"该患者2年生存概率为{p2*100:.1f}%，风险{risk_level}，{outlook}。" \
           f"其风险评分高于训练队列中{pct*100:.1f}%的患者。"


# =============================================================================
# 命令行解析
# =============================================================================
def build_parser():
    p = argparse.ArgumentParser("surv_2y_system_v2.py - 修复版2年生存预测系统")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    tr = sub.add_parser("train", help="训练并导出模型")
    tr.add_argument("--clinical_csv", required=True)
    tr.add_argument("--radiomics_csv", required=True)
    tr.add_argument("--split_json", required=True)
    tr.add_argument("--out_dir", required=True)
    tr.add_argument("--seed", type=int, default=91)

    tr.add_argument("--id_col", default="PatientID")
    tr.add_argument("--time_col", default="Survival.time")
    tr.add_argument("--event_col", default="deadstatus.event")

    tr.add_argument("--age_col", default="age")
    tr.add_argument("--gender_col", default="gender")
    tr.add_argument("--histology_col", default="Histology")
    tr.add_argument("--t_stage_col", default="clinical.T.Stage")
    tr.add_argument("--stage_col", default="Overall.Stage")

    tr.add_argument("--use_age", type=int, default=1)
    tr.add_argument("--use_gender", type=int, default=1)
    tr.add_argument("--use_histology", type=int, default=1)
    tr.add_argument("--use_t_stage", type=int, default=1)
    tr.add_argument("--use_stage", type=int, default=1)

    tr.add_argument("--selected_radiomics_txt", required=True)
    tr.add_argument("--gbsa_params_json", default=None)

    tr.add_argument("--scale", choices=["minmax", "zscore", "none"], default="minmax")
    tr.add_argument("--age_impute", choices=["mean", "median"], default="mean")

    tr.add_argument("--horizon", type=float, default=730.0)
    tr.add_argument("--calibrate", choices=["none", "platt", "isotonic"], default="platt",
                    help="校准方法: platt(推荐), isotonic, none")
    tr.add_argument("--calib_bins", type=int, default=10)
    
    # 新增：正则化选项
    tr.add_argument("--regularize", choices=["none", "light", "moderate", "strong"], default="moderate",
                    help="正则化级别(修复过拟合): none, light, moderate(推荐), strong")

    # eval
    ev = sub.add_parser("eval", help="评估模型")
    ev.add_argument("--model_joblib", required=True)
    ev.add_argument("--clinical_csv", required=True)
    ev.add_argument("--radiomics_csv", required=True)
    ev.add_argument("--split_json", required=True)
    ev.add_argument("--out_dir", required=True)

    # predict
    pr = sub.add_parser("predict", help="单患者预测")
    pr.add_argument("--model_joblib", required=True)
    pr.add_argument("--patient_id", default=None)
    pr.add_argument("--patient_clinical_json", default=None)
    pr.add_argument("--patient_clinical_csv", default=None)
    pr.add_argument("--patient_radiomics_csv", required=True)
    pr.add_argument("--out_dir", required=True)
    pr.add_argument("--explain_topk", type=int, default=10)

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "predict":
        cmd_predict(args)
    else:
        raise RuntimeError("未知命令")


if __name__ == "__main__":
    main()