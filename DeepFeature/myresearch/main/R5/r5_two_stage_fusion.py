#!/usr/bin/env python3
"""
两阶段融合模型：
Stage 1: Bagged Ridge on Radiomics → radiomics_risk (1维)
Stage 2: CoxPH on (radiomics_risk + Stage_dummies + Age) → final_risk

这样做的好处：
1. Radiomics 1130维 → 1维 risk score
2. Stage 作为强先验加入，参数少不易过拟合
3. 最终模型非常稳定
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

warnings.filterwarnings("ignore")


def make_structured_y(time_arr, event_arr):
    y = np.zeros(len(time_arr), dtype=[("event", bool), ("time", float)])
    y["event"] = event_arr.astype(bool)
    y["time"] = time_arr.astype(float)
    return y


def compute_tau(y_train, y_val, q=0.90, eps=1e-6):
    event_times = y_train["time"][y_train["event"]]
    if len(event_times) == 0:
        return float(np.max(y_val["time"]) - eps)
    tau_train = float(np.quantile(event_times, q))
    tau_train = min(tau_train, float(np.max(event_times)) - eps)
    tau_val_cap = float(np.max(y_val["time"]) - eps)
    tau = min(tau_train, tau_val_cap)
    if not np.isfinite(tau) or tau <= 0:
        tau = float(np.max(y_val["time"]) - eps)
    return tau


def encode_stage(stage_series):
    """
    将 Overall.Stage 编码为 dummy 变量
    Stage I → [1,0,0]
    Stage II → [0,1,0]
    Stage III → [0,0,1]
    Stage IV 或其他 → [0,0,0] (作为参考类)
    """
    stage_str = stage_series.astype(str).str.upper().str.strip()
    
    # 创建 dummy 变量
    stage_1 = stage_str.str.contains("I(?!I|V)", regex=True).astype(float)  # Stage I but not II, III, IV
    stage_2 = stage_str.str.contains("II(?!I)", regex=True).astype(float)   # Stage II but not III
    stage_3 = (stage_str.str.contains("III", regex=True) | stage_str.str.contains("IIIA|IIIB", regex=True)).astype(float)
    
    # 更简单的方式：直接匹配
    stage_1 = stage_str.apply(lambda x: 1.0 if x in ['I', 'IA', 'IB', '1', '1A', '1B'] else 0.0)
    stage_2 = stage_str.apply(lambda x: 1.0 if x in ['II', 'IIA', 'IIB', '2', '2A', '2B'] else 0.0)
    stage_3 = stage_str.apply(lambda x: 1.0 if x in ['III', 'IIIA', 'IIIB', '3', '3A', '3B'] else 0.0)
    
    return np.column_stack([stage_1.values, stage_2.values, stage_3.values])


def train_bagged_ridge_stage1(X_tr, y_tr, X_va, 
                               l1_ratio=0.0001, n_bags=32, 
                               n_alphas=80, alpha_min_ratio=1e-3,
                               inner_folds=3, seed=42):
    """
    Stage 1: 训练 Bagged Ridge，返回 train 和 val 的 radiomics_risk
    """
    rng = np.random.default_rng(seed)
    
    # 标准化
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)
    
    # 内层 CV 选择 alpha
    base = CoxnetSurvivalAnalysis(
        l1_ratio=l1_ratio,
        n_alphas=n_alphas,
        alpha_min_ratio=alpha_min_ratio,
        max_iter=200000,
        tol=1e-7,
        fit_baseline_model=False,
    )
    base.fit(X_tr_scaled, y_tr)
    alphas = base.alphas_
    
    inner = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    score_mat = np.full((inner_folds, len(alphas)), np.nan)
    
    for j, (itr, iva) in enumerate(inner.split(X_tr_scaled)):
        Xi_tr, Xi_va = X_tr_scaled[itr], X_tr_scaled[iva]
        yi_tr, yi_va = y_tr[itr], y_tr[iva]
        
        mdl = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            alphas=alphas,
            max_iter=200000,
            tol=1e-7,
            fit_baseline_model=False,
        )
        mdl.fit(Xi_tr, yi_tr)
        
        for k, a in enumerate(alphas):
            try:
                r = mdl.predict(Xi_va, alpha=a)
                score_mat[j, k] = concordance_index_censored(yi_va["event"], yi_va["time"], r)[0]
            except:
                pass
    
    mean_scores = np.nanmean(score_mat, axis=0)
    valid = np.isfinite(mean_scores)
    if valid.sum() == 0:
        idx_sel = len(alphas) // 2
    else:
        idx_sel = np.where(valid)[0][np.nanargmax(mean_scores[valid])]
    alpha_sel = alphas[idx_sel]
    
    # Bagging 预测
    risk_tr = np.zeros(len(X_tr_scaled))
    risk_va = np.zeros(len(X_va_scaled))
    
    for b in range(n_bags):
        bs_idx = rng.choice(len(X_tr_scaled), size=len(X_tr_scaled), replace=True)
        mdl_b = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            alphas=[alpha_sel],
            max_iter=200000,
            tol=1e-7,
            fit_baseline_model=False,
        )
        mdl_b.fit(X_tr_scaled[bs_idx], y_tr[bs_idx])
        risk_tr += mdl_b.predict(X_tr_scaled, alpha=alpha_sel)
        risk_va += mdl_b.predict(X_va_scaled, alpha=alpha_sel)
    
    risk_tr /= n_bags
    risk_va /= n_bags
    
    return risk_tr, risk_va


def train_stage2_coxph(X_tr_clinical, y_tr, X_va_clinical):
    """
    Stage 2: 简单 CoxPH on low-dimensional clinical features
    X_clinical 包含: [radiomics_risk, stage_1, stage_2, stage_3, age]
    """
    # 标准化 (对 radiomics_risk 和 age)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_clinical)
    X_va_scaled = scaler.transform(X_va_clinical)
    
    # 简单 CoxPH (无正则化，因为只有5个变量)
    try:
        cph = CoxPHSurvivalAnalysis(alpha=0.1)  # 轻微正则化
        cph.fit(X_tr_scaled, y_tr)
        risk_va = cph.predict(X_va_scaled)
    except Exception as e:
        # 如果 CoxPH 失败，用 Coxnet 替代
        cph = CoxnetSurvivalAnalysis(l1_ratio=0.01, n_alphas=50, fit_baseline_model=False)
        cph.fit(X_tr_scaled, y_tr)
        risk_va = cph.predict(X_va_scaled)
    
    return risk_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True, help="Radiomics features CSV")
    ap.add_argument("--labels_csv", required=True, help="Clinical labels CSV")
    ap.add_argument("--splits_json", required=True, help="Splits JSON")
    ap.add_argument("--outdir", required=True, help="Output directory")
    
    ap.add_argument("--id_col", default="PatientID")
    ap.add_argument("--time_col", default="Survival.time")
    ap.add_argument("--event_col", default="deadstatus.event")
    ap.add_argument("--stage_col", default="Overall.Stage")
    ap.add_argument("--age_col", default="age")
    
    ap.add_argument("--l1_ratio", type=float, default=0.0001)
    ap.add_argument("--n_bags", type=int, default=32)
    ap.add_argument("--n_alphas", type=int, default=80)
    ap.add_argument("--alpha_min_ratio", type=float, default=1e-3)
    ap.add_argument("--inner_folds", type=int, default=3)
    ap.add_argument("--tau_q", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)
    
    ap.add_argument("--use_age", action="store_true", default=True, help="Include age in Stage 2")
    args = ap.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # ========== 加载数据 ==========
    print("=" * 70)
    print("Two-Stage Fusion: Radiomics + Stage + Age")
    print("=" * 70)
    
    # 加载 radiomics features
    feat_df = pd.read_csv(args.features_csv)
    feat_df[args.id_col] = feat_df[args.id_col].astype(str)
    feat_cols = [c for c in feat_df.columns if c != args.id_col]
    print(f"[INFO] Radiomics features: {len(feat_cols)}")
    
    # 加载 clinical labels
    lab_df = pd.read_csv(args.labels_csv)
    lab_df[args.id_col] = lab_df[args.id_col].astype(str)
    
    # 检查列名
    print(f"[INFO] Available columns in labels: {list(lab_df.columns)}")
    
    if args.stage_col not in lab_df.columns:
        raise ValueError(f"Stage column '{args.stage_col}' not found. Available: {list(lab_df.columns)}")
    
    # 对齐数据
    common_ids = set(feat_df[args.id_col]) & set(lab_df[args.id_col])
    feat_df = feat_df[feat_df[args.id_col].isin(common_ids)].copy()
    lab_df = lab_df[lab_df[args.id_col].isin(common_ids)].copy()
    
    # 按 ID 排序对齐
    feat_df = feat_df.sort_values(args.id_col).reset_index(drop=True)
    lab_df = lab_df.sort_values(args.id_col).reset_index(drop=True)
    
    assert list(feat_df[args.id_col]) == list(lab_df[args.id_col]), "ID mismatch!"
    
    # 提取数据
    pids = feat_df[args.id_col].values
    X_radiomics = feat_df[feat_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    
    time_arr = pd.to_numeric(lab_df[args.time_col], errors="coerce").values
    event_arr = pd.to_numeric(lab_df[args.event_col], errors="coerce").fillna(0).astype(int).values
    y_all = make_structured_y(time_arr, event_arr)
    
    # Stage encoding
    stage_dummies = encode_stage(lab_df[args.stage_col])
    print(f"[INFO] Stage distribution:")
    print(f"  Stage I:   {stage_dummies[:, 0].sum():.0f}")
    print(f"  Stage II:  {stage_dummies[:, 1].sum():.0f}")
    print(f"  Stage III: {stage_dummies[:, 2].sum():.0f}")
    print(f"  Stage IV+: {(1 - stage_dummies.sum(axis=1) > 0).sum():.0f}")
    
    # Age
    if args.use_age and args.age_col in lab_df.columns:
        age = pd.to_numeric(lab_df[args.age_col], errors="coerce").fillna(lab_df[args.age_col].median()).values
        print(f"[INFO] Age: mean={age.mean():.1f}, std={age.std():.1f}")
    else:
        age = None
        print("[INFO] Age not used")
    
    n_samples = len(pids)
    print(f"[INFO] Total samples: {n_samples}")
    
    # ========== 加载 splits ==========
    with open(args.splits_json, "r") as f:
        splits = json.load(f)
    fold_keys = sorted(splits.keys())
    print(f"[INFO] Number of folds: {len(fold_keys)}")
    
    # ========== 运行实验 ==========
    rows = []
    
    for fk in fold_keys:
        tr_set = set(map(str, splits[fk]["train"]))
        va_set = set(map(str, splits[fk]["val"]))
        
        tr_idx = np.array([i for i, pid in enumerate(pids) if pid in tr_set])
        va_idx = np.array([i for i, pid in enumerate(pids) if pid in va_set])
        
        if len(tr_idx) == 0 or len(va_idx) == 0:
            print(f"[WARN] {fk}: empty train/val, skip")
            continue
        
        # 准备数据
        X_tr_rad = X_radiomics[tr_idx].copy()
        X_va_rad = X_radiomics[va_idx].copy()
        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]
        
        # 处理 NaN
        X_tr_rad = np.where(np.isfinite(X_tr_rad), X_tr_rad, np.nan)
        X_va_rad = np.where(np.isfinite(X_va_rad), X_va_rad, np.nan)
        med = np.nanmedian(X_tr_rad, axis=0)
        ok = np.isfinite(med)
        X_tr_rad = X_tr_rad[:, ok]
        X_va_rad = X_va_rad[:, ok]
        med = med[ok]
        X_tr_rad = np.where(np.isfinite(X_tr_rad), X_tr_rad, med)
        X_va_rad = np.where(np.isfinite(X_va_rad), X_va_rad, med)
        
        # ========== Stage 1: Radiomics Bagged Ridge ==========
        risk_tr_rad, risk_va_rad = train_bagged_ridge_stage1(
            X_tr_rad, y_tr, X_va_rad,
            l1_ratio=args.l1_ratio,
            n_bags=args.n_bags,
            n_alphas=args.n_alphas,
            alpha_min_ratio=args.alpha_min_ratio,
            inner_folds=args.inner_folds,
            seed=args.seed
        )
        
        # 计算 Stage 1 的 C-index (仅 radiomics)
        tau = compute_tau(y_tr, y_va, q=args.tau_q)
        har_rad = concordance_index_censored(y_va["event"], y_va["time"], risk_va_rad)[0]
        uno_rad = concordance_index_ipcw(y_tr, y_va, risk_va_rad, tau=tau)[0]
        
        # ========== Stage 2: Low-dim fusion ==========
        # 构建 Stage 2 的特征
        stage_tr = stage_dummies[tr_idx]
        stage_va = stage_dummies[va_idx]
        
        if age is not None:
            age_tr = age[tr_idx].reshape(-1, 1)
            age_va = age[va_idx].reshape(-1, 1)
            X_tr_clin = np.column_stack([risk_tr_rad.reshape(-1, 1), stage_tr, age_tr])
            X_va_clin = np.column_stack([risk_va_rad.reshape(-1, 1), stage_va, age_va])
            clin_names = ["radiomics_risk", "stage_I", "stage_II", "stage_III", "age"]
        else:
            X_tr_clin = np.column_stack([risk_tr_rad.reshape(-1, 1), stage_tr])
            X_va_clin = np.column_stack([risk_va_rad.reshape(-1, 1), stage_va])
            clin_names = ["radiomics_risk", "stage_I", "stage_II", "stage_III"]
        
        # 训练 Stage 2
        risk_va_final = train_stage2_coxph(X_tr_clin, y_tr, X_va_clin)
        
        # 计算 Stage 2 的 C-index (融合后)
        har_fusion = concordance_index_censored(y_va["event"], y_va["time"], risk_va_final)[0]
        uno_fusion = concordance_index_ipcw(y_tr, y_va, risk_va_final, tau=tau)[0]
        
        # 记录结果
        row = {
            "fold": fk,
            "n_train": len(tr_idx),
            "n_val": len(va_idx),
            "harrell_radiomics": har_rad,
            "uno_radiomics": uno_rad,
            "harrell_fusion": har_fusion,
            "uno_fusion": uno_fusion,
            "improvement": har_fusion - har_rad,
        }
        rows.append(row)
        
        print(f"[{fk}] Radiomics: {har_rad:.4f} | Fusion: {har_fusion:.4f} | Δ={har_fusion-har_rad:+.4f}")
    
    # ========== 汇总结果 ==========
    res = pd.DataFrame(rows)
    res.to_csv(outdir / "summary.csv", index=False)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\n[Radiomics Only (Stage 1)]")
    print(f"  Mean Harrell: {res['harrell_radiomics'].mean():.4f} ± {res['harrell_radiomics'].std():.4f}")
    print(f"  Mean Uno:     {res['uno_radiomics'].mean():.4f} ± {res['uno_radiomics'].std():.4f}")
    
    print(f"\n[Fusion: Radiomics + Stage + Age (Stage 2)]")
    print(f"  Mean Harrell: {res['harrell_fusion'].mean():.4f} ± {res['harrell_fusion'].std():.4f}")
    print(f"  Mean Uno:     {res['uno_fusion'].mean():.4f} ± {res['uno_fusion'].std():.4f}")
    
    print(f"\n[Improvement]")
    print(f"  Mean Δ Harrell: {res['improvement'].mean():+.4f}")
    print(f"  Positive folds: {(res['improvement'] > 0).sum()} / {len(res)}")
    
    # 配对 t 检验
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(res['harrell_fusion'], res['harrell_radiomics'])
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    print(f"\n[DONE] Saved to {outdir}")


if __name__ == "__main__":
    main()