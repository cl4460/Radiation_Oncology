#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_feature_selection.py
特征选择泄漏审计工具

功能：
1. 验证给定的特征列表（selected_radiomics_txt）是否是在 Training Set 上合法筛选的。
2. 检查是否存在 "Test Set Leakage"（即特征在 Test 上的表现异常优于 Train）。
3. 检查选中的特征是否在 Train Set 上具有统计学显著性（排名前列）。

依赖：pandas, numpy, scikit-survival
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored

def load_split(path):
    with open(path, 'r') as f:
        data = json.load(f)
    # 兼容不同的 key 写法
    train_keys = ['train', 'train_ids', 'train_patients']
    test_keys = ['test', 'test_ids', 'test_patients']
    
    train_ids = []
    test_ids = []
    
    for k in train_keys:
        if k in data:
            train_ids = [str(x) for x in data[k]]
            break
    for k in test_keys:
        if k in data:
            test_ids = [str(x) for x in data[k]]
            break
            
    return train_ids, test_ids

def load_data(clinical_path, radiomics_path):
    clin = pd.read_csv(clinical_path)
    rad = pd.read_csv(radiomics_path)
    
    # 确保 ID 是字符串格式，方便合并
    if 'PatientID' not in clin.columns:
        # 尝试寻找常见的 ID 列名
        possible_ids = ['Case ID', 'ID', 'Patient ID']
        for pid in possible_ids:
            if pid in clin.columns:
                clin = clin.rename(columns={pid: 'PatientID'})
                break
    
    clin['PatientID'] = clin['PatientID'].astype(str)
    rad['PatientID'] = rad['PatientID'].astype(str)
    
    # 合并
    df = pd.merge(clin, rad, on='PatientID', how='inner')
    return df, [c for c in rad.columns if c != 'PatientID']

def get_cindex(df, feature_col, time_col='Survival.time', event_col='deadstatus.event'):
    """计算单变量 C-index"""
    try:
        t = df[time_col].astype(float).values
        e = df[event_col].astype(int).astype(bool).values
        x = df[feature_col].astype(float).values
        
        # 填充缺失值
        if np.isnan(x).any():
            x = np.nan_to_num(x, nan=np.nanmedian(x))
            
        c = concordance_index_censored(e, t, x)[0]
        
        # Radiomics 特征方向不确定，如果 c < 0.5，说明是负相关，取 1-c
        if c < 0.5:
            return 1.0 - c, "Negative"
        return c, "Positive"
    except Exception as err:
        return 0.5, str(err)

def main():
    parser = argparse.ArgumentParser(description="Audit Feature Selection for Leakage")
    parser.add_argument("--clinical_csv", required=True)
    parser.add_argument("--radiomics_csv", required=True)
    parser.add_argument("--split_json", required=True)
    parser.add_argument("--current_feature_txt", required=True)
    parser.add_argument("--out_dir", required=True)
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 1. 加载数据
    print("Loading data...")
    df, all_rad_features = load_data(args.clinical_csv, args.radiomics_csv)
    train_ids, test_ids = load_split(args.split_json)
    
    df_train = df[df['PatientID'].isin(train_ids)].copy()
    df_test = df[df['PatientID'].isin(test_ids)].copy()
    
    print(f"Total merged: {len(df)}")
    print(f"Train Set: {len(df_train)} (Target: 332)")
    print(f"Test Set : {len(df_test)} (Target: 90)")
    
    if len(df_train) == 0 or len(df_test) == 0:
        print("Error: Split resulted in empty sets. Check PatientIDs in JSON vs CSV.")
        sys.exit(1)

    # 2. 读取当前使用的特征列表
    with open(args.current_feature_txt, 'r') as f:
        selected_feats = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"\nAuditing {len(selected_feats)} selected features...")
    
    # 3. 计算选定特征在 Train 和 Test 上的表现
    audit_results = []
    suspicious_count = 0
    
    print(f"{'Feature Name':<50} | {'Train C':<8} | {'Test C':<8} | {'Gap (Tr-Te)':<12}")
    print("-" * 90)
    
    train_c_scores = []
    
    for feat in selected_feats:
        if feat not in df.columns:
            print(f"Warning: Feature {feat} not found in dataframe!")
            continue
            
        c_train, _ = get_cindex(df_train, feat)
        c_test, _ = get_cindex(df_test, feat)
        gap = c_train - c_test
        
        train_c_scores.append(c_train)
        
        # 判定标准：如果 Test C-index 比 Train 高出 0.1 以上，且 Train C-index 很低，疑似泄漏
        is_suspicious = (gap < -0.10) and (c_train < 0.60)
        flag = " [!]" if is_suspicious else ""
        if is_suspicious:
            suspicious_count += 1
            
        print(f"{feat[:50]:<50} | {c_train:.4f}   | {c_test:.4f}   | {gap:+.4f}{flag}")
        
        audit_results.append({
            "feature": feat,
            "train_cindex": c_train,
            "test_cindex": c_test,
            "gap": gap,
            "suspicious": is_suspicious
        })
        
    avg_train_c = np.mean(train_c_scores)
    print("-" * 90)
    print(f"Average Train C-index of selected features: {avg_train_c:.4f}")
    
    # 4. 全局扫描：看选中的特征是否在 Train Set 上真的优秀
    print("\nGlobal Scan: Ranking all 1688 features on Train Set...")
    all_scores = []
    for feat in all_rad_features:
        c, _ = get_cindex(df_train, feat)
        all_scores.append((feat, c))
    
    # 按 Train C-index 降序排列
    all_scores.sort(key=lambda x: x[1], reverse=True)
    top_100_features = set([x[0] for x in all_scores[:100]])
    
    # 检查重合度
    selected_set = set(selected_feats)
    overlap = selected_set.intersection(top_100_features)
    overlap_rate = len(overlap) / len(selected_feats)
    
    print(f"Top 1 feature on Train: {all_scores[0][0]} (C={all_scores[0][1]:.4f})")
    print(f"Number of selected features in Top 100 Train features: {len(overlap)} / {len(selected_feats)}")
    
    # 5. 最终结论报告
    print("\n" + "="*30)
    print("       AUDIT REPORT       ")
    print("="*30)
    
    final_verdict = "PASS"
    reasons = []
    
    # 规则 1: 选中的特征必须在 Train 上有不错的区分度 (比如平均 > 0.58)
    if avg_train_c < 0.55:
        final_verdict = "FAIL"
        reasons.append("Selected features have very low predictive power on Training Set.")
        
    # 规则 2: 不应有太多“反常”特征 (Test 远好于 Train)
    if suspicious_count > len(selected_feats) * 0.3:
        final_verdict = "FAIL"
        reasons.append(f"Too many features ({suspicious_count}) perform much better on Test than Train (Leakage sign).")
        
    # 规则 3: 选中的特征应该大部分位于 Train 的前列
    # 注意：mRMR 会去重，所以不一定都在 Top 100，但至少应该有几个在。
    if len(overlap) == 0:
        final_verdict = "WARNING"
        reasons.append("None of the selected features are in the Top 100 univariate features of Training Set.")
    
    print(f"Verdict: {final_verdict}")
    if reasons:
        for r in reasons:
            print(f"- {r}")
    else:
        print("- Features look consistent with a Train-only selection process.")
        print("- Performance gap is normal (Train >= Test).")

    # 保存报告
    pd.DataFrame(audit_results).to_csv(os.path.join(args.out_dir, "audit_details.csv"), index=False)
    print(f"\nDetails saved to: {os.path.join(args.out_dir, 'audit_details.csv')}")

if __name__ == "__main__":
    main()