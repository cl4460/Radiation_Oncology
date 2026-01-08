#!/usr/bin/env python3
"""
特征选择数据泄漏验证脚本

目的：证明 gbsa91_selected_radiomics.txt 是严格在训练集上选择的

验证步骤：
1. 加载 split_json，获取 train_ids (n=332) 和 test_ids (n=90)
2. 仅使用 train_ids 的数据运行 mRMR 特征选择
3. 输出选定的 16 个特征
4. 计算 SHA256 哈希
5. 与现有的 gbsa91_selected_radiomics.txt 对比

如果完全一致，则证明特征选择无数据泄漏。
"""

import argparse
import hashlib
import json
import sys
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


def compute_sha256(feature_list: List[str]) -> str:
    """计算特征列表的 SHA256 哈希"""
    content = "\n".join(feature_list) + "\n"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def mrmr_select(
    X: pd.DataFrame,
    time: np.ndarray,
    event: np.ndarray,
    k: int,
    seed: int = 0,
    relevance: str = "event+time",
    redundancy: str = "corr",
) -> List[str]:
    """
    mRMR 特征选择（与 reproduce_report_baseline_step_by_step.py 完全一致）
    
    relevance:
      - "event": MI(feature, event)
      - "time":  MI(feature, log1p(time))
      - "event+time": sum of both
    redundancy:
      - "corr": mean absolute Pearson corr with selected features
    """
    rng = np.random.default_rng(seed)
    cols = list(X.columns)
    Xv = X.to_numpy(dtype=float)

    # Compute relevance scores
    rel = np.zeros(Xv.shape[1], dtype=float)
    if relevance in ("event", "event+time"):
        rel += mutual_info_classif(Xv, event.astype(int), random_state=seed, discrete_features=False)
    if relevance in ("time", "event+time"):
        rel += mutual_info_regression(Xv, np.log1p(time.astype(float)), random_state=seed)

    # Start with best relevance
    selected = []
    remaining = list(range(len(cols)))
    
    # First feature: max relevance
    idx = int(np.argmax(rel))
    selected.append(idx)
    remaining.remove(idx)

    # Iteratively select features
    for _ in range(k - 1):
        if not remaining:
            break
        
        scores = []
        for i in remaining:
            # Redundancy: mean absolute correlation with already selected
            if redundancy == "corr":
                red = 0.0
                for j in selected:
                    red += abs(np.corrcoef(Xv[:, i], Xv[:, j])[0, 1])
                red /= len(selected)
            else:
                red = 0.0
            
            # mRMR score: relevance - redundancy
            scores.append(rel[i] - red)
        
        best_idx = remaining[int(np.argmax(scores))]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [cols[i] for i in selected]


def main():
    parser = argparse.ArgumentParser(
        description="验证特征选择无数据泄漏",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  python verify_feature_selection_no_leakage.py \\
    --clinical_csv /path/to/Lung1.clinical.csv \\
    --radiomics_csv /path/to/Tb_features_1688.csv \\
    --split_json /path/to/split.json \\
    --existing_features /path/to/gbsa91_selected_radiomics.txt \\
    --seed 91 \\
    --k 16
        """
    )
    
    parser.add_argument("--clinical_csv", required=True, help="临床数据CSV")
    parser.add_argument("--radiomics_csv", required=True, help="影像组学特征CSV")
    parser.add_argument("--split_json", required=True, help="训练/测试集划分JSON")
    parser.add_argument("--existing_features", required=True, help="现有的特征列表文件")
    parser.add_argument("--seed", type=int, default=91, help="随机种子")
    parser.add_argument("--k", type=int, default=16, help="选择的特征数量")
    parser.add_argument("--time_col", default="Survival.time", help="生存时间列名")
    parser.add_argument("--event_col", default="deadstatus.event", help="事件列名")
    parser.add_argument("--out_txt", default=None, help="输出验证后的特征列表（可选）")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("特征选择数据泄漏验证")
    print("=" * 80)
    print()
    
    # 1. Load split
    print("[1/6] 加载数据集划分...")
    with open(args.split_json, 'r') as f:
        split = json.load(f)
    
    train_ids = split.get("train_ids", split.get("train", []))
    test_ids = split.get("test_ids", split.get("test", []))
    
    print(f"  训练集样本数: {len(train_ids)}")
    print(f"  测试集样本数: {len(test_ids)}")
    print(f"  总样本数: {len(train_ids) + len(test_ids)}")
    print(f"  Split seed: {split.get('seed', 'N/A')}")
    print()
    
    # 2. Load data
    print("[2/6] 加载临床和影像组学数据...")
    clin = pd.read_csv(args.clinical_csv)
    rad = pd.read_csv(args.radiomics_csv)
    
    # Merge
    if "PatientID" in clin.columns and "PatientID" in rad.columns:
        df = pd.merge(clin, rad, on="PatientID", how="inner")
    else:
        raise ValueError("Both CSV files must have 'PatientID' column")
    
    df = df.set_index("PatientID")
    print(f"  合并后总样本数: {len(df)}")
    print()
    
    # 3. Extract train subset ONLY
    print("[3/6] 提取训练集数据（仅用于特征选择）...")
    train_ids_exist = [pid for pid in train_ids if pid in df.index]
    test_ids_exist = [pid for pid in test_ids if pid in df.index]
    
    print(f"  训练集有效样本: {len(train_ids_exist)}")
    print(f"  测试集有效样本: {len(test_ids_exist)}")
    
    if len(train_ids_exist) != len(train_ids):
        print(f"  ⚠️  警告: {len(train_ids) - len(train_ids_exist)} 个训练集ID未找到")
    
    df_train = df.loc[train_ids_exist].copy()
    
    # Get radiomics features (exclude clinical columns)
    rad_cols = [c for c in rad.columns if c != "PatientID"]
    X_rad_train = df_train[rad_cols].copy()
    
    # Get survival data
    y_time_train = df_train[args.time_col].to_numpy(dtype=float)
    y_event_train = df_train[args.event_col].to_numpy(dtype=int)
    
    print(f"  影像组学特征数: {X_rad_train.shape[1]}")
    print(f"  ✅ 关键验证点: 特征选择仅使用 train_ids 的 {len(train_ids_exist)} 个样本")
    print(f"  ✅ 测试集的 {len(test_ids_exist)} 个样本完全未使用")
    print()
    
    # 4. Run mRMR on TRAIN ONLY
    print(f"[4/6] 运行 mRMR 特征选择（k={args.k}, seed={args.seed}）...")
    print(f"  输入: X_rad_train.shape = {X_rad_train.shape}")
    print(f"  方法: mRMR (relevance=event+time, redundancy=corr)")
    
    selected_features = mrmr_select(
        X=X_rad_train,
        time=y_time_train,
        event=y_event_train,
        k=args.k,
        seed=args.seed,
        relevance="event+time",
        redundancy="corr",
    )
    
    print(f"  ✅ 选定 {len(selected_features)} 个特征")
    print()
    
    # 5. Compute hash
    print("[5/6] 计算特征列表哈希...")
    new_hash = compute_sha256(selected_features)
    print(f"  SHA256: {new_hash}")
    print()
    
    # 6. Compare with existing
    print("[6/6] 与现有特征列表对比...")
    with open(args.existing_features, 'r') as f:
        existing_features = [line.strip() for line in f if line.strip()]
    
    existing_hash = compute_sha256(existing_features)
    print(f"  现有特征文件: {args.existing_features}")
    print(f"  现有特征数量: {len(existing_features)}")
    print(f"  现有特征哈希: {existing_hash}")
    print()
    
    # Detailed comparison
    print("=" * 80)
    print("验证结果")
    print("=" * 80)
    
    if new_hash == existing_hash:
        print("✅ **验证通过**: 特征列表完全一致")
        print()
        print("结论:")
        print(f"  - gbsa91_selected_radiomics.txt 中的 {args.k} 个特征")
        print(f"  - 确实是在训练集（n={len(train_ids_exist)}）上通过 mRMR 选择的")
        print(f"  - 测试集（n={len(test_ids_exist)}）完全未参与特征选择")
        print(f"  - 无数据泄漏")
        print()
        print("SHA256 哈希匹配证明特征列表未被手工修改。")
        exit_code = 0
    else:
        print("❌ **验证失败**: 特征列表不一致")
        print()
        print("差异分析:")
        
        # Find differences
        new_set = set(selected_features)
        existing_set = set(existing_features)
        
        only_in_new = new_set - existing_set
        only_in_existing = existing_set - new_set
        
        if only_in_new:
            print(f"  仅在新生成中: {len(only_in_new)} 个特征")
            for feat in list(only_in_new)[:5]:
                print(f"    - {feat}")
            if len(only_in_new) > 5:
                print(f"    ... 还有 {len(only_in_new) - 5} 个")
        
        if only_in_existing:
            print(f"  仅在现有文件中: {len(only_in_existing)} 个特征")
            for feat in list(only_in_existing)[:5]:
                print(f"    - {feat}")
            if len(only_in_existing) > 5:
                print(f"    ... 还有 {len(only_in_existing) - 5} 个")
        
        # Check order
        if new_set == existing_set:
            print()
            print("  注意: 特征集合相同，但顺序不同")
            print("  这可能是由于:")
            print("    - 不同的 numpy/sklearn 版本")
            print("    - 浮点数精度差异")
            print("    - 随机数生成器实现差异")
        
        exit_code = 1
    
    print()
    print("=" * 80)
    print("详细特征列表")
    print("=" * 80)
    print()
    print("新生成的特征列表:")
    for i, feat in enumerate(selected_features, 1):
        match = "✓" if feat in existing_features else "✗"
        print(f"  {i:2d}. [{match}] {feat}")
    
    # Save if requested
    if args.out_txt:
        with open(args.out_txt, 'w') as f:
            for feat in selected_features:
                f.write(feat + '\n')
        print()
        print(f"✅ 新特征列表已保存到: {args.out_txt}")
    
    print()
    print("=" * 80)
    print("验证完成")
    print("=" * 80)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

