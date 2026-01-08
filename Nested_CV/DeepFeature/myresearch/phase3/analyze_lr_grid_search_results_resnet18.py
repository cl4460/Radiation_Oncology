#!/usr/bin/env python3
"""
分析ResNet18学习率网格搜索结果，包括每个fold的详细值和排名
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import torch

def load_all_results():
    """加载所有学习率和fold的结果"""
    base_dir = Path("/home/lichengze/Research/DeepFeature/myresearch/phase3_outputs")
    
    # 学习率列表（按脚本中的顺序）
    lr_values = [
        "9.00e-05", "8.00e-05", "1.00e-04", "2.00e-04", "3.00e-04",
        "4.00e-04", "5.00e-04", "6.00e-04", "6.20e-04", "6.40e-04",
        "6.80e-04", "6.90e-04", "7.00e-04", "7.20e-04", "7.40e-04",
        "7.60e-04", "8.00e-04", "9.00e-04", "1.00e-03", "2.00e-03",
        "3.00e-03", "4.00e-03", "6.00e-03", "9.00e-03"
    ]
    
    results = []
    missing_folds = {}
    
    for lr in lr_values:
        exp_name = f"resnet18_lr{lr}"
        exp_dir = base_dir / exp_name
        
        fold_results = {}
        missing = []
        
        for fold in range(5):
            fold_dir = exp_dir / f"fold_{fold}"
            ckpt_file = fold_dir / "best.pt"
            
            if ckpt_file.exists():
                try:
                    # PyTorch 2.6+ requires weights_only=False for files with numpy objects
                    ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
                    fold_results[fold] = {
                        'uno_integral': ckpt.get('uno_integral', np.nan),
                        'uno_24m': ckpt.get('uno_24m', np.nan),
                        'harrell': ckpt.get('harrell', np.nan),
                        'epoch': ckpt.get('epoch', -1)
                    }
                except Exception as e:
                    print(f"Warning: Failed to load {ckpt_file}: {e}")
                    missing.append(fold)
            else:
                missing.append(fold)
        
        if missing:
            missing_folds[lr] = missing
        
        # 如果至少有一个fold的结果，就记录
        if fold_results:
            result_entry = {
                'learning_rate': lr,
                'lr_numeric': float(lr.replace('e-', 'e-').replace('e+', 'e+'))
            }
            
            # 添加每个fold的结果
            for fold in range(5):
                if fold in fold_results:
                    result_entry[f'fold_{fold}_uno'] = fold_results[fold]['uno_integral']
                    result_entry[f'fold_{fold}_harrell'] = fold_results[fold]['harrell']
                    result_entry[f'fold_{fold}_uno_24m'] = fold_results[fold]['uno_24m']
                else:
                    result_entry[f'fold_{fold}_uno'] = np.nan
                    result_entry[f'fold_{fold}_harrell'] = np.nan
                    result_entry[f'fold_{fold}_uno_24m'] = np.nan
            
            # 计算统计量（只使用有效的fold）
            valid_uno = [fold_results[f]['uno_integral'] for f in range(5) if f in fold_results and not np.isnan(fold_results[f]['uno_integral'])]
            valid_harrell = [fold_results[f]['harrell'] for f in range(5) if f in fold_results and not np.isnan(fold_results[f]['harrell'])]
            
            if valid_uno:
                result_entry['mean_uno'] = np.mean(valid_uno)
                result_entry['std_uno'] = np.std(valid_uno)
                result_entry['n_folds'] = len(valid_uno)
            else:
                result_entry['mean_uno'] = np.nan
                result_entry['std_uno'] = np.nan
                result_entry['n_folds'] = 0
            
            if valid_harrell:
                result_entry['mean_harrell'] = np.mean(valid_harrell)
                result_entry['std_harrell'] = np.std(valid_harrell)
            else:
                result_entry['mean_harrell'] = np.nan
                result_entry['std_harrell'] = np.nan
            
            results.append(result_entry)
    
    return pd.DataFrame(results), missing_folds

def format_lr_scientific(lr_str):
    """将学习率字符串转换为科学计数法格式，如 2.00E-04"""
    try:
        lr_float = float(lr_str.replace('e-', 'e-').replace('e+', 'e+'))
        # 格式化为科学计数法，保留两位小数
        return f"{lr_float:.2E}"
    except:
        return lr_str

def generate_detailed_report(df, missing_folds):
    """生成详细的分析报告"""
    report = []
    report.append("=" * 100)
    report.append("ResNet18 学习率网格搜索结果详细分析报告")
    report.append("=" * 100)
    report.append("")
    
    # 检查缺失的fold
    if missing_folds:
        report.append("⚠️  缺失的Fold检测:")
        for lr, folds in sorted(missing_folds.items()):
            fold_str = ', '.join([f'fold_{f}' for f in sorted(folds)])
            report.append(f"  - LR={lr}: 缺失 {fold_str} ({len(folds)}个fold)")
        report.append("")
    else:
        report.append("✅ 所有学习率都包含完整的5个fold (fold_0 到 fold_4)")
        report.append("")
    
    # 按mean_uno排序
    df_sorted = df.sort_values('mean_uno', ascending=False, na_position='last').reset_index(drop=True)
    df_sorted['rank'] = range(1, len(df_sorted) + 1)
    
    # 生成表格格式的输出（符合图片格式）
    report.append("=" * 100)
    report.append("结果表格 (按Uno C-index均值排序，格式符合CSV要求)")
    report.append("=" * 100)
    report.append("")
    report.append("说明: Result格式为 'mean, std' (三位小数), Each Fold格式为 'fold0, fold1, fold2, fold3, fold4' (三位小数，逗号分隔)")
    report.append("")
    
    # 表头
    report.append(f"{'Result':<25} {'Each Fold':<80} {'LR':<15}")
    report.append("-" * 120)
    
    for _, row in df_sorted.iterrows():
        if pd.notna(row['mean_uno']):
            # Result格式: "mean, std" (都是三位小数)
            result_str = f"{row['mean_uno']:.3f}, {row['std_uno']:.3f}"
            
            # Each Fold格式: "fold0, fold1, fold2, fold3, fold4" (都是三位小数)
            fold_values = []
            for fold in range(5):
                uno_val = row[f'fold_{fold}_uno']
                if pd.notna(uno_val):
                    fold_values.append(f"{uno_val:.3f}")
                else:
                    fold_values.append("")  # 缺失的fold留空
            
            # 过滤掉空值，只保留有效的fold值
            fold_values_filtered = [v for v in fold_values if v]
            fold_str = ", ".join(fold_values_filtered)
            
            # LR格式: 科学计数法
            lr_formatted = format_lr_scientific(row['learning_rate'])
            
            # 输出格式：Result | Each Fold | LR
            report.append(f"{result_str:<25} {fold_str:<80} {lr_formatted:<15}")
        else:
            lr_formatted = format_lr_scientific(row['learning_rate'])
            report.append(f"{'N/A':<25} {'N/A':<80} {lr_formatted:<15}")
    
    report.append("")
    
    # 总体排名（保留原有格式作为参考）
    report.append("=" * 100)
    report.append("总体排名 (按Uno C-index均值排序)")
    report.append("=" * 100)
    report.append("")
    report.append(f"{'排名':<6} {'学习率':<15} {'Uno均值':<12} {'Uno标准差':<12} {'Harrell均值':<12} {'完整fold数':<10}")
    report.append("-" * 100)
    
    for _, row in df_sorted.iterrows():
        if pd.notna(row['mean_uno']):
            report.append(
                f"{row['rank']:<6} {row['learning_rate']:<15} "
                f"{row['mean_uno']:<12.3f} {row['std_uno']:<12.3f} "
                f"{row['mean_harrell']:<12.3f} {row['n_folds']:<10.0f}"
            )
        else:
            report.append(
                f"{row['rank']:<6} {row['learning_rate']:<15} "
                f"{'N/A':<12} {'N/A':<12} {'N/A':<12} {row['n_folds']:<10.0f}"
            )
    
    report.append("")
    
    # 所有学习率详细结果
    report.append("=" * 100)
    report.append("所有学习率详细结果 (每个fold的具体值)")
    report.append("=" * 100)
    report.append("")
    
    for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
        report.append(f"排名 {idx}: LR = {row['learning_rate']}")
        if pd.notna(row['mean_uno']):
            report.append(f"  平均Uno C-index: {row['mean_uno']:.3f} ± {row['std_uno']:.3f}")
            report.append(f"  平均Harrell C-index: {row['mean_harrell']:.3f} ± {row['std_harrell']:.3f}")
        else:
            report.append(f"  平均Uno C-index: N/A")
            report.append(f"  平均Harrell C-index: N/A")
        report.append("  各Fold详细值:")
        report.append(f"    {'Fold':<8} {'Uno C-index':<15} {'Harrell C-index':<15} {'Uno 24m':<15}")
        report.append("    " + "-" * 60)
        
        for fold in range(5):
            uno_val = row[f'fold_{fold}_uno']
            harrell_val = row[f'fold_{fold}_harrell']
            uno_24m_val = row[f'fold_{fold}_uno_24m']
            
            if pd.notna(uno_val):
                report.append(
                    f"    fold_{fold:<3} {uno_val:<15.6f} {harrell_val:<15.6f} {uno_24m_val:<15.6f}"
                )
            else:
                report.append(f"    fold_{fold:<3} {'缺失':<15} {'缺失':<15} {'缺失':<15}")
        
        report.append("")
    
    # 统计信息
    report.append("=" * 100)
    report.append("统计摘要")
    report.append("=" * 100)
    report.append("")
    
    valid_df = df_sorted[df_sorted['n_folds'] == 5]
    if len(valid_df) > 0:
        report.append(f"完整结果的学习率数量: {len(valid_df)} / {len(df_sorted)}")
        report.append(f"最佳学习率: {valid_df.iloc[0]['learning_rate']}")
        report.append(f"最佳Uno C-index: {valid_df.iloc[0]['mean_uno']:.3f} ± {valid_df.iloc[0]['std_uno']:.3f}")
        report.append(f"最佳Harrell C-index: {valid_df.iloc[0]['mean_harrell']:.3f} ± {valid_df.iloc[0]['std_harrell']:.3f}")
        report.append("")
        
        report.append("Top 5 学习率:")
        for i in range(min(5, len(valid_df))):
            row = valid_df.iloc[i]
            report.append(f"  {i+1}. LR={row['learning_rate']:15s}  Uno={row['mean_uno']:.3f}±{row['std_uno']:.3f}  Harrell={row['mean_harrell']:.3f}±{row['std_harrell']:.3f}")
    
    report.append("")
    report.append("=" * 100)
    report.append("分析完成")
    report.append("=" * 100)
    
    return "\n".join(report)

def main():
    """主函数"""
    print("正在加载所有结果...")
    df, missing_folds = load_all_results()
    
    print(f"找到 {len(df)} 个学习率的结果")
    print(f"缺失fold的学习率数量: {len(missing_folds)}")
    
    print("生成详细报告...")
    report = generate_detailed_report(df, missing_folds)
    
    # 输出到控制台
    print("\n" + report)
    
    # 保存报告到文件
    output_dir = Path("/home/lichengze/Research/DeepFeature/myresearch/phase3")
    report_file = output_dir / "lr_grid_search_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n报告已保存到: {report_file}")
    
    # 保存CSV格式的详细数据（符合图片格式）
    csv_file = output_dir / "lr_grid_search_detailed_results.csv"
    df_sorted = df.sort_values('mean_uno', ascending=False, na_position='last').reset_index(drop=True)
    df_sorted['rank'] = range(1, len(df_sorted) + 1)
    
    # 创建符合图片格式的CSV
    csv_data = []
    for _, row in df_sorted.iterrows():
        if pd.notna(row['mean_uno']):
            # Result格式: "mean, std" (都是三位小数)
            result_str = f"{row['mean_uno']:.3f}, {row['std_uno']:.3f}"
            
            # Each Fold格式: "fold0, fold1, fold2, fold3, fold4" (都是三位小数)
            # 只包含有效的fold值，过滤掉缺失的fold
            fold_values = []
            for fold in range(5):
                uno_val = row[f'fold_{fold}_uno']
                if pd.notna(uno_val):
                    fold_values.append(f"{uno_val:.3f}")
            
            fold_str = ", ".join(fold_values)
            
            # LR格式: 科学计数法
            lr_formatted = format_lr_scientific(row['learning_rate'])
        else:
            result_str = "N/A"
            fold_str = "N/A"
            lr_formatted = format_lr_scientific(row['learning_rate'])
        
        csv_data.append({
            'Result': result_str,
            'Each Fold': fold_str,
            'LR': lr_formatted
        })
    
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_file, index=False)
    print(f"详细数据已保存到: {csv_file}")
    
    # 保存每个fold的排名数据
    per_fold_rank_file = output_dir / "lr_grid_search_per_fold_ranking.csv"
    per_fold_data = []
    for _, row in df_sorted.iterrows():
        for fold in range(5):
            if pd.notna(row[f'fold_{fold}_uno']):
                per_fold_data.append({
                    'learning_rate': row['learning_rate'],
                    'overall_rank': row['rank'],
                    'fold': fold,
                    'uno_integral': row[f'fold_{fold}_uno'],
                    'harrell': row[f'fold_{fold}_harrell'],
                    'uno_24m': row[f'fold_{fold}_uno_24m']
                })
    
    per_fold_df = pd.DataFrame(per_fold_data)
    # 按fold分组，在每个fold内按uno_integral排序
    per_fold_df['fold_rank'] = per_fold_df.groupby('fold')['uno_integral'].rank(ascending=False, method='min')
    per_fold_df = per_fold_df.sort_values(['fold', 'fold_rank'])
    per_fold_df.to_csv(per_fold_rank_file, index=False)
    print(f"每个fold的排名数据已保存到: {per_fold_rank_file}")

if __name__ == "__main__":
    main()

