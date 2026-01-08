#!/usr/bin/env python3
"""
分析R0测试结果，包括每个fold的详细值和排名
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_results():
    """加载所有R0测试结果"""
    base_dir = Path(__file__).parent
    
    results = {}
    for test_name in ['clinical', 'deep', 'fusion']:
        dir_name = f"R0_{test_name}_verified"
        results[test_name] = {
            'per_fold': pd.read_csv(base_dir / dir_name / 'r0_per_fold_summary.csv'),
            'summary': json.load(open(base_dir / dir_name / 'r0_summary.json'))
        }
    
    return results

def check_missing_folds(results):
    """检查是否有缺失的fold"""
    expected_folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']
    missing = {}
    
    for test_name, data in results.items():
        available_folds = set(data['per_fold']['fold'].tolist())
        missing_folds = set(expected_folds) - available_folds
        if missing_folds:
            missing[test_name] = sorted(missing_folds)
    
    return missing

def analyze_and_rank(results):
    """分析结果并进行排名"""
    analysis = {}
    
    for test_name, data in results.items():
        df = data['per_fold'].copy()
        
        # 计算每个fold的综合得分（使用Harrell C-index的均值）
        df['score'] = df['perm_y_harrell_mean']
        
        # 按得分排序
        df_ranked = df.sort_values('score', ascending=False).reset_index(drop=True)
        df_ranked['rank'] = range(1, len(df_ranked) + 1)
        
        analysis[test_name] = {
            'data': df_ranked,
            'summary': data['summary']
        }
    
    return analysis

def generate_report(results, analysis, missing_folds):
    """生成详细的分析报告"""
    report = []
    report.append("=" * 80)
    report.append("R0测试结果详细分析报告")
    report.append("=" * 80)
    report.append("")
    
    # 检查缺失的fold
    if missing_folds:
        report.append("⚠️  缺失的Fold检测:")
        for test_name, folds in missing_folds.items():
            report.append(f"  - {test_name}: 缺失 {', '.join(folds)}")
        report.append("")
    else:
        report.append("✅ 所有测试都包含完整的5个fold (fold_0 到 fold_4)")
        report.append("")
    
    # 对每个测试进行详细分析
    for test_name in ['clinical', 'deep', 'fusion']:
        report.append("=" * 80)
        report.append(f"测试类型: {test_name.upper()}")
        report.append("=" * 80)
        report.append("")
        
        df = analysis[test_name]['data']
        summary = analysis[test_name]['summary']
        
        # 总体统计
        report.append("总体统计:")
        report.append(f"  Harrell C-index: {summary['harrell']['mean']:.6f} ± {summary['harrell']['std']:.6f}")
        report.append(f"  Uno C-index: {summary['uno_ipcw']['mean']:.6f} ± {summary['uno_ipcw']['std']:.6f}")
        report.append(f"  状态: {summary['harrell']['flag']}")
        report.append(f"  说明: {summary['harrell']['reason']}")
        report.append("")
        
        # 每个fold的详细值
        report.append("各Fold详细结果 (按Harrell C-index均值排名):")
        report.append("-" * 80)
        report.append(f"{'排名':<6} {'Fold':<10} {'Harrell均值':<15} {'Harrell标准差':<15} {'Uno均值':<15} {'Uno标准差':<15}")
        report.append("-" * 80)
        
        for _, row in df.iterrows():
            report.append(
                f"{row['rank']:<6} {row['fold']:<10} "
                f"{row['perm_y_harrell_mean']:<15.6f} {row['perm_y_harrell_std']:<15.6f} "
                f"{row['perm_y_uno_mean']:<15.6f} {row['perm_y_uno_std']:<15.6f}"
            )
        
        report.append("")
        
        # 统计信息
        report.append("Fold统计信息:")
        report.append(f"  最高Harrell C-index: {df['perm_y_harrell_mean'].max():.6f} ({df.loc[df['perm_y_harrell_mean'].idxmax(), 'fold']})")
        report.append(f"  最低Harrell C-index: {df['perm_y_harrell_mean'].min():.6f} ({df.loc[df['perm_y_harrell_mean'].idxmin(), 'fold']})")
        report.append(f"  平均Harrell C-index: {df['perm_y_harrell_mean'].mean():.6f}")
        report.append(f"  标准差: {df['perm_y_harrell_mean'].std():.6f}")
        report.append("")
    
    # 跨测试比较
    report.append("=" * 80)
    report.append("跨测试比较")
    report.append("=" * 80)
    report.append("")
    
    comparison_data = []
    for test_name in ['clinical', 'deep', 'fusion']:
        summary = analysis[test_name]['summary']
        comparison_data.append({
            '测试类型': test_name,
            'Harrell均值': summary['harrell']['mean'],
            'Harrell标准差': summary['harrell']['std'],
            'Uno均值': summary['uno_ipcw']['mean'],
            'Uno标准差': summary['uno_ipcw']['std'],
            '状态': summary['harrell']['flag']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Harrell均值', ascending=False)
    comparison_df['排名'] = range(1, len(comparison_df) + 1)
    
    report.append("按总体Harrell C-index均值排名:")
    report.append("-" * 80)
    report.append(f"{'排名':<6} {'测试类型':<15} {'Harrell均值':<15} {'Harrell标准差':<15} {'Uno均值':<15} {'状态':<10}")
    report.append("-" * 80)
    
    for _, row in comparison_df.iterrows():
        report.append(
            f"{row['排名']:<6} {row['测试类型']:<15} "
            f"{row['Harrell均值']:<15.6f} {row['Harrell标准差']:<15.6f} "
            f"{row['Uno均值']:<15.6f} {row['状态']:<10}"
        )
    
    report.append("")
    report.append("=" * 80)
    report.append("分析完成")
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    """主函数"""
    print("正在加载结果...")
    results = load_results()
    
    print("检查缺失的fold...")
    missing_folds = check_missing_folds(results)
    
    print("分析结果并排名...")
    analysis = analyze_and_rank(results)
    
    print("生成报告...")
    report = generate_report(results, analysis, missing_folds)
    
    # 输出到控制台
    print("\n" + report)
    
    # 保存到文件
    output_file = Path(__file__).parent / "r0_analysis_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存到: {output_file}")
    
    # 同时保存CSV格式的详细数据
    for test_name in ['clinical', 'deep', 'fusion']:
        csv_file = Path(__file__).parent / f"r0_{test_name}_ranked.csv"
        analysis[test_name]['data'].to_csv(csv_file, index=False)
        print(f"{test_name}排名数据已保存到: {csv_file}")

if __name__ == "__main__":
    main()














