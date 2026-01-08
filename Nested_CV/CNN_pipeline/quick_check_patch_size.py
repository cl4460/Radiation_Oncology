#!/usr/bin/env python
"""
快速检查FIXED_PATCH是否合适
生成简洁的统计报告
"""

import numpy as np
import pandas as pd
from pathlib import Path

# 配置（自动检测路径）
import os
CROP_LOG_PATHS = [
    "/home/lichengze/Research/CNN_pipeline/phase2_outputs/crop_log.csv",
    "/home/lichengze/CNN_pipeline/phase2_outputs/crop_log.csv",
]
CROP_LOG = None
for path in CROP_LOG_PATHS:
    if os.path.exists(path):
        CROP_LOG = path
        break
FIXED_PATCH = (160, 192, 128)
MARGIN_MM = 20

def main():
    print("="*70)
    print("快速检查：FIXED_PATCH是否合适？")
    print("="*70)
    print(f"FIXED_PATCH: {FIXED_PATCH} (X, Y, Z)")
    print(f"MARGIN: {MARGIN_MM} mm")
    print("="*70)
    
    # 如果已经有crop_log.csv，直接使用
    if CROP_LOG is None:
        print(f"\n❌ 未找到crop_log.csv")
        print(f"请先运行以下之一：")
        print(f"  1. python phase2_preprocessing.py (生成crop_log.csv)")
        print(f"  2. python analyze_dataset_sizes.py (完整分析)")
        print("="*70)
        return
    
    crop_log_path = Path(CROP_LOG)
    if crop_log_path.exists():
        print(f"\n读取已有的crop_log.csv...")
        df = pd.read_csv(crop_log_path)
        
        if 'truncated' in df.columns and 'retained_mask_ratio' in df.columns:
            total = len(df)
            truncated = df['truncated'].sum()
            truncated_pct = truncated / total * 100
            
            low_retention = (df['retained_mask_ratio'] < 0.9).sum()
            low_retention_pct = low_retention / total * 100
            
            very_low_retention = (df['retained_mask_ratio'] < 0.8).sum()
            very_low_retention_pct = very_low_retention / total * 100
            
            print(f"\n总病例数: {total}")
            print(f"\n【Truncation统计】")
            print(f"  会被truncate的case数: {truncated} ({truncated_pct:.1f}%)")
            
            print(f"\n【Mask保留比例统计】")
            print(f"  保留比例 < 90%: {low_retention} ({low_retention_pct:.1f}%)")
            print(f"  保留比例 < 80%: {very_low_retention} ({very_low_retention_pct:.1f}%)")
            print(f"  平均保留比例: {df['retained_mask_ratio'].mean():.3f}")
            print(f"  中位数保留比例: {df['retained_mask_ratio'].median():.3f}")
            print(f"  最小保留比例: {df['retained_mask_ratio'].min():.3f}")
            
            print(f"\n【结论】")
            if truncated_pct < 5 and low_retention_pct < 5:
                print(f"  ✅ FIXED_PATCH设置合理！")
                print(f"     - 只有{truncated_pct:.1f}%的case会被truncate")
                print(f"     - 只有{low_retention_pct:.1f}%的case保留比例 < 90%")
            elif truncated_pct < 10 and low_retention_pct < 10:
                print(f"  ⚠️  FIXED_PATCH设置基本合理，但有一些case受影响")
                print(f"     - {truncated_pct:.1f}%的case会被truncate")
                print(f"     - {low_retention_pct:.1f}%的case保留比例 < 90%")
            else:
                print(f"  ❌ FIXED_PATCH可能过小")
                print(f"     - {truncated_pct:.1f}%的case会被truncate")
                print(f"     - {low_retention_pct:.1f}%的case保留比例 < 90%")
            
            # 显示最严重的case
            if truncated > 0:
                print(f"\n【最严重的truncation case（前5个）】")
                worst = df[df['truncated']].nlargest(5, 'retained_mask_ratio', keep='all')
                for idx, row in worst.iterrows():
                    print(f"  {row['patient_id']}: 保留比例 = {row['retained_mask_ratio']:.3f}")
            
            print("="*70)
            return
    
    # 如果没有crop_log.csv，提示运行完整分析
    print(f"\n未找到crop_log.csv，请运行完整分析：")
    print(f"  python analyze_dataset_sizes.py")
    print("="*70)

if __name__ == "__main__":
    main()

