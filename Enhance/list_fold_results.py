#!/usr/bin/env python3
"""
列出每个 fold 的最佳训练结果
"""

from pathlib import Path
import sys
import pickle
import csv

# 尝试导入 torch，如果失败则使用 pickle
try:
    import torch
    USE_TORCH = True
except ImportError:
    USE_TORCH = False

def load_checkpoint(ckpt_path):
    """加载 checkpoint 文件"""
    if USE_TORCH:
        return torch.load(ckpt_path, map_location='cpu', weights_only=False)
    else:
        with open(ckpt_path, 'rb') as f:
            return pickle.load(f)

def list_fold_results(base_dir):
    """列出指定目录下所有 fold 的最佳结果"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"错误: 目录不存在: {base_path}")
        return
    
    results = []
    
    # 查找所有 fold 目录
    fold_dirs = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('fold_')])
    
    if not fold_dirs:
        print(f"错误: 在 {base_path} 中未找到任何 fold 目录")
        return
    
    print("="*80)
    print(f"分析目录: {base_path}")
    print("="*80)
    print()
    
    for fold_dir in fold_dirs:
        fold_num = fold_dir.name.replace('fold_', '')
        best_pt = fold_dir / "best.pt"
        val_pred_csv = fold_dir / "val_pred.csv"
        
        if not best_pt.exists():
            print(f"fold_{fold_num}: ⚠️  best.pt 不存在")
            continue
        
        try:
            # 加载 checkpoint
            ckpt = load_checkpoint(best_pt)
            
            # 提取信息
            epoch = ckpt.get('epoch', 'N/A')
            uno_integral = ckpt.get('uno_integral', 'N/A')
            harrell = ckpt.get('harrell', 'N/A')
            n_time_bins = ckpt.get('n_time_bins', 'N/A')
            clinical_features = ckpt.get('clinical_features', [])
            clinical_dim = ckpt.get('clinical_dim', 'N/A')
            
            # 检查 val_pred.csv
            has_val_pred = val_pred_csv.exists()
            
            # 模型大小
            model_size_mb = best_pt.stat().st_size / (1024 * 1024)
            
            results.append({
                'fold': fold_num,
                'epoch': epoch,
                'uno_cindex': uno_integral,
                'harrell_cindex': harrell,
                'n_time_bins': n_time_bins,
                'clinical_dim': clinical_dim,
                'clinical_features': ', '.join(clinical_features) if clinical_features else 'N/A',
                'has_val_pred': '✅' if has_val_pred else '❌',
                'model_size_mb': model_size_mb
            })
            
        except Exception as e:
            print(f"fold_{fold_num}: ❌ 加载失败 - {e}")
            continue
    
    if not results:
        print("未找到任何有效的结果")
        return
    
    # 显示结果
    print("="*80)
    print("所有 Fold 的最佳结果汇总")
    print("="*80)
    print()
    
    for result in results:
        print(f"Fold {result['fold']}:")
        print(f"  Epoch: {result['epoch']}")
        
        uno_val = result['uno_cindex']
        if isinstance(uno_val, (int, float)):
            print(f"  Uno C-index: {uno_val:.4f}")
        else:
            print(f"  Uno C-index: {uno_val}")
        
        harrell_val = result['harrell_cindex']
        if isinstance(harrell_val, (int, float)):
            print(f"  Harrell C-index: {harrell_val:.4f}")
        else:
            print(f"  Harrell C-index: {harrell_val}")
        
        print(f"  Time bins: {result['n_time_bins']}")
        print(f"  Clinical dim: {result['clinical_dim']}")
        print(f"  Clinical features: {result['clinical_features']}")
        print(f"  Val predictions: {result['has_val_pred']}")
        print(f"  Model size: {result['model_size_mb']:.1f} MB")
        print()
    
    # 统计信息
    uno_values = [r['uno_cindex'] for r in results if isinstance(r['uno_cindex'], (int, float))]
    if uno_values:
        mean_uno = sum(uno_values) / len(uno_values)
        max_uno = max(uno_values)
        min_uno = min(uno_values)
        std_uno = (sum((x - mean_uno)**2 for x in uno_values) / len(uno_values))**0.5
        
        print("="*80)
        print("统计信息:")
        print("="*80)
        print(f"  Uno C-index: 平均 = {mean_uno:.4f}, 最大 = {max_uno:.4f}, "
              f"最小 = {min_uno:.4f}, 标准差 = {std_uno:.4f}")
    
    harrell_values = [r['harrell_cindex'] for r in results if isinstance(r['harrell_cindex'], (int, float))]
    if harrell_values:
        mean_harrell = sum(harrell_values) / len(harrell_values)
        max_harrell = max(harrell_values)
        min_harrell = min(harrell_values)
        std_harrell = (sum((x - mean_harrell)**2 for x in harrell_values) / len(harrell_values))**0.5
        
        print(f"  Harrell C-index: 平均 = {mean_harrell:.4f}, 最大 = {max_harrell:.4f}, "
              f"最小 = {min_harrell:.4f}, 标准差 = {std_harrell:.4f}")
    
    print()
    
    # 保存为 CSV
    output_csv = base_path / "fold_results_summary.csv"
    with open(output_csv, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"结果已保存到: {output_csv}")
    print("="*80)


def main():
    if len(sys.argv) < 2:
        print("用法: python list_fold_results.py <base_dir>")
        print("示例: python list_fold_results.py /home/lichengze/Research/Enhance/phase3_7e4_output/fixed_split_seed42_lr7e-04")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    list_fold_results(base_dir)


if __name__ == "__main__":
    main()
