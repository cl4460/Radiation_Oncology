#!/usr/bin/env python3
"""
从训练日志中提取所有结果，生成汇总报告
即使模型文件被覆盖了，我们仍然可以从日志中提取所有结果
"""

import re
from pathlib import Path
from collections import defaultdict
import csv

log_dir = Path("/home/lichengze/Research/Enhance/training_logs")
output_file = Path("/home/lichengze/Research/Enhance/training_results_summary.csv")

results = []

# 解析所有日志文件
for log_file in sorted(log_dir.glob("gpu*_seed*_*.log")):
    if "worker" in log_file.name:
        continue
    
    content = log_file.read_text()
    
    # Extract seed
    seed_match = re.search(r'seed=(\d+)', content)
    if not seed_match:
        continue
    seed = int(seed_match.group(1))
    
    # Extract GPU
    gpu_match = re.search(r'GPU (\d+):', content)
    if not gpu_match:
        continue
    gpu = int(gpu_match.group(1))
    
    # Extract best C-index
    cindex_match = re.search(r'Best Val Uno C-index: ([\d.]+)', content)
    if not cindex_match:
        continue
    cindex = float(cindex_match.group(1))
    
    # Extract Harrell C-index if available
    harrell_match = re.search(r'Harrell ([\d.]+)', content.split("Best Val Uno")[-1])
    harrell = float(harrell_match.group(1)) if harrell_match else None
    
    # Extract completion time
    time_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*SUCCESS', content)
    completion_time = time_match.group(1) if time_match else None
    
    # Extract model path
    model_match = re.search(r'Model saved to: (.+best\.pt)', content)
    model_path = model_match.group(1) if model_match else "N/A"
    
    results.append({
        'log_file': log_file.name,
        'gpu': gpu,
        'seed': seed,
        'uno_cindex': cindex,
        'harrell_cindex': harrell,
        'model_path': model_path,
        'completion_time': completion_time
    })

# 排序
results.sort(key=lambda x: (x['seed'], -x['uno_cindex']))

# 保存到CSV
if results:
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

# 按seed分组统计
by_seed = defaultdict(list)
for r in results:
    by_seed[r['seed']].append(r)

print("="*80)
print("训练结果汇总 (已保存到 CSV)")
print("="*80)
print(f"\n总共 {len(results)} 个任务完成")
print(f"\n按 Seed 分组统计:")
for seed in sorted(by_seed.keys()):
    seed_results = sorted(by_seed[seed], key=lambda x: -x['uno_cindex'])
    cindexes = [r['uno_cindex'] for r in seed_results]
    mean_cindex = sum(cindexes) / len(cindexes)
    std_cindex = (sum((x - mean_cindex)**2 for x in cindexes) / len(cindexes))**0.5
    best = seed_results[0]
    
    print(f"\nSeed {seed}:")
    print(f"  任务数: {len(seed_results)}")
    print(f"  最佳 C-index: {best['uno_cindex']:.4f} (GPU{best['gpu']})")
    print(f"  平均 C-index: {mean_cindex:.4f} ± {std_cindex:.4f}")
    print(f"  详细结果:")
    for r in seed_results:
        print(f"    GPU{r['gpu']:2d} | Uno: {r['uno_cindex']:.4f} | {r['completion_time']}")

print(f"\n\n结果已保存到: {output_file}")
print("="*80)

