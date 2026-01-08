#!/usr/bin/env python3
"""
从训练日志中提取结果，分析哪些任务的结果被覆盖了，
并生成只重新运行被覆盖任务的脚本
"""

import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

log_dir = Path("/home/lichengze/Research/Enhance/training_logs")
output_dir = Path("/home/lichengze/Research/Enhance/phase3_7e4_output")

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
    
    # Extract completion time
    time_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*SUCCESS', content)
    if time_match:
        completion_time = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S")
    else:
        completion_time = None
    
    results.append({
        'log_file': log_file.name,
        'gpu': gpu,
        'seed': seed,
        'cindex': cindex,
        'completion_time': completion_time
    })

# 按seed分组
by_seed = defaultdict(list)
for r in results:
    by_seed[r['seed']].append(r)

print("="*80)
print("分析结果:")
print("="*80)

# 检查每个seed的模型文件时间戳
for seed in sorted(by_seed.keys()):
    seed_results = sorted(by_seed[seed], key=lambda x: x['completion_time'] if x['completion_time'] else datetime.min)
    
    model_path = output_dir / f"fixed_split_seed{seed}_lr7e-04" / "fold_0" / "best.pt"
    if model_path.exists():
        model_time = datetime.fromtimestamp(model_path.stat().st_mtime)
    else:
        model_time = None
    
    print(f"\nSeed {seed}:")
    print(f"  模型文件时间: {model_time}")
    print(f"  任务结果 (按完成时间排序):")
    
    # 找出最后保存的任务
    last_task = None
    for r in seed_results:
        marker = ""
        if model_time and r['completion_time']:
            # 检查这个任务是否是最后保存的（时间最接近模型文件时间）
            time_diff = abs((r['completion_time'] - model_time).total_seconds())
            if time_diff < 300:  # 5分钟内
                marker = " <-- 当前保存的模型"
                last_task = r
        
        print(f"    GPU{r['gpu']:2d} | C-index: {r['cindex']:.4f} | {r['completion_time']}{marker}")
    
    # 找出最佳结果
    best_result = max(seed_results, key=lambda x: x['cindex'])
    
    # 如果最佳结果不是最后保存的，需要重新运行
    if last_task and best_result['cindex'] > last_task['cindex']:
        print(f"\n  ⚠️  警告: 最佳结果 (GPU{best_result['gpu']}, C-index={best_result['cindex']:.4f}) 被覆盖了!")
        print(f"     当前保存的是 GPU{last_task['gpu']}, C-index={last_task['cindex']:.4f}")
    elif best_result == last_task:
        print(f"\n  ✅ 最佳结果已保存 (GPU{best_result['gpu']}, C-index={best_result['cindex']:.4f})")

print("\n" + "="*80)
print("建议:")
print("="*80)
print("1. 检查上面的分析，确认哪些任务的结果被覆盖了")
print("2. 如果最佳结果被覆盖，可以:")
print("   - 从日志中提取所有结果用于分析（虽然模型文件丢失了）")
print("   - 或者只重新运行被覆盖的最佳任务（使用新的run_id）")
print("3. 生成恢复脚本...")

# 生成恢复脚本
recover_script = """#!/bin/bash
# 恢复被覆盖的训练任务
# 只重新运行那些性能更好但被覆盖的任务

SCRIPT_DIR="/home/lichengze/Research/Enhance"
PYTHON_SCRIPT="${SCRIPT_DIR}/phase3_0617_update.py"
LOG_DIR="${SCRIPT_DIR}/training_logs"
mkdir -p "${LOG_DIR}"

# 需要恢复的任务列表（根据上面的分析手动填写）
# 格式: run_id seed split_json
RECOVER_TASKS=(
    # 示例: "1 42 /home/lichengze/Research/Enhance/splits/fixed_split_seed42.json"
)

for task in "${RECOVER_TASKS[@]}"; do
    read -r run_id seed split_json <<< "${task}"
    log_file="${LOG_DIR}/recover_gpu_seed${seed}_run${run_id}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Recovering: run_id=${run_id}, seed=${seed}"
    
    cd "${SCRIPT_DIR}"
    PYTHONUNBUFFERED=1 python -u "${PYTHON_SCRIPT}" \\
        --seed ${seed} \\
        --split_json "${split_json}" \\
        --lr 7e-4 \\
        --run_id ${run_id} \\
        2>&1 | tee "${log_file}"
    
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Completed: run_id=${run_id}, seed=${seed}"
done

echo "All recovery tasks completed!"
"""

recover_script_path = log_dir.parent / "recover_training.sh"
recover_script_path.write_text(recover_script)
recover_script_path.chmod(0o755)

print(f"\n已生成恢复脚本: {recover_script_path}")
print("请根据上面的分析，手动编辑脚本中的 RECOVER_TASKS 数组")









