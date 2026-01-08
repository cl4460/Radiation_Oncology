#!/usr/bin/env python3
"""批量提取所有learning rate实验的embeddings"""
import sys
import os
from pathlib import Path
import subprocess

# 强制使用GPU 0（如果可用）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 添加项目路径
sys.path.append('/home/lichengze/Research/DeepFeature/myresearch')

# 导入extract_embeddings中的函数
from extract_embeddings import extract_for_experiment

def main():
    phase3_outputs = Path("/home/lichengze/Research/DeepFeature/myresearch/phase3_outputs")
    
    if not phase3_outputs.exists():
        print(f"❌ Error: {phase3_outputs} does not exist!")
        return
    
    # 查找所有lr_*目录
    lr_dirs = sorted([d for d in phase3_outputs.iterdir() 
                      if d.is_dir() and d.name.startswith('lr_')])
    
    if not lr_dirs:
        print("❌ No learning rate directories found!")
        return
    
    print("=" * 80)
    print(f"🚀 BATCH EMBEDDING EXTRACTION")
    print(f"Found {len(lr_dirs)} learning rate experiments")
    print("=" * 80)
    
    successful = []
    failed = []
    skipped = []
    
    for i, lr_dir in enumerate(lr_dirs, 1):
        lr_name = lr_dir.name
        print(f"\n{'='*80}")
        print(f"[{i}/{len(lr_dirs)}] Processing: {lr_name}")
        print(f"{'='*80}")
        
        # 检查是否已经有train_embeddings.npy
        all_train_exist = True
        for fold_idx in range(5):
            fold_dir = lr_dir / f"fold_{fold_idx}"
            train_emb_path = fold_dir / "train_embeddings.npy"
            if not train_emb_path.exists():
                all_train_exist = False
                break
        
        if all_train_exist:
            print(f"⏭️  Skipping {lr_name}: train_embeddings.npy already exist for all folds")
            skipped.append(lr_name)
            continue
        
        try:
            # 调用extract_for_experiment函数
            extract_for_experiment(lr_dir, seed=42)
            successful.append(lr_name)
            print(f"✅ Successfully extracted embeddings for {lr_name}")
        except Exception as e:
            failed.append((lr_name, str(e)))
            print(f"❌ Failed to extract {lr_name}: {e}")
            continue
    
    # 打印汇总报告
    print("\n" + "=" * 80)
    print("📊 SUMMARY REPORT")
    print("=" * 80)
    print(f"Total experiments: {len(lr_dirs)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"⏭️  Skipped (already exists): {len(skipped)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\n✅ Successfully extracted ({len(successful)}):")
        for name in successful:
            print(f"   - {name}")
    
    if skipped:
        print(f"\n⏭️  Skipped ({len(skipped)}):")
        for name in skipped:
            print(f"   - {name}")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for name, error in failed:
            print(f"   - {name}: {error}")
    
    print("\n" + "=" * 80)
    print("🎉 Batch extraction complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

