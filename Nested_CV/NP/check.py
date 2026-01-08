#!/usr/bin/env python3
"""
诊断脚本：检查checkpoint中的fold划分信息
"""
import torch
import json
import os
from pathlib import Path

def check_ckpt_structure(ckpt_root: str):
    """检查checkpoint中保存了什么信息"""
    root = Path(ckpt_root)
    fold_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    
    print(f"Found {len(fold_dirs)} folds in {ckpt_root}\n")
    
    for fold_dir in fold_dirs:
        print(f"\n{'='*60}")
        print(f"{fold_dir.name}")
        print(f"{'='*60}")
        
        # 检查 selection.json
        sel_json = fold_dir / "selection.json"
        if sel_json.exists():
            print(f"\n  [selection.json] 存在")
            with open(sel_json) as f:
                sel = json.load(f)
            print(f"    Keys: {list(sel.keys())}")
            
            # 检查是否有val_patients
            if "val_patients" in sel:
                vp = sel["val_patients"]
                print(f"    val_patients: {len(vp)} patients")
                print(f"    前5个: {vp[:5]}")
            if "val_indices" in sel:
                vi = sel["val_indices"]
                print(f"    val_indices: {len(vi)} indices")
                print(f"    前5个: {vi[:5]}")
        else:
            print(f"\n  [selection.json] 不存在")
        
        # 检查 best.pt
        best_pt = fold_dir / "best.pt"
        if best_pt.exists():
            ckpt = torch.load(best_pt, map_location="cpu")
            print(f"\n  [best.pt] Keys: {list(ckpt.keys())}")
            
            # 检查是否有val相关信息
            for key in ["val_patients", "val_patient_ids", "val_indices", "val_ids"]:
                if key in ckpt:
                    val_info = ckpt[key]
                    print(f"    {key}: {len(val_info)} items, 前5个: {val_info[:5]}")
            
            # 检查metrics
            if "metrics" in ckpt:
                print(f"    metrics: {ckpt['metrics']}")
            
            # 检查val_uno
            for key in ["val_uno", "uno", "best_uno"]:
                if key in ckpt:
                    print(f"    {key}: {ckpt[key]}")
            
            # 检查seed
            if "seed" in ckpt:
                print(f"    seed: {ckpt['seed']}")
            if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
                if "seed" in ckpt["cfg"]:
                    print(f"    cfg.seed: {ckpt['cfg']['seed']}")
        
        # 检查 model_img/best.pt 和 model_imgclin/best.pt
        for variant in ["model_img", "model_imgclin"]:
            variant_pt = fold_dir / variant / "best.pt"
            if variant_pt.exists():
                ckpt = torch.load(variant_pt, map_location="cpu")
                print(f"\n  [{variant}/best.pt]")
                for key in ["val_patients", "val_patient_ids", "val_indices"]:
                    if key in ckpt:
                        val_info = ckpt[key]
                        print(f"    {key}: {len(val_info)} items, 前5个: {val_info[:5]}")
                if "variant" in ckpt:
                    print(f"    variant: {ckpt['variant']}")
                for key in ["val_uno", "uno"]:
                    if key in ckpt:
                        print(f"    {key}: {ckpt[key]}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python check_ckpt_structure.py <ckpt_root>")
        print("Example: python check_ckpt_structure.py /home/lichengze/Research/NP/lr_7e-04_29night")
        sys.exit(1)
    
    check_ckpt_structure(sys.argv[1])