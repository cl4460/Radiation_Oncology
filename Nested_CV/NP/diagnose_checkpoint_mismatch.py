#!/usr/bin/env python3
"""
diagnose_checkpoint_mismatch.py

详细诊断checkpoint结构，检查：
1. fold_X/best.pt 的 val_uno 和 variant
2. fold_X/model_img/best.pt 的 val_uno
3. fold_X/model_imgclin/best.pt 的 val_uno
4. fold_X/selection.json 的内容
5. val_pred.csv 来自哪个variant

用法：
  python diagnose_checkpoint_mismatch.py /path/to/lr_7e-04_29night
"""

import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path


def diagnose_fold(fold_dir: str) -> dict:
    """诊断单个fold的checkpoint结构"""
    fold_path = Path(fold_dir)
    result = {"fold": fold_path.name}
    
    # 1. 检查 fold_X/best.pt
    main_best = fold_path / "best.pt"
    if main_best.exists():
        ckpt = torch.load(main_best, map_location="cpu")
        result["main_best"] = {
            "val_uno": ckpt.get("val_uno") or ckpt.get("metrics", {}).get("uno"),
            "variant": ckpt.get("variant"),
            "use_clinical": ckpt.get("use_clinical"),
            "keys": list(ckpt.keys())[:10],
        }
    else:
        result["main_best"] = "NOT FOUND"
    
    # 2. 检查 model_img/best.pt
    img_best = fold_path / "model_img" / "best.pt"
    if img_best.exists():
        ckpt = torch.load(img_best, map_location="cpu")
        result["model_img"] = {
            "val_uno": ckpt.get("val_uno") or ckpt.get("metrics", {}).get("uno"),
            "use_clinical": ckpt.get("use_clinical"),
        }
        # 检查val_pred.csv
        val_pred = fold_path / "model_img" / "val_pred.csv"
        if val_pred.exists():
            df = pd.read_csv(val_pred)
            result["model_img"]["val_pred_count"] = len(df)
            result["model_img"]["val_pred_pids"] = df["patient_id"].astype(str).tolist()[:3]
    else:
        result["model_img"] = "NOT FOUND"
    
    # 3. 检查 model_imgclin/best.pt
    imgclin_best = fold_path / "model_imgclin" / "best.pt"
    if imgclin_best.exists():
        ckpt = torch.load(imgclin_best, map_location="cpu")
        result["model_imgclin"] = {
            "val_uno": ckpt.get("val_uno") or ckpt.get("metrics", {}).get("uno"),
            "use_clinical": ckpt.get("use_clinical"),
        }
        val_pred = fold_path / "model_imgclin" / "val_pred.csv"
        if val_pred.exists():
            df = pd.read_csv(val_pred)
            result["model_imgclin"]["val_pred_count"] = len(df)
            result["model_imgclin"]["val_pred_pids"] = df["patient_id"].astype(str).tolist()[:3]
    else:
        result["model_imgclin"] = "NOT FOUND"
    
    # 4. 检查 selection.json
    sel_json = fold_path / "selection.json"
    if sel_json.exists():
        with open(sel_json) as f:
            sel = json.load(f)
        result["selection"] = sel
    else:
        result["selection"] = "NOT FOUND"
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_checkpoint_mismatch.py <ckpt_root>")
        sys.exit(1)
    
    ckpt_root = Path(sys.argv[1])
    fold_dirs = sorted([d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    
    print("=" * 80)
    print(f"Diagnosing: {ckpt_root}")
    print("=" * 80)
    
    summary = {
        "main_best_unos": [],
        "img_unos": [],
        "imgclin_unos": [],
        "winner_variants": [],
    }
    
    for fold_dir in fold_dirs:
        print(f"\n{'='*60}")
        print(f"{fold_dir.name}")
        print(f"{'='*60}")
        
        result = diagnose_fold(str(fold_dir))
        
        # main_best
        if isinstance(result["main_best"], dict):
            uno = result["main_best"]["val_uno"]
            var = result["main_best"]["variant"]
            use_clin = result["main_best"]["use_clinical"]
            print(f"  [fold_X/best.pt]")
            print(f"    val_uno: {uno}")
            print(f"    variant: {var}")
            print(f"    use_clinical: {use_clin}")
            summary["main_best_unos"].append(uno)
            summary["winner_variants"].append(var)
        else:
            print(f"  [fold_X/best.pt] NOT FOUND")
            summary["main_best_unos"].append(None)
            summary["winner_variants"].append(None)
        
        # model_img
        if isinstance(result["model_img"], dict):
            uno = result["model_img"]["val_uno"]
            print(f"  [model_img/best.pt]")
            print(f"    val_uno: {uno}")
            if "val_pred_count" in result["model_img"]:
                print(f"    val_pred: {result['model_img']['val_pred_count']} patients")
                print(f"    first 3: {result['model_img']['val_pred_pids']}")
            summary["img_unos"].append(uno)
        else:
            print(f"  [model_img] NOT FOUND")
            summary["img_unos"].append(None)
        
        # model_imgclin
        if isinstance(result["model_imgclin"], dict):
            uno = result["model_imgclin"]["val_uno"]
            print(f"  [model_imgclin/best.pt]")
            print(f"    val_uno: {uno}")
            if "val_pred_count" in result["model_imgclin"]:
                print(f"    val_pred: {result['model_imgclin']['val_pred_count']} patients")
                print(f"    first 3: {result['model_imgclin']['val_pred_pids']}")
            summary["imgclin_unos"].append(uno)
        else:
            print(f"  [model_imgclin] NOT FOUND")
            summary["imgclin_unos"].append(None)
        
        # selection.json
        if isinstance(result["selection"], dict):
            print(f"  [selection.json]")
            for k, v in result["selection"].items():
                print(f"    {k}: {v}")
        else:
            print(f"  [selection.json] NOT FOUND")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nfold_X/best.pt val_unos: {summary['main_best_unos']}")
    if all(u is not None for u in summary['main_best_unos']):
        print(f"  Mean: {sum(summary['main_best_unos'])/len(summary['main_best_unos']):.4f}")
    
    print(f"\nmodel_img val_unos: {summary['img_unos']}")
    if all(u is not None for u in summary['img_unos']):
        print(f"  Mean: {sum(summary['img_unos'])/len(summary['img_unos']):.4f}")
    
    print(f"\nmodel_imgclin val_unos: {summary['imgclin_unos']}")
    if all(u is not None for u in summary['imgclin_unos']):
        print(f"  Mean: {sum(summary['imgclin_unos'])/len(summary['imgclin_unos']):.4f}")
    
    print(f"\nWinner variants: {summary['winner_variants']}")
    
    # Diagnosis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    # Check for mismatch
    for i, (main_uno, img_uno, imgclin_uno, variant) in enumerate(zip(
        summary['main_best_unos'], 
        summary['img_unos'], 
        summary['imgclin_unos'],
        summary['winner_variants']
    )):
        if main_uno is None:
            print(f"  fold_{i}: main best.pt missing!")
            continue
            
        if variant == "model_img" and img_uno is not None:
            if abs(main_uno - img_uno) > 0.01:
                print(f"  fold_{i}: ⚠️ MISMATCH! main_uno={main_uno:.4f} vs img_uno={img_uno:.4f}")
            else:
                print(f"  fold_{i}: ✓ variant=img, uno matches")
        elif variant == "model_imgclin" and imgclin_uno is not None:
            if abs(main_uno - imgclin_uno) > 0.01:
                print(f"  fold_{i}: ⚠️ MISMATCH! main_uno={main_uno:.4f} vs imgclin_uno={imgclin_uno:.4f}")
            else:
                print(f"  fold_{i}: ✓ variant=imgclin, uno matches")
        else:
            print(f"  fold_{i}: variant={variant}, main_uno={main_uno}")


if __name__ == "__main__":
    main()