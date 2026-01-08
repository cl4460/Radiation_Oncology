#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_donoharm_winners.py

用途：
  对已经训练好的 do-no-harm checkpoint 目录（ckpt_root）重新应用更保守的 winner 选择规则，
  不需要重新训练。

前提：
  每个 fold 目录下存在：
    fold_k/model_img/best.pt
    fold_k/model_imgclin/best.pt
  并且 fold_k/best.pt 是当前 winner（会被覆盖为新 winner）。

选择规则（默认与你 peer 建议一致）：
  - 只有当 imgclin 的 val_uno >= img + margin 才选 imgclin
  - 可选：再加一个 brier guard（imgclin 的 val_brier_24m 不应明显更差）

示例：
  python fix_donoharm_winners.py --ckpt_root /path/to/lr_7e-04_29night --margin 0.02
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from typing import Dict, Tuple

import torch


def load_pt(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu")


def pick_winner(
    ckpt_img: Dict, 
    ckpt_imgclin: Dict, 
    selection_json: Dict | None,
    margin: float, 
    brier_margin: float | None
) -> Tuple[str, Dict]:
    """
    Pick winner based on validation Uno C-index.
    
    Priority order (most reliable first):
    1. selection.json['img_uno'] and ['imgclin_uno'] - These are the values used during training
    2. ckpt['metrics']['uno'] - Some checkpoint structures
    3. ckpt['uno'] - Direct checkpoint field
    4. ckpt['val_uno'] - Fallback
    """
    # PRIORITY 1: Read from selection.json (most reliable - these are the values used during training)
    u_img = float("nan")
    u_ic = float("nan")
    
    if selection_json is not None:
        u_img = float(selection_json.get("img_uno", float("nan")))
        u_ic = float(selection_json.get("imgclin_uno", float("nan")))
    
    # PRIORITY 2-4: Fallback to checkpoint if selection.json values are NaN
    if math.isnan(u_img):
        if "metrics" in ckpt_img and isinstance(ckpt_img["metrics"], dict) and "uno" in ckpt_img["metrics"]:
            u_img = float(ckpt_img["metrics"]["uno"])
        elif "uno" in ckpt_img:
            u_img = float(ckpt_img["uno"])
        elif "val_uno" in ckpt_img:
            u_img = float(ckpt_img["val_uno"])
    
    if math.isnan(u_ic):
        if "metrics" in ckpt_imgclin and isinstance(ckpt_imgclin["metrics"], dict) and "uno" in ckpt_imgclin["metrics"]:
            u_ic = float(ckpt_imgclin["metrics"]["uno"])
        elif "uno" in ckpt_imgclin:
            u_ic = float(ckpt_imgclin["uno"])
        elif "val_uno" in ckpt_imgclin:
            u_ic = float(ckpt_imgclin["val_uno"])

    # Try multiple ways to get brier (also prioritize selection.json if available)
    b_img = None
    b_ic = None
    
    if selection_json is not None:
        # selection.json doesn't store separate img/imgclin brier, so we still need to read from ckpt
        # But we could add this in the future if needed
        pass
    
    if b_img is None:
        if "metrics" in ckpt_img and isinstance(ckpt_img["metrics"], dict) and "brier_24m" in ckpt_img["metrics"]:
            b_img = float(ckpt_img["metrics"]["brier_24m"])
        elif "brier_24m" in ckpt_img:
            b_img = float(ckpt_img["brier_24m"])
        elif "val_brier_24m" in ckpt_img:
            b_img = float(ckpt_img["val_brier_24m"])
    
    if b_ic is None:
        if "metrics" in ckpt_imgclin and isinstance(ckpt_imgclin["metrics"], dict) and "brier_24m" in ckpt_imgclin["metrics"]:
            b_ic = float(ckpt_imgclin["metrics"]["brier_24m"])
        elif "brier_24m" in ckpt_imgclin:
            b_ic = float(ckpt_imgclin["brier_24m"])
        elif "val_brier_24m" in ckpt_imgclin:
            b_ic = float(ckpt_imgclin["val_brier_24m"])

    # Handle NaN values: if either is NaN, prefer the non-NaN one; if both NaN, default to img
    u_img_valid = not math.isnan(u_img)
    u_ic_valid = not math.isnan(u_ic)
    
    if not u_img_valid and not u_ic_valid:
        # Both are NaN, default to img
        choose_ic = False
    elif not u_img_valid:
        # img is NaN, imgclin is valid -> choose imgclin
        choose_ic = True
    elif not u_ic_valid:
        # imgclin is NaN, img is valid -> choose img
        choose_ic = False
    else:
        # Both are valid, apply margin rule
        choose_ic = (u_ic >= u_img + margin)

    if brier_margin is not None and (b_img is not None) and (b_ic is not None):
        # brier 越小越好；要求 imgclin 不比 img 差太多
        choose_ic = choose_ic and (b_ic <= b_img + brier_margin)

    winner = "model_imgclin" if choose_ic else "model_img"
    
    # Record data source for debugging
    data_source = "selection.json" if selection_json is not None and not (math.isnan(u_img) or math.isnan(u_ic)) else "checkpoint"
    
    info = {
        "uno_img": u_img,
        "uno_imgclin": u_ic,
        "uno_margin": margin,
        "brier_img": b_img,
        "brier_imgclin": b_ic,
        "brier_margin": brier_margin,
        "winner": winner,
        "data_source": data_source,  # Track where we got the values from
    }
    return winner, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--brier_margin", type=float, default=None)
    args = ap.parse_args()

    ckpt_root = args.ckpt_root
    fold_dirs = sorted([os.path.join(ckpt_root, d) for d in os.listdir(ckpt_root) if d.startswith("fold_") and os.path.isdir(os.path.join(ckpt_root, d))])

    if not fold_dirs:
        raise ValueError(f"No fold_* dirs under {ckpt_root}")

    all_info = {"ckpt_root": ckpt_root, "margin": args.margin, "brier_margin": args.brier_margin, "folds": []}

    for fd in fold_dirs:
        img_path = os.path.join(fd, "model_img", "best.pt")
        ic_path = os.path.join(fd, "model_imgclin", "best.pt")
        out_best = os.path.join(fd, "best.pt")
        selection_path = os.path.join(fd, "selection.json")

        ckpt_img = load_pt(img_path)
        ckpt_ic = load_pt(ic_path)
        
        # PRIORITY: Read from selection.json first (most reliable source)
        selection_json = None
        if os.path.exists(selection_path):
            try:
                with open(selection_path, "r", encoding="utf-8") as f:
                    selection_json = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to read {selection_path}: {e}")

        winner, info = pick_winner(ckpt_img, ckpt_ic, selection_json, args.margin, args.brier_margin)
        src = ic_path if winner == "model_imgclin" else img_path

        # keep backup
        if os.path.exists(out_best):
            shutil.copy2(out_best, out_best + ".bak")

        shutil.copy2(src, out_best)
        all_info["folds"].append({"fold_dir": fd, **info})

        source_marker = "✓" if info.get("data_source") == "selection.json" else "⚠"
        print(f"{os.path.basename(fd)}: winner={winner} | "
              f"uno_img={info['uno_img']:.4f} uno_imgclin={info['uno_imgclin']:.4f} "
              f"[{source_marker} {info.get('data_source', 'unknown')}]")

    out_json = os.path.join(ckpt_root, "donoharm_selection_margin.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_info, f, indent=2, ensure_ascii=False)

    print("\nSaved selection summary:", out_json)
    print("Note: original fold_k/best.pt has been backed up to best.pt.bak (if existed).")


if __name__ == "__main__":
    main()
