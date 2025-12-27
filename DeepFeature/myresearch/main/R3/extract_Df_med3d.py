#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R3 Step-A: Frozen Med3D/MedicalNet embedding extraction (D^f)

Key fix: Extract penultimate encoder features (512-dim for resnet10/18)
         instead of segmentation head logits (2-dim).
"""

import os
import sys
import json
import argparse
import inspect
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

try:
    import nibabel as nib
except ImportError as e:
    raise ImportError("Please install nibabel: pip install nibabel") from e

try:
    import scipy.ndimage as ndi
except ImportError as e:
    raise ImportError("Please install scipy: pip install scipy") from e


# -------------------------
# Utils: IO & geometry
# -------------------------
def load_nii(path: str) -> Tuple[np.ndarray, Tuple[float, float, float], np.ndarray]:
    img = nib.load(path)
    arr = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()[:3]  # (sx, sy, sz) in nib axis order
    affine = img.affine
    return arr, (float(zooms[0]), float(zooms[1]), float(zooms[2])), affine


def resample_to_spacing(
    vol: np.ndarray,
    src_spacing: Tuple[float, float, float],
    tgt_spacing: Tuple[float, float, float],
    order: int,
) -> np.ndarray:
    # zoom factor = src_spacing / tgt_spacing
    zoom = [src_spacing[i] / tgt_spacing[i] for i in range(3)]
    return ndi.zoom(vol, zoom=zoom, order=order)


def bbox_from_mask(mask: np.ndarray, margin: int) -> Optional[Tuple[slice, slice, slice]]:
    idx = np.where(mask > 0)
    if len(idx[0]) == 0:
        return None
    x0, x1 = int(idx[0].min()), int(idx[0].max())
    y0, y1 = int(idx[1].min()), int(idx[1].max())
    z0, z1 = int(idx[2].min()), int(idx[2].max())
    x0 = max(0, x0 - margin); y0 = max(0, y0 - margin); z0 = max(0, z0 - margin)
    x1 = min(mask.shape[0] - 1, x1 + margin)
    y1 = min(mask.shape[1] - 1, y1 + margin)
    z1 = min(mask.shape[2] - 1, z1 + margin)
    return (slice(x0, x1 + 1), slice(y0, y1 + 1), slice(z0, z1 + 1))


def center_crop_or_pad(vol: np.ndarray, out_shape: Tuple[int, int, int], pad_value: float = 0.0) -> np.ndarray:
    ox, oy, oz = out_shape
    x, y, z = vol.shape

    # pad if needed
    px0 = max(0, (ox - x) // 2); px1 = max(0, ox - x - px0)
    py0 = max(0, (oy - y) // 2); py1 = max(0, oy - y - py0)
    pz0 = max(0, (oz - z) // 2); pz1 = max(0, oz - z - pz0)
    if px0 or px1 or py0 or py1 or pz0 or pz1:
        vol = np.pad(vol, ((px0, px1), (py0, py1), (pz0, pz1)), mode="constant", constant_values=pad_value)

    # crop center
    x, y, z = vol.shape
    sx = (x - ox) // 2
    sy = (y - oy) // 2
    sz = (z - oz) // 2
    return vol[sx:sx + ox, sy:sy + oy, sz:sz + oz]


# -------------------------
# Intensity normalization
# -------------------------
def norm_med3d(vol: np.ndarray) -> np.ndarray:
    # Per Med3D paper: truncate 0.5–99.5 percentiles then z-score (per volume).
    lo, hi = np.percentile(vol, [0.5, 99.5])
    vol = np.clip(vol, lo, hi)
    m = float(vol.mean())
    s = float(vol.std())
    if s < 1e-6:
        s = 1.0
    vol = (vol - m) / s
    return vol.astype(np.float32)


def norm_lung_window(vol: np.ndarray, vmin: float, vmax: float, out_range: str) -> np.ndarray:
    vol = np.clip(vol, vmin, vmax)
    vol = (vol - vmin) / (vmax - vmin)  # [0,1]
    if out_range == "neg1_1":
        vol = vol * 2.0 - 1.0           # [-1,1]
    return vol.astype(np.float32)


# -------------------------
# MedicalNet model loading
# -------------------------
def load_checkpoint_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "net", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unrecognized checkpoint format: {type(ckpt)}")

    sd = {}
    for k, v in ckpt.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        sd[nk] = v
    return sd


def infer_in_channels(sd: Dict[str, torch.Tensor]) -> int:
    # Find first conv weight
    cand = [k for k in sd.keys() if k.endswith("conv1.weight") or k.endswith("conv1.0.weight")]
    if not cand:
        # fallback: any 5D weight with second dim = channels?
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.ndim == 5:
                return int(v.shape[1])
        return 1
    w = sd[cand[0]]
    return int(w.shape[1])


def build_medicalnet_resnet(medicalnet_root: str, arch: str, in_channels: int) -> nn.Module:
    # Expect you have cloned Tencent/MedicalNet. We import its model definition.
    sys.path.insert(0, medicalnet_root)

    try:
        from models import resnet as mn_resnet  # type: ignore
    except Exception as e:
        raise ImportError(
            f"Failed to import MedicalNet models from {medicalnet_root}. "
            f"Make sure this script can import 'models/resnet.py' from that repo."
        ) from e

    if not hasattr(mn_resnet, arch):
        raise ValueError(f"MedicalNet models.resnet has no function '{arch}'.")

    fn = getattr(mn_resnet, arch)
    
    # ---------------------------------------------------------
    # FIX: 直接 Inspect 底层的 ResNet 类，而不是 wrapper 函数
    # ---------------------------------------------------------
    # 获取 ResNet 类定义
    ResNetClass = getattr(mn_resnet, "ResNet")
    resnet_sig = inspect.signature(ResNetClass.__init__)
    
    # 准备所有可能的参数池
    candidates = {
        "sample_input_D": 128,
        "sample_input_H": 128,
        "sample_input_W": 128,
        "num_seg_classes": 2,
        "num_classes": 2,
        "shortcut_type": 'B',
        "no_max_pool": False,
        "n_input_channels": in_channels, # MedicalNet 常用名
        "input_channels": in_channels,   # 备用名
        "in_channels": in_channels       # 备用名
    }

    # 智能筛选：只传入 ResNet.__init__ 真正接受的参数
    kwargs = {}
    for k, v in candidates.items():
        if k in resnet_sig.parameters:
            kwargs[k] = v
    
    # 特殊兜底：如果 ResNet 定义里包含 **kwargs，则把关键参数强行塞进去
    # 因为有些版本的 parameters 可能只显示 **kwargs 而隐藏具体参数
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in resnet_sig.parameters.values())
    if has_kwargs:
        # 强制补充 MedicalNet 必须的参数，防止它们没在 signature 里显式声明
        required_keys = ["sample_input_D", "sample_input_H", "sample_input_W", "num_seg_classes", "n_input_channels"]
        for k in required_keys:
            if k not in kwargs and k in candidates:
                kwargs[k] = candidates[k]

    print(f"[Model Build] Calling {arch} with args: {kwargs.keys()}")
    
    model = fn(**kwargs)

    # Remove classification head (if exists)
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    
    # Remove segmentation head (if exists) - CRITICAL FIX
    if hasattr(model, "conv_seg"):
        model.conv_seg = nn.Identity()

    return model


# -------------------------
# KEY FIX: Extract encoder embedding, NOT head logits
# -------------------------
@torch.no_grad()
def forward_backbone_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Extract penultimate embedding from MedicalNet ResNet backbone.
    
    This function manually forwards through the encoder stages (layer1-4)
    and applies global average pooling, bypassing any segmentation or
    classification head.
    
    Returns:
        (N, 512) for resnet10/18
        (N, 2048) for resnet50/101/152/200
    """
    # Handle DataParallel wrapper
    m = model.module if hasattr(model, "module") else model
    
    # ---- Stem ----
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    
    # MaxPool (some implementations may not have it or name it differently)
    if hasattr(m, "maxpool") and m.maxpool is not None:
        x = m.maxpool(x)
    
    # ---- ResNet stages ----
    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)
    x = m.layer4(x)
    
    # ---- Global Average Pooling over spatial dims (D, H, W) ----
    # layer4 output shape: (N, C, D', H', W')
    # For resnet10/18: C=512
    # For resnet50/101: C=2048
    x = x.mean(dim=(2, 3, 4))  # (N, C)
    
    return x


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_csv", required=True)
    ap.add_argument("--medicalnet_root", required=True, help="Path to cloned Tencent/MedicalNet repo root")
    ap.add_argument("--ckpt", required=True, help="Path to pretrained .pth (e.g., resnet_10_23dataset.pth)")
    ap.add_argument("--arch", default="resnet10", help="resnet10/resnet18/resnet34/resnet50/...")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--target_spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--input_size", type=int, default=128)

    ap.add_argument("--use_mask_crop", action="store_true", help="Crop to mask bbox (recommended if mask is tumor/ROI)")
    ap.add_argument("--crop_margin_vox", type=int, default=8)

    ap.add_argument("--norm", choices=["med3d", "lung_window"], default="med3d")
    ap.add_argument("--window_min", type=float, default=-1000.0)
    ap.add_argument("--window_max", type=float, default=400.0)
    ap.add_argument("--out_range", choices=["0_1", "neg1_1"], default="0_1", help="Only used for lung_window")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.images_csv)
    assert {"PatientID", "image_path", "mask_path"}.issubset(df.columns), "images.csv must have PatientID,image_path,mask_path"
    pids = df["PatientID"].tolist()

    # Load ckpt and build model
    sd = load_checkpoint_state_dict(args.ckpt)
    ckpt_in_ch = infer_in_channels(sd)

    model = build_medicalnet_resnet(args.medicalnet_root, args.arch, ckpt_in_ch)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(args.device)

    print(f"[Model] arch={args.arch} | ckpt_in_channels={ckpt_in_ch} | device={args.device}")
    if missing:
        print(f"[Model] missing keys (ok if head removed): {len(missing)}")
    if unexpected:
        print(f"[Model] unexpected keys: {len(unexpected)}")

    # -------------------------
    # KEY FIX: Infer embedding dimension using dummy input
    # -------------------------
    out_shape = (args.input_size, args.input_size, args.input_size)
    tgt_spacing = tuple(float(x) for x in args.target_spacing)
    
    with torch.no_grad():
        dummy = torch.zeros(1, ckpt_in_ch, *out_shape, device=args.device)
        emb_dim = forward_backbone_embedding(model, dummy).shape[1]
    
    print(f"[Model] Encoder embedding dim = {emb_dim}")
    
    # Sanity check: embedding should be 512 or 2048, NOT 2
    if emb_dim <= 8:
        raise RuntimeError(
            f"Embedding dim = {emb_dim}, expected 512 (resnet10/18) or 2048 (resnet50+). "
            f"You may still be extracting head/logits instead of encoder features. "
            f"Check that forward_backbone_embedding() is being used correctly."
        )

    out_embs = []
    bad = 0

    for i, row in df.iterrows():
        pid = row["PatientID"]
        ip = row["image_path"]
        mp = row["mask_path"]

        try:
            vol, sp, aff = load_nii(ip)

            # Optional: load mask (for crop)
            mask = None
            if args.use_mask_crop:
                mask, sp_m, aff_m = load_nii(mp)
                if mask.shape != vol.shape:
                    raise ValueError(f"{pid}: image/mask shape mismatch {vol.shape} vs {mask.shape}")

            # Resample image (& mask) to target spacing
            if sp != tgt_spacing:
                vol = resample_to_spacing(vol, sp, tgt_spacing, order=1)
                if mask is not None:
                    mask = resample_to_spacing(mask, sp, tgt_spacing, order=0)

            # Crop to mask bbox
            if mask is not None:
                bb = bbox_from_mask(mask, margin=args.crop_margin_vox)
                if bb is not None:
                    vol = vol[bb]
                # If bb is None -> fallback to full volume
            
            # Normalize
            if args.norm == "med3d":
                vol = norm_med3d(vol)
            else:
                vol = norm_lung_window(vol, args.window_min, args.window_max, args.out_range)

            # Resize to fixed cube
            vol = center_crop_or_pad(vol, out_shape, pad_value=0.0)

            # To tensor: (1, C, D, H, W)
            x = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)

            # Channel replication fallback if ckpt wants 3ch
            if ckpt_in_ch == 3 and x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1, 1)

            x = x.to(args.device, non_blocking=True)

            # -------------------------
            # KEY FIX: Use new embedding extraction function
            # -------------------------
            emb = forward_backbone_embedding(model, x)  # (1, 512) or (1, 2048)
            emb = emb.squeeze(0).detach().cpu().numpy().astype(np.float32)
            
            if not np.isfinite(emb).all():
                raise FloatingPointError("NaN/Inf in embedding")
            
            out_embs.append(emb)
            
        except Exception as e:
            bad += 1
            print(f"[WARN] pid={pid} failed: {repr(e)}")
            # Keep alignment by inserting NaNs with CORRECT dimension
            out_embs.append(np.full((emb_dim,), np.nan, dtype=np.float32))

        if (i + 1) % 25 == 0:
            print(f"[Progress] {i+1}/{len(df)}")

    embs = np.stack(out_embs, axis=0)  # (N, emb_dim)
    np.save(os.path.join(args.out_dir, "Df_embeddings.npy"), embs)

    pd.DataFrame({"PatientID": pids}).to_csv(os.path.join(args.out_dir, "Df_pids.csv"), index=False)
    
    # Also save as CSV for direct R3 ingestion
    feat_cols = [f"deep_{j}" for j in range(emb_dim)]
    df_out = pd.DataFrame(embs, columns=feat_cols)
    df_out.insert(0, "PatientID", pids)
    df_out.to_csv(os.path.join(args.out_dir, "Df_features.csv"), index=False)

    params = {
        "images_csv": os.path.abspath(args.images_csv),
        "ckpt": os.path.abspath(args.ckpt),
        "arch": args.arch,
        "ckpt_in_channels": ckpt_in_ch,
        "target_spacing": list(tgt_spacing),
        "input_size": args.input_size,
        "use_mask_crop": bool(args.use_mask_crop),
        "crop_margin_vox": args.crop_margin_vox,
        "norm": args.norm,
        "lung_window": {"min": args.window_min, "max": args.window_max, "out_range": args.out_range},
        "device": args.device,
        "n_failed": bad,
        "embedding_dim": emb_dim,  # NEW: record actual dim
        "embedding_shape": list(embs.shape),
    }
    with open(os.path.join(args.out_dir, "Df_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    
    # Sanity stats
    sanity = {
        "n_patients": len(pids),
        "n_failed": bad,
        "embedding_dim": emb_dim,
        "embedding_shape": list(embs.shape),
        "emb_mean": float(np.nanmean(embs)),
        "emb_std": float(np.nanstd(embs)),
        "emb_min": float(np.nanmin(embs)),
        "emb_max": float(np.nanmax(embs)),
        "n_nan_patients": int(np.isnan(embs).any(axis=1).sum()),
    }
    with open(os.path.join(args.out_dir, "Df_sanity.json"), "w") as f:
        json.dump(sanity, f, indent=2)

    print(f"\n[Done] Saved to {args.out_dir}")
    print(f"[Done] embeddings shape = {embs.shape}")
    print(f"[Done] embedding dim = {emb_dim}")
    print(f"[Done] failed = {bad}")
    
    # Final assertion
    if embs.shape[1] != emb_dim:
        raise RuntimeError(f"Final shape mismatch: {embs.shape} vs expected dim={emb_dim}")
    if embs.shape[1] <= 8:
        raise RuntimeError(f"CRITICAL: Final embedding dim = {embs.shape[1]}, still wrong!")
    
    print(f"[Done] ✓ Embedding extraction successful!")


if __name__ == "__main__":
    main()


# ============================================================
# Run command example:
# ============================================================
# python extract_Df_med3d.py \
#   --images_csv /home/lichengze/Research/DeepFeature/myresearch/main/R3/images.csv \
#   --medicalnet_root /home/lichengze/Research/DeepFeature/MedicalNet \
#   --ckpt /home/lichengze/Research/DeepFeature/MedicalNet/pretrain/resnet_10_23dataset.pth \
#   --arch resnet10 \
#   --out_dir /home/lichengze/Research/DeepFeature/myresearch/main/R3 \
#   --norm lung_window \
#   --window_min -1000 --window_max 400 \
#   --out_range neg1_1 \
#   --target_spacing 1 1 3 \
#   --input_size 128 \
#   --use_mask_crop