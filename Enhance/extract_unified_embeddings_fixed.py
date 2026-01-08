# extract_unified_embeddings_fixed.py
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConcatItemsd, DeleteItemsd, Lambdad
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import resnet10
import torch.nn as nn


def build_backbone(in_channels=2, embedding_dim=512):
    backbone = resnet10(
        spatial_dims=3,
        n_input_channels=in_channels,
        num_classes=embedding_dim,
    )
    return backbone


def replace_bn_with_gn(module, num_groups=16):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features)
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups=num_groups)
    return module


def fix_path(path_str):
    """Fix paths that may be missing Nested_CV in the middle."""
    if pd.isna(path_str) or not path_str:
        return path_str
    path_str = str(path_str).strip()
    # If path doesn't exist, try adding Nested_CV after /Research/
    if not Path(path_str).exists() and "/Research/CNN_pipeline/" in path_str:
        fixed = path_str.replace("/Research/CNN_pipeline/", "/Research/Nested_CV/CNN_pipeline/")
        if Path(fixed).exists():
            return fixed
    return path_str


def load_ckpt_backbone(backbone, ckpt_path):
    # weights_only=False needed for PyTorch 2.6+ when checkpoint contains numpy objects
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # phase3_0617.py 里通常是 backbone.xxx
    filtered = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
    missing, unexpected = backbone.load_state_dict(filtered, strict=False)
    return ckpt, missing, unexpected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop_log_csv", required=True)
    ap.add_argument("--clinical_csv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=8)  # 减小默认batch_size以避免OOM
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="Device string (e.g., 'cuda:1') or None to use --gpu")
    ap.add_argument("--gpu", type=int, default=None, help="GPU device ID (e.g., 1 for cuda:1). If set, overrides --device")
    args = ap.parse_args()
    
    # 如果指定了 --gpu，设置 CUDA_VISIBLE_DEVICES 并使用 cuda:0（因为只有指定的GPU可见）
    if args.gpu is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device_str = "cuda:0"
    elif args.device is not None:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    crop = pd.read_csv(args.crop_log_csv)
    clin = pd.read_csv(args.clinical_csv)

    # 必须包含 out_ct / out_edt
    # crop_log 使用 patient_id (小写)，与 phase3_0617.py 一致
    need = {"patient_id", "out_ct", "out_edt"}
    miss = need - set(crop.columns)
    if miss:
        raise ValueError(f"crop_log missing columns: {miss}")

    # 查找临床数据中的 patient_id 列（可能是 PatientID, patient_id 等）
    clin_id_col = None
    for col in clin.columns:
        if col.lower() in ['patientid', 'patient_id', 'case']:
            clin_id_col = col
            break
    
    if clin_id_col is None:
        raise ValueError(f"clinical_csv missing patient ID column. Available columns: {list(clin.columns)}")
    
    # 标准化为 patient_id
    clin_renamed = clin.rename(columns={clin_id_col: "patient_id"})
    
    # 查找生存时间和事件列（与 phase3_0617.py 逻辑一致）
    time_col = None
    event_col = None
    for col in clin_renamed.columns:
        col_lower = col.lower()
        if 'survival' in col_lower and 'time' in col_lower:
            time_col = col
        elif 'dead' in col_lower or 'event' in col_lower:
            event_col = col
    
    if time_col is None or event_col is None:
        raise ValueError(f"clinical_csv missing time/event columns. Time: {time_col}, Event: {event_col}")
    
    # 合并数据
    df = crop.merge(clin_renamed[[col for col in ["patient_id", time_col, event_col] if col in clin_renamed.columns]], 
                    on="patient_id", how="inner")
    df = df.dropna(subset=[time_col, event_col]).copy()
    df = df.sort_values("patient_id").reset_index(drop=True)

    # 保存 patient_id 列表，用于后续匹配
    patient_ids = df["patient_id"].tolist()
    
    # 修复路径（CSV 中的路径可能缺少 Nested_CV）
    df["out_ct"] = df["out_ct"].apply(fix_path)
    df["out_edt"] = df["out_edt"].apply(fix_path)
    
    records = []
    for _, r in df.iterrows():
        ct_path = fix_path(r["out_ct"])
        edt_path = fix_path(r["out_edt"])
        records.append({
            "ct": ct_path,
            "edt": edt_path,
        })

    # 训练契约：CT+EDT，float32，clamp到[0,1]（如果你训练这么做）
    tfm = Compose([
        LoadImaged(keys=["ct", "edt"], image_only=True),
        EnsureChannelFirstd(keys=["ct", "edt"]),
        EnsureTyped(keys=["ct", "edt"], dtype=torch.float32, track_meta=False),
        Lambdad(keys=["ct", "edt"], func=lambda x: torch.nan_to_num(x).clamp_(0, 1)),
        ConcatItemsd(keys=["ct", "edt"], name="image", dim=0),
        DeleteItemsd(keys=["ct", "edt"]),
    ])

    ds = Dataset(records, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    backbone = build_backbone(in_channels=2, embedding_dim=512)
    backbone = replace_bn_with_gn(backbone, num_groups=16)
    ckpt, missing, unexpected = load_ckpt_backbone(backbone, args.ckpt)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    backbone.to(device).eval()

    all_ids, all_emb = [], []
    idx = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="Extracting"):
            img = batch["image"].to(device)
            emb = backbone(img).detach().cpu().numpy()  # (B,512)
            batch_size = emb.shape[0]
            # 使用索引匹配 patient_id（因为 Dataset 按顺序返回，且 shuffle=False）
            all_ids.extend(patient_ids[idx:idx+batch_size])
            all_emb.append(emb)
            idx += batch_size

    all_emb = np.concatenate(all_emb, axis=0)
    out = pd.DataFrame(all_emb, columns=[f"deep_{i}" for i in range(all_emb.shape[1])])
    # 使用 PatientID（大写）以匹配 step2_multimodal_stack_v1.py 的期望
    out.insert(0, "PatientID", all_ids)
    out.to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print(f"Loaded checkpoint epoch={ckpt.get('epoch')} uno={ckpt.get('uno_integral', None)}")

if __name__ == "__main__":
    main()
