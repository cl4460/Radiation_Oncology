import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from monai.networks.nets import resnet as monai_resnet
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,
    ConcatItemsd, DeleteItemsd, Lambdad
)
from monai.data import Dataset, DataLoader

# 配置
CROP_LOG_CSV = "/home/lichengze/Research/Nested_CV/CNN_pipeline/phase2_outputs/crop_log.csv"
CLINICAL_CSV = "/home/lichengze/Research/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
CKPT_PATH = "/home/lichengze/Research/Enhance/phase3_7e4_output/fixed_split_seed42_lr7e-04/fold_0/best.pt"
OUTPUT_DIR = "/home/lichengze/Research/Enhance/phase3_7e4_output/fixed_split_seed42_lr7e-04"
OUTPUT_CSV = f"{OUTPUT_DIR}/embeddings_512_fold0.csv"

BATCH_SIZE = 8
NUM_WORKERS = 4
GPU_ID = 1
DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")


def fix_path(path_str):
    """Fix paths that may be missing Nested_CV in the middle."""
    if pd.isna(path_str) or not path_str:
        return path_str
    path_str = str(path_str).strip()
    if not Path(path_str).exists() and "/Research/CNN_pipeline/" in path_str:
        fixed = path_str.replace("/Research/CNN_pipeline/", "/Research/Nested_CV/CNN_pipeline/")
        if Path(fixed).exists():
            return fixed
    return path_str


def _replace_bn_with_gn(module, num_groups=8):
    """Replace BatchNorm with GroupNorm (must match training)"""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            num_channels = child.num_features
            groups = min(num_groups, num_channels)
            while num_channels % groups != 0:
                groups = groups // 2
                if groups == 0:
                    groups = 1
                    break
            setattr(module, name, nn.GroupNorm(groups, num_channels))
        else:
            _replace_bn_with_gn(child, num_groups)


# 加载模型
print("="*80)
print("加载模型...")
print("="*80)
ckpt_path = Path(CKPT_PATH)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
print(f"Validation Uno C-index: {ckpt['uno_integral']:.4f}")

# 重建backbone（必须使用 GroupNorm，与训练时一致）
backbone = monai_resnet.resnet10(spatial_dims=3, n_input_channels=2, num_classes=512)
_replace_bn_with_gn(backbone)  # 关键：替换 BatchNorm 为 GroupNorm

# 加载权重
backbone_state = {k.replace('backbone.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('backbone.')}
backbone.load_state_dict(backbone_state, strict=True)
backbone.eval().to(DEVICE)

print("✅ Backbone loaded successfully!")
print()

# 加载数据
print("="*80)
print("加载数据...")
print("="*80)
crop = pd.read_csv(CROP_LOG_CSV)
clin = pd.read_csv(CLINICAL_CSV)

# 查找 patient_id 列
clin_id_col = None
for col in clin.columns:
    if col.lower() in ['patientid', 'patient_id', 'case']:
        clin_id_col = col
        break

if clin_id_col is None:
    raise ValueError(f"clinical_csv missing patient ID column")

clin_renamed = clin.rename(columns={clin_id_col: "patient_id"})

# 查找生存时间和事件列
time_col = None
event_col = None
for col in clin_renamed.columns:
    col_lower = col.lower()
    if 'survival' in col_lower and 'time' in col_lower:
        time_col = col
    elif 'dead' in col_lower or 'event' in col_lower:
        event_col = col

if time_col is None or event_col is None:
    raise ValueError(f"clinical_csv missing time/event columns")

# 合并数据
df = crop.merge(clin_renamed[[col for col in ["patient_id", time_col, event_col] if col in clin_renamed.columns]], 
                on="patient_id", how="inner")
df = df.dropna(subset=[time_col, event_col]).copy()
df = df.sort_values("patient_id").reset_index(drop=True)

patient_ids = df["patient_id"].tolist()

# 修复路径
df["out_ct"] = df["out_ct"].apply(fix_path)
df["out_edt"] = df["out_edt"].apply(fix_path)

# 构建 records
records = []
for _, r in df.iterrows():
    ct_path = fix_path(r["out_ct"])
    edt_path = fix_path(r["out_edt"])
    records.append({
        "ct": ct_path,
        "edt": edt_path,
    })

print(f"总患者数: {len(patient_ids)}")
print()

# 构建 transform（与训练时一致）
print("="*80)
print("提取 embeddings...")
print("="*80)
tfm = Compose([
    LoadImaged(keys=["ct", "edt"], image_only=True),
    EnsureChannelFirstd(keys=["ct", "edt"]),
    EnsureTyped(keys=["ct", "edt"], dtype=torch.float32, track_meta=False),
    Lambdad(keys=["ct", "edt"], func=lambda x: torch.nan_to_num(x).clamp_(0, 1)),
    ConcatItemsd(keys=["ct", "edt"], name="image", dim=0),
    DeleteItemsd(keys=["ct", "edt"]),
])

ds = Dataset(records, transform=tfm)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# 提取 embeddings
all_embeddings = []
all_patient_ids = []
idx = 0

with torch.no_grad():
    for batch in tqdm(dl, desc="Extracting embeddings"):
        img = batch["image"].to(DEVICE)
        emb = backbone(img).detach().cpu().numpy()  # (B, 512)
        batch_size = emb.shape[0]
        all_patient_ids.extend(patient_ids[idx:idx+batch_size])
        all_embeddings.append(emb)
        idx += batch_size

# 合并
all_embeddings = np.concatenate(all_embeddings, axis=0)
print(f"Extracted embeddings shape: {all_embeddings.shape}")
print()

# 保存为 CSV
print("="*80)
print("保存结果...")
print("="*80)
out_df = pd.DataFrame(all_embeddings, columns=[f"d{i}" for i in range(512)])
out_df.insert(0, "PatientID", all_patient_ids)

# 确保输出目录存在
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ 已保存到: {OUTPUT_CSV}")
print(f"   格式: PatientID, d0, d1, ..., d511")
print(f"   患者数: {len(all_patient_ids)}")
print(f"   维度: {all_embeddings.shape[1]}")
print("="*80)
