import numpy as np
import torch
from pathlib import Path
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Lambda

# 测试修复后的transforms
ct_path = "/home/lichengze/Research/CNN_pipeline/phase2_outputs/LUNG1-001/LUNG1-001_ct_patch.nii.gz"

# 修复后的transforms
transforms = Compose([
    LoadImaged(keys=["ct"]),
    EnsureChannelFirstd(keys=["ct"]),
    EnsureTyped(keys=["ct"], dtype=torch.float32, track_meta=False),
    Lambda(func=lambda d: {**d, "ct": torch.nan_to_num(d["ct"], nan=0.0)}),
    Lambda(func=lambda d: {**d, "ct": torch.clamp(d["ct"], min=0.0, max=1.0)}),
])

data = transforms({"ct": ct_path})
ct = data["ct"].numpy()

print("✅ Fixed transforms test:")
print(f"  Min: {ct.min():.6f}")
print(f"  Max: {ct.max():.6f}")
print(f"  Mean: {ct.mean():.6f}")
print(f"\n{'✅ PASS' if 0.0 <= ct.min() < 0.1 and 0.9 < ct.max() <= 1.0 else '❌ FAIL'}")
print(f"Expected: Min≈0.0, Max≈1.0, Mean≈0.3-0.5")