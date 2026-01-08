# test_edt.py
import nibabel as nib
import numpy as np

edt_path = "/home/lichengze/Research/CNN_pipeline/phase2_outputs/LUNG1-001/LUNG1-001_edt_patch.nii.gz"
edt = nib.load(edt_path).get_fdata()

print(f"EDT Phase 2 output:")
print(f"  Min: {edt.min():.6f}")
print(f"  Max: {edt.max():.6f}")
print(f"  Mean: {edt.mean():.6f}")
print(f"  ✅ Expected: Min≈0.0, Max≈1.0")

# 模拟Phase 3 transforms
import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped, Lambda

transforms = Compose([
    LoadImaged(keys=["edt"]),
    EnsureChannelFirstd(keys=["edt"]),
    EnsureTyped(keys=["edt"], dtype=torch.float32, track_meta=False),
    Lambda(func=lambda d: {**d, "edt": torch.clamp(d["edt"], min=0.0, max=1.0)}),
])

data = transforms({"edt": edt_path})
edt_after = data["edt"].numpy()

print(f"\nEDT after Phase 3 transforms:")
print(f"  Min: {edt_after.min():.6f}")
print(f"  Max: {edt_after.max():.6f}")
print(f"  Mean: {edt_after.mean():.6f}")
print(f"  ✅ Should be same as Phase 2")