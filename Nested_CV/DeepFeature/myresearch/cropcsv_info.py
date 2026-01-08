# check_phase2_output.py
import SimpleITK as sitk
import numpy as np
import pandas as pd

crop_log = pd.read_csv("/home/lichengze/Research/DeepFeature/myresearch/phase2_outputs/crop_log.csv")

# Check 5 random cases
for i in range(5):
    row = crop_log.sample(1).iloc[0]
    ct = sitk.GetArrayFromImage(sitk.ReadImage(row['out_ct']))
    
    print(f"\n{row['patient_id']}:")
    print(f"  Min:  {ct.min():.3f}")
    print(f"  Max:  {ct.max():.3f}")
    print(f"  Mean: {ct.mean():.3f}")
    print(f"  Std:  {ct.std():.3f}")
    print(f"  p01:  {np.percentile(ct, 1):.3f}")
    print(f"  p99:  {np.percentile(ct, 99):.3f}")