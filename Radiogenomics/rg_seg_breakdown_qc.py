import pandas as pd

seg_check_csv = "/home/lichengze/Research/Radiogenomics/radiogenomics_segmentation_check.csv"      # diagnose_segmentations.py 输出
seg_label_csv = "/home/lichengze/Research/Radiogenomics/radiogenomics_seg_label_count.csv"         # 你的 label 统计输出
phase2_crop_log = "/home/lichengze/Research/Radiogenomics/phase2_outputs/phase2_crop_log.csv"                     # RG Phase2 输出 (133 行)

seg_check = pd.read_csv(seg_check_csv)
seg_label = pd.read_csv(seg_label_csv)
p2 = pd.read_csv(phase2_crop_log)

has_seg_ids = set(seg_check.loc[seg_check["has_seg"] == True, "patient_id"])
used_ids = set(p2["patient_id"])

excluded = sorted(list(has_seg_ids - used_ids))

print("=======================================================")
print("RG segmentation availability breakdown")
print("=======================================================")
print(f"has_seg_file (diagnose): {len(has_seg_ids)}")
print(f"usable_in_phase2:        {len(used_ids)}")
print(f"excluded:                {len(excluded)}")
print("Excluded IDs:", excluded)

# join some metadata for paper-ready table
out = (
    seg_check.merge(seg_label, on="patient_id", how="left")
             .assign(used_in_phase2=lambda d: d["patient_id"].isin(used_ids))
)

# paper-ready summary
summary = out.loc[out["has_seg"] == True, ["patient_id","used_in_phase2","seg_file_count","total_segment_count","all_labels","num_series","modalities","total_dicom"]]
summary.to_csv("rg_has_seg_vs_usable_phase2.csv", index=False)
print("\nSaved: rg_has_seg_vs_usable_phase2.csv")

# excluded table
excluded_df = summary.loc[summary["used_in_phase2"] == False].copy()
excluded_df.to_csv("rg_excluded_11_cases.csv", index=False)
print("Saved: rg_excluded_11_cases.csv")
