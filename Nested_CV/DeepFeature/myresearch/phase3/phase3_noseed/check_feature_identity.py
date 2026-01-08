# check_feature_identity.py
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

def load_mod(py_path, name):
    spec = importlib.util.spec_from_file_location(name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

BASE = load_mod("phase3_fusion_age_gender_TNM_stage.py", "base")  # 你旧的 allfeatures（不含 histology）
HIST = load_mod("phase3_fusion_allfeatures.py", "hist")  # 新的 histology 版本

exp_dir = Path("/home/lichengze/Research/DeepFeature/myresearch/phase3_outputs/lr_6.2e-4")
fold_dir = exp_dir / "fold_0"
t_pids = pd.read_csv(fold_dir/"train_pids.csv", header=None)[0].astype(str).tolist()
v_pids = pd.read_csv(fold_dir/"val_pids.csv", header=None)[0].astype(str).tolist()

csv_path = "/home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
df_base = BASE.load_clinical_data(csv_path)
df_hist = HIST.load_clinical_data(csv_path)

# 先确保 clinical table 一致（index 对齐后比关键列）
common = df_base.index.intersection(df_hist.index)
for col in ["age","gender","t_stage","n_stage","m_stage","overall_stage"]:
    a = df_base.loc[common, col].values
    b = df_hist.loc[common, col].values
    # Skip gender as it may contain strings, check others numerically
    if col == "gender":
        # Check if values are identical (as strings)
        match = np.sum(a == b)
        print(f"{col}: {match}/{len(a)} identical")
    else:
        # Convert to float and check difference
        a_float = pd.to_numeric(a, errors='coerce')
        b_float = pd.to_numeric(b, errors='coerce')
        max_diff = np.nanmax(np.abs(a_float - b_float))
        print(f"{col}: max_diff = {max_diff}")

Xb_tr, Xb_va, *_ = BASE.build_clinical_features(df_base, t_pids, v_pids)
Xh_tr, Xh_va, *_ = HIST.build_clinical_features(df_hist, t_pids, v_pids)

print("Xb_tr", Xb_tr.shape, "Xh_tr", Xh_tr.shape)
print("max|diff train first6| =", np.max(np.abs(Xb_tr - Xh_tr[:, :Xb_tr.shape[1]])))
print("max|diff val first6|   =", np.max(np.abs(Xb_va - Xh_va[:, :Xb_va.shape[1]])))
