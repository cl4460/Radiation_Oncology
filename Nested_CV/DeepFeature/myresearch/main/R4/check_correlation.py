import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# 1. 设置路径 (请根据你的实际路径修改)
PATH_R3_OOF = "/home/lichengze/Research/DeepFeature/myresearch/main/R3/R3_coxnet_fix1se/oof_pred_lambda_min.csv"
PATH_R4_OOF = "/home/lichengze/Research/DeepFeature/myresearch/main/R4/R4_result/oof_pred_lambda_min.csv"

# 2. 读取
df3 = pd.read_csv(PATH_R3_OOF)
df4 = pd.read_csv(PATH_R4_OOF)

# 3. 合并
# 确保 PatientID 是字符串以防万一
df3["PatientID"] = df3["PatientID"].astype(str)
df4["PatientID"] = df4["PatientID"].astype(str)

merged = df3.merge(df4, on="PatientID", suffixes=("_R3", "_R4"))

# 4. 计算相关性
r_p, _ = pearsonr(merged["risk_R3"], merged["risk_R4"])
r_s, _ = spearmanr(merged["risk_R3"], merged["risk_R4"])

print(f"=== Correlation Check (N={len(merged)}) ===")
print(f"R3 (Deep) vs R4 (Radiomics)")
print(f"Pearson Correlation:  {r_p:.4f}")
print(f"Spearman Correlation: {r_s:.4f}")

if r_p < 0.8:
    print("\n✅ Good News! Correlations are low. Fusion implies high potential gain.")
else:
    print("\n⚠️ High correlation. They learned similar things. Fusion gain might be limited.")