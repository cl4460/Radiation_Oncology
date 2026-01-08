# make_splits.py
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

# 1. 修改这里为你真实的临床数据路径
CLINICAL_CSV = "/home/lichengze/Research/CNN_pipeline/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
OUTPUT_JSON = "/home/lichengze/Research/DeepFeature/myresearch/main/R0/splits.json"
SEED = 42

# 2. 读取数据
df = pd.read_csv(CLINICAL_CSV)

# 确保 ID 是字符串
# 假设你的 ID 列名是 "PatientID"，事件列是 "deadstatus.event"
# 如果列名不对，请修改这里
id_col = "PatientID"
event_col = "deadstatus.event" 

ids = df[id_col].astype(str).values
y = df[event_col].values

# 3. 生成 5 折划分
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

splits = {}
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(ids, y)):
    fold_name = f"fold_{fold_idx}"
    splits[fold_name] = {
        "train": ids[train_idx].tolist(),
        "val": ids[val_idx].tolist()
    }

# 4. 保存为 JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(splits, f, indent=2)

print(f"成功生成 {OUTPUT_JSON}！包含了 {len(splits)} 个 fold。")