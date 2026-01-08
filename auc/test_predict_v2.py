#!/bin/bash
# =============================================================================
# test_predict_v2.sh - 使用修复版模型测试单患者预测
# =============================================================================
#
# 测试目标：
# 1. 验证概率不会被截断为0%或100%
# 2. 验证解释中的delta不全为0
# 3. 验证预测结果临床合理
#
# =============================================================================

set -e

echo "=========================================="
echo "修复版模型测试 - 单患者预测 (v2.0)"
echo "=========================================="

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate zip_gbsa_infer_py310

# 测试患者
PATIENT_ID="LUNG1-006"
WORK_DIR="/home/lichengze/Research/auc/test_prediction_v2_${PATIENT_ID}"
SCRIPT_DIR="/home/lichengze/Research/auc"

mkdir -p $WORK_DIR
cd $WORK_DIR

echo ""
echo "测试患者: $PATIENT_ID"
echo "工作目录: $WORK_DIR"

# =============================================================================
# Step 1: 准备测试数据
# =============================================================================
echo ""
echo "=========================================="
echo "Step 1: 准备测试数据"
echo "=========================================="

python3 << 'PREPARE_PY'
import pandas as pd
import json
import numpy as np

# 编码映射
GENDER_MAP = {"male": 1, "female": 2}
HISTOLOGY_MAP = {
    "large cell": 0,
    "nos": 1,
    "squamous cell": 2,
    "adenocarcinoma": 3,
    "other": 4,
}

def safe_int(x, default=0):
    if pd.isna(x):
        return default
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        if np.isnan(x):
            return default
        return int(x)
    return default

def map_gender(x):
    if pd.isna(x):
        return 0
    if isinstance(x, (int, np.integer)):
        return int(x)
    x_str = str(x).strip().lower()
    return GENDER_MAP.get(x_str, 0)

def map_histology(x):
    if pd.isna(x):
        return 0
    if isinstance(x, (int, np.integer)):
        return int(x)
    x_str = str(x).strip().lower()
    if "squamous" in x_str:
        return 2
    if "adenocarcinoma" in x_str or "adeno" in x_str:
        return 3
    if "large cell" in x_str:
        return 0
    if "nos" in x_str:
        return 1
    return 0

# 加载数据
clinical = pd.read_csv("/home/lichengze/Research/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
radiomics = pd.read_csv("/home/lichengze/Research/com/Tb_features_1688_from_yaml.csv")

patient_id = "LUNG1-006"
clin_row = clinical[clinical['PatientID'] == patient_id].iloc[0]
rad_row = radiomics[radiomics['PatientID'] == patient_id]

# 创建临床JSON
clinical_data = {
    "PatientID": patient_id,
    "age": safe_int(clin_row['age'], 65),
    "gender": map_gender(clin_row['gender']),
    "Histology": map_histology(clin_row['Histology']),
    "clinical.T.Stage": safe_int(clin_row.get('clinical.T.Stage', 0), 0),
    "Clinical.N.Stage": safe_int(clin_row.get('Clinical.N.Stage', 0), 0),
    "Clinical.M.Stage": safe_int(clin_row.get('Clinical.M.Stage', 0), 0),
    "Overall.Stage": safe_int(clin_row.get('Overall.Stage', 0), 0)
}

with open(f"{patient_id}_clinical.json", 'w') as f:
    json.dump(clinical_data, f, indent=2)

print(f"✅ 创建: {patient_id}_clinical.json")
print(f"   临床数据: age={clinical_data['age']}, gender={clinical_data['gender']}, Histology={clinical_data['Histology']}")

# 保存radiomics
rad_row.to_csv(f"{patient_id}_radiomics.csv", index=False)
print(f"✅ 创建: {patient_id}_radiomics.csv")
print(f"   Radiomics特征: {rad_row.shape[1]-1} 个")

# 显示实际结局
print(f"\n📊 实际结局 (参考):")
print(f"   事件: {bool(clin_row['deadstatus.event'])}")
print(f"   生存时间: {clin_row['Survival.time']} 天")
if clin_row['Survival.time'] >= 730:
    actual_2y = "生存超过2年" if not clin_row['deadstatus.event'] else "2年后死亡"
else:
    actual_2y = "2年内死亡" if clin_row['deadstatus.event'] else "2年内删失"
print(f"   2年状态: {actual_2y}")
PREPARE_PY

# =============================================================================
# Step 2: 运行预测
# =============================================================================
echo ""
echo "=========================================="
echo "Step 2: 运行修复版预测"
echo "=========================================="

python $SCRIPT_DIR/surv_2y_system_v2.py predict \
  --model_joblib $SCRIPT_DIR/gbsa_2y_export_v2/gbsa_2y_model_v2.joblib \
  --patient_clinical_json ${PATIENT_ID}_clinical.json \
  --patient_radiomics_csv ${PATIENT_ID}_radiomics.csv \
  --out_dir predictions_v2 \
  --explain_topk 10

# =============================================================================
# Step 3: 验证修复效果
# =============================================================================
echo ""
echo "=========================================="
echo "Step 3: 验证修复效果"
echo "=========================================="

python3 << 'VERIFY_PY'
import json
import pandas as pd

# 加载预测结果
with open("predictions_v2/LUNG1-006_2y_prediction_v2.json") as f:
    pred = json.load(f)

# 加载解释结果
expl = pd.read_csv("predictions_v2/LUNG1-006_explanation_top10_v2.csv")

print("=" * 60)
print("修复验证报告")
print("=" * 60)

# 验证1: 概率不为0或1
p2 = pred['pred_survival_2y']
print(f"\n[验证1] 概率范围检查:")
print(f"  校准后概率: {p2:.4f} ({p2*100:.1f}%)")
if 0.01 <= p2 <= 0.99:
    print(f"  ✅ 通过: 概率在合理范围 [1%, 99%]")
else:
    print(f"  ❌ 失败: 概率超出范围")

# 验证2: 解释delta不全为0
deltas = expl['delta_surv2y_if_set_to_ref'].values
non_zero = (deltas != 0).sum()
print(f"\n[验证2] 解释有效性检查:")
print(f"  非零delta数量: {non_zero}/{len(deltas)}")
if non_zero > 0:
    print(f"  ✅ 通过: 解释系统正常工作")
    print(f"  最大正delta: {deltas.max():.4f} (改为参考值可提高生存概率)")
    print(f"  最小负delta: {deltas.min():.4f} (改为参考值会降低生存概率)")
else:
    print(f"  ❌ 失败: 所有delta为0，解释失效")

# 验证3: 临床合理性
print(f"\n[验证3] 临床合理性检查:")
print(f"  原始概率: {pred['pred_survival_2y_raw']:.4f}")
print(f"  校准概率: {pred['pred_survival_2y']:.4f}")
print(f"  风险评分: {pred['pred_risk']:.4f}")
print(f"  百分位: {pred['risk_percentile_vs_train']*100:.1f}%")

# 该患者实际在2年内死亡(173天)，预期低生存概率
if p2 < 0.5:
    print(f"  ✅ 合理: 模型预测低生存概率，与实际结局一致")
else:
    print(f"  ⚠️ 注意: 模型预测高生存概率，但患者实际2年内死亡")

print("\n" + "=" * 60)
print(f"解释: {pred.get('interpretation', 'N/A')}")
print("=" * 60)
VERIFY_PY

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "=========================================="
echo "✅ 测试完成！"
echo "=========================================="
echo ""
echo "输出文件:"
echo "  - predictions_v2/${PATIENT_ID}_2y_prediction_v2.json"
echo "  - predictions_v2/${PATIENT_ID}_explanation_top10_v2.csv"
echo ""
echo "查看结果:"
echo "  cat predictions_v2/${PATIENT_ID}_2y_prediction_v2.json"
echo "  cat predictions_v2/${PATIENT_ID}_explanation_top10_v2.csv"
echo ""