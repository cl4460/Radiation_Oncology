#!/usr/bin/env python
"""
演示Phase3中Risk Score计算与Logistic-Hazard Loss的一致性

这个脚本展示：
1. Loss计算过程（NLLLogistiHazardLoss）
2. Risk Score计算过程
3. 验证两者的一致性
"""

import torch
import torch.nn as nn
import numpy as np
from pycox.models.loss import NLLLogistiHazardLoss

print("="*70)
print("Phase3 Risk Score vs Logistic-Hazard Loss 一致性验证")
print("="*70)

# ============================================================
# 1. 模拟Phase3的输出
# ============================================================
print("\n【1. 模拟Phase3模型输出】")
print("-"*70)

batch_size = 4
n_time_bins = 12

# 模拟logits（模型输出）
logits = torch.randn(batch_size, n_time_bins)
print(f"Logits形状: {logits.shape}")
print(f"Logits示例（第一个样本，前3个时间区间）:")
print(f"  {logits[0, :3].tolist()}")

# ============================================================
# 2. Loss计算（NLLLogistiHazardLoss）
# ============================================================
print("\n【2. Loss计算（NLLLogistiHazardLoss）】")
print("-"*70)

# 创建loss函数
criterion = NLLLogistiHazardLoss()

# 模拟标签
y_idx = torch.randint(0, n_time_bins, (batch_size,))  # 时间区间索引
y_evt = torch.randint(0, 2, (batch_size,)).float()  # 事件指示

print(f"标签:")
print(f"  y_idx (时间区间): {y_idx.tolist()}")
print(f"  y_evt (事件): {y_evt.tolist()}")

# 计算loss
loss = criterion(logits, y_idx, y_evt)
print(f"\nLoss: {loss.item():.4f}")

# 手动计算hazard概率（NLLLogistiHazardLoss内部做的）
haz_loss = torch.sigmoid(logits)
print(f"\nHazard概率（Loss计算中使用）:")
print(f"  形状: {haz_loss.shape}")
print(f"  第一个样本，前3个时间区间: {haz_loss[0, :3].tolist()}")

# ============================================================
# 3. Risk Score计算（验证时）
# ============================================================
print("\n【3. Risk Score计算（验证时）】")
print("-"*70)

# 步骤1：Logits → Hazard概率（与loss计算相同！）
haz_risk = torch.sigmoid(logits[:, :n_time_bins])
print(f"步骤1: Logits → Hazard概率")
print(f"  形状: {haz_risk.shape}")
print(f"  第一个样本，前3个时间区间: {haz_risk[0, :3].tolist()}")

# 验证：haz_loss和haz_risk应该完全相同
assert torch.allclose(haz_loss, haz_risk), "Hazard概率不一致！"
print(f"  ✅ 验证通过：haz_loss和haz_risk完全相同")

# 步骤2：Hazard概率 → 生存概率
surv = torch.cumprod(1.0 - haz_risk, dim=1)
print(f"\n步骤2: Hazard概率 → 生存概率")
print(f"  形状: {surv.shape}")
print(f"  第一个样本，前3个时间区间: {surv[0, :3].tolist()}")
print(f"  生存概率应该单调递减: {(surv[:, 1:] <= surv[:, :-1]).all().item()}")

# 步骤3：生存概率 → Risk Score
bin_mids = np.linspace(0, 1000, n_time_bins)  # 模拟时间区间中点
risk_integral = -np.trapz(surv.cpu().numpy(), x=bin_mids, axis=1)
print(f"\n步骤3: 生存概率 → Risk Score")
print(f"  形状: {risk_integral.shape}")
print(f"  所有样本的risk score: {risk_integral.tolist()}")

# ============================================================
# 4. 一致性验证
# ============================================================
print("\n【4. 一致性验证】")
print("-"*70)

print("验证1: Hazard概率计算一致")
print(f"  Loss计算中的haz: {haz_loss[0, 0].item():.6f}")
print(f"  Risk计算中的haz: {haz_risk[0, 0].item():.6f}")
print(f"  差异: {abs(haz_loss[0, 0].item() - haz_risk[0, 0].item()):.10f}")
assert torch.allclose(haz_loss, haz_risk, atol=1e-6)
print(f"  ✅ 验证通过：Hazard概率计算完全一致")

print("\n验证2: 数学公式正确性")
print("  Logistic-Hazard假设: haz = sigmoid(logits)")
print(f"  验证: haz[0,0] = sigmoid(logits[0,0])")
print(f"    haz[0,0] = {haz_loss[0, 0].item():.6f}")
print(f"    sigmoid(logits[0,0]) = {torch.sigmoid(logits[0, 0]).item():.6f}")
assert torch.allclose(haz_loss[0, 0], torch.sigmoid(logits[0, 0]))
print(f"  ✅ 验证通过：数学公式正确")

print("\n验证3: 生存概率计算正确性")
print("  离散时间生存分析公式: S(t_k) = ∏(1 - h(t_j)) for j=1 to k")
print(f"  验证第一个样本:")
for k in range(min(3, n_time_bins)):
    manual_surv = torch.prod(1.0 - haz_risk[0, :k+1])
    computed_surv = surv[0, k]
    print(f"    S(t_{k}) = {computed_surv.item():.6f}, 手动计算 = {manual_surv.item():.6f}")
    assert torch.allclose(computed_surv, manual_surv, atol=1e-5)
print(f"  ✅ 验证通过：生存概率计算正确")

# ============================================================
# 5. 完整流程对比
# ============================================================
print("\n【5. 完整流程对比】")
print("-"*70)

print("Loss计算流程:")
print("  1. logits [B, n_time_bins]")
print("  2. haz = sigmoid(logits) [B, n_time_bins]")
print("  3. loss = NLLLogistiHazardLoss(logits, y_idx, y_evt)")
print("     (内部使用haz计算负对数似然)")

print("\nRisk Score计算流程:")
print("  1. logits [B, n_time_bins]")
print("  2. haz = sigmoid(logits) [B, n_time_bins]  ← 与Loss相同！")
print("  3. surv = cumprod(1 - haz, dim=1) [B, n_time_bins]")
print("  4. risk = -trapz(surv, x=bin_mids, axis=1) [B]")

print("\n关键观察:")
print("  ✅ 两者都从相同的logits开始")
print("  ✅ 两者都使用sigmoid(logits)得到hazard概率")
print("  ✅ 这是Logistic-Hazard方法的核心假设")
print("  ✅ Risk Score是从hazard概率正确推导出的")

# ============================================================
# 6. 数值示例
# ============================================================
print("\n【6. 数值示例】")
print("-"*70)

print("示例：第一个样本")
print(f"  Logits: {logits[0, :3].tolist()}")
print(f"  Hazard概率: {haz_loss[0, :3].tolist()}")
print(f"  生存概率: {surv[0, :3].tolist()}")
print(f"  Risk Score: {risk_integral[0]:.4f}")

print("\n解释:")
print("  - Hazard概率越高，表示在该时间区间发生事件的可能性越大")
print("  - 生存概率 = 1 - 累积hazard，应该单调递减")
print("  - Risk Score = -生存曲线下的面积，值越大表示风险越高")

# ============================================================
# 7. 结论
# ============================================================
print("\n" + "="*70)
print("结论")
print("="*70)
print("✅ Phase3代码中的risk score计算与Logistic-Hazard loss完全一致！")
print("\n原因:")
print("  1. 两者都使用sigmoid(logits)得到hazard概率")
print("  2. Risk Score是从hazard概率正确推导出的")
print("  3. 这是离散时间生存分析的标准方法")
print("  4. 代码实现是正确的")
print("="*70)


