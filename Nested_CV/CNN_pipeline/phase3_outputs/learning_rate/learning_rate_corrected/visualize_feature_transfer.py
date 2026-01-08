#!/usr/bin/env python
"""
可视化演示：为什么Logistic-Hazard训练的CNN特征可以用于DeepSurv

这个脚本展示了：
1. Phase3模型的完整架构
2. 训练时的数据流和反向传播
3. 特征提取时的数据流
4. DeepSurv训练时的数据流
"""

import torch
import torch.nn as nn
import numpy as np

print("="*70)
print("可视化演示：Logistic-Hazard → DeepSurv Feature Transfer")
print("="*70)

# ============================================================
# 1. Phase3模型架构（完整版）
# ============================================================
print("\n【1. Phase3模型架构】")
print("-"*70)

class Phase3Model(nn.Module):
    """Phase3完整模型：Backbone + Clinical Encoder + Head"""
    def __init__(self):
        super().__init__()
        # Backbone: 图像特征提取器
        self.backbone = nn.Sequential(
            nn.Conv3d(2, 64, 3, padding=1),  # 简化版ResNet-10
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, 512)  # 输出512维特征
        )
        
        # Clinical Encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        # Head: 生存预测头（12个时间区间）
        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 12)  # 12个时间区间的logits
        )
    
    def forward(self, x_img, x_clinical):
        f_img = self.backbone(x_img)        # [B, 512] ← ⭐ 这就是我们要提取的！
        f_clin = self.clinical_encoder(x_clinical)  # [B, 64]
        f_combined = torch.cat([f_img, f_clin], dim=1)  # [B, 576]
        logits = self.head(f_combined)      # [B, 12]
        return logits, f_img  # 返回logits和backbone特征（用于演示）

# 创建模型
phase3_model = Phase3Model()

print("Phase3模型结构：")
print("  Input: 图像 [B, 2, 128, 192, 160] + 临床特征 [B, 10]")
print("  ↓")
print("  Backbone (ResNet-10): 提取512维图像特征")
print("  ↓")
print("  Clinical Encoder: 编码临床特征为64维")
print("  ↓")
print("  Concatenate: [512 + 64] = 576维")
print("  ↓")
print("  Head: 576 → 256 → 128 → 12 (时间区间logits)")
print("  ↓")
print("  Output: [B, 12] logits (用于Logistic-Hazard loss)")

# ============================================================
# 2. Phase3训练过程（模拟）
# ============================================================
print("\n【2. Phase3训练过程（Logistic-Hazard Loss）】")
print("-"*70)

# 模拟输入
batch_size = 4
x_img = torch.randn(batch_size, 2, 128, 192, 160)
x_clinical = torch.randn(batch_size, 10)
y_idx = torch.randint(0, 12, (batch_size,))  # 时间区间索引
y_event = torch.randint(0, 2, (batch_size,)).float()  # 事件指示

# 前向传播
logits, backbone_features = phase3_model(x_img, x_clinical)

print(f"前向传播：")
print(f"  Input shape: 图像 {x_img.shape}, 临床 {x_clinical.shape}")
print(f"  Backbone输出: {backbone_features.shape} ← ⭐ 512维特征")
print(f"  Head输出: {logits.shape} ← 12个时间区间的logits")

# 模拟Logistic-Hazard Loss
def logistic_hazard_loss(logits, y_idx, y_event):
    """简化的Logistic-Hazard loss"""
    probs = torch.sigmoid(logits)
    # 简化版：只计算对应时间区间的loss
    loss = -torch.log(probs[range(len(y_idx)), y_idx] + 1e-8) * y_event
    return loss.mean()

loss = logistic_hazard_loss(logits, y_idx, y_event)
print(f"\n计算Loss: {loss.item():.4f} (NLLLogistiHazardLoss)")

print(f"\n反向传播路径：")
print(f"  Loss → Head (Linear 128→12)")
print(f"      → Head (Linear 256→128)")
print(f"      → Head (Linear 576→256)")
print(f"      ├─→ Clinical Encoder (更新64维编码器)")
print(f"      └─→ Backbone (更新512维特征提取器) ← ⭐ 这里！")
print(f"\n  ⚠️ 注意：Backbone的权重被Logistic-Hazard loss更新了")
print(f"  ✅ 但Backbone学到的是'生存相关特征'，不是'12个时间区间预测'")

# ============================================================
# 3. 特征提取过程
# ============================================================
print("\n【3. 特征提取过程（只使用Backbone）】")
print("-"*70)

# 设置为评估模式
phase3_model.eval()

# 只使用backbone提取特征
with torch.no_grad():
    extracted_features = phase3_model.backbone(x_img)

print(f"特征提取：")
print(f"  Input: 图像 {x_img.shape}")
print(f"  ↓")
print(f"  Backbone (frozen, 不更新)")
print(f"  ↓")
print(f"  Output: {extracted_features.shape} ← ⭐ 512维特征（保存为npy）")
print(f"\n  ✅ 只使用backbone，不使用head")
print(f"  ✅ Backbone权重来自Logistic-Hazard训练")
print(f"  ✅ 这些特征被保存，用于后续DeepSurv训练")

# 保存特征（模拟）
features_numpy = extracted_features.cpu().numpy()
print(f"\n保存特征: shape={features_numpy.shape}, dtype={features_numpy.dtype}")

# ============================================================
# 4. DeepSurv模型架构
# ============================================================
print("\n【4. DeepSurv模型架构（Cox PH Loss）】")
print("-"*70)

class DeepSurv(nn.Module):
    """DeepSurv模型：接收512维特征，输出单个risk score"""
    def __init__(self, in_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # 输出单个risk score
        )
    
    def forward(self, x):
        return self.network(x)

# 创建DeepSurv模型
deepsurv_model = DeepSurv(in_features=512 + 10)  # 512维图像特征 + 10维临床特征

print("DeepSurv模型结构：")
print("  Input: 512维图像特征 + 10维临床特征 = 522维")
print("  ↓")
print("  MLP: 522 → 128 → 64 → 1")
print("  ↓")
print("  Output: [B, 1] risk score (log hazard)")

# ============================================================
# 5. DeepSurv训练过程（模拟）
# ============================================================
print("\n【5. DeepSurv训练过程（Cox PH Loss）】")
print("-"*70)

# 加载提取的特征（模拟）
train_features = features_numpy  # [B, 512]
train_clinical = x_clinical.numpy()  # [B, 10]
train_combined = np.hstack([train_features, train_clinical])  # [B, 522]

# 转换为tensor
train_combined_tensor = torch.tensor(train_combined, dtype=torch.float32)

# 模拟survival数据
times = torch.rand(batch_size) * 1000  # 生存时间
events = torch.randint(0, 2, (batch_size,)).float()  # 事件指示

# 前向传播
risk_scores = deepsurv_model(train_combined_tensor)

print(f"前向传播：")
print(f"  Input: 提取的512维特征 + 临床特征 = {train_combined_tensor.shape}")
print(f"  ↓")
print(f"  DeepSurv MLP (trainable, 从零开始训练)")
print(f"  ↓")
print(f"  Output: {risk_scores.shape} ← 单个risk score")

# 模拟Cox PH Loss
def cox_ph_loss(risk_scores, times, events):
    """简化的Cox PH loss"""
    # 简化版：只计算partial likelihood
    risk_exp = torch.exp(risk_scores.squeeze())
    # 找到事件发生的样本
    event_mask = events == 1
    if event_mask.sum() == 0:
        return torch.tensor(0.0)
    
    # 计算每个事件的风险集
    loss = 0.0
    for i in torch.where(event_mask)[0]:
        risk_set = times >= times[i]  # 风险集：时间 >= 当前事件时间
        numerator = risk_scores[i]
        denominator = torch.log(torch.sum(risk_exp[risk_set]) + 1e-8)
        loss += -(numerator - denominator)
    
    return loss / event_mask.sum()

loss_cox = cox_ph_loss(risk_scores, times, events)
print(f"\n计算Loss: {loss_cox.item():.4f} (CoxPHLoss)")

print(f"\n反向传播路径：")
print(f"  Loss → DeepSurv MLP (Linear 64→1)")
print(f"      → DeepSurv MLP (Linear 128→64)")
print(f"      → DeepSurv MLP (Linear 522→128)")
print(f"      → 停止（Phase3 Backbone不参与训练）")
print(f"\n  ✅ 只更新DeepSurv的参数")
print(f"  ✅ Phase3 Backbone完全frozen，不参与训练")

# ============================================================
# 6. 关键对比
# ============================================================
print("\n【6. 关键对比：为什么可以Transfer？】")
print("-"*70)

print("Phase3训练（Logistic-Hazard）：")
print("  ┌─────────────────────────────────────────┐")
print("  │ 图像 → Backbone → 512维特征             │")
print("  │              ↓                          │")
print("  │  Head → 12个时间区间logits              │")
print("  │              ↓                          │")
print("  │  NLLLogistiHazardLoss                   │")
print("  │              ↓                          │")
print("  │  反向传播更新Backbone + Head            │")
print("  └─────────────────────────────────────────┘")
print("  ⭐ Backbone学到：'哪些图像模式与生存相关'")

print("\n特征提取：")
print("  ┌─────────────────────────────────────────┐")
print("  │ 图像 → Backbone (frozen) → 512维特征    │")
print("  │              ↓                          │")
print("  │  保存为npy文件                          │")
print("  └─────────────────────────────────────────┘")
print("  ⭐ 只使用Backbone，不使用Head")

print("\nDeepSurv训练（Cox PH）：")
print("  ┌─────────────────────────────────────────┐")
print("  │ 512维特征 → DeepSurv MLP → risk score   │")
print("  │              ↓                          │")
print("  │  CoxPHLoss                              │")
print("  │              ↓                          │")
print("  │  反向传播只更新DeepSurv MLP             │")
print("  │  (Backbone不参与)                       │")
print("  └─────────────────────────────────────────┘")
print("  ⭐ DeepSurv学到：'如何用特征计算risk score'")

print("\n" + "="*70)
print("【核心理解】")
print("="*70)
print("1. Backbone的权重确实是用Logistic-Hazard训练的")
print("2. 但Backbone学到的是'生存相关特征'，不是'12个时间区间预测'")
print("3. 这些特征对Cox PH模型同样有用（都是survival任务）")
print("4. DeepSurv只需要学习'如何用这些特征计算risk score'")
print("5. 这是标准的Transfer Learning场景")
print("="*70)

# ============================================================
# 7. 类比：ImageNet预训练
# ============================================================
print("\n【7. 类比：ImageNet预训练】")
print("-"*70)

print("ImageNet预训练：")
print("  图像 → ResNet-50 → 1000类logits → CrossEntropyLoss")
print("  ↓")
print("  Transfer到医学图像分类：")
print("  医学图像 → ResNet-50 (frozen) → 2048维特征 → 新的分类头 → 2类logits")
print("\n  为什么可以transfer？")
print("  ✅ ResNet-50学到的是'通用图像特征'（边缘、纹理、形状）")
print("  ✅ 这些特征对任何图像任务都有用")

print("\n你的情况：")
print("  CT图像 → ResNet-10 (Logistic-Hazard) → 512维特征 → Head → 12个时间区间")
print("  ↓")
print("  Transfer到DeepSurv：")
print("  CT图像 → ResNet-10 (frozen) → 512维特征 → DeepSurv MLP → risk score")
print("\n  为什么可以transfer？")
print("  ✅ ResNet-10学到的是'生存相关图像特征'（肿瘤大小、形状、位置）")
print("  ✅ 这些特征对任何survival模型都有用")

print("\n" + "="*70)
print("演示完成！")
print("="*70)



