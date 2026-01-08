#!/usr/bin/env python
"""
演示DeepSurv MLP的详细计算过程：522 → 128 → 64 → 1

这个脚本展示了：
1. MLP的每一层如何工作
2. 矩阵乘法的具体过程
3. 激活函数和归一化的作用
4. 完整的数值示例
"""

import torch
import torch.nn as nn
import numpy as np

print("="*70)
print("DeepSurv MLP详细计算演示：522 → 128 → 64 → 1")
print("="*70)

# ============================================================
# 1. 创建DeepSurv模型
# ============================================================
print("\n【1. 创建DeepSurv模型】")
print("-"*70)

class DeepSurv(nn.Module):
    def __init__(self, in_features, hidden_1=128, hidden_2=64, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_2, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# 创建模型
model = DeepSurv(in_features=522, hidden_1=128, hidden_2=64, dropout=0.3)
model.eval()  # 设置为评估模式（关闭dropout）

print("模型结构：")
print(f"  输入维度: 522")
print(f"  第一层: 522 → 128")
print(f"  第二层: 128 → 64")
print(f"  输出层: 64 → 1")
print(f"  总参数: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 2. 创建示例输入
# ============================================================
print("\n【2. 创建示例输入】")
print("-"*70)

batch_size = 2  # 2个患者
x = torch.randn(batch_size, 522)  # 随机输入

print(f"输入形状: {x.shape}")
print(f"  批次大小: {batch_size} (2个患者)")
print(f"  特征维度: 522 (512维图像特征 + 10维临床特征)")
print(f"\n输入数据示例（前5个特征）:")
print(f"  患者1: {x[0, :5].tolist()}")
print(f"  患者2: {x[1, :5].tolist()}")

# ============================================================
# 3. 逐层计算（手动演示）
# ============================================================
print("\n【3. 逐层计算过程】")
print("-"*70)

# 获取每一层的权重和偏置
layers = list(model.network.children())
linear1 = layers[0]  # Linear(522, 128)
layernorm1 = layers[1]  # LayerNorm(128)
relu1 = layers[2]  # ReLU
dropout1 = layers[3]  # Dropout
linear2 = layers[4]  # Linear(128, 64)
layernorm2 = layers[5]  # LayerNorm(64)
relu2 = layers[6]  # ReLU
dropout2 = layers[7]  # Dropout
linear3 = layers[8]  # Linear(64, 1)

print("\n【第一层：522 → 128】")
print("-"*70)

# 第一层：Linear
x1 = linear1(x)
print(f"1. Linear层 (522 → 128):")
print(f"   输入形状: {x.shape}")
print(f"   权重形状: {linear1.weight.shape} (522 × 128)")
print(f"   偏置形状: {linear1.bias.shape} (128)")
print(f"   输出形状: {x1.shape}")
print(f"   计算: y = xW₁ + b₁")
print(f"   [2, 522] × [522, 128] + [128] = [2, 128]")
print(f"   参数数量: {linear1.weight.numel() + linear1.bias.numel():,}")

# LayerNorm
x2 = layernorm1(x1)
print(f"\n2. LayerNorm层:")
print(f"   输入形状: {x1.shape}")
print(f"   对每个样本的128个特征进行归一化")
print(f"   输出形状: {x2.shape}")
print(f"   计算: y = (x - μ) / √(σ² + ε)")

# ReLU
x3 = relu1(x2)
print(f"\n3. ReLU激活函数:")
print(f"   输入形状: {x2.shape}")
print(f"   将负数变为0: y = max(0, x)")
print(f"   输出形状: {x3.shape}")
print(f"   负数数量: {(x3 < 0).sum().item()} (应该为0)")

# Dropout (评估模式下不起作用)
x4 = dropout1(x3)
print(f"\n4. Dropout层 (评估模式，不起作用):")
print(f"   输入形状: {x3.shape}")
print(f"   训练时: 随机将30%的神经元设为0")
print(f"   评估时: 不改变输入")
print(f"   输出形状: {x4.shape}")

print("\n【第二层：128 → 64】")
print("-"*70)

# 第二层：Linear
x5 = linear2(x4)
print(f"5. Linear层 (128 → 64):")
print(f"   输入形状: {x4.shape}")
print(f"   权重形状: {linear2.weight.shape} (128 × 64)")
print(f"   偏置形状: {linear2.bias.shape} (64)")
print(f"   输出形状: {x5.shape}")
print(f"   计算: y = xW₂ + b₂")
print(f"   [2, 128] × [128, 64] + [64] = [2, 64]")
print(f"   参数数量: {linear2.weight.numel() + linear2.bias.numel():,}")

# LayerNorm
x6 = layernorm2(x5)
print(f"\n6. LayerNorm层:")
print(f"   输入形状: {x5.shape}")
print(f"   对每个样本的64个特征进行归一化")
print(f"   输出形状: {x6.shape}")

# ReLU
x7 = relu2(x6)
print(f"\n7. ReLU激活函数:")
print(f"   输入形状: {x6.shape}")
print(f"   将负数变为0: y = max(0, x)")
print(f"   输出形状: {x7.shape}")
print(f"   负数数量: {(x7 < 0).sum().item()} (应该为0)")

# Dropout
x8 = dropout2(x7)
print(f"\n8. Dropout层 (评估模式，不起作用):")
print(f"   输入形状: {x7.shape}")
print(f"   输出形状: {x8.shape}")

print("\n【输出层：64 → 1】")
print("-"*70)

# 输出层：Linear
x9 = linear3(x8)
print(f"9. Linear层 (64 → 1):")
print(f"   输入形状: {x8.shape}")
print(f"   权重形状: {linear3.weight.shape} (64 × 1)")
print(f"   偏置形状: {linear3.bias.shape} (1)")
print(f"   输出形状: {x9.shape}")
print(f"   计算: y = xW₃ + b₃")
print(f"   [2, 64] × [64, 1] + [1] = [2, 1]")
print(f"   参数数量: {linear3.weight.numel() + linear3.bias.numel()}")

print(f"\n最终输出 (log-hazard):")
print(f"   患者1: {x9[0, 0].item():.4f}")
print(f"   患者2: {x9[1, 0].item():.4f}")

# ============================================================
# 4. 使用forward方法（完整流程）
# ============================================================
print("\n【4. 使用forward方法（完整流程）】")
print("-"*70)

output = model(x)
print(f"输入: {x.shape}")
print(f"输出: {output.shape}")
print(f"患者1的risk score: {output[0, 0].item():.4f}")
print(f"患者2的risk score: {output[1, 0].item():.4f}")

# 验证手动计算和forward方法结果一致
assert torch.allclose(x9, output, atol=1e-5), "手动计算和forward方法结果不一致！"
print("\n✅ 验证通过：手动计算和forward方法结果一致")

# ============================================================
# 5. 矩阵乘法详细示例
# ============================================================
print("\n【5. 矩阵乘法详细示例（第一层）】")
print("-"*70)

# 简化示例：只展示第一个患者的前3个特征
print("示例：计算第一个患者的第一层输出（前3个神经元）")
print("-"*70)

# 输入（前3个特征）
x_sample = x[0, :3]  # [3]
print(f"输入特征（前3个）: {x_sample.tolist()}")

# 权重矩阵（前3行，前3列）
W_sample = linear1.weight[:3, :3]  # [3, 3]
print(f"\n权重矩阵（前3×3）:")
print(W_sample)

# 偏置（前3个）
b_sample = linear1.bias[:3]  # [3]
print(f"\n偏置（前3个）: {b_sample.tolist()}")

# 计算
y_sample = torch.matmul(x_sample, W_sample) + b_sample
print(f"\n计算过程:")
print(f"  y₁ = x₁w₁₁ + x₂w₂₁ + x₃w₃₁ + b₁")
print(f"  y₂ = x₁w₁₂ + x₂w₂₂ + x₃w₃₂ + b₂")
print(f"  y₃ = x₁w₁₃ + x₂w₂₃ + x₃w₃₃ + b₃")
print(f"\n结果（前3个神经元）: {y_sample.tolist()}")

# ============================================================
# 6. 参数统计
# ============================================================
print("\n【6. 参数统计】")
print("-"*70)

total_params = 0
for name, param in model.named_parameters():
    num_params = param.numel()
    total_params += num_params
    print(f"{name:30s}: {num_params:>8,} 参数, 形状 {list(param.shape)}")

print(f"\n总参数数量: {total_params:,}")
print(f"内存占用 (float32): {total_params * 4 / 1024:.2f} KB")

# ============================================================
# 7. 维度变换总结
# ============================================================
print("\n【7. 维度变换总结】")
print("-"*70)

print("完整数据流:")
print(f"  输入: [B, 522]")
print(f"    ↓ Linear(522, 128)")
print(f"  [B, 128]")
print(f"    ↓ LayerNorm + ReLU + Dropout")
print(f"  [B, 128]")
print(f"    ↓ Linear(128, 64)")
print(f"  [B, 64]")
print(f"    ↓ LayerNorm + ReLU + Dropout")
print(f"  [B, 64]")
print(f"    ↓ Linear(64, 1)")
print(f"  输出: [B, 1] (log-hazard)")

print("\n" + "="*70)
print("演示完成！")
print("="*70)
print("\n关键理解：")
print("1. MLP = 多层全连接网络")
print("2. 每一层通过矩阵乘法进行维度变换")
print("3. 激活函数引入非线性")
print("4. 归一化和Dropout提高训练稳定性")
print("5. 最终输出是log-hazard，用于Cox PH模型")


