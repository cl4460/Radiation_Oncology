#!/usr/bin/env python3
"""
用最简单的例子解释：为什么随机预测的 C-index 是 0.5 而不是 0.1

通过手工计算一个 4 患者的例子来说明
"""

import numpy as np
from sksurv.metrics import concordance_index_censored

print("="*80)
print("🎓 用最简单的例子理解 C-index")
print("="*80)
print()

# ============================================================================
# 例子 1: 4 个患者的完整计算
# ============================================================================
print("📋 例子 1: 4 个患者，手工计算所有样本对")
print("="*80)
print()

# 真实数据
patients = ['A', 'B', 'C', 'D']
true_times = np.array([100, 200, 300, 400])  # A 最早死亡，D 最晚死亡
true_events = np.array([1, 1, 1, 1], dtype=bool)  # 都发生了事件

print("真实生存时间：")
for i, (p, t) in enumerate(zip(patients, true_times)):
    print(f"  患者 {p}: {t} 天")
print()

# 场景 1: 完美预测
print("场景 1：完美预测（风险分数 = -生存时间）")
print("-" * 80)
perfect_risk = -true_times  # [-100, -200, -300, -400]
print(f"风险分数: {dict(zip(patients, perfect_risk))}")
print()

pairs = [
    ('A', 'B', 100, 200, -100, -200),
    ('A', 'C', 100, 300, -100, -300),
    ('A', 'D', 100, 400, -100, -400),
    ('B', 'C', 200, 300, -200, -300),
    ('B', 'D', 200, 400, -200, -400),
    ('C', 'D', 300, 400, -300, -400),
]

print("比较所有样本对：")
correct = 0
for i, (p1, p2, t1, t2, r1, r2) in enumerate(pairs, 1):
    # 真实关系：t1 < t2（患者1先死）
    # 预测关系：r1 > r2（患者1风险更高）
    is_correct = r1 > r2
    correct += is_correct
    status = "✅" if is_correct else "❌"
    print(f"  {i}. {p1} vs {p2}: 真实({t1}<{t2}), 预测({r1}>{r2}?) {status}")

print(f"\nC-index = {correct}/{len(pairs)} = {correct/len(pairs):.2f}")
print()

# 场景 2: 随机预测（第一次）
print("场景 2：随机预测（第 1 次）")
print("-" * 80)
np.random.seed(42)
random_risk_1 = np.random.randn(4)
print(f"风险分数: {dict(zip(patients, random_risk_1))}")
print()

pairs_random_1 = [
    ('A', 'B', 100, 200, random_risk_1[0], random_risk_1[1]),
    ('A', 'C', 100, 300, random_risk_1[0], random_risk_1[2]),
    ('A', 'D', 100, 400, random_risk_1[0], random_risk_1[3]),
    ('B', 'C', 200, 300, random_risk_1[1], random_risk_1[2]),
    ('B', 'D', 200, 400, random_risk_1[1], random_risk_1[3]),
    ('C', 'D', 300, 400, random_risk_1[2], random_risk_1[3]),
]

print("比较所有样本对：")
correct_1 = 0
for i, (p1, p2, t1, t2, r1, r2) in enumerate(pairs_random_1, 1):
    is_correct = r1 > r2
    correct_1 += is_correct
    status = "✅" if is_correct else "❌"
    print(f"  {i}. {p1} vs {p2}: 真实({t1}<{t2}), 预测({r1:.2f}>{r2:.2f}?) {status}")

c1 = correct_1 / len(pairs_random_1)
print(f"\nC-index = {correct_1}/{len(pairs_random_1)} = {c1:.2f}")
print()

# 场景 3: 随机预测（第二次）
print("场景 3：随机预测（第 2 次）- 不同的随机数")
print("-" * 80)
np.random.seed(123)
random_risk_2 = np.random.randn(4)
print(f"风险分数: {dict(zip(patients, random_risk_2))}")
print()

pairs_random_2 = [
    ('A', 'B', 100, 200, random_risk_2[0], random_risk_2[1]),
    ('A', 'C', 100, 300, random_risk_2[0], random_risk_2[2]),
    ('A', 'D', 100, 400, random_risk_2[0], random_risk_2[3]),
    ('B', 'C', 200, 300, random_risk_2[1], random_risk_2[2]),
    ('B', 'D', 200, 400, random_risk_2[1], random_risk_2[3]),
    ('C', 'D', 300, 400, random_risk_2[2], random_risk_2[3]),
]

print("比较所有样本对：")
correct_2 = 0
for i, (p1, p2, t1, t2, r1, r2) in enumerate(pairs_random_2, 1):
    is_correct = r1 > r2
    correct_2 += is_correct
    status = "✅" if is_correct else "❌"
    print(f"  {i}. {p1} vs {p2}: 真实({t1}<{t2}), 预测({r1:.2f}>{r2:.2f}?) {status}")

c2 = correct_2 / len(pairs_random_2)
print(f"\nC-index = {correct_2}/{len(pairs_random_2)} = {c2:.2f}")
print()

# ============================================================================
# 统计：重复 1000 次
# ============================================================================
print("="*80)
print("📊 重复 1000 次随机预测，看 C-index 的分布")
print("="*80)
print()

c_indices = []
for seed in range(1000):
    np.random.seed(seed)
    random_risk = np.random.randn(4)
    
    # 手工计算 C-index（和上面一样的逻辑）
    correct = 0
    total = 0
    for i in range(4):
        for j in range(i+1, 4):
            if true_times[i] < true_times[j]:  # i 先死
                total += 1
                if random_risk[i] > random_risk[j]:  # 预测 i 风险更高
                    correct += 1
    
    c = correct / total if total > 0 else 0.5
    c_indices.append(c)

c_indices = np.array(c_indices)

print(f"1000 次随机预测的 C-index 统计：")
print(f"  - 均值: {c_indices.mean():.4f}")
print(f"  - 标准差: {c_indices.std():.4f}")
print(f"  - 中位数: {np.median(c_indices):.4f}")
print(f"  - 范围: [{c_indices.min():.4f}, {c_indices.max():.4f}]")
print()

print("分布情况：")
bins = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
for i in range(len(bins)-1):
    count = ((c_indices >= bins[i]) & (c_indices < bins[i+1])).sum()
    pct = count / len(c_indices) * 100
    print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {count:4d} 次 ({pct:5.1f}%)")
print()

# ============================================================================
# 关键解释
# ============================================================================
print("="*80)
print("🎯 关键解释：为什么是 0.5 而不是 0.1？")
print("="*80)
print()

print("1️⃣ C-index 不是'预测正确率'")
print("-" * 80)
print("❌ 错误理解: C-index = 预测对的患者数 / 总患者数")
print("   → 随机猜测预测对 10% 的患者 → C ≈ 0.1")
print()
print("✅ 正确理解: C-index = 排序正确的样本对数 / 总样本对数")
print("   → 4 个患者有 6 个样本对（C(4,2) = 6）")
print("   → 每一对有 50% 概率排序正确")
print("   → 平均 3 对正确 → C ≈ 0.5")
print()

print("2️⃣ 为什么每一对有 50% 概率正确？")
print("-" * 80)
print("对于样本对 (A, B)：")
print("  - 真实: A 先死 (time_A < time_B)")
print("  - 预测: risk_A 和 risk_B 都是随机数")
print()
print("问：P(risk_A > risk_B) = ?")
print("答：如果两个数都是随机的，一个比另一个大的概率 = 50%")
print()
print("这就像抛硬币：")
print("  - 正面 (50%): risk_A > risk_B → 排序正确 ✅")
print("  - 反面 (50%): risk_A < risk_B → 排序错误 ❌")
print()

print("3️⃣ 如果要达到 C = 0.1 呢？")
print("-" * 80)
print("需要 90% 的样本对排序错误！")
print()
print("这意味着模型要'系统性地反向预测'：")
print("  - 真实越早死的，预测风险越低")
print("  - 真实越晚死的，预测风险越高")
print()
print("但随机预测不会这样：")
print("  - 随机预测对每一对都是'抛硬币'")
print("  - 不会系统性地偏向错误")
print("  - 所以 C-index 会在 0.5 附近")
print()

print("4️⃣ 实际验证")
print("-" * 80)
print(f"我们刚才计算的 1000 次随机预测：")
print(f"  - 均值 = {c_indices.mean():.4f} ≈ 0.5 ✅")
print(f"  - 在 [0.45, 0.55] 范围的: {((c_indices >= 0.45) & (c_indices <= 0.55)).sum()} 次 ({((c_indices >= 0.45) & (c_indices <= 0.55)).sum()/10:.1f}%)")
print(f"  - 小于 0.2 的: {(c_indices < 0.2).sum()} 次 ({(c_indices < 0.2).sum()/10:.1f}%)")
print()

print("="*80)
print("💡 总结")
print("="*80)
print()
print("R0 中 C-index ≈ 0.5 说明：")
print("  ✅ 模型在打乱标签后完全失败了（预测 = 随机猜测）")
print("  ✅ 没有数据泄露（否则 C 会 > 0.5）")
print("  ✅ 评估代码正确")
print()
print("如果 C-index ≈ 0.1：")
print("  ❌ 说明模型系统性地反向预测（不正常）")
print("  ❌ 可能代码有严重 bug（风险符号反了）")
print()
print("类比：")
print("  - C = 0.5 就像抛硬币，正反面各 50%")
print("  - C = 0.1 就像一个'坏硬币'，90% 出反面")
print("  - 随机预测不会产生'坏硬币'，只会是公平硬币 (50%)")
print()
print("="*80)

