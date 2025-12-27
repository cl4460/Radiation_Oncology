#!/usr/bin/env python3
"""
演示 alpha_step 在三模态融合中的限制
"""

import math

def test_step(step: float, n_models: int = 3):
    """测试某个 step 值是否能用于 n_models 融合"""
    print(f"\n{'='*60}")
    print(f"测试: step = {step}, n_models = {n_models}")
    print(f"{'='*60}")
    
    inv = 1.0 / step
    N = int(round(inv))
    actual_sum = N * step
    
    print(f"1.0 / step = {inv:.10f}")
    print(f"N = int(round({inv:.10f})) = {N}")
    print(f"N * step = {N} * {step} = {actual_sum:.10f}")
    print(f"期望值 = 1.0")
    print(f"是否相等: {math.isclose(actual_sum, 1.0, rel_tol=0, abs_tol=1e-9)}")
    
    if math.isclose(actual_sum, 1.0, rel_tol=0, abs_tol=1e-9):
        print(f"✅ 可以使用！")
        print(f"   示例权重组合：")
        print(f"   - [0.0, 0.0, 1.0]  ← ({0}, {0}, {N}) * {step}")
        print(f"   - [0.0, {step}, {1.0-step:.2f}]  ← ({0}, {1}, {N-1}) * {step}")
        print(f"   - [1.0, 0.0, 0.0]  ← ({N}, {0}, {0}) * {step}")
    else:
        print(f"❌ 不能使用！")
        print(f"   因为 N * step = {actual_sum:.10f} ≠ 1.0")
        print(f"   这会导致权重向量和不等于 1.0")
    
    # 计算组合数量（仅用于 step=0.1 和 0.05 的情况）
    if n_models == 3:
        # C(N + n_models - 1, n_models - 1) = C(N + 2, 2)
        if N <= 100:
            num_combinations = (N + 2) * (N + 1) // 2
            print(f"\n   候选权重组合数量: {num_combinations}")
            if num_combinations > 1000:
                print(f"   ⚠️  组合数量较多，inner-CV 可能需要较长时间")


if __name__ == "__main__":
    print("=" * 60)
    print("alpha_step 在三模态融合中的限制验证")
    print("=" * 60)
    
    # 测试可以使用的步长
    print("\n✅ 可以使用的步长：")
    test_step(0.1, n_models=3)
    test_step(0.05, n_models=3)
    test_step(0.2, n_models=3)
    test_step(0.25, n_models=3)
    
    # 测试不能使用的步长
    print("\n\n❌ 不能使用的步长：")
    test_step(0.03, n_models=3)
    test_step(0.07, n_models=3)
    test_step(0.15, n_models=3)
    
    print("\n" + "=" * 60)
    print("总结：")
    print("  - 对于三模态，alpha_step 必须能整除 1.0")
    print("  - 建议先用 0.1（较粗糙，但快速稳定）")
    print("  - 如果需要更细的网格，再尝试 0.05（更多组合，更慢）")
    print("=" * 60)

