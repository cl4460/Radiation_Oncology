#!/usr/bin/env python3
"""
检查 Radiogenomics Phase2 输出是否与 LUNG1 兼容
用于验证是否可以安全地用于 phase3_0617 和 phase4_external_test_radiogenomics
"""
import pandas as pd
import ast
from pathlib import Path

def check_phase2_output(crop_log_path: str, dataset_name: str = "Dataset"):
    """检查 phase2 crop_log.csv 是否符合要求"""
    print(f"\n{'='*70}")
    print(f"{dataset_name} Phase2 输出检查")
    print(f"{'='*70}")
    
    try:
        df = pd.read_csv(crop_log_path)
        print(f"文件: {crop_log_path}")
        print(f"行数: {len(df)}")
        print(f"列数: {len(df.columns)}")
        
        results = {
            "file_exists": True,
            "has_required_cols": False,
            "has_out_edt": False,
            "norm_is_window": False,
            "patch_shape_correct": False,
            "spacing_correct": False,
            "no_med3d_cols": False,
            "all_ok": False
        }
        
        # 1. 必需列检查
        need = ["out_ct", "out_mask", "out_edt", "norm"]
        missing = [c for c in need if c not in df.columns]
        
        print(f"\n1. 必需列检查:")
        print(f"   需要: {need}")
        if missing:
            print(f"   ❌ 缺失: {missing}")
        else:
            print(f"   ✓ 所有必需列都存在")
            results["has_required_cols"] = True
        
        # 2. norm 检查
        print(f"\n2. Normalization 检查:")
        if "norm" in df.columns:
            norm_vals = df["norm"].unique()
            print(f"   norm unique: {norm_vals}")
            if len(norm_vals) == 1 and norm_vals[0] == "window":
                print(f"   ✓ norm = 'window' (正确)")
                results["norm_is_window"] = True
            else:
                print(f"   ❌ norm 不是 'window'")
        else:
            print(f"   ❌ 没有 'norm' 列")
        
        # 3. out_edt 检查
        print(f"\n3. EDT 输出检查:")
        if "out_edt" in df.columns:
            edt_count = df["out_edt"].notna().sum()
            print(f"   ✓ out_edt 列存在，{edt_count}/{len(df)} 行有值")
            results["has_out_edt"] = True
        else:
            print(f"   ❌ 没有 'out_edt' 列")
            print(f"   ⚠ 这是 ResNet10 pipeline 必需的列")
        
        # 4. patch_shape 检查
        print(f"\n4. Patch 形状检查:")
        if "patch_shape" in df.columns:
            try:
                shapes = [ast.literal_eval(str(s)) for s in df["patch_shape"].unique()]
                print(f"   unique shapes: {shapes}")
                expected = [160, 192, 128]  # Z, Y, X
                if shapes and list(shapes[0]) == expected:
                    print(f"   ✓ patch_shape = {expected} (正确)")
                    results["patch_shape_correct"] = True
                else:
                    print(f"   ⚠ patch_shape = {shapes[0] if shapes else 'N/A'}, 期望 {expected}")
            except Exception as e:
                print(f"   ⚠ 无法解析 patch_shape: {e}")
        else:
            print(f"   ❌ 没有 'patch_shape' 列")
        
        # 5. spacing 检查
        print(f"\n5. Spacing 检查:")
        if "spacing" in df.columns:
            try:
                spacings = [ast.literal_eval(str(s)) for s in df["spacing"].unique()]
                print(f"   unique spacings: {spacings}")
                expected = [1.0, 1.0, 3.0]  # X, Y, Z
                if spacings and list(spacings[0]) == expected:
                    print(f"   ✓ spacing = {expected} (正确)")
                    results["spacing_correct"] = True
                else:
                    print(f"   ⚠ spacing = {spacings[0] if spacings else 'N/A'}, 期望 {expected}")
            except Exception as e:
                print(f"   ⚠ 无法解析 spacing: {e}")
        else:
            print(f"   ❌ 没有 'spacing' 列")
        
        # 6. Med3D 列检查
        print(f"\n6. Med3D 特征列检查:")
        med3d_cols = ["p0.5", "p99.5", "mu", "sigma", "spacing_xyz", "patch_zyx"]
        found_med3d = [c for c in med3d_cols if c in df.columns]
        if found_med3d:
            print(f"   ⚠ 发现 Med3D 特征列: {found_med3d}")
            print(f"   ⚠ 这些列不应该出现在 ResNet10 pipeline 中")
            print(f"   ⚠ 如果看到这些列，说明使用了错误的 crop_log (可能是 DeepFeature/myresearch 的)")
        else:
            print(f"   ✓ 没有 Med3D 特征列（正确）")
            results["no_med3d_cols"] = True
        
        # 总结
        results["all_ok"] = all([
            results["has_required_cols"],
            results["has_out_edt"],
            results["norm_is_window"],
            results["patch_shape_correct"],
            results["spacing_correct"],
            results["no_med3d_cols"]
        ])
        
        return results
        
    except FileNotFoundError:
        print(f"❌ 文件不存在: {crop_log_path}")
        return {"file_exists": False, "all_ok": False}
    except Exception as e:
        print(f"❌ 读取错误: {e}")
        import traceback
        traceback.print_exc()
        return {"file_exists": False, "all_ok": False}

def main():
    print("="*70)
    print("Phase2 输出兼容性检查")
    print("="*70)
    print("\n此脚本检查 Phase2 输出是否符合 ResNet10 pipeline 的要求")
    print("必须满足以下条件:")
    print("  1. 有 out_ct, out_mask, out_edt, norm 列")
    print("  2. norm = 'window'")
    print("  3. 有 out_edt 列（EDT距离变换）")
    print("  4. patch_shape = [160, 192, 128]")
    print("  5. spacing = [1.0, 1.0, 3.0]")
    print("  6. 没有 Med3D 特征列 (p0.5, p99.5, mu, sigma 等)")
    
    # 检查 Radiogenomics
    p_rg = "/home/lichengze/Research/Radiogenomics/phase2_outputs/phase2_crop_log.csv"
    results_rg = check_phase2_output(p_rg, "Radiogenomics")
    
    # 检查 LUNG1 (作为参考)
    p_lung1 = "/home/lichengze/Research/CNN_pipeline/phase2_outputs/crop_log.csv"
    results_lung1 = check_phase2_output(p_lung1, "LUNG1 (参考)")
    
    # 最终结论
    print(f"\n{'='*70}")
    print("最终结论")
    print(f"{'='*70}")
    
    if results_rg.get("all_ok", False):
        print("✓ Radiogenomics Phase2 输出符合要求！")
        print("✓ 可以安全地用于 phase3_0617 和 phase4_external_test_radiogenomics")
    else:
        print("❌ Radiogenomics Phase2 输出不符合要求")
        print("\n需要检查的问题:")
        if not results_rg.get("has_required_cols", False):
            print("  - 缺少必需的列")
        if not results_rg.get("has_out_edt", False):
            print("  - 缺少 out_edt 列（这是 ResNet10 pipeline 必需的）")
        if not results_rg.get("norm_is_window", False):
            print("  - norm 不是 'window'")
        if not results_rg.get("patch_shape_correct", False):
            print("  - patch_shape 不正确")
        if not results_rg.get("spacing_correct", False):
            print("  - spacing 不正确")
        if not results_rg.get("no_med3d_cols", False):
            print("  - 发现了 Med3D 特征列（可能使用了错误的 crop_log）")
    
    if results_lung1.get("all_ok", False) and results_rg.get("all_ok", False):
        print("\n✓ LUNG1 和 Radiogenomics 配置一致，可以安全使用！")
    
    print("="*70)

if __name__ == "__main__":
    main()

