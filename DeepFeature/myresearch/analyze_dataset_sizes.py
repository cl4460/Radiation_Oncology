#!/usr/bin/env python
"""
分析NSCLC-Radiomics原始数据集的大小
用于验证FIXED_PATCH=(160, 192, 128)是否合适

这个脚本会：
1. 读取原始CT和mask文件
2. 计算肿瘤bounding box的大小
3. 考虑margin后的实际需要大小
4. 与FIXED_PATCH进行对比
5. 生成统计报告和可视化
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============================================================
# 配置（与phase2_preprocessing.py保持一致）
# ============================================================
# 尝试多个可能的路径（优先使用myresearch文件夹下的）
PHASE1_QC_PATHS = [
    "/home/lichengze/Research/DeepFeature/myresearch/phase1_outputs/phase1_qc.csv",
    "/home/lichengze/Research/CNN_pipeline/phase1_outputs/phase1_qc.csv",
    "/home/lichengze/CNN_pipeline/phase1_outputs/phase1_qc.csv",
]
PHASE1_QC = None
for path in PHASE1_QC_PATHS:
    if os.path.exists(path):
        PHASE1_QC = path
        break
if PHASE1_QC is None:
    raise FileNotFoundError(f"找不到phase1_qc.csv，尝试过的路径: {PHASE1_QC_PATHS}")
# FIXED_PATCH顺序说明：
# - phase2.py中定义为 (Z, Y, X)，例如 (128, 192, 160) 表示 Z=128, Y=192, X=160
# - 在analyze_dataset_sizes.py中转换为 (X, Y, Z) 进行比较
# - 如果phase2.py中 FIXED_PATCH = (Z, Y, X) = (128, 192, 160)
#   则这里应该是 FIXED_PATCH = (X, Y, Z) = (160, 192, 128)
# 注意：需要与phase2.py中的实际值保持一致！
FIXED_PATCH = (160, 192, 128)  # (X, Y, Z) in resampled space - 对应phase2.py中的(Z,Y,X)=(128,192,160)
MARGIN_MM = 20  # Physical margin around tumor
TARGET_SP = (1.0, 1.0, 3.0)  # Target spacing (X, Y, Z) in mm

OUTPUT_DIR = Path("/home/lichengze/Research/DeepFeature/myresearch/phase2_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 辅助函数
# ============================================================
def bbox_zyx(mask_arr: np.ndarray):
    """计算mask的bounding box"""
    idx = np.argwhere(mask_arr > 0)
    if idx.size == 0:
        return None
    return idx.min(0), idx.max(0)

def get_roi_size_with_margin(bb_min, bb_max, spacing, margin_mm):
    """计算考虑margin后的ROI大小（以voxel为单位）"""
    roi_size = bb_max - bb_min + 1
    margin_vox = np.ceil(margin_mm / np.array(spacing)).astype(int)
    return roi_size + 2 * margin_vox

def fix_path(path: str) -> str:
    """修复路径，处理路径不一致的情况"""
    if not os.path.exists(path):
        # 尝试添加 Research
        path_with_research = path.replace('/home/lichengze/CNN_pipeline/', 
                                          '/home/lichengze/Research/CNN_pipeline/')
        if os.path.exists(path_with_research):
            return path_with_research
        
        # 尝试移除 Research
        path_without_research = path.replace('/home/lichengze/Research/CNN_pipeline/', 
                                             '/home/lichengze/CNN_pipeline/')
        if os.path.exists(path_without_research):
            return path_without_research
    
    return path

def analyze_case(ct_path: str, mask_path: str, target_spacing: tuple):
    """分析单个case的大小"""
    try:
        # 修复路径
        ct_path = fix_path(ct_path)
        mask_path = fix_path(mask_path)
        
        # 读取原始图像
        ct_img = sitk.ReadImage(ct_path)
        mask_img = sitk.ReadImage(mask_path)
        
        # 获取原始spacing和size
        orig_spacing = ct_img.GetSpacing()
        orig_size = ct_img.GetSize()
        
        # Resample到target spacing（与phase2一致）
        def resample_to_target(image, is_label=False):
            new_spacing = target_spacing
            original_spacing = image.GetSpacing()
            original_size = image.GetSize()
            
            new_size = [
                int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
                for i in range(3)
            ]
            
            resample = sitk.ResampleImageFilter()
            resample.SetOutputSpacing(new_spacing)
            resample.SetSize(new_size)
            resample.SetOutputDirection(image.GetDirection())
            resample.SetOutputOrigin(image.GetOrigin())
            resample.SetTransform(sitk.Transform())
            resample.SetDefaultPixelValue(0 if is_label else -1000)
            resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
            
            return resample.Execute(image)
        
        ct_resampled = resample_to_target(ct_img, is_label=False)
        mask_resampled = resample_to_target(mask_img, is_label=True)
        
        # 转换为numpy数组
        mask_arr = sitk.GetArrayFromImage(mask_resampled).astype(np.uint8)
        ct_arr = sitk.GetArrayFromImage(ct_resampled)
        
        # 计算resampled后的size
        resampled_size = np.array(mask_arr.shape)  # (Z, Y, X)
        
        # 计算肿瘤bounding box
        bb = bbox_zyx(mask_arr)
        
        if bb is None:
            return {
                "error": "No mask found",
                "resampled_size_x": resampled_size[2],
                "resampled_size_y": resampled_size[1],
                "resampled_size_z": resampled_size[0],
            }
        
        bb_min, bb_max = bb
        roi_size = bb_max - bb_min + 1  # (Z, Y, X)
        
        # 计算考虑margin后的需要大小
        margin_vox = np.ceil(MARGIN_MM / np.array(target_spacing)).astype(int)
        needed_size = roi_size + 2 * margin_vox  # (Z, Y, X)
        
        # 转换为(X, Y, Z)顺序（与FIXED_PATCH一致）
        roi_size_xyz = np.array([roi_size[2], roi_size[1], roi_size[0]])
        needed_size_xyz = np.array([needed_size[2], needed_size[1], needed_size[0]])
        resampled_size_xyz = np.array([resampled_size[2], resampled_size[1], resampled_size[0]])
        
        # 检查是否会被truncate
        truncated = (needed_size_xyz > np.array(FIXED_PATCH)).any()
        truncation_axes = []
        for i, axis_name in enumerate(['X', 'Y', 'Z']):
            if needed_size_xyz[i] > FIXED_PATCH[i]:
                truncation_axes.append(axis_name)
        
        # 计算truncation比例
        truncation_ratio = needed_size_xyz / np.array(FIXED_PATCH)
        max_truncation_ratio = truncation_ratio.max()
        
        # 计算mask保留比例（如果使用FIXED_PATCH）
        total_mask_voxels = float((mask_arr > 0).sum())
        
        # 模拟crop（简化版，假设centered）
        center = (bb_min + bb_max) // 2
        half_patch = np.array(FIXED_PATCH[::-1]) // 2  # 转换为(Z, Y, X)
        start = np.maximum(center - half_patch, 0)
        end = np.minimum(start + np.array(FIXED_PATCH[::-1]), mask_arr.shape)
        
        cropped_mask = mask_arr[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        kept_mask_voxels = float((cropped_mask > 0).sum())
        retained_ratio = kept_mask_voxels / total_mask_voxels if total_mask_voxels > 0 else 0.0
        
        return {
            "resampled_size_x": int(resampled_size_xyz[0]),
            "resampled_size_y": int(resampled_size_xyz[1]),
            "resampled_size_z": int(resampled_size_xyz[2]),
            "roi_size_x": int(roi_size_xyz[0]),
            "roi_size_y": int(roi_size_xyz[1]),
            "roi_size_z": int(roi_size_xyz[2]),
            "needed_size_x": int(needed_size_xyz[0]),
            "needed_size_y": int(needed_size_xyz[1]),
            "needed_size_z": int(needed_size_xyz[2]),
            "truncated": bool(truncated),
            "truncation_axes": ", ".join(truncation_axes) if truncation_axes else "None",
            "max_truncation_ratio": float(max_truncation_ratio),
            "truncation_ratio_x": float(truncation_ratio[0]),
            "truncation_ratio_y": float(truncation_ratio[1]),
            "truncation_ratio_z": float(truncation_ratio[2]),
            "retained_mask_ratio": float(retained_ratio),
            "total_mask_voxels": int(total_mask_voxels),
            "kept_mask_voxels": int(kept_mask_voxels),
        }
        
    except Exception as e:
        return {"error": str(e)}

# ============================================================
# 主分析函数
# ============================================================
def main():
    print("="*70)
    print("NSCLC-Radiomics数据集大小分析")
    print("="*70)
    print(f"FIXED_PATCH: {FIXED_PATCH} (X, Y, Z)")
    print(f"MARGIN: {MARGIN_MM} mm")
    print(f"TARGET_SPACING: {TARGET_SP} mm (X, Y, Z)")
    print("="*70)
    
    # 读取数据
    print("\n读取数据...")
    df = pd.read_csv(PHASE1_QC)
    print(f"总病例数: {len(df)}")
    
    # 修复路径（phase1_qc.csv中的路径可能缺少Research）
    print("修复路径...")
    df['ct_path'] = df['ct_path'].apply(fix_path)
    df['mask_path'] = df['mask_path'].apply(fix_path)
    
    # 分析每个case
    print("\n分析每个case的大小...")
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        patient_id = row.get("patient_id", f"case_{idx}")
        ct_path = row.get("ct_path")
        mask_path = row.get("mask_path")
        
        if not ct_path or not mask_path:
            results.append({"patient_id": patient_id, "error": "Missing paths"})
            continue
        
        if not os.path.exists(ct_path) or not os.path.exists(mask_path):
            results.append({"patient_id": patient_id, "error": "File not found"})
            continue
        
        result = analyze_case(ct_path, mask_path, TARGET_SP)
        result["patient_id"] = patient_id
        results.append(result)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 保存详细结果
    output_csv = OUTPUT_DIR / "dataset_size_analysis.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n详细结果已保存到: {output_csv}")
    
    # 过滤有效结果
    if "error" in results_df.columns:
        valid_df = results_df[~results_df["error"].notna()].copy()
    else:
        # 如果没有error列，说明所有case都成功了
        valid_df = results_df.copy()
    
    print(f"\n有效分析结果: {len(valid_df)}/{len(results_df)}")
    
    if len(valid_df) == 0:
        print("❌ 没有有效的分析结果！")
        return
    
    # ============================================================
    # 统计分析
    # ============================================================
    print("\n" + "="*70)
    print("统计分析")
    print("="*70)
    
    # ROI大小统计
    print("\n【1. 肿瘤ROI大小统计（resampled后，单位：voxels）】")
    print("-"*70)
    for axis, name in [('x', 'X'), ('y', 'Y'), ('z', 'Z')]:
        col = f"roi_size_{axis}"
        print(f"\n{name}轴:")
        print(f"  均值: {valid_df[col].mean():.1f}")
        print(f"  中位数: {valid_df[col].median():.1f}")
        print(f"  标准差: {valid_df[col].std():.1f}")
        print(f"  最小值: {valid_df[col].min()}")
        print(f"  最大值: {valid_df[col].max()}")
        print(f"  25%分位数: {valid_df[col].quantile(0.25):.1f}")
        print(f"  75%分位数: {valid_df[col].quantile(0.75):.1f}")
        print(f"  95%分位数: {valid_df[col].quantile(0.95):.1f}")
        print(f"  99%分位数: {valid_df[col].quantile(0.99):.1f}")
    
    # 考虑margin后的需要大小
    print("\n【2. 考虑margin后的需要大小统计（resampled后，单位：voxels）】")
    print("-"*70)
    print(f"Margin: {MARGIN_MM} mm = {np.ceil(MARGIN_MM / np.array(TARGET_SP)).astype(int)} voxels (X, Y, Z)")
    
    for axis, name in [('x', 'X'), ('y', 'Y'), ('z', 'Z')]:
        col = f"needed_size_{axis}"
        fixed = FIXED_PATCH[{'x': 0, 'y': 1, 'z': 2}[axis]]
        print(f"\n{name}轴 (FIXED_PATCH = {fixed}):")
        print(f"  均值: {valid_df[col].mean():.1f}")
        print(f"  中位数: {valid_df[col].median():.1f}")
        print(f"  最大值: {valid_df[col].max()}")
        print(f"  95%分位数: {valid_df[col].quantile(0.95):.1f}")
        print(f"  99%分位数: {valid_df[col].quantile(0.99):.1f}")
        
        # 统计超过FIXED_PATCH的数量
        exceeds = (valid_df[col] > fixed).sum()
        exceeds_pct = exceeds / len(valid_df) * 100
        print(f"  超过FIXED_PATCH的数量: {exceeds} ({exceeds_pct:.1f}%)")
    
    # Truncation统计
    print("\n【3. Truncation统计】")
    print("-"*70)
    truncated_count = valid_df["truncated"].sum()
    truncated_pct = truncated_count / len(valid_df) * 100
    print(f"会被truncate的case数: {truncated_count} ({truncated_pct:.1f}%)")
    
    if truncated_count > 0:
        print(f"\nTruncation详情:")
        truncation_axes_counts = valid_df[valid_df["truncated"]]["truncation_axes"].value_counts()
        for axes, count in truncation_axes_counts.items():
            print(f"  {axes}: {count} cases")
        
        print(f"\n最大truncation比例:")
        print(f"  均值: {valid_df['max_truncation_ratio'].mean():.2f}x")
        print(f"  最大值: {valid_df['max_truncation_ratio'].max():.2f}x")
        print(f"  95%分位数: {valid_df['max_truncation_ratio'].quantile(0.95):.2f}x")
    
    # Mask保留比例
    print("\n【4. Mask保留比例统计（使用FIXED_PATCH）】")
    print("-"*70)
    print(f"均值: {valid_df['retained_mask_ratio'].mean():.3f}")
    print(f"中位数: {valid_df['retained_mask_ratio'].median():.3f}")
    print(f"最小值: {valid_df['retained_mask_ratio'].min():.3f}")
    
    low_retention = (valid_df['retained_mask_ratio'] < 0.9).sum()
    low_retention_pct = low_retention / len(valid_df) * 100
    print(f"保留比例 < 90% 的case数: {low_retention} ({low_retention_pct:.1f}%)")
    
    very_low_retention = (valid_df['retained_mask_ratio'] < 0.8).sum()
    very_low_retention_pct = very_low_retention / len(valid_df) * 100
    print(f"保留比例 < 80% 的case数: {very_low_retention} ({very_low_retention_pct:.1f}%)")
    
    # ============================================================
    # 可视化
    # ============================================================
    print("\n" + "="*70)
    print("生成可视化图表...")
    print("="*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Dataset Size Analysis (FIXED_PATCH={FIXED_PATCH}, MARGIN={MARGIN_MM}mm)', 
                 fontsize=16, fontweight='bold')
    
    # 1. ROI大小分布
    axes[0, 0].hist([valid_df['roi_size_x'], valid_df['roi_size_y'], valid_df['roi_size_z']], 
                    bins=30, alpha=0.7, label=['X', 'Y', 'Z'])
    axes[0, 0].axvline(FIXED_PATCH[0], color='r', linestyle='--', label=f'FIXED_PATCH X={FIXED_PATCH[0]}')
    axes[0, 0].axvline(FIXED_PATCH[1], color='g', linestyle='--', label=f'FIXED_PATCH Y={FIXED_PATCH[1]}')
    axes[0, 0].axvline(FIXED_PATCH[2], color='b', linestyle='--', label=f'FIXED_PATCH Z={FIXED_PATCH[2]}')
    axes[0, 0].set_xlabel('ROI Size (voxels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Tumor ROI Size Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 需要大小分布
    axes[0, 1].hist([valid_df['needed_size_x'], valid_df['needed_size_y'], valid_df['needed_size_z']], 
                    bins=30, alpha=0.7, label=['X', 'Y', 'Z'])
    axes[0, 1].axvline(FIXED_PATCH[0], color='r', linestyle='--', linewidth=2, label=f'FIXED_PATCH X={FIXED_PATCH[0]}')
    axes[0, 1].axvline(FIXED_PATCH[1], color='g', linestyle='--', linewidth=2, label=f'FIXED_PATCH Y={FIXED_PATCH[1]}')
    axes[0, 1].axvline(FIXED_PATCH[2], color='b', linestyle='--', linewidth=2, label=f'FIXED_PATCH Z={FIXED_PATCH[2]}')
    axes[0, 1].set_xlabel('Needed Size with Margin (voxels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Required Size Distribution (ROI + Margin)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Truncation比例分布
    axes[0, 2].hist(valid_df['max_truncation_ratio'], bins=30, alpha=0.7, color='orange')
    axes[0, 2].axvline(1.0, color='r', linestyle='--', linewidth=2, label='No truncation (1.0)')
    axes[0, 2].set_xlabel('Max Truncation Ratio')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Truncation Ratio Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Mask保留比例分布
    axes[1, 0].hist(valid_df['retained_mask_ratio'], bins=30, alpha=0.7, color='green')
    axes[1, 0].axvline(0.9, color='r', linestyle='--', linewidth=2, label='90% threshold')
    axes[1, 0].axvline(0.8, color='orange', linestyle='--', linewidth=2, label='80% threshold')
    axes[1, 0].set_xlabel('Retained Mask Ratio')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Mask Retention Ratio Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 箱线图：各轴的需要大小
    data_for_box = [valid_df['needed_size_x'], valid_df['needed_size_y'], valid_df['needed_size_z']]
    bp = axes[1, 1].boxplot(data_for_box, labels=['X', 'Y', 'Z'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['red', 'green', 'blue']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 1].axhline(FIXED_PATCH[0], color='r', linestyle='--', linewidth=2, label=f'FIXED_PATCH X={FIXED_PATCH[0]}')
    axes[1, 1].axhline(FIXED_PATCH[1], color='g', linestyle='--', linewidth=2, label=f'FIXED_PATCH Y={FIXED_PATCH[1]}')
    axes[1, 1].axhline(FIXED_PATCH[2], color='b', linestyle='--', linewidth=2, label=f'FIXED_PATCH Z={FIXED_PATCH[2]}')
    axes[1, 1].set_ylabel('Size (voxels)')
    axes[1, 1].set_title('Required Size by Axis (Boxplot)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. 散点图：需要大小 vs 保留比例
    scatter = axes[1, 2].scatter(valid_df['max_truncation_ratio'], 
                                 valid_df['retained_mask_ratio'],
                                 alpha=0.6, c=valid_df['retained_mask_ratio'], 
                                 cmap='RdYlGn', s=50)
    axes[1, 2].axvline(1.0, color='r', linestyle='--', linewidth=2, label='No truncation')
    axes[1, 2].axhline(0.9, color='orange', linestyle='--', linewidth=1, label='90% retention')
    axes[1, 2].set_xlabel('Max Truncation Ratio')
    axes[1, 2].set_ylabel('Retained Mask Ratio')
    axes[1, 2].set_title('Truncation vs Retention')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 2], label='Retention Ratio')
    
    plt.tight_layout()
    
    # 保存图表
    output_fig = OUTPUT_DIR / "dataset_size_analysis.png"
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存到: {output_fig}")
    
    # ============================================================
    # 生成总结报告
    # ============================================================
    print("\n" + "="*70)
    print("总结报告")
    print("="*70)
    
    report_lines = [
        "="*70,
        "NSCLC-Radiomics数据集大小分析报告",
        "="*70,
        f"分析日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"总病例数: {len(df)}",
        f"有效分析结果: {len(valid_df)}",
        "",
        f"配置参数:",
        f"  FIXED_PATCH: {FIXED_PATCH} (X, Y, Z)",
        f"  MARGIN: {MARGIN_MM} mm",
        f"  TARGET_SPACING: {TARGET_SP} mm (X, Y, Z)",
        "",
        "关键发现:",
        f"  1. 会被truncate的case数: {truncated_count} ({truncated_pct:.1f}%)",
        f"  2. 保留比例 < 90% 的case数: {low_retention} ({low_retention_pct:.1f}%)",
        f"  3. 保留比例 < 80% 的case数: {very_low_retention} ({very_low_retention_pct:.1f}%)",
        "",
        "各轴需要大小统计（95%分位数）:",
        f"  X轴: {valid_df['needed_size_x'].quantile(0.95):.1f} (FIXED_PATCH = {FIXED_PATCH[0]})",
        f"  Y轴: {valid_df['needed_size_y'].quantile(0.95):.1f} (FIXED_PATCH = {FIXED_PATCH[1]})",
        f"  Z轴: {valid_df['needed_size_z'].quantile(0.95):.1f} (FIXED_PATCH = {FIXED_PATCH[2]})",
        "",
        "结论:",
    ]
    
    if truncated_pct < 5:
        report_lines.append(f"  ✅ FIXED_PATCH设置合理，只有{truncated_pct:.1f}%的case会被truncate")
    elif truncated_pct < 10:
        report_lines.append(f"  ⚠️  FIXED_PATCH设置基本合理，但有{truncated_pct:.1f}%的case会被truncate")
    else:
        report_lines.append(f"  ❌ FIXED_PATCH可能过小，有{truncated_pct:.1f}%的case会被truncate")
    
    if low_retention_pct < 5:
        report_lines.append(f"  ✅ 大部分case（{100-low_retention_pct:.1f}%）的mask保留比例 > 90%")
    else:
        report_lines.append(f"  ⚠️  有{low_retention_pct:.1f}%的case的mask保留比例 < 90%")
    
    report_lines.append("="*70)
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # 保存报告
    report_file = OUTPUT_DIR / "dataset_size_analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    print(f"\n报告已保存到: {report_file}")
    
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)

if __name__ == "__main__":
    main()

