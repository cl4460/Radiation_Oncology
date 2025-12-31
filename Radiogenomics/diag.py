#!/usr/bin/env python3
"""
检查NSCLC_Radiogenomics数据集中每个patient的SEG标签数量
"""
import pydicom
import pydicom_seg
from pathlib import Path
import glob
import os
from collections import defaultdict

RAW_BASE = Path("/data0/lichengze/NSCLC_Radiogenomics/NSCLC Radiogenomics")

print("="*70)
print("检查NSCLC_Radiogenomics数据集中每个patient的SEG标签数量")
print("="*70)

# 获取所有患者目录
pats_amc = sorted(glob.glob(os.path.join(str(RAW_BASE), "AMC-*")))
pats_r01 = sorted(glob.glob(os.path.join(str(RAW_BASE), "R01-*")))
all_patients = pats_amc + pats_r01

print(f"\n找到 {len(pats_amc)} 个AMC-*患者和 {len(pats_r01)} 个R01-*患者")
print(f"总计: {len(all_patients)} 个患者\n")

results = []
patients_with_seg = 0
patients_without_seg = 0

for idx, patient_dir in enumerate(all_patients, 1):
    patient_id = Path(patient_dir).name

    if idx % 20 == 0:
        print(f"进度: {idx}/{len(all_patients)}...")
    
    # 找所有SEG文件（不只是一个）
    seg_files = []
    for f in Path(patient_dir).rglob("*"):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            if getattr(ds, 'Modality', '') in ('SEG', 'RTSEG'):
                seg_files.append(f)
        except:
            pass
    
    if not seg_files:
        patients_without_seg += 1
        results.append({
            'patient': patient_id,
            'seg_file_count': 0,
            'total_segment_count': 0,
            'segments_per_file': [],
            'all_labels': [],
            'error': None
        })
        continue
    
    patients_with_seg += 1
    total_segments = 0
    segments_per_file = []
    all_labels = []
    
    # 检查每个SEG文件
    for seg_file in seg_files:
    try:
        ds = pydicom.dcmread(str(seg_file))
        reader = pydicom_seg.SegmentReader()
        res = reader.read(ds)
        
        seg_count = len(res.available_segments)
            total_segments += seg_count
            segments_per_file.append(seg_count)
        
            file_labels = []
        for seg_id in res.available_segments:
            info = res.segment_infos[seg_id]
            lab = str(info.get("SegmentLabel", ""))
            desc = str(info.get("SegmentDescription", ""))
                label_str = f"{lab} ({desc})" if desc else lab
                file_labels.append(label_str)
                all_labels.append(label_str)
        
    except Exception as e:
            results.append({
                'patient': patient_id,
                'seg_file_count': len(seg_files),
                'total_segment_count': total_segments,
                'segments_per_file': segments_per_file,
                'all_labels': all_labels,
                'error': str(e)
            })
            continue
    
    results.append({
        'patient': patient_id,
        'seg_file_count': len(seg_files),
        'total_segment_count': total_segments,
        'segments_per_file': segments_per_file,
        'all_labels': all_labels,
        'error': None
    })

print("\n" + "="*70)
print("统计汇总")
print("="*70)

print(f"\n总患者数: {len(all_patients)}")
print(f"有SEG文件的患者: {patients_with_seg}")
print(f"无SEG文件的患者: {patients_without_seg}")

# 统计每个患者的segment数量分布
segment_count_distribution = defaultdict(int)
for r in results:
    if r['error'] is None:
        segment_count_distribution[r['total_segment_count']] += 1

print("\n每个患者的SEG标签数量分布:")
for seg_count in sorted(segment_count_distribution.keys()):
    patient_count = segment_count_distribution[seg_count]
    print(f"  {seg_count} 个标签: {patient_count} 个患者")

# 统计有多个SEG文件的患者
multi_file_patients = [r for r in results if r['seg_file_count'] > 1]
if multi_file_patients:
    print(f"\n有多个SEG文件的患者数: {len(multi_file_patients)}")
    print("这些患者的详细信息:")
    for r in multi_file_patients[:10]:  # 只显示前10个
        print(f"  {r['patient']}: {r['seg_file_count']} 个SEG文件, "
              f"每文件segment数: {r['segments_per_file']}, "
              f"总计: {r['total_segment_count']} 个标签")

# 统计label模式
label_counts = defaultdict(int)
for r in results:
    for label in r['all_labels']:
        # 提取主要label（括号前的部分）
        main_label = label.split('(')[0].strip()
        if main_label:
            label_counts[main_label] += 1

if label_counts:
    print("\n最常见的标签类型:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  {label}: {count} 次")

# 保存详细结果到CSV
import pandas as pd
df_data = []
for r in results:
    df_data.append({
        'patient_id': r['patient'],
        'seg_file_count': r['seg_file_count'],
        'total_segment_count': r['total_segment_count'],
        'segments_per_file': str(r['segments_per_file']),
        'all_labels': '; '.join(r['all_labels']),
        'error': r['error']
    })

df = pd.DataFrame(df_data)
output_file = "radiogenomics_seg_label_count.csv"
df.to_csv(output_file, index=False)
print(f"\n✓ 详细结果已保存到: {output_file}")

print("\n" + "="*70)
print("结论")
print("="*70)

# 检查segment数量模式
single_seg_patients = sum(1 for r in results if r['total_segment_count'] == 1)
multi_seg_patients = sum(1 for r in results if r['total_segment_count'] > 1)

print(f"\n只有1个SEG标签的患者: {single_seg_patients}")
print(f"有多个SEG标签的患者: {multi_seg_patients}")

if single_seg_patients > 0 and multi_seg_patients == 0:
    print("→ 所有患者都只有1个segment")
    print("→ 策略：直接使用第一个segment")
elif single_seg_patients > multi_seg_patients:
    print("→ 大多数患者只有1个segment")
    print("→ 策略：对于单segment患者直接使用，多segment患者需要智能选择")
else:
    print("→ 很多患者有多个segments")
    print("→ 策略：需要智能选择segment")

print("="*70)