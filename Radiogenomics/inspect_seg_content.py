#!/usr/bin/env python3
"""
检查SEG文件的实际内容
找出为什么extract_mask函数没有提取到分割
"""
import pydicom
import pydicom_seg

seg_file = "/data0/lichengze/NSCLC_Radiogenomics/NSCLC Radiogenomics/R01-001/09-06-1990-NA-CT CHEST ABD PELVIS WITH CON-98785/1000.000000-3D Slicer segmentation result-67652/1-1.dcm"

print("="*70)
print("检查SEG文件的segment信息")
print("="*70)
print(f"文件: {seg_file}")
print()

# 1. 基本DICOM信息
print("步骤1: 基本DICOM属性")
print("-"*70)
ds = pydicom.dcmread(seg_file)
print(f"Modality: {getattr(ds, 'Modality', 'N/A')}")
print(f"SeriesDescription: {getattr(ds, 'SeriesDescription', 'N/A')}")
print()

# 2. 使用pydicom_seg读取
print("步骤2: 使用pydicom_seg读取segment信息")
print("-"*70)
try:
    reader = pydicom_seg.SegmentReader()
    res = reader.read(ds)
    
    print(f"Available segments: {res.available_segments}")
    print()
    
    print("Segment详细信息:")
    for seg_id in res.available_segments:
        info = res.segment_infos[seg_id]
        print(f"\nSegment {seg_id}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # 特别检查SegmentLabel和SegmentDescription
        lab = str(info.get("SegmentLabel", ""))
        desc = str(info.get("SegmentDescription", ""))
        
        print(f"\n  关键检查:")
        print(f"    SegmentLabel: '{lab}'")
        print(f"    SegmentDescription: '{desc}'")
        print(f"    Label包含'gtv': {'gtv' in lab.lower()}")
        print(f"    Desc包含'gtv': {'gtv' in desc.lower()}")
        print(f"    Label包含'primary': {'primary' in lab.lower()}")
        print(f"    Desc包含'primary': {'primary' in desc.lower()}")
        print(f"    Label包含'tumor': {'tumor' in lab.lower()}")
        print(f"    Desc包含'tumor': {'tumor' in desc.lower()}")
        
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*70)
print("分析")
print("="*70)
print("如果SegmentLabel和SegmentDescription都不包含:")
print("  - 'gtv', 'primary', 'tumor' 等关键词")
print("那么extract_mask函数的条件判断会失败，导致不提取这个segment")
print()
print("解决方案:")
print("  1. 如果segment只有1个，直接使用它")
print("  2. 放宽关键词匹配条件")
print("  3. 使用默认的第一个segment")
print("="*70)