#!/usr/bin/env python3
"""
快速检查：找到第一个SEG文件
运行这个脚本来确认SEG文件的位置
"""
import os
from pathlib import Path
import pydicom

RAW_BASE = "/data0/lichengze/NSCLC_Radiogenomics/NSCLC Radiogenomics"

print("快速查找第一个SEG文件...")
print("这可能需要1-2分钟...")
print()

# 检查前3个R01患者
r01_dirs = sorted([d for d in Path(RAW_BASE).glob("R01-*")])[:3]

found_seg = False

for patient_dir in r01_dirs:
    patient_id = patient_dir.name
    print(f"检查 {patient_id}...")
    
    # 递归搜索
    file_count = 0
    for filepath in patient_dir.rglob("*"):
        if not filepath.is_file() or filepath.name.startswith('.'):
            continue
        
        file_count += 1
        if file_count > 1000:  # 限制搜索数量避免太慢
            break
        
        try:
            ds = pydicom.dcmread(str(filepath), stop_before_pixels=True)
            mod = str(getattr(ds, 'Modality', ''))
            
            if mod in ('SEG', 'RTSEG'):
                print()
                print("="*70)
                print("✓ 找到SEG文件！")
                print("="*70)
                print(f"患者: {patient_id}")
                print(f"完整路径: {filepath}")
                print(f"相对路径: {filepath.relative_to(RAW_BASE)}")
                print()
                print(f"父目录: {filepath.parent}")
                print(f"文件名: {filepath.name}")
                print()
                
                # 分析目录结构
                parts = filepath.relative_to(patient_dir).parts
                print(f"目录层级: {len(parts)-1} 层")
                print(f"结构: {' / '.join(parts)}")
                print()
                
                # SeriesInstanceUID
                try:
                    series_uid = getattr(ds, 'SeriesInstanceUID', 'N/A')
                    print(f"SeriesInstanceUID: {series_uid}")
                except:
                    pass
                
                print("="*70)
                found_seg = True
                break
        except:
            continue
    
    if found_seg:
        break

if not found_seg:
    print()
    print("⚠ 在前3个R01患者的前1000个文件中未找到SEG文件")
    print()
    print("可能原因：")
    print("1. SEG文件在靠后的文件中（继续搜索更多文件）")
    print("2. SEG文件在单独的目录")
    print("3. 文件扩展名不是标准的")
    print()
    print("建议：运行完整搜索")
    print("  find /data0/lichengze/NSCLC_Radiogenomics -name '*.dcm' | head -20")