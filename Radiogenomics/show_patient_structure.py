#!/usr/bin/env python3
"""
显示Radiogenomics数据集中指定患者的目录结构
用法: python show_patient_structure.py R01-022
"""
import sys
from pathlib import Path
import pydicom

RAW_BASE = Path("/data0/lichengze/NSCLC_Radiogenomics/NSCLC Radiogenomics")

def show_patient_structure(patient_id: str, max_depth: int = 3):
    """显示患者的目录结构"""
    patient_dir = RAW_BASE / patient_id
    
    if not patient_dir.exists():
        print(f"错误: 患者目录不存在: {patient_dir}")
        return
    
    print("="*70)
    print(f"Radiogenomics 数据集目录结构: {patient_id}")
    print("="*70)
    print()
    
    # 使用类似tree的格式
    def tree_format(path, prefix="", max_depth=3, current_depth=0, is_last=True):
        if current_depth >= max_depth:
            return
        
        items = sorted([p for p in path.iterdir()])
        dirs = [p for p in items if p.is_dir()]
        files = [p for p in items if p.is_file() and (p.suffix == '.dcm' or p.suffix == '')]
        
        # 只显示有DICOM文件的目录
        display_dirs = []
        for d in dirs:
            if list(d.rglob("*.dcm")):
                display_dirs.append(d)
        
        # 显示文件（每个目录最多3个）
        display_files = files[:3] if current_depth == max_depth - 1 else []
        
        all_items = display_dirs + display_files
        
        for i, item in enumerate(all_items):
            is_last_item = (i == len(all_items) - 1)
            
            if item.is_dir():
                # 获取目录信息
                dcm_files = list(item.rglob("*.dcm"))
                if dcm_files:
                    try:
                        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                        mod = getattr(ds, 'Modality', 'UNKNOWN')
                        desc = getattr(ds, 'SeriesDescription', 'N/A')
                        if len(desc) > 35:
                            desc = desc[:35] + "..."
                        info = f" [{mod}] {desc} ({len(dcm_files)} files)"
                    except:
                        info = f" ({len(dcm_files)} DICOM files)"
                else:
                    info = ""
                
                connector = "└── " if is_last_item else "├── "
                print(f"{prefix}{connector}{item.name}{info}")
                
                next_prefix = prefix + ("    " if is_last_item else "│   ")
                tree_format(item, next_prefix, max_depth, current_depth + 1, is_last_item)
            else:
                # 显示文件
                try:
                    ds = pydicom.dcmread(str(item), stop_before_pixels=True)
                    mod = getattr(ds, 'Modality', 'UNKNOWN')
                    print(f"{prefix}{'└── ' if is_last_item else '├── '}{item.name} [{mod}]")
                except:
                    print(f"{prefix}{'└── ' if is_last_item else '├── '}{item.name}")
        
        # 如果有更多文件未显示
        if len(files) > 3 and current_depth == max_depth - 1:
            print(f"{prefix}└── ... ({len(files) - 3} more files)")
    
    print(f"{patient_id}/")
    tree_format(patient_dir, "", max_depth, 0)
    
    # 统计信息
    print()
    print("="*70)
    print("统计信息:")
    print("="*70)
    
    all_dcm = list(patient_dir.rglob("*.dcm"))
    modality_count = {}
    series_info = {}
    
    for f in all_dcm:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            mod = getattr(ds, 'Modality', 'UNKNOWN')
            series_uid = str(ds.SeriesInstanceUID)
            
            modality_count[mod] = modality_count.get(mod, 0) + 1
            
            if series_uid not in series_info:
                series_info[series_uid] = {
                    'modality': mod,
                    'count': 0,
                    'description': str(getattr(ds, 'SeriesDescription', 'N/A')),
                    'dir': str(f.parent.relative_to(patient_dir))
                }
            series_info[series_uid]['count'] += 1
        except:
            pass
    
    print(f"\n总DICOM文件数: {len(all_dcm)}")
    print(f"\nModality统计:")
    for mod, count in sorted(modality_count.items()):
        print(f"  {mod}: {count} files")
    
    print(f"\nSeries统计 (共 {len(series_info)} 个series):")
    for uid, info in sorted(series_info.items(), key=lambda x: (x[1]['modality'], -x[1]['count'])):
        print(f"  [{info['modality']}] {info['description'][:50]}")
        print(f"    - 目录: {info['dir']}")
        print(f"    - 文件数: {info['count']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python show_patient_structure.py <patient_id>")
        print("示例: python show_patient_structure.py R01-022")
        sys.exit(1)
    
    patient_id = sys.argv[1]
    show_patient_structure(patient_id)


"""
显示Radiogenomics数据集中指定患者的目录结构
用法: python show_patient_structure.py R01-022
"""
import sys
from pathlib import Path
import pydicom

RAW_BASE = Path("/data0/lichengze/NSCLC_Radiogenomics/NSCLC Radiogenomics")

def show_patient_structure(patient_id: str, max_depth: int = 3):
    """显示患者的目录结构"""
    patient_dir = RAW_BASE / patient_id
    
    if not patient_dir.exists():
        print(f"错误: 患者目录不存在: {patient_dir}")
        return
    
    print("="*70)
    print(f"Radiogenomics 数据集目录结构: {patient_id}")
    print("="*70)
    print()
    
    # 使用类似tree的格式
    def tree_format(path, prefix="", max_depth=3, current_depth=0, is_last=True):
        if current_depth >= max_depth:
            return
        
        items = sorted([p for p in path.iterdir()])
        dirs = [p for p in items if p.is_dir()]
        files = [p for p in items if p.is_file() and (p.suffix == '.dcm' or p.suffix == '')]
        
        # 只显示有DICOM文件的目录
        display_dirs = []
        for d in dirs:
            if list(d.rglob("*.dcm")):
                display_dirs.append(d)
        
        # 显示文件（每个目录最多3个）
        display_files = files[:3] if current_depth == max_depth - 1 else []
        
        all_items = display_dirs + display_files
        
        for i, item in enumerate(all_items):
            is_last_item = (i == len(all_items) - 1)
            
            if item.is_dir():
                # 获取目录信息
                dcm_files = list(item.rglob("*.dcm"))
                if dcm_files:
                    try:
                        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
                        mod = getattr(ds, 'Modality', 'UNKNOWN')
                        desc = getattr(ds, 'SeriesDescription', 'N/A')
                        if len(desc) > 35:
                            desc = desc[:35] + "..."
                        info = f" [{mod}] {desc} ({len(dcm_files)} files)"
                    except:
                        info = f" ({len(dcm_files)} DICOM files)"
                else:
                    info = ""
                
                connector = "└── " if is_last_item else "├── "
                print(f"{prefix}{connector}{item.name}{info}")
                
                next_prefix = prefix + ("    " if is_last_item else "│   ")
                tree_format(item, next_prefix, max_depth, current_depth + 1, is_last_item)
            else:
                # 显示文件
                try:
                    ds = pydicom.dcmread(str(item), stop_before_pixels=True)
                    mod = getattr(ds, 'Modality', 'UNKNOWN')
                    print(f"{prefix}{'└── ' if is_last_item else '├── '}{item.name} [{mod}]")
                except:
                    print(f"{prefix}{'└── ' if is_last_item else '├── '}{item.name}")
        
        # 如果有更多文件未显示
        if len(files) > 3 and current_depth == max_depth - 1:
            print(f"{prefix}└── ... ({len(files) - 3} more files)")
    
    print(f"{patient_id}/")
    tree_format(patient_dir, "", max_depth, 0)
    
    # 统计信息
    print()
    print("="*70)
    print("统计信息:")
    print("="*70)
    
    all_dcm = list(patient_dir.rglob("*.dcm"))
    modality_count = {}
    series_info = {}
    
    for f in all_dcm:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            mod = getattr(ds, 'Modality', 'UNKNOWN')
            series_uid = str(ds.SeriesInstanceUID)
            
            modality_count[mod] = modality_count.get(mod, 0) + 1
            
            if series_uid not in series_info:
                series_info[series_uid] = {
                    'modality': mod,
                    'count': 0,
                    'description': str(getattr(ds, 'SeriesDescription', 'N/A')),
                    'dir': str(f.parent.relative_to(patient_dir))
                }
            series_info[series_uid]['count'] += 1
        except:
            pass
    
    print(f"\n总DICOM文件数: {len(all_dcm)}")
    print(f"\nModality统计:")
    for mod, count in sorted(modality_count.items()):
        print(f"  {mod}: {count} files")
    
    print(f"\nSeries统计 (共 {len(series_info)} 个series):")
    for uid, info in sorted(series_info.items(), key=lambda x: (x[1]['modality'], -x[1]['count'])):
        print(f"  [{info['modality']}] {info['description'][:50]}")
        print(f"    - 目录: {info['dir']}")
        print(f"    - 文件数: {info['count']}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python show_patient_structure.py <patient_id>")
        print("示例: python show_patient_structure.py R01-022")
        sys.exit(1)
    
    patient_id = sys.argv[1]
    show_patient_structure(patient_id)



