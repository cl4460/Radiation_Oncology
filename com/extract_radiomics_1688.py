#!/usr/bin/env python3
"""
extract_radiomics_1688.py

使用PyRadiomics提取~1688个特征以复现实验报告

使用方法:
python extract_radiomics_1688.py \
  --image_dir /path/to/ct_images \
  --mask_dir /path/to/masks \
  --output_csv /path/to/output.csv \
  --config pyradiomics_1688_config.yaml

或者如果你有现成的image/mask路径CSV:
python extract_radiomics_1688.py \
  --path_csv /path/to/paths.csv \
  --output_csv /path/to/output.csv \
  --config pyradiomics_1688_config.yaml
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

# Suppress radiomics warnings
logging.getLogger('radiomics').setLevel(logging.ERROR)

try:
    from radiomics import featureextractor
except ImportError:
    print("请安装PyRadiomics: pip install pyradiomics")
    exit(1)


def extract_features_for_patient(
    extractor,
    image_path: str,
    mask_path: str,
    patient_id: str,
) -> dict:
    """为单个患者提取特征"""
    try:
        result = extractor.execute(image_path, mask_path)
        
        # 过滤掉诊断信息，只保留特征
        features = {}
        for key, value in result.items():
            if not key.startswith('diagnostics_'):
                # 转换为Python原生类型
                if hasattr(value, 'item'):
                    features[key] = value.item()
                else:
                    features[key] = float(value) if isinstance(value, (int, float, np.number)) else value
        
        features['PatientID'] = patient_id
        return features
    
    except Exception as e:
        print(f"  警告: {patient_id} 提取失败: {e}")
        return {'PatientID': patient_id}


def main():
    parser = argparse.ArgumentParser(description='提取1688个PyRadiomics特征')
    
    # 输入方式1: 目录
    parser.add_argument('--image_dir', type=str, help='CT图像目录')
    parser.add_argument('--mask_dir', type=str, help='Mask目录')
    
    # 输入方式2: CSV文件
    parser.add_argument('--path_csv', type=str, 
                       help='包含PatientID, image_path, mask_path的CSV文件')
    
    # 输出和配置
    parser.add_argument('--output_csv', type=str, required=True, help='输出CSV路径')
    parser.add_argument('--config', type=str, default='pyradiomics_1688_config.yaml',
                       help='PyRadiomics配置文件路径')
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return
    
    # 初始化提取器
    print("初始化PyRadiomics提取器...")
    extractor = featureextractor.RadiomicsFeatureExtractor(args.config)
    
    # 获取待处理的患者列表
    patients = []
    
    if args.path_csv:
        # 从CSV读取路径
        df = pd.read_csv(args.path_csv)
        # ================== 修改开始 ==================
        # 强制将列名去除空格
        df.columns = [c.strip() for c in df.columns]
        
        for idx, row in df.iterrows():
            # 强制转换为字符串并去除首尾空格/换行符
            img_p = str(row['image_path']).strip()
            msk_p = str(row['mask_path']).strip()
            
            # 自动修正路径：如果路径包含 /Research/DeepFeature/ 但缺少 Nested_CV，则添加
            if '/Research/DeepFeature/' in img_p and '/Research/Nested_CV/DeepFeature/' not in img_p:
                img_p = img_p.replace('/Research/DeepFeature/', '/Research/Nested_CV/DeepFeature/')
            if '/Research/DeepFeature/' in msk_p and '/Research/Nested_CV/DeepFeature/' not in msk_p:
                msk_p = msk_p.replace('/Research/DeepFeature/', '/Research/Nested_CV/DeepFeature/')
            
            # 添加调试检查：如果文件不存在，直接打印出来
            if not os.path.exists(img_p):
                print(f"[错误] 找不到图像文件 (第{idx}行): '{img_p}'")
                continue # 跳过该患者
            if not os.path.exists(msk_p):
                print(f"[错误] 找不到Mask文件 (第{idx}行): '{msk_p}'")
                continue # 跳过该患者

            patients.append({
                'patient_id': str(row['PatientID']).strip(),
                'image_path': img_p,
                'mask_path': msk_p
            })
    
    elif args.image_dir and args.mask_dir:
        # 从目录读取
        image_dir = Path(args.image_dir)
        mask_dir = Path(args.mask_dir)
        
        # 支持多种格式
        for ext in ['*.nii.gz', '*.nii', '*.nrrd', '*.mha']:
            for img_path in image_dir.glob(ext):
                patient_id = img_path.stem.replace('.nii', '')
                
                # 尝试找到对应的mask
                mask_candidates = [
                    mask_dir / img_path.name,
                    mask_dir / f"{patient_id}_mask{img_path.suffix}",
                    mask_dir / f"{patient_id}.nii.gz",
                    mask_dir / f"{patient_id}_seg.nii.gz",
                ]
                
                mask_path = None
                for mc in mask_candidates:
                    if mc.exists():
                        mask_path = str(mc)
                        break
                
                if mask_path:
                    patients.append({
                        'patient_id': patient_id,
                        'image_path': str(img_path),
                        'mask_path': mask_path
                    })
    
    else:
        print("错误: 请提供 --path_csv 或 (--image_dir 和 --mask_dir)")
        return
    
    print(f"找到 {len(patients)} 个患者")
    
    if len(patients) == 0:
        print("没有找到任何患者数据!")
        return
    
    # 提取特征
    print("\n开始提取特征...")
    all_features = []
    
    for p in tqdm(patients, desc="提取进度"):
        features = extract_features_for_patient(
            extractor,
            p['image_path'],
            p['mask_path'],
            p['patient_id']
        )
        all_features.append(features)
    
    # 转换为DataFrame
    df = pd.DataFrame(all_features)
    
    # 重新排列列，PatientID在前
    cols = ['PatientID'] + [c for c in df.columns if c != 'PatientID']
    df = df[cols]
    
    # 保存
    df.to_csv(args.output_csv, index=False)
    
    print(f"\n完成!")
    print(f"  患者数: {len(df)}")
    print(f"  特征数: {len(df.columns) - 1}")
    print(f"  保存到: {args.output_csv}")


if __name__ == '__main__':
    main()
