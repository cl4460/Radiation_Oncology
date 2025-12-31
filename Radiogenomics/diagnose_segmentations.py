#!/usr/bin/env python3
"""
Diagnose NSCLC-Radiogenomics dataset segmentation availability
Run this to see which patients have RTSTRUCT or DICOM-SEG files
"""
import os, glob
from pathlib import Path
import pydicom
from collections import defaultdict
import pandas as pd

RAW_BASE = "/data0/lichengze/NSCLC_Radiogenomics/NSCLC Radiogenomics"

def check_patient_segmentations(patient_dir):
    """Check if a patient has RTSTRUCT or SEG files"""
    patient_id = Path(patient_dir).name
    
    has_rtstruct = False
    has_seg = False
    has_ct = False
    total_dicom = 0
    
    modality_counts = defaultdict(int)
    series_info = {}
    
    # Scan all DICOM files
    for dcm_file in Path(patient_dir).rglob("*"):
        if not dcm_file.is_file() or dcm_file.name.startswith('.'):
            continue
            
        try:
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            mod = str(getattr(ds, 'Modality', 'UNKNOWN'))
            series_uid = str(ds.SeriesInstanceUID)
            
            modality_counts[mod] += 1
            total_dicom += 1
            
            if series_uid not in series_info:
                series_info[series_uid] = {
                    'modality': mod,
                    'count': 0,
                    'description': str(getattr(ds, 'SeriesDescription', 'N/A'))
                }
            series_info[series_uid]['count'] += 1
            
            if mod == 'RTSTRUCT':
                has_rtstruct = True
            elif mod in ('SEG', 'RTSEG'):
                has_seg = True
            elif mod == 'CT':
                has_ct = True
                
        except:
            continue
    
    return {
        'patient_id': patient_id,
        'has_ct': has_ct,
        'has_rtstruct': has_rtstruct,
        'has_seg': has_seg,
        'total_dicom': total_dicom,
        'num_series': len(series_info),
        'modalities': dict(modality_counts)
    }

def main():
    print("="*70)
    print("NSCLC-Radiogenomics Segmentation Availability Check")
    print("="*70)
    
    # Get all patient directories
    pats_amc = sorted(glob.glob(os.path.join(RAW_BASE, "AMC-*")))
    pats_r01 = sorted(glob.glob(os.path.join(RAW_BASE, "R01-*")))
    all_patients = pats_amc + pats_r01
    
    print(f"\nFound {len(pats_amc)} AMC-* and {len(pats_r01)} R01-* patients")
    print(f"Total: {len(all_patients)} patients\n")
    
    results = []
    
    print("Scanning patients...")
    for i, pdir in enumerate(all_patients, 1):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(all_patients)}")
        
        result = check_patient_segmentations(pdir)
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total patients:           {len(df)}")
    print(f"Patients with CT:         {df['has_ct'].sum()}")
    print(f"Patients with RTSTRUCT:   {df['has_rtstruct'].sum()}")
    print(f"Patients with DICOM-SEG:  {df['has_seg'].sum()}")
    print(f"Patients with ANY seg:    {(df['has_rtstruct'] | df['has_seg']).sum()}")
    print(f"Patients with NO seg:     {(~df['has_rtstruct'] & ~df['has_seg']).sum()}")
    
    # Save detailed report
    output_file = "radiogenomics_segmentation_check.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed report saved to: {output_file}")
    
    # Show sample of patients WITH segmentations
    with_seg = df[df['has_rtstruct'] | df['has_seg']]
    if len(with_seg) > 0:
        print(f"\nSample patients WITH segmentations (showing first 10):")
        for idx, row in with_seg.head(10).iterrows():
            seg_type = "RTSTRUCT" if row['has_rtstruct'] else "SEG"
            print(f"  {row['patient_id']:15s} - {seg_type}")
    else:
        print("\n⚠ WARNING: NO PATIENTS HAVE SEGMENTATIONS!")
        print("\nPossible issues:")
        print("  1. Segmentations not downloaded from TCIA")
        print("  2. Segmentations in separate files/folders not included in download")
        print("  3. Need to download additional annotations from external source")
    
    # Show sample of patients WITHOUT segmentations
    without_seg = df[~df['has_rtstruct'] & ~df['has_seg']]
    if len(without_seg) > 0:
        print(f"\nSample patients WITHOUT segmentations (showing first 10):")
        for idx, row in without_seg.head(10).iterrows():
            print(f"  {row['patient_id']:15s} - CT: {'✓' if row['has_ct'] else '✗'}, "
                  f"DICOM files: {row['total_dicom']}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()