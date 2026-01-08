#!/usr/bin/env python3
"""
Extract radiomics features for a single patient using PyRadiomics.
"""
import argparse
import pandas as pd
from pathlib import Path
from radiomics import featureextractor

def extract_radiomics(args):
    """Extract radiomics features for one patient."""
    
    print("=" * 70)
    print("Radiomics Feature Extraction for Single Patient")
    print("=" * 70)
    print(f"Patient ID: {args.patient_id}")
    print(f"CT Image: {args.ct_path}")
    print(f"Mask: {args.mask_path}")
    print(f"Config: {args.yaml_config}")
    print(f"Label: {args.label}")
    print("=" * 70)
    
    # Check files exist
    if not Path(args.ct_path).exists():
        raise FileNotFoundError(f"CT image not found: {args.ct_path}")
    if not Path(args.mask_path).exists():
        raise FileNotFoundError(f"Mask not found: {args.mask_path}")
    if not Path(args.yaml_config).exists():
        raise FileNotFoundError(f"Config not found: {args.yaml_config}")
    
    # Initialize extractor
    print("\n[1/3] Initializing PyRadiomics extractor...")
    extractor = featureextractor.RadiomicsFeatureExtractor(args.yaml_config)
    print("✅ Extractor initialized")
    
    # Extract features
    print("\n[2/3] Extracting features (this may take a minute)...")
    try:
        result = extractor.execute(args.ct_path, args.mask_path, label=args.label)
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        raise
    
    # Filter out diagnostics
    features = {k: v for k, v in result.items() if not k.startswith('diagnostics_')}
    features['PatientID'] = args.patient_id
    
    print(f"✅ Extracted {len(features)-1} radiomics features")
    
    # Show some example features
    print("\nExample features (first 5):")
    for i, (k, v) in enumerate(list(features.items())[:5]):
        if k != 'PatientID':
            print(f"  {k}: {v}")
    
    # Save to CSV
    print(f"\n[3/3] Saving to CSV...")
    df = pd.DataFrame([features])
    df.to_csv(args.output, index=False)
    
    print("=" * 70)
    print(f"✅ SUCCESS!")
    print("=" * 70)
    print(f"Output file: {args.output}")
    print(f"Total features: {len(features)-1} (+ PatientID)")
    print(f"CSV shape: {df.shape}")
    print("=" * 70)
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Extract radiomics features for a single patient",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python extract_single_patient_radiomics.py \\
    --patient_id NEW001 \\
    --ct_path /path/to/CT.nii.gz \\
    --mask_path /path/to/mask.nii.gz \\
    --yaml_config /home/lichengze/Research/com/radiomic_setting.yaml \\
    --output patient_NEW001_radiomics.csv \\
    --label 255

Note:
  - Make sure the mask label value matches your configuration
  - LUNG1 dataset uses label=255
  - NSCLC-Radiogenomics dataset uses label=1
""")
    
    parser.add_argument("--patient_id", required=True,
                       help="Patient identifier")
    parser.add_argument("--ct_path", required=True,
                       help="Path to CT image (NIfTI format)")
    parser.add_argument("--mask_path", required=True,
                       help="Path to segmentation mask (NIfTI format)")
    parser.add_argument("--yaml_config", required=True,
                       help="Path to PyRadiomics configuration YAML")
    parser.add_argument("--label", type=int, default=255,
                       help="Mask label value to use (default: 255)")
    parser.add_argument("--output", required=True,
                       help="Output CSV file path")
    
    args = parser.parse_args()
    
    try:
        extract_radiomics(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())



