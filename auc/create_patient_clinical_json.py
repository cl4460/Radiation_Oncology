#!/usr/bin/env python3
"""
Helper script to create clinical JSON file for a new patient.
"""
import json
import argparse

# Encoding maps
GENDER_MAP = {"male": 1, "female": 2, "unknown": 0}
HISTOLOGY_MAP = {
    "large cell": 0,
    "nos": 1,
    "squamous cell": 2,
    "adenocarcinoma": 3,
    "other": 4,
    "unknown": 0
}

def create_clinical_json(args):
    """Create a clinical JSON file for a new patient."""
    
    # Map gender
    gender_code = GENDER_MAP.get(args.gender.lower(), 0)
    
    # Map histology
    histology_code = HISTOLOGY_MAP.get(args.histology.lower(), 0)
    
    clinical_data = {
        "PatientID": args.patient_id,
        "age": args.age,
        "gender": gender_code,
        "Histology": histology_code,
        "clinical.T.Stage": args.t_stage,
        "Clinical.N.Stage": args.n_stage,
        "Clinical.M.Stage": args.m_stage,
        "Overall.Stage": args.overall_stage
    }
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(clinical_data, f, indent=2)
    
    print("=" * 60)
    print("✅ Created clinical JSON file")
    print("=" * 60)
    print(f"Output: {args.output}")
    print(f"\nPatient Information:")
    print(f"  ID: {args.patient_id}")
    print(f"  Age: {args.age}")
    print(f"  Gender: {args.gender} (code: {gender_code})")
    print(f"  Histology: {args.histology} (code: {histology_code})")
    print(f"  TNM Stage: T{args.t_stage} N{args.n_stage} M{args.m_stage}")
    print(f"  Overall Stage: {args.overall_stage}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Create clinical JSON for a new patient",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create JSON for a 66-year-old male with adenocarcinoma, T2N1M0, Stage III
  python create_patient_clinical_json.py \\
    --patient_id NEW001 \\
    --age 66 \\
    --gender male \\
    --histology adenocarcinoma \\
    --t_stage 2 \\
    --n_stage 1 \\
    --m_stage 0 \\
    --overall_stage 3 \\
    --output patient_NEW001_clinical.json

Encoding Rules:
  Gender: male=1, female=2, unknown=0
  Histology: large_cell=0, nos=1, squamous_cell=2, adenocarcinoma=3, other=4
  Stages: 0-5 (where 5 usually means unknown/X)
""")
    
    parser.add_argument("--patient_id", required=True, help="Patient identifier")
    parser.add_argument("--age", type=int, required=True, help="Age in years")
    parser.add_argument("--gender", required=True, 
                       choices=["male", "female", "unknown"],
                       help="Patient gender")
    parser.add_argument("--histology", required=True,
                       choices=["large cell", "nos", "squamous cell", "adenocarcinoma", "other", "unknown"],
                       help="Tumor histology type")
    parser.add_argument("--t_stage", type=int, required=True,
                       help="T stage (0-5, where 5=TX/unknown)")
    parser.add_argument("--n_stage", type=int, required=True,
                       help="N stage (0-5, where 5=NX/unknown)")
    parser.add_argument("--m_stage", type=int, required=True,
                       help="M stage (0-3)")
    parser.add_argument("--overall_stage", type=int, required=True,
                       help="Overall stage (0-5)")
    parser.add_argument("--output", required=True,
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Validate stages
    if not (0 <= args.t_stage <= 5):
        parser.error("T stage must be 0-5")
    if not (0 <= args.n_stage <= 5):
        parser.error("N stage must be 0-5")
    if not (0 <= args.m_stage <= 3):
        parser.error("M stage must be 0-3")
    if not (0 <= args.overall_stage <= 5):
        parser.error("Overall stage must be 0-5")
    if not (0 <= args.age <= 120):
        parser.error("Age must be 0-120")
    
    create_clinical_json(args)

if __name__ == "__main__":
    main()



