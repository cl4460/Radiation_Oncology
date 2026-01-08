#!/usr/bin/env python3
"""
External validation on NSCLC-Radiogenomics using the RETRAINED GBSA model
that successfully reproduced the internal C-index.

This script:
1. Loads the retrained model from retrain_gbsa91/
2. Loads preprocessing artifacts (scaler, encoder, fill_values)
3. Applies the same preprocessing to RG external data
4. Makes predictions and calculates C-index
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sksurv.metrics import concordance_index_censored

# ============================================================================
# Configuration
# ============================================================================
RETRAIN_DIR = Path("retrain_gbsa91")
RG_CLINICAL_CSV = Path("rg_clinical.csv")
RG_RADIOMICS_CSV = Path("rg_radiomics_fixed.csv")
OUTPUT_DIR = Path("external_validation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# Helper functions for categorical encoding
# ============================================================================
_GENDER_MAP = {"male": 1, "female": 2}
_HISTO_MAP = {
    "large cell": 0,
    "nos": 1,
    "squamous cell": 2,
    "adenocarcinoma": 3,
    "other": 4,
}
_STAGE_MAP = {"0": 0, "i": 1, "ii": 2, "iii": 3, "iv": 4, "x": 5}

def _canon_str(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower()

def _canon_gender(s):
    c = _canon_str(s)
    return _GENDER_MAP.get(c, 0)

def _canon_histology(s):
    c = _canon_str(s)
    for k, v in _HISTO_MAP.items():
        if k in c:
            return v
    return 0

def _parse_stage_int(s):
    """Parse T/N/M stage strings like 'T1a', '1a', 'T2' into integer codes."""
    if pd.isna(s):
        return 0
    c = _canon_str(s)
    # Remove 'T', 'N', 'M' prefix
    c = c.replace('t', '').replace('n', '').replace('m', '')
    # Remove suffixes like 'a', 'b', 'c'
    c = c.replace('a', '').replace('b', '').replace('c', '')
    # Try to parse as integer
    if c in ['0', '1', '2', '3', '4', '5']:
        return int(c)
    # Special case for 'x'
    if c == 'x':
        return 5
    return 0

def _canon_overall_stage(s):
    c = _canon_str(s)
    # Remove prefix like "stage "
    c = c.replace("stage", "").strip()
    # Remove suffixes
    c = c.replace('a', '').replace('b', '').replace('c', '')
    return _STAGE_MAP.get(c, 0)

def _encode_cat_int(df, clinical_fill, cat_cols):
    """
    Encode categorical features as integers, matching the training script logic.
    
    Key logic:
    - gender: map strings to codes, fill missing with clinical_fill['gender']
    - Histology: map strings to codes, fill missing with clinical_fill['Histology']
    - Overall.Stage: map strings to codes, fill missing with clinical_fill['Overall.Stage']
    - Clinical.T/N/M.Stage: parse strings, fill missing with 0
    """
    X_cat = pd.DataFrame(index=df.index)
    
    for c in cat_cols:
        if c not in df.columns:
            # If column doesn't exist, fill with default
            if c == 'gender':
                X_cat[c] = clinical_fill.get('gender', 0)
            elif c == 'Histology':
                X_cat[c] = clinical_fill.get('Histology', 0)
            elif c == 'Overall.Stage':
                X_cat[c] = clinical_fill.get('Overall.Stage', 0)
            elif c in ['Clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage']:
                X_cat[c] = 0
            else:
                X_cat[c] = 0
        else:
            raw = df[c].copy()
            
            if c == 'gender':
                X_cat[c] = raw.apply(_canon_gender)
                X_cat[c] = X_cat[c].fillna(clinical_fill.get('gender', 0))
            elif c == 'Histology':
                X_cat[c] = raw.apply(_canon_histology)
                X_cat[c] = X_cat[c].fillna(clinical_fill.get('Histology', 0))
            elif c == 'Overall.Stage':
                X_cat[c] = raw.apply(_canon_overall_stage)
                X_cat[c] = X_cat[c].fillna(clinical_fill.get('Overall.Stage', 0))
            elif c in ['Clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage']:
                # For T/N/M stages, parse and fill with 0
                X_cat[c] = raw.apply(_parse_stage_int)
                X_cat[c] = X_cat[c].fillna(0)
            else:
                # Unknown categorical column, try to convert to int
                X_cat[c] = pd.to_numeric(raw, errors='coerce').fillna(0)
    
    # Ensure all are integers
    X_cat = X_cat.astype(int)
    return X_cat


# ============================================================================
# Load retrained model and artifacts
# ============================================================================
def load_retrained_artifacts():
    """Load the retrained model and preprocessing artifacts."""
    print("=" * 80)
    print("Loading retrained model and artifacts...")
    print("=" * 80)
    
    # Load model
    model_path = RETRAIN_DIR / "result_Clinic_Radiomics_gbsa_scratch.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model summary not found: {model_path}")
    
    with open(model_path) as f:
        model_info = json.load(f)[0]
    
    print(f"Model: {model_info['model']}")
    print(f"Task: {model_info['task']}")
    print(f"Train C-index: {model_info['train_cindex']:.4f}")
    print(f"Test C-index: {model_info['test_cindex']:.4f}")
    print(f"Selected features: {len(model_info['selected_features'])}")
    
    # The reproduce script saves artifacts in a specific location
    # We need to find where it saved the model, scaler, encoder, etc.
    # Based on the script structure, artifacts should be in scratch_out/
    
    artifacts_dir = Path("scratch_out/Clinic_Radiomics/gbsa")
    if not artifacts_dir.exists():
        # Try alternate location
        artifacts_dir = Path("scratch_out")
        if not artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
    
    # Find the latest artifacts
    model_file = None
    normalizer_file = None
    fill_file = None
    
    # Search for files
    for root, dirs, files in artifacts_dir.walk():
        for f in files:
            if f.endswith("_model.joblib"):
                model_file = root / f
            elif f == "normalizer.joblib":
                normalizer_file = root / f
            elif f == "train_fill_values.joblib":
                fill_file = root / f
    
    # If not found, try loading from reproduce script's expected locations
    # The script uses a timestamp-based naming, so we'll look for the pattern
    if model_file is None or normalizer_file is None or fill_file is None:
        print("\n⚠️  Artifacts not found in scratch_out/, will use reproduce script's API")
        return None
    
    print(f"\nLoading model from: {model_file}")
    print(f"Loading normalizer from: {normalizer_file}")
    print(f"Loading fill values from: {fill_file}")
    
    model = joblib.load(model_file)
    normalizer = joblib.load(normalizer_file)
    fill_values = joblib.load(fill_file)
    
    artifacts = {
        'model': model,
        'scaler': normalizer['normalizers']['numeric'],
        'encoder': normalizer['normalizers']['categorical'],
        'num_cols': normalizer['columns']['numeric'],
        'cat_cols': normalizer['columns']['categorical'],
        'fill_values': fill_values,
        'selected_features': model_info['selected_features'],
        'model_params': model_info['model_params'],
    }
    
    return artifacts


# ============================================================================
# Load and preprocess external data
# ============================================================================
def load_external_data():
    """Load RG clinical and radiomics data."""
    print("\n" + "=" * 80)
    print("Loading external validation data (NSCLC-Radiogenomics)...")
    print("=" * 80)
    
    clinical_df = pd.read_csv(RG_CLINICAL_CSV)
    print(f"Clinical data: {clinical_df.shape}")
    print(f"Columns: {clinical_df.columns.tolist()}")
    
    radiomics_df = pd.read_csv(RG_RADIOMICS_CSV)
    print(f"Radiomics data: {radiomics_df.shape}")
    
    # Handle patient ID column
    if 'PatientID' in radiomics_df.columns:
        pid_col = 'PatientID'
    elif 'patient_id' in radiomics_df.columns:
        pid_col = 'patient_id'
        radiomics_df.rename(columns={'patient_id': 'PatientID'}, inplace=True)
    else:
        raise ValueError("Radiomics CSV must contain PatientID or patient_id")
    
    # Merge clinical and radiomics
    joint_df = pd.merge(
        clinical_df,
        radiomics_df,
        left_on='Case ID',
        right_on='PatientID',
        how='inner'
    )
    print(f"Merged data: {joint_df.shape}")
    
    return joint_df


def preprocess_external_data(joint_df, artifacts):
    """Preprocess external data using the same pipeline as training."""
    print("\n" + "=" * 80)
    print("Preprocessing external data...")
    print("=" * 80)
    
    scaler = artifacts['scaler']
    encoder = artifacts['encoder']
    num_cols = artifacts['num_cols']
    cat_cols = artifacts['cat_cols']
    fill_values = artifacts['fill_values']
    selected_features = artifacts['selected_features']
    
    # Separate numeric and categorical fill values
    clinical_fill = {k: v for k, v in fill_values.items() 
                     if k in ['age', 'gender', 'Histology', 'Overall.Stage',
                             'Clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage']}
    radiomics_fill = {k: v for k, v in fill_values.items() if k not in clinical_fill}
    
    print(f"Clinical fill values: {clinical_fill}")
    print(f"Radiomics fill values: {len(radiomics_fill)} features")
    
    # Build numeric features
    X_num = pd.DataFrame(index=joint_df.index)
    for c in num_cols:
        if c == 'age':
            # Age comes from clinical data
            if 'Age at Histological Diagnosis' in joint_df.columns:
                X_num[c] = joint_df['Age at Histological Diagnosis']
            else:
                X_num[c] = fill_values.get('age', 65.0)
            X_num[c] = X_num[c].fillna(fill_values.get('age', 65.0))
        elif c in joint_df.columns:
            X_num[c] = joint_df[c]
            X_num[c] = X_num[c].fillna(radiomics_fill.get(c, 0.0))
        else:
            # Feature not in external data, fill with training mean
            X_num[c] = radiomics_fill.get(c, 0.0)
    
    # Build categorical features
    # Map external column names to training column names
    clinical_mapping = {
        'Gender': 'gender',
        'Histology': 'Histology',
        'Clinical Stage': 'Overall.Stage',
        'Clinical T Stage': 'Clinical.T.Stage',
        'Clinical N Stage': 'Clinical.N.Stage',
        'Clinical M Stage': 'Clinical.M.Stage',
    }
    
    # Create a temporary df with standardized column names
    temp_df = pd.DataFrame(index=joint_df.index)
    for ext_col, train_col in clinical_mapping.items():
        if ext_col in joint_df.columns:
            temp_df[train_col] = joint_df[ext_col]
    
    X_cat_int = _encode_cat_int(temp_df, clinical_fill, cat_cols)
    
    # Apply scaler to numeric features
    print(f"\nScaling {len(num_cols)} numeric features...")
    X_num_scaled = pd.DataFrame(
        scaler.transform(X_num),
        columns=num_cols,
        index=X_num.index
    )
    
    # Apply encoder to categorical features
    print(f"Encoding {len(cat_cols)} categorical features...")
    X_cat_encoded = encoder.transform(X_cat_int)
    
    # Get feature names from encoder
    cat_feature_names = []
    for i, cat_col in enumerate(cat_cols):
        n_categories = len(encoder.categories_[i])
        for j in range(n_categories):
            cat_feature_names.append(f"{cat_col}_{j}")
    
    X_cat_encoded_df = pd.DataFrame(
        X_cat_encoded,
        columns=cat_feature_names,
        index=X_cat_int.index
    )
    
    # Concatenate numeric and categorical
    X_all = pd.concat([X_num_scaled, X_cat_encoded_df], axis=1)
    print(f"Total features after preprocessing: {X_all.shape[1]}")
    
    # Select only features used by the model
    print(f"\nSelecting {len(selected_features)} features used by the model...")
    missing_features = [f for f in selected_features if f not in X_all.columns]
    if missing_features:
        print(f"⚠️  Missing {len(missing_features)} features, will fill with 0:")
        for f in missing_features[:10]:
            print(f"  - {f}")
        if len(missing_features) > 10:
            print(f"  ... and {len(missing_features) - 10} more")
        # Add missing features with 0
        for f in missing_features:
            X_all[f] = 0.0
    
    X_final = X_all[selected_features]
    
    print(f"Final feature matrix shape: {X_final.shape}")
    
    return X_final, joint_df


# ============================================================================
# Make predictions and evaluate
# ============================================================================
def make_predictions(X, artifacts, joint_df):
    """Make predictions using the retrained model."""
    print("\n" + "=" * 80)
    print("Making predictions on external data...")
    print("=" * 80)
    
    model = artifacts['model']
    
    # Get predictions
    pred_risk = model.predict(X)
    
    print(f"Predictions shape: {pred_risk.shape}")
    print(f"Predictions stats:")
    print(f"  Mean: {pred_risk.mean():.4f}")
    print(f"  Std: {pred_risk.std():.4f}")
    print(f"  Min: {pred_risk.min():.4f}")
    print(f"  Max: {pred_risk.max():.4f}")
    
    # Get survival outcomes
    if 'Survival Status' in joint_df.columns:
        event_col = 'Survival Status'
        events = joint_df[event_col].map({'Dead': 1, 'Alive': 0}).values
    elif 'deadstatus.event' in joint_df.columns:
        events = joint_df['deadstatus.event'].astype(bool).values
    else:
        raise ValueError("Cannot find event column in external data")
    
    if 'Survival Days' in joint_df.columns:
        time_col = 'Survival Days'
        times = joint_df[time_col].values
    elif 'time' in joint_df.columns:
        times = joint_df['time'].values
    else:
        raise ValueError("Cannot find time column in external data")
    
    # Calculate C-index
    cindex = concordance_index_censored(events, times, pred_risk)[0]
    
    print(f"\n{'=' * 80}")
    print(f"EXTERNAL VALIDATION C-INDEX: {cindex:.4f}")
    print(f"{'=' * 80}")
    
    # Save predictions
    results_df = pd.DataFrame({
        'PatientID': joint_df['Case ID'].values if 'Case ID' in joint_df.columns else joint_df.index,
        'time': times,
        'event': events,
        'pred_risk': pred_risk,
    })
    
    output_file = OUTPUT_DIR / "external_predictions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved predictions to: {output_file}")
    
    # Save summary
    summary = {
        'external_dataset': 'NSCLC-Radiogenomics',
        'n_samples': len(X),
        'n_features': len(artifacts['selected_features']),
        'external_cindex': float(cindex),
        'internal_train_cindex': artifacts['model_params'].get('train_cindex', 'N/A'),
        'internal_test_cindex': artifacts['model_params'].get('test_cindex', 'N/A'),
    }
    
    summary_file = OUTPUT_DIR / "validation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_file}")
    
    return cindex, results_df


# ============================================================================
# Main
# ============================================================================
def main():
    try:
        # Load artifacts
        artifacts = load_retrained_artifacts()
        
        if artifacts is None:
            print("\n⚠️  Cannot load artifacts directly. Using reproduce script's API instead...")
            print("Please run reproduce script in 'predict' mode first to generate artifacts.")
            return 1
        
        # Load external data
        joint_df = load_external_data()
        
        # Preprocess
        X, joint_df = preprocess_external_data(joint_df, artifacts)
        
        # Predict
        cindex, results_df = make_predictions(X, artifacts, joint_df)
        
        print("\n" + "=" * 80)
        print("✅ External validation completed successfully!")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

