#!/usr/bin/env python3
"""
Debug script to inspect what the zip's preprocessors expect as input.
This will reveal the exact categories, feature order, and encoding expected by the saved model.

Run this in your zip_gbsa_infer_py310 environment:
  python debug_zip_preprocessing.py
"""

import zipfile
import tempfile
import os
import json
import joblib
import pandas as pd
import numpy as np

def main():
    zip_path = "/home/lichengze/Research/com/Model.zip"
    task = "Clinic+Radiomics"
    seed = 91
    model = "GradientBoostingSurvivalAnalysis"
    
    base = f"{task}/seed_{seed}"
    model_base = f"{base}/{model}"
    
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        
        print("=" * 70)
        print("1. ZIP CONTENTS (relevant files)")
        print("=" * 70)
        for n in sorted(names):
            if base in n:
                info = z.getinfo(n)
                print(f"  {n}  ({info.file_size} bytes)")
        
        print("\n" + "=" * 70)
        print("2. NORMALIZER.JOBLIB CONTENTS")
        print("=" * 70)
        
        with tempfile.TemporaryDirectory() as tmp:
            # Load normalizer
            norm_path = f"{base}/normalizer.joblib"
            z.extract(norm_path, tmp)
            normalizer = joblib.load(os.path.join(tmp, norm_path))
            
            print(f"\nKeys in normalizer dict: {list(normalizer.keys())}")
            
            # Numeric columns
            print(f"\n--- numeric_cols ---")
            num_cols = normalizer.get('numeric_cols', [])
            print(f"Count: {len(num_cols)}")
            print(f"First 10: {num_cols[:10]}")
            
            # Category columns
            print(f"\n--- category_columns ---")
            cat_cols = normalizer.get('category_columns', [])
            print(f"Count: {len(cat_cols)}")
            print(f"Columns: {cat_cols}")
            
            # Scaler
            print(f"\n--- Numeric Scaler ---")
            scaler = normalizer['normalizers']['numeric']
            print(f"Type: {type(scaler)}")
            if hasattr(scaler, 'feature_names_in_'):
                print(f"feature_names_in_: {list(scaler.feature_names_in_[:10])}...")
            if hasattr(scaler, 'n_features_in_'):
                print(f"n_features_in_: {scaler.n_features_in_}")
            
            # Encoder
            print(f"\n--- Categorical Encoder ---")
            encoder = normalizer['normalizers']['categorical']
            print(f"Type: {type(encoder)}")
            if hasattr(encoder, 'feature_names_in_'):
                print(f"feature_names_in_: {list(encoder.feature_names_in_)}")
            
            print(f"\n*** CRITICAL: encoder.categories_ ***")
            print("This shows what INTEGER VALUES the encoder expects for each column:")
            if hasattr(encoder, 'categories_'):
                for i, cats in enumerate(encoder.categories_):
                    col = encoder.feature_names_in_[i] if hasattr(encoder, 'feature_names_in_') else f"col_{i}"
                    print(f"  [{i}] {col}: {list(cats)} (dtype={cats.dtype})")
            
            # Get encoded feature names
            if hasattr(encoder, 'get_feature_names_out'):
                out_names = encoder.get_feature_names_out(cat_cols)
                print(f"\nEncoded feature names ({len(out_names)}): {list(out_names)}")
        
        print("\n" + "=" * 70)
        print("3. TRAIN_FILL_VALUES.JOBLIB")
        print("=" * 70)
        
        with tempfile.TemporaryDirectory() as tmp:
            fill_path = f"{base}/train_fill_values.joblib"
            z.extract(fill_path, tmp)
            fill_obj = joblib.load(os.path.join(tmp, fill_path))
            
            print(f"Type: {type(fill_obj)}")
            if isinstance(fill_obj, dict):
                print(f"Top-level keys: {list(fill_obj.keys())}")
                if 'fill_values' in fill_obj:
                    fv = fill_obj['fill_values']
                    print(f"\nfill_values dict ({len(fv)} entries):")
                    # Print clinical-related ones
                    for k, v in fv.items():
                        if any(x in k.lower() for x in ['age', 'gender', 'histology', 'stage', 'clinical', 'overall']):
                            print(f"  {k}: {v} (type={type(v).__name__})")
        
        print("\n" + "=" * 70)
        print("4. SELECTED FEATURES")
        print("=" * 70)
        
        with tempfile.TemporaryDirectory() as tmp:
            sel_path = f"{model_base}/selected_features_{model}.xlsx"
            z.extract(sel_path, tmp)
            sel_df = pd.read_excel(os.path.join(tmp, sel_path))
            
            print(f"Columns in xlsx: {list(sel_df.columns)}")
            for col in sel_df.columns:
                feats = sel_df[col].dropna().tolist()
                print(f"\n{col} ({len(feats)} features):")
                for f in feats:
                    print(f"  - {f}")
        
        print("\n" + "=" * 70)
        print("5. FITTED MODEL feature_names_in_")
        print("=" * 70)
        
        with tempfile.TemporaryDirectory() as tmp:
            # Find model joblib
            model_joblobs = [n for n in names if n.startswith(model_base) and n.endswith('.joblib') and 'surv_fn' not in n.lower()]
            print(f"Found joblobs: {model_joblobs}")
            
            for mjp in model_joblobs:
                z.extract(mjp, tmp)
                model_obj = joblib.load(os.path.join(tmp, mjp))
                print(f"\n{os.path.basename(mjp)}:")
                print(f"  Type: {type(model_obj)}")
                if hasattr(model_obj, 'feature_names_in_'):
                    fni = list(model_obj.feature_names_in_)
                    print(f"  feature_names_in_ ({len(fni)} features):")
                    for i, f in enumerate(fni):
                        print(f"    [{i}] {f}")
                if hasattr(model_obj, 'n_features_in_'):
                    print(f"  n_features_in_: {model_obj.n_features_in_}")
        
        print("\n" + "=" * 70)
        print("6. VERIFY: Load Lung1 test data and check preprocessing")
        print("=" * 70)
        
        # Load Lung1 clinical and check what values are in the categorical columns
        lung1_clinical = "/home/lichengze/Research/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
        lung1_df = pd.read_csv(lung1_clinical)
        
        print("\nLung1 clinical categorical column values:")
        for col in ['gender', 'Histology', 'Overall.Stage', 'clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage']:
            if col in lung1_df.columns:
                vals = lung1_df[col].dropna().unique()
                print(f"\n  {col}: {sorted(vals)[:20]}")
                print(f"    dtype: {lung1_df[col].dtype}")
        
        print("\n" + "=" * 70)
        print("7. TEST: Try to reproduce the exact preprocessing")
        print("=" * 70)
        
        with tempfile.TemporaryDirectory() as tmp:
            # Load all artifacts
            norm_path = f"{base}/normalizer.joblib"
            z.extract(norm_path, tmp)
            normalizer = joblib.load(os.path.join(tmp, norm_path))
            
            fill_path = f"{base}/train_fill_values.joblib"
            z.extract(fill_path, tmp)
            fill_obj = joblib.load(os.path.join(tmp, fill_path))
            fill_values = fill_obj.get('fill_values', fill_obj)
            
            # Load model
            model_joblobs = [n for n in names if n.startswith(model_base) and n.endswith('.joblib') and 'surv_fn' not in n.lower()]
            model_joblobs.sort(key=lambda n: z.getinfo(n).file_size, reverse=True)
            z.extract(model_joblobs[0], tmp)
            gbsa = joblib.load(os.path.join(tmp, model_joblobs[0]))
            
            # Load test_preds
            test_path = f"{model_base}/test_preds_{model}.xlsx"
            z.extract(test_path, tmp)
            test_df = pd.read_excel(os.path.join(tmp, test_path))
            # Column name is 'pid', not 'PatientID'
            pid_col = 'pid' if 'pid' in test_df.columns else 'PatientID'
            test_df[pid_col] = test_df[pid_col].astype(str)
            
            # Get scaler/encoder
            scaler = normalizer['normalizers']['numeric']
            encoder = normalizer['normalizers']['categorical']
            num_cols = normalizer.get('numeric_cols', [])
            cat_cols = normalizer.get('category_columns', [])
            
            # Load Lung1 data
            lung1_clin = pd.read_csv(lung1_clinical)
            lung1_clin['PatientID'] = lung1_clin['PatientID'].astype(str)
            lung1_clin = lung1_clin.set_index('PatientID')
            
            lung1_rad = pd.read_csv("/home/lichengze/Research/com/Tb_features_1688_from_yaml.csv")
            lung1_rad['PatientID'] = lung1_rad['PatientID'].astype(str)
            lung1_rad = lung1_rad.set_index('PatientID')
            
            # Get test IDs from test_preds (column name is 'pid', not 'PatientID')
            pid_col = 'pid' if 'pid' in test_df.columns else 'PatientID'
            test_ids = test_df[pid_col].astype(str).tolist()
            
            # Merge clinical + radiomics for test set
            joint = lung1_clin.loc[lung1_clin.index.intersection(test_ids)].join(
                lung1_rad.loc[lung1_rad.index.intersection(test_ids)],
                how='inner'
            )
            print(f"\nTest set size: {len(joint)}")
            
            # Apply fill values
            for k, v in fill_values.items():
                if k in joint.columns:
                    joint[k] = joint[k].fillna(v)
            
            # Prepare numeric features
            X_num = joint[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            
            # Prepare categorical features
            X_cat_raw = joint[cat_cols].copy()
            
            print("\n--- Categorical values in test set BEFORE encoding ---")
            for col in cat_cols:
                vals = X_cat_raw[col].unique()
                try:
                    sorted_vals = sorted([str(v) for v in vals])[:10]
                except:
                    sorted_vals = list(vals)[:10]
                print(f"  {col}: {sorted_vals}")
                print(f"    dtype: {X_cat_raw[col].dtype}")
            
            # Convert categorical to proper format for encoder
            # The encoder expects INTEGER values, not strings!
            print("\n--- Checking encoder input requirements ---")
            print(f"Encoder categories_ dtypes:")
            for i, cats in enumerate(encoder.categories_):
                print(f"  {cat_cols[i]}: {cats.dtype}, values={list(cats)}")
            
            # Step 1: Convert strings to integers using the same mapping as training
            print("\n--- Step 1: Converting strings to integers ---")
            X_cat_int = pd.DataFrame(index=X_cat_raw.index)
            
            # Gender mapping
            if "gender" in cat_cols:
                g = X_cat_raw["gender"].map(lambda x: str(x).strip().lower())
                X_cat_int["gender"] = g.map({"female": 0, "male": 1}).fillna(1).astype(int)
                print(f"  gender: {sorted(X_cat_int['gender'].unique())}")
            
            # Histology mapping
            if "Histology" in cat_cols:
                h = X_cat_raw["Histology"].map(lambda x: str(x).strip().lower() if pd.notna(x) else "")
                histo_map = {
                    "adenocarcinoma": 0,
                    "large cell": 1,
                    "nos": 2,
                    "squamous cell carcinoma": 3,
                }
                X_cat_int["Histology"] = h.map(histo_map).fillna(4).astype(int)
                print(f"  Histology: {sorted(X_cat_int['Histology'].unique())}")
            
            # Overall.Stage mapping
            if "Overall.Stage" in cat_cols:
                st = X_cat_raw["Overall.Stage"].map(lambda x: str(x).strip().upper() if pd.notna(x) else "")
                stage_map = {"I": 0, "II": 1, "IIIA": 2, "IIIB": 3, "IIIa": 2, "IIIb": 3}
                X_cat_int["Overall.Stage"] = st.map(stage_map).fillna(3).astype(int)
                print(f"  Overall.Stage: {sorted(X_cat_int['Overall.Stage'].unique())}")
            
            # T/N/M stages (already numeric, but ensure int)
            if "clinical.T.Stage" in cat_cols:
                t = pd.to_numeric(X_cat_raw["clinical.T.Stage"], errors='coerce')
                X_cat_int["clinical.T.Stage"] = t.fillna(0).astype(int)
                print(f"  clinical.T.Stage: {sorted(X_cat_int['clinical.T.Stage'].unique())}")
            
            if "Clinical.N.Stage" in cat_cols:
                n = pd.to_numeric(X_cat_raw["Clinical.N.Stage"], errors='coerce')
                X_cat_int["Clinical.N.Stage"] = n.fillna(0).astype(int)
                print(f"  Clinical.N.Stage: {sorted(X_cat_int['Clinical.N.Stage'].unique())}")
            
            if "Clinical.M.Stage" in cat_cols:
                m = pd.to_numeric(X_cat_raw["Clinical.M.Stage"], errors='coerce')
                m = m.map(lambda x: 2 if x == 3 else x)  # map 3 -> 2
                X_cat_int["Clinical.M.Stage"] = m.fillna(0).astype(int)
                print(f"  Clinical.M.Stage: {sorted(X_cat_int['Clinical.M.Stage'].unique())}")
            
            # Step 2: Now try encoding with integer values
            print("\n--- Step 2: Attempting to encode with integer values ---")
            try:
                X_cat_arr = X_cat_int[cat_cols].values
                print(f"X_cat_int shape: {X_cat_arr.shape}")
                print(f"X_cat_int dtypes: {X_cat_int[cat_cols].dtypes.tolist()}")
                print(f"X_cat_int first row: {X_cat_arr[0]}")
                
                cat_enc = encoder.transform(X_cat_int[cat_cols])
                print(f"SUCCESS: encoded shape {cat_enc.shape}")
                print(f"Encoded feature names: {encoder.get_feature_names_out(cat_cols).tolist()}")
            except Exception as e:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()