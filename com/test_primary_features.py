#!/usr/bin/env python3
"""
Test if zip test_preds were generated using primary_selected_features (34 features)
instead of secondary_selected_features (16 features).
"""

import zipfile
import tempfile
import os
import json
import joblib
import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_censored


def main():
    # Paths
    zip_path = "/home/lichengze/Research/com/Model.zip"
    lung1_clinical = "/home/lichengze/Research/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
    lung1_radiomics = "/home/lichengze/Research/com/Tb_features_1688_from_yaml.csv"
    split_json = "/home/lichengze/Research/com/out_zip_audit_splits/split_ClinicplusRadiomics_6cf7bc2c.json"
    
    print("=" * 80)
    print("TEST: Were test_preds generated with PRIMARY features (34)?")
    print("=" * 80)
    
    # Load artifacts
    with zipfile.ZipFile(zip_path, "r") as z:
        with tempfile.TemporaryDirectory() as tmp:
            # Load selected features
            z.extract('Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/selected_features_GradientBoostingSurvivalAnalysis.xlsx', tmp)
            sel_df = pd.read_excel(f"{tmp}/Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/selected_features_GradientBoostingSurvivalAnalysis.xlsx")
            
            primary_features = sel_df['primary_selected_features'].dropna().tolist()
            secondary_features = sel_df['secondary_selected_features'].dropna().tolist()
            
            print(f"\nPrimary features: {len(primary_features)}")
            print(f"Secondary features: {len(secondary_features)}")
            
            # Load model to see what it was trained with
            z.extract('Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/all_train_data_GradientBoostingSurvivalAnalysis.joblib', tmp)
            gbsa = joblib.load(f"{tmp}/Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/all_train_data_GradientBoostingSurvivalAnalysis.joblib")
            
            model_features = list(gbsa.feature_names_in_)
            
            print(f"\nModel feature_names_in_: {len(model_features)}")
            
            # Check which list matches
            if model_features == secondary_features:
                print("\n✅ Model matches SECONDARY features (16 radiomics)")
            elif model_features == primary_features:
                print("\n✅ Model matches PRIMARY features (34 with clinical)")
            else:
                print("\n❓ Model matches neither! Checking partial matches...")
                
                # Check if model features are subset/superset
                model_set = set(model_features)
                primary_set = set(primary_features)
                secondary_set = set(secondary_features)
                
                print(f"   Model ∩ Primary: {len(model_set & primary_set)}")
                print(f"   Model ∩ Secondary: {len(model_set & secondary_set)}")
                
                print("\n   Model features:")
                for f in model_features[:10]:
                    in_primary = "YES" if f in primary_set else "NO"
                    in_secondary = "YES" if f in secondary_set else "NO"
                    print(f"     {f[:60]:<60} | Primary: {in_primary} | Secondary: {in_secondary}")
    
    print("\n" + "=" * 80)
    print("HYPOTHESIS: Maybe the issue is with CATEGORICAL encoding?")
    print("=" * 80)
    
    # Check if primary features include clinical OneHot features
    clinical_features = [f for f in primary_features if any(x in f for x in ['gender_', 'Histology_', 'Stage_', 'Clinical.'])]
    print(f"\nClinical OneHot features in primary_selected_features: {len(clinical_features)}")
    for f in clinical_features:
        print(f"  - {f}")
    
    # Check model features
    clinical_in_model = [f for f in model_features if any(x in f for x in ['gender_', 'Histology_', 'Stage_', 'Clinical.'])]
    print(f"\nClinical OneHot features in model: {len(clinical_in_model)}")
    for f in clinical_in_model:
        print(f"  - {f}")
    
    if len(clinical_in_model) == 0:
        print("\n❌ Model does NOT use any clinical features!")
        print("   → test_preds must have been generated with only radiomics features")
        print("   → BUT our predictions don't match, so something else is wrong...")
    
    print("\n" + "=" * 80)
    print("NEXT IDEA: Check if there's a different model joblib file")
    print("=" * 80)
    
    # List all joblib files in model directory
    with zipfile.ZipFile(zip_path, "r") as z:
        model_dir = "Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/"
        joblibs = [n for n in z.namelist() if n.startswith(model_dir) and n.endswith('.joblib')]
        
        print(f"\nJoblib files in {model_dir}:")
        for jb in joblibs:
            info = z.getinfo(jb)
            print(f"  {jb.replace(model_dir, '')} ({info.file_size} bytes)")
    
    print("\n" + "=" * 80)
    print("CRITICAL QUESTION: Are we using the RIGHT model joblib?")
    print("=" * 80)
    
    # Maybe test_preds was generated with a different model?
    # Let's check train_preds too
    with zipfile.ZipFile(zip_path, "r") as z:
        with tempfile.TemporaryDirectory() as tmp:
            z.extract('Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/train_preds_GradientBoostingSurvivalAnalysis.xlsx', tmp)
            train_preds = pd.read_excel(f"{tmp}/Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/train_preds_GradientBoostingSurvivalAnalysis.xlsx")
            
            z.extract('Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/test_preds_GradientBoostingSurvivalAnalysis.xlsx', tmp)
            test_preds = pd.read_excel(f"{tmp}/Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/test_preds_GradientBoostingSurvivalAnalysis.xlsx")
    
    print(f"\nTrain preds: n={len(train_preds)}")
    print(f"  Columns: {train_preds.columns.tolist()}")
    print(f"  Risk stats: mean={train_preds['pred_risk'].mean():.4f}, std={train_preds['pred_risk'].std():.4f}")
    
    print(f"\nTest preds: n={len(test_preds)}")
    print(f"  Columns: {test_preds.columns.tolist()}")
    print(f"  Risk stats: mean={test_preds['pred_risk'].mean():.4f}, std={test_preds['pred_risk'].std():.4f}")
    
    # Check time/event distributions
    print(f"\nTrain event rate: {train_preds['deadstatus.event'].mean():.2%}")
    print(f"Test event rate: {test_preds['deadstatus.event'].mean():.2%}")
    
    # Compute C-index for train_preds using loaded model
    print("\n" + "=" * 80)
    print("ULTIMATE TEST: Can we reproduce TRAIN predictions?")
    print("=" * 80)
    
    # Load data
    with open(split_json) as f:
        split = json.load(f)
    train_ids = [str(x) for x in split['train_ids']]
    
    clin = pd.read_csv(lung1_clinical)
    clin['PatientID'] = clin['PatientID'].astype(str)
    clin = clin.set_index('PatientID')
    
    rad = pd.read_csv(lung1_radiomics)
    rad['PatientID'] = rad['PatientID'].astype(str)
    rad = rad.set_index('PatientID')
    
    # Build feature matrix for train
    train_ids_available = [pid for pid in train_ids if pid in rad.index and pid in clin.index]
    print(f"\nTrain samples: {len(train_ids_available)}")
    
    # Load preprocessing artifacts
    with zipfile.ZipFile(zip_path, "r") as z:
        with tempfile.TemporaryDirectory() as tmp:
            z.extract('Clinic+Radiomics/seed_91/normalizer.joblib', tmp)
            norm = joblib.load(f"{tmp}/Clinic+Radiomics/seed_91/normalizer.joblib")
            scaler = norm['normalizers']['numeric']
            numeric_cols = norm['numeric_cols']
            
            z.extract('Clinic+Radiomics/seed_91/train_fill_values.joblib', tmp)
            fill_obj = joblib.load(f"{tmp}/Clinic+Radiomics/seed_91/train_fill_values.joblib")
            fill_values = fill_obj['fill_values']
    
    # Build feature matrix (same as test)
    numeric_cols_in_rad = [c for c in numeric_cols if c in rad.columns]
    X_train_full = pd.DataFrame(index=train_ids_available, columns=numeric_cols)
    X_train_full[numeric_cols_in_rad] = rad.loc[train_ids_available, numeric_cols_in_rad]
    
    if 'age' in numeric_cols:
        X_train_full['age'] = clin.loc[train_ids_available, 'age']
    
    for c in numeric_cols:
        if c in fill_values:
            X_train_full[c] = X_train_full[c].fillna(fill_values[c])
    
    X_train_scaled = scaler.transform(X_train_full)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train_full.index, columns=numeric_cols)
    X_train_model = X_train_scaled_df[model_features].values
    
    # Predict
    risk_train_pred = gbsa.predict(X_train_model)
    
    print(f"\nOur train predictions:")
    print(f"  mean={risk_train_pred.mean():.4f}, std={risk_train_pred.std():.4f}")
    print(f"  min={risk_train_pred.min():.4f}, max={risk_train_pred.max():.4f}")
    
    # Compare with zip train_preds
    train_preds_indexed = train_preds.copy()
    train_preds_indexed['pid'] = train_preds_indexed['pid'].astype(str)
    train_preds_indexed = train_preds_indexed.set_index('pid')
    
    # Align
    aligned_our_train = []
    aligned_zip_train = []
    for pid in train_ids_available:
        if pid in train_preds_indexed.index:
            idx = train_ids_available.index(pid)
            aligned_our_train.append(risk_train_pred[idx])
            aligned_zip_train.append(train_preds_indexed.loc[pid, 'pred_risk'])
    
    corr_train = np.corrcoef(aligned_our_train, aligned_zip_train)[0, 1]
    print(f"\nCorrelation with zip train_preds: {corr_train:.6f}")
    
    if corr_train > 0.95:
        print("✅ We can reproduce TRAIN predictions!")
    else:
        print(f"❌ We CANNOT reproduce train predictions either!")
        print(f"   → This confirms the model/preprocessing mismatch")


if __name__ == "__main__":
    main()



