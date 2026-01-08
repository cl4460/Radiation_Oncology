#!/usr/bin/env python3
"""
Deep debugging script to find the exact cause of prediction mismatch.

This script will:
1. Load the first test sample
2. Build its feature vector step by step
3. Compare with zip predictions at each stage
4. Identify where the divergence occurs
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
    print("DEEP DEBUG: Find the exact cause of prediction mismatch")
    print("=" * 80)
    
    # Load artifacts
    with zipfile.ZipFile(zip_path, "r") as z:
        with tempfile.TemporaryDirectory() as tmp:
            # Load normalizer
            z.extract('Clinic+Radiomics/seed_91/normalizer.joblib', tmp)
            norm = joblib.load(f"{tmp}/Clinic+Radiomics/seed_91/normalizer.joblib")
            scaler = norm['normalizers']['numeric']
            numeric_cols = norm['numeric_cols']
            
            # Load fill values
            z.extract('Clinic+Radiomics/seed_91/train_fill_values.joblib', tmp)
            fill_obj = joblib.load(f"{tmp}/Clinic+Radiomics/seed_91/train_fill_values.joblib")
            fill_values = fill_obj['fill_values']
            
            # Load model
            z.extract('Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/all_train_data_GradientBoostingSurvivalAnalysis.joblib', tmp)
            gbsa = joblib.load(f"{tmp}/Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/all_train_data_GradientBoostingSurvivalAnalysis.joblib")
            
            # Load test_preds
            z.extract('Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/test_preds_GradientBoostingSurvivalAnalysis.xlsx', tmp)
            test_preds_df = pd.read_excel(f"{tmp}/Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/test_preds_GradientBoostingSurvivalAnalysis.xlsx")
    
    model_features = list(gbsa.feature_names_in_)
    print(f"\nModel uses {len(model_features)} features")
    
    # Load split
    with open(split_json) as f:
        split = json.load(f)
    test_ids = [str(x) for x in split['test_ids']]
    
    # Load data
    clin = pd.read_csv(lung1_clinical)
    clin['PatientID'] = clin['PatientID'].astype(str)
    clin = clin.set_index('PatientID')
    
    rad = pd.read_csv(lung1_radiomics)
    rad['PatientID'] = rad['PatientID'].astype(str)
    rad = rad.set_index('PatientID')
    
    # Focus on first 3 test samples
    sample_ids = test_ids[:3]
    
    print("\n" + "=" * 80)
    print("STEP 1: Check raw feature values")
    print("=" * 80)
    
    for pid in sample_ids:
        print(f"\n--- Patient {pid} ---")
        
        # Check the 16 model features
        print("Raw radiomics features:")
        for feat in model_features[:3]:  # Show first 3
            val = rad.loc[pid, feat]
            print(f"  {feat}: {val}")
        
        # Check age
        age = clin.loc[pid, 'age']
        print(f"  age (from clinical): {age}")
    
    print("\n" + "=" * 80)
    print("STEP 2: Build feature matrix and check scaling")
    print("=" * 80)
    
    # Build feature matrix for all test samples
    test_ids_available = [pid for pid in test_ids if pid in rad.index and pid in clin.index]
    
    # Method 1: What we're currently doing
    numeric_cols_in_rad = [c for c in numeric_cols if c in rad.columns]
    X_full = pd.DataFrame(index=test_ids_available, columns=numeric_cols)
    
    # Fill radiomics features
    X_full[numeric_cols_in_rad] = rad.loc[test_ids_available, numeric_cols_in_rad]
    
    # Fill age
    if 'age' in numeric_cols:
        X_full['age'] = clin.loc[test_ids_available, 'age']
    
    # Fill missing with fill_values
    for c in numeric_cols:
        if c in fill_values:
            X_full[c] = X_full[c].fillna(fill_values[c])
    
    print(f"\nBefore scaling:")
    for pid in sample_ids:
        print(f"  {pid}: age={X_full.loc[pid, 'age']:.2f}, {model_features[0]}={X_full.loc[pid, model_features[0]]:.6f}")
    
    # Check scaler properties
    print(f"\n--- Scaler properties ---")
    print(f"Scaler type: {type(scaler)}")
    print(f"n_features_in: {scaler.n_features_in_}")
    
    if hasattr(scaler, 'data_min_'):
        age_idx = list(numeric_cols).index('age')
        print(f"\nAge statistics in scaler:")
        print(f"  data_min_[age]: {scaler.data_min_[age_idx]}")
        print(f"  data_max_[age]: {scaler.data_max_[age_idx]}")
        print(f"  data_range_[age]: {scaler.data_range_[age_idx]}")
        
        # Check one model feature
        feat_idx = list(numeric_cols).index(model_features[0])
        print(f"\n{model_features[0]} statistics in scaler:")
        print(f"  data_min_: {scaler.data_min_[feat_idx]}")
        print(f"  data_max_: {scaler.data_max_[feat_idx]}")
        print(f"  data_range_: {scaler.data_range_[feat_idx]}")
    
    # Scale
    X_scaled = scaler.transform(X_full)
    X_scaled_df = pd.DataFrame(X_scaled, index=X_full.index, columns=numeric_cols)
    
    print(f"\nAfter scaling:")
    for pid in sample_ids:
        age_scaled = X_scaled_df.loc[pid, 'age']
        feat_scaled = X_scaled_df.loc[pid, model_features[0]]
        print(f"  {pid}: age={age_scaled:.6f}, {model_features[0]}={feat_scaled:.6f}")
    
    # Extract model features
    X_model = X_scaled_df[model_features].values
    
    print("\n" + "=" * 80)
    print("STEP 3: Predict and compare")
    print("=" * 80)
    
    # Predict
    risk_pred = gbsa.predict(X_model)
    
    # Get zip predictions
    test_preds_df['pid'] = test_preds_df['pid'].astype(str)
    test_preds_df = test_preds_df.set_index('pid')
    
    print(f"\nPrediction comparison:")
    print(f"{'PatientID':<15} {'Our Risk':<15} {'ZIP Risk':<15} {'Diff':<15}")
    for i, pid in enumerate(sample_ids):
        our_risk = risk_pred[test_ids_available.index(pid)]
        zip_risk = test_preds_df.loc[pid, 'pred_risk']
        diff = abs(our_risk - zip_risk)
        print(f"{pid:<15} {our_risk:<15.6f} {zip_risk:<15.6f} {diff:<15.6f}")
    
    print("\n" + "=" * 80)
    print("STEP 4: Check if scaler.transform is deterministic")
    print("=" * 80)
    
    # Transform the same data twice
    X_scaled_1 = scaler.transform(X_full)
    X_scaled_2 = scaler.transform(X_full)
    
    if np.allclose(X_scaled_1, X_scaled_2):
        print("✅ Scaler.transform is deterministic")
    else:
        print("❌ Scaler.transform is NOT deterministic!")
        print(f"Max difference: {np.abs(X_scaled_1 - X_scaled_2).max()}")
    
    print("\n" + "=" * 80)
    print("STEP 5: Try a different approach - use sklearn's scaler directly")
    print("=" * 80)
    
    # Maybe the issue is with how we're using the scaler?
    # Let's try using it exactly as sklearn intends
    
    # Convert to numpy array first
    X_full_array = X_full.values
    X_scaled_array = scaler.transform(X_full_array)
    
    # Extract model features by index
    model_feat_indices = [list(numeric_cols).index(f) for f in model_features]
    X_model_array = X_scaled_array[:, model_feat_indices]
    
    # Predict
    risk_pred_2 = gbsa.predict(X_model_array)
    
    print(f"\nPrediction comparison (using array approach):")
    print(f"{'PatientID':<15} {'Method1':<15} {'Method2':<15} {'Diff':<15}")
    for i, pid in enumerate(sample_ids):
        idx = test_ids_available.index(pid)
        risk1 = risk_pred[idx]
        risk2 = risk_pred_2[idx]
        diff = abs(risk1 - risk2)
        print(f"{pid:<15} {risk1:<15.6f} {risk2:<15.6f} {diff:<15.6f}")
    
    if np.allclose(risk_pred, risk_pred_2):
        print("✅ Both methods give same predictions")
    else:
        print("❌ Different methods give different predictions!")
    
    print("\n" + "=" * 80)
    print("STEP 6: Check model properties")
    print("=" * 80)
    
    print(f"Model type: {type(gbsa)}")
    print(f"Model n_features: {gbsa.n_features_in_}")
    
    # Check if model has any randomness
    risk_pred_a = gbsa.predict(X_model)
    risk_pred_b = gbsa.predict(X_model)
    
    if np.allclose(risk_pred_a, risk_pred_b):
        print("✅ Model predictions are deterministic")
    else:
        print("❌ Model predictions are NOT deterministic!")
    
    print("\n" + "=" * 80)
    print("STEP 7: Hypothesis - maybe we need to look at train_preds too?")
    print("=" * 80)
    
    # Load train_preds to see if there's a pattern
    with zipfile.ZipFile(zip_path, "r") as z:
        with tempfile.TemporaryDirectory() as tmp:
            z.extract('Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/train_preds_GradientBoostingSurvivalAnalysis.xlsx', tmp)
            train_preds_df = pd.read_excel(f"{tmp}/Clinic+Radiomics/seed_91/GradientBoostingSurvivalAnalysis/train_preds_GradientBoostingSurvivalAnalysis.xlsx")
    
    print(f"\nTrain preds stats:")
    print(f"  mean: {train_preds_df['pred_risk'].mean():.6f}")
    print(f"  std: {train_preds_df['pred_risk'].std():.6f}")
    print(f"  min: {train_preds_df['pred_risk'].min():.6f}")
    print(f"  max: {train_preds_df['pred_risk'].max():.6f}")
    
    print(f"\nTest preds stats (from zip):")
    print(f"  mean: {test_preds_df['pred_risk'].mean():.6f}")
    print(f"  std: {test_preds_df['pred_risk'].std():.6f}")
    print(f"  min: {test_preds_df['pred_risk'].min():.6f}")
    print(f"  max: {test_preds_df['pred_risk'].max():.6f}")
    
    print(f"\nOur test preds stats:")
    print(f"  mean: {risk_pred.mean():.6f}")
    print(f"  std: {risk_pred.std():.6f}")
    print(f"  min: {risk_pred.min():.6f}")
    print(f"  max: {risk_pred.max():.6f}")
    
    # Check ratio
    ratio = test_preds_df['pred_risk'].std() / risk_pred.std()
    print(f"\nStd ratio (zip / ours): {ratio:.2f}x")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Calculate correlation
    aligned_our = []
    aligned_zip = []
    for pid in test_ids_available:
        if pid in test_preds_df.index:
            idx = test_ids_available.index(pid)
            aligned_our.append(risk_pred[idx])
            aligned_zip.append(test_preds_df.loc[pid, 'pred_risk'])
    
    corr = np.corrcoef(aligned_our, aligned_zip)[0, 1]
    print(f"Correlation: {corr:.6f}")
    
    # Try different scaling of our predictions
    print(f"\nTrying to match by scaling:")
    
    # Method 1: Linear scaling
    scale_factor = test_preds_df['pred_risk'].std() / risk_pred.std()
    mean_shift = test_preds_df['pred_risk'].mean() - risk_pred.mean() * scale_factor
    risk_scaled = risk_pred * scale_factor + mean_shift
    
    print(f"  Scale factor: {scale_factor:.4f}")
    print(f"  Mean shift: {mean_shift:.4f}")
    print(f"  After scaling - mean: {risk_scaled.mean():.6f}, std: {risk_scaled.std():.6f}")
    
    # Check if scaled predictions match
    aligned_scaled = []
    for pid in test_ids_available:
        if pid in test_preds_df.index:
            idx = test_ids_available.index(pid)
            aligned_scaled.append(risk_scaled[idx])
    
    mse = np.mean((np.array(aligned_scaled) - np.array(aligned_zip))**2)
    print(f"  MSE after scaling: {mse:.6f}")
    
    # If MSE is small, scaling works!
    if mse < 0.01:
        print("\n✅ FOUND IT! Our predictions are correct, just need linear scaling!")
        print(f"   risk_correct = risk_pred * {scale_factor:.6f} + {mean_shift:.6f}")
    else:
        print(f"\n❌ Scaling doesn't fix it. Problem is deeper.")


if __name__ == "__main__":
    main()



