#!/usr/bin/env python3
"""
Simplified external validation script.
Key insight: The GBSA model only uses 16 RADIOMICS features (secondary_selected_features),
NO clinical features at all!

This script:
1. Loads the saved GBSA model from zip
2. Loads the MinMaxScaler from zip
3. Extracts ONLY the 16 radiomics features used by the model
4. Applies the scaler to normalize them
5. Predicts and computes C-index

Run in zip_gbsa_infer_py310 environment.
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
    rg_clinical = "/home/lichengze/Research/Radiogenomics/Radiogenomics_clinical_aligned.csv"
    rg_radiomics = "/home/lichengze/Research/com/rg_Tb_features_1688_from_yaml.csv"
    out_dir = "/home/lichengze/Research/com/external_validation_simple"
    
    os.makedirs(out_dir, exist_ok=True)
    
    task = "Clinic+Radiomics"
    seed = 91
    model_name = "GradientBoostingSurvivalAnalysis"
    
    base = f"{task}/seed_{seed}"
    model_base = f"{base}/{model_name}"
    
    print("=" * 70)
    print("STEP 1: Load artifacts from zip")
    print("=" * 70)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        with tempfile.TemporaryDirectory() as tmp:
            # Load normalizer
            norm_path = f"{base}/normalizer.joblib"
            z.extract(norm_path, tmp)
            normalizer = joblib.load(os.path.join(tmp, norm_path))
            scaler = normalizer['normalizers']['numeric']
            numeric_cols = normalizer.get('numeric_cols', [])
            
            # Load fill values
            fill_path = f"{base}/train_fill_values.joblib"
            z.extract(fill_path, tmp)
            fill_obj = joblib.load(os.path.join(tmp, fill_path))
            fill_values = fill_obj.get('fill_values', fill_obj)
            
            # Load model
            model_path = f"{model_base}/all_train_data_{model_name}.joblib"
            z.extract(model_path, tmp)
            gbsa = joblib.load(os.path.join(tmp, model_path))
            
            # Load test_preds for ground truth
            test_preds_path = f"{model_base}/test_preds_{model_name}.xlsx"
            z.extract(test_preds_path, tmp)
            test_preds_df = pd.read_excel(os.path.join(tmp, test_preds_path))
    
    # Get the 16 features the model actually uses
    model_features = list(gbsa.feature_names_in_)
    print(f"\nModel uses {len(model_features)} features:")
    for i, f in enumerate(model_features):
        print(f"  [{i}] {f}")
    
    # Verify all are in numeric_cols (i.e., radiomics features)
    print(f"\nAll features in numeric_cols? {all(f in numeric_cols for f in model_features)}")
    
    print("\n" + "=" * 70)
    print("STEP 2: Load and prepare Lung1 test data")
    print("=" * 70)
    
    # Load split
    with open(split_json) as f:
        split = json.load(f)
    test_ids = [str(x) for x in (split.get('test_ids') or split.get('test') or [])]
    print(f"Test set size from split: {len(test_ids)}")
    
    # Load clinical (for survival outcomes only)
    clin = pd.read_csv(lung1_clinical)
    clin['PatientID'] = clin['PatientID'].astype(str)
    clin = clin.set_index('PatientID')
    
    # Load radiomics
    rad = pd.read_csv(lung1_radiomics)
    rad['PatientID'] = rad['PatientID'].astype(str)
    rad = rad.set_index('PatientID')
    
    # Get test set
    test_ids_available = [pid for pid in test_ids if pid in rad.index and pid in clin.index]
    print(f"Test samples available: {len(test_ids_available)}")
    
    # Extract only the 16 features needed
    X_test_raw = rad.loc[test_ids_available, model_features].copy()
    
    # Apply fill values for any missing
    for f in model_features:
        if f in fill_values:
            X_test_raw[f] = X_test_raw[f].fillna(fill_values[f])
    X_test_raw = X_test_raw.fillna(0.0)
    
    print(f"\nX_test_raw shape: {X_test_raw.shape}")
    print(f"X_test_raw sample (first row):\n{X_test_raw.iloc[0]}")
    
    # Apply scaler - BUT we need to scale using the FULL numeric_cols, then extract
    # Because the scaler was fitted on all 1689 numeric features
    print("\n--- Normalizing features ---")
    
    # Method A: Scale all 1689 features, then extract the 16
    # But only use features that exist in radiomics CSV (exclude age)
    numeric_cols_available = [c for c in numeric_cols if c in rad.columns]
    X_all_raw = rad.loc[test_ids_available, numeric_cols_available].copy()
    for c in numeric_cols_available:
        if c in fill_values:
            X_all_raw[c] = X_all_raw[c].fillna(fill_values[c])
    X_all_raw = X_all_raw.fillna(0.0)
    
    # Transform - but scaler expects ALL numeric_cols, so we need to pad for missing columns
    # Create a full matrix with all numeric_cols
    X_all_full = pd.DataFrame(index=X_all_raw.index, columns=numeric_cols)
    # Fill with actual values from radiomics
    X_all_full[numeric_cols_available] = X_all_raw
    # Fill age from clinical data (NOT fill_value!)
    if 'age' in numeric_cols and 'age' not in numeric_cols_available:
        # Get age from clinical CSV
        age_values = clin.loc[test_ids_available, 'age']
        X_all_full['age'] = age_values
    # Fill other missing columns with their fill_values
    for c in numeric_cols:
        if c not in numeric_cols_available and c != 'age':
            fill_val = fill_values.get(c, 0.0)
            X_all_full[c] = fill_val
    # Apply fill_values for any remaining NaNs
    for c in numeric_cols:
        if c in fill_values:
            X_all_full[c] = X_all_full[c].fillna(fill_values[c])
    
    # Transform
    X_all_scaled = scaler.transform(X_all_full)
    X_all_scaled_df = pd.DataFrame(X_all_scaled, index=X_all_raw.index, columns=numeric_cols)
    
    # Extract the 16 features
    X_test = X_all_scaled_df[model_features].values
    
    print(f"X_test scaled shape: {X_test.shape}")
    print(f"X_test scaled sample (first row): {X_test[0]}")
    
    # Get survival outcomes
    time_test = clin.loc[test_ids_available, 'Survival.time'].values.astype(float)
    event_test = clin.loc[test_ids_available, 'deadstatus.event'].values.astype(bool)
    
    print("\n" + "=" * 70)
    print("STEP 3: Predict and evaluate on Lung1 test")
    print("=" * 70)
    
    risk_pred = gbsa.predict(X_test)
    print(f"Predicted risk stats: mean={risk_pred.mean():.6f}, std={risk_pred.std():.6f}, min={risk_pred.min():.6f}, max={risk_pred.max():.6f}")
    
    ci_internal = concordance_index_censored(event_test, time_test, risk_pred)[0]
    print(f"\n*** INTERNAL C-index (our prediction): {ci_internal:.6f} ***")
    
    # Compare with zip test_preds
    pid_col = 'pid' if 'pid' in test_preds_df.columns else 'PatientID'
    test_preds_df[pid_col] = test_preds_df[pid_col].astype(str)
    test_preds_df = test_preds_df.set_index(pid_col)
    
    # Find risk column
    risk_col = [c for c in test_preds_df.columns if 'risk' in c.lower() or 'pred' in c.lower()][0]
    
    # Align with our test set
    common_ids = [pid for pid in test_ids_available if pid in test_preds_df.index]
    risk_zip = test_preds_df.loc[common_ids, risk_col].values
    time_zip = clin.loc[common_ids, 'Survival.time'].values.astype(float)
    event_zip = clin.loc[common_ids, 'deadstatus.event'].values.astype(bool)
    
    ci_zip = concordance_index_censored(event_zip, time_zip, risk_zip)[0]
    print(f"*** ZIP test_preds C-index: {ci_zip:.6f} ***")
    
    # Compare predictions
    our_risk_aligned = risk_pred[[test_ids_available.index(pid) for pid in common_ids]]
    print(f"\nPrediction comparison (first 10 samples):")
    print(f"{'PatientID':<15} {'Our Risk':<15} {'ZIP Risk':<15} {'Diff':<15}")
    for i, pid in enumerate(common_ids[:10]):
        our_r = our_risk_aligned[i]
        zip_r = risk_zip[i]
        print(f"{pid:<15} {our_r:<15.6f} {zip_r:<15.6f} {abs(our_r - zip_r):<15.6f}")
    
    # Check if predictions match
    corr = np.corrcoef(our_risk_aligned, risk_zip)[0, 1]
    print(f"\nCorrelation between our predictions and ZIP predictions: {corr:.6f}")
    
    if abs(ci_internal - ci_zip) < 0.001:
        print("\n✅ SANITY CHECK PASSED: Internal C-index matches ZIP!")
    else:
        print(f"\n❌ SANITY CHECK FAILED: Internal C-index differs by {abs(ci_internal - ci_zip):.6f}")
        print("Possible causes:")
        print("  1. Feature values are different")
        print("  2. Scaler was not applied correctly")
        print("  3. Feature order is wrong")
        
        # Debug: Compare raw feature values
        print("\n--- Debugging: Check feature values ---")
        # We need to see if our features match what the model expects
        
    print("\n" + "=" * 70)
    print("STEP 4: External validation on Radiogenomics")
    print("=" * 70)
    
    if abs(ci_internal - ci_zip) >= 0.01:
        print("⚠️  Skipping external validation because internal sanity check failed.")
        print("Please fix the internal alignment first.")
        return
    
    # Load Radiogenomics data
    rg_clin = pd.read_csv(rg_clinical)
    rg_clin['PatientID'] = rg_clin['patient_id'].astype(str).str.replace('AMC-', 'R01-')
    rg_clin = rg_clin.set_index('PatientID')
    
    rg_rad = pd.read_csv(rg_radiomics)
    rg_rad['PatientID'] = rg_rad['PatientID'].astype(str)
    rg_rad = rg_rad.set_index('PatientID')
    
    # Get common patients
    rg_ids = list(set(rg_clin.index) & set(rg_rad.index))
    
    # Filter for valid survival time
    rg_ids = [pid for pid in rg_ids if rg_clin.loc[pid, 'time_days'] > 0]
    
    # Optionally filter out M1 (metastatic)
    # rg_ids = [pid for pid in rg_ids if rg_clin.loc[pid, 'M'] != 'M1']
    
    print(f"Radiogenomics samples: {len(rg_ids)}")
    
    # Check if all model features exist in RG radiomics
    missing_feats = [f for f in model_features if f not in rg_rad.columns]
    if missing_feats:
        print(f"⚠️  Missing features in RG radiomics: {missing_feats}")
        return
    
    # Prepare features - scale all 1689, then extract 16
    numeric_cols_rg = [c for c in numeric_cols if c in rg_rad.columns]
    X_rg_all_raw = rg_rad.loc[rg_ids, numeric_cols_rg].copy()
    for c in numeric_cols_rg:
        if c in fill_values:
            X_rg_all_raw[c] = X_rg_all_raw[c].fillna(fill_values[c])
    X_rg_all_raw = X_rg_all_raw.fillna(0.0)
    
    # Pad for missing columns
    X_rg_all_full = pd.DataFrame(index=X_rg_all_raw.index, columns=numeric_cols)
    X_rg_all_full[numeric_cols_rg] = X_rg_all_raw
    # Fill age from clinical data (NOT fill_value!)
    if 'age' in numeric_cols and 'age' not in numeric_cols_rg:
        # Get age from RG clinical CSV
        age_values = rg_clin.loc[rg_ids, 'age']
        X_rg_all_full['age'] = age_values
    # Fill other missing columns with their fill_values
    for c in numeric_cols:
        if c not in numeric_cols_rg and c != 'age':
            fill_val = fill_values.get(c, 0.0)
            X_rg_all_full[c] = fill_val
    # Apply fill_values for any remaining NaNs
    for c in numeric_cols:
        if c in fill_values:
            X_rg_all_full[c] = X_rg_all_full[c].fillna(fill_values[c])
    
    X_rg_all_scaled = scaler.transform(X_rg_all_full)
    X_rg_all_scaled_df = pd.DataFrame(X_rg_all_scaled, index=X_rg_all_raw.index, columns=numeric_cols)
    X_rg = X_rg_all_scaled_df[model_features].values
    
    # Survival outcomes
    time_rg = rg_clin.loc[rg_ids, 'time_days'].values.astype(float)
    event_rg = rg_clin.loc[rg_ids, 'event'].values.astype(bool)
    
    # Predict
    risk_rg = gbsa.predict(X_rg)
    
    # Evaluate
    ci_external = concordance_index_censored(event_rg, time_rg, risk_rg)[0]
    
    print(f"\n*** EXTERNAL C-index (Radiogenomics): {ci_external:.6f} ***")
    
    # Bootstrap CI
    n_boot = 200
    rng = np.random.default_rng(42)
    boot_vals = []
    n = len(rg_ids)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ci = concordance_index_censored(event_rg[idx], time_rg[idx], risk_rg[idx])[0]
        boot_vals.append(ci)
    boot_vals = np.array(boot_vals)
    
    print(f"Bootstrap: mean={boot_vals.mean():.6f}, std={boot_vals.std():.6f}")
    print(f"95% CI: [{np.percentile(boot_vals, 2.5):.6f}, {np.percentile(boot_vals, 97.5):.6f}]")
    
    # Save results
    summary = {
        "internal": {
            "n_test": len(test_ids_available),
            "cindex_our": float(ci_internal),
            "cindex_zip": float(ci_zip),
            "match": abs(ci_internal - ci_zip) < 0.001
        },
        "external": {
            "n_test": len(rg_ids),
            "n_events": int(event_rg.sum()),
            "event_rate": float(event_rg.mean()),
            "cindex": float(ci_external),
            "boot_mean": float(boot_vals.mean()),
            "boot_std": float(boot_vals.std()),
            "ci95": [float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5))]
        }
    }
    
    with open(os.path.join(out_dir, "validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save predictions
    pd.DataFrame({
        "PatientID": rg_ids,
        "time": time_rg,
        "event": event_rg.astype(int),
        "risk": risk_rg
    }).to_csv(os.path.join(out_dir, "radiogenomics_preds.csv"), index=False)
    
    print(f"\n[OK] Saved: {out_dir}/validation_summary.json")
    print(f"[OK] Saved: {out_dir}/radiogenomics_preds.csv")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Internal (Lung1 test):       C-index = {ci_internal:.4f}")
    print(f"Internal (ZIP ground truth): C-index = {ci_zip:.4f}")
    print(f"External (Radiogenomics):    C-index = {ci_external:.4f} (95% CI: [{np.percentile(boot_vals, 2.5):.4f}, {np.percentile(boot_vals, 97.5):.4f}])")


if __name__ == "__main__":
    main()