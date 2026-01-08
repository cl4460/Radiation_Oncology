#!/usr/bin/env python3
"""
Final external validation using the retrained GBSA model.

This script:
1. Trains a new GBSA model on LUNG1 training data using the same parameters
2. Applies it to RG external data
3. Reports external C-index

Run in zip_gbsa_infer_py310 environment with sklearn==1.3.2, sksurv==0.22.2
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Categorical mappings
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
    if pd.isna(s):
        return 0
    c = _canon_str(s).replace('t', '').replace('n', '').replace('m', '')
    c = c.replace('a', '').replace('b', '').replace('c', '')
    if c in ['0', '1', '2', '3', '4', '5']:
        return int(c)
    if c == 'x':
        return 5
    return 0

def _canon_overall_stage(s):
    c = _canon_str(s).replace("stage", "").strip()
    c = c.replace('a', '').replace('b', '').replace('c', '')
    return _STAGE_MAP.get(c, 0)

def main():
    # Load retrained model info
    with open("retrain_gbsa91/summary.json") as f:
        summary = json.load(f)[0]
    
    selected_features = summary['selected_features']
    model_params = summary['model_params']
    
    print("=" * 80)
    print("EXTERNAL VALIDATION - NSCLC-RADIOGENOMICS")
    print("=" * 80)
    print(f"Model: GBSA with {len(selected_features)} features")
    print(f"Internal train C-index: {summary['train_cindex']:.4f}")
    print(f"Internal test C-index: {summary['test_cindex']:.4f}")
    
    # Paths
    LUNG1_CLINICAL = "/home/lichengze/Research/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv"
    LUNG1_RADIOMICS = "/home/lichengze/Research/com/Tb_features_1688_from_yaml.csv"
    SPLIT_JSON = "/home/lichengze/Research/com/out_zip_audit_splits/split_ClinicplusRadiomics_6cf7bc2c.json"
    RG_CLINICAL = "/home/lichengze/Research/Radiogenomics/Radiogenomics_clinical_aligned.csv"
    RG_RADIOMICS = "/home/lichengze/Research/com/rg_Tb_features_1688_from_yaml.csv"
    
    # Load LUNG1 data
    print("\n[1/5] Loading LUNG1 training data...")
    lung1_clin = pd.read_csv(LUNG1_CLINICAL)
    lung1_rad = pd.read_csv(LUNG1_RADIOMICS)
    lung1 = pd.merge(lung1_clin, lung1_rad, on='PatientID', how='inner')
    
    with open(SPLIT_JSON) as f:
        split = json.load(f)
        train_ids = split['train_ids']
    
    lung1_train = lung1[lung1['PatientID'].isin(train_ids)].copy()
    print(f"   Training samples: {len(lung1_train)}")
    
    # Load RG data
    print("\n[2/5] Loading RG external data...")
    rg_clin = pd.read_csv(RG_CLINICAL)
    rg_rad = pd.read_csv(RG_RADIOMICS)
    
    if 'patient_id' in rg_rad.columns:
        rg_rad.rename(columns={'patient_id': 'PatientID'}, inplace=True)
    if 'patient_id' in rg_clin.columns and 'PatientID' not in rg_clin.columns:
        rg_clin.rename(columns={'patient_id': 'PatientID'}, inplace=True)
    
    rg = pd.merge(rg_clin, rg_rad, on='PatientID', how='inner')
    print(f"   External samples: {len(rg)}")
    
    # Identify feature types
    radiomics_feats = [f for f in selected_features 
                      if not f.endswith(('_0', '_1', '_2', '_3', '_4', '_5')) and f != 'age']
    numeric_feats = ['age'] + radiomics_feats
    cat_feats = ['gender', 'Histology', 'Overall.Stage', 
                 'Clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage']
    
    print(f"\n[3/5] Building feature matrices...")
    print(f"   Numeric features: {len(numeric_feats)} (age + {len(radiomics_feats)} radiomics)")
    print(f"   Categorical features: {len(cat_feats)}")
    
    # Build LUNG1 training features
    X_train_num = pd.DataFrame(index=lung1_train.index)
    X_train_num['age'] = pd.to_numeric(lung1_train['age'], errors='coerce')
    for f in radiomics_feats:
        X_train_num[f] = lung1_train.get(f, np.nan)
    
    # LUNG1 categorical features - check if already integer or need mapping
    X_train_cat = pd.DataFrame(index=lung1_train.index)
    
    # LUNG1 data should already be integer-coded, but let's be safe
    gender_col = lung1_train['gender']
    if gender_col.dtype == 'object':
        X_train_cat['gender'] = gender_col.apply(_canon_gender)
    else:
        X_train_cat['gender'] = gender_col
    
    histo_col = lung1_train['Histology']
    if histo_col.dtype == 'object':
        X_train_cat['Histology'] = histo_col.apply(_canon_histology)
    else:
        X_train_cat['Histology'] = histo_col
    
    stage_col = lung1_train['Overall.Stage']
    if stage_col.dtype == 'object':
        X_train_cat['Overall.Stage'] = stage_col.apply(_canon_overall_stage)
    else:
        X_train_cat['Overall.Stage'] = stage_col
    
    t_col = lung1_train['clinical.T.Stage']
    if t_col.dtype == 'object':
        X_train_cat['Clinical.T.Stage'] = t_col.apply(_parse_stage_int)
    else:
        X_train_cat['Clinical.T.Stage'] = t_col
    
    n_col = lung1_train['Clinical.N.Stage']
    if n_col.dtype == 'object':
        X_train_cat['Clinical.N.Stage'] = n_col.apply(_parse_stage_int)
    else:
        X_train_cat['Clinical.N.Stage'] = n_col
    
    m_col = lung1_train['Clinical.M.Stage']
    if m_col.dtype == 'object':
        X_train_cat['Clinical.M.Stage'] = m_col.apply(_parse_stage_int)
    else:
        X_train_cat['Clinical.M.Stage'] = m_col
    
    # Build RG features (with string-to-int mapping)
    X_rg_num = pd.DataFrame(index=rg.index)
    X_rg_num['age'] = pd.to_numeric(rg.get('age_at_diagnosis', rg.get('age', np.nan)), errors='coerce')
    for f in radiomics_feats:
        X_rg_num[f] = rg.get(f, np.nan)
    
    X_rg_cat = pd.DataFrame(index=rg.index)
    X_rg_cat['gender'] = rg.get('gender', pd.Series([np.nan]*len(rg))).apply(_canon_gender)
    X_rg_cat['Histology'] = rg.get('histology', pd.Series([np.nan]*len(rg))).apply(_canon_histology)
    X_rg_cat['Overall.Stage'] = rg.get('clinical_stage', pd.Series([np.nan]*len(rg))).apply(_canon_overall_stage)
    X_rg_cat['Clinical.T.Stage'] = rg.get('clinical_t_stage', pd.Series([np.nan]*len(rg))).apply(_parse_stage_int)
    X_rg_cat['Clinical.N.Stage'] = rg.get('clinical_n_stage', pd.Series([np.nan]*len(rg))).apply(_parse_stage_int)
    X_rg_cat['Clinical.M.Stage'] = rg.get('clinical_m_stage', pd.Series([np.nan]*len(rg))).apply(_parse_stage_int)
    
    print(f"   LUNG1 raw features: {X_train_num.shape[1]} numeric + {X_train_cat.shape[1]} categorical")
    print(f"   RG raw features: {X_rg_num.shape[1]} numeric + {X_rg_cat.shape[1]} categorical")
    
    # Preprocessing
    print(f"\n[4/5] Preprocessing...")
    
    # Numeric: impute + scale
    num_imputer = SimpleImputer(strategy='mean')
    X_train_num_imp = pd.DataFrame(
        num_imputer.fit_transform(X_train_num),
        columns=numeric_feats,
        index=X_train_num.index
    )
    
    num_scaler = MinMaxScaler()
    X_train_num_scaled = pd.DataFrame(
        num_scaler.fit_transform(X_train_num_imp),
        columns=numeric_feats,
        index=X_train_num.index
    )
    
    X_rg_num_imp = pd.DataFrame(
        num_imputer.transform(X_rg_num),
        columns=numeric_feats,
        index=X_rg_num.index
    )
    
    X_rg_num_scaled = pd.DataFrame(
        num_scaler.transform(X_rg_num_imp),
        columns=numeric_feats,
        index=X_rg_num.index
    )
    
    # Categorical: fill 0 + OneHotEncode
    X_train_cat_filled = X_train_cat.fillna(0).astype(int)
    X_rg_cat_filled = X_rg_cat.fillna(0).astype(int)
    
    cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat_enc = cat_encoder.fit_transform(X_train_cat_filled)
    X_rg_cat_enc = cat_encoder.transform(X_rg_cat_filled)
    
    cat_names = []
    for i, col in enumerate(cat_feats):
        for j in range(len(cat_encoder.categories_[i])):
            cat_names.append(f"{col}_{j}")
    
    X_train_cat_enc = pd.DataFrame(X_train_cat_enc, columns=cat_names, index=X_train_cat.index)
    X_rg_cat_enc = pd.DataFrame(X_rg_cat_enc, columns=cat_names, index=X_rg_cat.index)
    
    # Combine
    X_train_all = pd.concat([X_train_num_scaled, X_train_cat_enc], axis=1)
    X_rg_all = pd.concat([X_rg_num_scaled, X_rg_cat_enc], axis=1)
    
    # Select model features
    missing_train = [f for f in selected_features if f not in X_train_all.columns]
    missing_rg = [f for f in selected_features if f not in X_rg_all.columns]
    
    if missing_train:
        for f in missing_train:
            X_train_all[f] = 0.0
    
    if missing_rg:
        for f in missing_rg:
            X_rg_all[f] = 0.0
    
    X_train = X_train_all[selected_features]
    X_rg = X_rg_all[selected_features]
    
    print(f"   Final training matrix: {X_train.shape}")
    print(f"   Final RG matrix: {X_rg.shape}")
    
    # Train model
    print(f"\n[5/5] Training and evaluating...")
    y_train = np.array([
        (bool(e), float(t))
        for e, t in zip(lung1_train['deadstatus.event'], lung1_train['Survival.time'])
    ], dtype=[('event', bool), ('time', float)])
    
    model = GradientBoostingSurvivalAnalysis(**model_params)
    model.fit(X_train, y_train)
    
    # Predict on RG
    pred_rg = model.predict(X_rg)
    
    # Get RG outcomes
    events_rg = rg['event'].astype(bool).values
    times_rg = rg['time_days'].values
    
    # Calculate C-index
    cindex_rg = concordance_index_censored(events_rg, times_rg, pred_rg)[0]
    
    # Also predict on LUNG1 train for sanity check
    pred_train = model.predict(X_train)
    cindex_train = concordance_index_censored(
        lung1_train['deadstatus.event'].astype(bool).values,
        lung1_train['Survival.time'].values,
        pred_train
    )[0]
    
    # Bootstrap CI for external validation (200 iterations, matching internal)
    print(f"\n[6/6] Computing bootstrap CI for external validation (n=200)...")
    np.random.seed(model_params['random_state'])
    n_boot = 200
    boot_cindices = []
    
    rg_df = pd.DataFrame({
        'event': events_rg,
        'time': times_rg,
        'pred_risk': pred_rg,
    })
    
    for i in range(n_boot):
        # Resample with replacement
        boot_sample = rg_df.sample(n=len(rg_df), replace=True, random_state=model_params['random_state']+i)
        
        # Calculate C-index on bootstrap sample
        try:
            boot_ci = concordance_index_censored(
                boot_sample['event'].values,
                boot_sample['time'].values,
                boot_sample['pred_risk'].values
            )[0]
            boot_cindices.append(boot_ci)
        except:
            # Skip if bootstrap sample has issues
            continue
    
    boot_cindices = np.array(boot_cindices)
    boot_mean = boot_cindices.mean()
    boot_std = boot_cindices.std()
    
    # 95% CI using normal approximation
    ci_norm_lower = boot_mean - 1.96 * boot_std
    ci_norm_upper = boot_mean + 1.96 * boot_std
    
    # 95% CI using percentile method
    ci_pct_lower = np.percentile(boot_cindices, 2.5)
    ci_pct_upper = np.percentile(boot_cindices, 97.5)
    
    print(f"   Bootstrap mean: {boot_mean:.4f} ± {boot_std:.4f}")
    print(f"   CI(norm) 95%: [{ci_norm_lower:.4f}, {ci_norm_upper:.4f}]")
    print(f"   CI(pct)  95%: [{ci_pct_lower:.4f}, {ci_pct_upper:.4f}]")
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Training C-index (recomputed):   {cindex_train:.4f}")
    print(f"Internal test C-index (original): {summary['test_cindex']:.4f}")
    print(f"External RG C-index:              {cindex_rg:.4f}")
    print(f"External RG bootstrap:            {boot_mean:.4f} ± {boot_std:.4f}")
    print(f"External RG CI(norm) 95%:         [{ci_norm_lower:.4f}, {ci_norm_upper:.4f}]")
    print(f"External RG CI(pct)  95%:         [{ci_pct_lower:.4f}, {ci_pct_upper:.4f}]")
    print("=" * 80)
    
    # Save results
    out_dir = Path("external_validation_results")
    out_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame({
        'PatientID': rg['PatientID'].values,
        'time': times_rg,
        'event': events_rg.astype(int),
        'pred_risk': pred_rg,
    })
    results_df.to_csv(out_dir / "rg_predictions.csv", index=False)
    
    validation_summary = {
        'external_dataset': 'NSCLC-Radiogenomics',
        'n_external_samples': len(X_rg),
        'n_training_samples': len(X_train),
        'n_features': len(selected_features),
        'training_cindex_recomputed': float(cindex_train),
        'internal_test_cindex_original': float(summary['test_cindex']),
        'external_cindex': float(cindex_rg),
        'external_boot_mean': float(boot_mean),
        'external_boot_std': float(boot_std),
        'external_ci_norm_2.5': float(ci_norm_lower),
        'external_ci_norm_97.5': float(ci_norm_upper),
        'external_ci_pct_2.5': float(ci_pct_lower),
        'external_ci_pct_97.5': float(ci_pct_upper),
        'model_params': model_params,
    }
    
    with open(out_dir / "validation_summary.json", 'w') as f:
        json.dump(validation_summary, f, indent=2)
    
    print(f"\n✅ Results saved to {out_dir}/")
    print(f"   - rg_predictions.csv")
    print(f"   - validation_summary.json")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

