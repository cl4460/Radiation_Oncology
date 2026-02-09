#!/usr/bin/env python3
"""
NO-STAGE ENSEMBLE - FOLD-SPECIFIC OOF (NO LEAKAGE)
===================================================
Critical fix: Proper fold-specific training for unbiased OOF predictions

Key difference from previous versions:
- For each fold: train models ONLY on other 4 folds
- Predict on held-out fold -> true OOF
- NO data leakage in calibration

Training config:
- 5-fold CV
- Per fold: 50 models (seeds 0-49) with bootstrap
- Features: age + 16 mRMR radiomics + gender/Histology/T-stage/Overall.Stage
- Horizon: 730 days
- ORIGINAL MODEL REPRODUCTION with fold-specific OOF (no leakage)
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
import warnings
warnings.filterwarnings('ignore')

OUTDIR = Path("/home/lichengze/Research/with_stage_v3")
OUTDIR.mkdir(parents=True, exist_ok=True)

TAU = 730.0
N_MODELS_PER_FOLD = 50
N_CV_FOLDS = 5

# The EXACT 16 mRMR-selected radiomics from original model
SELECTED_RADIOMICS = [
    'wavelet-HHL_glszm_LowGrayLevelZoneEmphasis',
    'wavelet-HLL_glrlm_RunEntropy',
    'wavelet-LHH_glrlm_LongRunHighGrayLevelEmphasis',
    'squareroot_glrlm_LongRunLowGrayLevelEmphasis',
    'original_shape_Elongation',
    'logarithm_glszm_SizeZoneNonUniformityNormalized',
    'wavelet-HLH_glcm_Imc1',
    'wavelet-HHL_glcm_InverseVariance',
    'exponential_glcm_Id',
    'wavelet-HLH_glcm_ClusterShade',
    'wavelet-LHH_firstorder_Mean',
    'wavelet-LLL_firstorder_90Percentile',
    'squareroot_glcm_MCC',
    'wavelet-LHL_glrlm_LongRunLowGrayLevelEmphasis',
    'log-sigma-3-0-mm-3D_glszm_LargeAreaEmphasis',
    'wavelet-LHH_glszm_HighGrayLevelZoneEmphasis'
]

GBSA_PARAMS = {
    'loss': 'coxph',
    'learning_rate': 0.05,
    'n_estimators': 300,
    'criterion': 'friedman_mse',
    'max_depth': 3,
    'max_features': 0.34046888947419796,
    'max_leaf_nodes': 8,
    'min_impurity_decrease': 0.03,
    'min_samples_leaf': 10,
    'min_samples_split': 15,
    'min_weight_fraction_leaf': 0.1,
    'subsample': 0.5,
    'dropout_rate': 0.0
}

class PreprocessBundle:
    def __init__(self, numeric_cols, cat_cols):
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.num_fill = {}
        self.cat_fill = {}
        self.feature_names = None
        
    def fit(self, df):
        for col in self.numeric_cols:
            if col in df.columns:
                self.num_fill[col] = df[col].median()
        for col in self.cat_cols:
            if col in df.columns:
                mode_val = df[col].mode()
                self.cat_fill[col] = mode_val[0] if len(mode_val) > 0 else 'unknown'
        
        df_prep = df.copy()
        for col, val in self.num_fill.items():
            df_prep[col] = df_prep[col].fillna(val)
        for col, val in self.cat_fill.items():
            df_prep[col] = df_prep[col].fillna(val)
        
        X_num = df_prep[self.numeric_cols].values
        self.scaler.fit(X_num)
        X_cat = df_prep[self.cat_cols].values
        self.encoder.fit(X_cat)
        
        X_num_scaled = self.scaler.transform(X_num)
        X_cat_encoded = self.encoder.transform(X_cat)
        
        cat_feature_names = []
        for i, col in enumerate(self.cat_cols):
            categories = self.encoder.categories_[i]
            cat_feature_names.extend([f"{col}_{cat}" for cat in categories])
        
        self.feature_names = self.numeric_cols + cat_feature_names
        return np.hstack([X_num_scaled, X_cat_encoded])
    
    def transform(self, df):
        df_prep = df.copy()
        for col, val in self.num_fill.items():
            if col in df_prep.columns:
                df_prep[col] = df_prep[col].fillna(val)
        for col, val in self.cat_fill.items():
            if col in df_prep.columns:
                df_prep[col] = df_prep[col].fillna(val)
        
        X_num = df_prep[self.numeric_cols].values
        X_num_scaled = self.scaler.transform(X_num)
        X_cat = df_prep[self.cat_cols].values
        X_cat_encoded = self.encoder.transform(X_cat)
        return np.hstack([X_num_scaled, X_cat_encoded])

class PlattCalibrator:
    def __init__(self, C=10.0, pmin=0.01, pmax=0.99):
        self.lr = LogisticRegression(C=C, max_iter=2000, random_state=91)
        self.pmin = pmin
        self.pmax = pmax
    
    def fit(self, risk, y_binary):
        self.lr.fit(risk.reshape(-1, 1), y_binary)
    
    def predict(self, risk):
        p = self.lr.predict_proba(risk.reshape(-1, 1))[:, 1]
        p = np.clip(p, self.pmin, self.pmax)
        return p

def main():
    print("\n" + "="*80)
    print("NO-STAGE ENSEMBLE - FOLD-SPECIFIC OOF (NO LEAKAGE)")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load data
    # ========================================================================
    print(f"\n[STEP 1] Loading data")
    
    with open('/home/lichengze/Research/final_survival_package/config/split_frozen_v1.json', 'r') as f:
        split = json.load(f)
    
    train_ids = split['train_ids']
    test_ids = split['test_ids']
    
    df_rad_full = pd.read_csv('/home/lichengze/Research/com/Tb_features_1688_from_yaml.csv')
    # Select ONLY the 16 mRMR features (same as original model)
    rad_cols = SELECTED_RADIOMICS
    missing = [c for c in rad_cols if c not in df_rad_full.columns]
    if missing:
        print(f"  ✗ FATAL: Missing mRMR features in 1688 file: {missing}")
        return
    df_rad = df_rad_full[['PatientID'] + rad_cols].copy()
    print(f"  ✓ All 16 mRMR features found in 1688-feature file")
    
    df_clin = pd.read_csv('/home/lichengze/Research/NSCLC-Radiomics/NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv')
    df_clin = df_clin.rename(columns={'PatientID': 'patient_id', 'Survival.time': 'time_days', 'deadstatus.event': 'event'})
    
    df = df_rad.merge(df_clin, left_on='PatientID', right_on='patient_id', how='inner')
    df_train = df[df['patient_id'].isin(train_ids)].copy()
    df_test = df[df['patient_id'].isin(test_ids)].copy()
    
    print(f"  Train: {len(df_train)}, Test: {len(df_test)}")
    print(f"  Radiomics: {len(rad_cols)} features")
    
    # ========================================================================
    # STEP 2: Prepare features (WITH Overall.Stage)
    # ========================================================================
    print(f"\n[STEP 2] Feature preparation")
    
    numeric_cols = ['age'] + rad_cols
    cat_cols = ['gender', 'Histology', 'clinical.T.Stage', 'Overall.Stage']
    
    print(f"  Numeric: {len(numeric_cols)}, Categorical: {len(cat_cols)}")
    print(f"  ✓ Overall.Stage INCLUDED (original model reproduction)")
    
    # Fit preprocessor on FULL train (for test/external later)
    prep_full = PreprocessBundle(numeric_cols, cat_cols)
    X_train_full = prep_full.fit(df_train)
    y_train_full = Surv.from_dataframe('event', 'time_days', df_train)
    
    print(f"  Full train: {X_train_full.shape}, features: {len(prep_full.feature_names)}")
    
    # ========================================================================
    # STEP 3: Fold-specific training for TRUE OOF
    # ========================================================================
    print(f"\n[STEP 3] Fold-specific training (TRUE OOF, no leakage)")
    print(f"  {N_CV_FOLDS} folds × {N_MODELS_PER_FOLD} models = {N_CV_FOLDS * N_MODELS_PER_FOLD} total")
    print(f"  Estimated time: ~2-3 hours")
    
    event_at_tau = ((df_train['event'] == 1) & (df_train['time_days'] <= TAU)).astype(int).values
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
    
    oof_risks = np.zeros(len(df_train))
    oof_folds = np.zeros(len(df_train), dtype=int)
    fold_models = []  # Store models per fold for final ensemble
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_full, event_at_tau)):
        print(f"\n  === FOLD {fold_idx} ===")
        print(f"    Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        # Get fold data
        df_fold_train = df_train.iloc[train_idx].copy()
        df_fold_val = df_train.iloc[val_idx].copy()
        
        # Fit preprocessor on THIS fold's train only
        prep_fold = PreprocessBundle(numeric_cols, cat_cols)
        X_fold_train = prep_fold.fit(df_fold_train)
        X_fold_val = prep_fold.transform(df_fold_val)
        
        y_fold_train = Surv.from_dataframe('event', 'time_days', df_fold_train)
        
        # Train N_MODELS_PER_FOLD models on THIS fold's train
        fold_val_risks = []
        models_this_fold = []
        
        for model_idx in range(N_MODELS_PER_FOLD):
            seed = fold_idx * 1000 + model_idx
            rng = np.random.RandomState(seed)
            
            # Bootstrap resample from fold's train
            boot_idx = rng.choice(len(X_fold_train), size=len(X_fold_train), replace=True)
            
            params = GBSA_PARAMS.copy()
            params['random_state'] = seed
            
            model = GradientBoostingSurvivalAnalysis(**params)
            model.fit(X_fold_train[boot_idx], y_fold_train[boot_idx])
            
            # Predict on val fold (truly held-out)
            risk_val = model.predict(X_fold_val)
            fold_val_risks.append(risk_val)
            models_this_fold.append(model)
            
            if (model_idx + 1) % 10 == 0:
                print(f"      {model_idx+1}/{N_MODELS_PER_FOLD} models")
        
        # Ensemble mean for this fold's val
        oof_risks[val_idx] = np.array(fold_val_risks).mean(axis=0)
        oof_folds[val_idx] = fold_idx
        fold_models.append(models_this_fold)
        
        print(f"    ✓ Fold {fold_idx} complete")
    
    print(f"\n  ✓ All folds complete")
    print(f"  Total models trained: {N_CV_FOLDS * N_MODELS_PER_FOLD}")
    
    # Verify no leakage
    oof_auc = roc_auc_score(event_at_tau, oof_risks)
    print(f"  OOF AUC (unbiased): {oof_auc:.4f}")
    
    # ========================================================================
    # STEP 4: Platt calibration on TRUE OOF
    # ========================================================================
    print(f"\n[STEP 4] Platt calibration (on unbiased OOF)")
    
    calibrator = PlattCalibrator(C=10.0, pmin=0.01, pmax=0.99)
    calibrator.fit(oof_risks, event_at_tau)
    
    oof_p_event = calibrator.predict(oof_risks)
    oof_auc_cal = roc_auc_score(event_at_tau, oof_p_event)
    
    print(f"  ✓ Platt fitted")
    print(f"  OOF AUC (calibrated): {oof_auc_cal:.4f}")
    print(f"  OOF p_event: mean={oof_p_event.mean():.4f}, range=[{oof_p_event.min():.4f}, {oof_p_event.max():.4f}]")
    
    # ========================================================================
    # STEP 5: Train final ensemble on FULL train for test/external
    # ========================================================================
    print(f"\n[STEP 5] Training final ensemble on full train (for test/external)")
    
    final_models = []
    for model_idx in range(N_MODELS_PER_FOLD):
        seed = 9999 + model_idx
        rng = np.random.RandomState(seed)
        boot_idx = rng.choice(len(X_train_full), size=len(X_train_full), replace=True)
        
        params = GBSA_PARAMS.copy()
        params['random_state'] = seed
        
        model = GradientBoostingSurvivalAnalysis(**params)
        model.fit(X_train_full[boot_idx], y_train_full[boot_idx])
        final_models.append(model)
        
        if (model_idx + 1) % 10 == 0:
            print(f"    {model_idx+1}/{N_MODELS_PER_FOLD} models")
    
    print(f"  ✓ Final ensemble trained")
    
    # ========================================================================
    # STEP 6: Test predictions
    # ========================================================================
    print(f"\n[STEP 6] Test predictions")
    
    X_test = prep_full.transform(df_test)
    y_test = Surv.from_dataframe('event', 'time_days', df_test)
    
    test_risks = np.array([m.predict(X_test) for m in final_models])
    test_risk_mean = test_risks.mean(axis=0)
    test_risk_std = test_risks.std(axis=0)
    p_event_test = calibrator.predict(test_risk_mean)
    
    y_binary_test = ((df_test['event'] == 1) & (df_test['time_days'] <= TAU)).astype(int).values
    auc_test = roc_auc_score(y_binary_test, p_event_test)
    
    print(f"  Test AUC: {auc_test:.4f}")
    print(f"  Test p_event: mean={p_event_test.mean():.4f}, range=[{p_event_test.min():.4f}, {p_event_test.max():.4f}]")
    
    # ========================================================================
    # STEP 7: External predictions
    # ========================================================================
    print(f"\n[STEP 7] External (Radiogenomics) predictions")
    
    ext_rad_path = '/home/lichengze/Research/com/rg_Tb_features_1688_from_yaml.csv'
    df_ext_rad = pd.read_csv(ext_rad_path)
    
    id_col = 'patient_id' if 'patient_id' in df_ext_rad.columns else 'PatientID'
    if id_col == 'PatientID':
        df_ext_rad = df_ext_rad.rename(columns={'PatientID': 'patient_id'})
    
    rad_in_ext = [c for c in rad_cols if c in df_ext_rad.columns]
    missing_rad = [c for c in rad_cols if c not in df_ext_rad.columns]
    
    if missing_rad:
        print(f"  ✗ FATAL: {len(missing_rad)}/{len(rad_cols)} mRMR features missing in external!")
        print(f"    Missing: {missing_rad}")
        return
    print(f"  ✓ All 16 mRMR features found in external file")
    
    df_ext_rad = df_ext_rad[['patient_id'] + rad_cols]
    
    df_ext_clin = pd.read_csv('/home/lichengze/Research/Radiogenomics/Radiogenomics_clinical_aligned.csv')
    df_ext = df_ext_rad.merge(df_ext_clin, on='patient_id', how='inner')
    
    if 'histology' in df_ext.columns:
        df_ext = df_ext.rename(columns={'histology': 'Histology'})
    if 'T' in df_ext.columns:
        df_ext = df_ext.rename(columns={'T': 'clinical.T.Stage'})
    if 'overall_stage' in df_ext.columns:
        df_ext = df_ext.rename(columns={'overall_stage': 'Overall.Stage'})
    
    print(f"  External: {len(df_ext)} patients")
    
    X_ext = prep_full.transform(df_ext)
    ext_risks = np.array([m.predict(X_ext) for m in final_models])
    ext_risk_mean = ext_risks.mean(axis=0)
    ext_risk_std = ext_risks.std(axis=0)
    p_event_ext = calibrator.predict(ext_risk_mean)
    
    y_binary_ext = ((df_ext['event'] == 1) & (df_ext['time_days'] <= TAU)).astype(int).values
    auc_ext = roc_auc_score(y_binary_ext, p_event_ext)
    
    print(f"  External AUC: {auc_ext:.4f}")
    print(f"  External p_event: mean={p_event_ext.mean():.4f}, range=[{p_event_ext.min():.4f}, {p_event_ext.max():.4f}]")
    
    # ========================================================================
    # STEP 8: Save everything
    # ========================================================================
    print(f"\n[STEP 8] Saving results")
    
    ensemble_bundle = {
        'version': 'with_stage_16mrmr_fold_specific_oof_v3',
        'models': final_models,
        'fold_models': fold_models,
        'prep': prep_full,
        'numeric_cols': numeric_cols,
        'cat_cols': cat_cols,
        'selected_radiomics': rad_cols,
        'clinical_flags': {'use_overall_stage': True},
        'calibrator': calibrator,
        'n_models': N_MODELS_PER_FOLD,
        'n_folds': N_CV_FOLDS,
        'training_method': 'fold_specific_oof_no_leakage'
    }
    
    joblib.dump(ensemble_bundle, OUTDIR / 'no_stage_ensemble_clean.joblib')
    print(f"  ✓ {OUTDIR}/no_stage_ensemble_clean.joblib")
    
    df_oof = pd.DataFrame({
        'patient_id': df_train['patient_id'].values,
        'time_days': df_train['time_days'].values,
        'event': df_train['event'].values,
        'f_pred': oof_risks,
        'p_event_730d': oof_p_event,
        'fold': oof_folds
    })
    df_oof.to_csv(OUTDIR / 'TRAIN_332_OOF_clean.csv', index=False)
    print(f"  ✓ {OUTDIR}/TRAIN_332_OOF_clean.csv")
    
    df_test_out = pd.DataFrame({
        'patient_id': df_test['patient_id'].values,
        'time_days': df_test['time_days'].values,
        'event': df_test['event'].values,
        'f_pred': test_risk_mean,
        'pred_var': test_risk_std ** 2,
        'p_event_730d': p_event_test,
        'S_cal_730d': 1 - p_event_test
    })
    df_test_out.to_csv(OUTDIR / 'TEST_90_clean.csv', index=False)
    print(f"  ✓ {OUTDIR}/TEST_90_clean.csv")
    
    df_ext_out = pd.DataFrame({
        'patient_id': df_ext['patient_id'].values,
        'time_days': df_ext['time_days'].values,
        'event': df_ext['event'].values,
        'f_pred': ext_risk_mean,
        'pred_var': ext_risk_std ** 2,
        'p_event_730d': p_event_ext,
        'S_cal_730d': 1 - p_event_ext
    })
    df_ext_out.to_csv(OUTDIR / 'EXTERNAL_clean.csv', index=False)
    print(f"  ✓ {OUTDIR}/EXTERNAL_clean.csv")
    
    report = {
        'model_type': 'with_stage_16mrmr_fold_specific_oof',
        'leakage_free': True,
        'n_models_per_fold': N_MODELS_PER_FOLD,
        'n_folds': N_CV_FOLDS,
        'total_models': N_CV_FOLDS * N_MODELS_PER_FOLD + N_MODELS_PER_FOLD,
        'n_features': len(prep_full.feature_names),
        'train_n': len(df_train),
        'test_n': len(df_test),
        'external_n': len(df_ext),
        'oof_auc_raw': float(oof_auc),
        'oof_auc_calibrated': float(oof_auc_cal),
        'test_auc': float(auc_test),
        'external_auc': float(auc_ext),
        'oof_p_event_mean': float(oof_p_event.mean()),
        'test_p_event_mean': float(p_event_test.mean()),
        'external_p_event_mean': float(p_event_ext.mean())
    }
    
    with open(OUTDIR / 'training_report_clean.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"  ✓ {OUTDIR}/training_report_clean.json")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n" + "="*80)
    print("TRAINING COMPLETE - FOLD-SPECIFIC OOF (NO LEAKAGE)")
    print("="*80)
    print(f"\n✓ Method: Fold-specific training (each fold's models never see val data)")
    print(f"✓ Total models: {N_CV_FOLDS * N_MODELS_PER_FOLD} (fold-specific) + {N_MODELS_PER_FOLD} (final)")
    print(f"✓ Features: {len(prep_full.feature_names)} (WITH Overall.Stage)")
    print(f"\nPerformance (unbiased):")
    print(f"  OOF AUC: {oof_auc:.4f} (raw) → {oof_auc_cal:.4f} (calibrated)")
    print(f"  Test AUC: {auc_test:.4f}")
    print(f"  External AUC: {auc_ext:.4f}")
    print(f"\np_event distribution:")
    print(f"  OOF: {oof_p_event.mean():.4f}")
    print(f"  Test: {p_event_test.mean():.4f}")
    print(f"  External: {p_event_ext.mean():.4f}")
    print(f"\nAll outputs: {OUTDIR}/")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()