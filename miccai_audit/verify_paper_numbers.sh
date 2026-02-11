#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/lichengze/miniconda3/envs/ucla/bin/python3

echo "================================================================================"
echo "VERIFY PAPER NUMBERS AGAINST SINGLE SOURCE OF TRUTH"
echo "================================================================================"

$PYTHON - <<'PY'
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Load master cohort (single source of truth)
master = pd.read_csv("EXTERNAL_CLEAN_123.csv")
y_true = master["y_2y"].values
p_rad = master["p_event_730d"].values
pi = y_true.mean()

# Expected values from master
expected = {
    'n': len(master),
    'events': int(y_true.sum()),
    'pi': round(pi, 4),
    'rad_mean_p': round(p_rad.mean(), 4),
    'rad_overest': round(p_rad.mean()/pi, 2),
    'rad_auc': round(roc_auc_score(y_true, p_rad), 4),
    'rad_brier': round(np.mean((p_rad - y_true)**2), 4)
}

# Load Table 1
table1 = pd.read_csv("paper_tables/table1_external_audit.csv")
rad_row = table1[table1['model'] == 'Radiomics'].iloc[0]

# Verify
checks = [
    ('n', expected['n'], rad_row['n']),
    ('events', expected['events'], rad_row['events']),
    ('pi', expected['pi'], rad_row['pi']),
    ('mean_p', expected['rad_mean_p'], rad_row['mean_p']),
    ('overest', expected['rad_overest'], rad_row['overest']),
    ('auc', expected['rad_auc'], rad_row['auc']),
    ('brier', expected['rad_brier'], rad_row['brier']),
]

print("\nVerifying Radiomics row in Table 1:")
print("="*80)
all_pass = True
for name, exp, obs in checks:
    status = "✓ PASS" if exp == obs else "✗ FAIL"
    print(f"{name:12s} | Expected: {exp:8} | Table1: {obs:8} | {status}")
    if exp != obs:
        all_pass = False

print("="*80)
if all_pass:
    print("✓ ALL CHECKS PASSED")
else:
    print("✗ VERIFICATION FAILED - TABLE 1 DOES NOT MATCH MASTER")
    exit(1)

# Verify reversal frequencies exist
print("\nVerifying reversal frequency files:")
import os
for alpha in ['0p05', '0p10', '0p20']:
    path = f"exp7c_v2_CLEAN_123_alpha{alpha}/reversal_frequency_standard.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  {alpha.replace('p','.')}: {df['reversal_pct'].values[0]:.1f}% reversal ✓")
    else:
        print(f"  {alpha.replace('p','.')}: FILE MISSING ✗")
        exit(1)

print("\n✓ VERIFICATION COMPLETE")
PY
