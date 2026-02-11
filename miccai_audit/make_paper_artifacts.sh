#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/lichengze/miniconda3/envs/ucla/bin/python3

echo "================================================================================"
echo "PAPER ARTIFACTS GENERATION"
echo "================================================================================"

echo ""
echo "[1/5] Regenerate Table 1"
$PYTHON make_table1.py

echo ""
echo "[2/5] Reversal frequency (Standard CP)"
$PYTHON compute_reversal_frequency.py

echo ""
echo "[3/5] Floor overlay figure"
$PYTHON make_floor_overlay.py

echo ""
echo "[4/5] Reversal evidence figure (if exists)"
if [ -f make_exp7c_reversal_figure.py ]; then
    $PYTHON make_exp7c_reversal_figure.py
else
    echo "  (Script not found, skipping)"
fi

echo ""
echo "[5/5] Sanity summary from master cohort"
$PYTHON - <<'PY'
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

m = pd.read_csv("EXTERNAL_CLEAN_123.csv")
y = m["y_2y"].values
p = m["p_event_730d"].values
pi = y.mean()

print("="*80)
print("MASTER CHECK: n=%d events=%d pi=%.4f" % (len(m), int(y.sum()), pi))
print("Radiomics: mean_p=%.4f overest=%.2fx AUC=%.4f Brier=%.4f" % (
    p.mean(), p.mean()/pi, roc_auc_score(y,p), np.mean((p-y)**2)
))
print("="*80)
PY

echo ""
echo "================================================================================"
echo "DONE. All paper artifacts generated."
echo "================================================================================"
