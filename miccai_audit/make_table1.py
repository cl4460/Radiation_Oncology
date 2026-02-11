#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

def ece(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.linspace(0, 1, n_bins+1)
    idx = np.digitize(p, bins) - 1
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0: continue
        e += (m.sum()/len(y)) * abs(p[m].mean() - y[m].mean())
    return e

master = pd.read_csv("EXTERNAL_CLEAN_123.csv")
rad = pd.read_csv("exp1_calibration_FINAL/radiomics_external_predictions_CLEAN.csv")
cli = pd.read_csv("exp2_clinical_baseline_FINAL/clinical_external_predictions_CLEAN.csv")

# Merge to get y_2y from master
rad = rad.merge(master[['patient_id', 'y_2y']], on='patient_id', how='left')
cli = cli.merge(master[['patient_id', 'y_2y']], on='patient_id', how='left')

data = []
for name, df, pcol in [('Radiomics', rad, 'p_event'), ('Clinical', cli, 'p_event')]:
    y = df['y_2y'].astype(int).values
    p = df[pcol].astype(float).values
    n, ev, pi = len(y), int(y.sum()), y.mean()
    mp = p.mean()
    auc = roc_auc_score(y, p)
    br = brier_score_loss(y, p)
    bc = pi * (1-pi)
    bs = 1 - br/bc if bc>0 else 0
    data.append({
        'model':name, 'n':n, 'events':ev, 'pi':round(pi,4),
        'mean_p':round(mp,4), 'overest':round(mp/pi,2),
        'auc':round(auc,4), 'brier':round(br,4),
        'brier_clim':round(bc,4), 'skill':round(bs,4),
        'ece':round(ece(y,p),4)
    })

tab = pd.DataFrame(data)
tab.to_csv("paper_tables/table1_external_audit.csv", index=False)
print(tab)
print("\nSaved: paper_tables/table1_external_audit.csv")
