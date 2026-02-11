#!/usr/bin/env python3
"""Compute reversal frequency from exp7c full results"""
import pandas as pd
import numpy as np
import sys
import os

def compute_reversal_freq(full_csv):
    df = pd.read_csv(full_csv)
    
    # Only Standard CP deferral reversal: Raw vs CalOnly_Corrected
    std = df[df['method'] == 'Standard']
    
    # Pivot to get defer by (repeat, version, model)
    piv = std.pivot_table(
        index=['repeat', 'version'], 
        columns='model', 
        values='defer'
    )
    
    # Extract Raw and CalOnly
    raw = piv.xs('Raw', level='version')
    cal = piv.xs('CalOnly_Corrected', level='version')
    
    # Compute deltas
    d_raw = raw['Clinical'] - raw['Radiomics']
    d_cal = cal['Clinical'] - cal['Radiomics']
    
    # Reversal = sign flip
    reversal = (np.sign(d_raw) != np.sign(d_cal))
    
    res = {
        'n_repeats': len(raw),
        'raw_clin_higher_pct': float((d_raw > 0).mean() * 100),
        'cal_clin_higher_pct': float((d_cal > 0).mean() * 100),
        'reversal_pct': float(reversal.mean() * 100),
        'median_delta_raw': float(np.median(d_raw)),
        'median_delta_cal': float(np.median(d_cal))
    }
    
    return res

if __name__ == '__main__':
    import glob
    
    print("="*80)
    print("REVERSAL FREQUENCY ANALYSIS")
    print("="*80)
    
    for d in sorted(glob.glob("exp7c_v2_CLEAN_123_alpha*/")):
        full_csv = os.path.join(d, "exp7c_full_results.csv")
        if not os.path.exists(full_csv):
            continue
        
        alpha = d.split('alpha')[1].rstrip('/').replace('p', '.')
        res = compute_reversal_freq(full_csv)
        
        print(f"\nAlpha = {alpha}:")
        print(f"  Repeats: {res['n_repeats']}")
        print(f"  Raw: Clinical higher in {res['raw_clin_higher_pct']:.1f}% of splits")
        print(f"  CalOnly: Clinical higher in {res['cal_clin_higher_pct']:.1f}% of splits")
        print(f"  Reversal frequency: {res['reversal_pct']:.1f}%")
        print(f"  Median Δ(defer): Raw={res['median_delta_raw']:.3f}, Cal={res['median_delta_cal']:.3f}")
        
        # Save
        out_csv = os.path.join(d, "reversal_frequency_standard.csv")
        pd.DataFrame([res]).to_csv(out_csv, index=False)
        print(f"  ✓ Saved: {out_csv}")
