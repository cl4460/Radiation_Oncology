#!/usr/bin/env python3
"""
SANITY CHECK for classwise_conformal.py
========================================
Verifies critical invariants that MUST hold:

1. Prob threshold baseline: marginal_cov MUST equal accuracy (NOT 1.0)
2. All methods: marginal_cov <= 1.0
3. All methods: event_cov, surv_cov <= 1.0
4. Deferral + retention = 1.0
5. No NaN in critical metrics

This script MUST pass before any results can be trusted.
"""
import sys
import numpy as np
from classwise_conformal import run_prob_threshold, run_standard_cp, run_classwise_cp

def test_prob_threshold_marginal_cov():
    """CRITICAL: Prob threshold marginal_cov must equal accuracy, NOT 1.0"""
    np.random.seed(42)
    n = 100
    p_test = np.random.uniform(0.3, 0.8, n)
    y_test = (np.random.rand(n) < p_test).astype(int)
    
    result = run_prob_threshold(p_test, y_test, threshold=0.5)
    
    # Compute expected accuracy
    pred = (p_test >= 0.5).astype(int)
    expected_acc = (pred == y_test).mean()
    
    assert abs(result.marginal_cov - expected_acc) < 1e-6, \
        f"FATAL: Prob threshold marginal_cov={result.marginal_cov:.4f}, but accuracy={expected_acc:.4f}"
    
    assert result.marginal_cov != 1.0 or expected_acc == 1.0, \
        f"FATAL: Prob threshold marginal_cov is hardcoded to 1.0 (accuracy={expected_acc:.4f})"
    
    print(f"✓ Prob threshold: marginal_cov={result.marginal_cov:.4f} == accuracy={expected_acc:.4f}")

def test_conformal_coverage_bounds():
    """All coverage metrics must be in [0, 1]"""
    np.random.seed(43)
    n_cal, n_test = 50, 50
    p_cal = np.random.uniform(0.2, 0.9, n_cal)
    y_cal = (np.random.rand(n_cal) < p_cal).astype(int)
    p_test = np.random.uniform(0.2, 0.9, n_test)
    y_test = (np.random.rand(n_test) < p_test).astype(int)
    
    for method_name, method_func in [
        ("Standard CP", lambda: run_standard_cp(p_cal, y_cal, p_test, y_test, alpha=0.1)),
        ("Classwise CP", lambda: run_classwise_cp(p_cal, y_cal, p_test, y_test, alpha=0.1)),
    ]:
        result = method_func()
        
        assert 0 <= result.marginal_cov <= 1.0, \
            f"FATAL: {method_name} marginal_cov={result.marginal_cov:.4f} out of [0,1]"
        assert 0 <= result.event_cov <= 1.0, \
            f"FATAL: {method_name} event_cov={result.event_cov:.4f} out of [0,1]"
        assert 0 <= result.surv_cov <= 1.0, \
            f"FATAL: {method_name} surv_cov={result.surv_cov:.4f} out of [0,1]"
        assert 0 <= result.defer_rate <= 1.0, \
            f"FATAL: {method_name} defer_rate={result.defer_rate:.4f} out of [0,1]"
        
        # Deferral + retention = 1.0
        assert abs(result.defer_rate + result.retention_rate - 1.0) < 1e-6, \
            f"FATAL: {method_name} defer_rate + retention_rate != 1.0"
        
        print(f"✓ {method_name}: all metrics in valid range")

def test_no_nan_in_critical_metrics():
    """Critical metrics must not be NaN"""
    np.random.seed(44)
    n_cal, n_test = 30, 30
    p_cal = np.random.uniform(0.3, 0.7, n_cal)
    y_cal = (np.random.rand(n_cal) < 0.5).astype(int)
    p_test = np.random.uniform(0.3, 0.7, n_test)
    y_test = (np.random.rand(n_test) < 0.5).astype(int)
    
    result = run_standard_cp(p_cal, y_cal, p_test, y_test, alpha=0.1)
    
    critical_metrics = [
        "marginal_cov", "event_cov", "surv_cov", 
        "defer_rate", "retention_rate", "acc_accepted"
    ]
    
    for metric in critical_metrics:
        val = getattr(result, metric)
        assert not np.isnan(val), f"FATAL: {metric} is NaN"
    
    print(f"✓ No NaN in critical metrics")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SANITY CHECK: classwise_conformal.py")
    print("="*80 + "\n")
    
    try:
        test_prob_threshold_marginal_cov()
        test_conformal_coverage_bounds()
        test_no_nan_in_critical_metrics()
        
        print("\n" + "="*80)
        print("✓ ALL SANITY CHECKS PASSED")
        print("="*80 + "\n")
        sys.exit(0)
        
    except AssertionError as e:
        print("\n" + "="*80)
        print("✗ SANITY CHECK FAILED")
        print("="*80)
        print(f"\n{e}\n")
        print("DO NOT USE ANY RESULTS UNTIL THIS IS FIXED.\n")
        sys.exit(1)
