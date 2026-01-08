#!/usr/bin/env python3
"""
Learning Rate Results Viewer for learning_rate_corrected
Analyzes and displays results from run_all_lr_experiments.sh
Only prints results, no file saving
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

# =====================================================================
# CONFIGURATION
# =====================================================================

BASE_DIR = Path("/home/lichengze/Research/CNN_pipeline/phase3_outputs/learning_rate/learning_rate_corrected")
OUTPUT_DIR = BASE_DIR / "output"

# Learning rates - auto-detected from output directory
def get_all_learning_rates():
    """Automatically detect all learning rates from output directory."""
    if not OUTPUT_DIR.exists():
        return []
    lr_dirs = sorted([d.name.replace('lr_', '') for d in OUTPUT_DIR.iterdir() 
                      if d.is_dir() and d.name.startswith('lr_')])
    return lr_dirs

LEARNING_RATES = get_all_learning_rates()

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def load_lr_results(lr: str) -> Dict:
    """Load results for a specific LR from results_complete.csv or individual fold files."""
    lr_dir = OUTPUT_DIR / f"lr_{lr}"
    
    if not lr_dir.exists():
        return None
    
    # Try loading from results_complete.csv first
    results_file = lr_dir / "results_complete.csv"
    if results_file.exists():
        try:
            df = pd.read_csv(results_file)
            if 'best_uno' in df.columns and len(df) > 0:
                return {
                    'fold_unos': df['best_uno'].tolist(),
                    'lr': lr,
                    'complete': True
                }
        except Exception as e:
            pass
    
    # Try loading individual fold files
    fold_unos = []
    for fold_idx in range(5):
        fold_file = lr_dir / f"results_fold{fold_idx}.csv"
        if fold_file.exists():
            try:
                df_fold = pd.read_csv(fold_file)
                if 'best_uno' in df_fold.columns and len(df_fold) > 0:
                    fold_unos.append(df_fold['best_uno'].values[0])
            except Exception as e:
                pass
    
    if len(fold_unos) > 0:
        return {
            'fold_unos': fold_unos,
            'lr': lr,
            'complete': len(fold_unos) == 5
        }
    
    return None


def collect_all_results() -> pd.DataFrame:
    """Collect results from all LR experiments."""
    data = []
    
    print("📥 Collecting results from all experiments...")
    print()
    
    for lr in LEARNING_RATES:
        results = load_lr_results(lr)
        
        if results and results['fold_unos']:
            fold_unos = results['fold_unos']
            n_folds = len(fold_unos)
            status = "✅" if n_folds == 5 else f"⚠️  ({n_folds}/5)"
            print(f"  {lr:<15} {status:<15} Mean: {np.mean(fold_unos):.3f}")
            
            data.append({
                'LR': lr,
                'Fold_0': fold_unos[0] if len(fold_unos) > 0 else np.nan,
                'Fold_1': fold_unos[1] if len(fold_unos) > 1 else np.nan,
                'Fold_2': fold_unos[2] if len(fold_unos) > 2 else np.nan,
                'Fold_3': fold_unos[3] if len(fold_unos) > 3 else np.nan,
                'Fold_4': fold_unos[4] if len(fold_unos) > 4 else np.nan,
                'Mean': np.nanmean(fold_unos),
                'Std': np.nanstd(fold_unos),
                'Min': np.nanmin(fold_unos),
                'Max': np.nanmax(fold_unos),
                'Range': np.nanmax(fold_unos) - np.nanmin(fold_unos),
                'N_Folds': n_folds
            })
        else:
            print(f"  {lr:<15} ❌ No results found")
    
    print()
    return pd.DataFrame(data)


def print_summary_table(df: pd.DataFrame):
    """Print comprehensive summary table."""
    print("=" * 140)
    print(f"LEARNING RATE COMPARISON ({len(df)} LRs)")
    print("=" * 140)
    
    # Sort by Mean Uno (descending)
    df_sorted = df.sort_values('Mean', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    
    # Calculate relative performance
    best_mean = df_sorted['Mean'].max()
    df_sorted['vs_Best'] = df_sorted['Mean'] - best_mean
    df_sorted['Relative_%'] = (df_sorted['vs_Best'] / best_mean) * 100
    
    # Add status emoji
    def get_status(rel_perf):
        if rel_perf > -0.005:
            return "🏆"
        elif rel_perf > -0.01:
            return "✅"
        elif rel_perf > -0.02:
            return "📊"
        else:
            return "⚠️"
    
    df_sorted['Status'] = df_sorted['vs_Best'].apply(get_status)
    
    # Print table header
    print(f"{'Rank':<7} {'LR':<12} {'Mean':<10} {'Std':<10} {'Min':<10} "
          f"{'Max':<10} {'Range':<10} {'vs Best':<10} {'Relative%':<12} {'Folds':<8} {'Status':<8}")
    print("-" * 140)
    
    # Print each row
    for _, row in df_sorted.iterrows():
        rank_str = f"{int(row['Rank'])}"
        if row['Rank'] == 1:
            rank_str += " 🥇"
        elif row['Rank'] == 2:
            rank_str += " 🥈"
        elif row['Rank'] == 3:
            rank_str += " 🥉"
        
        n_folds_str = f"{int(row['N_Folds'])}/5"
        
        print(f"{rank_str:<7} {row['LR']:<12} {row['Mean']:<10.3f} {row['Std']:<10.3f} "
              f"{row['Min']:<10.3f} {row['Max']:<10.3f} {row['Range']:<10.3f} "
              f"{row['vs_Best']:+.3f}    {row['Relative_%']:+.2f}%       "
              f"{n_folds_str:<8} {row['Status']:<8}")
    
    print("=" * 140)
    print()


def print_top_performers(df: pd.DataFrame, n=5):
    """Print top N performers."""
    print("=" * 100)
    print(f"🏆 TOP {n} LEARNING RATES")
    print("=" * 100)
    
    df_sorted = df.sort_values('Mean', ascending=False)
    top_n = df_sorted.head(n)
    
    print(f"{'Rank':<7} {'LR':<15} {'Mean Uno':<12} {'Std':<12} {'Range':<20} {'Folds':<8}")
    print("-" * 100)
    
    for rank, (_, row) in enumerate(top_n.iterrows(), 1):
        rank_str = f"{rank}"
        if rank == 1:
            rank_str += " 🥇"
        elif rank == 2:
            rank_str += " 🥈"
        elif rank == 3:
            rank_str += " 🥉"
        
        range_str = f"[{row['Min']:.3f}, {row['Max']:.3f}]"
        folds_str = f"{int(row['N_Folds'])}/5"
        
        print(f"{rank_str:<7} {row['LR']:<15} {row['Mean']:<12.3f} {row['Std']:<12.3f} "
              f"{range_str:<20} {folds_str:<8}")
    
    print("=" * 100)
    print()


def print_fold_details(df: pd.DataFrame):
    """Print detailed fold-by-fold results."""
    print("=" * 110)
    print("FOLD-BY-FOLD RESULTS")
    print("=" * 110)
    
    # Sort by Mean Uno (descending)
    df_sorted = df.sort_values('Mean', ascending=False)
    
    print(f"{'LR':<12} {'Fold 0':<12} {'Fold 1':<12} {'Fold 2':<12} "
          f"{'Fold 3':<12} {'Fold 4':<12} {'Mean':<12}")
    print("-" * 110)
    
    for _, row in df_sorted.iterrows():
        status = "🏆" if row['Mean'] == df_sorted['Mean'].max() else ""
        f0 = f"{row['Fold_0']:.3f}" if not np.isnan(row['Fold_0']) else "N/A"
        f1 = f"{row['Fold_1']:.3f}" if not np.isnan(row['Fold_1']) else "N/A"
        f2 = f"{row['Fold_2']:.3f}" if not np.isnan(row['Fold_2']) else "N/A"
        f3 = f"{row['Fold_3']:.3f}" if not np.isnan(row['Fold_3']) else "N/A"
        f4 = f"{row['Fold_4']:.3f}" if not np.isnan(row['Fold_4']) else "N/A"
        
        print(f"{row['LR']:<12} {f0:<12} {f1:<12} {f2:<12} "
              f"{f3:<12} {f4:<12} {row['Mean']:<12.3f} {status}")
    
    print("=" * 110)
    print()


def analyze_lr_ranges(df: pd.DataFrame):
    """Analyze performance by LR ranges."""
    print("=" * 100)
    print("PERFORMANCE BY LEARNING RATE RANGE")
    print("=" * 100)
    
    # Convert LR to float
    def lr_to_float(lr_str):
        return float(lr_str.replace('e-', 'e-'))
    
    df['LR_float'] = df['LR'].apply(lr_to_float)
    
    ranges = [
        ("Very Low (≤8e-5)", 0, 8e-5),
        ("Low-Mid (8e-5 to 9.5e-5)", 8e-5, 9.5e-5),
        ("High (≥6.8e-4)", 6.8e-4, 1.0)
    ]
    
    print(f"{'Range':<35} {'N':<6} {'Mean':<12} {'Std':<12} {'Best':<15}")
    print("-" * 100)
    
    for range_name, lower, upper in ranges:
        mask = (df['LR_float'] > lower) & (df['LR_float'] <= upper)
        subset = df[mask]
        
        if not subset.empty:
            n = len(subset)
            mean = subset['Mean'].mean()
            std = subset['Mean'].std() if len(subset) > 1 else 0.0
            best = subset['Mean'].max()
            best_lr = subset.loc[subset['Mean'].idxmax(), 'LR']
            
            print(f"{range_name:<35} {n:<6} {mean:<12.3f} {std:<12.3f} "
                  f"{best:.3f} ({best_lr})")
    
    print("=" * 100)
    print()


def print_detailed_analysis(df: pd.DataFrame):
    """Print detailed analysis and trends."""
    print("=" * 100)
    print("DETAILED ANALYSIS")
    print("=" * 100)
    
    # Best performer
    best_idx = df['Mean'].idxmax()
    best_row = df.loc[best_idx]
    
    print(f"\n🏆 BEST LEARNING RATE:")
    print(f"   LR: {best_row['LR']}")
    print(f"   Mean Uno: {best_row['Mean']:.3f}")
    print(f"   Std: {best_row['Std']:.3f}")
    print(f"   Range: [{best_row['Min']:.3f}, {best_row['Max']:.3f}]")
    print(f"   Variability: {best_row['Range']:.3f}")
    print()
    
    # Most stable (among top 5)
    top5 = df.nlargest(5, 'Mean')
    most_stable = top5.nsmallest(1, 'Std').iloc[0]
    
    print(f"📊 MOST STABLE (among top 5):")
    print(f"   LR: {most_stable['LR']}")
    print(f"   Mean: {most_stable['Mean']:.3f}")
    print(f"   Std: {most_stable['Std']:.3f}")
    print()
    
    # Fold analysis
    print("📁 FOLD PERFORMANCE:")
    fold_cols = ['Fold_0', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4']
    fold_means = df[fold_cols].mean()
    fold_stds = df[fold_cols].std()
    
    for i, (mean, std) in enumerate(zip(fold_means, fold_stds)):
        threshold = fold_means.mean() - 0.5 * fold_means.std()
        status = "⚠️" if mean < threshold else "✅"
        print(f"   Fold {i}: Mean={mean:.3f}, Std={std:.3f} {status}")
    
    print()
    
    # Completion status
    complete_count = (df['N_Folds'] == 5).sum()
    incomplete_count = len(df) - complete_count
    
    print(f"📈 COMPLETION STATUS:")
    print(f"   Complete (5/5 folds): {complete_count}/{len(df)}")
    if incomplete_count > 0:
        print(f"   Incomplete: {incomplete_count}/{len(df)}")
        incomplete_lrs = df[df['N_Folds'] < 5]['LR'].tolist()
        print(f"   Missing folds: {', '.join(incomplete_lrs)}")
    
    print("=" * 100)
    print()


def print_recommendations(df: pd.DataFrame):
    """Print final recommendations."""
    print("=" * 100)
    print("🎯 RECOMMENDATIONS")
    print("=" * 100)
    
    best_row = df.loc[df['Mean'].idxmax()]
    
    print(f"\n✅ RECOMMENDED LEARNING RATE: {best_row['LR']}")
    print(f"   Expected C-index: {best_row['Mean']:.3f} ± {best_row['Std']:.3f}")
    print()
    
    # Check if all experiments are complete
    if (df['N_Folds'] < 5).any():
        incomplete = df[df['N_Folds'] < 5]
        print("⚠️  INCOMPLETE EXPERIMENTS:")
        for _, row in incomplete.iterrows():
            print(f"   {row['LR']}: {int(row['N_Folds'])}/5 folds completed")
        print()
    else:
        print("✅ All experiments completed successfully!")
        print()
    
    print("📋 NEXT STEPS:")
    print("   1. Use best LR for final model training")
    print("   2. Consider ensemble of top 3 LRs")
    print("   3. Implement SWA (Stochastic Weight Averaging)")
    print("   4. Enable TTA for final testing")
    
    print("=" * 100)
    print()


# =====================================================================
# MAIN
# =====================================================================

def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 100)
    print("LEARNING RATE EXPERIMENT RESULTS")
    print(f"Directory: {OUTPUT_DIR}")
    print(f"Total LRs tested: {len(LEARNING_RATES)}")
    print(f"Learning rates: {', '.join(LEARNING_RATES)}")
    print("=" * 100 + "\n")
    
    # Collect results
    df = collect_all_results()
    
    if df.empty:
        print("❌ No results found! Check that experiments have completed.")
        print(f"   Expected location: {OUTPUT_DIR}")
        return
    
    print(f"✅ Successfully loaded results for {len(df)}/{len(LEARNING_RATES)} learning rates")
    print()
    
    # Print all analyses
    print_summary_table(df)
    print_top_performers(df, n=5)
    print_fold_details(df)
    analyze_lr_ranges(df)
    print_detailed_analysis(df)
    print_recommendations(df)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE!")
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()

