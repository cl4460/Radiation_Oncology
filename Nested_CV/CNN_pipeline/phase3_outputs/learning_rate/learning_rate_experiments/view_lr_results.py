#!/usr/bin/env python3
"""
Comprehensive Learning Rate Experiment Results Viewer
Automatically detects and analyzes ALL learning rate experiments in learning_rate_experiments/
Displays complete results table, TOP 10 performers, and detailed statistics
All numerical values displayed with 3 decimal places
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# CONFIGURATION
# =====================================================================

BASE_DIR = Path("/home/lichengze/Research/CNN_pipeline")
RESULTS_DIR = BASE_DIR / "phase3_outputs" / "learning_rate_experiments" / "outputs"

# Automatically detect all LR experiments from directory
def get_all_experiments():
    """Scan directory for all LR experiments."""
    if not RESULTS_DIR.exists():
        return []
    
    lr_dirs = sorted([d.name for d in RESULTS_DIR.iterdir() 
                     if d.is_dir() and d.name.startswith('lr_')])
    
    # Extract LR names (remove 'lr_' prefix)
    return [d.replace('lr_', '') for d in lr_dirs]

LEARNING_RATES = get_all_experiments()
BASELINE_LR = "2e-5"  # Original baseline

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def load_lr_results(lr: str) -> Dict:
    """Load all results for a specific LR from results_complete.csv or individual fold files."""
    lr_dir = RESULTS_DIR / f"lr_{lr}"
    
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
                    'lr': lr
                }
        except Exception as e:
            pass
    
    # If results_complete.csv doesn't exist, try loading individual fold files
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
            'lr': lr
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
            status = "✅" if len(fold_unos) == 5 else f"⚠️ ({len(fold_unos)}/5)"
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
                'N_Folds': len(fold_unos)
            })
        else:
            print(f"  {lr:<15} ❌ No results found")
    
    print()
    return pd.DataFrame(data)


def get_baseline_mean(df: pd.DataFrame) -> float:
    """Get baseline mean Uno C-index."""
    baseline_row = df[df['LR'] == BASELINE_LR]
    if not baseline_row.empty:
        return baseline_row['Mean'].values[0]
    else:
        # Use first LR as baseline if 2e-5 not found
        return df['Mean'].values[0]


def print_summary_table(df: pd.DataFrame):
    """Print comprehensive summary table with 3 decimal places."""
    print("=" * 130)
    print(f"COMPREHENSIVE LEARNING RATE COMPARISON ({len(df)} LRs)")
    print("=" * 130)
    
    # Sort by Mean Uno (descending)
    df_sorted = df.sort_values('Mean', ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    
    # Get baseline
    baseline_mean = get_baseline_mean(df)
    if BASELINE_LR in df['LR'].values:
        print(f"📊 Baseline: {BASELINE_LR} (Mean Uno: {baseline_mean:.3f})")
    else:
        print(f"⚠️  Baseline {BASELINE_LR} not found, using {df.iloc[0]['LR']} as reference (Mean: {baseline_mean:.3f})")
    print()
    
    # Calculate vs baseline
    df_sorted['vs_Baseline'] = df_sorted['Mean'] - baseline_mean
    df_sorted['Improvement_%'] = (df_sorted['vs_Baseline'] / baseline_mean) * 100
    
    # Add status emoji
    def get_status(improvement):
        if improvement > 0.015:
            return "🏆"
        elif improvement > 0.005:
            return "✅"
        elif improvement > -0.005:
            return "📊"
        elif improvement > -0.015:
            return "⚠️"
        else:
            return "❌"
    
    df_sorted['Status'] = df_sorted['vs_Baseline'].apply(get_status)
    
    # Print table header
    print(f"{'Rank':<7} {'LR':<12} {'Mean':<10} {'Std':<10} {'Min':<10} "
          f"{'Max':<10} {'Range':<10} {'vs Base':<10} {'Improv%':<10} {'Status':<8}")
    print("-" * 130)
    
    # Print each row
    for _, row in df_sorted.iterrows():
        rank_str = f"{int(row['Rank'])}"
        if row['Rank'] == 1:
            rank_str += " 🏆"
        elif row['Rank'] == 2:
            rank_str += " 🥈"
        elif row['Rank'] == 3:
            rank_str += " 🥉"
        
        print(f"{rank_str:<7} {row['LR']:<12} {row['Mean']:<10.3f} {row['Std']:<10.3f} "
              f"{row['Min']:<10.3f} {row['Max']:<10.3f} {row['Range']:<10.3f} "
              f"{row['vs_Baseline']:+.3f}    {row['Improvement_%']:+.2f}%     {row['Status']:<8}")
    
    print("=" * 130)
    
    # Summary statistics
    best_row = df_sorted.iloc[0]
    print(f"\n🏆 BEST LEARNING RATE: {best_row['LR']}")
    print(f"   Mean Uno: {best_row['Mean']:.3f}")
    print(f"   Std: {best_row['Std']:.3f}")
    print(f"   Range: [{best_row['Min']:.3f}, {best_row['Max']:.3f}]")
    print(f"   Improvement: {best_row['vs_Baseline']:+.3f} ({best_row['Improvement_%']:+.2f}%)")
    print()
    
    # TOP 10 Summary
    print("\n" + "=" * 130)
    print("🏆 TOP 10 LEARNING RATES")
    print("=" * 130)
    print(f"{'Rank':<7} {'LR':<15} {'Mean Uno':<12} {'Std':<12} {'Min':<10} {'Max':<10} {'vs Baseline':<15}")
    print("-" * 130)
    
    top10 = df_sorted.head(10)
    for _, row in top10.iterrows():
        rank_str = f"{int(row['Rank'])}"
        if row['Rank'] == 1:
            rank_str += " 🥇"
        elif row['Rank'] == 2:
            rank_str += " 🥈"
        elif row['Rank'] == 3:
            rank_str += " 🥉"
        
        print(f"{rank_str:<7} {row['LR']:<15} {row['Mean']:<12.3f} {row['Std']:<12.3f} "
              f"{row['Min']:<10.3f} {row['Max']:<10.3f} {row['vs_Baseline']:+.3f} ({row['Improvement_%']:+.2f}%)")
    
    print("=" * 130)
    print()


def print_fold_details(df: pd.DataFrame):
    """Print detailed fold-by-fold results with 3 decimal places."""
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
    
    # Convert LR to float for categorization (remove suffixes like _2nd, _3rd first)
    def lr_to_float(lr_str):
        # Remove suffixes like _2nd, _3rd
        lr_clean = lr_str.split('_')[0]
        return float(lr_clean)
    
    df['LR_float'] = df['LR'].apply(lr_to_float)
    
    ranges = [
        ("Very Low (<1e-5)", 0, 1e-5),
        ("Low (1e-5 to 3e-5)", 1e-5, 3e-5),
        ("Mid-Low (3e-5 to 5e-5)", 3e-5, 5e-5),
        ("Mid (5e-5 to 7e-5)", 5e-5, 7e-5),
        ("Mid-High (7e-5 to 9e-5)", 7e-5, 9e-5),
        ("High (9e-5 to 2e-4)", 9e-5, 2e-4),
        ("Very High (≥2e-4)", 2e-4, 1.0)
    ]
    
    print(f"{'Range':<30} {'N':<6} {'Mean':<12} {'Std':<12} {'Best':<12}")
    print("-" * 100)
    
    for range_name, lower, upper in ranges:
        mask = (df['LR_float'] >= lower) & (df['LR_float'] < upper)
        subset = df[mask]
        
        if not subset.empty:
            n = len(subset)
            mean = subset['Mean'].mean()
            std = subset['Mean'].std() if len(subset) > 1 else 0.0
            best = subset['Mean'].max()
            best_lr = subset.loc[subset['Mean'].idxmax(), 'LR']
            
            print(f"{range_name:<30} {n:<6} {mean:<12.3f} {std:<12.3f} "
                  f"{best:.3f} ({best_lr})")
    
    print("=" * 100)
    print()


def analyze_trends(df: pd.DataFrame):
    """Analyze trends and patterns with 3 decimal places."""
    print("=" * 100)
    print("TREND ANALYSIS")
    print("=" * 100)
    
    # Find optimal point
    best_idx = df['Mean'].idxmax()
    best_row = df.loc[best_idx]
    
    print(f"🏆 OPTIMAL LEARNING RATE:")
    print(f"   LR: {best_row['LR']}")
    print(f"   Mean Uno: {best_row['Mean']:.3f}")
    print(f"   Std: {best_row['Std']:.3f}")
    print(f"   Range: [{best_row['Min']:.3f}, {best_row['Max']:.3f}]")
    print()
    
    # Top 5 performers
    top5 = df.nlargest(5, 'Mean')
    print("🎯 TOP 5 PERFORMERS:")
    for idx, row in top5.iterrows():
        print(f"   {row['LR']:<12} Mean={row['Mean']:.3f}, Std={row['Std']:.3f}")
    print()
    
    # Most stable (lowest std among top performers)
    top10 = df.nlargest(10, 'Mean')
    most_stable = top10.nsmallest(3, 'Std')
    print("📊 MOST STABLE (among top 10):")
    for idx, row in most_stable.iterrows():
        print(f"   {row['LR']:<12} Mean={row['Mean']:.3f}, Std={row['Std']:.3f}")
    print()
    
    # Fold analysis
    print("⚠️  FOLD ANALYSIS:")
    fold_cols = ['Fold_0', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4']
    fold_means = df[fold_cols].mean()
    fold_stds = df[fold_cols].std()
    
    for i, (mean, std) in enumerate(zip(fold_means, fold_stds)):
        threshold = fold_means.mean() - fold_means.std()
        status = "⚠️" if mean < threshold else "✅"
        print(f"   Fold {i}: Mean={mean:.3f}, Std={std:.3f} {status}")
    
    worst_fold_idx = fold_means.idxmin()
    worst_fold_num = int(worst_fold_idx.split('_')[1])
    print(f"\n   🔍 Fold {worst_fold_num} shows relatively lower performance")
    print(f"      Avg performance: {fold_means[worst_fold_idx]:.3f}")
    print()
    
    # LR curve shape analysis
    print("📈 LEARNING RATE CURVE ANALYSIS:")
    
    # Use the same lr_to_float function to avoid suffix issues
    def lr_to_float(lr_str):
        lr_clean = lr_str.split('_')[0]
        return float(lr_clean)
    
    if 'LR_float' not in df.columns:
        df['LR_float'] = df['LR'].apply(lr_to_float)
    df_sorted = df.sort_values('LR_float')
    
    # Find peak region
    peak_lr = df.loc[df['Mean'].idxmax(), 'LR']
    peak_mean = df['Mean'].max()
    
    print(f"   Peak LR: {peak_lr} (Mean: {peak_mean:.3f})")
    
    # Analyze slope before and after peak
    peak_idx = df_sorted[df_sorted['LR'] == peak_lr].index[0]
    peak_position = df_sorted.index.get_loc(peak_idx)
    
    if peak_position > 0:
        before_peak = df_sorted.iloc[:peak_position]
        if len(before_peak) > 1:
            slope_before = (peak_mean - before_peak['Mean'].iloc[0]) / peak_position
            print(f"   Slope before peak: {slope_before:+.4f} per LR step")
    
    if peak_position < len(df_sorted) - 1:
        after_peak = df_sorted.iloc[peak_position+1:]
        if len(after_peak) > 0:
            slope_after = (after_peak['Mean'].iloc[-1] - peak_mean) / len(after_peak)
            print(f"   Slope after peak: {slope_after:+.4f} per LR step")
    
    print("=" * 100)
    print()


def plot_results(df: pd.DataFrame, save_path: Path = None):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Convert LR to float for plotting (handle suffixes)
    def lr_to_float(lr_str):
        lr_clean = lr_str.split('_')[0]
        return float(lr_clean)
    
    if 'LR_float' not in df.columns:
        df['LR_float'] = df['LR'].apply(lr_to_float)
    df_sorted = df.sort_values('LR_float')
    
    # Plot 1: Full LR curve with error bars (large, main plot)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.errorbar(range(len(df_sorted)), df_sorted['Mean'], 
                 yerr=df_sorted['Std'], marker='o', capsize=5, 
                 linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Learning Rate', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Uno C-index', fontsize=14, fontweight='bold')
    ax1.set_title('Complete Learning Rate Curve (20 LRs)', fontsize=16, fontweight='bold')
    ax1.set_xticks(range(len(df_sorted)))
    ax1.set_xticklabels(df_sorted['LR'], rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Mark best LR
    best_idx = df_sorted['Mean'].idxmax()
    best_lr_idx = df_sorted.index.get_loc(best_idx)
    ax1.plot(best_lr_idx, df_sorted.loc[best_idx, 'Mean'], 
             'r*', markersize=25, label=f"Best: {df_sorted.loc[best_idx, 'LR']}")
    ax1.legend(fontsize=12)
    
    # Plot 2: Box plot
    ax2 = fig.add_subplot(gs[1, 0])
    fold_data = []
    labels = []
    for _, row in df_sorted.iterrows():
        folds = [row['Fold_0'], row['Fold_1'], row['Fold_2'], 
                 row['Fold_3'], row['Fold_4']]
        fold_data.append([f for f in folds if not np.isnan(f)])
        labels.append(row['LR'])
    
    bp = ax2.boxplot(fold_data, patch_artist=True)
    ax2.set_xlabel('Learning Rate', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Uno C-index', fontsize=10, fontweight='bold')
    ax2.set_title('Fold Variability', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(1, len(labels)+1, 2))
    ax2.set_xticklabels([labels[i] for i in range(0, len(labels), 2)], 
                        rotation=45, ha='right', fontsize=7)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    # Plot 3: Stability vs Performance
    ax3 = fig.add_subplot(gs[1, 1])
    scatter = ax3.scatter(df_sorted['Mean'], df_sorted['Std'], 
                         s=100, alpha=0.6, c=range(len(df_sorted)), 
                         cmap='viridis')
    
    # Annotate top 5
    top5_idx = df.nlargest(5, 'Mean').index
    for idx in top5_idx:
        if idx in df_sorted.index:
            row = df_sorted.loc[idx]
            ax3.annotate(row['LR'], (row['Mean'], row['Std']), 
                        fontsize=8, ha='center', fontweight='bold')
    
    ax3.set_xlabel('Mean Uno C-index', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Std', fontsize=10, fontweight='bold')
    ax3.set_title('Performance vs Stability', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap (top 10)
    ax4 = fig.add_subplot(gs[1, 2])
    top10 = df.nlargest(10, 'Mean')
    fold_matrix = top10[['Fold_0', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4']].values
    im = ax4.imshow(fold_matrix, aspect='auto', cmap='YlGnBu', vmin=0.5, vmax=0.65)
    
    ax4.set_xticks(range(5))
    ax4.set_xticklabels([f'F{i}' for i in range(5)])
    ax4.set_yticks(range(len(top10)))
    ax4.set_yticklabels(top10['LR'], fontsize=9)
    ax4.set_title('Top 10 LRs - Fold Heatmap', fontsize=12, fontweight='bold')
    
    plt.colorbar(im, ax=ax4, label='Uno C-index')
    
    # Plot 5: LR range comparison (bar plot)
    ax5 = fig.add_subplot(gs[2, :])
    
    ranges_data = []
    range_names = []
    
    ranges = [
        ("<1e-5", 0, 1e-5),
        ("1-3e-5", 1e-5, 3e-5),
        ("3-5e-5", 3e-5, 5e-5),
        ("5-7e-5", 5e-5, 7e-5),
        ("7-9e-5", 7e-5, 9e-5),
        ("9e-5-2e-4", 9e-5, 2e-4),
        ("≥2e-4", 2e-4, 1.0)
    ]
    
    for name, lower, upper in ranges:
        mask = (df['LR_float'] >= lower) & (df['LR_float'] < upper)
        subset = df[mask]
        if not subset.empty:
            range_names.append(name)
            ranges_data.append(subset['Mean'].mean())
    
    bars = ax5.bar(range(len(ranges_data)), ranges_data, 
                   color='#A23B72', alpha=0.7, edgecolor='black')
    ax5.set_xticks(range(len(range_names)))
    ax5.set_xticklabels(range_names, fontsize=11)
    ax5.set_ylabel('Mean Uno C-index', fontsize=12, fontweight='bold')
    ax5.set_title('Performance by LR Range', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Annotate bars
    for bar, val in zip(bars, ranges_data):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    
    plt.show()


def export_results(df: pd.DataFrame, output_path: Path):
    """Export results to CSV with 3 decimal places."""
    # Format all numeric columns to 3 decimals
    numeric_cols = ['Mean', 'Std', 'Min', 'Max', 'Range', 
                    'Fold_0', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4']
    
    df_export = df.copy()
    for col in numeric_cols:
        if col in df_export.columns:
            df_export[col] = df_export[col].apply(lambda x: f"{x:.3f}" if not np.isnan(x) else "N/A")
    
    df_export.to_csv(output_path, index=False)
    print(f"💾 Results exported to: {output_path}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE LEARNING RATE EXPERIMENT ANALYSIS")
    print(f"Total LRs available: {len(LEARNING_RATES)}")
    if LEARNING_RATES:
        print(f"Experiments: {', '.join(LEARNING_RATES[:5])} ... {', '.join(LEARNING_RATES[-3:])}")
    print("=" * 100 + "\n")
    
    # Collect results
    df = collect_all_results()
    
    if df.empty:
        print("❌ No results found! Check experiment directories.")
        return
    
    print(f"✅ Successfully loaded results for {len(df)}/{len(LEARNING_RATES)} learning rates")
    print()
    
    # Print summary table
    print_summary_table(df)
    
    # Print fold details
    print_fold_details(df)
    
    # Analyze LR ranges
    analyze_lr_ranges(df)
    
    # Analyze trends
    analyze_trends(df)
    
    # Create plots
    print("📊 Generating comprehensive visualizations...")
    plot_path = BASE_DIR / "lr_complete_analysis.png"
    plot_results(df, save_path=plot_path)
    
    # Export results
    csv_path = BASE_DIR / "lr_complete_results.csv"
    export_results(df, csv_path)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE!")
    print("=" * 100)
    
    # Final recommendations
    best_row = df.loc[df['Mean'].idxmax()]
    baseline_mean = get_baseline_mean(df)
    improvement = ((best_row['Mean'] - baseline_mean) / baseline_mean) * 100
    
    print(f"\n🎯 FINAL RECOMMENDATIONS:")
    print(f"   Best LR: {best_row['LR']}")
    print(f"   Expected C-index: {best_row['Mean']:.3f} ± {best_row['Std']:.3f}")
    print(f"   Improvement: {improvement:+.2f}%")
    print()
    
    # Suggest next steps
    print("📋 SUGGESTED NEXT STEPS:")
    print("   1. Verify best LR with additional runs")
    print("   2. Implement SWA (Stochastic Weight Averaging)")
    print("   3. Add 5-fold ensemble")
    print("   4. Enable TTA for final testing")
    print("   5. Expected final C-index: 0.630-0.650")
    print()


if __name__ == "__main__":
    main()