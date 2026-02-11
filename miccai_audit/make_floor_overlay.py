#!/usr/bin/env python3
"""Create floor figure with real model points overlaid"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("CREATING DISCRIMINATION FLOOR WITH REAL MODEL OVERLAY")
print("="*80)

# Load floor simulation results
floor = pd.read_csv("exp5_v2_CLEAN_123_alpha0p10/discrimination_floor_results.csv")
cw = floor[floor["method"] == "Classwise CP"].copy()

print(f"\nFloor simulation: {len(cw)} AUC points")
print(f"  AUC range: {cw['target_auc'].min():.2f} to {cw['target_auc'].max():.2f}")

# Real model points from exp7c summary (alpha=0.10, Classwise, CalOnly_Corrected)
# We'll read from exp7c_v2 once it's done, for now use approximate values
# These will be updated once exp7c_v2 finishes

# Placeholder: will update after exp7c_v2 completes
rad_auc, rad_defer = 0.618, 0.824  # From previous exp7c
cli_auc, cli_defer = 0.689, 0.763  # From previous exp7c

print(f"\nReal model points (from exp7c, alpha=0.10, Classwise):")
print(f"  Radiomics: AUC={rad_auc:.3f}, defer={rad_defer:.3f}")
print(f"  Clinical:  AUC={cli_auc:.3f}, defer={cli_defer:.3f}")

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot simulation curve
ax.plot(cw["target_auc"], cw["defer_rate_mean"], 
        marker="o", linewidth=2, markersize=6,
        color='#2C3E50', label="Simulated (Classwise CP)", zorder=2)

# Confidence band
ax.fill_between(
    cw["target_auc"],
    cw["defer_rate_mean"] - cw["defer_rate_std"],
    cw["defer_rate_mean"] + cw["defer_rate_std"],
    alpha=0.2, color='#2C3E50', zorder=1
)

# Real model points
ax.scatter([rad_auc], [rad_defer], s=200, marker='D', 
          color='#E74C3C', edgecolors='black', linewidths=2,
          label="Radiomics (real)", zorder=5)

ax.scatter([cli_auc], [cli_defer], s=200, marker='s', 
          color='#3498DB', edgecolors='black', linewidths=2,
          label="Clinical (real)", zorder=5)

# Annotate real points
ax.annotate(f'Radiomics\nAUC={rad_auc:.3f}\nDefer={rad_defer:.1%}',
           xy=(rad_auc, rad_defer), xytext=(rad_auc-0.05, rad_defer+0.08),
           fontsize=9, ha='right',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#E74C3C', alpha=0.3),
           arrowprops=dict(arrowstyle='->', lw=1.5))

ax.annotate(f'Clinical\nAUC={cli_auc:.3f}\nDefer={cli_defer:.1%}',
           xy=(cli_auc, cli_defer), xytext=(cli_auc+0.05, cli_defer-0.08),
           fontsize=9, ha='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#3498DB', alpha=0.3),
           arrowprops=dict(arrowstyle='->', lw=1.5))

# Threshold line
ax.axhline(0.30, color='orange', linestyle='--', linewidth=2, 
          alpha=0.7, label='30% deferral threshold', zorder=3)

# Shaded "infeasible" region
ax.fill_between([0.55, 1.0], [0, 0], [0.30, 0.30], 
               alpha=0.1, color='green', label='Feasible region (<30% defer)')

ax.set_xlabel("Base Model AUC", fontsize=13, fontweight='bold')
ax.set_ylabel("Deferral Rate (to achieve 90% event coverage)", fontsize=13, fontweight='bold')
ax.set_title("Discrimination Floor for Classwise Conformal Triage (α=0.10)\n" +
            "Real models fall in high-deferral region", 
            fontsize=14, fontweight='bold')

ax.set_xlim([0.55, 0.97])
ax.set_ylim([0, 1.0])
ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
ax.grid(alpha=0.3, linestyle=':', linewidth=0.8)

plt.tight_layout()
plt.savefig("paper_figs/fig_floor_overlay.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: paper_figs/fig_floor_overlay.png")
plt.close()

# Also save data for supplementary
overlay_data = pd.DataFrame({
    'model': ['Radiomics', 'Clinical'],
    'auc': [rad_auc, cli_auc],
    'defer': [rad_defer, cli_defer],
    'method': ['Classwise CP', 'Classwise CP'],
    'alpha': [0.10, 0.10]
})
overlay_data.to_csv("paper_tables/real_model_floor_points.csv", index=False)
print(f"✓ Saved: paper_tables/real_model_floor_points.csv")

