#!/usr/bin/env python3
"""
Cross-validation scatter plot: predicted win probability vs observed win rate
for all 140 held-out configurations across 5 folds.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "top_down_theoretical"
df = pd.read_csv(RESULTS_DIR / "cv_predictions.csv")

# Binomial uncertainty on observed rate (n=100 games per config)
df['obs_err'] = np.sqrt(df['observed_rate'] * (1 - df['observed_rate']) / df['total'])

r, _ = pearsonr(df['predicted_p'], df['observed_rate'])

# ── Figure ─────────────────────────────────────────────────────────────────
plt.ioff()
plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

fold_colors = {1: '#2E86AB', 2: '#E63946', 3: '#2A9D8F', 4: '#E9C46A', 5: '#9B5DE5'}

fig, ax = plt.subplots(figsize=(6, 6))

# Perfect-prediction diagonal
ax.plot([0, 1], [0, 1], color='#999999', linewidth=1.2, linestyle='--', zorder=0,
        label='Perfect prediction')

# Scatter, one fold at a time for the legend
for fold_id, color in fold_colors.items():
    sub = df[df['fold'] == fold_id]
    ax.errorbar(
        sub['predicted_p'], sub['observed_rate'],
        yerr=sub['obs_err'],
        fmt='o', color=color, alpha=0.75,
        markersize=5, capsize=2, capthick=0.8, elinewidth=0.8,
        label=f'Fold {fold_id}',
        zorder=3,
    )

# Pearson r annotation
ax.text(0.04, 0.93, f'$r = {r:.3f}$', transform=ax.transAxes,
        fontsize=13, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc', alpha=0.85))

ax.set_xlabel('Predicted win probability (p̂)')
ax.set_ylabel('Observed win rate')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=10, loc='lower right', framealpha=0.85)
ax.set_aspect('equal')
ax.grid(True, alpha=0.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
out = RESULTS_DIR / "cv_scatter.png"
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out}")
print(f"Pearson r = {r:.3f}  (n=140 held-out configs)")
