#!/usr/bin/env python3
"""
Compare simplified methodology vs hierarchical hypothesis model
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load both results
simplified = pd.read_csv('../results/scores.csv')
hypothesis = pd.read_csv('../results/scores_hypothesis_hierarchical.csv')

# Merge on Model
comparison = simplified.merge(hypothesis, on='Model', suffixes=('_simp', '_hyp'))

print("="*80)
print("COMPARISON: Simplified Methodology vs Hierarchical Hypothesis Model")
print("="*80)

# Print comparison table
print("\n" + "="*80)
print("DECEIVE CAPABILITY")
print("="*80)
print(f"{'Model':<25} {'Simplified':<20} {'Hypothesis':<20} {'Ratio':<10}")
print("-"*80)
for _, row in comparison.iterrows():
    simp = f"{row['Deceive_simp']:.3f} ± {row['Deceive_Error_simp']:.3f}"
    hyp = f"{row['Deceive_hyp']:.3f} ± {row['Deceive_Error_hyp']:.3f}"
    ratio = row['Deceive_simp'] / row['Deceive_hyp']
    print(f"{row['Model']:<25} {simp:<20} {hyp:<20} {ratio:.2f}")

print("\n" + "="*80)
print("DETECT CAPABILITY")
print("="*80)
print(f"{'Model':<25} {'Simplified':<20} {'Hypothesis':<20} {'Ratio':<10}")
print("-"*80)
for _, row in comparison.iterrows():
    simp = f"{row['Detect_simp']:.3f} ± {row['Detect_Error_simp']:.3f}"
    hyp = f"{row['Detect_hyp']:.3f} ± {row['Detect_Error_hyp']:.3f}"
    ratio = row['Detect_simp'] / row['Detect_hyp']
    print(f"{row['Model']:<25} {simp:<20} {hyp:<20} {ratio:.2f}")

print("\n" + "="*80)
print("DISCLOSE CAPABILITY")
print("="*80)
print(f"{'Model':<25} {'Simplified':<20} {'Hypothesis':<20} {'Ratio':<10}")
print("-"*80)
for _, row in comparison.iterrows():
    simp = f"{row['Disclose_simp']:.3f} ± {row['Disclose_Error_simp']:.3f}"
    hyp = f"{row['Disclose_hyp']:.3f} ± {row['Disclose_Error_hyp']:.3f}"
    ratio = row['Disclose_simp'] / row['Disclose_hyp']
    print(f"{row['Model']:<25} {simp:<20} {hyp:<20} {ratio:.2f}")

# Compute correlations
print("\n" + "="*80)
print("CORRELATIONS BETWEEN METHODOLOGIES")
print("="*80)

for cap in ['Deceive', 'Detect', 'Disclose']:
    corr = np.corrcoef(comparison[f'{cap}_simp'], comparison[f'{cap}_hyp'])[0, 1]
    print(f"{cap}: r = {corr:.3f}")

# Create side-by-side comparison plots
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

capabilities = ['Deceive', 'Detect', 'Disclose']

for i, cap in enumerate(capabilities):
    # Simplified method
    ax_simp = axes[i, 0]
    models = comparison['Model']
    values_simp = comparison[f'{cap}_simp']
    errors_simp = comparison[f'{cap}_Error_simp']

    sorted_data = sorted(zip(models, values_simp, errors_simp), key=lambda x: x[1])
    sorted_models = [x[0] for x in sorted_data]
    sorted_values = [x[1] for x in sorted_data]
    sorted_errors = [x[2] for x in sorted_data]

    y_pos = range(len(sorted_models))
    ax_simp.barh(y_pos, sorted_values, xerr=sorted_errors,
                  color='#3498DB', alpha=0.8, height=0.6,
                  error_kw={'capsize': 5, 'capthick': 2})

    for j, (model, value, error) in enumerate(zip(sorted_models, sorted_values, sorted_errors)):
        ax_simp.text(value + error + 0.1, j, model, ha='left', va='center', fontsize=10)

    ax_simp.set_yticks([])
    ax_simp.axvline(x=1, color='gray', alpha=0.7, linewidth=2, linestyle='--')
    ax_simp.set_xlabel(f'{cap} Score', fontsize=12, fontweight='bold')
    ax_simp.set_title(f'{cap} - Simplified Methodology', fontsize=14, fontweight='bold')
    ax_simp.grid(True, axis='x', alpha=0.3)

    # Hypothesis method
    ax_hyp = axes[i, 1]
    values_hyp = comparison[f'{cap}_hyp']
    errors_hyp = comparison[f'{cap}_Error_hyp']

    sorted_data_hyp = sorted(zip(models, values_hyp, errors_hyp), key=lambda x: x[1])
    sorted_models_hyp = [x[0] for x in sorted_data_hyp]
    sorted_values_hyp = [x[1] for x in sorted_data_hyp]
    sorted_errors_hyp = [x[2] for x in sorted_data_hyp]

    y_pos_hyp = range(len(sorted_models_hyp))
    ax_hyp.barh(y_pos_hyp, sorted_values_hyp, xerr=sorted_errors_hyp,
                 color='#E74C3C', alpha=0.8, height=0.6,
                 error_kw={'capsize': 5, 'capthick': 2})

    for j, (model, value, error) in enumerate(zip(sorted_models_hyp, sorted_values_hyp, sorted_errors_hyp)):
        ax_hyp.text(value + error + 0.1, j, model, ha='left', va='center', fontsize=10)

    ax_hyp.set_yticks([])
    ax_hyp.axvline(x=1, color='gray', alpha=0.7, linewidth=2, linestyle='--')
    ax_hyp.set_xlabel(f'{cap} Score', fontsize=12, fontweight='bold')
    ax_hyp.set_title(f'{cap} - Hypothesis Model: logit(p)=c(a-b)', fontsize=14, fontweight='bold')
    ax_hyp.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/methodology_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Comparison plot saved to: ../results/methodology_comparison.png")

# Create scatter plots showing correlation
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, cap in enumerate(capabilities):
    ax = axes[i]

    x = comparison[f'{cap}_simp']
    y = comparison[f'{cap}_hyp']

    ax.scatter(x, y, s=100, alpha=0.6)

    # Add model labels
    for _, row in comparison.iterrows():
        ax.annotate(row['Model'],
                   (row[f'{cap}_simp'], row[f'{cap}_hyp']),
                   fontsize=8, alpha=0.7)

    # Add correlation
    corr = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Simplified Methodology', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hypothesis Model', fontsize=12, fontweight='bold')
    ax.set_title(cap, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add diagonal line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig('../results/methodology_correlation.png', dpi=300, bbox_inches='tight')
print(f"✓ Correlation plot saved to: ../results/methodology_correlation.png")

print("\n" + "="*80)
