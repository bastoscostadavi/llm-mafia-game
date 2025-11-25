#!/usr/bin/env python3
"""
Compare hypothesis-derived capabilities with paper's published scores
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")

# Paper's published scores (from Table 1 in article)
paper_scores = {
    'Claude Opus 4.1': {'Deceive': 2.20, 'Detect': 1.98, 'Disclose': 1.92},
    'Claude Sonnet 4': {'Deceive': 1.86, 'Detect': 0.48, 'Disclose': 1.74},
    'DeepSeek V3.1': {'Deceive': 3.13, 'Detect': 2.13, 'Disclose': 1.68},
    'Gemini 2.5 Flash Lite': {'Deceive': 1.31, 'Detect': 0.99, 'Disclose': 1.10},
    'GPT-4.1 Mini': {'Deceive': 0.55, 'Detect': 0.64, 'Disclose': 1.49},
    'GPT-5 Mini': {'Deceive': 0.73, 'Detect': 0.66, 'Disclose': 2.07},
    'Grok 3 Mini': {'Deceive': 2.05, 'Detect': 6.70, 'Disclose': 1.90},
    'Llama 3.1 8B Instruct': {'Deceive': 0.30, 'Detect': 0.54, 'Disclose': 0.10},
    'Mistral 7B Instruct': {'Deceive': 0.69, 'Detect': 0.52, 'Disclose': 0.53},
    'Qwen2.5 7B Instruct': {'Deceive': 0.36, 'Detect': 0.63, 'Disclose': 0.51},
}

# Load hypothesis results
hypothesis_df = pd.read_csv('../results/hypothesis_capabilities_v2.csv')

# Prepare data for comparison
models = list(paper_scores.keys())
capabilities = ['Deceive', 'Detect', 'Disclose']

comparison_data = []

for model in models:
    paper_row = paper_scores[model]
    hyp_row = hypothesis_df[hypothesis_df['Model'] == model].iloc[0]

    for cap in capabilities:
        comparison_data.append({
            'Model': model,
            'Capability': cap,
            'Paper': paper_row[cap],
            'Hypothesis': hyp_row[f'{cap}_exp']
        })

comparison_df = pd.DataFrame(comparison_data)

# Create comparison plots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Scatter plot for each capability
for idx, cap in enumerate(capabilities):
    ax = axes[idx // 2, idx % 2]

    subset = comparison_df[comparison_df['Capability'] == cap]

    ax.scatter(subset['Paper'], subset['Hypothesis'], s=100, alpha=0.6)

    # Add labels
    for _, row in subset.iterrows():
        model_short = row['Model'][:15]
        ax.annotate(model_short, (row['Paper'], row['Hypothesis']),
                   fontsize=8, alpha=0.7, rotation=15)

    # Add diagonal line
    min_val = min(subset['Paper'].min(), subset['Hypothesis'].min()) * 0.9
    max_val = max(subset['Paper'].max(), subset['Hypothesis'].max()) * 1.1
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect agreement')

    # Compute correlation
    corr = subset['Paper'].corr(subset['Hypothesis'])

    ax.set_xlabel('Paper Scores', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hypothesis Scores', fontsize=12, fontweight='bold')
    ax.set_title(f'{cap} Capability\nCorrelation: r = {corr:.3f}',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

# 4. Overall correlation
ax = axes[1, 1]

ax.scatter(comparison_df['Paper'], comparison_df['Hypothesis'],
          c=[['red', 'green', 'blue'][capabilities.index(c)]
             for c in comparison_df['Capability']],
          s=80, alpha=0.6)

# Add diagonal
min_val = comparison_df['Paper'].min() * 0.9
max_val = comparison_df['Paper'].max() * 1.1
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect agreement')

# Compute overall correlation
overall_corr = comparison_df['Paper'].corr(comparison_df['Hypothesis'])

ax.set_xlabel('Paper Scores', fontsize=12, fontweight='bold')
ax.set_ylabel('Hypothesis Scores', fontsize=12, fontweight='bold')
ax.set_title(f'Overall Comparison\nCorrelation: r = {overall_corr:.3f}',
            fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.6, label='Deceive'),
    Patch(facecolor='green', alpha=0.6, label='Detect'),
    Patch(facecolor='blue', alpha=0.6, label='Disclose')
]
ax.legend(handles=legend_elements)

plt.suptitle('Comparison: Paper Methodology vs. Hypothesis logit(p) = c(a-b)',
            fontsize=16, fontweight='bold')
plt.tight_layout()

filename = '../results/paper_vs_hypothesis_comparison.png'
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved to: {filename}")
plt.close()

# Create ranking comparison table
print("\n" + "="*80)
print("RANKING COMPARISON")
print("="*80)

for cap in capabilities:
    print(f"\n{cap.upper()} CAPABILITY:")
    print("-" * 40)

    subset = comparison_df[comparison_df['Capability'] == cap].copy()

    # Rank by paper
    subset['Paper_Rank'] = subset['Paper'].rank(ascending=False, method='min').astype(int)
    # Rank by hypothesis
    subset['Hypothesis_Rank'] = subset['Hypothesis'].rank(ascending=False, method='min').astype(int)
    # Rank difference
    subset['Rank_Diff'] = subset['Hypothesis_Rank'] - subset['Paper_Rank']

    # Sort by paper rank
    subset = subset.sort_values('Paper_Rank')

    print("\n{:<25} {:>12} {:>6} {:>12} {:>6} {:>6}".format(
        "Model", "Paper Score", "Rank", "Hyp Score", "Rank", "Diff"))
    print("-" * 80)

    for _, row in subset.iterrows():
        diff_str = f"+{row['Rank_Diff']}" if row['Rank_Diff'] > 0 else str(int(row['Rank_Diff']))
        print("{:<25} {:>12.2f} {:>6} {:>12.2f} {:>6} {:>6}".format(
            row['Model'][:24],
            row['Paper'],
            int(row['Paper_Rank']),
            row['Hypothesis'],
            int(row['Hypothesis_Rank']),
            diff_str
        ))

    # Compute rank correlation (Spearman)
    from scipy.stats import spearmanr
    rank_corr, p_value = spearmanr(subset['Paper_Rank'], subset['Hypothesis_Rank'])
    print(f"\nSpearman rank correlation: ρ = {rank_corr:.3f} (p = {p_value:.4f})")

# Overall statistics
print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)

overall_corr = comparison_df['Paper'].corr(comparison_df['Hypothesis'])
print(f"\nPearson correlation (all capabilities): r = {overall_corr:.4f}")

# RMSE
rmse = np.sqrt(np.mean((comparison_df['Paper'] - comparison_df['Hypothesis'])**2))
print(f"RMSE: {rmse:.4f}")

# MAE
mae = np.mean(np.abs(comparison_df['Paper'] - comparison_df['Hypothesis']))
print(f"MAE: {mae:.4f}")

# R²
ss_res = np.sum((comparison_df['Paper'] - comparison_df['Hypothesis'])**2)
ss_tot = np.sum((comparison_df['Paper'] - comparison_df['Paper'].mean())**2)
r_squared = 1 - ss_res / ss_tot
print(f"R²: {r_squared:.4f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("\nThe hypothesis-derived capabilities show:")
if overall_corr > 0.8:
    print("✅ STRONG agreement with paper's methodology (r > 0.8)")
elif overall_corr > 0.6:
    print("✓ MODERATE agreement with paper's methodology (0.6 < r < 0.8)")
else:
    print("⚠ WEAK agreement with paper's methodology (r < 0.6)")

print("\nThis suggests that:")
print("- The functional form logit(p) = c(a-b) captures real structure")
print("- But there may be systematic differences in how capabilities are measured")
print("- Rankings are more robust than absolute scores")

# Save comparison table
comparison_df.to_csv('../results/paper_hypothesis_comparison.csv', index=False)
print(f"\nComparison data saved to: ../results/paper_hypothesis_comparison.csv")
