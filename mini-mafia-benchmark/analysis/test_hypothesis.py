#!/usr/bin/env python3
"""
Test the hypothesis: logit(p) = c(a - b)

where:
- a = deceive capability (z-score)
- b = disclose capability (z-score)
- c = detect capability (z-score)
- p = mafioso win rate
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'results'))
from utils import bayesian_win_rate

def logit(p):
    """Convert probability to logit scale"""
    # Clip to avoid log(0) or log(1)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))

def inv_logit(x):
    """Convert logit to probability"""
    return 1 / (1 + np.exp(-x))

def load_data():
    """Load scores and win counts data"""
    # Load aggregated scores (α values)
    scores_df = pd.read_csv('../results/scores.csv')

    # Convert to z-scores: z = ln(α)
    scores_df['deceive_z'] = np.log(scores_df['Deceive'])
    scores_df['detect_z'] = np.log(scores_df['Detect'])
    scores_df['disclose_z'] = np.log(scores_df['Disclose'])

    # Load raw win counts
    win_counts_df = pd.read_csv('../results/win_counts.csv')

    return scores_df, win_counts_df

def prepare_data_for_capability(win_counts_df, scores_df, capability):
    """
    Prepare data for testing hypothesis for a given capability.

    Args:
        win_counts_df: Raw win counts dataframe
        scores_df: Scores with z-scores
        capability: 'Deceive', 'Detect', or 'Disclose'

    Returns:
        DataFrame with observed and predicted values
    """
    # Filter for this capability
    cap_data = win_counts_df[win_counts_df['capability'] == capability].copy()

    # Get background columns (all columns except 'capability' and 'model')
    background_cols = [col for col in cap_data.columns if col not in ['capability', 'model']]

    # Create model name to z-score mapping
    model_to_z = {
        row['Model']: {
            'deceive': row['deceive_z'],
            'detect': row['detect_z'],
            'disclose': row['disclose_z']
        }
        for _, row in scores_df.iterrows()
    }

    results = []

    for _, row in cap_data.iterrows():
        target_model = row['model']

        # Get target model's capability z-score
        if capability == 'Deceive':
            a = model_to_z[target_model]['deceive']
        elif capability == 'Detect':
            c = model_to_z[target_model]['detect']
        elif capability == 'Disclose':
            b = model_to_z[target_model]['disclose']

        # For each background
        for background in background_cols:
            wins = row[background]
            n_games = 100

            # Calculate observed win rate using Bayesian estimate
            p_obs, _ = bayesian_win_rate(wins, n_games)
            p_obs = p_obs / 100  # Convert from percentage

            # IMPORTANT: For Detect and Disclose, CSV counts TOWN wins
            # but our formula predicts MAFIA win rate, so we need to flip
            if capability in ['Detect', 'Disclose']:
                p_obs = 1 - p_obs  # Convert town win rate to mafia win rate

            logit_obs = logit(p_obs)

            # Get background model's z-scores
            # Backgrounds use the same model for both non-target roles
            background_model = background

            if background_model not in model_to_z:
                # Try to map background name to model name
                # Background names might be slightly different
                continue

            bg_z = model_to_z[background_model]

            # Calculate predicted logit(p) based on capability being tested
            if capability == 'Deceive':
                # Testing mafioso (a), background has detective (b) and villager (c)
                b = bg_z['disclose']
                c = bg_z['detect']
                logit_pred = c * (a - b)
            elif capability == 'Detect':
                # Testing villager (c), background has mafioso (a) and detective (b)
                a = bg_z['deceive']
                b = bg_z['disclose']
                logit_pred = c * (a - b)
            elif capability == 'Disclose':
                # Testing detective (b), background has mafioso (a) and villager (c)
                a = bg_z['deceive']
                c = bg_z['detect']
                logit_pred = c * (a - b)

            results.append({
                'capability': capability,
                'target_model': target_model,
                'background': background,
                'wins': wins,
                'n_games': n_games,
                'p_obs': p_obs,
                'logit_obs': logit_obs,
                'logit_pred': logit_pred,
                'p_pred': inv_logit(logit_pred)
            })

    return pd.DataFrame(results)

def compute_statistics(df):
    """Compute R², correlation, RMSE, etc."""
    # Remove any NaN or inf values
    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()

    logit_obs = df_clean['logit_obs'].values
    logit_pred = df_clean['logit_pred'].values

    # Pearson correlation
    corr, p_value = stats.pearsonr(logit_obs, logit_pred)

    # R²
    ss_res = np.sum((logit_obs - logit_pred) ** 2)
    ss_tot = np.sum((logit_obs - np.mean(logit_obs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # RMSE
    rmse = np.sqrt(np.mean((logit_obs - logit_pred) ** 2))

    # Mean Absolute Error
    mae = np.mean(np.abs(logit_obs - logit_pred))

    return {
        'correlation': corr,
        'p_value': p_value,
        'r_squared': r_squared,
        'rmse': rmse,
        'mae': mae,
        'n_points': len(df_clean)
    }

def plot_predictions(df, capability, stats_dict):
    """Create scatter plot of predicted vs observed"""
    plt.figure(figsize=(10, 10))

    # Plot identity line
    min_val = min(df['logit_obs'].min(), df['logit_pred'].min())
    max_val = max(df['logit_obs'].max(), df['logit_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Perfect prediction')

    # Scatter plot
    plt.scatter(df['logit_pred'], df['logit_obs'], alpha=0.6, s=50)

    plt.xlabel('Predicted logit(p)', fontsize=14, fontweight='bold')
    plt.ylabel('Observed logit(p)', fontsize=14, fontweight='bold')
    plt.title(f'{capability} Capability\nHypothesis: logit(p) = c(a-b)', fontsize=16, fontweight='bold')

    # Add statistics text
    stats_text = f"R² = {stats_dict['r_squared']:.3f}\n"
    stats_text += f"r = {stats_dict['correlation']:.3f}\n"
    stats_text += f"p < {stats_dict['p_value']:.1e}\n"
    stats_text += f"RMSE = {stats_dict['rmse']:.3f}\n"
    stats_text += f"n = {stats_dict['n_points']}"

    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filename = f'../results/hypothesis_test_{capability.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    return filename

def main():
    print("Testing hypothesis: logit(p) = c(a - b)")
    print("=" * 60)

    # Load data
    scores_df, win_counts_df = load_data()

    print("\nZ-scores (from aggregated scores):")
    print(scores_df[['Model', 'deceive_z', 'detect_z', 'disclose_z']].to_string())

    # Test for each capability
    all_results = {}
    all_stats = {}

    for capability in ['Deceive', 'Detect', 'Disclose']:
        print(f"\n\n{'='*60}")
        print(f"Testing {capability.upper()} capability")
        print('='*60)

        # Prepare data
        df = prepare_data_for_capability(win_counts_df, scores_df, capability)

        if len(df) == 0:
            print(f"No data for {capability}")
            continue

        print(f"\nPrepared {len(df)} data points")

        # Compute statistics
        stats_dict = compute_statistics(df)

        print(f"\nStatistics:")
        print(f"  R² = {stats_dict['r_squared']:.4f}")
        print(f"  Correlation = {stats_dict['correlation']:.4f}")
        print(f"  P-value = {stats_dict['p_value']:.4e}")
        print(f"  RMSE = {stats_dict['rmse']:.4f}")
        print(f"  MAE = {stats_dict['mae']:.4f}")
        print(f"  N = {stats_dict['n_points']}")

        # Plot
        plot_file = plot_predictions(df, capability, stats_dict)
        print(f"\nPlot saved to: {plot_file}")

        all_results[capability] = df
        all_stats[capability] = stats_dict

    # Combined plot
    print(f"\n\n{'='*60}")
    print("COMBINED ANALYSIS")
    print('='*60)

    # Combine all data
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    combined_stats = compute_statistics(combined_df)

    print(f"\nOverall Statistics (all capabilities combined):")
    print(f"  R² = {combined_stats['r_squared']:.4f}")
    print(f"  Correlation = {combined_stats['correlation']:.4f}")
    print(f"  P-value = {combined_stats['p_value']:.4e}")
    print(f"  RMSE = {combined_stats['rmse']:.4f}")
    print(f"  MAE = {combined_stats['mae']:.4f}")
    print(f"  N = {combined_stats['n_points']}")

    # Create combined plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, (capability, df) in enumerate(all_results.items()):
        ax = axes[idx]
        stats_dict = all_stats[capability]

        # Plot identity line
        min_val = min(df['logit_obs'].min(), df['logit_pred'].min())
        max_val = max(df['logit_obs'].max(), df['logit_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

        # Scatter plot
        ax.scatter(df['logit_pred'], df['logit_obs'], alpha=0.6, s=50)

        ax.set_xlabel('Predicted logit(p)', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Observed logit(p)', fontsize=12, fontweight='bold')
        ax.set_title(f'{capability}', fontsize=14, fontweight='bold')

        # Add statistics text
        stats_text = f"R² = {stats_dict['r_squared']:.3f}\n"
        stats_text += f"r = {stats_dict['correlation']:.3f}"

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.grid(True, alpha=0.3)

    plt.suptitle('Hypothesis Test: logit(p) = c(a-b)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    combined_file = '../results/hypothesis_test_combined.png'
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nCombined plot saved to: {combined_file}")

    # Save results to CSV
    combined_df.to_csv('../results/hypothesis_test_results.csv', index=False)
    print(f"\nDetailed results saved to: ../results/hypothesis_test_results.csv")

    return all_results, all_stats, combined_stats

if __name__ == '__main__':
    main()
