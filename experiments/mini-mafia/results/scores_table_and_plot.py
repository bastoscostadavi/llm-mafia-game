#!/usr/bin/env python3
"""
Script 3: Create Aggregated Score Plots

Reads the win rates CSV and creates:
1. A results table CSV with exponential aggregated scores and uncertainties (models × capabilities)
2. Exponential aggregated score plots using the formula from the paper:

1. Calculate z-scores: zᵢᵦ = (p̄ᵢᵦ - μᵦ) / σᵦ
2. Average z-scores: z̄ᵢ = (1/B) Σᵦ zᵢᵦ  
3. Exponential scores: αᵢ = e^z̄ᵢ

Where μᵦ and σᵦ are the mean and std of win rates for background b.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path


def load_win_rates_data():
    """Load win rates and uncertainties from the multi-index CSV file"""
    df = pd.read_csv('win_rates.csv')
    
    # Group by capability and separate win rates from uncertainties
    capabilities_data = {}
    for capability in df['Capability'].unique():
        cap_data = df[df['Capability'] == capability].set_index('Model')
        cap_data = cap_data.drop(columns=['Capability'])  # Remove capability column
        
        # Separate win rates and uncertainties
        background_cols = [col for col in cap_data.columns if not col.endswith('_Error')]
        error_cols = [col for col in cap_data.columns if col.endswith('_Error')]
        
        win_rates = cap_data[background_cols]
        uncertainties = cap_data[error_cols]
        
        # Rename uncertainty columns to match background names
        uncertainties.columns = [col.replace('_Error', '') for col in uncertainties.columns]
        
        capabilities_data[capability] = {
            'win_rates': win_rates,
            'uncertainties': uncertainties
        }
    
    return capabilities_data

def calculate_aggregated_scores(win_rates_df, uncertainties_df, capability_name):
    """
    Calculate aggregated exponential scores with proper uncertainty propagation.
    
    Algorithm:
    1. For each background, calculate mean and std across models  
    2. Calculate z-scores for each model in each background
    3. Propagate uncertainties through z-score calculation
    4. Average z-scores across backgrounds for each model
    5. Compute exponential scores: αᵢ = e^z̄ᵢ
    6. Propagate uncertainties to exponential scores
    """
    
    print(f"  Win rates shape: {win_rates_df.shape}")
    print(f"  Sample win rates: {win_rates_df.iloc[0].to_dict()}")
    
    # Calculate z-scores and z-score uncertainties for each model in each background
    z_scores_df = pd.DataFrame(index=win_rates_df.index, columns=win_rates_df.columns)
    z_errors_df = pd.DataFrame(index=win_rates_df.index, columns=win_rates_df.columns)
    
    for background in win_rates_df.columns:
        # Get win rates and uncertainties for this background across all models
        background_win_rates = win_rates_df[background].values
        background_uncertainties = uncertainties_df[background].values
        
        # Calculate background mean and standard deviation
        bg_mean = np.mean(background_win_rates)
        bg_std = np.std(background_win_rates, ddof=1)  # Sample standard deviation
        
        print(f"  Background {background}: mean={bg_mean:.1f}%, std={bg_std:.1f}%")
        
        # Calculate z-scores for each model in this background
        if bg_std > 0:
            z_scores = (background_win_rates - bg_mean) / bg_std
            z_scores_df[background] = z_scores
            
            # Propagate uncertainties to z-scores: δz = δp / σ_background
            z_errors = background_uncertainties / bg_std
            z_errors_df[background] = z_errors
        else:
            # If std is 0, all models have same performance
            z_scores_df[background] = 0.0
            z_errors_df[background] = 0.0
    
    # Calculate average z-score for each model across backgrounds
    avg_z_scores = z_scores_df.mean(axis=1)
    
    # Calculate uncertainty in average z-scores (quadrature sum divided by number of backgrounds)
    z_variance_sum = (z_errors_df ** 2).sum(axis=1)  # Sum of variances
    avg_z_errors = np.sqrt(z_variance_sum) / len(z_scores_df.columns)  # Standard error of mean
    
    # Calculate exponential scores: αᵢ = e^z̄ᵢ
    exp_scores = np.exp(avg_z_scores)
    
    # Propagate uncertainties to exponential scores: δα = α * δz̄
    exp_errors = exp_scores * avg_z_errors
    
    print(f"  Exponential scores range: {exp_scores.min():.3f} to {exp_scores.max():.3f}")
    print(f"  Average z-score errors range: {avg_z_errors.min():.3f} to {avg_z_errors.max():.3f}")
    
    return exp_scores, exp_errors

def create_exponential_score_plot(exp_scores, exp_errors, capability_name):
    """Create exponential score plot for a specific capability"""
    
    # Sort models by score (ascending for consistent ordering)
    sorted_data = sorted(zip(exp_scores.index, exp_scores.values, exp_errors.values), 
                        key=lambda x: x[1])
    
    models = [x[0] for x in sorted_data]
    scores = [x[1] for x in sorted_data] 
    errors = [x[2] for x in sorted_data]
    
    filename = f"scores_{capability_name.lower()}.png"
    
    # Use non-interactive backend
    plt.ioff()
    
    # Set font size to match LaTeX document
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
        'figure.titlesize': 24
    })
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    y_positions = range(len(models))
    
    # Use original orange color from template
    bar_color = '#E74C3C'  # Original aggregate score color
    
    # Create bars
    bars = ax.barh(y_positions, scores, xerr=errors,
                   color=bar_color, alpha=0.8, height=0.6,
                   error_kw={'capsize': 5, 'capthick': 2})
    
    # Add model names on the right side of bars
    for i, model in enumerate(models):
        ax.text(scores[i] + errors[i] + 0.1, i, f'{model}',
                ha='left', va='center', fontweight='bold', fontsize=24)

    # Formatting - Update legend text to show specific capability
    ax.set_xlabel(f'{capability_name} Score', fontsize=24, fontweight='bold')
    ax.set_yticks([])  # Remove y-axis labels

    # Set data-driven x-axis limits with padding
    max_val = max([s + e for s, e in zip(scores, errors)])
    min_val = min([s - e for s, e in zip(scores, errors)])
    padding = (max_val - min_val) * 0.1
    x_min = max(0, min_val - padding)
    x_max = max_val + padding
    ax.set_xlim(x_min, x_max)

    # Add vertical line at exp(0) = 1 - same as hierarchical plots
    if x_min <= 1 <= x_max:  # Only show if 1 is in the visible range
        ax.axvline(x=1, color='gray', alpha=0.7, linewidth=2, linestyle='--')

    # Grid and styling
    ax.grid(True, axis='x', color='gray', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Plot saved: {filename}")

def create_results_table():
    """Create results table CSV with exponential scores and uncertainties"""
    
    print("🔄 Creating results table with exponential scores...")
    
    if not Path('win_rates.csv').exists():
        print("❌ win_rates.csv not found. Run script 02 first.")
        return None
    
    # Load win rates data
    capabilities_data = load_win_rates_data()
    
    # Dictionary to store all exponential scores and errors
    scores_data = {}
    errors_data = {}
    
    for capability_name in ['Deceive', 'Detect', 'Disclose']:
        if capability_name not in capabilities_data:
            print(f"❌ {capability_name} data not found")
            continue
            
        print(f"\n📊 Calculating {capability_name} scores...")
        
        # Get win rates and uncertainties for this capability
        cap_data = capabilities_data[capability_name]
        win_rates_df = cap_data['win_rates']
        uncertainties_df = cap_data['uncertainties']
        
        # Calculate aggregated scores
        exp_scores, exp_errors = calculate_aggregated_scores(win_rates_df, uncertainties_df, capability_name)
        
        # Store scores and errors
        scores_data[capability_name] = exp_scores
        errors_data[f'{capability_name}_Error'] = exp_errors
    
    # Create results table with scores and uncertainties
    if scores_data:
        # Combine scores and errors
        all_data = {**scores_data, **errors_data}
        results_df = pd.DataFrame(all_data)
        results_df = results_df.sort_index()  # Sort models alphabetically
        
        # Reorder columns: Deceive, Deceive_Error, Detect, Detect_Error, Disclose, Disclose_Error
        ordered_cols = []
        for cap in ['Deceive', 'Detect', 'Disclose']:
            if cap in results_df.columns:
                ordered_cols.extend([cap, f'{cap}_Error'])
        results_df = results_df[ordered_cols]
        
        results_df.to_csv('results_table.csv')
        print(f"\n💾 Results table saved: results_table.csv")
        print(f"   Shape: {results_df.shape} (models × capabilities+errors)")
        print(f"   Sample data:\n{results_df.head().round(3)}")
        
        return results_df
    
    return None

def create_score_plots():
    """Create aggregated score plots for all capabilities"""
    
    print("\n🔄 Creating aggregated score plots...")
    
    if not Path('win_rates.csv').exists():
        print("❌ win_rates.csv not found. Run script 02 first.")
        return 0
    
    # Load win rates data
    capabilities_data = load_win_rates_data()
    
    plots_created = 0
    
    for capability_name in ['Deceive', 'Detect', 'Disclose']:
        if capability_name not in capabilities_data:
            print(f"❌ {capability_name} data not found")
            continue
            
        print(f"\n📊 Creating {capability_name} score plot...")
        
        # Get win rates and uncertainties for this capability
        cap_data = capabilities_data[capability_name]
        win_rates_df = cap_data['win_rates']
        uncertainties_df = cap_data['uncertainties']
        
        # Calculate aggregated scores
        exp_scores, exp_errors = calculate_aggregated_scores(win_rates_df, uncertainties_df, capability_name)
        
        # Create plot
        create_exponential_score_plot(exp_scores, exp_errors, capability_name)
        
        plots_created += 1
    
    return plots_created

def main():
    """Create results table and aggregated score plots (separated)"""
    
    # Create results table
    results_df = create_results_table()
    
    # Create score plots
    plots_created = create_score_plots()
    
    if results_df is not None:
        print(f"\n✅ Created results table and {plots_created} score plots successfully!")
    else:
        print(f"\n⚠️ Results table creation failed, but created {plots_created} plots")

if __name__ == "__main__":
    main()