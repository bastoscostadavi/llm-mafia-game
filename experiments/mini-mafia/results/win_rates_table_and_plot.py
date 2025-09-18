#!/usr/bin/env python3
"""
Script 2: Create Win Rates and Plots

Reads win_counts.csv and creates:
1. win_rates.csv: Single CSV with 3 win rate tables using Bayesian estimation
2. 15 individual background bar plots with win rates and uncertainties

Formula: p̄ᵢᵦ = (kᵢᵦ + 1) / (nᵢᵦ + 2)  - Laplace rule of succession
Uncertainty: δp̄ᵢᵦ = √[(p̄ᵢᵦ(1-p̄ᵢᵦ))/(n+3)]  - Bayesian uncertainty
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from pathlib import Path

def get_background_color(background_key):
    """Get color based on background model type"""
    if 'mistral' in background_key.lower():
        return '#FF6600'  # Mistral orange
    elif 'gpt-5' in background_key.lower() and 'mini' in background_key.lower():
        return '#006B3C'  # GPT-5 Mini dark green
    elif 'gpt-4.1' in background_key.lower() and 'mini' in background_key.lower():
        return '#10A37F'  # GPT-4.1 Mini green
    elif 'grok' in background_key.lower() and 'mini' in background_key.lower():
        return '#8A2BE2'  # Grok Mini purple
    elif 'deepseek' in background_key.lower():
        return '#007ACC'  # DeepSeek blue
    else:
        return '#666666'  # Default gray

def calculate_bayesian_win_rate(wins, total_games=100):
    """Calculate Bayesian win rate and uncertainty using Beta distribution"""
    if total_games == 0:
        return 0.0, 0.0
    
    # Laplace rule of succession (Bayesian mean)
    bayesian_mean = (wins + 1) / (total_games + 2)
    
    # Bayesian uncertainty 
    bayesian_var = (bayesian_mean * (1 - bayesian_mean)) / (total_games + 3)
    bayesian_sd = math.sqrt(bayesian_var)
    
    # Convert to percentages
    return bayesian_mean * 100, bayesian_sd * 100

def create_win_rates_csv(win_counts_file='win_counts.csv', output_file='win_rates.csv'):
    """Convert win counts to win rates and uncertainties using Bayesian estimation"""
    
    # Read win counts
    df = pd.read_csv(win_counts_file)
    # Capitalize column names to match expected format
    df.columns = [col.replace('capability', 'Capability').replace('model', 'Model') for col in df.columns]
    df = df.set_index(['Capability', 'Model'])  # Multi-index: Capability, Model
    
    print(f"📊 Converting win counts to Bayesian win rates with uncertainties...")
    print(f"   Input shape: {df.shape}")
    
    # Create DataFrames for win rates and uncertainties
    win_rates_df = df.copy()
    uncertainties_df = df.copy()
    
    for index, row in df.iterrows():
        for background in df.columns:
            wins = row[background]
            win_rate, uncertainty = calculate_bayesian_win_rate(wins, 100)
            win_rates_df.loc[index, background] = round(win_rate, 2)
            uncertainties_df.loc[index, background] = round(uncertainty, 2)
    
    # Create combined DataFrame with alternating win rate and uncertainty columns
    # Reset index to work with the data more easily
    win_rates_reset = win_rates_df.reset_index()
    uncertainties_reset = uncertainties_df.reset_index()
    
    # Create new DataFrame with alternating columns
    combined_data = []
    for _, row in win_rates_reset.iterrows():
        capability = row['Capability']
        model = row['Model']
        
        # Get corresponding uncertainty row
        uncertainty_row = uncertainties_reset[
            (uncertainties_reset['Capability'] == capability) & 
            (uncertainties_reset['Model'] == model)
        ].iloc[0]
        
        # Create row with alternating win rate and uncertainty
        new_row = {'Capability': capability, 'Model': model}
        for background in df.columns:
            new_row[background] = row[background]
            new_row[f'{background}_Error'] = uncertainty_row[background]
        
        combined_data.append(new_row)
    
    combined_df = pd.DataFrame(combined_data)
    
    # Reorder columns: Capability, Model, then alternating background and background_Error
    ordered_cols = ['Capability', 'Model']
    for background in df.columns:
        ordered_cols.extend([background, f'{background}_Error'])
    
    combined_df = combined_df[ordered_cols]
    
    # Save to CSV
    output_path = Path(output_file)
    combined_df.to_csv(output_path, index=False)
    
    print(f"✅ Win rates with uncertainties saved to: {output_path}")
    print(f"   Shape: {combined_df.shape} (includes uncertainties)")
    print(f"   Preview:")
    print(combined_df.head(3))
    
    # Return the original win rates format for plotting compatibility
    return win_rates_df

def create_benchmark_plot(models, win_rates, uncertainties, background_name, capability_name,
                         role_name, use_good_wins=False, output_dir=''):
    """Create horizontal bar plot with exact same formatting as original"""
    
    # Sort by win rate (descending) and reverse for plotting (highest at top)
    sorted_data = sorted(zip(models, win_rates, uncertainties), key=lambda x: x[1], reverse=True)
    sorted_models = [x[0] for x in reversed(sorted_data)]
    sorted_rates = [x[1] for x in reversed(sorted_data)]  
    sorted_errors = [x[2] for x in reversed(sorted_data)]
    
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
    
    y_positions = range(len(sorted_models))
    
    # Get color based on background
    bar_color = get_background_color(background_name)
    
    # Create main bars
    bars = ax.barh(y_positions, sorted_rates, xerr=sorted_errors, 
                   color=bar_color, alpha=0.8, height=0.6,
                   error_kw={'capsize': 5, 'capthick': 2})
    
    # Add model names on the right side of bars
    for i, model in enumerate(sorted_models):
        ax.text(sorted_rates[i] + sorted_errors[i] + 1.5, i, f'{model}', 
                ha='left', va='center', fontweight='bold', fontsize=24)
    
    # Set axis labels based on metric type
    if use_good_wins:
        xlabel = 'Town Win Rate (%)'
    else:
        xlabel = 'Mafia Win Rate (%)'
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold')
    ax.set_yticks([])  # Remove y-axis labels
    
    # Grid and styling
    ax.grid(True, axis='x', color='gray', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Create filename with new naming convention: win_rates_capability_background
    background_clean = background_name.lower().replace(' ', '_').replace('.', '')
    capability_clean = capability_name.lower()
    filename = f"{output_dir}win_rates_{capability_clean}_{background_clean}.png"
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  📈 Plot saved: {filename}")
    return filename

def create_plots_from_win_rates(win_rates_df, output_dir=''):
    """Create all background plots from win rates dataframe"""
    
    print(f"\n📊 Creating win rate bar plots...")
    
    # Define capabilities and their settings
    capability_settings = {
        'Deceive': {'role_name': 'mafioso', 'use_good_wins': False},
        'Detect': {'role_name': 'villager', 'use_good_wins': True},
        'Disclose': {'role_name': 'detective', 'use_good_wins': True}
    }
    
    backgrounds = win_rates_df.columns.tolist()
    plots_created = 0
    
    for capability in ['Deceive', 'Detect', 'Disclose']:
        if capability not in win_rates_df.index.get_level_values(0):
            continue
            
        settings = capability_settings[capability]
        print(f"\n  {capability} plots:")
        
        # Get data for this capability
        capability_data = win_rates_df.loc[capability]
        models = capability_data.index.tolist()
        
        for background in backgrounds:
            print(f"    Background: {background}")
            
            # Get win rates for this background
            win_rates = capability_data[background].tolist()
            
            # Calculate uncertainties for each win rate
            # We need to reverse-calculate wins from win rates for uncertainty
            uncertainties = []
            for win_rate in win_rates:
                # Reverse Laplace formula: wins ≈ (win_rate/100 * 102) - 1
                estimated_wins = max(0, (win_rate/100 * 102) - 1)
                _, uncertainty = calculate_bayesian_win_rate(estimated_wins, 100)
                uncertainties.append(uncertainty)
            
            # Create plot
            create_benchmark_plot(
                models, win_rates, uncertainties, background,
                capability, settings['role_name'], settings['use_good_wins'], output_dir
            )
            plots_created += 1
    
    return plots_created

def main():
    """Create win rates CSV and plots from win counts"""

    print("🔄 Creating win rates and plots from win counts...")

    # Check if win counts file exists
    win_counts_path = 'win_counts.csv'
    if not Path(win_counts_path).exists():
        print("❌ win_counts.csv not found. Run script 01 first.")
        return

    # Create win rates CSV
    win_rates_df = create_win_rates_csv(win_counts_path, 'win_rates.csv')

    # Create plots
    plots_created = create_plots_from_win_rates(win_rates_df, '')
    
    print(f"\n✅ Created win_rates.csv and {plots_created} plots successfully!")

if __name__ == "__main__":
    main()