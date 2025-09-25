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
import numpy as np
import math
from pathlib import Path
from utils import bayesian_win_rate, get_background_color, create_horizontal_bar_plot

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
            win_rate, uncertainty = bayesian_win_rate(wins, 100)
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
    
    # Return both win rates and uncertainties dataframes for plotting compatibility
    return win_rates_df, uncertainties_df

def create_benchmark_plot(models, win_rates, uncertainties, background_name, capability_name,
                         role_name, use_good_wins=False, output_dir=''):
    """Create horizontal bar plot with exact same formatting as original"""

    # Set axis labels based on metric type
    if use_good_wins:
        xlabel = 'Town Win Rate (%)'
    else:
        xlabel = 'Mafia Win Rate (%)'

    # Create filename with new naming convention: win_rates_capability_background
    background_clean = background_name.lower().replace(' ', '_').replace('.', '')
    capability_clean = capability_name.lower()
    filename = f"{output_dir}win_rates_{capability_clean}_{background_clean}.png"

    # Get color based on background
    bar_color = get_background_color(background_name)

    # Use unified plotting function with win-rates specific sorting (descending then reversed)
    create_horizontal_bar_plot(
        models=models,
        values=win_rates,
        errors=uncertainties,
        xlabel=xlabel,
        filename=filename,
        color=bar_color,
        sort_ascending=False,  # Sort descending first
        show_reference_line=False,  # Win rates don't need reference line
        text_offset=1.5,  # Win rates use larger offset
        reverse_after_sort=True  # Then reverse to put highest at top
    )

    print(f"  📈 Plot saved: {filename}")
    return filename

def create_plots_from_win_rates(win_rates_df, uncertainties_df, output_dir=''):
    """Create all background plots from win rates dataframe with computed uncertainties"""

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
        capability_win_rates = win_rates_df.loc[capability]
        capability_uncertainties = uncertainties_df.loc[capability]
        models = capability_win_rates.index.tolist()

        for background in backgrounds:
            print(f"    Background: {background}")

            # Get win rates and uncertainties for this background
            win_rates = capability_win_rates[background].tolist()
            uncertainties = capability_uncertainties[background].tolist()

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
    win_rates_df, uncertainties_df = create_win_rates_csv(win_counts_path, 'win_rates.csv')

    # Create plots using computed uncertainties
    plots_created = create_plots_from_win_rates(win_rates_df, uncertainties_df, '')
    
    print(f"\n✅ Created win_rates.csv and {plots_created} plots successfully!")

if __name__ == "__main__":
    main()