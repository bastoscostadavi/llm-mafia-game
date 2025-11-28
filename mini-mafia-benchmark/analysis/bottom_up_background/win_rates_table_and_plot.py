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
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import bayesian_win_rate, get_background_color, create_horizontal_bar_plot

# Set up paths
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "bottom_up_background"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def create_win_rates_csv(win_counts_file=None, output_file=None):
    if win_counts_file is None:
        win_counts_file = RESULTS_DIR / 'win_counts.csv'
    if output_file is None:
        output_file = RESULTS_DIR / 'win_rates.csv'
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

    # Define reference lines
    # Non-information exchange baseline: 5/12 mafia win rate (7/12 town win rate)
    # = 41.67% mafia, 58.33% town
    if use_good_wins:
        # Town win rate plot (Detect and Disclose capabilities)
        reference_lines = [
            (100 * 7/12, 'Non-Info Exchange', 'purple', ':')
        ]
    else:
        # Mafia win rate plot (Deceive capability)
        reference_lines = [
            (100 * 5/12, 'Non-Info Exchange', 'purple', ':')
        ]

    # Use unified plotting function with win-rates specific sorting (descending then reversed)
    create_horizontal_bar_plot(
        models=models,
        values=win_rates,
        errors=uncertainties,
        xlabel=xlabel,
        filename=filename,
        color=bar_color,
        sort_ascending=False,  # Sort descending first
        show_reference_line=False,  # Win rates don't need reference line at 1
        text_offset=1.5,  # Win rates use larger offset
        reverse_after_sort=True,  # Then reverse to put highest at top
        reference_lines=reference_lines
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
    win_counts_path = RESULTS_DIR / 'win_counts.csv'
    if not win_counts_path.exists():
        print("❌ win_counts.csv not found. Run win_counts_table.py first.")
        return

    # Create win rates CSV
    win_rates_path = RESULTS_DIR / 'win_rates.csv'
    win_rates_df, uncertainties_df = create_win_rates_csv(win_counts_path, win_rates_path)

    # Create plots using computed uncertainties (output_dir needs trailing slash)
    plots_created = create_plots_from_win_rates(win_rates_df, uncertainties_df, str(RESULTS_DIR) + '/')

    print(f"\n✅ Created win_rates.csv and {plots_created} plots successfully!")

if __name__ == "__main__":
    main()