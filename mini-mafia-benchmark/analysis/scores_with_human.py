#!/usr/bin/env python3
"""
Create Deceive Score Plot with Human Data

Extends the deceive score analysis to include human gameplay data.
Only includes backgrounds where human data exists.

Algorithm:
1. Load AI win rates from win_rates.csv (Deceive capability)
2. Load human gameplay data from database
3. Calculate human win rate per background
4. Filter to only backgrounds present in human data
5. Calculate z-scores and exponential scores for AI + human
6. Create combined plot
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import sys

# Add parent directory to path for utils import
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BENCHMARK_DIR / 'results'))

from utils import create_horizontal_bar_plot

# Paths
DB_PATH = BENCHMARK_DIR / 'database' / 'mini_mafia_human.db'
WIN_RATES_CSV = BENCHMARK_DIR / 'results' / 'win_rates.csv'
OUTPUT_DIR = BENCHMARK_DIR / 'results'

# Background name mapping from human DB format to AI data format
BACKGROUND_NAME_MAPPING = {
    'gpt-5-mini': 'GPT-5 Mini',
    'gpt-4.1-mini': 'GPT-4.1 Mini',
    'gpt-4.1_mini': 'GPT-4.1 Mini',  # Added: underscore version
    'grok-3-mini': 'Grok 3 Mini',
    'deepseek': 'DeepSeek V3.1',
    'mistral': 'Mistral 7B Instruct',
    # Old format compatibility
    'gpt-5_mini_gpt-5_mini': 'GPT-5 Mini',
    'gpt-4.1_mini_gpt-4.1_mini': 'GPT-4.1 Mini',
    'grok-3_mini_grok-3_mini': 'Grok 3 Mini',
    'deepseek_deepseek': 'DeepSeek V3.1',
    'mistral_mistral': 'Mistral 7B Instruct',
}


def load_human_data(capability='Deceive'):
    """Load human gameplay data from database and calculate win rates per background for a specific capability

    Args:
        capability: 'Deceive' (human as mafioso) or 'Detect' (human as villager/detective)
    """

    if not DB_PATH.exists():
        print(f"❌ Human database not found: {DB_PATH}")
        return None

    conn = sqlite3.connect(DB_PATH)

    # Load all games including human_role
    query = "SELECT game_id, background_name, winner, human_role FROM games ORDER BY game_id"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if len(df) == 0:
        print("❌ No human games found in database")
        return None

    # Filter by capability
    if capability == 'Deceive':
        # Deceive = human plays as mafioso
        df = df[df['human_role'] == 'mafioso'].copy()
    else:
        # Detect/Disclose = human plays as villager or detective
        df = df[df['human_role'].isin(['villager', 'detective'])].copy()

    if len(df) == 0:
        print(f"❌ No human {capability} games found in database")
        return None

    print(f"📊 Loaded {len(df)} human {capability} games")

    # Map background names first
    df['mapped_background'] = df['background_name'].map(lambda x: BACKGROUND_NAME_MAPPING.get(x, x))

    # Calculate win rate per mapped background (aggregating across different name formats)
    # Display faction win rates:
    # - For Deceive: Display mafia faction win rate (EVIL = mafia wins)
    # - For Detect: Display town faction win rate (GOOD = town wins)
    background_stats = []

    for mapped_background in df['mapped_background'].unique():
        # Get all games for this mapped background (may include multiple name formats)
        bg_games = df[df['mapped_background'] == mapped_background]

        # Show which original names were aggregated
        original_names = bg_games['background_name'].unique()

        total_games = len(bg_games)

        # Count faction wins based on capability
        if capability == 'Deceive':
            # EVIL = mafia wins
            wins = (bg_games['winner'] == 'EVIL').sum()
        else:
            # GOOD = town wins (for Detect/Disclose)
            wins = (bg_games['winner'] == 'GOOD').sum()

        win_rate = (wins / total_games) * 100 if total_games > 0 else 0

        # Calculate binomial standard error: sqrt(p*(1-p)/n)
        p = wins / total_games if total_games > 0 else 0
        error = np.sqrt(p * (1 - p) / total_games) * 100 if total_games > 0 else 0

        background_stats.append({
            'background': mapped_background,
            'original_names': ', '.join(original_names),
            'win_rate': win_rate,
            'error': error,
            'games': total_games,
            'wins': wins
        })

        names_str = ', '.join(original_names) if len(original_names) > 1 else original_names[0]
        print(f"  {names_str} → {mapped_background}: {wins}/{total_games} wins = {win_rate:.1f}% ± {error:.1f}%")

    return pd.DataFrame(background_stats)


def load_ai_capability_data(capability):
    """Load AI win rates for a specific capability from win_rates.csv"""

    if not WIN_RATES_CSV.exists():
        print(f"❌ Win rates CSV not found: {WIN_RATES_CSV}")
        return None, None

    df = pd.read_csv(WIN_RATES_CSV)

    # Filter to specified capability
    cap_df = df[df['Capability'] == capability].copy()

    if len(cap_df) == 0:
        print(f"❌ No {capability} data found in win_rates.csv")
        return None, None

    # Set model as index
    cap_df = cap_df.set_index('Model')
    cap_df = cap_df.drop(columns=['Capability'])

    # Separate win rates and uncertainties
    background_cols = [col for col in cap_df.columns if not col.endswith('_Error')]
    error_cols = [col for col in cap_df.columns if col.endswith('_Error')]

    win_rates = cap_df[background_cols]
    uncertainties = cap_df[error_cols]

    # Rename uncertainty columns to match background names
    uncertainties.columns = [col.replace('_Error', '') for col in uncertainties.columns]

    print(f"📊 Loaded AI {capability} data: {len(win_rates)} models × {len(background_cols)} backgrounds")

    return win_rates, uncertainties


def load_ai_deceive_data():
    """Load AI Deceive win rates from win_rates.csv"""
    return load_ai_capability_data('Deceive')


def load_ai_detect_data():
    """Load AI Detect win rates from win_rates.csv"""
    return load_ai_capability_data('Detect')


def add_human_to_ai_data(ai_win_rates, ai_uncertainties, human_data):
    """Add human data to AI dataframes, filtering to common backgrounds"""

    # Get backgrounds present in human data
    human_backgrounds = set(human_data['background'].values)
    ai_backgrounds = set(ai_win_rates.columns)

    # Find common backgrounds
    common_backgrounds = list(human_backgrounds & ai_backgrounds)

    if len(common_backgrounds) == 0:
        print("❌ No common backgrounds between human and AI data")
        print(f"   Human backgrounds: {sorted(human_backgrounds)}")
        print(f"   AI backgrounds: {sorted(ai_backgrounds)}")
        return None, None

    print(f"\n📊 Common backgrounds: {len(common_backgrounds)}")
    for bg in sorted(common_backgrounds):
        print(f"   • {bg}")

    # Filter AI data to common backgrounds
    ai_win_rates_filtered = ai_win_rates[common_backgrounds].copy()
    ai_uncertainties_filtered = ai_uncertainties[common_backgrounds].copy()

    # Create human row
    human_win_rates = {}
    human_uncertainties = {}

    for bg in common_backgrounds:
        human_bg_data = human_data[human_data['background'] == bg].iloc[0]
        human_win_rates[bg] = human_bg_data['win_rate']
        human_uncertainties[bg] = human_bg_data['error']

    # Add human row to dataframes
    human_win_rates_series = pd.Series(human_win_rates, name='Human')
    human_uncertainties_series = pd.Series(human_uncertainties, name='Human')

    combined_win_rates = pd.concat([ai_win_rates_filtered, human_win_rates_series.to_frame().T])
    combined_uncertainties = pd.concat([ai_uncertainties_filtered, human_uncertainties_series.to_frame().T])

    print(f"\n📊 Combined data: {len(combined_win_rates)} models (including Human) × {len(common_backgrounds)} backgrounds")

    return combined_win_rates, combined_uncertainties


def calculate_reference_exp_score(reference_value, win_rates_df):
    """
    Calculate exponential score for a reference win rate value.
    Uses the same z-score calculation as for models.

    Args:
        reference_value: Win rate percentage (e.g., 50.0 for 50%)
        win_rates_df: DataFrame of win rates across backgrounds

    Returns:
        Exponential score for the reference value
    """
    z_scores = []

    for background in win_rates_df.columns:
        background_win_rates = win_rates_df[background].values
        bg_mean = np.mean(background_win_rates)
        bg_std = np.std(background_win_rates, ddof=1)

        if bg_std > 0:
            z = (reference_value - bg_mean) / bg_std
            z_scores.append(z)
        else:
            z_scores.append(0.0)

    avg_z = np.mean(z_scores)
    exp_score = np.exp(avg_z)

    return exp_score


def calculate_aggregated_scores(win_rates_df, uncertainties_df):
    """
    Calculate aggregated exponential scores with uncertainty propagation.
    Same algorithm as scores_table_and_plot.py
    """

    # Calculate z-scores for each model in each background
    z_scores_df = pd.DataFrame(index=win_rates_df.index, columns=win_rates_df.columns)
    z_errors_df = pd.DataFrame(index=win_rates_df.index, columns=win_rates_df.columns)

    print("\n📊 Calculating z-scores per background:")

    for background in win_rates_df.columns:
        # Get win rates and uncertainties for this background
        background_win_rates = win_rates_df[background].values
        background_uncertainties = uncertainties_df[background].values

        # Calculate background mean and standard deviation
        bg_mean = np.mean(background_win_rates)
        bg_std = np.std(background_win_rates, ddof=1)

        print(f"  {background}: mean={bg_mean:.1f}%, std={bg_std:.1f}%")

        # Calculate z-scores
        if bg_std > 0:
            z_scores = (background_win_rates - bg_mean) / bg_std
            z_scores_df[background] = z_scores

            # Propagate uncertainties to z-scores
            z_errors = background_uncertainties / bg_std
            z_errors_df[background] = z_errors
        else:
            z_scores_df[background] = 0.0
            z_errors_df[background] = 0.0

    # Average z-scores across backgrounds
    avg_z_scores = z_scores_df.mean(axis=1)

    # Calculate uncertainty in average z-scores
    z_variance_sum = (z_errors_df ** 2).sum(axis=1)
    avg_z_errors = np.sqrt(z_variance_sum) / len(z_scores_df.columns)

    # Calculate exponential scores
    exp_scores = np.exp(avg_z_scores)
    exp_errors = exp_scores * avg_z_errors

    print(f"\n📊 Exponential scores:")
    for model in exp_scores.index:
        print(f"  {model}: {exp_scores[model]:.3f} ± {exp_errors[model]:.3f}")

    return avg_z_scores, avg_z_errors, exp_scores, exp_errors


def create_capability_score_plot(exp_scores, exp_errors, win_rates_df, capability):
    """Create capability score plot including human data"""

    models = exp_scores.index.tolist()
    scores = exp_scores.values.tolist()
    errors = exp_errors.values.tolist()

    filename = str(OUTPUT_DIR / f"scores_{capability.lower()}_with_human.png")
    xlabel = f'{capability} Score'

    # Sort data first
    sorted_data = sorted(zip(models, scores, errors), key=lambda x: x[1])
    sorted_models = [x[0] for x in sorted_data]
    sorted_scores = [x[1] for x in sorted_data]
    sorted_errors = [x[2] for x in sorted_data]

    # Assign colors AFTER sorting: Human in green, others in red
    colors = ['#2ECC71' if model == 'Human' else '#E74C3C' for model in sorted_models]

    # Calculate reference lines based on capability
    # Deceive is mafia win rate, Detect/Disclose are town win rates
    if capability == 'Deceive':
        # Mafia win rates: 50% and 5/12 ≈ 41.67%
        ref_50 = calculate_reference_exp_score(50.0, win_rates_df)
        ref_noinfo = calculate_reference_exp_score(100 * 5/12, win_rates_df)
    else:
        # Town win rates: 50% and 7/12 ≈ 58.33%
        ref_50 = calculate_reference_exp_score(50.0, win_rates_df)
        ref_noinfo = calculate_reference_exp_score(100 * 7/12, win_rates_df)

    reference_lines = [
        (ref_50, '50%', 'black', '--'),
        (ref_noinfo, 'Non-Info Exchange', 'purple', ':')
    ]

    print(f"  Reference lines: 50%={ref_50:.3f}, Non-Info={ref_noinfo:.3f}")

    create_horizontal_bar_plot(
        models=sorted_models,
        values=sorted_scores,
        errors=sorted_errors,
        xlabel=xlabel,
        filename=filename,
        color=colors,
        sort_ascending=None,  # Already sorted
        show_reference_line=False,  # Don't show the x=1 reference line
        x_min=0,  # Force plot to start at 0
        reference_lines=reference_lines
    )

    print(f"\n💾 Plot saved: {filename}")


def create_single_background_plot(combined_win_rates, combined_uncertainties, background, capability):
    """Create a single background win rate plot with human data"""

    from utils import get_background_color

    # Get win rates and uncertainties for this background
    win_rates = combined_win_rates[background].tolist()
    uncertainties = combined_uncertainties[background].tolist()
    models = combined_win_rates.index.tolist()

    # Create filename
    background_clean = background.lower().replace(' ', '_').replace('.', '')
    filename = str(OUTPUT_DIR / f"win_rates_{capability.lower()}_{background_clean}_with_human.png")

    # Set xlabel based on capability
    if capability == 'Deceive':
        xlabel = 'Mafia Win Rate (%)'
    else:
        xlabel = 'Town Win Rate (%)'

    # Get color for this background
    bg_color = get_background_color(background)

    # Sort by win rate (descending) then reverse to put highest at top
    sorted_data = sorted(zip(models, win_rates, uncertainties), key=lambda x: x[1], reverse=True)
    sorted_models = [x[0] for x in sorted_data]
    sorted_win_rates = [x[1] for x in sorted_data]
    sorted_uncertainties = [x[2] for x in sorted_data]

    # Reverse to put highest at top
    sorted_models.reverse()
    sorted_win_rates.reverse()
    sorted_uncertainties.reverse()

    # Assign colors: green for Human, background color for AI models
    colors = ['#2ECC71' if model == 'Human' else bg_color for model in sorted_models]

    # Define reference lines based on capability
    if capability == 'Deceive':
        # Non-information exchange baseline: 5/12 ≈ 41.67% mafia win rate
        reference_lines = [
            (100 * 5/12, 'Non-Info Exchange', 'purple', ':')
        ]
    else:
        # Non-information exchange baseline: 7/12 ≈ 58.33% town win rate
        reference_lines = [
            (100 * 7/12, 'Non-Info Exchange', 'purple', ':')
        ]

    create_horizontal_bar_plot(
        models=sorted_models,
        values=sorted_win_rates,
        errors=sorted_uncertainties,
        xlabel=xlabel,
        filename=filename,
        color=colors,
        sort_ascending=None,  # Already sorted
        show_reference_line=False,
        text_offset=1.5,
        reference_lines=reference_lines
    )

    print(f"  📈 {capability} - {background}: {filename}")


def create_background_win_rates_plots(combined_win_rates, combined_uncertainties, capability='Deceive'):
    """Create win rates plots for each background with human data"""

    print(f"\n📊 Creating {capability} background-specific win rates plots...")

    for background in combined_win_rates.columns:
        create_single_background_plot(combined_win_rates, combined_uncertainties, background, capability)


def main():
    """Main execution"""

    print("🎯 Creating Score Plots with Human Data\n")
    print("=" * 60)

    # ========== DECEIVE CAPABILITY ==========
    print("\n" + "=" * 60)
    print("DECEIVE CAPABILITY")
    print("=" * 60)

    # Load human Deceive data
    print("\n1️⃣ Loading human Deceive gameplay data...")
    human_deceive_data = load_human_data('Deceive')

    if human_deceive_data is None:
        print("❌ Failed to load human Deceive data")
        return

    # Load AI Deceive data
    print("\n2️⃣ Loading AI deceive win rates...")
    ai_deceive_win_rates, ai_deceive_uncertainties = load_ai_deceive_data()

    if ai_deceive_win_rates is None:
        print("❌ Failed to load AI deceive data")
        return

    # Combine data for Deceive
    print("\n3️⃣ Combining human and AI data for Deceive...")
    deceive_win_rates, deceive_uncertainties = add_human_to_ai_data(
        ai_deceive_win_rates, ai_deceive_uncertainties, human_deceive_data
    )

    if deceive_win_rates is None:
        print("❌ Failed to combine deceive data")
        return

    # Calculate Deceive scores
    print("\n4️⃣ Calculating aggregated Deceive scores...")
    deceive_avg_z, deceive_avg_z_err, deceive_exp, deceive_exp_err = calculate_aggregated_scores(
        deceive_win_rates, deceive_uncertainties
    )

    # Create aggregated Deceive score plot
    print("\n5️⃣ Creating aggregated Deceive score plot...")
    create_capability_score_plot(deceive_exp, deceive_exp_err, deceive_win_rates, 'Deceive')

    # Create background-specific Deceive win rates plots
    print("\n6️⃣ Creating background-specific Deceive win rates plots...")
    create_background_win_rates_plots(deceive_win_rates, deceive_uncertainties, 'Deceive')

    # ========== DETECT CAPABILITY ==========
    print("\n" + "=" * 60)
    print("DETECT CAPABILITY")
    print("=" * 60)

    # Load human Detect data
    print("\n7️⃣ Loading human Detect gameplay data...")
    human_detect_data = load_human_data('Detect')

    if human_detect_data is None:
        print("❌ Failed to load human Detect data")
        return

    # Load AI Detect data
    print("\n8️⃣ Loading AI detect win rates...")
    ai_detect_win_rates, ai_detect_uncertainties = load_ai_detect_data()

    if ai_detect_win_rates is None:
        print("❌ Failed to load AI detect data")
        return

    # Combine data for Detect
    print("\n9️⃣ Combining human and AI data for Detect...")
    detect_win_rates, detect_uncertainties = add_human_to_ai_data(
        ai_detect_win_rates, ai_detect_uncertainties, human_detect_data
    )

    if detect_win_rates is None:
        print("❌ Failed to combine detect data")
        return

    # Calculate Detect scores
    print("\n🔟 Calculating aggregated Detect scores...")
    detect_avg_z, detect_avg_z_err, detect_exp, detect_exp_err = calculate_aggregated_scores(
        detect_win_rates, detect_uncertainties
    )

    # Create aggregated Detect score plot
    print("\n1️⃣1️⃣ Creating aggregated Detect score plot...")
    create_capability_score_plot(detect_exp, detect_exp_err, detect_win_rates, 'Detect')

    # Create GPT-5 Mini background Detect win rate plot
    print("\n1️⃣2️⃣ Creating GPT-5 Mini Detect win rate plot...")
    if 'GPT-5 Mini' in detect_win_rates.columns:
        create_single_background_plot(detect_win_rates, detect_uncertainties, 'GPT-5 Mini', 'Detect')
    else:
        print("  ⚠️ GPT-5 Mini background not found in detect data")

    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("✅ Done! Summary:")
    print(f"   Human Deceive Score: {deceive_exp['Human']:.3f} ± {deceive_exp_err['Human']:.3f}")
    print(f"   Human Deceive Rank: {(deceive_exp > deceive_exp['Human']).sum() + 1} out of {len(deceive_exp)}")
    print(f"   Human Detect Score: {detect_exp['Human']:.3f} ± {detect_exp_err['Human']:.3f}")
    print(f"   Human Detect Rank: {(detect_exp > detect_exp['Human']).sum() + 1} out of {len(detect_exp)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
