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


def load_human_data():
    """Load human gameplay data from database and calculate win rates per background"""

    if not DB_PATH.exists():
        print(f"❌ Human database not found: {DB_PATH}")
        return None

    conn = sqlite3.connect(DB_PATH)

    # Load all games (include game_id for ordering)
    query = "SELECT game_id, background_name, winner FROM games ORDER BY game_id"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if len(df) == 0:
        print("❌ No human games found in database")
        return None

    print(f"📊 Loaded {len(df)} human games")

    # Map background names first
    df['mapped_background'] = df['background_name'].map(lambda x: BACKGROUND_NAME_MAPPING.get(x, x))

    # Calculate win rate per mapped background (aggregating across different name formats)
    # Human plays as mafioso, wins when winner = 'EVIL'
    background_stats = []

    for mapped_background in df['mapped_background'].unique():
        # Get all games for this mapped background (may include multiple name formats)
        bg_games = df[df['mapped_background'] == mapped_background]

        # Show which original names were aggregated
        original_names = bg_games['background_name'].unique()

        total_games = len(bg_games)
        wins = (bg_games['winner'] == 'EVIL').sum()
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


def load_ai_deceive_data():
    """Load AI Deceive win rates from win_rates.csv"""

    if not WIN_RATES_CSV.exists():
        print(f"❌ Win rates CSV not found: {WIN_RATES_CSV}")
        return None, None

    df = pd.read_csv(WIN_RATES_CSV)

    # Filter to Deceive capability
    deceive_df = df[df['Capability'] == 'Deceive'].copy()

    if len(deceive_df) == 0:
        print("❌ No Deceive data found in win_rates.csv")
        return None, None

    # Set model as index
    deceive_df = deceive_df.set_index('Model')
    deceive_df = deceive_df.drop(columns=['Capability'])

    # Separate win rates and uncertainties
    background_cols = [col for col in deceive_df.columns if not col.endswith('_Error')]
    error_cols = [col for col in deceive_df.columns if col.endswith('_Error')]

    win_rates = deceive_df[background_cols]
    uncertainties = deceive_df[error_cols]

    # Rename uncertainty columns to match background names
    uncertainties.columns = [col.replace('_Error', '') for col in uncertainties.columns]

    print(f"📊 Loaded AI data: {len(win_rates)} models × {len(background_cols)} backgrounds")

    return win_rates, uncertainties


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


def create_plot_with_human(exp_scores, exp_errors, win_rates_df):
    """Create deceive score plot including human data"""

    models = exp_scores.index.tolist()
    scores = exp_scores.values.tolist()
    errors = exp_errors.values.tolist()

    filename = str(OUTPUT_DIR / "scores_deceive_with_human.png")
    xlabel = 'Deceive Score'

    # Sort data first
    sorted_data = sorted(zip(models, scores, errors), key=lambda x: x[1])
    sorted_models = [x[0] for x in sorted_data]
    sorted_scores = [x[1] for x in sorted_data]
    sorted_errors = [x[2] for x in sorted_data]

    # Assign colors AFTER sorting: Human in green, others in red
    colors = ['#2ECC71' if model == 'Human' else '#E74C3C' for model in sorted_models]

    # Calculate reference lines (Deceive is mafia win rate)
    # Non-info exchange: 5/12 ≈ 41.67% mafia win rate
    ref_50 = calculate_reference_exp_score(50.0, win_rates_df)
    ref_noinfo = calculate_reference_exp_score(100 * 5/12, win_rates_df)

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


def create_background_win_rates_plots(combined_win_rates, combined_uncertainties):
    """Create win rates plots for each background with human data"""

    from utils import get_background_color

    print("\n📊 Creating background-specific win rates plots...")

    for background in combined_win_rates.columns:
        # Get win rates and uncertainties for this background
        win_rates = combined_win_rates[background].tolist()
        uncertainties = combined_uncertainties[background].tolist()
        models = combined_win_rates.index.tolist()

        # Create filename
        background_clean = background.lower().replace(' ', '_').replace('.', '')
        filename = str(OUTPUT_DIR / f"win_rates_deceive_{background_clean}_with_human.png")
        xlabel = 'Mafia Win Rate (%)'

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

        # Define reference lines
        # Non-information exchange baseline: 5/12 ≈ 41.67% mafia win rate
        reference_lines = [
            (100 * 5/12, 'Non-Info Exchange', 'purple', ':')
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

        print(f"  📈 {background}: {filename}")


def main():
    """Main execution"""

    print("🎯 Creating Deceive Score Plot with Human Data\n")
    print("=" * 60)

    # Load human data
    print("\n1️⃣ Loading human gameplay data...")
    human_data = load_human_data()

    if human_data is None:
        print("❌ Failed to load human data")
        return

    # Load AI data
    print("\n2️⃣ Loading AI deceive win rates...")
    ai_win_rates, ai_uncertainties = load_ai_deceive_data()

    if ai_win_rates is None:
        print("❌ Failed to load AI data")
        return

    # Combine data
    print("\n3️⃣ Combining human and AI data...")
    combined_win_rates, combined_uncertainties = add_human_to_ai_data(
        ai_win_rates, ai_uncertainties, human_data
    )

    if combined_win_rates is None:
        print("❌ Failed to combine data")
        return

    # Calculate scores
    print("\n4️⃣ Calculating aggregated scores...")
    avg_z_scores, avg_z_errors, exp_scores, exp_errors = calculate_aggregated_scores(
        combined_win_rates, combined_uncertainties
    )

    # Create aggregated deceive score plot
    print("\n5️⃣ Creating aggregated deceive score plot...")
    create_plot_with_human(exp_scores, exp_errors, combined_win_rates)

    # Create background-specific win rates plots
    print("\n6️⃣ Creating background-specific win rates plots...")
    create_background_win_rates_plots(combined_win_rates, combined_uncertainties)

    print("\n" + "=" * 60)
    print("✅ Done! Show these plots to your class!")
    print(f"   Human Deceive Score: {exp_scores['Human']:.3f} ± {exp_errors['Human']:.3f}")
    print(f"   Rank: {(exp_scores > exp_scores['Human']).sum() + 1} out of {len(exp_scores)}")


if __name__ == "__main__":
    main()
