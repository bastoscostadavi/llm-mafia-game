#!/usr/bin/env python3
"""
Theoretical Model with Human Data

Extends the theoretical model to include human players:
    logit(p_ijk) = c_k × (a_i - b_j)

Combines:
- 14,000 LLM vs LLM games from benchmark
- Human games from mini_mafia_human.db

Outputs deceive and detect plots with humans included.
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# Set up paths
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "top_down_theoretical"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(__file__).parent.parent.parent / "database" / "mini_mafia.db"
HUMAN_DB_PATH = Path(__file__).parent.parent.parent / "database" / "mini_mafia_human.db"
HUMAN_OLD_DB_PATH = Path(__file__).parent.parent.parent / "database" / "mini_mafia_human_old.db"

# For Bayesian inference (required)
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False

warnings.filterwarnings('ignore')

def get_display_name(model_name):
    """Convert internal model names to display names"""
    display_names = {
        'claude-opus-4-1-20250805': 'Claude Opus 4.1',
        'claude-sonnet-4-20250514': 'Claude Sonnet 4',
        'deepseek-chat': 'DeepSeek V3.1',
        'gemini-2.5-flash-lite': 'Gemini 2.5 Flash Lite',
        'grok-3-mini': 'Grok 3 Mini',
        'gpt-4.1-mini': 'GPT-4.1 Mini',
        'gpt-5-mini': 'GPT-5 Mini',
        'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf': 'Mistral 7B Instruct',
        'Qwen2.5-7B-Instruct-Q4_K_M.gguf': 'Qwen2.5 7B Instruct',
        'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf': 'Llama 3.1 8B Instruct',
        'human': 'Human'
    }
    return display_names.get(model_name, model_name)

def get_llm_games_data():
    """Get unique games data from LLM benchmark table."""
    conn = sqlite3.connect(DB_PATH)

    print("🔍 Loading LLM games from benchmark table...")

    query = """
    SELECT DISTINCT
        g.game_id,
        g.winner,
        mafioso.model_name as mafioso_model,
        villager.model_name as villager_model,
        detective.model_name as detective_model
    FROM games g
    JOIN benchmark b ON g.game_id = b.game_id
    JOIN game_players gp_m ON g.game_id = gp_m.game_id AND gp_m.role = 'mafioso'
    JOIN players mafioso ON gp_m.player_id = mafioso.player_id
    JOIN game_players gp_v ON g.game_id = gp_v.game_id AND gp_v.role = 'villager'
    JOIN players villager ON gp_v.player_id = villager.player_id
    JOIN game_players gp_d ON g.game_id = gp_d.game_id AND gp_d.role = 'detective'
    JOIN players detective ON gp_d.player_id = detective.player_id
    WHERE g.winner IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"📊 Loaded {len(df)} LLM games")

    return df

def get_human_games_data():
    """Get human games data and map to model format."""

    if not HUMAN_DB_PATH.exists():
        print("⚠️  No human database found, skipping human data")
        return pd.DataFrame()

    conn = sqlite3.connect(HUMAN_DB_PATH)

    print("\n🔍 Loading human games...")

    query = """
    SELECT
        game_id,
        winner,
        human_role,
        background_name,
        mafioso_name,
        detective_name,
        villager_name
    FROM games
    WHERE winner IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if len(df) == 0:
        print("⚠️  No human games found")
        return pd.DataFrame()

    # Map character names to roles and determine AI models
    # The human plays one role, AIs play the others from background_name

    games_list = []

    for _, row in df.iterrows():
        human_role = row['human_role']
        background = row['background_name']
        winner = 'mafia' if row['winner'] == 'EVIL' else 'town'

        # Determine which models are playing
        # Background format: "gpt-4.1-mini" or "gpt-5-mini" etc.
        # We need to figure out detective and villager models

        if human_role == 'mafioso':
            # Human is mafioso, AIs are detective and villager
            # We need to determine which AI is which from the background
            mafioso_model = 'human'
            # For now, assume background is the detective model
            # and we don't have villager info - skip these games for now
            # We need more info about the game setup
            continue  # Skip for now until we understand the setup better

        elif human_role == 'villager':
            # Human is villager (detect capability)
            # Mafioso and detective are AIs from background
            villager_model = 'human'
            # Need to know which AI is mafioso vs detective
            continue  # Skip for now

    # Actually, let me re-examine the database to understand the setup better
    # The background_name tells us which AI models were used
    # But we need to map character names to models

    print(f"⚠️  Human game integration requires more info about AI role assignments")
    print(f"   Found {len(df)} human games but need to map character names to models")

    return pd.DataFrame()  # Return empty for now

def get_human_games_simple():
    """
    Get human games with simplified assumptions:
    - When human is mafioso: we're testing human deceive capability
    - When human is villager: we're testing human detect capability
    - When human is detective: we're testing human disclose capability
    - We'll aggregate across all AI opponents
    - Includes both new and old database (old = all mafioso games)
    """

    games_summary = []

    # Load from new database (has human_role column)
    if HUMAN_DB_PATH.exists():
        conn = sqlite3.connect(HUMAN_DB_PATH)
        query = """
        SELECT
            human_role,
            winner,
            COUNT(*) as game_count,
            SUM(CASE WHEN winner = 'EVIL' THEN 1 ELSE 0 END) as mafia_wins
        FROM games
        WHERE winner IS NOT NULL
        GROUP BY human_role
        """
        df_new = pd.read_sql_query(query, conn)
        conn.close()

        if len(df_new) > 0:
            games_summary.append(df_new)

    # Load from old database (all mafioso games)
    if HUMAN_OLD_DB_PATH.exists():
        conn = sqlite3.connect(HUMAN_OLD_DB_PATH)
        query = """
        SELECT
            COUNT(*) as game_count,
            SUM(CASE WHEN winner = 'EVIL' THEN 1 ELSE 0 END) as mafia_wins
        FROM games
        WHERE winner IS NOT NULL
        """
        result = conn.execute(query).fetchone()
        conn.close()

        if result and result[0] > 0:
            # All old games are mafioso games
            df_old = pd.DataFrame([{
                'human_role': 'mafioso',
                'game_count': result[0],
                'mafia_wins': result[1]
            }])
            games_summary.append(df_old)

    if not games_summary:
        print("⚠️  No human games found")
        return pd.DataFrame()

    # Combine and aggregate
    df_all = pd.concat(games_summary, ignore_index=True)

    # Aggregate by role
    df = df_all.groupby('human_role').agg({
        'game_count': 'sum',
        'mafia_wins': 'sum'
    }).reset_index()

    print("\n🔍 Loading human games...")
    print(f"📊 Loaded {df['game_count'].sum()} total human games")
    for _, row in df.iterrows():
        print(f"   Role: {row['human_role']}, Games: {row['game_count']}, Mafia wins: {row['mafia_wins']}")

    return df

def aggregate_game_outcomes(df):
    """Aggregate games by (mafioso, villager, detective) configuration."""
    print("\n🔄 Aggregating games by configuration...")

    # Add binary outcome
    df['mafia_win'] = (df['winner'] == 'mafia').astype(int)

    grouped = df.groupby(['mafioso_model', 'villager_model', 'detective_model']).agg({
        'mafia_win': ['sum', 'count']
    }).reset_index()

    # Flatten column names
    grouped.columns = ['mafioso', 'villager', 'detective', 'wins', 'total']

    print(f"📊 Found {len(grouped)} unique configurations")
    print(f"   Average games per configuration: {grouped['total'].mean():.1f}")

    return grouped

def create_synthetic_human_games(human_summary_df, llm_df):
    """
    Create synthetic game records for humans by pairing with average AI opponents.

    Strategy:
    - For human deceive: pair with average AI detective and average AI villager
    - For human detect: pair with average AI mafioso and average AI detective
    - For human disclose: pair with average AI mafioso and average AI villager
    """

    synthetic_games = []

    # Get most common AI models to use as "average opponents"
    all_models = set(llm_df['mafioso_model'].unique()) | \
                 set(llm_df['villager_model'].unique()) | \
                 set(llm_df['detective_model'].unique())

    # Use GPT-4.1 Mini as the "average" opponent (it's middle-tier)
    avg_model = 'gpt-4.1-mini'

    for _, row in human_summary_df.iterrows():
        role = row['human_role']
        total_games = row['game_count']
        mafia_wins = row['mafia_wins']
        town_wins = total_games - mafia_wins

        if role == 'mafioso':
            # Human deceive capability
            # Create a pseudo-configuration: human vs avg_model (villager) vs avg_model (detective)
            synthetic_games.append({
                'mafioso_model': 'human',
                'villager_model': avg_model,
                'detective_model': avg_model,
                'winner': 'mafia',
                'game_count': mafia_wins
            })
            synthetic_games.append({
                'mafioso_model': 'human',
                'villager_model': avg_model,
                'detective_model': avg_model,
                'winner': 'town',
                'game_count': town_wins
            })

        elif role == 'villager':
            # Human detect capability
            # Create a pseudo-configuration: avg_model (mafioso) vs human (villager) vs avg_model (detective)
            synthetic_games.append({
                'mafioso_model': avg_model,
                'villager_model': 'human',
                'detective_model': avg_model,
                'winner': 'mafia',
                'game_count': mafia_wins
            })
            synthetic_games.append({
                'mafioso_model': avg_model,
                'villager_model': 'human',
                'detective_model': avg_model,
                'winner': 'town',
                'game_count': town_wins
            })

        elif role == 'detective':
            # Human disclose capability
            # Create a pseudo-configuration: avg_model (mafioso) vs avg_model (villager) vs human (detective)
            synthetic_games.append({
                'mafioso_model': avg_model,
                'villager_model': avg_model,
                'detective_model': 'human',
                'winner': 'mafia',
                'game_count': mafia_wins
            })
            synthetic_games.append({
                'mafioso_model': avg_model,
                'villager_model': avg_model,
                'detective_model': 'human',
                'winner': 'town',
                'game_count': town_wins
            })

    return pd.DataFrame(synthetic_games)

def combine_llm_and_human_data():
    """Combine LLM and human game data into single aggregated format."""

    # Get LLM games
    llm_df = get_llm_games_data()

    # Get human summary
    human_summary = get_human_games_simple()

    if len(human_summary) == 0:
        print("\n⚠️  No human data available, using LLM-only model")
        llm_df['mafia_win'] = (llm_df['winner'] == 'mafia').astype(int)
        return aggregate_game_outcomes(llm_df)

    # Create synthetic human game records
    print("\n🔄 Creating synthetic human game configurations...")
    human_games = create_synthetic_human_games(human_summary, llm_df)

    # Expand synthetic records into individual game rows
    human_expanded = []
    for _, row in human_games.iterrows():
        for _ in range(row['game_count']):
            human_expanded.append({
                'mafioso_model': row['mafioso_model'],
                'villager_model': row['villager_model'],
                'detective_model': row['detective_model'],
                'winner': row['winner']
            })

    human_df = pd.DataFrame(human_expanded)

    print(f"   Created {len(human_df)} human game records")
    print(f"   Human as mafioso: {(human_df['mafioso_model'] == 'human').sum()} games")
    print(f"   Human as villager: {(human_df['villager_model'] == 'human').sum()} games")
    print(f"   Human as detective: {(human_df['detective_model'] == 'human').sum()} games")

    # Combine
    combined_df = pd.concat([llm_df[['mafioso_model', 'villager_model', 'detective_model', 'winner']],
                             human_df], ignore_index=True)

    print(f"\n📊 Combined dataset: {len(combined_df)} total games")
    print(f"   LLM games: {len(llm_df)}")
    print(f"   Human games: {len(human_df)}")

    # Aggregate
    combined_df['mafia_win'] = (combined_df['winner'] == 'mafia').astype(int)
    return aggregate_game_outcomes(combined_df)

def run_theoretical_model_pymc(data_df):
    """Run theoretical model using PyMC with humans included."""
    print("\n🔬 Running theoretical model with PyMC (including humans)...")

    # Get unique models (including 'human')
    all_models = sorted(set(data_df['mafioso'].unique()) |
                       set(data_df['villager'].unique()) |
                       set(data_df['detective'].unique()))

    n_models = len(all_models)
    print(f"   Models: {n_models} (including humans)")
    print(f"   Model list: {[get_display_name(m) for m in all_models]}")

    # Create indices
    model_idx = {model: i for i, model in enumerate(all_models)}

    # Prepare data
    mafioso_indices = [model_idx[m] for m in data_df['mafioso']]
    villager_indices = [model_idx[v] for v in data_df['villager']]
    detective_indices = [model_idx[d] for d in data_df['detective']]
    wins = data_df['wins'].values
    totals = data_df['total'].values
    n_obs = len(data_df)

    print(f"   Observations: {n_obs}")
    print(f"   Total games: {totals.sum()}")

    with pm.Model() as model:
        # Priors
        a = pm.Normal('a', mu=0, sigma=2, shape=n_models)  # deceive
        b = pm.Normal('b', mu=0, sigma=2, shape=n_models)  # disclose
        c = pm.Normal('c', mu=0, sigma=2, shape=n_models)  # detect

        # Identifiability constraint
        pm.Potential('c_sum_positive', pm.math.switch(pm.math.sum(c) > 0, 0, -np.inf))

        # Theoretical model
        logit_p = c[villager_indices] * (a[mafioso_indices] - b[detective_indices])

        # Likelihood
        y = pm.Binomial('y', n=totals, logit_p=logit_p, observed=wins)

        # Sample
        print("   🔄 Sampling from posterior...")
        trace = pm.sample(2000, tune=1000, chains=2, cores=1,
                         target_accept=0.95, random_seed=42,
                         progressbar=True)

        # Convergence diagnostics
        print("\n   📊 Convergence diagnostics:")
        rhat = az.rhat(trace)
        max_rhat_a = float(rhat['a'].max())
        max_rhat_b = float(rhat['b'].max())
        max_rhat_c = float(rhat['c'].max())
        print(f"   Max R-hat: a={max_rhat_a:.4f}, b={max_rhat_b:.4f}, c={max_rhat_c:.4f}")

    # Extract posterior samples
    a_samples = trace.posterior['a'].values.reshape(-1, n_models)
    b_samples = trace.posterior['b'].values.reshape(-1, n_models)
    c_samples = trace.posterior['c'].values.reshape(-1, n_models)

    # Remove additive gauge freedom by enforcing mean(m) = 0
    mean_shift = a_samples.mean(axis=1, keepdims=True)
    a_samples = a_samples - mean_shift
    b_samples = b_samples - mean_shift

    # Rescale so that mean(v) = 1
    detect_mean_raw = np.mean(c_samples, axis=0)
    scale_factor = np.mean(detect_mean_raw)

    m_samples = a_samples * scale_factor
    d_samples = b_samples * scale_factor
    v_samples = c_samples / scale_factor

    deceive_mean = np.mean(m_samples, axis=0)
    deceive_std = np.std(m_samples, axis=0)

    disclose_mean = np.mean(d_samples, axis=0)
    disclose_std = np.std(d_samples, axis=0)

    detect_mean = np.mean(v_samples, axis=0)
    detect_std = np.std(v_samples, axis=0)

    print(f"\n✅ Sampling complete!")
    print(f"   Deceive (m): [{deceive_mean.min():.3f}, {deceive_mean.max():.3f}]")
    print(f"   Disclose (d): [{disclose_mean.min():.3f}, {disclose_mean.max():.3f}]")
    print(f"   Detect (v): [{detect_mean.min():.3f}, {detect_mean.max():.3f}], mean = {detect_mean.mean():.3f}")

    return all_models, deceive_mean, deceive_std, detect_mean, detect_std, disclose_mean, disclose_std

def create_score_plot(models, scores, errors, capability_name, output_suffix="with_humans"):
    """Create score plot with humans highlighted."""

    plt.ioff()

    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })

    # Sort by score (descending, so best at top)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    sorted_errors = [errors[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Reverse y_positions
    y_positions = range(len(sorted_models) - 1, -1, -1)

    # Color: highlight human in soft pastel green, others based on sign
    colors = []
    for model, score in zip(sorted_models, sorted_scores):
        if model == 'human':
            colors.append('#A8D5BA')  # Soft pastel green for human
        elif score > 0:
            colors.append('#2E86AB')  # Blue for positive
        else:
            colors.append('#E63946')  # Red for negative

    # Create bars
    ax.barh(y_positions, sorted_scores, xerr=sorted_errors,
           color=colors, alpha=0.7, height=0.7,
           error_kw={'capsize': 3, 'capthick': 1.5, 'elinewidth': 1.5})

    # Add value labels
    for i, (score, error) in enumerate(zip(sorted_scores, sorted_errors)):
        x_pos = score + error if score > 0 else score - error
        x_pos = x_pos + (0.1 if score > 0 else -0.1)
        label = f'{score:.2f}'
        ax.text(x_pos, y_positions[i], label,
               va='center',
               ha='left' if score > 0 else 'right',
               fontsize=11,
               color='#333333')

    # Model names with special formatting for human
    display_models = [get_display_name(m) for m in sorted_models]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(display_models)

    # Bold human label
    ytick_labels = ax.get_yticklabels()
    for i, (label, model) in enumerate(zip(ytick_labels, sorted_models)):
        if model == 'human':
            label.set_weight('bold')

    # Labels
    behavior_labels = {
        'deceive': 'Deception Parameter (m)',
        'detect': 'Detection Parameter (v)',
        'disclose': 'Disclosure Parameter (d)'
    }
    xlabel = behavior_labels.get(capability_name.lower(), 'Parameter Value')
    ax.set_xlabel(xlabel, fontsize=12)

    # X-axis limits
    max_val = max([s + e for s, e in zip(sorted_scores, sorted_errors)])
    min_val = min([s - e for s, e in zip(sorted_scores, sorted_errors)])
    padding = (max_val - min_val) * 0.25
    ax.set_xlim(min_val - padding, max_val + padding)

    # Reference line at 0
    if min_val <= 0 <= max_val:
        ax.axvline(x=0, color='#333333', alpha=0.5, linewidth=1.5, linestyle='-', zorder=0)

    # Grid and spines
    ax.grid(True, alpha=0.2, axis='x', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    plt.tight_layout()

    # Save
    capability_clean = capability_name.lower()
    filename = RESULTS_DIR / f"scores_{capability_clean}_{output_suffix}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"   📊 Plot saved: {filename}")
    return filename

def create_combined_plot_with_humans(models, deceive_mean, deceive_std, detect_mean, detect_std, disclose_mean, disclose_std):
    """Create a single figure with three subplots side by side, with humans highlighted in pastel green"""

    print("\n📊 Creating combined plot with humans...")

    plt.ioff()

    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    # Sort models alphabetically
    display_models = [get_display_name(m) for m in models]
    sorted_indices = np.argsort(display_models)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_display = [display_models[i] for i in sorted_indices]
    n_models = len(models)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)

    capabilities = [
        ('Deceive', deceive_mean, deceive_std, 'Deception Parameter (m)'),
        ('Detect', detect_mean, detect_std, 'Detection Parameter (v)'),
        ('Disclose', disclose_mean, disclose_std, 'Disclosure Parameter (d)')
    ]

    y_positions = range(n_models)

    for idx, (ax, (cap_name, means, stds, xlabel)) in enumerate(zip(axes, capabilities)):
        # Get sorted scores and errors
        scores = [means[i] for i in sorted_indices]
        errors = [stds[i] for i in sorted_indices]

        # Color bars: soft pastel green for humans, blue/red for others
        colors = []
        for model, score in zip(sorted_models, scores):
            if model == 'human':
                colors.append('#A8D5BA')  # Soft pastel green/mint
            elif score > 0:
                colors.append('#2E86AB')  # Blue for positive
            else:
                colors.append('#E63946')  # Red for negative

        # Create horizontal bars
        bars = ax.barh(y_positions, scores, xerr=errors,
                color=colors, alpha=0.7, height=0.7,
                error_kw={'capsize': 3, 'capthick': 1.5, 'elinewidth': 1.5})

        # Add value labels on the bars
        for i, (score, error) in enumerate(zip(scores, errors)):
            x_pos = score + error if score > 0 else score - error
            x_pos = x_pos + (0.1 if score > 0 else -0.1)
            label = f'{score:.2f}'
            ax.text(x_pos, i, label,
                   va='center',
                   ha='left' if score > 0 else 'right',
                   fontsize=10,
                   color='#333333')

        # Set x-axis label
        ax.set_xlabel(xlabel, fontsize=13)

        # Set x-axis limits with padding
        max_val = max([s + e for s, e in zip(scores, errors)])
        min_val = min([s - e for s, e in zip(scores, errors)])
        padding = (max_val - min_val) * 0.25
        x_min = min_val - padding
        x_max = max_val + padding
        ax.set_xlim(x_min, x_max)

        # Add vertical line at 0
        if x_min <= 0 <= x_max:
            ax.axvline(x=0, color='#333333', alpha=0.5, linewidth=1.5, linestyle='-', zorder=0)

        # Grid
        ax.grid(True, alpha=0.2, axis='x', linewidth=0.5)
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.spines['bottom'].set_linewidth(0.5)

        if idx == 0:
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['left'].set_linewidth(0.5)
        else:
            ax.spines['left'].set_visible(False)

        # Set y-axis ticks and limits for all subplots
        ax.set_yticks(y_positions)
        ax.set_ylim(-0.5, n_models - 0.5)

        # Only show y-axis labels on the leftmost plot
        if idx == 0:
            ax.set_yticklabels(sorted_display, fontsize=12)
            ax.tick_params(axis='y', which='both', length=0, pad=5)

            # Bold human label
            ytick_labels = ax.get_yticklabels()
            for i, (label, model) in enumerate(zip(ytick_labels, sorted_models)):
                if model == 'human':
                    label.set_weight('bold')
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', left=False)

    plt.tight_layout(pad=1.5)

    # Save plot
    filename = RESULTS_DIR / "scores_combined_with_humans.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"   📊 Combined plot saved: {filename}")
    return filename

def main():
    """Main function."""

    print("=" * 70)
    print("THEORETICAL MODEL WITH HUMANS")
    print("=" * 70)

    if not HAS_PYMC:
        raise ImportError("PyMC required: pip install pymc arviz")

    # Combine data
    data_df = combine_llm_and_human_data()

    # Run model
    models, deceive_mean, deceive_std, detect_mean, detect_std, disclose_mean, disclose_std = \
        run_theoretical_model_pymc(data_df)

    # Create plots for all three capabilities
    print("\n" + "=" * 70)
    print("CREATING PLOTS WITH HUMANS")
    print("=" * 70)

    print("\n📊 Creating Deceive plot...")
    create_score_plot(models, deceive_mean, deceive_std, 'Deceive', 'with_humans')

    print("\n📊 Creating Detect plot...")
    create_score_plot(models, detect_mean, detect_std, 'Detect', 'with_humans')

    print("\n📊 Creating Disclose plot...")
    create_score_plot(models, disclose_mean, disclose_std, 'Disclose', 'with_humans')

    print("\n📊 Creating combined plot...")
    create_combined_plot_with_humans(models, deceive_mean, deceive_std, detect_mean, detect_std, disclose_mean, disclose_std)

    # Save CSV
    display_models = [get_display_name(m) for m in models]
    results_df = pd.DataFrame({
        'Model': display_models,
        'Deceive': deceive_mean,
        'Deceive_Error': deceive_std,
        'Detect': detect_mean,
        'Detect_Error': detect_std,
        'Disclose': disclose_mean,
        'Disclose_Error': disclose_std
    })

    output_file = RESULTS_DIR / "scores_with_humans.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved: {output_file}")

    # Show human results
    human_row = results_df[results_df['Model'] == 'Human']
    if len(human_row) > 0:
        print("\n" + "=" * 70)
        print("HUMAN RESULTS")
        print("=" * 70)
        print(f"Deceive:  {human_row.iloc[0]['Deceive']:.3f} ± {human_row.iloc[0]['Deceive_Error']:.3f}")
        print(f"Detect:   {human_row.iloc[0]['Detect']:.3f} ± {human_row.iloc[0]['Detect_Error']:.3f}")
        print(f"Disclose: {human_row.iloc[0]['Disclose']:.3f} ± {human_row.iloc[0]['Disclose_Error']:.3f}")

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
