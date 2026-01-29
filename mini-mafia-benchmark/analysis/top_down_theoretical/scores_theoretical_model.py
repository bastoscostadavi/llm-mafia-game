#!/usr/bin/env python3
"""
Theoretical Model for Mini-Mafia Benchmark Results

Tests the theoretical model:
    logit(p_ijk) = c_k × (a_i - b_j)

where:
- p_ijk = probability mafia wins when mafioso=i, villager=k, detective=j
- a_i = deceive capability of model i
- b_j = disclose capability of model j
- c_k = detect capability of model k

Uses the 14,000 unique games from the benchmark (filtering duplicates).
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
import sys
from pathlib import Path

# Set up paths
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "top_down_theoretical"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(__file__).parent.parent.parent / "database" / "mini_mafia.db"

# For Bayesian inference (required)
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    # Will raise error in main function with install instructions

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
        'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf': 'Llama 3.1 8B Instruct'
    }
    return display_names.get(model_name, model_name)

def get_unique_games_data():
    """
    Get unique games data from benchmark table.

    Each unique game has a unique (mafioso, villager, detective) configuration.
    We avoid counting the same game multiple times when it appears in different
    capability rows of the benchmark table.
    """
    conn = sqlite3.connect(DB_PATH)

    print("🔍 Loading unique games from benchmark table...")

    # Get all games with their role assignments by joining through game_players
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

    print(f"📊 Loaded {len(df)} unique games")
    print(f"   Unique mafiosos: {df['mafioso_model'].nunique()}")
    print(f"   Unique villagers: {df['villager_model'].nunique()}")
    print(f"   Unique detectives: {df['detective_model'].nunique()}")

    # Add binary outcome (1 = mafia wins, 0 = town wins)
    df['mafia_win'] = (df['winner'] == 'mafia').astype(int)

    print(f"   Mafia wins: {df['mafia_win'].sum()} ({100*df['mafia_win'].mean():.1f}%)")
    print(f"   Town wins: {(1-df['mafia_win']).sum()} ({100*(1-df['mafia_win'].mean()):.1f}%)")

    return df

def aggregate_game_outcomes(df):
    """
    Aggregate games by (mafioso, villager, detective) configuration.

    Returns:
        DataFrame with columns: mafioso, villager, detective, wins, total
    """
    print("\n🔄 Aggregating games by configuration...")

    grouped = df.groupby(['mafioso_model', 'villager_model', 'detective_model']).agg({
        'mafia_win': ['sum', 'count']
    }).reset_index()

    # Flatten column names
    grouped.columns = ['mafioso', 'villager', 'detective', 'wins', 'total']

    print(f"📊 Found {len(grouped)} unique (mafioso, villager, detective) configurations")
    print(f"   Average games per configuration: {grouped['total'].mean():.1f}")
    print(f"   Min games: {grouped['total'].min()}, Max games: {grouped['total'].max()}")

    return grouped

def run_theoretical_model_pymc(data_df):
    """
    Run theoretical model using PyMC:
        logit(p_ijk) = c_k × (a_i - b_j)

    where i=mafioso, k=villager, j=detective
    """
    print("\n🔬 Running theoretical model with PyMC...")

    # Get unique models
    all_models = sorted(set(data_df['mafioso'].unique()) |
                       set(data_df['villager'].unique()) |
                       set(data_df['detective'].unique()))

    n_models = len(all_models)
    print(f"   Models: {n_models}")

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
        # Priors for capabilities
        # a_i: deceive capability (higher = better at deception)
        a = pm.Normal('a', mu=0, sigma=2, shape=n_models)

        # b_j: disclose capability (higher = better at disclosure, hurts mafia)
        b = pm.Normal('b', mu=0, sigma=2, shape=n_models)

        # c_k: detect capability (positive = correct detection, negative = anti-detection)
        # Allow negative values - some models might systematically vote wrong!
        c = pm.Normal('c', mu=0, sigma=2, shape=n_models)

        # Fix identifiability by constraining sum(c) > 0
        # This ensures we don't have the global sign flip ambiguity
        # Most models should detect correctly, so sum should be positive
        pm.Potential('c_sum_positive', pm.math.switch(pm.math.sum(c) > 0, 0, -np.inf))

        # Theoretical model: logit(p_ijk) = c_k × (a_i - b_j)
        logit_p = c[villager_indices] * (a[mafioso_indices] - b[detective_indices])

        # Likelihood
        y = pm.Binomial('y', n=totals, logit_p=logit_p, observed=wins)

        # Sample
        print("   🔄 Sampling from posterior (this may take a few minutes)...")
        trace = pm.sample(2000, tune=1000, chains=2, cores=1,
                         target_accept=0.95, random_seed=42,
                         progressbar=True)

        # Check convergence diagnostics and persist them for reporting
        diag_summary = az.summary(trace, var_names=['a', 'b', 'c'], round_to=4)
        diag_path = RESULTS_DIR / "mcmc_diagnostics.csv"
        diag_summary.to_csv(diag_path)

        max_rhat = float(diag_summary['r_hat'].max())
        min_bulk_ess = float(diag_summary['ess_bulk'].min())
        min_tail_ess = float(diag_summary['ess_tail'].min())

        print("\n   📊 Convergence diagnostics:")
        print(f"   Max R-hat: {max_rhat:.4f}")
        print(f"   Min ESS (bulk/tail): {min_bulk_ess:.0f} / {min_tail_ess:.0f}")
        print(f"   Diagnostics saved to: {diag_path}")
        if max_rhat > 1.01:
            print("   ⚠️  Some R-hat values > 1.01 (chain convergence issues)")
        else:
            print("   ✓ All R-hat values < 1.01 (good convergence)")

    # Extract results
    a_samples = trace.posterior['a'].values.reshape(-1, n_models)
    b_samples = trace.posterior['b'].values.reshape(-1, n_models)
    c_samples = trace.posterior['c'].values.reshape(-1, n_models)

    # Fix additive gauge freedom by forcing mean(m)=0 while preserving
    # the gaps m_i - d_j.
    mean_shift = a_samples.mean(axis=1, keepdims=True)
    a_samples = a_samples - mean_shift
    b_samples = b_samples - mean_shift

    # Compute statistics - keep raw values from the model
    # m_i: deceive capability (logit scale)
    # d_j: disclose capability (logit scale)
    # v_k: detect capability (can be positive or negative)

    deceive_mean_raw = np.mean(a_samples, axis=0)
    disclose_mean_raw = np.mean(b_samples, axis=0)
    detect_mean_raw = np.mean(c_samples, axis=0)

    # Rescale to fix identifiability: set mean(v) = 1
    # If we scale v by λ, we must scale m and d by 1/λ
    v_mean_value = np.mean(detect_mean_raw)
    scale_factor = v_mean_value  # λ = mean(v)

    print(f"\n✅ Sampling complete!")
    print(f"   Raw parameter ranges (before rescaling):")
    print(f"   Deceive (m): [{deceive_mean_raw.min():.3f}, {deceive_mean_raw.max():.3f}]")
    print(f"   Disclose (d): [{disclose_mean_raw.min():.3f}, {disclose_mean_raw.max():.3f}]")
    print(f"   Detect (v): [{detect_mean_raw.min():.3f}, {detect_mean_raw.max():.3f}], mean = {v_mean_value:.3f}")

    # Apply rescaling to all samples (for proper uncertainty propagation)
    # If we divide v by λ, we must multiply m and d by λ to preserve predictions
    m_samples_rescaled = a_samples * scale_factor
    d_samples_rescaled = b_samples * scale_factor
    v_samples_rescaled = c_samples / scale_factor

    # Compute rescaled statistics
    deceive_mean = np.mean(m_samples_rescaled, axis=0)
    deceive_std = np.std(m_samples_rescaled, axis=0)

    disclose_mean = np.mean(d_samples_rescaled, axis=0)
    disclose_std = np.std(d_samples_rescaled, axis=0)

    detect_mean = np.mean(v_samples_rescaled, axis=0)
    detect_std = np.std(v_samples_rescaled, axis=0)

    print(f"\n   Rescaled parameters (mean(v) = 1):")
    print(f"   Deceive (m): [{deceive_mean.min():.3f}, {deceive_mean.max():.3f}]")
    print(f"   Disclose (d): [{disclose_mean.min():.3f}, {disclose_mean.max():.3f}]")
    print(f"   Detect (v): [{detect_mean.min():.3f}, {detect_mean.max():.3f}], mean = {np.mean(detect_mean):.3f}")

    # Print uncertainty details for debugging
    print(f"\n   Parameter uncertainties (std):")
    for i, model in enumerate(all_models):
        print(f"   {get_display_name(model):30s} - m:{deceive_std[i]:.3f}  d:{disclose_std[i]:.3f}  v:{detect_std[i]:.3f}")

    return all_models, deceive_mean, deceive_std, detect_mean, detect_std, disclose_mean, disclose_std

def create_score_plot(models, scores, errors, capability_name, output_suffix="theoretical"):
    """Create minimalist score plot for theoretical model parameters"""

    # Use non-interactive backend
    plt.ioff()

    # Set font size
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })

    # Sort by score (descending, so best performers at top)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_models = [models[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    sorted_errors = [errors[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(9, 6))

    # Reverse y_positions so best performers appear at top of plot
    y_positions = range(len(sorted_models) - 1, -1, -1)

    # Color bars based on positive/negative values
    colors = ['#2E86AB' if s > 0 else '#E63946' for s in sorted_scores]

    # Create bars
    bars = ax.barh(y_positions, sorted_scores, xerr=sorted_errors,
                   color=colors, alpha=0.7, height=0.7,
                   error_kw={'capsize': 3, 'capthick': 1.5, 'elinewidth': 1.5})

    # Add value labels on the bars
    for i, (score, error) in enumerate(zip(sorted_scores, sorted_errors)):
        # Position text at the end of the bar (including error bar)
        x_pos = score + error if score > 0 else score - error
        # Add minimal padding
        x_pos = x_pos + (0.1 if score > 0 else -0.1)

        # Format the value
        label = f'{score:.2f}'

        # Add text
        ax.text(x_pos, y_positions[i], label,
               va='center',
               ha='left' if score > 0 else 'right',
               fontsize=11,
               color='#333333')

    # Add model names on y-axis
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_models)

    # Set axis labels
    behavior_labels = {
        'deceive': 'Deception Parameter (m)',
        'detect': 'Detection Parameter (v)',
        'disclose': 'Disclosure Parameter (d)'
    }
    xlabel = behavior_labels.get(capability_name.lower(), 'Parameter Value')
    ax.set_xlabel(xlabel, fontsize=12)

    # Set x-axis limits with padding
    max_val = max([s + e for s, e in zip(sorted_scores, sorted_errors)])
    min_val = min([s - e for s, e in zip(sorted_scores, sorted_errors)])
    padding = (max_val - min_val) * 0.25
    x_min = min_val - padding
    x_max = max_val + padding
    ax.set_xlim(x_min, x_max)

    # Add vertical line at 0 (reference point)
    if x_min <= 0 <= x_max:
        ax.axvline(x=0, color='#333333', alpha=0.5, linewidth=1.5, linestyle='-', zorder=0)

    # Minimal grid
    ax.grid(True, alpha=0.2, axis='x', linewidth=0.5)
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    plt.tight_layout()

    # Save plot
    capability_clean = capability_name.lower()
    filename = RESULTS_DIR / f"scores_{capability_clean}_{output_suffix}.png"

    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"   📊 Plot saved as {filename}")
    return filename

def create_theoretical_scores():
    """Main function to create theoretical model scores"""

    print("=" * 70)
    print("THEORETICAL MODEL: logit(p_ijk) = c_k × (a_i - b_j)")
    print("=" * 70)

    # Step 1: Get unique games data
    games_df = get_unique_games_data()

    # Step 2: Aggregate by configuration
    data_df = aggregate_game_outcomes(games_df)

    # Step 3: Run theoretical model (requires PyMC)
    if not HAS_PYMC:
        raise ImportError(
            "PyMC is required for Bayesian inference. Install with: pip install pymc arviz"
        )

    models, deceive_mean, deceive_std, detect_mean, detect_std, disclose_mean, disclose_std = \
        run_theoretical_model_pymc(data_df)

    # Create results DataFrame
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

    # Save results
    output_file = RESULTS_DIR / "scores_theoretical.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\n💾 Theoretical model scores saved to: {output_file}")
    print(f"   Shape: {results_df.shape}")
    print("\n📋 Sample results:")
    print(results_df.head(10))

    return results_df

def create_combined_theoretical_plot(df):
    """Create a single figure with three subplots side by side, models sorted alphabetically"""

    print("\n📊 Creating combined theoretical model plot...")

    # Use non-interactive backend
    plt.ioff()

    # Set font size
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    # Sort models alphabetically
    sorted_df = df.sort_values('Model', ascending=True).reset_index(drop=True)
    models = sorted_df['Model'].tolist()
    n_models = len(models)

    # Create figure with 3 subplots - don't share y-axis so we can control labels independently
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)

    capabilities = ['Deceive', 'Detect', 'Disclose']
    capability_labels = {
        'Deceive': 'Deception Parameter (m)',
        'Detect': 'Detection Parameter (v)',
        'Disclose': 'Disclosure Parameter (d)'
    }

    y_positions = range(n_models)

    for idx, (ax, capability) in enumerate(zip(axes, capabilities)):
        scores = sorted_df[capability].tolist()
        errors = sorted_df[f'{capability}_Error'].tolist()

        # Color bars based on positive/negative values
        colors = ['#2E86AB' if s > 0 else '#E63946' for s in scores]

        # Create horizontal bars
        bars = ax.barh(y_positions, scores, xerr=errors,
                color=colors, alpha=0.7, height=0.7,
                error_kw={'capsize': 3, 'capthick': 1.5, 'elinewidth': 1.5})

        # Add value labels on the bars
        for i, (score, error) in enumerate(zip(scores, errors)):
            # Position text at the end of the bar (including error bar)
            x_pos = score + error if score > 0 else score - error
            # Add some padding
            x_pos = x_pos + (0.3 if score > 0 else -0.3)

            # Format the value
            label = f'{score:.2f}'

            # Add text
            ax.text(x_pos, i, label,
                   va='center',
                   ha='left' if score > 0 else 'right',
                   fontsize=10,
                   color='#333333')

        # Set x-axis label instead of title
        ax.set_xlabel(capability_labels[capability], fontsize=13)

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
            # Keep left spine visible for first plot
            ax.spines['left'].set_color('#CCCCCC')
            ax.spines['left'].set_linewidth(0.5)
        else:
            # Hide left spine for other plots
            ax.spines['left'].set_visible(False)

        # Set y-axis ticks and limits for all subplots
        ax.set_yticks(y_positions)
        ax.set_ylim(-0.5, n_models - 0.5)

        # Only show y-axis labels on the leftmost plot
        if idx == 0:
            ax.set_yticklabels(models, fontsize=12)
            ax.tick_params(axis='y', which='both', length=0, pad=5)  # Remove tick marks, add padding
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', left=False)  # Hide ticks on other plots

    plt.tight_layout(pad=1.5)

    # Save plot
    filename = RESULTS_DIR / "scores_theoretical_combined.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"   📊 Combined plot saved as {filename}")
    return filename

def create_all_theoretical_plots(df):
    """Create all three theoretical model capability plots"""

    print("\n" + "=" * 70)
    print("CREATING THEORETICAL MODEL PLOTS")
    print("=" * 70)

    capabilities = ['Deceive', 'Detect', 'Disclose']
    created_files = []

    for capability in capabilities:
        print(f"\n📊 Creating {capability} plot...")
        models = df['Model'].tolist()
        scores = df[capability].tolist()
        errors = df[f'{capability}_Error'].tolist()

        filename = create_score_plot(models, scores, errors, capability, "theoretical")
        created_files.append(filename)

        # Print some stats
        print(f"   Range: [{min(scores):.3f}, {max(scores):.3f}]")
        print(f"   Best: {models[np.argmax(scores)]} ({max(scores):.3f})")
        print(f"   Worst: {models[np.argmin(scores)]} ({min(scores):.3f})")

    # Create combined plot
    print(f"\n📊 Creating combined plot...")
    combined_file = create_combined_theoretical_plot(df)
    created_files.append(combined_file)

    print(f"\n✅ Created {len(created_files)} theoretical model plots:")
    for filename in created_files:
        print(f"   ✓ {filename}")

    return created_files

if __name__ == "__main__":
    # Create scores
    scores_df = create_theoretical_scores()

    # Create plots
    create_all_theoretical_plots(scores_df)

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
