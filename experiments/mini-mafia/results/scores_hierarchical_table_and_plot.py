#!/usr/bin/env python3
"""
Hierarchical Bayesian Analysis for Mini-Mafia Benchmark Results

Uses all 15,000 entries from the benchmark table to implement the hierarchical
Bayesian model described in the paper:
- k_ib ~ Binomial(n_ib, p_ib)
- logit(p_ib) = z_i + β_b (where α_i = exp(z_i))
- z_i ~ Normal(μ_z, σ_z²)  # Model abilities
- β_b ~ Normal(0, σ_β²)    # Background effects

Generates hierarchical Bayesian scores for comparison with simplified methodology.
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
import math

# For Bayesian inference
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    print("⚠️  PyMC not available. Using approximate hierarchical Bayesian method.")
    HAS_PYMC = False

warnings.filterwarnings('ignore')

def get_company_color(model_name):
    """Get color based on company, matching existing plots"""
    company_colors = {
        'Claude Opus 4.1': '#FF6B35',      # Anthropic orange
        'Claude Sonnet 4': '#FF6B35',      # Anthropic orange
        'DeepSeek V3.1': '#4ECDC4',        # DeepSeek teal
        'GPT-4.1 Mini': '#45B7D1',        # OpenAI blue
        'GPT-5 Mini': '#45B7D1',          # OpenAI blue
        'Gemini 2.5 Flash Lite': '#96CEB4', # Google green
        'Grok 3 Mini': '#FFEAA7',         # X yellow
        'Llama 3.1 8B Instruct': '#DDA0DD', # Meta purple
        'Mistral 7B Instruct': '#FFB6C1',  # Mistral pink
        'Qwen2.5 7B Instruct': '#F0E68C'   # Alibaba khaki
    }
    return company_colors.get(model_name, '#666666')

def create_hierarchical_bayesian_plot(models, scores, errors, capability_name):
    """Create hierarchical Bayesian score plot matching existing format exactly"""

    # Use non-interactive backend
    plt.ioff()

    # Set font size - exact match
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
        'figure.titlesize': 24
    })

    # Sort by score (ascending, so best performers at top)
    sorted_indices = np.argsort(scores)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_scores = [scores[i] for i in sorted_indices]
    sorted_errors = [errors[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(14, 7))

    y_positions = range(len(sorted_models))

    # Create bars - use single color like original
    bars = ax.barh(y_positions, sorted_scores, xerr=sorted_errors,
                   color='#E74C3C', alpha=0.8, height=0.6,
                   error_kw={'capsize': 5, 'capthick': 2})

    # Add model names - exact positioning like original
    for i, (model, score, error) in enumerate(zip(sorted_models, sorted_scores, sorted_errors)):
        x_pos = max(score + error + 0.05, 0.05)
        ax.text(x_pos, i, f'{model}',
                ha='left', va='center', fontweight='bold', fontsize=24)

    # Set axis labels - match behavior labels
    behavior_labels = {
        'deceive': 'Deceive Score',
        'detect': 'Detect Score',
        'disclose': 'Disclose Score'
    }
    xlabel = behavior_labels.get(capability_name.lower(), 'Model Score')
    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold')
    ax.set_yticks([])

    # Set data-driven x-axis limits with padding - exact match
    max_val = max([s + e for s, e in zip(sorted_scores, sorted_errors)])
    min_val = min([s - e for s, e in zip(sorted_scores, sorted_errors)])
    padding = (max_val - min_val) * 0.1
    x_min = 0
    x_max = max_val + padding
    ax.set_xlim(x_min, x_max)

    # Add vertical line at exp(0) = 1 - exact match
    if x_min <= 1 <= x_max:  # Only show if 1 is in the visible range
        ax.axvline(x=1, color='gray', alpha=0.7, linewidth=2, linestyle='--')

    # Add grid - exact match
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)

    # Hide spines - exact match
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()

    # Save plot with new naming convention: scores_capability_hierarchical.png
    capability_clean = capability_name.lower()
    filename = f"scores_{capability_clean}_hierarchical.png"

    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Hierarchical Bayesian plot saved as {filename}")
    return filename

def get_display_name(model_name):
    """Convert internal model names to display names"""
    display_names = {
        'claude_opus_4_1': 'Claude Opus 4.1',
        'claude_sonnet_4': 'Claude Sonnet 4',
        'deepseek_v3_1': 'DeepSeek V3.1',
        'gemini_2_5_flash_lite': 'Gemini 2.5 Flash Lite',
        'grok_3_mini': 'Grok 3 Mini',
        'gpt_4_1_mini': 'GPT-4.1 Mini',
        'gpt_5_mini': 'GPT-5 Mini',
        'mistral_7b_instruct': 'Mistral 7B Instruct',
        'qwen2_5_7b_instruct': 'Qwen2.5 7B Instruct',
        'llama_3_1_8b_instruct': 'Llama 3.1 8B Instruct'
    }
    return display_names.get(model_name, model_name)

def get_benchmark_data():
    """Get win/total counts from benchmark table using all 15,000 entries"""
    db_path = "../database/mini_mafia.db"
    conn = sqlite3.connect(db_path)

    print("🔍 Loading benchmark data for hierarchical Bayesian analysis...")

    # Query to get all benchmark entries with game outcomes
    query = """
    SELECT
        b.capability,
        b.background,
        b.target,
        g.winner,
        COUNT(*) as total_games
    FROM benchmark b
    JOIN games g ON b.game_id = g.game_id
    WHERE g.winner IS NOT NULL
    GROUP BY b.capability, b.background, b.target, g.winner
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Process the data to get win counts per capability-background-target combination
    results = {}
    capabilities = ['deceive', 'detect', 'disclose']

    for capability in capabilities:
        cap_data = df[df['capability'] == capability]
        win_counts = defaultdict(lambda: defaultdict(int))

        for _, row in cap_data.iterrows():
            background = row['background']
            target = row['target']
            winner = row['winner']
            games = row['total_games']

            # Determine if this is a win for the target model
            if (capability == 'deceive' and winner == 'mafia') or \
               (capability in ['detect', 'disclose'] and winner == 'town'):
                win_counts[(target, background)]['wins'] += games

            win_counts[(target, background)]['total'] += games

        results[capability] = win_counts

    print(f"📊 Loaded data for {len(capabilities)} capabilities")
    return results

def run_hierarchical_bayesian_pymc(win_counts, capability):
    """Run hierarchical Bayesian analysis using PyMC"""

    # Prepare data
    data_list = []
    for (target, background), counts in win_counts.items():
        data_list.append({
            'target': target,
            'background': background,
            'wins': counts['wins'],
            'total': counts['total']
        })

    data_df = pd.DataFrame(data_list)

    # Create indices
    targets = sorted(data_df['target'].unique())
    backgrounds = sorted(data_df['background'].unique())

    target_idx = {target: i for i, target in enumerate(targets)}
    background_idx = {bg: i for i, bg in enumerate(backgrounds)}

    # Prepare arrays
    n_targets = len(targets)
    n_backgrounds = len(backgrounds)
    n_obs = len(data_df)

    target_indices = [target_idx[target] for target in data_df['target']]
    background_indices = [background_idx[bg] for bg in data_df['background']]
    wins = data_df['wins'].values
    totals = data_df['total'].values

    print(f"  📈 Running PyMC model for {capability} with {n_targets} targets, {n_backgrounds} backgrounds, {n_obs} observations")

    with pm.Model() as model:
        # Hyperpriors
        mu_z = pm.Normal('mu_z', mu=0, sigma=2)
        sigma_z = pm.HalfNormal('sigma_z', sigma=1)
        sigma_beta = pm.HalfNormal('sigma_beta', sigma=1)

        # Model abilities (z_i)
        z = pm.Normal('z', mu=mu_z, sigma=sigma_z, shape=n_targets)

        # Background effects (β_j) with sum-to-zero constraint
        beta_raw = pm.Normal('beta_raw', mu=0, sigma=sigma_beta, shape=n_backgrounds-1)
        beta = pm.Deterministic('beta', pm.math.concatenate([beta_raw, [-pm.math.sum(beta_raw)]]))

        # Linear predictor
        logit_p = z[target_indices] + beta[background_indices]

        # Likelihood
        y = pm.Binomial('y', n=totals, logit_p=logit_p, observed=wins)

        # Sample
        trace = pm.sample(2000, tune=1000, chains=2, cores=1,
                         target_accept=0.90, random_seed=42)

    # Extract results
    z_samples = trace.posterior['z'].values.reshape(-1, n_targets)
    alpha_samples = np.exp(z_samples)  # α_i = exp(z_i)

    # Compute statistics
    alpha_mean = np.mean(alpha_samples, axis=0)
    alpha_std = np.std(alpha_samples, axis=0)

    return targets, alpha_mean, alpha_std

def run_approximate_hierarchical_bayesian(win_counts, capability):
    """Approximate hierarchical Bayesian analysis without PyMC"""

    print(f"  📈 Running approximate hierarchical Bayesian for {capability}")

    # Prepare data
    data_list = []
    for (target, background), counts in win_counts.items():
        if counts['total'] > 0:
            win_rate = counts['wins'] / counts['total']
            data_list.append({
                'target': target,
                'background': background,
                'win_rate': win_rate,
                'wins': counts['wins'],
                'total': counts['total']
            })

    data_df = pd.DataFrame(data_list)

    # Get targets and backgrounds
    targets = sorted(data_df['target'].unique())
    backgrounds = sorted(data_df['background'].unique())

    # Simple hierarchical approximation
    # 1. Compute overall mean and variance
    overall_logit = np.mean([np.log(max(0.01, min(0.99, wr)) / (1 - max(0.01, min(0.99, wr))))
                            for wr in data_df['win_rate']])

    # 2. Estimate target effects
    target_effects = {}
    target_errors = {}

    for target in targets:
        target_data = data_df[data_df['target'] == target]

        if len(target_data) > 0:
            # Weighted average of logits
            logits = []
            weights = []
            for _, row in target_data.iterrows():
                p = max(0.01, min(0.99, row['win_rate']))
                logit = np.log(p / (1 - p))
                weight = row['total']  # Weight by sample size
                logits.append(logit)
                weights.append(weight)

            if weights:
                weighted_logit = np.average(logits, weights=weights)
                target_effects[target] = weighted_logit
                # Approximate standard error
                target_errors[target] = 1.0 / np.sqrt(sum(weights))
            else:
                target_effects[target] = overall_logit
                target_errors[target] = 1.0
        else:
            target_effects[target] = overall_logit
            target_errors[target] = 1.0

    # Convert to alpha scale
    alpha_mean = np.array([np.exp(target_effects[target]) for target in targets])
    alpha_std = np.array([target_errors[target] * alpha_mean[i] for i, target in enumerate(targets)])

    return targets, alpha_mean, alpha_std

def create_hierarchical_bayesian_scores():
    """Create hierarchical Bayesian scores for all capabilities"""

    print("🔄 Creating hierarchical Bayesian scores...")

    # Get benchmark data
    benchmark_data = get_benchmark_data()

    all_results = {}

    for capability in ['deceive', 'detect', 'disclose']:
        print(f"\n📊 Processing {capability} capability...")

        win_counts = benchmark_data[capability]

        if HAS_PYMC:
            try:
                targets, alpha_mean, alpha_std = run_hierarchical_bayesian_pymc(win_counts, capability)
            except Exception as e:
                print(f"  ⚠️  PyMC failed, using approximate method: {e}")
                targets, alpha_mean, alpha_std = run_approximate_hierarchical_bayesian(win_counts, capability)
        else:
            targets, alpha_mean, alpha_std = run_approximate_hierarchical_bayesian(win_counts, capability)

        # Store results
        results_df = pd.DataFrame({
            'Model': [get_display_name(target) for target in targets],
            f'{capability.capitalize()}': alpha_mean,
            f'{capability.capitalize()}_Error': alpha_std
        })

        all_results[capability] = results_df

        print(f"  ✅ Completed {capability}: α range [{alpha_mean.min():.3f}, {alpha_mean.max():.3f}]")

    # Combine results into final table
    final_df = all_results['deceive'][['Model']].copy()

    for capability in ['deceive', 'detect', 'disclose']:
        df = all_results[capability]
        final_df[capability.capitalize()] = df[capability.capitalize()]
        final_df[f'{capability.capitalize()}_Error'] = df[f'{capability.capitalize()}_Error']

    # Save results
    output_file = "scores_hierarchical.csv"
    final_df.to_csv(output_file, index=False)

    print(f"\n✅ Hierarchical Bayesian scores saved to: {output_file}")
    print(f"   Shape: {final_df.shape}")
    print("\nSample results:")
    print(final_df.head())

    return final_df

def create_all_hierarchical_bayesian_plots(df):
    """Create all three hierarchical Bayesian capability plots from scores DataFrame"""

    print("\n🔄 Creating hierarchical Bayesian score plots...")

    capabilities = ['Deceive', 'Detect', 'Disclose']
    created_files = []

    for capability in capabilities:
        models = df['Model'].tolist()
        scores = df[capability].tolist()
        errors = df[f'{capability}_Error'].tolist()

        filename = create_hierarchical_bayesian_plot(models, scores, errors, capability)
        created_files.append(filename)

        # Print some stats
        print(f"  {capability}: α range [{min(scores):.3f}, {max(scores):.3f}]")

    print(f"\n✅ Created {len(created_files)} hierarchical Bayesian score plots:")
    for filename in created_files:
        print(f"   {filename}")

    return created_files

if __name__ == "__main__":
    # Create scores
    scores_df = create_hierarchical_bayesian_scores()

    # Create plots
    create_all_hierarchical_bayesian_plots(scores_df)