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

import numpy as np
import pandas as pd
from collections import defaultdict
import math
from utils import get_display_name, create_horizontal_bar_plot, _MODEL_DISPLAY_NAMES

# For Bayesian inference
import pymc as pm
import arviz as az



def create_hierarchical_bayesian_plot(models, scores, errors, capability_name):
    """Create hierarchical Bayesian score plot matching existing format exactly"""

    # Set axis labels - match behavior labels
    behavior_labels = {
        'deceive': 'Deceive Score',
        'detect': 'Detect Score',
        'disclose': 'Disclose Score'
    }
    xlabel = behavior_labels.get(capability_name.lower(), 'Model Score')

    # Save plot with hierarchical naming convention
    capability_clean = capability_name.lower()
    filename = f"scores_{capability_clean}_hierarchical.png"

    # Use unified plotting function
    create_horizontal_bar_plot(
        models=models,
        values=scores,
        errors=errors,
        xlabel=xlabel,
        filename=filename,
        color='#E74C3C',
        sort_ascending=True,
        show_reference_line=True
    )

    print(f"Hierarchical Bayesian plot saved as {filename}")
    return filename


def load_win_counts_from_csv():
    """Load win counts from the existing CSV file instead of querying database"""
    print("🔍 Loading win counts from CSV...")

    # Read the win counts CSV
    df = pd.read_csv('win_counts.csv')

    # Process into the format expected by PyMC function
    results = {}

    for capability in ['Deceive', 'Detect', 'Disclose']:
        cap_data = df[df['capability'] == capability]
        win_counts = defaultdict(lambda: defaultdict(int))

        for _, row in cap_data.iterrows():
            target = row['model']
            # Convert display names back to internal names for consistency
            target_internal = None
            for internal, display in _MODEL_DISPLAY_NAMES.items():
                if display == target:
                    target_internal = internal
                    break

            if target_internal is None:
                continue  # Skip if we can't map back

            # Process each background column
            for background_display in ['DeepSeek V3.1', 'GPT-4.1 Mini', 'GPT-5 Mini', 'Grok 3 Mini', 'Mistral 7B Instruct']:
                if background_display in row:
                    wins = int(row[background_display])
                    total = 100  # Each combination has 100 games

                    # Convert display name back to internal name
                    background_internal = None
                    for internal, display in _MODEL_DISPLAY_NAMES.items():
                        if display == background_display:
                            background_internal = internal
                            break

                    if background_internal is not None:
                        win_counts[(target_internal, background_internal)]['wins'] = wins
                        win_counts[(target_internal, background_internal)]['total'] = total

        results[capability.lower()] = win_counts

    print(f"📊 Loaded data for {len(results)} capabilities from CSV")
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


def create_hierarchical_bayesian_scores():
    """Create hierarchical Bayesian scores for all capabilities"""

    print("🔄 Creating hierarchical Bayesian scores...")

    # Load data from existing CSV instead of database
    benchmark_data = load_win_counts_from_csv()

    all_results = {}

    for capability in ['deceive', 'detect', 'disclose']:
        print(f"\n📊 Processing {capability} capability...")

        win_counts = benchmark_data[capability]

        targets, alpha_mean, alpha_std = run_hierarchical_bayesian_pymc(win_counts, capability)

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