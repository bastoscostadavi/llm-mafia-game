#!/usr/bin/env python3
"""
Hierarchical Bayesian model fitting the hypothesis: logit(p) = c(a - b)

Estimates intrinsic capabilities {a_i, b_i, c_i} for all models by fitting
to all game data simultaneously:
- a_i: deceive capability
- b_i: disclose capability
- c_i: detect capability

Fits all three datasets (deceive, detect, disclose) in one unified model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'results'))
from utils import create_horizontal_bar_plot, _MODEL_DISPLAY_NAMES

# For Bayesian inference
import pymc as pm
import arviz as az

def load_all_game_data():
    """Load all game data from CSV"""
    print("Loading game data...")

    # Load win counts
    df = pd.read_csv('../results/win_counts.csv')

    # Get unique models
    all_models = sorted(df['model'].unique())

    # Create model index mapping
    model_to_idx = {model: i for i, model in enumerate(all_models)}

    print(f"  Found {len(all_models)} models")

    # Prepare data for each capability
    data = {}

    for capability in ['Deceive', 'Detect', 'Disclose']:
        cap_data = df[df['capability'] == capability].copy()

        # Get background columns
        background_cols = [col for col in cap_data.columns
                          if col not in ['capability', 'model']]

        # Prepare arrays
        target_indices = []
        background_indices = []
        wins = []
        totals = []

        for _, row in cap_data.iterrows():
            target_model = row['model']
            target_idx = model_to_idx[target_model]

            for background in background_cols:
                background_idx = model_to_idx[background]
                n_wins = int(row[background])
                n_total = 100

                target_indices.append(target_idx)
                background_indices.append(background_idx)
                wins.append(n_wins)
                totals.append(n_total)

        data[capability] = {
            'target_indices': np.array(target_indices),
            'background_indices': np.array(background_indices),
            'wins': np.array(wins),
            'totals': np.array(totals)
        }

        print(f"  {capability}: {len(wins)} games")

    return data, all_models, model_to_idx

def fit_hierarchical_model(data, all_models):
    """
    Fit hierarchical Bayesian model with hypothesis: logit(p) = c(a - b)

    All three capabilities fitted simultaneously to ensure consistency.
    """
    print("\nFitting hierarchical Bayesian model...")
    print("Hypothesis: logit(p) = c(a - b)")

    n_models = len(all_models)

    # Extract data
    deceive_data = data['Deceive']
    detect_data = data['Detect']
    disclose_data = data['Disclose']

    with pm.Model() as model:
        print("\nBuilding PyMC model...")

        # Priors for intrinsic capabilities
        # These are on the log-odds scale (z-scores)
        a = pm.Normal('a_deceive', mu=0, sigma=2, shape=n_models)
        b = pm.Normal('b_disclose', mu=0, sigma=2, shape=n_models)
        c = pm.Normal('c_detect', mu=0, sigma=2, shape=n_models)

        # DECEIVE experiments: mafioso i vs background j (detective + villager)
        # logit(p_mafia) = c[j] * (a[i] - b[j])
        logit_p_deceive = c[deceive_data['background_indices']] * (
            a[deceive_data['target_indices']] - b[deceive_data['background_indices']]
        )
        y_deceive = pm.Binomial('y_deceive',
                                n=deceive_data['totals'],
                                logit_p=logit_p_deceive,
                                observed=deceive_data['wins'])

        # DETECT experiments: villager i vs background j (mafioso + detective)
        # Data has TOWN wins, but formula predicts MAFIA wins
        # logit(p_mafia) = c[i] * (a[j] - b[j])
        # logit(p_town) = -logit(p_mafia) = c[i] * (b[j] - a[j])
        logit_p_detect = c[detect_data['target_indices']] * (
            b[detect_data['background_indices']] - a[detect_data['background_indices']]
        )
        y_detect = pm.Binomial('y_detect',
                               n=detect_data['totals'],
                               logit_p=logit_p_detect,
                               observed=detect_data['wins'])

        # DISCLOSE experiments: detective i vs background j (mafioso + villager)
        # Data has TOWN wins
        # logit(p_mafia) = c[j] * (a[j] - b[i])
        # logit(p_town) = c[j] * (b[i] - a[j])
        logit_p_disclose = c[disclose_data['background_indices']] * (
            b[disclose_data['target_indices']] - a[disclose_data['background_indices']]
        )
        y_disclose = pm.Binomial('y_disclose',
                                 n=disclose_data['totals'],
                                 logit_p=logit_p_disclose,
                                 observed=disclose_data['wins'])

        print("Sampling from posterior...")
        print("  This may take a few minutes...")

        # Sample
        trace = pm.sample(
            2000,
            tune=1000,
            chains=4,
            cores=4,
            target_accept=0.95,
            random_seed=42,
            progressbar=True
        )

    print("\nSampling complete!")

    return trace, model

def extract_capabilities(trace, all_models):
    """Extract capability estimates from trace"""
    print("\nExtracting capability estimates...")

    # Get posterior samples
    a_samples = trace.posterior['a_deceive'].values.reshape(-1, len(all_models))
    b_samples = trace.posterior['b_disclose'].values.reshape(-1, len(all_models))
    c_samples = trace.posterior['c_detect'].values.reshape(-1, len(all_models))

    # Convert to exponential scale (α = exp(z))
    alpha_a = np.exp(a_samples)
    alpha_b = np.exp(b_samples)
    alpha_c = np.exp(c_samples)

    # Compute statistics
    results = []

    for i, model in enumerate(all_models):
        results.append({
            'Model': model,
            'Deceive': np.mean(alpha_a[:, i]),
            'Deceive_Error': np.std(alpha_a[:, i]),
            'Detect': np.mean(alpha_c[:, i]),
            'Detect_Error': np.std(alpha_c[:, i]),
            'Disclose': np.mean(alpha_b[:, i]),
            'Disclose_Error': np.std(alpha_b[:, i])
        })

    results_df = pd.DataFrame(results)

    # Print summary
    print("\nCapability estimates (α = exp(z)):")
    print(results_df.to_string(index=False))

    return results_df

def create_score_plots(results_df):
    """Create score plots for all three capabilities"""
    print("\nCreating score plots...")

    capabilities = ['Deceive', 'Detect', 'Disclose']
    filenames = []

    for capability in capabilities:
        models = results_df['Model'].tolist()
        scores = results_df[capability].tolist()
        errors = results_df[f'{capability}_Error'].tolist()

        filename = f'../results/scores_{capability.lower()}_hypothesis.png'

        create_horizontal_bar_plot(
            models=models,
            values=scores,
            errors=errors,
            xlabel=f'{capability} Score',
            filename=filename,
            color='#E74C3C',
            sort_ascending=True,
            show_reference_line=True
        )

        filenames.append(filename)
        print(f"  {capability}: {filename}")

    return filenames

def main():
    print("="*60)
    print("HIERARCHICAL BAYESIAN MODEL: logit(p) = c(a - b)")
    print("="*60)

    # Load data
    data, all_models, model_to_idx = load_all_game_data()

    total_games = sum(len(data[cap]['wins']) for cap in data)
    print(f"\nTotal games: {total_games}")

    # Fit model
    trace, model = fit_hierarchical_model(data, all_models)

    # Extract capabilities
    results_df = extract_capabilities(trace, all_models)

    # Save results
    output_file = '../results/scores_hypothesis_hierarchical.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Create plots
    filenames = create_score_plots(results_df)

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nGenerated {len(filenames)} plots:")
    for fn in filenames:
        print(f"  {fn}")

    return results_df, trace

if __name__ == '__main__':
    results_df, trace = main()
