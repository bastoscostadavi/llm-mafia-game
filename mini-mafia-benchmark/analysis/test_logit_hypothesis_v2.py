#!/usr/bin/env python3
"""
Test the hypothesis: logit(p) = c(a - b)

FIXED VERSION: Addresses identifiability problem by normalizing parameters.

where:
- p = mafia win rate
- a = deceive capability (mafioso)
- b = disclose capability (detective)
- c = detect capability (villager)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit  # inverse logit
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def extract_data_from_tables():
    """
    Extract the raw experimental data from Tables in the article.
    Returns win counts k_ib out of n=100 games.
    """

    # Table from article: Deceive data (mafioso wins)
    # Rows = mafioso model, Columns = background (detective+villager)
    deceive_data = {
        'Claude Opus 4.1': {'DeepSeek V3.1': 23, 'GPT-4.1 Mini': 57, 'GPT-5 Mini': 43, 'Grok 3 Mini': 15, 'Mistral 7B Instruct': 48},
        'Claude Sonnet 4': {'DeepSeek V3.1': 17, 'GPT-4.1 Mini': 55, 'GPT-5 Mini': 37, 'Grok 3 Mini': 19, 'Mistral 7B Instruct': 50},
        'DeepSeek V3.1': {'DeepSeek V3.1': 30, 'GPT-4.1 Mini': 58, 'GPT-5 Mini': 40, 'Grok 3 Mini': 20, 'Mistral 7B Instruct': 51},
        'Gemini 2.5 Flash Lite': {'DeepSeek V3.1': 24, 'GPT-4.1 Mini': 48, 'GPT-5 Mini': 34, 'Grok 3 Mini': 7, 'Mistral 7B Instruct': 50},
        'GPT-4.1 Mini': {'DeepSeek V3.1': 11, 'GPT-4.1 Mini': 37, 'GPT-5 Mini': 26, 'Grok 3 Mini': 7, 'Mistral 7B Instruct': 45},
        'GPT-5 Mini': {'DeepSeek V3.1': 17, 'GPT-4.1 Mini': 34, 'GPT-5 Mini': 35, 'Grok 3 Mini': 7, 'Mistral 7B Instruct': 49},
        'Grok 3 Mini': {'DeepSeek V3.1': 14, 'GPT-4.1 Mini': 47, 'GPT-5 Mini': 49, 'Grok 3 Mini': 8, 'Mistral 7B Instruct': 59},
        'Llama 3.1 8B Instruct': {'DeepSeek V3.1': 12, 'GPT-4.1 Mini': 20, 'GPT-5 Mini': 30, 'Grok 3 Mini': 1, 'Mistral 7B Instruct': 35},
        'Mistral 7B Instruct': {'DeepSeek V3.1': 11, 'GPT-4.1 Mini': 36, 'GPT-5 Mini': 30, 'Grok 3 Mini': 2, 'Mistral 7B Instruct': 54},
        'Qwen2.5 7B Instruct': {'DeepSeek V3.1': 3, 'GPT-4.1 Mini': 25, 'GPT-5 Mini': 30, 'Grok 3 Mini': 2, 'Mistral 7B Instruct': 45},
    }

    # Table from article: Detect data (town wins, need to convert to mafia wins)
    detect_data = {
        'Claude Opus 4.1': {'DeepSeek V3.1': 62, 'GPT-4.1 Mini': 82, 'GPT-5 Mini': 93, 'Grok 3 Mini': 78, 'Mistral 7B Instruct': 43},
        'Claude Sonnet 4': {'DeepSeek V3.1': 62, 'GPT-4.1 Mini': 54, 'GPT-5 Mini': 70, 'Grok 3 Mini': 44, 'Mistral 7B Instruct': 42},
        'DeepSeek V3.1': {'DeepSeek V3.1': 70, 'GPT-4.1 Mini': 73, 'GPT-5 Mini': 87, 'Grok 3 Mini': 75, 'Mistral 7B Instruct': 52},
        'Gemini 2.5 Flash Lite': {'DeepSeek V3.1': 58, 'GPT-4.1 Mini': 60, 'GPT-5 Mini': 71, 'Grok 3 Mini': 65, 'Mistral 7B Instruct': 59},
        'GPT-4.1 Mini': {'DeepSeek V3.1': 49, 'GPT-4.1 Mini': 63, 'GPT-5 Mini': 69, 'Grok 3 Mini': 68, 'Mistral 7B Instruct': 46},
        'GPT-5 Mini': {'DeepSeek V3.1': 57, 'GPT-4.1 Mini': 56, 'GPT-5 Mini': 65, 'Grok 3 Mini': 66, 'Mistral 7B Instruct': 45},
        'Grok 3 Mini': {'DeepSeek V3.1': 76, 'GPT-4.1 Mini': 82, 'GPT-5 Mini': 98, 'Grok 3 Mini': 92, 'Mistral 7B Instruct': 70},
        'Llama 3.1 8B Instruct': {'DeepSeek V3.1': 53, 'GPT-4.1 Mini': 63, 'GPT-5 Mini': 64, 'Grok 3 Mini': 52, 'Mistral 7B Instruct': 48},
        'Mistral 7B Instruct': {'DeepSeek V3.1': 52, 'GPT-4.1 Mini': 63, 'GPT-5 Mini': 65, 'Grok 3 Mini': 52, 'Mistral 7B Instruct': 46},
        'Qwen2.5 7B Instruct': {'DeepSeek V3.1': 50, 'GPT-4.1 Mini': 70, 'GPT-5 Mini': 64, 'Grok 3 Mini': 54, 'Mistral 7B Instruct': 50},
    }

    # Table from article: Disclose data (town wins, need to convert to mafia wins)
    disclose_data = {
        'Claude Opus 4.1': {'DeepSeek V3.1': 59, 'GPT-4.1 Mini': 62, 'GPT-5 Mini': 76, 'Grok 3 Mini': 97, 'Mistral 7B Instruct': 66},
        'Claude Sonnet 4': {'DeepSeek V3.1': 62, 'GPT-4.1 Mini': 69, 'GPT-5 Mini': 64, 'Grok 3 Mini': 96, 'Mistral 7B Instruct': 63},
        'DeepSeek V3.1': {'DeepSeek V3.1': 70, 'GPT-4.1 Mini': 64, 'GPT-5 Mini': 65, 'Grok 3 Mini': 98, 'Mistral 7B Instruct': 57},
        'Gemini 2.5 Flash Lite': {'DeepSeek V3.1': 50, 'GPT-4.1 Mini': 52, 'GPT-5 Mini': 61, 'Grok 3 Mini': 97, 'Mistral 7B Instruct': 57},
        'GPT-4.1 Mini': {'DeepSeek V3.1': 60, 'GPT-4.1 Mini': 63, 'GPT-5 Mini': 66, 'Grok 3 Mini': 88, 'Mistral 7B Instruct': 62},
        'GPT-5 Mini': {'DeepSeek V3.1': 69, 'GPT-4.1 Mini': 75, 'GPT-5 Mini': 72, 'Grok 3 Mini': 95, 'Mistral 7B Instruct': 59},
        'Grok 3 Mini': {'DeepSeek V3.1': 64, 'GPT-4.1 Mini': 79, 'GPT-5 Mini': 75, 'Grok 3 Mini': 92, 'Mistral 7B Instruct': 54},
        'Llama 3.1 8B Instruct': {'DeepSeek V3.1': 17, 'GPT-4.1 Mini': 19, 'GPT-5 Mini': 23, 'Grok 3 Mini': 28, 'Mistral 7B Instruct': 26},
        'Mistral 7B Instruct': {'DeepSeek V3.1': 45, 'GPT-4.1 Mini': 54, 'GPT-5 Mini': 45, 'Grok 3 Mini': 62, 'Mistral 7B Instruct': 46},
        'Qwen2.5 7B Instruct': {'DeepSeek V3.1': 28, 'GPT-4.1 Mini': 45, 'GPT-5 Mini': 57, 'Grok 3 Mini': 75, 'Mistral 7B Instruct': 46},
    }

    return deceive_data, detect_data, disclose_data


def create_dataset_for_fitting():
    """
    Create dataset in format needed for optimization.
    """

    deceive_data, detect_data, disclose_data = extract_data_from_tables()

    models = list(deceive_data.keys())

    dataset = []

    # Deceive experiments
    for mafioso in deceive_data:
        for background in deceive_data[mafioso]:
            wins = deceive_data[mafioso][background]
            dataset.append({
                'mafioso': mafioso,
                'detective': background,
                'villager': background,
                'mafia_wins': wins,
                'n_games': 100,
                'experiment_type': 'deceive'
            })

    # Detect experiments (convert town wins to mafia wins)
    for villager in detect_data:
        for background in detect_data[villager]:
            town_wins = detect_data[villager][background]
            mafia_wins = 100 - town_wins
            dataset.append({
                'mafioso': background,
                'detective': background,
                'villager': villager,
                'mafia_wins': mafia_wins,
                'n_games': 100,
                'experiment_type': 'detect'
            })

    # Disclose experiments (convert town wins to mafia wins)
    for detective in disclose_data:
        for background in disclose_data[detective]:
            town_wins = disclose_data[detective][background]
            mafia_wins = 100 - town_wins
            dataset.append({
                'mafioso': background,
                'detective': detective,
                'villager': background,
                'mafia_wins': mafia_wins,
                'n_games': 100,
                'experiment_type': 'disclose'
            })

    df = pd.DataFrame(dataset)

    # Create model index
    all_models = sorted(models)
    model_to_idx = {model: i for i, model in enumerate(all_models)}

    df['mafioso_idx'] = df['mafioso'].map(model_to_idx)
    df['detective_idx'] = df['detective'].map(model_to_idx)
    df['villager_idx'] = df['villager'].map(model_to_idx)

    # Observed win rate
    df['p_obs'] = df['mafia_wins'] / df['n_games']

    return df, all_models, model_to_idx


def logit(p):
    """Convert probability to logit"""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))


def fit_hypothesis_constrained(df, n_models):
    """
    Fit: logit(p) = c(a - b)

    To fix identifiability, we constrain: sum(c) = n_models (mean c = 1)
    This allows a and b to have natural scale.
    """

    print("\nFitting hypothesis: logit(p) = c(a - b)")
    print("With constraint: mean(c) = 1")
    print("="*60)

    # Initialize with small random values
    np.random.seed(42)
    initial_params = np.random.randn(3 * n_models) * 0.1

    def neg_log_likelihood(params):
        """Negative log-likelihood for binomial data"""

        a = params[0:n_models]
        b = params[n_models:2*n_models]
        c_raw = params[2*n_models:3*n_models]

        # Apply constraint: normalize c to have mean of 1
        c = c_raw - c_raw.mean() + 1.0

        nll = 0

        for _, row in df.iterrows():
            i = row['mafioso_idx']
            j = row['detective_idx']
            k = row['villager_idx']

            # Hypothesis: logit(p) = c[k] * (a[i] - b[j])
            logit_p = c[k] * (a[i] - b[j])

            # Convert to probability
            p = expit(logit_p)
            p = np.clip(p, 1e-10, 1 - 1e-10)

            # Binomial log-likelihood
            wins = row['mafia_wins']
            n = row['n_games']

            nll -= wins * np.log(p) + (n - wins) * np.log(1 - p)

        # Add small L2 regularization to prevent extreme values
        nll += 0.001 * np.sum(params**2)

        return nll

    # Optimize
    print("Optimizing parameters...")
    result = minimize(neg_log_likelihood, initial_params, method='L-BFGS-B')

    if not result.success:
        print(f"Warning: Optimization may not have converged: {result.message}")

    print(f"Final negative log-likelihood: {result.fun:.2f}")

    # Extract and normalize parameters
    a = result.x[0:n_models]
    b = result.x[n_models:2*n_models]
    c_raw = result.x[2*n_models:3*n_models]
    c = c_raw - c_raw.mean() + 1.0

    return a, b, c, result


def compute_predictions(df, a, b, c, all_models):
    """Compute predicted probabilities for each game"""

    predictions = []

    for _, row in df.iterrows():
        i = row['mafioso_idx']
        j = row['detective_idx']
        k = row['villager_idx']

        # Hypothesis
        logit_p = c[k] * (a[i] - b[j])
        p_pred = expit(logit_p)

        predictions.append({
            'mafioso': all_models[i],
            'detective': all_models[j],
            'villager': all_models[k],
            'p_obs': row['p_obs'],
            'p_pred': p_pred,
            'logit_obs': logit(row['p_obs']),
            'logit_pred': logit_p,
            'experiment_type': row['experiment_type'],
            'mafia_wins': row['mafia_wins'],
            'n_games': row['n_games']
        })

    return pd.DataFrame(predictions)


def evaluate_fit(pred_df):
    """Compute goodness of fit statistics"""

    # R-squared on logit scale
    ss_res = np.sum((pred_df['logit_obs'] - pred_df['logit_pred'])**2)
    ss_tot = np.sum((pred_df['logit_obs'] - pred_df['logit_obs'].mean())**2)
    r_squared_logit = 1 - ss_res / ss_tot

    # R-squared on probability scale
    ss_res_p = np.sum((pred_df['p_obs'] - pred_df['p_pred'])**2)
    ss_tot_p = np.sum((pred_df['p_obs'] - pred_df['p_obs'].mean())**2)
    r_squared_p = 1 - ss_res_p / ss_tot_p

    # Correlation
    corr_logit = np.corrcoef(pred_df['logit_obs'], pred_df['logit_pred'])[0, 1]
    corr_p = np.corrcoef(pred_df['p_obs'], pred_df['p_pred'])[0, 1]

    # RMSE
    rmse = np.sqrt(np.mean((pred_df['p_obs'] - pred_df['p_pred'])**2))

    # MAE
    mae = np.mean(np.abs(pred_df['p_obs'] - pred_df['p_pred']))

    return {
        'r_squared_logit': r_squared_logit,
        'r_squared_p': r_squared_p,
        'correlation_logit': corr_logit,
        'correlation_p': corr_p,
        'rmse': rmse,
        'mae': mae,
        'n_points': len(pred_df)
    }


def plot_results(pred_df, stats, all_models, a, b, c):
    """Create visualization of results"""

    fig = plt.figure(figsize=(18, 12))

    # 1. Observed vs Predicted (overall)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(pred_df['logit_pred'], pred_df['logit_obs'], alpha=0.5, s=30)

    min_val = min(pred_df['logit_pred'].min(), pred_df['logit_obs'].min())
    max_val = max(pred_df['logit_pred'].max(), pred_df['logit_obs'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect fit')

    ax1.set_xlabel('Predicted logit(p)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Observed logit(p)', fontsize=12, fontweight='bold')
    ax1.set_title(f"Overall Fit\nR² = {stats['r_squared_logit']:.3f}, r = {stats['correlation_logit']:.3f}",
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2-4. By experiment type
    for idx, exp_type in enumerate(['deceive', 'detect', 'disclose']):
        ax = plt.subplot(2, 3, idx + 2)

        subset = pred_df[pred_df['experiment_type'] == exp_type]
        ax.scatter(subset['logit_pred'], subset['logit_obs'], alpha=0.5, s=30)

        min_val = min(subset['logit_pred'].min(), subset['logit_obs'].min())
        max_val = max(subset['logit_pred'].max(), subset['logit_obs'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

        # Compute R² for this subset
        ss_res = np.sum((subset['logit_obs'] - subset['logit_pred'])**2)
        ss_tot = np.sum((subset['logit_obs'] - subset['logit_obs'].mean())**2)
        r2 = 1 - ss_res / ss_tot

        ax.set_xlabel('Predicted logit(p)', fontsize=10)
        ax.set_ylabel('Observed logit(p)', fontsize=10)
        ax.set_title(f"{exp_type.capitalize()}\nR² = {r2:.3f}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # 5. Estimated capabilities (bar plots) - normalized to exp scale
    ax5 = plt.subplot(2, 3, 5)

    # For display, show on exp scale but keep interpretation
    alpha_a = np.exp(a - a.mean())  # Center around geometric mean
    alpha_b = np.exp(b - b.mean())
    alpha_c = np.exp(c - 1.0)  # c is already normalized to mean 1

    x = np.arange(len(all_models))
    width = 0.25

    ax5.barh(x - width, alpha_a, width, label='Deceive (a)', alpha=0.7)
    ax5.barh(x, alpha_b, width, label='Disclose (b)', alpha=0.7)
    ax5.barh(x + width, alpha_c, width, label='Detect (c)', alpha=0.7)

    ax5.set_yticks(x)
    ax5.set_yticklabels([m[:20] for m in all_models], fontsize=8)
    ax5.set_xlabel('Relative Capability', fontsize=10, fontweight='bold')
    ax5.set_title('Estimated Capabilities', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.axvline(1, color='red', linestyle='--', alpha=0.3, label='Average')
    ax5.grid(True, alpha=0.3, axis='x')

    # 6. Residuals
    ax6 = plt.subplot(2, 3, 6)
    residuals = pred_df['logit_obs'] - pred_df['logit_pred']
    ax6.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax6.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Residual (logit scale)', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax6.set_title(f'Residuals\nMean = {residuals.mean():.3f}, Std = {residuals.std():.3f}',
                  fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    plt.suptitle('Hypothesis Test: logit(p) = c(a - b)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = '../results/hypothesis_test_fixed.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {filename}")
    plt.close()


def create_capability_comparison_plot(a, b, c, all_models):
    """Create a nice comparison plot with the paper's results"""

    # Normalize for interpretability
    alpha_a = np.exp(a - a.mean())
    alpha_b = np.exp(b - b.mean())
    alpha_c = np.exp(c - 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (values, label, title) in enumerate([
        (alpha_a, 'Deceive', 'Deceive Capability (a)'),
        (alpha_c, 'Detect', 'Detect Capability (c)'),
        (alpha_b, 'Disclose', 'Disclose Capability (b)')
    ]):
        ax = axes[idx]

        # Sort by value
        sorted_idx = np.argsort(values)
        sorted_models = [all_models[i] for i in sorted_idx]
        sorted_values = values[sorted_idx]

        y_pos = np.arange(len(sorted_models))

        colors = plt.cm.RdYlGn(sorted_values / sorted_values.max())

        ax.barh(y_pos, sorted_values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_models, fontsize=10)
        ax.set_xlabel('Relative Capability', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axvline(1, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Average')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend()

    plt.suptitle('Estimated Capabilities from logit(p) = c(a - b)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    filename = '../results/capabilities_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Capabilities plot saved to: {filename}")
    plt.close()


def main():
    print("="*60)
    print("TESTING HYPOTHESIS: logit(p) = c(a - b)")
    print("="*60)
    print("\nwhere:")
    print("  p = mafia win probability")
    print("  a = deceive capability (mafioso)")
    print("  b = disclose capability (detective)")
    print("  c = detect capability (villager)")

    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    df, all_models, model_to_idx = create_dataset_for_fitting()
    print(f"\nLoaded {len(df)} game configurations")
    print(f"Models: {len(all_models)}")
    print(f"Deceive experiments: {len(df[df['experiment_type']=='deceive'])}")
    print(f"Detect experiments: {len(df[df['experiment_type']=='detect'])}")
    print(f"Disclose experiments: {len(df[df['experiment_type']=='disclose'])}")

    # Fit hypothesis
    a, b, c, result = fit_hypothesis_constrained(df, len(all_models))

    # Compute predictions
    print("\nComputing predictions...")
    pred_df = compute_predictions(df, a, b, c, all_models)

    # Evaluate fit
    print("\nEvaluating fit...")
    stats = evaluate_fit(pred_df)

    print("\n" + "="*60)
    print("GOODNESS OF FIT")
    print("="*60)
    print(f"R² (logit scale) = {stats['r_squared_logit']:.4f}")
    print(f"R² (probability scale) = {stats['r_squared_p']:.4f}")
    print(f"Correlation (logit) = {stats['correlation_logit']:.4f}")
    print(f"Correlation (probability) = {stats['correlation_p']:.4f}")
    print(f"RMSE (probability scale) = {stats['rmse']:.4f}")
    print(f"MAE (probability scale) = {stats['mae']:.4f}")

    # Display estimated capabilities
    print("\n" + "="*60)
    print("ESTIMATED CAPABILITIES (raw values)")
    print("="*60)

    capability_df = pd.DataFrame({
        'Model': all_models,
        'Deceive (a)': a,
        'Detect (c)': c,
        'Disclose (b)': b,
        'Deceive_exp': np.exp(a - a.mean()),
        'Detect_exp': np.exp(c - 1.0),
        'Disclose_exp': np.exp(b - b.mean()),
    })
    print("\n" + capability_df.to_string(index=False))

    # Save results
    capability_df.to_csv('../results/hypothesis_capabilities_v2.csv', index=False)
    pred_df.to_csv('../results/hypothesis_predictions_v2.csv', index=False)

    # Plot results
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    plot_results(pred_df, stats, all_models, a, b, c)
    create_capability_comparison_plot(a, b, c, all_models)

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("\nKey properties of your hypothesis logit(p) = c(a-b):")
    print("1. When c=0: villager can't detect → p=0.5 (coin flip)")
    print("2. When a=b: deceive equals disclose → p=0.5 (balanced)")
    print("3. c acts as SCALING: higher c amplifies the contest")
    print("\nPotential issues:")
    print("- R² indicates moderate fit, suggesting the model captures some structure")
    print("- But the multiplicative form may be too rigid")
    print("- Consider alternatives like: logit(p) = a - b + f(c)")

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print("\nFiles saved:")
    print("  ../results/hypothesis_capabilities_v2.csv")
    print("  ../results/hypothesis_predictions_v2.csv")
    print("  ../results/hypothesis_test_fixed.png")
    print("  ../results/capabilities_comparison.png")

    return capability_df, pred_df, stats


if __name__ == '__main__':
    capability_df, pred_df, stats = main()
