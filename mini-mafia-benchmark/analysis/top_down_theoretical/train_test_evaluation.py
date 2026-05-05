#!/usr/bin/env python3
"""
Train/Test Evaluation for the Mini-Mafia Theoretical Model

5-fold cross-validation on the 140 aggregated (mafioso, villager, detective)
configurations to test whether logit(p_ijk) = v_k × (m_i − d_j) has
predictive power beyond goodness-of-fit.
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import expit
from scipy.stats import binom as binom_dist
from scipy.optimize import minimize

try:
    import pymc as pm
    import arviz as az
except ImportError:
    raise ImportError("PyMC is required: pip install pymc arviz")

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent.parent / "results" / "top_down_theoretical"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))
from scores_theoretical_model import (
    get_unique_games_data,
    aggregate_game_outcomes,
    get_display_name,
)

warnings.filterwarnings('ignore')


def fit_model_map(data_df: pd.DataFrame, n_restarts: int = 10) -> tuple:
    """
    Fit logit(p_ijk) = v_k × (m_i − d_j) via MAP using scipy L-BFGS-B.

    Reparameterizes v = exp(log_v) so v is always positive, which:
    - Resolves the sign degeneracy (avoids the (m,d,v) ↔ (−m,−d,−v) flip)
    - Keeps the optimizer away from the zero saddle point where NUTS diverges

    Multiple random restarts guard against local optima.

    Returns: (all_models, m_map, zeros, v_map, zeros, d_map, zeros)
    """
    all_models = sorted(
        set(data_df['mafioso']) | set(data_df['villager']) | set(data_df['detective'])
    )
    n = len(all_models)
    idx = {m: i for i, m in enumerate(all_models)}

    maf_idx = np.array([idx[m] for m in data_df['mafioso']])
    vil_idx = np.array([idx[m] for m in data_df['villager']])
    det_idx = np.array([idx[m] for m in data_df['detective']])
    wins   = data_df['wins'].values.astype(float)
    totals = data_df['total'].values.astype(float)
    sigma2 = 4.0  # N(0, 2) prior

    def neg_log_posterior(params):
        a      = params[:n]
        b      = params[n:2*n]
        log_c  = params[2*n:]
        c      = np.exp(log_c)

        logit_p = c[vil_idx] * (a[maf_idx] - b[det_idx])
        p = expit(logit_p)
        eps = 1e-15
        ll = np.sum(wins * np.log(p + eps) + (totals - wins) * np.log(1 - p + eps))
        log_prior = -0.5 / sigma2 * (np.sum(a**2) + np.sum(b**2) + np.sum(log_c**2))
        return -(ll + log_prior)

    best_result, best_val = None, np.inf
    for seed in range(n_restarts):
        rng = np.random.default_rng(seed * 13 + 7)
        # Initialize away from zero: a,b ~ N(0,1), log_c ~ N(0, 0.5)
        x0 = np.concatenate([
            rng.normal(0, 1.0, n),
            rng.normal(0, 1.0, n),
            rng.normal(0, 0.5, n),
        ])
        result = minimize(neg_log_posterior, x0, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-12})
        if result.fun < best_val:
            best_val, best_result = result.fun, result

    params = best_result.x
    a_map  = params[:n]
    b_map  = params[n:2*n]
    c_map  = np.exp(params[2*n:])

    # Fix shift symmetry: E[m] = 0
    shift = a_map.mean()
    a_map -= shift
    b_map -= shift

    # Fix scale symmetry: E[v] = 1
    scale = c_map.mean()
    m_map = a_map * scale
    d_map = b_map * scale
    v_map = c_map / scale

    zeros = np.zeros(n)
    print(f"  ✓  MAP converged (neg-log-posterior: {best_val:.2f})")
    return (all_models, m_map, zeros, v_map, zeros, d_map, zeros)


def get_and_aggregate_data() -> pd.DataFrame:
    games_df = get_unique_games_data()
    return aggregate_game_outcomes(games_df)


def create_kfold_splits(data_df: pd.DataFrame, n_splits: int = 5, random_state: int = 42):
    """
    Split aggregated configs into k folds at the configuration level.
    Verifies all models appear in each training fold across all three roles.

    Returns: list of (train_df, test_df) tuples
    """
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(data_df))
    rng.shuffle(indices)

    folds = np.array_split(indices, n_splits)
    all_models = (
        set(data_df['mafioso']) | set(data_df['villager']) | set(data_df['detective'])
    )

    splits = []
    for fold_idx, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != fold_idx])
        train_df = data_df.iloc[train_idx].reset_index(drop=True)
        test_df = data_df.iloc[test_idx].reset_index(drop=True)

        missing = {
            role: all_models - set(train_df[col])
            for role, col in [('mafioso', 'mafioso'), ('villager', 'villager'), ('detective', 'detective')]
        }
        any_missing = any(len(v) > 0 for v in missing.values())
        if any_missing:
            print(f"  ⚠️  Fold {fold_idx + 1}: models missing from training: {missing}")
        else:
            print(f"  ✓ Fold {fold_idx + 1}: all {len(all_models)} models present in all roles")

        splits.append((train_df, test_df))

    return splits


def run_theoretical_model_train(train_df: pd.DataFrame) -> dict:
    """
    Fit the theoretical model on training data using robust sampler settings.

    Returns: {'m': {model: value}, 'd': {model: value}, 'v': {model: value}}
    """
    all_models, m_mean, m_std, v_mean, v_std, d_mean, d_std = fit_model_map(train_df)
    return {
        'm': dict(zip(all_models, m_mean)),
        'd': dict(zip(all_models, d_mean)),
        'v': dict(zip(all_models, v_mean)),
    }


def predict(params: dict, test_df: pd.DataFrame) -> np.ndarray:
    """
    Predict win probabilities: p̂_ijk = sigmoid(v_k × (m_i − d_j)).
    Configs with unseen models get NaN.
    """
    m, d, v = params['m'], params['d'], params['v']
    preds = np.full(len(test_df), np.nan)

    for i, row in enumerate(test_df.itertuples(index=False)):
        maf, det, vil = row.mafioso, row.detective, row.villager
        if maf in m and det in d and vil in v:
            preds[i] = expit(v[vil] * (m[maf] - d[det]))

    return preds


def compute_metrics(test_df: pd.DataFrame, preds: np.ndarray, baseline_p: float) -> dict:
    """
    Binomial log-likelihood and Brier score for model and baseline.
    baseline_p is computed from training to avoid data leakage.
    """
    valid = ~np.isnan(preds)
    k = test_df['wins'].values[valid]
    n = test_df['total'].values[valid]
    obs_rate = k / n

    eps = 1e-9
    p_hat = np.clip(preds[valid], eps, 1 - eps)
    p_base = np.clip(baseline_p, eps, 1 - eps)

    return {
        'log_likelihood': float(np.sum(binom_dist.logpmf(k, n, p_hat))),
        'brier_score': float(np.mean((p_hat - obs_rate) ** 2)),
        'baseline_log_likelihood': float(np.sum(binom_dist.logpmf(k, n, p_base))),
        'baseline_brier': float(np.mean((p_base - obs_rate) ** 2)),
        'n_test_configs': int(valid.sum()),
        'n_skipped': int((~valid).sum()),
    }


def run_cross_validation(n_splits: int = 5):
    """
    Full k-fold CV: split → train → predict → evaluate.

    Returns: (fold_metrics_df, all_predictions_df)
    """
    print("=" * 70)
    print(f"CROSS-VALIDATION: {n_splits}-fold")
    print("=" * 70)

    data_df = get_and_aggregate_data()
    print(f"\nTotal configurations: {len(data_df)}")
    print(f"\nCreating {n_splits}-fold splits...")
    splits = create_kfold_splits(data_df, n_splits=n_splits)

    fold_records = []
    pred_records = []

    for fold_idx, (train_df, test_df) in enumerate(splits):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{n_splits}  ({len(train_df)} train, {len(test_df)} test configs)")
        print(f"{'='*70}")

        # Baseline: empirical win rate from training set only (no leakage)
        baseline_p = train_df['wins'].sum() / train_df['total'].sum()
        print(f"  Training baseline win rate: {baseline_p:.3f}")

        params = run_theoretical_model_train(train_df)
        preds = predict(params, test_df)
        metrics = compute_metrics(test_df, preds, baseline_p)

        print(f"\n  Fold {fold_idx + 1} results:")
        print(f"  Log-likelihood — Model: {metrics['log_likelihood']:.2f}  |  Baseline: {metrics['baseline_log_likelihood']:.2f}")
        print(f"  Brier score    — Model: {metrics['brier_score']:.4f}  |  Baseline: {metrics['baseline_brier']:.4f}")
        if metrics['n_skipped'] > 0:
            print(f"  ⚠️  Skipped {metrics['n_skipped']} configs (unseen model in test)")

        fold_records.append({
            'fold': fold_idx + 1,
            'n_test': metrics['n_test_configs'],
            'log_likelihood': metrics['log_likelihood'],
            'brier_score': metrics['brier_score'],
            'baseline_log_likelihood': metrics['baseline_log_likelihood'],
            'baseline_brier': metrics['baseline_brier'],
        })

        for i, row in enumerate(test_df.itertuples(index=False)):
            pred_records.append({
                'fold': fold_idx + 1,
                'mafioso': get_display_name(row.mafioso),
                'villager': get_display_name(row.villager),
                'detective': get_display_name(row.detective),
                'predicted_p': preds[i],
                'observed_rate': row.wins / row.total,
                'wins': row.wins,
                'total': row.total,
            })

    fold_df = pd.DataFrame(fold_records)

    # Append mean row
    mean_vals = fold_df.drop(columns=['fold', 'n_test']).mean().to_dict()
    mean_vals['fold'] = 'mean'
    mean_vals['n_test'] = fold_df['n_test'].sum()
    fold_df = pd.concat([fold_df, pd.DataFrame([mean_vals])], ignore_index=True)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(fold_df.to_string(index=False))

    return fold_df, pd.DataFrame(pred_records)


def create_calibration_plot(pred_df: pd.DataFrame, output_path: Path):
    """Predicted win probability vs observed win rate, binned into 10 equal-width buckets."""
    plt.ioff()
    plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})

    valid = pred_df.dropna(subset=['predicted_p'])
    bins = np.linspace(0, 1, 11)

    bin_pred, bin_obs, bin_err, bin_n = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (valid['predicted_p'] >= lo) & (valid['predicted_p'] < hi)
        if mask.sum() == 0:
            continue
        sub = valid[mask]
        bin_pred.append(sub['predicted_p'].mean())
        bin_obs.append(sub['observed_rate'].mean())
        bin_err.append(sub['observed_rate'].std() / np.sqrt(len(sub)) if len(sub) > 1 else 0)
        bin_n.append(len(sub))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1.5, label='Perfect calibration')
    ax.errorbar(bin_pred, bin_obs, yerr=bin_err, fmt='o', color='#2E86AB',
                markersize=8, capsize=4, capthick=1.5, elinewidth=1.5, label='Model')

    for x, y, c in zip(bin_pred, bin_obs, bin_n):
        ax.annotate(f'n={c}', (x, y), textcoords='offset points', xytext=(5, 5),
                    fontsize=9, color='#555555')

    ax.set_xlabel('Predicted win probability (p̂)')
    ax.set_ylabel('Observed win rate')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  📊 Saved: {output_path}")


def create_metrics_summary(fold_df: pd.DataFrame, output_path: Path):
    """Grouped bar chart: model vs baseline for log-likelihood and Brier score per fold."""
    plt.ioff()
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13})

    numeric = fold_df[fold_df['fold'] != 'mean'].copy()
    numeric['fold'] = numeric['fold'].astype(int)
    mean_row = fold_df[fold_df['fold'] == 'mean'].iloc[0]

    x = np.arange(len(numeric))
    w = 0.35
    labels = [f'Fold {i}' for i in numeric['fold']]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, baseline_metric, ylabel, better in zip(
        axes,
        ['log_likelihood', 'brier_score'],
        ['baseline_log_likelihood', 'baseline_brier'],
        ['Binomial log-likelihood (higher is better)', 'Brier score (lower is better)'],
        ['higher', 'lower'],
    ):
        ax.bar(x - w/2, numeric[metric], w, label='Model', color='#2E86AB', alpha=0.8)
        ax.bar(x + w/2, numeric[baseline_metric], w, label='Baseline (mean)', color='#E63946', alpha=0.8)
        ax.axhline(float(mean_row[metric]), color='#2E86AB', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(float(mean_row[baseline_metric]), color='#E63946', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  📊 Saved: {output_path}")


if __name__ == "__main__":
    fold_df, pred_df = run_cross_validation(n_splits=5)

    fold_df.to_csv(RESULTS_DIR / "cv_metrics.csv", index=False)
    pred_df.to_csv(RESULTS_DIR / "cv_predictions.csv", index=False)
    print(f"\n💾 Saved: {RESULTS_DIR / 'cv_metrics.csv'}")
    print(f"💾 Saved: {RESULTS_DIR / 'cv_predictions.csv'}")

    print("\nCreating plots...")
    create_calibration_plot(pred_df, RESULTS_DIR / "calibration_plot.png")
    create_metrics_summary(fold_df, RESULTS_DIR / "metrics_summary.png")

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print("=" * 70)
