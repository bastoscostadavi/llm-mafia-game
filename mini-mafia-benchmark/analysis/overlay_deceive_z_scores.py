#!/usr/bin/env python3
"""Overlay Deceive average z-score comparisons (z & exp) and output table."""

from __future__ import annotations

import csv
import math
import os
import sqlite3
from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_MODULE_DIR = PROJECT_ROOT / 'results'

if str(RESULTS_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(RESULTS_MODULE_DIR))

from short_prompt_analysis import (
    DB_PATH,
    RESULTS_DIR,
    compute_aggregate_scores,
    build_model_matrices,
    enrich_stats,
    fetch_background_stats,
    display_name,
)
from scores_table_and_plot import (
    load_win_rates_data,
    calculate_aggregated_scores,
)

WIN_RATES_CSV = RESULTS_MODULE_DIR / 'win_rates.csv'
ROUND8_DB = PROJECT_ROOT / 'database' / 'mini_mafia_round8.db'
SHORT_PROMPT_ROUND8_DB = PROJECT_ROOT / 'database' / 'mini_mafia_short_prompt_round8.db'
OUTPUT_Z_PLOT = RESULTS_DIR / 'overlay_deceive_z_scores.png'
OUTPUT_EXP_PLOT = RESULTS_DIR / 'overlay_deceive_exp_scores.png'
OUTPUT_TABLE = RESULTS_DIR / 'overlay_deceive_z_scores.csv'
OUTPUT_BA_PLOT = RESULTS_DIR / 'overlay_deceive_z_scores_bland_altman.png'


def bayesian_win_rate(wins: int, total: int) -> Tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    mean = (wins + 1) / (total + 2)
    variance = (mean * (1 - mean)) / (total + 3)
    std = math.sqrt(variance)
    return mean * 100, std * 100


def compute_short_prompt_scores():
    raw_stats = fetch_background_stats(DB_PATH)
    enriched = enrich_stats(raw_stats)
    backgrounds, models, win_rates, uncertainties = build_model_matrices(enriched)
    exp_scores, z_scores = compute_aggregate_scores(backgrounds, models, win_rates, uncertainties)

    z_map = {display_name(model): (score, err) for model, score, err in z_scores}
    exp_map = {display_name(model): (score, err) for model, score, err in exp_scores}
    return z_map, exp_map


def compute_short_prompt_round8_scores():
    if not SHORT_PROMPT_ROUND8_DB.exists():
        print(f'Short-prompt round8 database not found at {SHORT_PROMPT_ROUND8_DB}; skipping.')
        return {}, {}

    raw_stats = fetch_background_stats(SHORT_PROMPT_ROUND8_DB)
    enriched = enrich_stats(raw_stats)
    backgrounds, models, win_rates, uncertainties = build_model_matrices(enriched)
    exp_scores, z_scores = compute_aggregate_scores(backgrounds, models, win_rates, uncertainties)

    z_map = {display_name(model): (score, err) for model, score, err in z_scores}
    exp_map = {display_name(model): (score, err) for model, score, err in exp_scores}
    return z_map, exp_map


def compute_article_scores():
    if not WIN_RATES_CSV.exists():
        raise FileNotFoundError(f'win_rates.csv not found at {WIN_RATES_CSV}')

    cwd = os.getcwd()
    os.chdir(RESULTS_MODULE_DIR)
    try:
        capabilities = load_win_rates_data()
    finally:
        os.chdir(cwd)

    if 'Deceive' not in capabilities:
        raise ValueError('Deceive capability not found in win_rates.csv')

    data = capabilities['Deceive']
    win_rates_df = data['win_rates']
    uncertainties_df = data['uncertainties']
    avg_z_scores, avg_z_errors, exp_scores, exp_errors = calculate_aggregated_scores(win_rates_df, uncertainties_df, 'Deceive')

    z_map = {
        display_name(model): (avg_z_scores.loc[model], avg_z_errors.loc[model])
        for model in avg_z_scores.index
    }
    exp_map = {
        display_name(model): (exp_scores.loc[model], exp_errors.loc[model])
        for model in exp_scores.index
    }
    return z_map, exp_map


def compute_round8_scores() -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """Compute z-scores and exp scores for round8 database.

    Since round8 has only 1 background, we use the z-score directly as the total score
    (no averaging across backgrounds).

    Returns:
        Tuple of (z_map, exp_map) where each is a dict mapping model name to (score, error)
    """
    if not ROUND8_DB.exists():
        print(f'Round8 database not found at {ROUND8_DB}; skipping round8 overlay.')
        return {}, {}

    with sqlite3.connect(ROUND8_DB) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT p.model_name, COUNT(*) AS total_games,
                   SUM(CASE WHEN g.winner = 'mafia' THEN 1 ELSE 0 END) AS mafia_wins
            FROM games g
            JOIN game_players gp ON g.game_id = gp.game_id AND gp.role = 'mafioso'
            JOIN players p ON gp.player_id = p.player_id
            GROUP BY p.model_name
            """
        ).fetchall()

    if not rows:
        print('Round8 database has no mafioso games; skipping round8 overlay.')
        return {}, {}

    stats = []
    for row in rows:
        win_rate, err = bayesian_win_rate(int(row['mafia_wins']), int(row['total_games']))
        stats.append({
            'model_name': row['model_name'],
            'win_rate_percent': win_rate,
            'uncertainty_percent': err
        })

    rates = [entry['win_rate_percent'] for entry in stats]
    mean = sum(rates) / len(rates)
    variance = sum((r - mean) ** 2 for r in rates) / (len(rates) - 1) if len(rates) > 1 else 0.0
    std = math.sqrt(variance) if variance > 0 else 1.0

    z_map = {}
    exp_map = {}
    for entry in stats:
        z = (entry['win_rate_percent'] - mean) / std
        z_err = entry['uncertainty_percent'] / std
        exp_score = math.exp(z)
        exp_err = exp_score * z_err
        model_name = display_name(entry['model_name'])
        # Since round8 has only 1 background, use z-score directly (no averaging)
        z_map[model_name] = (z, z_err)
        exp_map[model_name] = (exp_score, exp_err)

    return z_map, exp_map


def save_table(rows, include_short_prompt_round8):
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_TABLE.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = [
            'Model',
            'ShortPrompt Avg Z', 'ShortPrompt Z Err',
            'Article Avg Z', 'Article Z Err',
        ]
        if include_short_prompt_round8:
            headers.extend(['ShortPrompt Round8 Avg Z', 'ShortPrompt Round8 Z Err'])
        headers.extend([
            'ShortPrompt exp(z)', 'ShortPrompt exp Err',
            'Article exp(z)', 'Article exp Err'
        ])
        if include_short_prompt_round8:
            headers.extend(['ShortPrompt Round8 exp(z)', 'ShortPrompt Round8 exp Err'])
        writer.writerow(headers)
        for row in rows:
            model = row['model']
            sp_z, sp_z_err = row['short_prompt']
            art_z, art_z_err = row['article']
            sp_exp, sp_exp_err = row['short_prompt_exp']
            art_exp, art_exp_err = row['article_exp']
            csv_row = [
                model,
                f"{sp_z:.4f}", f"{sp_z_err:.4f}",
                f"{art_z:.4f}", f"{art_z_err:.4f}",
            ]
            if include_short_prompt_round8:
                sp_r8_z, sp_r8_z_err = row['short_prompt_round8']
                csv_row.extend([f"{sp_r8_z:.4f}", f"{sp_r8_z_err:.4f}"])
            csv_row.extend([
                f"{sp_exp:.4f}", f"{sp_exp_err:.4f}",
                f"{art_exp:.4f}", f"{art_exp_err:.4f}"
            ])
            if include_short_prompt_round8:
                sp_r8_exp, sp_r8_exp_err = row['short_prompt_round8_exp']
                csv_row.extend([f"{sp_r8_exp:.4f}", f"{sp_r8_exp_err:.4f}"])
            writer.writerow(csv_row)
    print(f'Saved overlay table to {OUTPUT_TABLE}')


def _plot_overlay(models, sp_values, sp_errors, art_values, art_errors,
                  filename, reference_line):
    plt.ioff()
    fig, ax = plt.subplots(figsize=(14, 7))
    y_pos = list(range(len(models)))
    bar_height = 0.35

    ax.barh([y - bar_height/2 for y in y_pos], sp_values, height=bar_height,
            xerr=sp_errors, label='Short-Prompt', color='#1f77b4', alpha=0.85,
            error_kw={'capsize': 5, 'capthick': 2})
    ax.barh([y + bar_height/2 for y in y_pos], art_values, height=bar_height,
            xerr=art_errors, label='Article', color='#E74C3C', alpha=0.85,
            error_kw={'capsize': 5, 'capthick': 2})

    ax.set_xlabel('Deceive Score', fontsize=24, fontweight='bold')
    ax.set_yticks([])
    ax.legend()

    all_right = [max(sp + se, art + ae) for sp, se, art, ae in zip(sp_values, sp_errors, art_values, art_errors)]
    all_left = [min(sp - se, art - ae) for sp, se, art, ae in zip(sp_values, sp_errors, art_values, art_errors)]
    max_val = max(all_right)
    min_val = min(all_left)
    padding = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
    ax.set_xlim(min_val - padding, max_val + padding)

    data_range = max_val - min_val if max_val > min_val else 1.0
    text_offset = data_range * 0.02
    for idx, model in enumerate(models):
        right_edge = art_values[idx] + art_errors[idx]
        ax.text(right_edge + text_offset, idx + bar_height/2, model, ha='left', va='center',
                fontweight='bold', fontsize=20)

    if reference_line is not None:
        ax.axvline(reference_line, color='gray', linestyle='--', linewidth=2)

    ax.grid(True, axis='x', color='gray', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved overlay plot to {filename}')


def plot_z_overlay(rows, include_short_prompt_round8):
    if not rows:
        raise ValueError('No data to plot.')

    models = [row['model'] for row in rows]
    sp_scores = [row['short_prompt'][0] for row in rows]
    sp_errs = [row['short_prompt'][1] for row in rows]
    art_scores = [row['article'][0] for row in rows]
    art_errs = [row['article'][1] for row in rows]

    if include_short_prompt_round8:
        sp_r8_scores = [row['short_prompt_round8'][0] for row in rows]
        sp_r8_errs = [row['short_prompt_round8'][1] for row in rows]

    # Plot with multiple bars per model - matching theoretical model style
    plt.ioff()

    # Set font size to match theoretical model exactly
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })

    num_datasets = 2 + (1 if include_short_prompt_round8 else 0)
    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = list(range(len(models)))
    bar_height = 0.7 / num_datasets

    # Use distinct colors for each dataset
    datasets = [
        (sp_scores, sp_errs, 'Short-prompt', '#2E86AB'),
        (art_scores, art_errs, 'Default', '#E63946')
    ]
    if include_short_prompt_round8:
        datasets.append((sp_r8_scores, sp_r8_errs, 'Short-prompt and 8 Rounds', '#9B59B6'))

    for idx, (scores, errs, label, color) in enumerate(datasets):
        offset = (idx - (num_datasets - 1) / 2) * bar_height
        ax.barh([y + offset for y in y_pos], scores, height=bar_height,
                xerr=errs, label=label, color=color, alpha=0.7,
                error_kw={'capsize': 3, 'capthick': 1.5, 'elinewidth': 1.5})

    ax.set_xlabel('Deceive Score', fontsize=12)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.legend(fontsize=11)

    all_values = sp_scores + art_scores
    all_errors = sp_errs + art_errs
    if include_short_prompt_round8:
        all_values += sp_r8_scores
        all_errors += sp_r8_errs

    all_right = [v + e for v, e in zip(all_values, all_errors)]
    all_left = [v - e for v, e in zip(all_values, all_errors)]
    max_val = max(all_right)
    min_val = min(all_left)
    padding = (max_val - min_val) * 0.15
    ax.set_xlim(min_val - padding, max_val + padding)

    # Add vertical line at 0 (reference point)
    if min_val <= 0 <= max_val:
        ax.axvline(x=0, color='#333333', alpha=0.5, linewidth=1.5, linestyle='-', zorder=0)

    # Minimal grid
    ax.grid(True, alpha=0.2, axis='x', linewidth=0.5)
    ax.set_axisbelow(True)

    # Clean spines - exactly like theoretical model
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_Z_PLOT, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f'Saved overlay plot to {OUTPUT_Z_PLOT}')


def plot_exp_overlay(rows, include_short_prompt_round8):
    if not rows:
        raise ValueError('No data to plot.')

    models = [row['model'] for row in rows]
    sp_scores = [row['short_prompt_exp'][0] for row in rows]
    sp_errs = [row['short_prompt_exp'][1] for row in rows]
    art_scores = [row['article_exp'][0] for row in rows]
    art_errs = [row['article_exp'][1] for row in rows]

    if include_short_prompt_round8:
        sp_r8_scores = [row['short_prompt_round8_exp'][0] for row in rows]
        sp_r8_errs = [row['short_prompt_round8_exp'][1] for row in rows]

    # Plot with multiple bars per model - matching theoretical model style
    plt.ioff()

    # Set font size to match theoretical model exactly
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
    })

    num_datasets = 2 + (1 if include_short_prompt_round8 else 0)
    fig, ax = plt.subplots(figsize=(9, 6))
    y_pos = list(range(len(models)))
    bar_height = 0.7 / num_datasets

    # Use distinct colors for each dataset
    datasets = [
        (sp_scores, sp_errs, 'Short-prompt', '#2E86AB'),
        (art_scores, art_errs, 'Default', '#E63946')
    ]
    if include_short_prompt_round8:
        datasets.append((sp_r8_scores, sp_r8_errs, 'Short-prompt and 8 Rounds', '#9B59B6'))

    for idx, (scores, errs, label, color) in enumerate(datasets):
        offset = (idx - (num_datasets - 1) / 2) * bar_height
        ax.barh([y + offset for y in y_pos], scores, height=bar_height,
                xerr=errs, label=label, color=color, alpha=0.7,
                error_kw={'capsize': 3, 'capthick': 1.5, 'elinewidth': 1.5})

    ax.set_xlabel('Deceive Score', fontsize=12)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.legend(fontsize=11)

    all_values = sp_scores + art_scores
    all_errors = sp_errs + art_errs
    if include_short_prompt_round8:
        all_values += sp_r8_scores
        all_errors += sp_r8_errs

    all_right = [v + e for v, e in zip(all_values, all_errors)]
    all_left = [v - e for v, e in zip(all_values, all_errors)]
    max_val = max(all_right)
    min_val = min(all_left)
    padding = (max_val - min_val) * 0.15
    ax.set_xlim(min_val - padding, max_val + padding)

    # Add vertical line at 1 (reference point)
    if min_val <= 1 <= max_val:
        ax.axvline(x=1, color='#333333', alpha=0.5, linewidth=1.5, linestyle='-', zorder=0)

    # Minimal grid
    ax.grid(True, alpha=0.2, axis='x', linewidth=0.5)
    ax.set_axisbelow(True)

    # Clean spines - exactly like theoretical model
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_EXP_PLOT, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f'Saved exp overlay plot to {OUTPUT_EXP_PLOT}')


def compute_correlations_with_uncertainty(exp_maps, labels, n_samples=10000):
    """
    Compute pairwise correlation coefficients with Monte Carlo uncertainty estimation.

    Args:
        exp_maps: List of dictionaries mapping model names to (score, error) tuples
        labels: List of labels for each dataset
        n_samples: Number of Monte Carlo samples for uncertainty estimation

    Returns:
        Dictionary mapping (label1, label2) tuples to (correlation, uncertainty) tuples
    """
    n_datasets = len(exp_maps)
    correlations = {}

    # For each pair of datasets
    for i in range(n_datasets):
        for j in range(i + 1, n_datasets):
            # Find common models
            common_models = list(set(exp_maps[i].keys()) & set(exp_maps[j].keys()))
            if not common_models:
                continue

            # Extract scores and errors for common models
            scores_i = np.array([exp_maps[i][m][0] for m in common_models])
            errors_i = np.array([exp_maps[i][m][1] for m in common_models])
            scores_j = np.array([exp_maps[j][m][0] for m in common_models])
            errors_j = np.array([exp_maps[j][m][1] for m in common_models])

            # Monte Carlo sampling
            sampled_correlations = []
            for _ in range(n_samples):
                # Sample from normal distributions
                sampled_i = np.random.normal(scores_i, errors_i)
                sampled_j = np.random.normal(scores_j, errors_j)

                # Compute Pearson correlation
                corr = np.corrcoef(sampled_i, sampled_j)[0, 1]
                sampled_correlations.append(corr)

            # Calculate mean and std dev
            mean_corr = np.mean(sampled_correlations)
            std_corr = np.std(sampled_correlations)

            correlations[(labels[i], labels[j])] = (mean_corr, std_corr, len(common_models))

    return correlations


def plot_bland_altman(rows):
    if not rows:
        raise ValueError('No data to plot.')

    means = []
    diffs = []
    for row in rows:
        sp, art = row['short_prompt'][0], row['article'][0]
        means.append((sp + art) / 2)
        diffs.append(sp - art)

    bias = sum(diffs) / len(diffs)
    variance = sum((d - bias) ** 2 for d in diffs) / (len(diffs) - 1) if len(diffs) > 1 else 0.0
    sd = variance ** 0.5
    upper = bias + 1.96 * sd
    lower = bias - 1.96 * sd

    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(means, diffs, color='#34495e')
    ax.axhline(bias, color='red', linestyle='--', label=f'Bias = {bias:.3f}')
    ax.axhline(upper, color='gray', linestyle=':', label=f'+1.96 SD = {upper:.3f}')
    ax.axhline(lower, color='gray', linestyle=':', label=f'-1.96 SD = {lower:.3f}')
    ax.set_xlabel('Mean of Short-Prompt and Article z-scores')
    ax.set_ylabel('Difference (Short-Prompt - Article)')
    ax.set_title('Bland–Altman Comparison: Deceive z-scores')
    ax.legend(loc='best')
    ax.grid(True, color='gray', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_BA_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved Bland–Altman plot to {OUTPUT_BA_PLOT}')


def main():
    short_z, short_exp = compute_short_prompt_scores()
    article_z, article_exp = compute_article_scores()
    short_prompt_round8_z, short_prompt_round8_exp = compute_short_prompt_round8_scores()

    common_models = set(short_z) & set(article_z)
    include_short_prompt_round8 = bool(short_prompt_round8_z)

    if include_short_prompt_round8:
        common_models &= set(short_prompt_round8_z)

    common_models = list(common_models)
    if not common_models:
        raise ValueError('No overlapping models between datasets to plot.')

    # Order by article z-score ascending (to match plot ordering)
    common_models.sort(key=lambda m: article_z[m][0])

    rows = [
        {
            'model': model,
            'short_prompt': short_z[model],
            'article': article_z[model],
            'short_prompt_exp': short_exp[model],
            'article_exp': article_exp[model]
        }
        for model in common_models
    ]

    if include_short_prompt_round8:
        for row in rows:
            row['short_prompt_round8'] = short_prompt_round8_z[row['model']]
            row['short_prompt_round8_exp'] = short_prompt_round8_exp[row['model']]

    save_table(rows, include_short_prompt_round8)
    plot_z_overlay(rows, include_short_prompt_round8)
    plot_exp_overlay(rows, include_short_prompt_round8)
    plot_bland_altman(rows)

    # Compute correlation coefficients with Monte Carlo uncertainty estimation
    print('\n' + '=' * 60)
    print('CORRELATION ANALYSIS (Exponential Scores)')
    print('=' * 60)

    exp_maps = [short_exp, article_exp]
    labels = ['Short-prompt', 'Article']

    if include_short_prompt_round8:
        exp_maps.append(short_prompt_round8_exp)
        labels.append('Short-prompt Round8')

    correlations = compute_correlations_with_uncertainty(exp_maps, labels, n_samples=10000)

    for (label1, label2), (corr, err, n_models) in sorted(correlations.items()):
        print(f'\n{label1} vs {label2}:')
        print(f'  Pearson r = {corr:.4f} ± {err:.4f}')
        print(f'  n = {n_models} models')

    print('\n' + '=' * 60)


if __name__ == '__main__':
    main()
