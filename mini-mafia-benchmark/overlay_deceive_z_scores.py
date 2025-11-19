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

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_MODULE_DIR = SCRIPT_DIR / 'results'

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
ROUND8_DB = SCRIPT_DIR / 'database' / 'mini_mafia_round8.db'
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


def compute_round8_exp_scores() -> Dict[str, Tuple[float, float]]:
    if not ROUND8_DB.exists():
        print(f'Round8 database not found at {ROUND8_DB}; skipping round8 overlay.')
        return {}

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
        return {}

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

    exp_map = {}
    for entry in stats:
        z = (entry['win_rate_percent'] - mean) / std
        z_err = entry['uncertainty_percent'] / std
        exp_score = math.exp(z)
        exp_err = exp_score * z_err
        exp_map[display_name(entry['model_name'])] = (exp_score, exp_err)

    return exp_map


def save_table(rows, include_round8):
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_TABLE.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = [
            'Model',
            'ShortPrompt Avg Z', 'ShortPrompt Z Err',
            'Article Avg Z', 'Article Z Err',
            'ShortPrompt exp(z)', 'ShortPrompt exp Err',
            'Article exp(z)', 'Article exp Err'
        ]
        if include_round8:
            headers.extend(['Round8 exp(z)', 'Round8 exp Err'])
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
                f"{sp_exp:.4f}", f"{sp_exp_err:.4f}",
                f"{art_exp:.4f}", f"{art_exp_err:.4f}"
            ]
            if include_round8:
                rnd_exp, rnd_err = row['round8_exp']
                csv_row.extend([f"{rnd_exp:.4f}", f"{rnd_err:.4f}"])
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


def plot_z_overlay(rows):
    if not rows:
        raise ValueError('No data to plot.')

    models = [row['model'] for row in rows]
    sp_scores = [row['short_prompt'][0] for row in rows]
    sp_errs = [row['short_prompt'][1] for row in rows]
    art_scores = [row['article'][0] for row in rows]
    art_errs = [row['article'][1] for row in rows]

    _plot_overlay(models, sp_scores, sp_errs, art_scores, art_errs,
                  OUTPUT_Z_PLOT, reference_line=0)


def plot_exp_overlay(rows, include_round8):
    if not rows:
        raise ValueError('No data to plot.')

    models = [row['model'] for row in rows]
    sp_scores = [row['short_prompt_exp'][0] for row in rows]
    sp_errs = [row['short_prompt_exp'][1] for row in rows]
    art_scores = [row['article_exp'][0] for row in rows]
    art_errs = [row['article_exp'][1] for row in rows]
    if include_round8:
        round_scores = [row['round8_exp'][0] for row in rows]
        round_errs = [row['round8_exp'][1] for row in rows]

    plt.ioff()
    fig, ax = plt.subplots(figsize=(14, 7))
    y_pos = list(range(len(models)))
    bar_height = 0.22
    offset = bar_height + 0.02

    ax.barh([y - offset for y in y_pos], sp_scores, height=bar_height,
            xerr=sp_errs, label='Short-Prompt exp(z)', color='#1f77b4', alpha=0.85,
            error_kw={'capsize': 5, 'capthick': 2})
    ax.barh(y_pos, art_scores, height=bar_height,
            xerr=art_errs, label='Article exp(z)', color='#E74C3C', alpha=0.85,
            error_kw={'capsize': 5, 'capthick': 2})
    if include_round8:
        ax.barh([y + offset for y in y_pos], round_scores, height=bar_height,
                xerr=round_errs, label='Round8 exp(z)', color='#2ECC71', alpha=0.85,
                error_kw={'capsize': 5, 'capthick': 2})

    ax.set_xlabel('exp(average z-score)', fontsize=24, fontweight='bold')
    ax.set_yticks([])
    ax.legend()

    combined_right = [art_scores[i] + art_errs[i] for i in range(len(models))]
    combined_left = [min(sp_scores[i] - sp_errs[i], art_scores[i] - art_errs[i]) for i in range(len(models))]
    if include_round8:
        for i in range(len(models)):
            combined_right[i] = max(combined_right[i], round_scores[i] + round_errs[i])
            combined_left[i] = min(combined_left[i], round_scores[i] - round_errs[i])
    max_val = max(combined_right)
    min_val = min(combined_left)
    padding = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
    ax.set_xlim(min_val - padding, max_val + padding)

    data_range = max_val - min_val if max_val > min_val else 1.0
    text_offset = data_range * 0.02
    for idx, model in enumerate(models):
        right_edge = art_scores[idx] + art_errs[idx]
        ax.text(right_edge + text_offset, y_pos[idx], model, ha='left', va='center',
                fontweight='bold', fontsize=20)

    ax.axvline(1, color='gray', linestyle='--', linewidth=2)
    ax.grid(True, axis='x', color='gray', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_EXP_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved exp overlay plot to {OUTPUT_EXP_PLOT}')


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
    round8_exp = compute_round8_exp_scores()

    common_models = set(short_z) & set(article_z)
    include_round8 = bool(round8_exp)
    if include_round8:
        common_models &= set(round8_exp)
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

    if include_round8:
        for row in rows:
            row['round8_exp'] = round8_exp[row['model']]

    save_table(rows, include_round8)
    plot_z_overlay(rows)
    plot_exp_overlay(rows, include_round8)
    plot_bland_altman(rows)


if __name__ == '__main__':
    main()
