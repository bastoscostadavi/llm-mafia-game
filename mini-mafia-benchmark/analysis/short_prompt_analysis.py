#!/usr/bin/env python3
"""Analyze short-prompt experiment results and generate plots."""

from __future__ import annotations

import os
import sqlite3
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault('MPLBACKEND', 'Agg')

# Add parent directory to path so we can import from results
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from results.utils import (
    bayesian_win_rate,
    create_horizontal_bar_plot,
    get_background_color,
)

DB_PATH = PROJECT_ROOT / 'database' / 'mini_mafia_short_prompt.db'
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BACKGROUND_ORDER = ['gpt-4.1-mini', 'gpt-5-mini', 'grok-3-mini', 'deepseek-chat', 'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf']
BACKGROUND_DISPLAY = {
    'gpt-5-mini': 'GPT-5 Mini Background',
    'gpt-4.1-mini': 'GPT-4.1 Mini Background',
    'grok-3-mini': 'Grok 3 Mini Background',
    'deepseek-chat': 'DeepSeek Background',
    'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf': 'Mistral 7B Instruct',
}
PLOT_TEMPLATE = RESULTS_DIR / 'win_rates_short_prompt_{background}.png'
TOTAL_SCORE_PATH = RESULTS_DIR / 'scores_short_prompt_total.png'
TOTAL_COLOR = '#FF6600'

MODEL_DISPLAY_NAMES = {
    'gpt-5-mini': 'GPT-5 Mini',
    'gpt-4.1-mini': 'GPT-4.1 Mini',
    'grok-3-mini': 'Grok 3 Mini',
    'deepseek-chat': 'DeepSeek V3.1',
    'mistral': 'Mistral 7B Instruct',
    'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf': 'Mistral 7B Instruct',
    'claude-sonnet-4-20250514': 'Claude Sonnet 4',
    'claude-opus-4-1-20250805': 'Claude Opus 4.1',
    'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf': 'Llama 3.1 8B Instruct',
    'Qwen2.5-7B-Instruct-Q4_K_M.gguf': 'Qwen2.5 7B Instruct'
}


def display_name(model_key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def fetch_background_stats(db_path: Path) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Aggregate mafioso stats grouped by background and model."""
    if not db_path.exists():
        raise FileNotFoundError(f'Database not found: {db_path}')

    data = defaultdict(lambda: defaultdict(lambda: {'total_games': 0, 'mafia_wins': 0}))

    query = """
        SELECT
            bg.model_name AS background_model,
            mafia.model_name AS mafioso_model,
            COUNT(*) AS total_games,
            SUM(CASE WHEN g.winner = 'mafia' THEN 1 ELSE 0 END) AS mafia_wins
        FROM games g
        JOIN game_players gp_mafia ON g.game_id = gp_mafia.game_id AND gp_mafia.role = 'mafioso'
        JOIN players mafia ON mafia.player_id = gp_mafia.player_id
        JOIN game_players gp_detective ON g.game_id = gp_detective.game_id AND gp_detective.role = 'detective'
        JOIN players bg ON bg.player_id = gp_detective.player_id
        GROUP BY bg.model_name, mafia.model_name
    """

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(query):
            background = row['background_model']
            if background not in BACKGROUND_DISPLAY:
                continue
            model = row['mafioso_model']
            data[background][model]['total_games'] += int(row['total_games'])
            data[background][model]['mafia_wins'] += int(row['mafia_wins'])

    return data


def enrich_stats(raw_stats: Dict[str, Dict[str, Dict[str, int]]]):
    """Compute Bayesian win rates for each model in every background."""
    enriched = {}
    for background, models in raw_stats.items():
        rows = []
        for model, counts in models.items():
            total = counts['total_games']
            wins = counts['mafia_wins']
            if total == 0:
                continue
            win_rate, err = bayesian_win_rate(wins, total)
            rows.append({
                'model_name': model,
                'win_rate_percent': round(win_rate, 2),
                'uncertainty_percent': round(err, 2),
                'total_games': total
            })
        enriched[background] = rows
    return enriched


def plot_background_win_rates(background: str, stats: List[Dict]):
    if not stats:
        print(f'No data for background {background}; skipping plot.')
        return

    models = [display_name(row['model_name']) for row in stats]
    values = [row['win_rate_percent'] for row in stats]
    errors = [row['uncertainty_percent'] for row in stats]

    filename = str(PLOT_TEMPLATE).format(background=background.replace('.', '').replace('-', '_'))
    color = get_background_color(BACKGROUND_DISPLAY[background])

    create_horizontal_bar_plot(
        models=models,
        values=values,
        errors=errors,
        xlabel='Mafia Win Rate (%)',
        filename=filename,
        color=color,
        sort_ascending=False,
        show_reference_line=False,
        text_offset=1.5,
        reverse_after_sort=True,
        x_min=0,
        x_max=100
    )
    print(f'Saved background plot: {filename}')


def build_model_matrices(enriched_stats: Dict[str, List[Dict]]):
    backgrounds = [bg for bg in BACKGROUND_ORDER if bg in enriched_stats]
    bg_lookup = {bg: {row['model_name']: row for row in rows} for bg, rows in enriched_stats.items() if rows}
    models = sorted({model for lookup in bg_lookup.values() for model in lookup})
    complete_models = [model for model in models if all(model in bg_lookup.get(bg, {}) for bg in backgrounds)]

    win_rates = {model: {bg: bg_lookup[bg][model]['win_rate_percent'] for bg in backgrounds}
                 for model in complete_models}
    uncertainties = {model: {bg: bg_lookup[bg][model]['uncertainty_percent'] for bg in backgrounds}
                     for model in complete_models}

    return backgrounds, complete_models, win_rates, uncertainties


def compute_aggregate_scores(backgrounds: List[str], models: List[str],
                              win_rates: Dict[str, Dict[str, float]],
                              uncertainties: Dict[str, Dict[str, float]]):
    if not backgrounds or not models:
        return [], []

    z_scores = {model: {} for model in models}
    z_errors = {model: {} for model in models}

    for background in backgrounds:
        rates = [win_rates[model][background] for model in models]
        errs = [uncertainties[model][background] for model in models]
        mean = sum(rates) / len(rates)
        if len(rates) > 1:
            variance = sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)
            std = math.sqrt(variance)
        else:
            std = 1.0
        if std == 0:
            std = 1.0

        for idx, model in enumerate(models):
            z_scores[model][background] = (rates[idx] - mean) / std
            z_errors[model][background] = errs[idx] / std

    exp_scores = []
    z_score_rows = []
    for model in models:
        avg_z = sum(z_scores[model].values()) / len(backgrounds)
        z_variance = sum((z_errors[model][bg]) ** 2 for bg in backgrounds)
        avg_z_err = math.sqrt(z_variance) / len(backgrounds)
        exp_score = math.exp(avg_z)
        exp_err = exp_score * avg_z_err
        exp_scores.append((model, exp_score, exp_err))
        z_score_rows.append((model, avg_z, avg_z_err))

    return exp_scores, z_score_rows


def plot_total_scores(exp_scores: List[Tuple[str, float, float]]):
    if not exp_scores:
        print('Not enough data to compute aggregated scores.')
        return

    models = [display_name(model) for model, _, _ in exp_scores]
    values = [score for _, score, _ in exp_scores]
    errors = [err for _, _, err in exp_scores]

    create_horizontal_bar_plot(
        models=models,
        values=values,
        errors=errors,
        xlabel='exp(average z-score)',
        filename=str(TOTAL_SCORE_PATH),
        color=TOTAL_COLOR,
        sort_ascending=True,
        show_reference_line=True,
        x_min=0
    )
    print(f'Saved aggregated score plot: {TOTAL_SCORE_PATH}')


def plot_avg_z_scores(z_scores: List[Tuple[str, float, float]]):
    if not z_scores:
        print('Not enough data to compute average z-score plot.')
        return

    models = [display_name(model) for model, _, _ in z_scores]
    values = [score for _, score, _ in z_scores]
    errors = [err for _, _, err in z_scores]

    filename = RESULTS_DIR / 'scores_short_prompt_avg_z.png'
    create_horizontal_bar_plot(
        models=models,
        values=values,
        errors=errors,
        xlabel='Average z-score',
        filename=str(filename),
        color='#3498DB',
        sort_ascending=True,
        show_reference_line=True,
        x_min=None
    )
    print(f'Saved average z-score plot: {filename}')


def main():
    print('Analyzing short-prompt experiments...')
    raw_stats = fetch_background_stats(DB_PATH)
    if not raw_stats:
        print('No data found in short-prompt database.')
        return

    enriched_stats = enrich_stats(raw_stats)

    for background in BACKGROUND_ORDER:
        if background in enriched_stats:
            plot_background_win_rates(background, enriched_stats[background])
        else:
            print(f'Background {background} has no games recorded yet.')

    backgrounds, models, win_rates, uncertainties = build_model_matrices(enriched_stats)
    exp_scores, z_scores = compute_aggregate_scores(backgrounds, models, win_rates, uncertainties)
    plot_total_scores(exp_scores)
    plot_avg_z_scores(z_scores)
    print('Done.')


if __name__ == '__main__':
    main()
