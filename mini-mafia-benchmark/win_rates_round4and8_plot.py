#!/usr/bin/env python3
"""Generate mafioso win-rate plots combining round 4 and round 8 databases."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

from results.utils import (
    bayesian_win_rate,
    create_horizontal_bar_plot,
    get_background_color,
    get_display_name,
)

DB_PATHS = [
    Path('database/mini_mafia_round4.db'),
    Path('database/mini_mafia_round8.db'),
]
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_PATH = OUTPUT_DIR / 'win_rates_round4and8.png'
Z_PLOT_PATH = OUTPUT_DIR / 'win_rates_round4and8_zscores.png'
BAR_COLOR = get_background_color('gpt-5 mini background')
Z_COLOR = '#FF6600'


def fetch_mafioso_stats(db_path: Path):
    """Return mafioso stats grouped by model for a single database."""
    if not db_path.exists():
        print(f"Warning: database not found at {db_path}; skipping.")
        return []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        query = """
            SELECT
                p.model_name,
                COUNT(*) AS total_games,
                SUM(CASE WHEN g.winner = 'mafia' THEN 1 ELSE 0 END) AS mafia_wins
            FROM games g
            JOIN game_players gp ON g.game_id = gp.game_id AND gp.role = 'mafioso'
            JOIN players p        ON gp.player_id = p.player_id
            GROUP BY p.model_name
            ORDER BY total_games DESC, p.model_name
        """
        rows = conn.execute(query).fetchall()
        return [dict(row) for row in rows]


def aggregate_stats(db_paths):
    """Combine mafioso stats from multiple databases."""
    combined = defaultdict(lambda: {'model_name': None, 'total_games': 0, 'mafia_wins': 0})

    for db_path in db_paths:
        stats = fetch_mafioso_stats(db_path)
        for row in stats:
            model = row['model_name']
            entry = combined[model]
            entry['model_name'] = model
            entry['total_games'] += int(row['total_games'])
            entry['mafia_wins'] += int(row['mafia_wins'])

    aggregated = [value for value in combined.values() if value['total_games'] > 0]
    aggregated.sort(key=lambda r: (-r['total_games'], r['model_name']))
    return aggregated


def compute_win_rates(stats):
    enriched = []
    for row in stats:
        total_games = int(row['total_games'])
        mafia_wins = int(row['mafia_wins'])
        win_rate, err = bayesian_win_rate(mafia_wins, total_games)
        enriched.append({
            **row,
            'win_rate_percent': round(win_rate, 2),
            'uncertainty_percent': round(err, 2)
        })
    return enriched


def compute_z_scores(data):
    if not data:
        return

    rates = np.array([row['win_rate_percent'] for row in data], dtype=float)
    errs = np.array([row['uncertainty_percent'] for row in data], dtype=float)
    mean = rates.mean()
    std = rates.std(ddof=1) if len(rates) > 1 else 1.0
    if std == 0:
        std = 1.0

    for row, rate, err in zip(data, rates, errs):
        z = (rate - mean) / std
        z_err = err / std
        exp_score = float(np.exp(z))
        exp_err = exp_score * z_err
        row['z_score'] = float(round(z, 4))
        row['z_uncertainty'] = float(round(z_err, 4))
        row['exp_score'] = float(round(exp_score, 4))
        row['exp_uncertainty'] = float(round(exp_err, 4))


def create_plot(data):
    if not data:
        print('No mafioso records to plot.')
        return

    models = [get_display_name(row['model_name']) for row in data]
    values = [row['win_rate_percent'] for row in data]
    errors = [row['uncertainty_percent'] for row in data]

    create_horizontal_bar_plot(
        models=models,
        values=values,
        errors=errors,
        xlabel='Mafia Win Rate (%)',
        filename=str(PLOT_PATH),
        color=BAR_COLOR,
        sort_ascending=True,
        show_reference_line=False,
        x_min=0,
        x_max=100
    )
    print(f"Saved combined win-rate plot: {PLOT_PATH}")


def create_exponential_z_plot(data):
    if not data:
        return

    models = [get_display_name(row['model_name']) for row in data]
    exp_values = [row.get('exp_score', 0.0) for row in data]
    exp_errors = [row.get('exp_uncertainty', 0.0) for row in data]

    create_horizontal_bar_plot(
        models=models,
        values=exp_values,
        errors=exp_errors,
        xlabel='exp(z)',
        filename=str(Z_PLOT_PATH),
        color=Z_COLOR,
        sort_ascending=True,
        show_reference_line=True,
        x_min=0
    )
    print(f"Saved combined exp(z) plot: {Z_PLOT_PATH}")


def main():
    print('Creating combined (round4 + round8) mafioso win-rate plots...')
    stats = aggregate_stats(DB_PATHS)
    if not stats:
        print('No mafioso data found across the provided databases.')
        return

    data = compute_win_rates(stats)
    compute_z_scores(data)
    create_plot(data)
    create_exponential_z_plot(data)
    print('Done.')


if __name__ == '__main__':
    main()
