#!/usr/bin/env python3
"""Generate mafioso win-rate plot from the 4-round Mini-Mafia database."""

import sqlite3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from results.utils import (
    bayesian_win_rate,
    create_horizontal_bar_plot,
    get_background_color,
)

DB_PATH = Path('database/mini_mafia_round8.db')
BASELINE_DB_PATH = Path('database/mini_mafia.db')
OUTPUT_DIR = Path('results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_PATH = OUTPUT_DIR / 'win_rates_round8.png'
Z_PLOT_PATH = OUTPUT_DIR / 'win_rates_round8_zscores.png'

ROUND4_COLOR = get_background_color('gpt-5 mini background')
ORANGE_COLOR = '#FF6600'


def fetch_mafioso_stats(db_path: Path, capability_filter: str = None,
                        background_filter: str = None):
    """Return aggregated mafioso results grouped by model.

    If background_filter is provided, use the benchmark table to focus on a
    specific capability/background slice (2-round dataset). Otherwise fall back
    to aggregating over the mafioso role in game_players (4-round dataset).
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        if background_filter is not None:
            query = """
                SELECT b.target AS model_name,
                       COUNT(*) AS total_games,
                       SUM(CASE WHEN g.winner = 'mafia' THEN 1 ELSE 0 END) AS mafia_wins
                FROM benchmark b
                JOIN games g ON b.game_id = g.game_id
                WHERE b.capability = ? AND b.background = ?
                GROUP BY b.target
                ORDER BY total_games DESC, b.target
            """
            params = (capability_filter or 'deceive', background_filter)
            rows = conn.execute(query, params).fetchall()
        else:
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


MODEL_DISPLAY_NAMES = {
    'gpt-5-mini': 'GPT-5 Mini',
    'gpt_5_mini': 'GPT-5 Mini',
    'deepseek-chat': 'DeepSeek V3.1',
    'deepseek_v3_1': 'DeepSeek V3.1',
    'claude-opus-4-1-20250805': 'Claude Opus 4.1',
    'claude_opus_4_1': 'Claude Opus 4.1',
    'claude-sonnet-4-20250514': 'Claude Sonnet 4',
    'claude_sonnet_4': 'Claude Sonnet 4',
    'gemini-2.5-flash-lite': 'Gemini 2.5 Flash Lite',
    'gemini_2_5_flash_lite': 'Gemini 2.5 Flash Lite',
    'grok-3-mini': 'Grok 3 Mini',
    'grok_3_mini': 'Grok 3 Mini',
    'gpt-4.1-mini': 'GPT-4.1 Mini',
    'gpt_4_1_mini': 'GPT-4.1 Mini',
    'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf': 'Mistral 7B Instruct',
    'mistral_7b_instruct': 'Mistral 7B Instruct',
    'Qwen2.5-7B-Instruct-Q4_K_M.gguf': 'Qwen2.5 7B Instruct',
    'qwen2_5_7b_instruct': 'Qwen2.5 7B Instruct',
    'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf': 'Llama 3.1 8B Instruct',
    'llama_3_1_8b_instruct': 'Llama 3.1 8B Instruct'
}


def display_name(model_key: str) -> str:
    """Convert internal model identifier to a readable label."""
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def compute_win_rates(stats):
    """Add Bayesian win rate + uncertainty to each stat row."""
    enriched = []
    for row in stats:
        total_games = int(row['total_games'])
        mafia_wins = int(row['mafia_wins'])
        win_rate, err = bayesian_win_rate(mafia_wins, total_games)
        row_copy = dict(row)
        row_copy['win_rate_percent'] = round(win_rate, 2)
        row_copy['uncertainty_percent'] = round(err, 2)
        enriched.append(row_copy)
    return enriched


def compute_z_scores(data):
    """Attach z-score/exp-score using the benchmark template."""
    if not data:
        return

    rates = np.array([row['win_rate_percent'] for row in data], dtype=float)
    errs = np.array([row['uncertainty_percent'] for row in data], dtype=float)

    mean = rates.mean()
    std = rates.std(ddof=1)
    if std == 0:
        std = 1.0

    for row, rate, err in zip(data, rates, errs):
        z = (rate - mean) / std
        z_err = err / std
        exp_score = np.exp(z)
        exp_err = exp_score * z_err

        row['z_score'] = float(round(z, 4))
        row['z_uncertainty'] = float(round(z_err, 4))
        row['exp_score'] = float(round(exp_score, 4))
        row['exp_uncertainty'] = float(round(exp_err, 4))


def create_plot(data):
    if not data:
        return

    models = [display_name(row['model_name']) for row in data]
    values = [row['win_rate_percent'] for row in data]
    errors = [row['uncertainty_percent'] for row in data]

    create_horizontal_bar_plot(
        models=models,
        values=values,
        errors=errors,
        xlabel='Mafia Win Rate (%)',
        filename=str(PLOT_PATH),
        color=ROUND4_COLOR,
        sort_ascending=True,
        show_reference_line=False
    )
    print(f"Saved plot: {PLOT_PATH}")


def create_exponential_z_plot(round4_data):
    if not round4_data:
        print('Skipping z-score plot (no 4-round data).')
        return

    models = [display_name(row['model_name']) for row in round4_data]
    exp_values = [row.get('exp_score', np.exp(row['z_score'])) for row in round4_data]
    exp_errors = [row.get('exp_uncertainty', row.get('z_uncertainty', 0.0)) for row in round4_data]

    create_horizontal_bar_plot(
        models=models,
        values=exp_values,
        errors=exp_errors,
        xlabel='exp(z)',
        filename=str(Z_PLOT_PATH),
        color=ORANGE_COLOR,
        sort_ascending=True,
        show_reference_line=True
    )
    print(f"Saved exp(z) plot: {Z_PLOT_PATH}")


def main():
    print('Creating 4-round mafioso win-rate plot...')
    stats = fetch_mafioso_stats(DB_PATH)
    if not stats:
        print('No mafioso records found in the 4-round database.')
        return

    data = compute_win_rates(stats)
    compute_z_scores(data)
    create_plot(data)

    # Baseline comparison (2-round dataset) -> used for data sanity but not plotting
    if BASELINE_DB_PATH.exists():
        baseline_stats = fetch_mafioso_stats(
            BASELINE_DB_PATH,
            capability_filter='deceive',
            background_filter='gpt_5_mini'
        )
        if baseline_stats:
            baseline_data = compute_win_rates(baseline_stats)
            compute_z_scores(baseline_data)
            # you can inspect baseline_data if needed
        else:
            print('Baseline database has no mafioso games; skipping baseline check.')
    else:
        print(f"Baseline database not found at {BASELINE_DB_PATH}; skipping baseline check.")

    create_exponential_z_plot(data)
    print('Done.')


if __name__ == '__main__':
    main()
