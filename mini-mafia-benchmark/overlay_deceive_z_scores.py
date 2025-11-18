#!/usr/bin/env python3
"""Overlay Deceive average z-score comparisons from short-prompt and article analyses."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
os.environ.setdefault('MPLBACKEND', 'Agg')

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
OUTPUT_PATH = RESULTS_DIR / 'overlay_deceive_z_scores.png'


def compute_short_prompt_z_scores() -> Dict[str, Tuple[float, float]]:
    raw_stats = fetch_background_stats(DB_PATH)
    enriched = enrich_stats(raw_stats)
    backgrounds, models, win_rates, uncertainties = build_model_matrices(enriched)
    _, z_scores = compute_aggregate_scores(backgrounds, models, win_rates, uncertainties)
    return {display_name(model): (score, err) for model, score, err in z_scores}


def compute_article_z_scores() -> Dict[str, Tuple[float, float]]:
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
    avg_z_scores, avg_z_errors, _, _ = calculate_aggregated_scores(win_rates_df, uncertainties_df, 'Deceive')

    return {
        str(model): (avg_z_scores.loc[model], avg_z_errors.loc[model])
        for model in avg_z_scores.index
    }


def plot_overlay(short_prompt: Dict[str, Tuple[float, float]],
                 article: Dict[str, Tuple[float, float]]) -> None:
    common_models = list(set(short_prompt) & set(article))
    if not common_models:
        raise ValueError('No overlapping models between datasets to plot.')

    # Order models by the article Deceive average z-score (ascending)
    common_models.sort(key=lambda m: article[m][0])

    y_pos = range(len(common_models))
    bar_height = 0.35

    short_scores = [short_prompt[m][0] for m in common_models]
    short_errs = [short_prompt[m][1] for m in common_models]
    article_scores = [article[m][0] for m in common_models]
    article_errs = [article[m][1] for m in common_models]
    labels = [display_name(m) for m in common_models]

    plt.ioff()
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.barh([y - bar_height/2 for y in y_pos], short_scores, height=bar_height,
            xerr=short_errs, label='Short-Prompt', color='#1f77b4', alpha=0.8, capsize=5)
    ax.barh([y + bar_height/2 for y in y_pos], article_scores, height=bar_height,
            xerr=article_errs, label='Article Deceive', color='#ff7f0e', alpha=0.8, capsize=5)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Average z-score')
    ax.set_title('Deceive Average z-score Comparison')
    ax.legend()
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', color='gray', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved overlay plot to {OUTPUT_PATH}')


def main():
    short_scores = compute_short_prompt_z_scores()
    article_scores = compute_article_z_scores()
    plot_overlay(short_scores, article_scores)


if __name__ == '__main__':
    main()
