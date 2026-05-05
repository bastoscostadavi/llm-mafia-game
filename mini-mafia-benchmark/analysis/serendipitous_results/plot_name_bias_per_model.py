#!/usr/bin/env python3
"""
Heatmap of per-model name bias in Mini-Mafia.
Rows = models, columns = character names, color = deviation from model mean.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "serendipitous_results"

DISPLAY_NAMES = {
    'claude_opus_4_1':        'Claude Opus 4.1',
    'claude_sonnet_4':        'Claude Sonnet 4',
    'deepseek_v3_1':          'DeepSeek V3.1',
    'gemini_2_5_flash_lite':  'Gemini 2.5 Flash',
    'gpt_4_1_mini':           'GPT-4.1 Mini',
    'gpt_5_mini':             'GPT-5 Mini',
    'grok_3_mini':            'Grok-3 Mini',
    'llama_3_1_8b_instruct':  'Llama 3.1 8B',
    'mistral_7b_instruct':    'Mistral 7B',
    'qwen2_5_7b_instruct':    'Qwen 2.5 7B',
}

NAMES = ['Alice', 'Bob', 'Charlie', 'Diana']


def compute_centered_matrix(df):
    """
    TODO(human): Given df with columns [model, character_name, win_rate_bayesian],
    return a (models x names) DataFrame of centered values.

    Each row should be the win rate minus that model's mean across names,
    then multiplied by 100 to express as percentage points.

    Args:
        df: raw per-model name bias dataframe

    Returns:
        centered: DataFrame with models as index, character names as columns,
                  values in percentage points relative to each model's mean
    """
    raw = df.pivot(index='model', columns='character_name', values='win_rate_bayesian')[NAMES]
    centered = (raw.subtract(raw.mean(axis=1), axis=0)) * 100
    return centered


def plot_heatmap(centered, raw, err, output_path):
    models = centered.index.tolist()
    model_labels = [DISPLAY_NAMES.get(m, m) for m in models]

    # Color scale symmetric around 0
    vmax = np.abs(centered.values).max()
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = 'RdBu_r'

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(centered.values, cmap=cmap, norm=norm, aspect='auto')

    # Axes labels
    ax.set_xticks(range(len(NAMES)))
    ax.set_xticklabels(NAMES, fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_labels, fontsize=12)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Annotate each cell: raw win rate ± 1.96*std, then centered delta
    for i, model in enumerate(models):
        for j, name in enumerate(NAMES):
            raw_val = raw.loc[model, name] * 100
            ci = err.loc[model, name] * 100 * 1.96   # 95% half-width
            delta = centered.loc[model, name]
            sign = '+' if delta >= 0 else ''
            text_color = 'white' if abs(delta) > vmax * 0.55 else 'black'
            ax.text(j, i, f'{raw_val:.1f}±{ci:.1f}%\n({sign}{delta:.1f}pp)',
                    ha='center', va='center', fontsize=8.5,
                    color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Deviation from model mean (pp)', fontsize=11)

    ax.set_title('Name bias per model\n(deviation from each model\'s mean win rate)',
                 fontsize=13, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    df = pd.read_csv(RESULTS_DIR / "name_bias_per_model.csv")

    # Build raw (model x name) win rate and uncertainty matrices
    raw = df.pivot(index='model', columns='character_name', values='win_rate_bayesian')[NAMES]
    err = df.pivot(index='model', columns='character_name', values='uncertainty_std')[NAMES]

    centered = compute_centered_matrix(df)
    if centered is None:
        print("compute_centered_matrix returned None — implement the TODO first.")
        return

    output_path = RESULTS_DIR / "name_bias_per_model_heatmap.png"
    plot_heatmap(centered, raw, err, output_path)


if __name__ == "__main__":
    main()
