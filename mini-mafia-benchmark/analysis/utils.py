"""
Shared utility functions for Mini-Mafia analysis scripts.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def bayesian_win_rate(wins, total_games):
    """
    Calculate Bayesian win rate and uncertainty using Beta distribution.

    Uses uniform prior Beta(1,1) which leads to posterior Beta(wins+1, losses+1).
    This implements the Laplace rule of succession.

    Args:
        wins: Number of wins
        total_games: Total number of games played

    Returns:
        tuple: (win_rate_percent, uncertainty_percent)
    """
    if total_games == 0:
        return (0.0, 0.0)

    a = wins + 1
    b = total_games - wins + 1
    n = a + b  # = total_games + 2

    mean = a / n
    variance = (a * b) / (n ** 2 * (n + 1))
    std = np.sqrt(variance)

    return (mean * 100, std * 100)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_DISPLAY_NAMES = {
    'gpt-5-mini':          'GPT-5 Mini',
    'gpt-4o':              'GPT-4o',
    'gpt-4.1-mini':        'GPT-4.1 Mini',
    'claude-sonnet':       'Claude Sonnet 4',
    'claude-opus':         'Claude Opus 4',
    'grok-3-mini':         'Grok-3 Mini',
    'deepseek-v3':         'DeepSeek V3.1',
    'gemini-2.5-flash':    'Gemini 2.5 Flash',
    'mistral-7b':          'Mistral 7B',
    'llama-3.1-8b':        'Llama 3.1 8B',
}


def get_display_name(model_key):
    """Convert internal model name to display name."""
    return _DISPLAY_NAMES.get(model_key, model_key)


def get_background_color(background_key):
    """Get color based on background model type."""
    key = background_key.lower()
    if 'mistral' in key:
        return '#FF6600'
    if 'gpt-5' in key and 'mini' in key:
        return '#006B3C'
    if 'gpt-4.1' in key:
        return '#10A37F'
    if 'grok' in key:
        return '#8A2BE2'
    if 'deepseek' in key:
        return '#007ACC'
    return '#666666'


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def configure_plot_style():
    """Configure matplotlib with consistent style settings for all plots."""
    size = 24
    for key in ('font.size', 'axes.labelsize', 'axes.titlesize',
                'xtick.labelsize', 'ytick.labelsize',
                'legend.fontsize', 'figure.titlesize'):
        plt.rcParams[key] = size


def create_horizontal_bar_plot(
    models, values, errors, xlabel, filename,
    color='red',
    sort_ascending=True,
    show_reference_line=True,
    text_offset=0.1,
    reverse_after_sort=False,
    x_min=None,
    x_max=None,
    reference_lines=None,
):
    """
    Create a standardized horizontal bar plot with error bars.

    Args:
        models: List of model names
        values: List of values to plot
        errors: List of error values for error bars
        xlabel: Label for x-axis
        filename: Output filename
        color: Bar color (default: red)
        sort_ascending: Whether to sort by ascending values (default: True)
        show_reference_line: Whether to show reference line at x=1 (default: True)
        reference_lines: List of tuples (value, label, color, linestyle) for
                         custom reference lines

    Returns:
        filename: The saved plot filename
    """
    configure_plot_style()

    # Sort data
    paired = sorted(zip(values, errors, models), key=lambda x: x[0],
                    reverse=not sort_ascending)
    if reverse_after_sort:
        paired = list(reversed(paired))
    values_s, errors_s, models_s = zip(*paired)

    fig, ax = plt.subplots(figsize=(14, 7))

    y_pos = range(len(models_s))
    ax.barh(y_pos, values_s, xerr=errors_s, color=color, alpha=0.8,
            height=0.6, capsize=5, capthick=2,
            error_kw={'capsize': 5, 'capthick': 2})

    # Model labels
    for i, (model, val) in enumerate(zip(models_s, values_s)):
        ax.text(val + text_offset, i, get_display_name(model),
                ha='left', va='center', fontweight='bold', fontsize=24)

    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold')
    ax.set_yticks([])

    if show_reference_line:
        ax.axvline(x=1, color='gray', alpha=0.7, linewidth=3, linestyle='--')

    if reference_lines:
        for ref in reference_lines:
            val, label, ref_color, linestyle = ref
            ax.axvline(x=val, color=ref_color, alpha=0.7, linewidth=3,
                       linestyle=linestyle, label=label)
        ax.legend(loc='best', fontsize=20, framealpha=0.9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set(color='gray', alpha=0.7, linewidth=3)

    if x_min is not None or x_max is not None:
        ax.set_xlim(left=x_min, right=x_max)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    return filename
