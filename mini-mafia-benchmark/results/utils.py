#!/usr/bin/env python3
"""
Shared utilities for Mini-Mafia benchmark analysis scripts.

Contains common functions for Bayesian statistics, model name mappings,
and other shared functionality to reduce code duplication.
"""

import math
import matplotlib.pyplot as plt

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
        return 0.0, 0.0

    # Laplace rule of succession (Bayesian mean)
    bayesian_mean = (wins + 1) / (total_games + 2)

    # Bayesian uncertainty
    bayesian_var = (bayesian_mean * (1 - bayesian_mean)) / (total_games + 3)
    bayesian_std = math.sqrt(bayesian_var)

    # Convert to percentages
    return bayesian_mean * 100, bayesian_std * 100


_MODEL_DISPLAY_NAMES = {
    'claude_opus_4_1': 'Claude Opus 4.1',
    'claude_sonnet_4': 'Claude Sonnet 4',
    'deepseek_v3_1': 'DeepSeek V3.1',
    'gemini_2_5_flash_lite': 'Gemini 2.5 Flash Lite',
    'gpt_4_1_mini': 'GPT-4.1 Mini',
    'gpt_5_mini': 'GPT-5 Mini',
    'grok_3_mini': 'Grok 3 Mini',
    'llama_3_1_8b_instruct': 'Llama 3.1 8B Instruct',
    'mistral_7b_instruct': 'Mistral 7B Instruct',
    'qwen2_5_7b_instruct': 'Qwen2.5 7B Instruct'
}


def get_background_color(background_key):
    """Get color based on background model type"""
    background_lower = background_key.lower()

    if 'mistral' in background_lower:
        return '#FF6600'  # Mistral orange
    elif 'gpt-5' in background_lower and 'mini' in background_lower:
        return '#006B3C'  # GPT-5 Mini dark green
    elif 'gpt-4.1' in background_lower and 'mini' in background_lower:
        return '#10A37F'  # GPT-4.1 Mini green
    elif 'grok' in background_lower and 'mini' in background_lower:
        return '#8A2BE2'  # Grok Mini purple
    elif 'deepseek' in background_lower:
        return '#007ACC'  # DeepSeek blue
    else:
        return '#666666'  # Default gray


def get_display_name(model_key):
    """Convert internal model name to display name"""
    return _MODEL_DISPLAY_NAMES.get(model_key, model_key)


def configure_plot_style():
    """Configure matplotlib with consistent style settings for all plots"""
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
        'figure.titlesize': 24
    })


def create_horizontal_bar_plot(models, values, errors, xlabel, filename,
                             color='#E74C3C', sort_ascending=True, show_reference_line=True,
                             text_offset=0.05, reverse_after_sort=False,
                             x_min=None, x_max=None, reference_lines=None):
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
        reference_lines: List of tuples (value, label, color, linestyle) for custom reference lines

    Returns:
        filename: The saved plot filename
    """
    # Configure consistent styling
    configure_plot_style()

    # Sort data if requested
    if sort_ascending is not None:
        sorted_data = sorted(zip(models, values, errors), key=lambda x: x[1], reverse=not sort_ascending)
        if reverse_after_sort:
            sorted_data = list(reversed(sorted_data))
        sorted_models = [x[0] for x in sorted_data]
        sorted_values = [x[1] for x in sorted_data]
        sorted_errors = [x[2] for x in sorted_data]
    else:
        sorted_models, sorted_values, sorted_errors = models, values, errors

    # Create plot
    plt.ioff()  # Use non-interactive backend
    fig, ax = plt.subplots(figsize=(14, 7))

    y_positions = range(len(sorted_models))

    # Create bars with error bars
    bars = ax.barh(y_positions, sorted_values, xerr=sorted_errors,
                   color=color, alpha=0.8, height=0.6,
                   error_kw={'capsize': 5, 'capthick': 2})

    # Add model names on the right side of bars
    for i, (model, value, error) in enumerate(zip(sorted_models, sorted_values, sorted_errors)):
        x_pos = max(value + error + text_offset, text_offset)
        ax.text(x_pos, i, model, ha='left', va='center',
                fontweight='bold', fontsize=24)

    # Set axis labels and remove y-axis ticks
    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold')
    ax.set_yticks([])

    # Set data-driven x-axis limits with padding
    max_val = max([v + e for v, e in zip(sorted_values, sorted_errors)])
    min_val = min([v - e for v, e in zip(sorted_values, sorted_errors)])
    padding = (max_val - min_val) * 0.1
    axis_min = x_min if x_min is not None else min_val - padding
    axis_max = x_max if x_max is not None else max_val + padding
    if axis_min > axis_max:
        axis_min, axis_max = axis_max, axis_min
    ax.set_xlim(axis_min, axis_max)

    # Add reference line at 1 if requested and in range
    if show_reference_line and axis_min is not None and axis_max is not None:
        if axis_min <= 1 <= axis_max:
            ax.axvline(x=1, color='gray', alpha=0.7, linewidth=2, linestyle='--')

    # Add custom reference lines if provided
    if reference_lines is not None:
        for line_spec in reference_lines:
            if len(line_spec) >= 2:
                value, label = line_spec[0], line_spec[1]
                line_color = line_spec[2] if len(line_spec) > 2 else 'gray'
                linestyle = line_spec[3] if len(line_spec) > 3 else '--'

                if axis_min is not None and axis_max is not None and axis_min <= value <= axis_max:
                    ax.axvline(x=value, color=line_color, alpha=0.7, linewidth=2, linestyle=linestyle, label=label)

    # Grid and styling
    ax.grid(True, axis='x', color='gray', alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Add legend if there are labeled reference lines
    if reference_lines is not None and any(len(line) >= 2 and line[1] for line in reference_lines):
        # Only add legend if there are actually labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='best', fontsize=20, framealpha=0.9)

    plt.tight_layout()

    # Save and close
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    return filename
