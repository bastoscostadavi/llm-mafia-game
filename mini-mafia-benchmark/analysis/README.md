# Analysis Scripts

This directory contains all analysis scripts organized by research methodology from the paper "Deceive, Detect, and Disclose: Large Language Models Playing Mini-Mafia".

## Directory Structure

```
analysis/
├── bottom_up_background/      # Bottom-up background methodology (Section 4.1)
├── top_down_theoretical/       # Top-down theoretical model (Section 4.2)
├── serendipitous_results/      # Emergent phenomena (Section 5)
├── ablation_experiments/       # Robustness checks (Section 6)
└── utils.py                    # Shared utilities (Bayesian stats, plotting)
```

## Methodologies

### Bottom-up Background Approach
Evaluates capabilities by varying the target model against fixed backgrounds.
- **Scripts**: 3 sequential scripts
- **Outputs**: Win counts, win rates, z-scores, and exponential scores
- **Location**: `bottom_up_background/`

### Top-down Theoretical Model
Fits the theoretical model: `logit(p_ijk) = v_k × (m_i - d_j)` using Bayesian inference.
- **Scripts**: 1 comprehensive script
- **Outputs**: Model parameters (m, d, v) for all models
- **Location**: `top_down_theoretical/`

### Serendipitous Results
Analyzes emergent phenomena observed during experiments.
- **Scripts**: 2 independent scripts
- **Outputs**: Name bias and last-speaker advantage analyses
- **Location**: `serendipitous_results/`

### Ablation Experiments
Tests robustness across different experimental conditions.
- **Scripts**: 1 comparison script
- **Outputs**: Overlay plots comparing conditions
- **Location**: `ablation_experiments/`

## Running Analysis

Each methodology folder contains its own README with detailed instructions. Generally:

```bash
# Bottom-up approach (run in sequence)
cd bottom_up_background/
python3 win_counts_table.py
python3 win_rates_table_and_plot.py
python3 scores_table_and_plot.py

# Top-down theoretical model
cd ../top_down_theoretical/
python3 scores_theoretical_model.py

# Serendipitous results
cd ../serendipitous_results/
python3 name_bias.py
python3 last_speaker_advantage.py
```

## Outputs

All analysis outputs are saved to `../results/` with the same folder structure:
- `../results/bottom_up_background/` - 25 files (CSVs + PNGs)
- `../results/top_down_theoretical/` - 5 files
- `../results/serendipitous_results/` - 2 CSVs
- `../results/ablation_experiments/` - 4 files

## Dependencies

All scripts require:
- Python 3.8+
- pandas, numpy, matplotlib
- sqlite3 (built-in)

Top-down theoretical model additionally requires:
- pymc (optional, falls back to maximum likelihood if unavailable)
- arviz (optional)

Install dependencies:
```bash
pip install pandas numpy matplotlib pymc arviz
```

## Shared Utilities

`utils.py` contains common functions:
- `bayesian_win_rate()` - Bayesian win rate estimation with Laplace rule of succession
- `get_display_name()` - Convert internal model names to display names
- `get_background_color()` - Consistent color mapping for plots
- `create_horizontal_bar_plot()` - Standardized plotting function

## Paper Sections

- **Section 3**: Experimental methodology → `bottom_up_background/`
- **Section 4.1**: Background-based results → `bottom_up_background/`
- **Section 4.2**: Theoretical model → `top_down_theoretical/`
- **Section 5**: Emergent phenomena → `serendipitous_results/`
- **Section 6**: Ablation studies → `ablation_experiments/`
