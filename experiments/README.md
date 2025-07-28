# Mafia LLM Experiments

This directory contains all research experiments for the "Deceive, Disclose and Detect: Large Language Models Playing Mafia" paper.

## Structure

- `scripts/` - Experiment runner scripts
- `configs/` - Experiment configuration files 
- `results/` - Organized experimental results
  - `bias_studies/` - Bias analysis results
  - `permutation_tests/` - Permutation test results
  - `baseline_comparisons/` - Baseline comparison results
  - `game_logs/` - Individual game transcripts
- `analysis/` - Analysis and visualization scripts

## Running Experiments

From the project root directory:

```bash
# Run batch experiments
python experiments/scripts/batch_experiment.py

# Run bias analysis
python experiments/scripts/bias_analysis.py

# Run permutation tests
python experiments/scripts/permutation_test.py

# Run baseline comparisons
python experiments/scripts/baseline_comparison.py
```

## Results

All results are automatically saved to the appropriate subdirectories in `results/`.