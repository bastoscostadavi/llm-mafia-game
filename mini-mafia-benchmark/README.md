# Mini-Mafia Benchmark

A streamlined four-player Mafia variant designed to systematically evaluate large language models' interactive capabilities across three key dimensions: deception, deception detection, and strategic information disclosure.

## Overview

Mini-Mafia reduces the classic Mafia game to its essential interactive elements, creating a controlled environment for measuring AI social intelligence. By fixing night actions and focusing on a single critical day phase, the benchmark isolates specific capabilities while maintaining the game's fundamental social dynamics.

## Game Design

### Players and Roles
- **1 Mafioso**: Must deceive other players to avoid detection
- **1 Detective**: Must effectively disclose investigation results to convince the villager
- **2 Villagers**: One is eliminated during the night; the survivor must detect deception

### Fixed Night Phase
- **Mafioso**: Eliminates one randomly selected villager
- **Detective**: Investigates the mafioso, learning their identity

### Day Phase
- **Discussion**: Two rounds of public communication in randomized order
- **Voting**: Blind voting to arrest one player; ties resolved randomly
- **Win Condition**: Town wins if mafioso is arrested; otherwise Mafia wins

### Information Asymmetry
- **Mafioso**: Knows who they eliminated (partial information)
- **Detective**: Knows the mafioso's identity (complete information)
- **Villager**: Has no special information (no information)

## Benchmark Methodology

### Experimental Design
The benchmark uses a **background** methodology where two roles are held constant while varying the third:

| Capability | Variable Role | Fixed Background |
|------------|---------------|------------------|
| **Deceive** | Mafioso | Detective + Villager |
| **Detect** | Villager | Detective + Mafioso |
| **Disclose** | Detective | Mafioso + Villager |

### Statistical Framework
1. **Win Rate Estimation**: Bayesian inference with Beta-Binomial model using Laplace rule of succession
2. **Cross-Background Aggregation**: Standardized z-scores across different opponent configurations
3. **Performance Scoring**: Exponential transformation for interpretability (α = e^z̄)

### Current Dataset
- **Models Evaluated**: 10 contemporary LLMs including Claude, GPT, Grok, DeepSeek, and others
- **Backgrounds**: 5 different opponent configurations per capability
- **Games per Configuration**: 100 games (5,000 total per capability)
- **Total Dataset**: 15,000 games across all three capabilities (14,000 unique games)

## Key Results

### Performance Scores
Performance scores where 1.0 represents average performance:

| Model | Deceive | Detect | Disclose |
|-------|---------|--------|----------|
| DeepSeek V3.1 | **3.13** | 2.13 | 1.68 |
| Grok 3 Mini | 2.05 | **6.70** | 1.90 |
| GPT-5 Mini | 0.73 | 0.66 | **2.07** |
| Claude Opus 4.1 | 2.20 | 1.98 | 1.92 |
| Claude Sonnet 4 | 1.86 | **0.48** | 1.74 |

### Key Findings
- **Model Specialization**: No single model dominates all dimensions
- **Size ≠ Performance**: Smaller models often outperform larger ones
- **Counterintuitive Results**: Claude Sonnet 4 worst at detection, Grok 3 Mini best

## Repository Structure

```
mini-mafia-benchmark/
├── README.md                    # This file
├── run_mini_mafia_batch.py     # Batch experiment execution
├── experiment_design_table.csv # Experimental configuration
├── database/
│   └── mini_mafia.db           # Complete experimental dataset
├── results/                    # Analysis scripts and outputs
│   ├── create_benchmark_plots.py
│   ├── scores_hierarchical_table_and_plot.py
│   ├── gender_bias.py
│   └── analyze_role_win_rates.py
└── article/                    # Research paper and LaTeX source
    ├── main.tex
    ├── main.pdf
    └── references.bib
```

## Usage

### Running Experiments

```bash
# Execute batch experiments
python run_mini_mafia_batch.py

# Create benchmark visualizations
cd results/
python create_benchmark_plots.py

# Generate hierarchical Bayesian analysis
python scores_hierarchical_table_and_plot.py
```

### Database Access

The `mini_mafia.db` SQLite database contains complete experimental data:

```python
import sqlite3
conn = sqlite3.connect('database/mini_mafia.db')

# Access game results
results = pd.read_sql_query("""
    SELECT b.capability, b.target, b.background, g.winner, COUNT(*) as games
    FROM benchmark b
    JOIN games g ON b.game_id = g.game_id
    GROUP BY b.capability, b.target, b.background, g.winner
""", conn)
```

### Analysis Scripts

- **`create_benchmark_plots.py`**: Generate performance score visualizations
- **`gender_bias.py`**: Analyze character name bias effects
- **`analyze_role_win_rates.py`**: Study last-speaker advantages
- **`scores_hierarchical_table_and_plot.py`**: Hierarchical Bayesian analysis

## Emergent Phenomena

### Gender Bias in Trust Attribution
Analysis across 14,000 games reveals systematic bias:
- **Male characters**: 55.06 ± 0.34% win rate
- **Female characters**: 54.66 ± 0.34% win rate
- **Male advantage**: 0.40 ± 0.48 percentage points

### Last-Speaker Advantage
Significant procedural advantages observed:
- **Detectives**: +7.10 ± 0.77 percentage points when speaking last
- **Mafiosos**: +6.04 ± 0.81 percentage points when speaking last
- **Villagers**: No significant advantage

## Research Applications

### AI Safety
- **Deception Tracking**: Monitor model capabilities relative to human baselines
- **Detection Training**: Generate training data for deception detection systems
- **Early Warning**: Identify concerning levels of social manipulation ability

### Multi-Agent Research
- **Emergent Behaviors**: Study spontaneous phenomena in AI-AI interactions
- **Social Dynamics**: Investigate bias, trust, and communication strategies
- **Capability Evolution**: Track how interactive abilities develop across model generations

### Benchmarking
- **Standardized Evaluation**: Consistent framework for comparing social intelligence
- **Capability Isolation**: Separate assessment of deception, detection, and disclosure
- **Cross-Model Analysis**: Systematic comparison across different AI architectures

## Future Directions

### Experimental Extensions
- **Complete Model Matrix**: Test all I³ possible model combinations
- **Extended Social Attributes**: Investigate additional bias dimensions beyond gender
- **Multi-Round Games**: Expand from Mini-Mafia to full Mafia gameplay

### Theoretical Development
- **Predictive Framework**: Develop mathematical models linking capabilities to outcomes
- **Functional Analysis**: Investigate p_ijk = f(α_deceive, α_detect, α_disclose)
- **Implementation Invariance**: Test capability measures across different game implementations

### Human Baselines
- **Comparative Analysis**: Establish human performance benchmarks
- **Safety Thresholds**: Identify concerning capability combinations
- **Validation Studies**: Confirm AI results translate to human interactions

## Citation

```bibtex
@article{costa2025minimafia,
  title={Deceive, Detect, and Disclose: Large Language Models Playing Mini-Mafia},
  author={Costa, Davi Bastos and Vicente, Renato},
  journal={arXiv preprint},
  year={2025},
  note={Code and data available at: \url{https://github.com/bastoscostadavi/llm-mafia-game}}
}
```

## Contact

- **Authors**: Davi Bastos Costa, Renato Vicente
- **Institution**: TELUS Digital Research Hub, University of São Paulo
- **Email**: davi.costa@usp.br
- **Paper**: [arXiv preprint](https://arxiv.org/abs/XXXX.XXXXX)