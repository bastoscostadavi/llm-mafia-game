# Mini-Mafia Experiment  

4-player Mafia variant designed for rapid experimentation and statistical analysis. Tests LLM capabilities in deception (Mafioso), detection (Villager), and disclosure (Detective).

## Game Setup

- **Players**: 4 (Alice, Bob, Charlie, Diana)
- **Roles**: 1 Detective, 1 Mafioso, 2 Villagers
- **Pre-game**: 1 villager eliminated, detective investigates mafioso
- **Gameplay**: Single day phase with discussion and voting
- **Win**: Town wins if mafioso arrested, Mafia wins if innocent arrested

## Usage

```bash
# Single game (interactive)
python mini_mafia.py

# Batch experiments (100 games)
python run_mini_mafia_batch.py 100

# Debug mode (shows all prompts/responses)
python run_mini_mafia_batch.py 10 --debug

# Custom temperature setting
python run_mini_mafia_batch.py 50 --temperature 0.9

# Interactive mode with prompts
python run_mini_mafia_batch.py --interactive
```

## Database

SQLite database stores game results in `database/mini_mafia.db`:
- Games and outcomes
- Player actions and responses
- Voting patterns
- Character assignments

## Analysis Tools

**Visualization**:
- `create_benchmark_plots.py` - Role-specific performance benchmarks
- `create_character_win_rates_plots.py` - Character bias analysis
- `create_aggregated_score_plots.py` - Trust score distributions

**Data Inspection**:
- `view_game.py` - Individual game transcript viewer
- `games_table.py` - Interactive database browser

**Outputs**: PNG plots, JSON data, statistical summaries, research paper

## Paper

Research paper and analysis in `article/main.tex`.