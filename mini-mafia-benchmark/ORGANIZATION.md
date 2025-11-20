# Project Organization

This project is organized into distinct directories based on functionality:

## Directory Structure

```
mini-mafia-benchmark/
├── mini_mafia.py              # Core game logic (shared by all)
├── web/                       # Teaching & human gameplay
│   ├── web_interface.py       # Flask web server
│   ├── setup_web_game.py      # Configure AI opponents
│   ├── view_human_game.py     # View human game data
│   ├── web_game_config.json   # Current configuration
│   └── SIMPLE_START.md        # Quick start guide
├── experiments/               # AI benchmarks & experiments
│   ├── run_mini_mafia_batch.py
│   ├── run_short_prompt_experiment.py
│   └── view_game.py           # View AI benchmark games
├── analysis/                  # Data analysis & plotting
│   ├── *_analysis.py          # Statistical analysis scripts
│   └── *_plot.py              # Visualization scripts
├── database/                  # All SQLite databases
│   ├── mini_mafia_human.db    # Human gameplay data
│   └── mini_mafia*.db         # AI benchmark data
└── article/                   # Research paper & presentation
```

## Quick Start Guides

### For Teaching (Web Interface)
See `web/SIMPLE_START.md` for instructions on running the web interface for students.

```bash
cd web/
python3 setup_web_game.py    # Configure AI opponents
python3 web_interface.py     # Start server
```

### For Research (Benchmarks)
See main `README.md` for instructions on running experiments.

```bash
cd experiments/
python3 run_mini_mafia_batch.py
```

### For Analysis
```bash
cd analysis/
python3 short_prompt_analysis.py
python3 win_rates_round8_plot.py
```

## Data Access

**View human gameplay:**
```bash
cd web/
python3 view_human_game.py --list
python3 view_human_game.py 1
```

**View AI benchmarks:**
```bash
cd experiments/
python3 view_game.py --list
```

## Key Files

- **mini_mafia.py**: Core game implementation, used by both web and experiments
- **web/web_interface.py**: Flask app for human gameplay
- **experiments/run_*.py**: Batch experiment runners
- **database/**: All game data (human + AI experiments)
