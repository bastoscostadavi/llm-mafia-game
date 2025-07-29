# Mafia Game Experiments

This folder contains experimental scripts and tools for studying LLM behavior in Mafia games.

## 📁 Folder Structure

```
experiments/
├── run_mini_mafia_batch.py    # Batch experiment runner
├── game_viewer.py             # Game viewer for batch results
├── data/
│   └── mini_mafia/            # Mini-mafia experiment results
│       ├── batch_YYYYMMDD_HHMMSS_game_NNNN.json
│       └── batch_YYYYMMDD_HHMMSS_summary.json
└── README.md                  # This file
```

## 🎮 Available Experiments

### Mini-Mafia Batch Runner (`run_mini_mafia_batch.py`)
Runs N mini-mafia games and saves results for analysis.

- **Setup**: 1 Detective, 1 Assassin, 2 Villagers (1 killed before day phase)
- **Special Rule**: Detective and Assassin know each other from start
- **Output**: Individual game logs + batch summary statistics

**Usage:**
```bash
# Run 10 games silently
python run_mini_mafia_batch.py 10

# Run 5 games with debug prompts
python run_mini_mafia_batch.py 5 --debug

# Interactive mode
python run_mini_mafia_batch.py 0 --interactive
```

## 🔍 Analysis Tools

### Game Viewer (`game_viewer.py`)
Interactive tool for browsing batch experiment results.

**Usage:**
```bash
# Interactive mode
python game_viewer.py

# View batch summary
python game_viewer.py batch_20250729_141943

# View specific game
python game_viewer.py batch_20250729_141943 0
```


## ⚡ Quick Start

1. **Run an experiment:**
   ```bash
   python run_mini_mafia_batch.py 10
   ```

2. **View the results:**
   ```bash
   python game_viewer.py
   ```

## 📝 Data Format

Each batch creates:
- **Game files**: `batch_YYYYMMDD_HHMMSS_game_NNNN.json` - Complete game logs with player memories
- **Summary files**: `batch_YYYYMMDD_HHMMSS_summary.json` - Batch statistics and win rates

All files are saved in the `data/mini_mafia/` directory for easy analysis.