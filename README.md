# LLM Mafia Game

A research framework for studying Large Language Model behavior in social deduction games. This system allows LLMs to play Mafia/Werewolf with different roles, providing insights into AI reasoning, deception, and social dynamics.

## Overview

This project implements a multi-agent Mafia game where different LLMs can play various roles:
- **Detective**: Investigates players each night to identify mafiosos
- **Mafioso**: Tries to eliminate good players while avoiding detection
- **Villager**: Participates in discussions and voting to identify mafiosos

The framework supports multiple LLM backends and includes comprehensive analysis tools for studying gameplay patterns.

## Features

- **Multi-LLM Support**: Local models (llama.cpp), OpenAI GPT, and Anthropic Claude
- **Flexible Game Modes**: Classic 6-player and Mini-mafia 4-player variants
- **Research Tools**: Batch experiment runner, game viewer, and statistical analysis
- **Prompt Versioning**: Reproducible research with versioned prompt configurations
- **Memory System**: Each agent maintains individual memory of game events

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# For local models, place GGUF files in models/ directory:
# - models/mistral.gguf
# - models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
# - models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# For API models, set environment variables:
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Run a Game

```bash
# Interactive game launcher with preset configurations
python run_game.py

# Available options:
# 1. Classic Game (6 players: 2 mafiosos, 1 detective, 3 villagers)
# 2. Mini-mafia (4 players: 1 mafioso, 1 detective, 2 villagers)
```

### Research Experiments

```bash
# Run batch experiments (Mini-mafia)
cd experiments/mini-mafia

# Run 100 games with current configuration
python run_mini_mafia_batch.py 100

# View results
python game_viewer.py                    # Interactive viewer
python analyze_results.py                # Win rate analysis
python analyze_voting.py                 # Voting pattern analysis
```

## Project Structure

```
llm-mafia-game/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_game.py                  # Main game launcher
├── preset_games.py              # Predefined game configurations
├── src/                         # Core game engine
│   ├── main.py                  # Game logic and state management
│   ├── agents.py                # Agent classes and behaviors
│   ├── llm_utils.py             # LLM wrapper utilities
│   └── prompts.py               # Prompt templates and versioning
├── models/                      # Local model files (GGUF format)
└── experiments/                 # Research experiments
    └── mini-mafia/              # Mini-mafia specific experiments
        ├── README.md            # Experiment documentation
        ├── run_mini_mafia_batch.py  # Batch runner
        ├── game_viewer.py       # Game visualization tool
        ├── analyze_results.py   # Statistical analysis
        ├── analyze_voting.py    # Voting pattern analysis
        └── data/                # Experimental data
            └── batch_*/         # Individual batch results
```

## Game Modes

### Classic Mode (6 Players)
- **Roles**: 2 Mafiosos, 1 Detective, 3 Villagers
- **Gameplay**: Full night/day cycle with eliminations
- **Win Conditions**: 
  - Good wins: Arrest all mafiosos
  - Evil wins: Equal or outnumber good players

### Mini-Mafia Mode (4 Players)
- **Setup**: 1 Mafioso, 1 Detective, 2 Villagers (1 killed at night)
- **Gameplay**: Single day phase with 3 surviving players
- **Research Focus**: Optimized for studying deception and detection patterns

## LLM Backends

### Local Models (via llama.cpp)
- Fast inference with GPU acceleration
- No API costs or rate limits
- Supports any GGUF format model

### API Models
- **OpenAI**: GPT-3.5, GPT-4 series
- **Anthropic**: Claude-3 series (Haiku, Sonnet, Opus)
- Configurable per role for comparative studies

## Research Applications

This framework enables research into:

- **Deception Detection**: How well can LLMs identify lies and inconsistencies?
- **Strategic Reasoning**: Do different models employ different strategies?
- **Social Dynamics**: How do models adapt their behavior in group settings?
- **Model Comparison**: Comparative analysis across different LLM architectures
- **Prompt Engineering**: Impact of different prompt formulations on gameplay

## Data and Analysis

The system generates rich datasets including:
- Complete game transcripts with all communications
- Individual agent memory states
- Voting patterns and decision rationales
- Win/loss statistics across different configurations

Analysis tools provide insights into:
- Model-specific behavioral patterns
- Voting accuracy and strategic choices  
- Communication styles and persuasion attempts
- Success rates across different prompt versions

## Configuration

### Prompt Versioning
```python
# Available prompt versions
v0.0: Original research prompts
v1.0: Enhanced with strategic guidance
v2.0: Latest with improved clarity and structure
```

### Model Configuration
```python
model_configs = {
    'detective': {'type': 'local', 'model_path': 'models/mistral.gguf'},
    'mafioso': {'type': 'openai', 'model': 'gpt-3.5-turbo'},
    'villager': {'type': 'anthropic', 'model': 'claude-3-haiku-20240307'}
}
```

## Contributing

This is a research framework. Contributions welcome for:
- New game variants or mechanics
- Additional LLM backend integrations  
- Enhanced analysis tools
- Performance optimizations


## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{llm-mafia-game,
  title={LLM Mafia Game: A Framework for Studying AI Behavior in Social Deduction},
  author={Davi Bastos Costa},
  year={2025},
  url={https://github.com/bastoscostadavi/llm-mafia-game}
}
```