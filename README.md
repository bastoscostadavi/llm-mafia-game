# Mafia LLM Game

A multi-agent social deduction game where Large Language Models play Mafia against each other.

## Features
- **Simple Interface**: Clean preset games and flexible custom game creation
- **Multi-LLM Support**: Use different models for different players
- **Game Types**: Classic (6 players) and Mini-mafia (3-4 players) presets
- **Core roles**: Assassin, Detective, Villager
- **Strategic communication**: 200-character message limits for realistic gameplay
- **Memory system**: Each agent remembers game events and conversations

## Supported LLM Types
- **Local Models**: llama.cpp (Mistral, Llama, etc.)
- **OpenAI**: GPT-3.5, GPT-4, and other OpenAI models (requires API key)
- **Anthropic**: Claude 3 (Haiku, Sonnet, Opus) (requires API key)

## Quick Start

**Play preset games:**
```bash
python run_game.py
```

**Options:**
1. **Classic** (6 players, all local) - 2 assassins, 1 detective, 3 villagers
2. **Mini-mafia** (3 players, all local) - 1 assassin, 1 detective, 1 villager (single day phase)

## Setup

### Installation
```bash
# Install required package
pip install llama-cpp-python

# For API models (optional)
pip install openai anthropic
```

### Model Setup
```bash
# Make sure you have a model file
ls models/mistral.gguf

# For API models, set environment variables
export OPENAI_API_KEY="your-key-here"      # Optional
export ANTHROPIC_API_KEY="your-key-here"   # Optional
```

## Custom Games

Create custom games using the `create_game()` function:

```python
from src.main import create_game

# Define players with roles and LLMs
players = [
    {'name': 'Alice', 'role': 'detective', 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
    {'name': 'Bob', 'role': 'assassin', 'llm': {'type': 'openai', 'model': 'gpt-4'}},
    {'name': 'Charlie', 'role': 'villager', 'llm': {'type': 'anthropic', 'model': 'claude-3-sonnet'}},
]

# Create and play the game
game = create_game(players, discussion_rounds=2)
game.play()
```

**LLM Configuration:**
- `'local'`: Uses local llama.cpp model (requires `model_path`)
- `'openai'`: Uses OpenAI API (requires `model` and API key in environment)
- `'anthropic'`: Uses Anthropic API (requires `model` and API key in environment)

## Game Roles

- **Assassin**: Works with other assassins to eliminate good players at night
- **Detective**: Investigates players at night to learn their alignment  
- **Villager**: No special abilities, helps vote out evil players during day

## Win Conditions

- **Good Team**: Eliminate all assassins by voting them out during day phases
- **Assassins**: Reduce good players to equal or fewer than assassins

## Research Experiments

Run and analyze batch experiments:

```bash
cd experiments

# Run mini-mafia experiments
python run_mini_mafia_batch.py 10

# View results
python game_viewer.py
```

Features:
- Batch experiment runner for reproducible studies
- Interactive game viewer with detailed breakdowns
- Automatic statistics and win rate analysis
- Organized data storage by experiment type

## Research Use

This system makes it easy to conduct LLM behavior research:

### Key Benefits
- **Head-to-head comparisons** between different LLMs in identical scenarios
- **Behavioral analysis** of reasoning patterns across models  
- **Cost optimization** by mixing local and API models
- **Reproducible experiments** with simple configuration
- **Automatic saving** in research scripts (games are not saved during casual play)

### Architecture
The system is designed for research simplicity:
- **Simple entry point**: `python run_game.py` for casual play
- **Flexible game creation**: `create_game()` function for custom experiments
- **Minimal dependencies**: Just llama-cpp-python + optional API packages
- **Clean separation**: Game logic separate from LLM interfaces
- **No bloat**: Casual games don't create save files

## Project Structure

```
mafia_game/
├── run_game.py              # Main entry point
├── preset_games.py          # Preset game configurations
├── src/main.py              # Core game engine with create_game()
├── src/agents/              # Agent and LLM interface code
├── src/day_phase.py         # Day phase logic
├── src/night_phase.py       # Night phase logic
└── experiments/
    ├── run_mini_mafia_batch.py  # Mini-mafia batch experiments
    ├── game_viewer.py           # Interactive results viewer
    └── data/
        └── mini_mafia/          # Saved experiment data
```

## License

[Add your license information here]