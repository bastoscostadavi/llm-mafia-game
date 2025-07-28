# Mafia LLM Game

A multi-agent social deduction game where Large Language Models play Mafia against each other.

## Features
- **Simple Interface**: Clean preset games and flexible custom game creation
- **Multi-LLM Support**: Use different models for different players
- **Game Types**: Classic (8 players) and Simple (4 players) presets
- Multiple roles: Assassin, Detective, Angel, Psychopath, Villager
- Strategic communication between agents
- Memory system for each agent

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
1. **Classic** (8 players, all local) - Traditional full game
2. **Simple** (4 players, all local) - Quick game with detective advantage

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
- **Angel**: Protects players from assassination at night
- **Psychopath**: Works alone, kills at night, wins by being last alive
- **Villager**: No special abilities, helps vote out evil players during day

## Win Conditions

- **Good Team**: Eliminate all evil players (assassins + psychopath)
- **Assassins**: Reduce good players to equal or fewer than assassins
- **Psychopath**: Be the last player alive

## Viewing Saved Games

Analyze saved research data:

```bash
cd experiments/analysis
python view_games.py
```

Features:
- List recent games
- View specific games with detailed breakdown
- Search games by criteria
- Show win/loss statistics

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
├── src/game/phases/         # Day and night phase logic
└── experiments/
    ├── analysis/
    │   └── view_games.py    # Game analysis and viewing
    └── results/             # Saved research data
```

## License

[Add your license information here]