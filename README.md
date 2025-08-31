# LLM Mafia Game

Research framework for studying Large Language Model behavior in social deduction games. LLMs play different roles (Detective, Mafioso, Villager) with asymmetric information, enabling analysis of reasoning, deception, and social dynamics.

## Structure

```
llm-mafia-game/
├── src/                    # Core game engine
│   ├── agents.py          # Game agents (Detective, Mafioso, Villager)
│   ├── prompt.txt         # Game prompt template
│   ├── prompt_utils.py    # Prompt formatting utilities
│   ├── config.py          # Model configurations
│   ├── main.py            # Game state management
│   └── llm_utils.py       # LLM interface wrappers
├── experiments/           # Research experiments
│   └── mini-mafia/        # 4-player Mafia variant
├── models/                # Local model files (.gguf)
├── run_game.py           # Interactive game launcher
└── preset_games.py       # Predefined game configurations
```

## Quick Start

```bash
# Interactive game menu
python run_game.py

# Run mini-mafia experiment
cd experiments/mini-mafia
python run_mini_mafia_batch.py 10
```

## Models Supported

**Local**: Mistral 7B, Llama 3.1 8B, Qwen 2.5 7B, Gemma 2 27B, GPT-OSS 20B  
**API**: GPT-4o/5, Claude Sonnet/Opus, Grok-3/4, DeepSeek V3, Gemini 2.5

## Game Types

**Classic**: 6 players (2 mafiosos, 1 detective, 3 villagers)  
**Mini-Mafia**: 4 players (1 mafioso, 1 detective, 2 villagers)

## Features

- **Batch Experiments**: Run hundreds of games with automated data collection
- **SQLite Storage**: Structured database with games, actions, votes, outcomes  
- **Analysis Tools**: Performance benchmarks, bias detection, statistical plots
- **Research Output**: Automated paper generation with LaTeX integration
- **Interactive Tools**: Game viewer, database browser, real-time plotting

## Configuration

Edit `src/config.py` for default model settings:

```python
DEFAULT_MODEL_CONFIGS = {
    'detective': {'type': 'xai', 'model': 'grok-3-mini'},
    'mafioso': {'type': 'anthropic', 'model': 'claude-sonnet-4'},
    'villager': {'type': 'xai', 'model': 'grok-3-mini'}
}
```

## Requirements

- Python 3.8+
- API keys for cloud models (optional)
- Local models in `models/` directory (optional)