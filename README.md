# LLM Mafia Game

A comprehensive research framework for studying Large Language Models (LLMs) in social deduction scenarios through the classic Mafia party game. This system enables systematic investigation of AI social intelligence, theory of mind, and strategic reasoning capabilities.

## Research Purpose

This framework is designed for scientific study of fundamental AI capabilities in social contexts:

- **Deception**: How effectively can AI agents mislead others while maintaining plausible cover?
- **Detection**: Can AI systems identify lies and inconsistencies in others' behavior? 
- **Disclosure**: How do agents decide what information to reveal or conceal strategically?
- **Theory of Mind**: Do LLMs demonstrate understanding of others' knowledge, beliefs, and intentions?
- **Social Intelligence**: Can AI navigate complex multi-party interactions with competing objectives?

## Game Mechanics

**Roles:**
- **Detective** (Town): Has investigative powers, wins by eliminating the Mafioso
- **Mafioso** (Mafia): Secretly eliminates opponents, wins by avoiding detection  
- **Villager** (Town): No special powers, must identify threats through reasoning

**Information Asymmetry:** Each role has different knowledge and abilities, creating rich strategic dynamics where agents must infer hidden information from limited observations.

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