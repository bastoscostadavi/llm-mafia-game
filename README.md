# LLM Mafia Game

A flexible implementation of the classic Mafia social deduction game where LLM agents (or humans) play as town members trying to identify hidden mafiosos through strategic discussion, voting, and night actions.

## Overview

This project enables:
- **Interactive gameplay** with AI agents from multiple providers (OpenAI, Anthropic, Google, xAI, DeepSeek, local models)
- **Human participation** through a web interface or command-line
- **Benchmarking and research** on LLM deception, detection, and strategic reasoning
- **Flexible configuration** for different game variants and model combinations

## Game Rules

### Roles

**Town Team (eliminate all mafia to win):**
- **Detective**: Investigates one player each night to learn their true role
- **Villager**: No special abilities, relies on discussion and voting

**Mafia Team (eliminate all town members to win):**
- **Mafioso**: Knows other mafiosos, kills one town member each night

### Game Flow

**Night Phase:**
1. Detectives investigate players to learn their roles (private information)
2. One randomly selected mafioso chooses a target to eliminate
3. Deaths are announced publicly at the start of the next day

**Day Phase:**
1. **Discussion rounds**: Players speak in random order, sharing suspicions and information
2. **Voting**: All active players vote to arrest someone
3. The player with the most votes is arrested and removed from play

**Victory Conditions:**
- Town wins when all mafiosos are arrested
- Mafia wins when all non-mafiosos are eliminated

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm-mafia-game

# Install dependencies
pip install -r requirements.txt
```

### Running a Game

**Option 1: Preset Games (Easiest)**
```bash
python run_game.py
```

This launches a menu where you can choose:
- **Classic** (6 players: 2 mafiosos, 1 detective, 3 villagers)
- **Mini-mafia** (4 players: 1 mafioso, 1 detective, 2 villagers)

**Option 2: Custom Game (Python API)**
```python
from src.main import create_game

players = [
    {'name': 'Alice', 'role': 'detective', 'llm': {'type': 'openai', 'model': 'gpt-4'}},
    {'name': 'Bob', 'role': 'mafioso', 'llm': {'type': 'local', 'model': 'mistral.gguf'}},
    {'name': 'Charlie', 'role': 'villager', 'llm': {'type': 'anthropic', 'model': 'claude-3-haiku'}},
    {'name': 'Diana', 'role': 'villager', 'llm': {'type': 'human'}}
]

game = create_game(players, discussion_rounds=2, debug_prompts=False)
game.play()
```

**Option 3: Web Interface (Human Players)**
```bash
cd mini-mafia-benchmark/web
python web_interface.py
# Open browser to http://localhost:5000
```

## Supported LLM Providers

### Local Models (llama-cpp)
```python
{'type': 'local', 'model': 'Mistral-7B-Instruct-v0.3-Q4_K_M.gguf', 'n_ctx': 2048}
```
- Download GGUF models to the `models/` directory
- Supports GPU acceleration via llama-cpp-python

### OpenAI
```python
{'type': 'openai', 'model': 'gpt-4o-mini', 'temperature': 0.7}
{'type': 'openai', 'model': 'gpt-5-mini', 'reasoning_effort': 'minimal'}
```
- Requires `OPENAI_API_KEY` environment variable

### Anthropic (Claude)
```python
{'type': 'anthropic', 'model': 'claude-3-5-sonnet-20241022', 'use_cache': True}
```
- Requires `ANTHROPIC_API_KEY` environment variable
- Supports prompt caching to reduce costs

### Google (Gemini)
```python
{'type': 'google', 'model': 'gemini-2.0-flash-exp', 'temperature': 0.7}
```
- Requires `GOOGLE_API_KEY` environment variable

### xAI (Grok)
```python
{'type': 'xai', 'model': 'grok-4', 'temperature': 0.7}
```
- Requires `XAI_API_KEY` environment variable

### DeepSeek
```python
{'type': 'deepseek', 'model': 'deepseek-v3.1', 'temperature': 0.7}
```
- Requires `DEEPSEEK_API_KEY` environment variable

### Human Players
```python
{'type': 'human', 'player_name': 'Alice'}
```
- Input responses via command line or web interface

## Configuration

### Environment Variables

Create a `.env` file in the root directory:
```bash
# API Keys (only needed for respective providers)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
XAI_API_KEY=...
DEEPSEEK_API_KEY=...
```

### Prompt Customization

Game prompts are defined in `src/prompt.txt` (standard version) and `src/prompt_short.txt` (concise version). You can modify these to change how agents perceive the game rules and their objectives.

## Project Structure

```
llm-mafia-game/
├── src/                        # Core game engine
│   ├── main.py                 # Game controller and state management
│   ├── agents.py               # Agent behavior (discuss, vote, kill, investigate)
│   ├── agent_interfaces.py     # LLM provider wrappers
│   ├── prompt_utils.py         # Prompt formatting and parsing
│   ├── config.py               # Token limits and model configurations
│   ├── prompt.txt              # Standard game prompt
│   └── prompt_short.txt        # Concise game prompt
│
├── mini-mafia-benchmark/       # Research and benchmarking suite
│   ├── mini_mafia.py           # 4-player variant implementation
│   ├── experiments/            # Experimental game configurations
│   ├── database/               # SQLite database for game records
│   ├── analysis/               # Analysis scripts and notebooks
│   ├── results/                # Plots and result tables
│   ├── article/                # Research paper (LaTeX)
│   └── web/                    # Web interface for human data collection
│       ├── web_interface.py    # Flask server
│       └── setup_web_game.py   # Web game configuration
│
├── models/                     # Local model files (.gguf)
├── run_game.py                 # Simple launcher for preset games
├── preset_games.py             # Predefined game configurations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Advanced Features

### Debug Mode

Enable debug mode to see all prompts and raw LLM responses:
```python
game = create_game(players, discussion_rounds=2, debug_prompts=True)
```

### Custom Discussion Rounds

Adjust the number of discussion rounds per day phase:
```python
game = create_game(players, discussion_rounds=3)  # 3 rounds of discussion
```

### Game State Logging

All game actions (discuss, vote, kill, investigate) are automatically logged to `game.state.game_sequence` with raw responses and parsed results for analysis.

### Prompt Caching

For repeated games with the same configuration:
- **Anthropic**: Set `'use_cache': True` in model config
- **Local models**: Automatic KV cache for static prompt sections

## Research & Benchmarking

The `mini-mafia-benchmark/` directory contains tools for systematic evaluation:

- **Automated experiments**: Run hundreds of games with different model configurations
- **Database storage**: SQLite database tracks all games, actions, and outcomes
- **Statistical analysis**: Scripts for analyzing win rates, deception patterns, detection accuracy
- **Web interface**: Collect human gameplay data for comparison

See `mini-mafia-benchmark/analysis/README.md` for details on analysis tools and methodologies.

## Tips for Effective Games

**For AI Agents:**
- Use temperature 0.3-0.7 for more strategic play
- Larger models (70B+, GPT-4, Claude-3.5-Sonnet) perform better at deception and strategic reasoning
- Smaller models may struggle with output format compliance

**For Human Players:**
- Web interface provides the best experience for humans
- Command-line interface works but requires precise format adherence
- Mix humans with AI for interesting emergent dynamics

**For Research:**
- Use `debug_prompts=False` and collect data via `game.state.game_sequence`
- Store results in the database for longitudinal analysis
- Control for confounds (player names, speaking order, role assignment)

## Troubleshooting

**Model loading issues:**
- Ensure `.gguf` model files are in the `models/` directory
- Check that `llama-cpp-python` is installed with GPU support if needed

**API errors:**
- Verify API keys are set in `.env` file
- Check rate limits for your provider
- Some providers may filter game content (deception themes)

**Format parsing failures:**
- Agents sometimes fail to follow output format requirements
- Failed actions fall back to random selection
- Enable `debug_prompts=True` to diagnose parsing issues

## Citation

If you use this codebase for research, please cite:
```
@misc{costa2025deceivedetectdiscloselarge,
      title={Deceive, Detect, and Disclose: Large Language Models Play Mini-Mafia}, 
      author={Davi Bastos Costa and Renato Vicente},
      year={2025},
      eprint={2509.23023},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.23023}, 
}
```

## License

[License information to be added]

## Contributing

Contributions welcome! Areas of interest:
- New game variants (e.g., multiple mafia teams, additional roles)
- Additional LLM provider integrations
- Improved prompt engineering for better strategic play
- Analysis tools and visualizations
