# LLM Mafia Game

A flexible implementation of the classic Mafia social deduction game where LLM agents (and humans) play as town members trying to identify hidden mafiosos through discussion, voting, and night actions. Based on the paper [Deceive, Detect, and Disclose: Large Language Models Play Mini-Mafia](https://arxiv.org/abs/2509.23023).

## Setup

```bash
git clone <repository-url>
cd llm-mafia-game
pip install -r requirements.txt
```

Create a `.env` file in the root with the API keys for the providers you want to use:

```bash
# Only the keys for the providers you intend to use are required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
XAI_API_KEY=...
DEEPSEEK_API_KEY=...
```

## Running a Game

### Option 1: Preset launcher (easiest)

```bash
python run_game.py
```

Presents a menu to choose between:
- **Classic** — 6 players: 2 mafiosos, 1 detective, 3 villagers
- **Mini-Mafia** — 4 players: 1 mafioso, 1 detective, 2 villagers

Edit `preset_games.py` to change which models are used in each preset.

### Option 2: Custom game (Python API)

```python
from src.main import create_game

players = [
    {'name': 'Alice', 'role': 'detective',  'llm': {'type': 'anthropic', 'model': 'claude-3-5-sonnet-20241022'}},
    {'name': 'Bob',   'role': 'mafioso',    'llm': {'type': 'openai',    'model': 'gpt-4o-mini'}},
    {'name': 'Carol', 'role': 'villager',   'llm': {'type': 'google',    'model': 'gemini-2.0-flash-exp'}},
    {'name': 'Dave',  'role': 'villager',   'llm': {'type': 'human'}},
]

game = create_game(players, discussion_rounds=2, debug_prompts=False)
game.play()
```

### Option 3: Web interface (human players)

```bash
cd mini-mafia-benchmark/web
python web_interface.py
# Open http://localhost:5000
```

## Supported Models

| Provider   | `type`       | Example model                        | Required key          |
|------------|--------------|--------------------------------------|-----------------------|
| OpenAI     | `openai`     | `gpt-4o-mini`, `gpt-4o`             | `OPENAI_API_KEY`      |
| Anthropic  | `anthropic`  | `claude-3-5-sonnet-20241022`         | `ANTHROPIC_API_KEY`   |
| Google     | `google`     | `gemini-2.0-flash-exp`               | `GOOGLE_API_KEY`      |
| xAI        | `xai`        | `grok-3`                             | `XAI_API_KEY`         |
| DeepSeek   | `deepseek`   | `deepseek-chat`                      | `DEEPSEEK_API_KEY`    |
| Local GGUF | `local`      | `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` | —                  |
| Human      | `human`      | —                                    | —                     |

Local models (GGUF format) go in the `models/` directory. Install llama-cpp-python with GPU support if needed.

### Model config options

```python
# Temperature and prompt caching
{'type': 'anthropic', 'model': 'claude-3-5-sonnet-20241022', 'use_cache': True}

# Reasoning models
{'type': 'openai', 'model': 'o1-mini', 'reasoning_effort': 'minimal'}

# Local model with context size
{'type': 'local', 'model': 'mistral.gguf', 'n_ctx': 4096}
```

## Game Rules

**Roles:**
- **Mafioso** — knows other mafiosos, kills one town member each night
- **Detective** — investigates one player each night to learn their role
- **Villager** — no special ability; wins through discussion and voting

**Each round:**
1. **Night** — Mafioso picks a target to kill; Detective picks a player to investigate
2. **Day** — Deaths announced; players discuss in random order; blind vote to arrest someone

**Win conditions:** Town wins when all mafiosos are arrested. Mafia wins when mafiosos equal or outnumber town.

## Options

```python
# See all prompts and raw LLM responses
game = create_game(players, debug_prompts=True)

# Change discussion rounds per day phase (default: 2)
game = create_game(players, discussion_rounds=3)
```

Prompts are in `src/prompt.txt` (standard) and `src/prompt_short.txt` (concise). Modify them to change how agents understand their roles and objectives.

## Research & Benchmarking

The `mini-mafia-benchmark/` directory contains the full research suite: automated batch experiments, a SQLite database (~15k games), analysis scripts, and the web interface for collecting human gameplay data.

See [`mini-mafia-benchmark/README.md`](mini-mafia-benchmark/README.md) for the benchmarking workflow.

## Citation

```bibtex
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
