# LLM Mafia Game

A comprehensive implementation of the classic Mafia social deduction game designed for evaluating large language models' interactive capabilities through gameplay.

## Overview

This repository provides a general-purpose Mafia game implementation that can be configured with different numbers of players, roles, and game mechanics. The system enables systematic evaluation of AI models in multi-agent scenarios involving deception, reasoning, and strategic communication.

## Game Implementation

### Core Features

- **Flexible Game Configuration**: Support for variable numbers of players, roles, and game mechanics
- **Multi-Agent LLM Support**: Compatible with various language models through a unified interface
- **Complete Game Flow**: Implements full Mafia gameplay including night phases, day discussions, and voting
- **Structured Communication**: Standardized prompt templates and response parsing for consistent AI interactions
- **Game State Management**: Comprehensive tracking of player roles, actions, and game progression

### Supported Roles

- **Mafia**: Informed minority that eliminates town members during night phases
- **Detective**: Town members with investigation abilities to identify mafia
- **Villager**: Town members with voting power but no special abilities
- **Extensible Role System**: Framework supports additional custom roles

### Game Phases

1. **Night Phase**: Secret actions including mafia eliminations and detective investigations
2. **Day Phase**: Public discussion rounds followed by voting to eliminate suspects
3. **Win Conditions**: Town wins by eliminating all mafia; mafia wins by achieving parity

## Project Structure

```
├── src/                      # Core game implementation
├── models/                   # LLM integration and model configurations
├── mini-mafia-benchmark/     # Mini-Mafia benchmark implementation
├── run_game.py              # Basic game execution script
├── preset_games.py          # Predefined game configurations
└── requirements.txt         # Python dependencies
```

## Mini-Mafia Benchmark

This repository includes the **Mini-Mafia Benchmark**, a specialized four-player variant designed for systematic evaluation of LLM capabilities. Mini-Mafia isolates three key interactive dimensions:

- **Deceive**: Mafioso must mislead other players
- **Detect**: Villager must identify deception
- **Disclose**: Detective must effectively share information

For detailed information about the Mini-Mafia benchmark, methodology, and results, see [`mini-mafia-benchmark/`](mini-mafia-benchmark/).

## Quick Start

### Installation

```bash
git clone https://github.com/bastoscostadavi/llm-mafia-game
cd llm-mafia-game
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run a basic game
python run_game.py

# Run preset game configurations
python preset_games.py
```

### Configuration

Set up your model API keys in `.env`:
```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
# Add other model provider keys as needed
```

## Research Applications

### Benchmarking
- Systematic evaluation of LLM social intelligence capabilities
- Cross-model performance comparisons in interactive scenarios
- Capability-specific assessments (deception, detection, disclosure)

### Multi-Agent Studies
- Investigation of emergent behaviors in AI-AI interactions
- Analysis of social biases and procedural advantages
- Communication strategy evolution across different models

### AI Safety Research
- Tracking deception capabilities relative to human baselines
- Training data generation for deception detection systems
- Early warning system for concerning social manipulation abilities

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{costa2025minimafia,
  title={Deceive, Detect, and Disclose: Large Language Models Playing Mini-Mafia},
  author={Costa, Davi Bastos and Vicente, Renato},
  journal={arXiv preprint},
  year={2025}
}
```

## Contributing

We welcome contributions to improve the game implementation, add new features, or extend the benchmark capabilities. Please see our contribution guidelines and submit pull requests for review.

## License

This project is released under [appropriate license]. See LICENSE file for details.

## Support

For questions, issues, or collaboration inquiries, please:
- Open an issue on GitHub
- Contact: davi.costa@usp.br