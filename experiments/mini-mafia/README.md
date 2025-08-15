# Mini-Mafia Experiment

A streamlined 4-player Mafia variant optimized for studying LLM deception and detection patterns in social deduction games. Features advanced prompt caching for performance optimization and supports human vs LLM gameplay data collection.

## Overview

Mini-Mafia simplifies the classic Mafia game to focus on the core interaction between a detective and mafioso, with one villager as a swing vote. This creates an ideal environment for studying:

- Deception detection capabilities
- Strategic reasoning under uncertainty  
- Voting behavior and persuasion tactics
- Model-specific behavioral patterns

## Game Setup

### Players (4 total)
- **1 Detective**: Investigates and knows the mafioso's identity
- **1 Mafioso**: Must avoid detection while influencing the vote
- **2 Villagers**: Initially unaware of roles, 1 killed at night

### Gameplay Flow
1. **Night Phase**: One villager is eliminated (predetermined)
2. **Day Phase**: 3 survivors discuss and vote to arrest someone
   - Detective knows mafioso identity but must convince the villager
   - Mafioso knows detective identity and tries to deflect suspicion
   - Surviving villager must choose who to trust

### Win Conditions
- **Good Wins**: Mafioso is arrested
- **Evil Wins**: Detective or villager is arrested

## Files and Tools

### Core Scripts

- **`run_mini_mafia_batch.py`**: Batch experiment runner
- **`game_viewer.py`**: Interactive game viewer and analyzer  
- **`analyze_results.py`**: Win/loss statistical analysis
- **`analyze_voting.py`**: Detailed voting pattern analysis

### Usage Examples

```bash
# Run 100 games with current configuration  
python run_mini_mafia_batch.py 100

# Run 50 games with specific prompt version
python run_mini_mafia_batch.py 50 --prompt-version v4.0

# Run with caching enabled (recommended for v4.0 prompts)
python run_mini_mafia_batch.py 100 --prompt-version v4.0 --enable-cache

# View batch results interactively
python game_viewer.py

# View specific batch
python game_viewer.py batch_20250814_183322_v4.0

# View specific game
python game_viewer.py batch_20250814_183322_v4.0 0

# Generate analysis reports
python analyze_results.py
python analyze_voting.py

# Human vs LLM data collection (from project root)
cd ../../
python web_app.py  # Creates data in web_games/ folder
```

## Data Format

### Batch Structure
```
data/
└── batch_YYYYMMDD_HHMMSS_vX.X/
    ├── batch_config.json       # Configuration and metadata
    ├── game_0000.json         # Individual game results  
    ├── game_0001.json
    └── ...

# Human vs LLM data (from web interface)
../../web_games/
└── web_game_YYYYMMDD_HHMMSS_PlayerName_Role.json
```

### Game Data (Streamlined Format)
```json
{
  "game_id": "batch_20250801_081244_game_0000",
  "game_number": 0,
  "timestamp": "2025-08-01T08:13:11.967813",
  "players": [
    {
      "name": "Alice",
      "role": "villager", 
      "alive": false,
      "imprisoned": false,
      "memory": [
        "You're Alice, the villager.",
        "Night 1: The villager Bob was found dead.",
        "Charlie: I'm concerned about the death...",
        // ... complete memory sequence
      ]
    }
    // ... other players
  ]
}
```

### Batch Configuration
```json
{
  "batch_id": "batch_20250801_081244",
  "timestamp": "2025-08-01T13:41:52.034454", 
  "game_type": "mini_mafia",
  "prompt_config": {
    "version": "v1.0",
    "message_limit": 200,
    "templates": { /* prompt templates */ }
  },
  "model_configs": {
    "detective": {"type": "local", "model_path": "models/mistral.gguf"},
    "mafioso": {"type": "local", "model_path": "models/mistral.gguf"}, 
    "villager": {"type": "local", "model_path": "models/mistral.gguf"}
  }
}
```

## Analysis Capabilities

### Win Rate Analysis (`analyze_results.py`)
- Overall good vs evil win percentages
- Batch-by-batch comparison
- Statistical uncertainty calculations
- Model configuration impact assessment

### Voting Pattern Analysis (`analyze_voting.py`)
- Detective voting accuracy (% voting for mafioso)
- Mafioso targeting patterns (% voting for detective)
- Villager decision-making (% voting for mafioso)
- Voting tie frequency and resolution

### Game Viewer (`game_viewer.py`)
- Interactive batch browser
- Complete game replay with discussion rounds
- Real-time decision analysis
- Memory-based reconstruction of game flow

## Research Applications

### Model Comparison Studies
```bash
# Different models for each role
python run_mini_mafia_batch.py 100 --detective-model gpt-3.5-turbo --mafioso-model claude-haiku

# Compare prompt versions
python run_mini_mafia_batch.py 100 --prompt-version v0.0
python run_mini_mafia_batch.py 100 --prompt-version v1.0
```

### Key Research Questions

1. **Deception Detection**: How accurately do detectives convince villagers?
2. **Deception Generation**: How effectively do mafiosos deflect suspicion?  
3. **Trust Dynamics**: What factors influence villager voting decisions?
4. **Model Differences**: Do different LLMs exhibit distinct strategies?
5. **Prompt Engineering**: How do prompt modifications affect behavior?

## Statistical Guidelines

### Sample Sizes
- **Preliminary analysis**: 100 games minimum
- **Model comparison**: 500+ games per configuration
- **Publication-quality**: 1000+ games per condition

### Batch Organization
- Keep batch sizes manageable (≤500 games) for analysis tools
- Use consistent naming conventions
- Document configuration changes between batches

### Analysis Best Practices
- Always report confidence intervals with win rates
- Control for prompt version when comparing models
- Consider multiple random seeds for robust conclusions

## Configuration Options

### Model Selection
Available model types and recommended usage:

```python
# Local models (fast, no API costs)
{'type': 'local', 'model_path': 'models/mistral.gguf'}
{'type': 'local', 'model_path': 'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf'}

# API models (higher capability, API costs apply)
{'type': 'openai', 'model': 'gpt-3.5-turbo'}
{'type': 'anthropic', 'model': 'claude-3-haiku-20240307'}
```

### Prompt Versions
- **v0.0**: Original research prompts, basic role descriptions
- **v1.0**: Enhanced with strategic guidance and clearer instructions
- **v2.0**: Improved structure and clarity
- **v3.0**: Forced response format for faster API model inference
- **v4.0**: Optimized for prompt caching with 50%+ performance improvements

### Performance Tuning
- **Context Window**: 2048 tokens (optimized for caching and v4.0 prompts)
- **Temperature**: 0.7 (balanced creativity and consistency)
- **Discussion Rounds**: 2 (adequate for 3-player dynamics)
- **KV Caching**: Enabled for local models with v4.0 prompts (50%+ speedup)
- **Prompt Caching**: Automatic cache boundary detection for repeated content

## Troubleshooting

### Common Issues

**Empty Responses**: Check model path and context window settings
**Memory Errors**: Reduce batch size or check available RAM
**API Rate Limits**: Add delays between games for API models
**Inconsistent Results**: Verify prompt version consistency across batches

### Performance Optimization
- Use local models for large-scale experiments  
- Enable GPU acceleration for llama.cpp models
- **Use v4.0 prompts with caching** for 50%+ performance improvement
- Enable KV caching: `--enable-cache` flag for batch experiments
- Monitor disk space (batches can be large despite optimization)
- Consider parallel execution for independent experiments

## Future Directions

Potential extensions to the mini-mafia framework:

- **Multi-round variants**: Extended gameplay with multiple elimination rounds
- **Role variants**: Additional roles like doctor or serial killer
- **Communication restrictions**: Limited or structured communication phases  
- **Mixed-model scenarios**: Human players vs AI agents (implemented via web interface)
- **Adversarial testing**: Robustness against prompt manipulation
- **Production deployment**: Scale web interface for internet-wide data collection