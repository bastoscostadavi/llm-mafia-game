# Experiments

This folder contains scripts for running Mini-Mafia benchmark experiments. These scripts orchestrate gameplay between LLM agents and store results in the database.

## Scripts

### `run_mini_mafia_batch.py`
Main experiment runner for the standard benchmark configuration.

**Purpose:** Run systematic tournaments to evaluate model capabilities

**Configuration:**
- 2 discussion rounds per game
- Standard detailed prompt
- All model combinations specified in config
- 100 games per configuration

**Usage:**
```bash
python3 run_mini_mafia_batch.py
```

**Output:** Game results stored in `../database/mini_mafia.db`

### `run_mini_mafia_batch_roundn.py`
Ablation experiment runner with configurable discussion rounds.

**Purpose:** Test robustness across different discussion lengths

**Configuration:**
- Configurable number of rounds (typically 8)
- Same models as standard benchmark
- Used for ablation studies

**Usage:**
```bash
python3 run_mini_mafia_batch_roundn.py
```

**Output:** Game results stored in ablation-specific database

## Running Experiments

### Prerequisites

1. **API Keys:** Set up API keys in `.env` file in project root:
   ```bash
   OPENAI_API_KEY=your_key_here
   ANTHROPIC_API_KEY=your_key_here
   DEEPSEEK_API_KEY=your_key_here
   XAI_API_KEY=your_key_here
   ```

2. **Database:** Ensure `../database/mini_mafia.db` exists (auto-created if missing)

3. **Game Logic:** Requires `../mini_mafia.py`

### Standard Benchmark

Run the complete benchmark:

```bash
python3 run_mini_mafia_batch.py
```

**Experiment Design:**
- **Models:** 10 LLMs (Claude, GPT, Grok, DeepSeek, Gemini, local models)
- **Configurations:** I³ possible combinations (where I = number of models)
- **Games per config:** 100
- **Total games:** ~15,000 (with capability-based filtering)

**Duration:** Depends on API rate limits and model speeds
- Fast models (GPT-4o-mini): ~10 hours
- Slower models (Claude Opus): ~20-30 hours
- Can run in parallel with multiple API keys

### Monitoring Progress

Check database for game counts:

```bash
sqlite3 ../database/mini_mafia.db "SELECT COUNT(*) FROM games;"
```

View benchmark progress:

```bash
sqlite3 ../database/mini_mafia.db "
  SELECT capability, COUNT(*) as games
  FROM benchmark
  GROUP BY capability;
"
```

## Experimental Design

The benchmark uses a **capability-based design** where each model is evaluated in three roles:

1. **Deceive (Mafioso):** Can the model deceive others to avoid detection?
2. **Detect (Villager):** Can the model detect deception from others?
3. **Disclose (Detective):** Can the model effectively disclose information?

For each capability, the target model is varied while opponent models are held fixed in **backgrounds**.

## Database Schema

Games are stored with:
- **games table:** Complete game state, outcomes, player info
- **game_actions table:** Detailed transcript of all actions
- **game_players table:** Player assignments and roles
- **benchmark table:** Links games to capability being tested

See `../database/README.md` for full schema details.

## Model Configuration

Models are configured with:
- **API endpoint:** OpenAI, Anthropic, DeepSeek, XAI, or local
- **Model name:** Specific model version
- **Temperature:** Sampling randomness (typically 1.0)
- **System prompts:** Role-specific instructions

## Cost Estimation

Approximate costs for full benchmark (per model):
- **GPT-4o-mini:** ~$5-10
- **GPT-4:** ~$50-100
- **Claude models:** ~$30-80
- **Local models:** Free (but slower)

**Total for 10 models:** ~$200-500 depending on model mix

## Error Handling

Scripts include:
- **Retry logic:** Automatic retries for API failures
- **Rate limiting:** Respects API rate limits
- **Validation:** Checks for valid game states
- **Logging:** Detailed logs of all games

## Reproducibility

For reproducible results:
1. **Fix random seeds:** Set in script configuration
2. **Use same model versions:** Note exact model IDs
3. **Consistent prompts:** Don't modify game instructions
4. **Same API parameters:** Keep temperature, max_tokens consistent

## Advanced: Custom Experiments

To run custom experiments:

1. **Modify model list:** Edit model configuration in script
2. **Change game count:** Adjust games per configuration
3. **Add new backgrounds:** Update background definitions
4. **Custom prompts:** Modify prompt templates in `../mini_mafia.py`

## Troubleshooting

**API rate limits:**
- Add delays between requests
- Use multiple API keys
- Switch to cheaper models for testing

**Database locked:**
- Close any open database connections
- Run experiments sequentially, not in parallel

**Invalid game states:**
- Check model outputs for format compliance
- Verify prompt templates
- Review game_actions table for debugging

## Citation

If you use these experiments in your research:

```bibtex
@article{costa2025minimafia,
  title={Deceive, Detect, and Disclose: Large Language Models Playing Mini-Mafia},
  author={Costa, Davi Bastos and Vicente, Renato},
  year={2025}
}
```
