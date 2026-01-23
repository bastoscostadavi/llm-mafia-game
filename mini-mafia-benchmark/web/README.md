# Web Interface for Human Gameplay

This folder contains a Flask-based web interface that allows humans to play Mini-Mafia in **any role** (Mafioso, Detective, or Villager) against AI opponents. Perfect for teaching, demonstrations, and collecting human performance data across all three social capabilities.

## Files

- **`web_interface.py`** - Main Flask application
- **`setup_web_game.py`** - Configure human role and AI opponents
- **`web_game_config.json`** - Current configuration (auto-generated)
- **`DETECTIVE_SETUP.md`** - Guide for running detective (disclose) games
- **`START_HERE.sh`** - Quick start script
- **`SIMPLE_START.md`** - Quick start guide
- **`flask_session/`** - Session data (auto-generated at runtime)

## Quick Start

### 1. Configure Game Setup

```bash
python3 setup_web_game.py
```

**First, select the human player role:**
1. **Mafioso (Deceive)** - Human tries to deceive the villager
2. **Detective (Disclose)** - Human reveals who the mafioso is
3. **Villager (Detect)** - Human tries to identify the mafioso

**Then select AI models** for the other two roles

### 2. Start the Server

```bash
python3 web_interface.py
```

The server will start at `http://localhost:5000`

### 3. Share with Students/Players

**Local network (classroom):**
```bash
# Find your IP address
ifconfig | grep "inet " | grep -v 127.0.0.1

# Share: http://YOUR_IP:5000
```

**Remote access:**
```bash
# Install ngrok
brew install ngrok

# Create tunnel
ngrok http 5000

# Share the HTTPS URL provided
```

## Game Flow

1. Player enters their name
2. Player reads brief game instructions
3. Player plays as **Mafioso** (trying to deceive the villager)
4. Two rounds of discussion in random order
5. Blind voting phase
6. Results displayed immediately

Each game takes ~5-10 minutes.

## Data Collection

All games are automatically saved to:
```
../database/mini_mafia_human.db
```

**Database contains:**
- Player names
- Game outcomes (win/loss)
- Complete transcripts
- AI opponent configurations
- Timestamps

### View Collected Data

```bash
# Count total games
sqlite3 ../database/mini_mafia_human.db "SELECT COUNT(*) FROM games;"

# View recent games
sqlite3 ../database/mini_mafia_human.db "
  SELECT student_name, winner, arrested_name, background_name
  FROM games
  ORDER BY timestamp DESC
  LIMIT 10;
"

# Calculate human win rate
sqlite3 ../database/mini_mafia_human.db "
  SELECT
    COUNT(*) as total_games,
    SUM(CASE WHEN winner = 'mafia' THEN 1 ELSE 0 END) as human_wins,
    ROUND(100.0 * SUM(CASE WHEN winner = 'mafia' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate_pct
  FROM games;
"
```

## Use Cases

### 1. Teaching/Lectures
- Demonstrate AI social intelligence concepts
- Let students experience playing against AI
- Compare human vs. AI performance

### 2. Research
- Collect human baseline data for comparison
- Study human deception strategies
- Analyze human vs. AI differences

### 3. Demonstrations
- Conference demos
- Research presentations
- Public engagement

## Configuration

### Choosing AI Opponents

**Easy (for beginners):**
- GPT-4.1 Mini + GPT-4.1 Mini

**Medium:**
- Claude Opus 4.1 + GPT-5 Mini

**Hard:**
- DeepSeek V3.1 + Grok 3 Mini

### Multiple Backgrounds

To test different opponent combinations:
1. Stop the server (Ctrl+C)
2. Run `setup_web_game.py` again
3. Select new models
4. Restart server

All games go to the same database with different `background_name` labels.

## API Requirements

You'll need API keys for the models you select:

```bash
# In project root .env file
OPENAI_API_KEY=your_key_here           # For GPT models
ANTHROPIC_API_KEY=your_key_here        # For Claude models
DEEPSEEK_API_KEY=your_key_here         # For DeepSeek
XAI_API_KEY=your_key_here              # For Grok
```

**Cost per game:**
- GPT-4o-mini: ~$0.01-0.02
- Claude models: ~$0.05-0.10
- Local models: Free

## Technical Details

- **Backend:** Flask (Python web framework)
- **Frontend:** Vanilla HTML/CSS/JavaScript (embedded in Python)
- **Database:** SQLite (no setup required)
- **Session Management:** Flask-Session (filesystem)
- **Game Logic:** Uses `../mini_mafia.py`

## Troubleshooting

**Students can't connect:**
- Verify server is running
- Check firewall settings (allow port 5000)
- Confirm IP address is correct
- Ensure on same network (for local access)

**API rate limits:**
- Use cheaper models (GPT-4o-mini)
- Add delays between games
- Consider local models for high-volume use

**Games are slow:**
- Choose faster models
- Check internet connection
- Reduce model response length

**Session errors:**
- Clear `flask_session/` folder
- Restart server
- Have students clear browser cookies

## Comparing Human vs. AI Performance

After collecting data, compare with AI benchmark:

```python
import sqlite3
import pandas as pd

# Load human data
conn = sqlite3.connect('../database/mini_mafia_human.db')
human_games = pd.read_sql_query("SELECT * FROM games", conn)
human_win_rate = (human_games['winner'] == 'mafia').mean()

print(f"Human win rate as Mafioso: {human_win_rate:.2%}")
print(f"Total games collected: {len(human_games)}")

# Compare with AI benchmark results from paper
# DeepSeek V3.1: 3.09 (highest deceive score)
# Humans: ??? (to be determined!)
# Llama 3.1 8B: 0.29 (lowest deceive score)
```

## Dependencies

```bash
pip install flask pandas sqlite3
```

Plus API packages for chosen models (openai, anthropic, etc.)

## Safety & Ethics

When collecting human data:
- ✅ Obtain informed consent
- ✅ Anonymize if publishing
- ✅ Store data securely
- ✅ Follow institutional review board guidelines
- ✅ Allow participants to withdraw data

## Citation

If you use this web interface in research:

```bibtex
@article{costa2025minimafia,
  title={Deceive, Detect, and Disclose: Large Language Models Playing Mini-Mafia},
  author={Costa, Davi Bastos and Vicente, Renato},
  year={2025}
}
```
