# Mini-Mafia Web Game - Setup Instructions

## Overview
This web interface allows students to play Mini-Mafia as the **Mafioso** role against AI opponents. Perfect for lectures, workshops, and collecting human deception data.

## Quick Start (5 minutes)

### 1. Install Dependencies (if not already done)
```bash
cd /Users/davicosta/Desktop/projects/llm-mafia-game
pip install -r requirements.txt
```

### 2. Configure Background
Run the setup script to choose which AI models students will play against:

```bash
python mini-mafia-benchmark/setup_web_game.py
```

You'll be prompted to select:
- **Detective model** (will investigate the student and know they're the mafioso)
- **Villager model** (the swing vote that students must deceive)

**Popular choices:**
- GPT-4 / GPT-5 Mini for consistent, strong performance
- Claude Opus 4.1 for variety
- Mistral 7B (Local) if you want to avoid API costs

### 3. Check API Keys
Make sure your `.env` file in the project root contains the necessary API keys:

```bash
# For OpenAI-based models (GPT, DeepSeek, Grok)
OPENAI_API_KEY=your_key_here

# For Anthropic models (Claude)
ANTHROPIC_API_KEY=your_key_here

# For DeepSeek (if using)
DEEPSEEK_API_KEY=your_key_here

# For Grok (if using)
XAI_API_KEY=your_key_here
```

### 4. Start the Server
```bash
python mini-mafia-benchmark/web_game.py
```

You should see:
```
Initializing Mini-Mafia Web Interface...
Database: /Users/davicosta/Desktop/projects/llm-mafia-game/mini-mafia-benchmark/database/mini_mafia_human.db
Config: /Users/davicosta/Desktop/projects/llm-mafia-game/mini-mafia-benchmark/web_game_config.json

Current background: gpt4-detective_gpt4-villager
  Detective: {'type': 'openai', 'model': 'gpt-4'}
  Villager: {'type': 'openai', 'model': 'gpt-4'}

Starting server...
Students can access at: http://localhost:5000
```

### 5. Share Link with Students

**Option A: Same Network (Classroom)**
1. Find your computer's local IP address:
   ```bash
   # On Mac/Linux:
   ifconfig | grep "inet " | grep -v 127.0.0.1

   # On Windows:
   ipconfig
   ```
2. Share: `http://YOUR_IP:5000` (e.g., `http://192.168.1.100:5000`)

**Option B: External Access (Remote Students)**
1. Install ngrok: `brew install ngrok` (Mac) or download from https://ngrok.com
2. In a new terminal: `ngrok http 5000`
3. Share the HTTPS URL ngrok provides (e.g., `https://abc123.ngrok.io`)

### 6. Students Play!
Students will:
1. Click your link
2. Enter their name
3. Play as Mafioso
4. See results immediately
5. Done! (takes ~5-10 minutes per game)

## Game Data Collection

All gameplay data is automatically saved to:
```
mini-mafia-benchmark/database/mini_mafia_human.db
```

### Database Schema

**games table:**
- `game_id`: Unique game identifier
- `timestamp`: When game was played
- `winner`: 'GOOD' or 'EVIL'
- `mafioso_name`: Always 'Alice' (the student)
- `detective_name`, `villager_name`: AI opponents
- `arrested_name`: Who was voted out
- `background_name`: Which AI configuration was used
- `student_name`: Student's entered name

**game_actions table:**
- Complete transcript of all messages, votes, and actions
- Linked to `game_id` for detailed analysis

### Viewing Data

Quick check:
```bash
sqlite3 mini-mafia-benchmark/database/mini_mafia_human.db "SELECT student_name, winner, arrested_name, background_name FROM games ORDER BY timestamp DESC LIMIT 10;"
```

Full analysis with Python:
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('mini-mafia-benchmark/database/mini_mafia_human.db')

# Get all games
games = pd.read_sql_query("SELECT * FROM games", conn)

# Calculate win rate
human_win_rate = (games['winner'] == 'EVIL').mean()
print(f"Human win rate (as Mafioso): {human_win_rate:.2%}")

# Win rate by background
win_rate_by_background = games.groupby('background_name')['winner'].apply(
    lambda x: (x == 'EVIL').mean()
)
print(win_rate_by_background)
```

## Running Multiple Backgrounds

To test students against different backgrounds during the same session:

1. **Stop the server** (Ctrl+C)
2. **Run setup again**: `python mini-mafia-benchmark/setup_web_game.py`
3. **Select new background**
4. **Restart server**: `python mini-mafia-benchmark/web_game.py`
5. **Students use the same link** - they'll now play against the new background

All data goes into the same database with different `background_name` values.

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### Students can't connect
- Check firewall settings (allow port 5000)
- Verify IP address is correct
- Make sure server is running (should show "Running on http://0.0.0.0:5000")

### API rate limits
- Use local models (Mistral, Llama) to avoid API costs
- Or use cheaper models like GPT-4o-mini

### Games are slow
- Reduce model complexity (use GPT-4o-mini instead of GPT-4)
- Use local models
- Check internet connection

### Database errors
- Database is created automatically on first run
- Located at: `mini-mafia-benchmark/database/mini_mafia_human.db`
- Can delete and restart if corrupted

## For Your Article

After collecting data, you can compare human performance against AI models:

```python
import sqlite3
import pandas as pd
import numpy as np

# Load human data
conn = sqlite3.connect('mini-mafia-benchmark/database/mini_mafia_human.db')
human_games = pd.read_sql_query("SELECT * FROM games", conn)

# Calculate human "Deceive" score
# (same methodology as AI benchmark)
human_win_rate = (human_games['winner'] == 'EVIL').mean()

print(f"Human Deceive capability: {human_win_rate:.2%} win rate")
print(f"Total games: {len(human_games)}")
print(f"Games per background:")
print(human_games['background_name'].value_counts())
```

Compare with AI scores from your benchmark to see how humans stack up!

## Technical Details

- **Backend**: Flask (Python web framework)
- **Frontend**: Vanilla HTML/CSS/JavaScript (no build step needed)
- **Database**: SQLite (no setup required)
- **Session Management**: Flask-Session (filesystem-based)
- **Game Logic**: Your existing Mini-Mafia implementation

## Support

If you encounter issues:
1. Check that all API keys are correctly set in `.env`
2. Verify the correct Python environment is active
3. Look at terminal output for error messages
4. Check `flask_session/` folder is being created (for session storage)

## Example Lecture Flow

**Before class:**
- [ ] Configure background #1
- [ ] Start server
- [ ] Test with your own playthrough
- [ ] Note your computer's IP / ngrok URL

**During class:**
- [ ] Explain Mini-Mafia game mechanics (5 min)
- [ ] Demo the interface by playing yourself (5 min)
- [ ] Share link, students play (10-15 min)
- [ ] While they play, explain benchmark methodology (10 min)
- [ ] Show aggregate results (5 min)

**After class:**
- [ ] Stop server
- [ ] Export data from database
- [ ] Add to your article analysis

Good luck with your lecture! 🎭
