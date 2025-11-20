# Tomorrow's Lecture - Quick Checklist ✅

## 30 Minutes Before Class

### 1. Choose Your Background (2 minutes)
```bash
cd /Users/davicosta/Desktop/projects/llm-mafia-game
python3 mini-mafia-benchmark/setup_web_game.py
```

**Recommended combinations:**
- **Easy to beat**: GPT-4.1 Mini + GPT-4.1 Mini
- **Challenging**: Claude Opus 4.1 + GPT-5 Mini
- **Very Hard**: DeepSeek V3.1 + Grok 3 Mini

### 2. Test It Yourself (5 minutes)
```bash
python3 mini-mafia-benchmark/web_game.py
```

Then open: http://localhost:5000

Play through one game to:
- Make sure everything works
- Get familiar with the interface
- Understand the student experience

Press Ctrl+C to stop when done.

### 3. Get Your Connection URL

**Option A: Same Network (Recommended for classroom)**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```
Use the IP address shown (e.g., `192.168.1.100`)
Students will access: `http://YOUR_IP:5001`

**Option B: External Access (if students are remote)**
1. Install ngrok: `brew install ngrok`
2. In a NEW terminal: `ngrok http 5001`
3. Copy the HTTPS URL (e.g., `https://abc123.ngrok.io`)

### 4. Start the Server
```bash
python3 mini-mafia-benchmark/web_game.py
```

Leave this running! Don't close this terminal.

---

## During Class

### Intro (5 minutes)
- Explain Mini-Mafia: 4 players, 1 Mafioso (them), 1 Detective, 2 Villagers
- One villager dies at night, detective knows who the mafioso is
- Goal: Convince the villager you're innocent

### Share Link (Write on board)
```
http://YOUR_IP:5000
or
https://YOUR_NGROK_URL
```

### While Students Play (10-15 minutes)
They will:
1. Enter their name
2. Read brief instructions
3. Play one game (~5-10 minutes)
4. See if they won or lost

You can:
- Explain benchmark methodology
- Show your results from the article
- Discuss emergent behaviors (name bias, last-speaker advantage)
- Answer questions

### After Games Complete
Stop the server (Ctrl+C) and show aggregate results:

```bash
cd mini-mafia-benchmark
sqlite3 database/mini_mafia_human.db "
SELECT
    COUNT(*) as total_games,
    SUM(CASE WHEN winner = 'EVIL' THEN 1 ELSE 0 END) as human_wins,
    ROUND(100.0 * SUM(CASE WHEN winner = 'EVIL' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate_pct
FROM games;
"
```

Compare with AI results!

---

## Testing Different Backgrounds (Optional)

If you want to test multiple backgrounds during class:

1. Stop server (Ctrl+C)
2. Run: `python3 mini-mafia-benchmark/setup_web_game.py`
3. Select new models
4. Restart: `python3 mini-mafia-benchmark/web_game.py`
5. Students use same URL

All data goes to the same database with different background labels.

---

## Troubleshooting

### Students can't connect
- [ ] Server is running (check terminal)
- [ ] Using correct IP/URL
- [ ] Firewall allows port 5000
- [ ] On same network (if using local IP)

### Games are slow
- Use faster/cheaper models (GPT-4o-mini instead of GPT-4)
- Check internet connection
- Consider local models (no API costs)

### API errors
- [ ] Check .env file has correct API keys
- [ ] Check API key has credits/quota
- [ ] Try simpler model

### "No game in session" error
- Student needs to start from home page (not /game directly)
- Clear cookies and try again

---

## After Class

### Export Data for Article
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('mini-mafia-benchmark/database/mini_mafia_human.db')

# Get all games
games = pd.read_sql_query("SELECT * FROM games", conn)

# Human win rate
human_win_rate = (games['winner'] == 'EVIL').mean()

# Export to CSV
games.to_csv('human_deception_data.csv', index=False)

print(f"Human deceive capability: {human_win_rate:.2%}")
print(f"Total games collected: {len(games)}")
```

### Add to Your Article
Compare human win rates with AI scores from your benchmark:
- DeepSeek V3.1: 3.09 (very high deception)
- Humans: ??? (to be determined!)
- Claude Sonnet 4: 1.83
- GPT-5 Mini: 0.86
- Llama 3.1 8B: 0.29 (very low deception)

---

## Quick Command Reference

```bash
# Setup background
python3 mini-mafia-benchmark/setup_web_game.py

# Start server
python3 mini-mafia-benchmark/web_game.py

# Or use the all-in-one script
./mini-mafia-benchmark/START_HERE.sh

# View data
sqlite3 mini-mafia-benchmark/database/mini_mafia_human.db "SELECT * FROM games;"

# Count games
sqlite3 mini-mafia-benchmark/database/mini_mafia_human.db "SELECT COUNT(*) FROM games;"
```

---

## File Locations

- **Server code**: `mini-mafia-benchmark/web_game.py`
- **Config**: `mini-mafia-benchmark/web_game_config.json`
- **Database**: `mini-mafia-benchmark/database/mini_mafia_human.db`
- **Full instructions**: `mini-mafia-benchmark/WEB_GAME_README.md`

---

Good luck with your lecture! 🎭🎓

The system is ready to go. Just test it once before class and you're all set!
