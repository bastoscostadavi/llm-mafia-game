# Simple Start Guide

## What This Is
- Uses the **real** `mini_mafia.py` game logic
- Shows the **actual** game explanation from `prompt.txt`
- Student plays as Mafioso via web browser
- **No name input** - just start playing!

## To Run

### 1. Configure Background (if needed)
```bash
cd /Users/davicosta/Desktop/projects/llm-mafia-game
python3 mini-mafia-benchmark/setup_web_game.py
```

### 2. Start Server
```bash
python3 mini-mafia-benchmark/web_interface.py
```

### 3. Open Browser
```
http://localhost:5001
```

That's it!

## For Students (Tomorrow)

Share this URL with students:
- **Same network**: `http://YOUR_IP:5001`
- **External**: Use ngrok: `ngrok http 5001`

## How It Works

1. Student opens the link
2. Sees full game rules from `prompt.txt`
3. Game starts automatically (no name needed)
4. When it's their turn, they see the prompt and respond
5. AI opponents (Detective + Villager) play automatically
6. Results shown at the end
7. Data saved to database

## View Results

```bash
sqlite3 mini-mafia-benchmark/database/mini_mafia_human.db "SELECT * FROM games;"
```

## Key Features

✅ Uses existing `mini_mafia.py` structure
✅ Uses existing `agent_interfaces.py` (with WebHuman adapter)
✅ Shows real game explanation from `prompt.txt`
✅ No name collection - direct gameplay
✅ Clean, simple interface
✅ Full game data collection
