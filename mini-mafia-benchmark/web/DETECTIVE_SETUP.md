# Playing as Detective (Disclose Games)

The web interface now supports all three roles: **Mafioso (Deceive)**, **Detective (Disclose)**, and **Villager (Detect)**.

## Quick Setup for Detective Games

### 1. Configure the Game

```bash
cd web
python3 setup_web_game.py
```

When prompted:
- **Select human player role:** Choose `2` for Detective (Disclose)
- **Select AI model for Mafioso:** Choose which AI will be the mafioso
- **Select AI model for Villager:** Choose which AI will be the villager

The human will play as the detective who knows who the mafioso is and must convince the villager.

### 2. Start the Server

```bash
python3 web_interface.py
```

The server starts at `http://localhost:5000`

### 3. Play the Game

Open your browser to `http://localhost:5000` and play!

**As Detective:**
- You know who the mafioso is (you saw them kill someone at night)
- During discussion, you reveal this information
- Your goal: convince the villager to vote correctly
- The villager makes the final voting decision

**Win condition:** Villager votes for the mafioso (GOOD wins)

## Role Comparison

| Role | Capability | Task | Win Condition |
|------|-----------|------|---------------|
| **Mafioso** | Deceive | Convince villager you're innocent | EVIL wins (mafioso not arrested) |
| **Detective** | Disclose | Reveal who the mafioso is convincingly | GOOD wins (mafioso arrested) |
| **Villager** | Detect | Identify who the mafioso is | GOOD wins (mafioso arrested) |

## Data Collection

All games are saved to `../database/mini_mafia_human.db` with:
- `human_role` field indicating which role the human played
- `background_name` showing which AI opponents were used
- Complete game transcripts

### View Detective Game Data

```bash
sqlite3 ../database/mini_mafia_human.db "
  SELECT
    COUNT(*) as total_games,
    SUM(CASE WHEN winner = 'GOOD' THEN 1 ELSE 0 END) as human_wins,
    ROUND(100.0 * SUM(CASE WHEN winner = 'GOOD' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate_pct
  FROM games
  WHERE human_role = 'detective';
"
```

## Running the Theoretical Model with All Human Data

After collecting detective games, update the theoretical model:

```bash
cd ../analysis/top_down_theoretical
python3 scores_theoretical_with_humans.py
```

This will include:
- Deceive scores (from mafioso games)
- Detect scores (from villager games)
- Disclose scores (from detective games) ← **NEW!**

All three human capabilities will be estimated and plotted with uncertainty bars.

## Tips for Detective Games

**Good detective disclosure:**
- State clearly who you saw kill someone
- Explain why you're certain (you're the detective)
- Address potential mafioso counter-arguments
- Build trust with the villager

**Poor detective disclosure:**
- Vague accusations without reasoning
- Sounding uncertain or hedging
- Making it sound like a guess rather than knowledge
- Not differentiating yourself from the mafioso's claims

The AI villager must decide who to believe based on the discussion!
