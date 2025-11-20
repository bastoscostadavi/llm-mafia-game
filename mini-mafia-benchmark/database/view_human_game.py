#!/usr/bin/env python3
"""
Human Game Viewer Script

Shows games played by humans via the web interface
Usage: python view_human_game.py [game_id]
       python view_human_game.py --list
"""

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent / 'mini_mafia_human.db'

def view_game(game_id):
    """Display a human game from the database."""

    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        # Get game info
        game = conn.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
        if not game:
            print(f"Game {game_id} not found")
            return

        print(f"\n{'='*80}")
        print(f"GAME #{game_id}")
        print(f"{'='*80}")
        print(f"Timestamp: {game['timestamp']}")
        print(f"Winner: {game['winner']}")
        print(f"Background: {game['background_name']}")
        print()

        print(f"PLAYERS:")
        print(f"  Mafioso (HUMAN): {game['mafioso_name']}")
        print(f"  Detective (AI):  {game['detective_name']}")
        print(f"  Villager (AI):   {game['villager_name']}")
        print()

        print(f"RESULT:")
        print(f"  Arrested: {game['arrested_name']}")
        if game['winner'] == 'EVIL':
            print(f"  → Human WON! (Mafioso survived)")
        else:
            print(f"  → Human LOST (Mafioso was arrested)")
        print()

        # Get game actions
        actions = conn.execute("""
            SELECT step, action_type, actor, raw_response, parsed_result
            FROM game_actions
            WHERE game_id = ?
            ORDER BY step
        """, (game_id,)).fetchall()

        print(f"GAME SEQUENCE ({len(actions)} events):")
        print(f"{'-'*80}")

        current_phase = None
        for action in actions:
            action_type = action['action_type']
            actor = action['actor']
            parsed = action['parsed_result']

            # Determine phase
            if action_type in ['investigate', 'kill']:
                phase = "NIGHT PHASE"
            elif action_type == 'discuss':
                phase = "DAY PHASE - Discussion"
            elif action_type == 'vote':
                phase = "DAY PHASE - Voting"
            else:
                phase = "UNKNOWN"

            if phase != current_phase:
                print(f"\n{phase}")
                current_phase = phase

            # Display action
            if action_type == 'discuss':
                if parsed == 'remained silent':
                    print(f"  {action['step']:2}. {actor}: (remained silent)")
                else:
                    human_marker = " (HUMAN)" if actor == game['mafioso_name'] else ""
                    print(f"  {action['step']:2}. {actor}{human_marker}: {parsed}")
            elif action_type == 'kill':
                print(f"  {action['step']:2}. {actor} killed {parsed}")
            elif action_type == 'investigate':
                print(f"  {action['step']:2}. {actor} investigated {parsed}")
            elif action_type == 'vote':
                print(f"  {action['step']:2}. {actor} voted for {parsed}")

        print(f"\n{'='*80}\n")

    finally:
        conn.close()

def list_games():
    """List all human games."""

    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        games = conn.execute("""
            SELECT game_id, timestamp, winner, mafioso_name, arrested_name, background_name
            FROM games
            ORDER BY timestamp DESC
        """).fetchall()

        if not games:
            print("No games found")
            return

        print(f"\nHUMAN GAMES ({len(games)} total):")
        print(f"{'ID':<5} {'Timestamp':<20} {'Winner':<6} {'Human':<10} {'Arrested':<10} {'Background':<30}")
        print(f"{'-'*90}")

        for game in games:
            timestamp = game['timestamp'][:19]  # Trim microseconds
            print(f"{game['game_id']:<5} {timestamp:<20} {game['winner']:<6} {game['mafioso_name']:<10} {game['arrested_name']:<10} {game['background_name']:<30}")

        # Show win rate
        wins = sum(1 for g in games if g['winner'] == 'EVIL')
        win_rate = (wins / len(games) * 100) if games else 0
        print(f"\nHuman win rate: {wins}/{len(games)} = {win_rate:.1f}%")
        print()

    finally:
        conn.close()

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python view_human_game.py <game_id>  - View specific game")
        print("  python view_human_game.py --list     - List all games")
        print()
        print("Examples:")
        print("  python view_human_game.py 1")
        print("  python view_human_game.py --list")
        sys.exit(1)

    if sys.argv[1] == '--list':
        list_games()
    else:
        try:
            game_id = int(sys.argv[1])
            view_game(game_id)
        except ValueError:
            print(f"Invalid game ID: {sys.argv[1]}")
            sys.exit(1)

if __name__ == "__main__":
    main()
