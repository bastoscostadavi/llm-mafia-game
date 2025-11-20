#!/usr/bin/env python3
"""
Game Viewer Script

Shows a complete, human-readable view of a game stored in SQLite.
Usage: python view_game.py <game_id> [db_path]
Example: python view_game.py 20250826_142553_0096
"""

import sqlite3
import sys
from pathlib import Path

def view_game(game_id, db_path='mini_mafia.db'):
    """Display a complete view of a game from the database."""
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        # Get game info
        game_row = conn.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
        if not game_row:
            print(f"Game {game_id} not found in database")
            return
        
        print(f"GAME: {game_id}")
        print(f"{'='*80}")
        print(f"Timestamp: {game_row['timestamp']}")
        print(f"Winner: {game_row['winner'].upper()}")
        print()
        
        # Get players and their models
        players_query = """
            SELECT gp.character_name, gp.role, gp.final_status,
                   p.player_type, p.model_name, p.model_provider, p.temperature
            FROM game_players gp
            LEFT JOIN players p ON gp.player_id = p.player_id
            WHERE gp.game_id = ?
            ORDER BY 
                CASE gp.role 
                    WHEN 'detective' THEN 1
                    WHEN 'mafioso' THEN 2  
                    WHEN 'villager' THEN 3
                END,
                gp.character_name
        """
        
        players = conn.execute(players_query, (game_id,)).fetchall()
        
        print(f"PLAYERS ({len(players)}):")
        print(f"{'Name':<10} {'Role':<10} {'Status':<10} {'Type':<8} {'Model':<20} {'Provider':<10}")
        print(f"{'-'*78}")
        
        for player in players:
            player_type = player['player_type'] or 'N/A'
            model_name = player['model_name'] or 'N/A'
            model_provider = player['model_provider'] or 'N/A'
            print(f"{player['character_name']:<10} {player['role']:<10} {player['final_status']:<10} {player_type:<8} {model_name:<20} {model_provider:<10}")
        
        print()
        
        # Show voting information from votes table
        votes_query = """
            SELECT character_name, role, voted_for, parsed_successfully
            FROM votes 
            WHERE game_id = ?
            ORDER BY 
                CASE role 
                    WHEN 'detective' THEN 1
                    WHEN 'mafioso' THEN 2
                    WHEN 'villager' THEN 3
                END,
                character_name
        """
        
        votes = conn.execute(votes_query, (game_id,)).fetchall()
        
        print(f"VOTING RESULTS:")
        print(f"{'Character':<10} {'Role':<10} {'Voted For':<10} {'Parsed':<8}")
        print(f"{'-'*45}")
        
        for vote in votes:
            parsed_str = 'Yes' if vote['parsed_successfully'] else 'No'
            voted_for = vote['voted_for'] if vote['voted_for'] else '(none)'
            print(f"{vote['character_name']:<10} {vote['role']:<10} {voted_for:<10} {parsed_str:<8}")
        
        print()
        
        # Get game sequence
        sequence = conn.execute("""
            SELECT step, action, actor, raw_response, parsed_result
            FROM game_sequence 
            WHERE game_id = ?
            ORDER BY step
        """, (game_id,)).fetchall()
        
        print(f"GAME SEQUENCE ({len(sequence)} events):")
        print(f"{'-'*80}")
        
        current_phase = None
        for event in sequence:
            step = event['step']
            action = event['action']
            actor = event['actor']
            parsed_result = event['parsed_result']
            
            # Determine game phase based on sequence position, not action type
            # In Mini-Mafia: Night actions (investigate/kill) happen first, then Day actions (discuss/vote)
            if action in ['investigate', 'kill']:
                phase = "NIGHT PHASE"
            elif action == 'discuss':
                phase = "DAY PHASE - Discussion"  
            elif action == 'vote':
                phase = "DAY PHASE - Voting"
            else:
                phase = "UNKNOWN PHASE"
            
            if phase != current_phase:
                print(f"\n{phase}")
                current_phase = phase
            
            if action == 'discuss':
                if parsed_result == 'remained silent':
                    print(f"  {step:2}. {actor}: (remained silent)")
                else:
                    # Show complete message without truncation
                    print(f"  {step:2}. {actor}: \"{parsed_result}\"")
            else:
                print(f"  {step:2}. {actor} {action}s {parsed_result}")
        
        print(f"\n{'='*80}")
        
        # Quick stats
        action_counts = {}
        for event in sequence:
            action = event['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        print(f"ACTION SUMMARY:")
        for action, count in sorted(action_counts.items()):
            print(f"  {action}: {count}")
        
    finally:
        conn.close()

def list_all_games(db_path='mini_mafia.db'):
    """List all games in the database."""
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        games = conn.execute("""
            SELECT g.game_id, g.timestamp, g.winner, 
                   COUNT(gp.character_name) as player_count
            FROM games g
            LEFT JOIN game_players gp ON g.game_id = gp.game_id
            GROUP BY g.game_id, g.timestamp, g.winner
            ORDER BY g.timestamp DESC
        """).fetchall()
        
        if not games:
            print("No games found in database")
            return
        
        print(f"ALL GAMES ({len(games)} total):")
        print(f"{'Game ID':<25} {'Timestamp':<20} {'Winner':<6} {'Players':<7}")
        print(f"{'-'*65}")
        
        for game in games:
            print(f"{game['game_id']:<25} {game['timestamp'][:19]:<20} {game['winner']:<6} {game['player_count']:<7}")
            
    finally:
        conn.close()

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python view_game.py <game_id> [db_path]  - View specific game")
        print("  python view_game.py --list [db_path]     - List all games")
        print()
        print("Examples:")
        print("  python view_game.py 20250826_142553_0096")
        print("  python view_game.py --list")
        sys.exit(1)
    
    if sys.argv[1] == '--list':
        db_path = sys.argv[2] if len(sys.argv) > 2 else 'mini_mafia.db'
        list_all_games(db_path)
    else:
        game_id = sys.argv[1]
        db_path = sys.argv[2] if len(sys.argv) > 2 else 'mini_mafia.db'
        view_game(game_id, db_path)

if __name__ == "__main__":
    main()