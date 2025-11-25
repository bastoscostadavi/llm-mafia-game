import sqlite3
import pandas as pd
from collections import defaultdict

# Connect to database
conn = sqlite3.connect('database/mini_mafia.db')

# Get all DeepSeek mafioso games
query = """
SELECT
    g.game_id,
    g.winner,
    gp.character_name as deepseek_character,
    gp.final_status
FROM games g
JOIN game_players gp ON g.game_id = gp.game_id
JOIN players p ON gp.player_id = p.player_id
WHERE p.model_name LIKE '%deepseek%'
AND gp.role = 'mafioso'
"""

games_df = pd.read_sql_query(query, conn)
print(f"Total DeepSeek mafioso games: {len(games_df)}")
print(f"Wins: {(games_df['winner'] == 'mafia').sum()}")
print(f"Losses: {(games_df['winner'] == 'town').sum()}")
print(f"Win rate: {100 * (games_df['winner'] == 'mafia').sum() / len(games_df):.2f}%\n")

# Analyze game sequences
def analyze_game_patterns(game_ids, label):
    print(f"\n=== Analyzing {label} ===")

    patterns = {
        'fake_detective_claim': 0,
        'counter_accuse_detective': 0,
        'deflect_with_questioning': 0,
        'claim_investigated_innocent': 0,
        'total_games_analyzed': 0
    }

    for game_id in game_ids[:100]:  # Analyze first 100 games
        query = f"""
        SELECT step, action, actor, parsed_result
        FROM game_sequence
        WHERE game_id = '{game_id}'
        ORDER BY step
        """

        sequence_df = pd.read_sql_query(query, conn)

        # Get DeepSeek's character name for this game
        char_query = f"""
        SELECT gp.character_name
        FROM game_players gp
        JOIN players p ON gp.player_id = p.player_id
        WHERE gp.game_id = '{game_id}'
        AND p.model_name LIKE '%deepseek%'
        AND gp.role = 'mafioso'
        """
        deepseek_char = pd.read_sql_query(char_query, conn).iloc[0]['character_name']

        # Get detective's character name
        det_query = f"""
        SELECT character_name
        FROM game_players
        WHERE game_id = '{game_id}' AND role = 'detective'
        """
        detective_char = pd.read_sql_query(det_query, conn).iloc[0]['character_name']

        # Analyze DeepSeek's discussion moves
        deepseek_discusses = sequence_df[
            (sequence_df['action'] == 'discuss') &
            (sequence_df['actor'] == deepseek_char)
        ]['parsed_result'].tolist()

        # Analyze detective's discussion moves
        detective_discusses = sequence_df[
            (sequence_df['action'] == 'discuss') &
            (sequence_df['actor'] == detective_char)
        ]['parsed_result'].tolist()

        patterns['total_games_analyzed'] += 1

        # Check for patterns
        for discuss in deepseek_discusses:
            if discuss and discuss.lower() != 'remained silent':
                # Fake detective claim
                if 'detective' in discuss.lower() and 'i am' in discuss.lower():
                    patterns['fake_detective_claim'] += 1
                    break

                # Counter-accuse the detective
                if detective_char.lower() in discuss.lower():
                    for det_discuss in detective_discusses:
                        if det_discuss and deepseek_char.lower() in det_discuss.lower():
                            patterns['counter_accuse_detective'] += 1
                            break

                # Deflect with questioning
                if '?' in discuss and ('why' in discuss.lower() or 'how' in discuss.lower()):
                    patterns['deflect_with_questioning'] += 1
                    break

                # Claim to have investigated someone innocent
                if 'investigated' in discuss.lower() and 'innocent' in discuss.lower():
                    patterns['claim_investigated_innocent'] += 1
                    break

    print(f"Games analyzed: {patterns['total_games_analyzed']}")
    print(f"Fake detective claims: {patterns['fake_detective_claim']} ({100*patterns['fake_detective_claim']/patterns['total_games_analyzed']:.1f}%)")
    print(f"Counter-accused detective: {patterns['counter_accuse_detective']} ({100*patterns['counter_accuse_detective']/patterns['total_games_analyzed']:.1f}%)")
    print(f"Deflected with questions: {patterns['deflect_with_questioning']} ({100*patterns['deflect_with_questioning']/patterns['total_games_analyzed']:.1f}%)")
    print(f"Claimed investigated innocent: {patterns['claim_investigated_innocent']} ({100*patterns['claim_investigated_innocent']/patterns['total_games_analyzed']:.1f}%)")

    return patterns

# Analyze winning vs losing games
winning_games = games_df[games_df['winner'] == 'mafia']['game_id'].tolist()
losing_games = games_df[games_df['winner'] == 'town']['game_id'].tolist()

winning_patterns = analyze_game_patterns(winning_games, "WINNING GAMES")
losing_patterns = analyze_game_patterns(losing_games, "LOSING GAMES")

conn.close()
