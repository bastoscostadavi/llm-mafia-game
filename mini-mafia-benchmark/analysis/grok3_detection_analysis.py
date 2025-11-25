import sqlite3
import pandas as pd
from collections import Counter

# Connect to database
conn = sqlite3.connect('database/mini_mafia.db')

# Get Grok 3 Mini's villager performance
print("=== GROK 3 MINI DETECTION ANALYSIS ===\n")

# Get all games where Grok 3 Mini played as villager
query = """
SELECT
    g.game_id,
    g.winner,
    gp.character_name as grok_character
FROM games g
JOIN game_players gp ON g.game_id = gp.game_id
JOIN players p ON gp.player_id = p.player_id
WHERE p.model_name LIKE '%grok-3-mini%'
AND gp.role = 'villager'
"""

grok_games = pd.read_sql_query(query, conn)
print(f"Total Grok 3 Mini villager games: {len(grok_games)}")
print(f"Town wins (voted correctly): {(grok_games['winner'] == 'town').sum()}")
print(f"Mafia wins (voted incorrectly): {(grok_games['winner'] == 'mafia').sum()}")
print(f"Accuracy: {100 * (grok_games['winner'] == 'town').sum() / len(grok_games):.1f}%\n")

# Compare with Claude Sonnet 4
query_claude = """
SELECT
    g.game_id,
    g.winner,
    gp.character_name as claude_character
FROM games g
JOIN game_players gp ON g.game_id = gp.game_id
JOIN players p ON gp.player_id = p.player_id
WHERE p.model_name LIKE '%claude-sonnet-4%'
AND gp.role = 'villager'
"""

claude_games = pd.read_sql_query(query_claude, conn)
print(f"=== CLAUDE SONNET 4 DETECTION COMPARISON ===\n")
print(f"Total Claude Sonnet 4 villager games: {len(claude_games)}")
print(f"Town wins: {(claude_games['winner'] == 'town').sum()}")
print(f"Accuracy: {100 * (claude_games['winner'] == 'town').sum() / len(claude_games):.1f}%\n")

# Analyze voting behavior - who did they vote for?
def analyze_voting_behavior(games_df, model_name):
    print(f"\n=== {model_name} VOTING BEHAVIOR ===")

    voted_for_detective = 0
    voted_for_mafioso = 0
    total_analyzed = 0

    for _, row in games_df.iterrows():
        game_id = row['game_id']
        villager_char = row.get('grok_character') or row.get('claude_character')

        # Get the vote
        vote_query = f"""
        SELECT gs.parsed_result
        FROM game_sequence gs
        WHERE gs.game_id = '{game_id}'
        AND gs.action = 'vote'
        AND gs.actor = '{villager_char}'
        """
        vote_result = pd.read_sql_query(vote_query, conn)

        if len(vote_result) == 0:
            continue

        voted_for = vote_result.iloc[0]['parsed_result']

        # Get roles
        roles_query = f"""
        SELECT character_name, role
        FROM game_players
        WHERE game_id = '{game_id}'
        """
        roles = pd.read_sql_query(roles_query, conn)

        detective_char = roles[roles['role'] == 'detective']['character_name'].values[0]
        mafioso_char = roles[roles['role'] == 'mafioso']['character_name'].values[0]

        if voted_for == detective_char:
            voted_for_detective += 1
        elif voted_for == mafioso_char:
            voted_for_mafioso += 1

        total_analyzed += 1

    print(f"Games analyzed: {total_analyzed}")
    print(f"Voted for mafioso: {voted_for_mafioso} ({100*voted_for_mafioso/total_analyzed:.1f}%)")
    print(f"Voted for detective: {voted_for_detective} ({100*voted_for_detective/total_analyzed:.1f}%)")

analyze_voting_behavior(grok_games, "GROK 3 MINI")
analyze_voting_behavior(claude_games, "CLAUDE SONNET 4")

# Analyze discussion patterns - sample some winning and losing games
print("\n\n=== SAMPLING GAME EXAMPLES ===")

# Get 5 random winning games for Grok 3 Mini
winning_games = grok_games[grok_games['winner'] == 'town'].sample(min(5, len(grok_games[grok_games['winner'] == 'town'])))
losing_games = grok_games[grok_games['winner'] == 'mafia'].sample(min(5, len(grok_games[grok_games['winner'] == 'mafia'])))

print(f"\nGrok 3 Mini winning game IDs: {winning_games['game_id'].tolist()}")
print(f"Grok 3 Mini losing game IDs: {losing_games['game_id'].tolist()}")

# Get 5 random winning games for Claude Sonnet 4
claude_winning = claude_games[claude_games['winner'] == 'town'].sample(min(5, len(claude_games[claude_games['winner'] == 'town'])))
claude_losing = claude_games[claude_games['winner'] == 'mafia'].sample(min(5, len(claude_games[claude_games['winner'] == 'mafia'])))

print(f"\nClaude Sonnet 4 winning game IDs: {claude_winning['game_id'].tolist()}")
print(f"Claude Sonnet 4 losing game IDs: {claude_losing['game_id'].tolist()}")

conn.close()
