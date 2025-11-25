import sqlite3
import pandas as pd
from collections import Counter

# Connect to database
conn = sqlite3.connect('database/mini_mafia.db')

# Get all DeepSeek mafioso games
query = """
SELECT
    g.game_id,
    g.winner,
    gp.character_name as deepseek_character
FROM games g
JOIN game_players gp ON g.game_id = gp.game_id
JOIN players p ON gp.player_id = p.player_id
WHERE p.model_name LIKE '%deepseek%'
AND gp.role = 'mafioso'
"""

games_df = pd.read_sql_query(query, conn)

print(f"Total DeepSeek mafioso games: {len(games_df)}\n")

# Analyze what DeepSeek says when speaking second after villager
second_speaker_statements = []
winning_statements = []
losing_statements = []

for idx, row in games_df.iterrows():
    game_id = row['game_id']
    deepseek_char = row['deepseek_character']
    winner = row['winner']

    # Get all player roles
    roles_query = f"""
    SELECT character_name, role
    FROM game_players
    WHERE game_id = '{game_id}'
    """
    roles_df = pd.read_sql_query(roles_query, conn)

    # Find villager character
    villager_char = roles_df[roles_df['role'] == 'villager']['character_name'].values
    if len(villager_char) == 0:
        continue
    villager_char = villager_char[0]

    # Get discussion sequence
    discuss_query = f"""
    SELECT step, actor, parsed_result
    FROM game_sequence
    WHERE game_id = '{game_id}'
    AND action = 'discuss'
    ORDER BY step
    """
    discuss_df = pd.read_sql_query(discuss_query, conn)

    if len(discuss_df) < 2:
        continue

    # Check if villager is first to speak and DeepSeek is second
    first_speaker = discuss_df.iloc[0]['actor']
    second_speaker = discuss_df.iloc[1]['actor'] if len(discuss_df) > 1 else None

    if first_speaker == villager_char and second_speaker == deepseek_char:
        statement = discuss_df.iloc[1]['parsed_result']
        if statement and isinstance(statement, str) and statement != 'remained silent':
            second_speaker_statements.append(statement)
            if winner == 'mafia':
                winning_statements.append(statement)
            else:
                losing_statements.append(statement)

print(f"=== WHEN VILLAGER SPEAKS FIRST, DEEPSEEK SPEAKS SECOND ===")
print(f"Total instances: {len(second_speaker_statements)}")
print(f"  Winning games: {len(winning_statements)}")
print(f"  Losing games: {len(losing_statements)}")

print("\n=== RANDOM SAMPLE OF WINNING STATEMENTS (first 15) ===")
for i, stmt in enumerate(winning_statements[:15], 1):
    print(f"{i}. {stmt[:200]}...")

print("\n=== RANDOM SAMPLE OF LOSING STATEMENTS (first 15) ===")
for i, stmt in enumerate(losing_statements[:15], 1):
    print(f"{i}. {stmt[:200]}...")

# Analyze content patterns
def analyze_patterns(statements, label):
    print(f"\n=== CONTENT ANALYSIS: {label} ===")
    patterns = {
        'agrees_with_villager': 0,
        'questions_or_neutral': 0,
        'suggests_caution': 0,
        'mentions_detective': 0,
        'asks_question': 0,
    }

    for stmt in statements:
        stmt_lower = stmt.lower()
        if 'agree' in stmt_lower:
            patterns['agrees_with_villager'] += 1
        if '?' in stmt:
            patterns['asks_question'] += 1
        if 'careful' in stmt_lower or 'caution' in stmt_lower or 'watch' in stmt_lower:
            patterns['suggests_caution'] += 1
        if 'detective' in stmt_lower:
            patterns['mentions_detective'] += 1

    total = len(statements)
    for pattern, count in patterns.items():
        print(f"  {pattern}: {count}/{total} ({100*count/total:.1f}%)")

analyze_patterns(winning_statements, "WINNING GAMES")
analyze_patterns(losing_statements, "LOSING GAMES")

conn.close()
