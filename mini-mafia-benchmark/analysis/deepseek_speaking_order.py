import sqlite3
import pandas as pd

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

# Analyze speaking order pattern
results = {
    'villager_first_deepseek_second': 0,
    'deepseek_claims_detective': 0,
    'deepseek_claims_detective_wins': 0,
    'deepseek_claims_detective_losses': 0,
    'deepseek_no_claim_wins': 0,
    'deepseek_no_claim_losses': 0,
}

detailed_examples = []

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
        results['villager_first_deepseek_second'] += 1

        # Check if DeepSeek claims to be detective in their first statement
        deepseek_first_statement = discuss_df.iloc[1]['parsed_result']

        claims_detective = False
        if deepseek_first_statement and isinstance(deepseek_first_statement, str):
            statement_lower = deepseek_first_statement.lower()
            # Look for explicit detective claims only
            if ('i am the detective' in statement_lower or
                'i\'m the detective' in statement_lower or
                'i am detective' in statement_lower or
                'i\'m detective' in statement_lower or
                ('i investigated' in statement_lower and 'mafioso' in statement_lower) or
                ('i investigated' in statement_lower and 'mafia' in statement_lower) or
                ('i investigated' in statement_lower and 'innocent' in statement_lower)):
                claims_detective = True

        if claims_detective:
            results['deepseek_claims_detective'] += 1
            if winner == 'mafia':
                results['deepseek_claims_detective_wins'] += 1
            else:
                results['deepseek_claims_detective_losses'] += 1

            # Save examples
            if len(detailed_examples) < 10:
                detailed_examples.append({
                    'game_id': game_id,
                    'winner': winner,
                    'villager_statement': discuss_df.iloc[0]['parsed_result'],
                    'deepseek_statement': deepseek_first_statement
                })
        else:
            if winner == 'mafia':
                results['deepseek_no_claim_wins'] += 1
            else:
                results['deepseek_no_claim_losses'] += 1

print("=== SPEAKING ORDER ANALYSIS ===")
print(f"Games where villager spoke first and DeepSeek spoke second: {results['villager_first_deepseek_second']}")
print(f"\nOf those games:")
print(f"  DeepSeek claimed detective: {results['deepseek_claims_detective']} ({100*results['deepseek_claims_detective']/results['villager_first_deepseek_second']:.1f}%)")
print(f"    - Wins when claiming: {results['deepseek_claims_detective_wins']}")
print(f"    - Losses when claiming: {results['deepseek_claims_detective_losses']}")
if results['deepseek_claims_detective'] > 0:
    print(f"    - Win rate when claiming: {100*results['deepseek_claims_detective_wins']/results['deepseek_claims_detective']:.1f}%")

print(f"\n  DeepSeek did NOT claim detective: {results['villager_first_deepseek_second'] - results['deepseek_claims_detective']}")
print(f"    - Wins without claiming: {results['deepseek_no_claim_wins']}")
print(f"    - Losses without claiming: {results['deepseek_no_claim_losses']}")
no_claim_total = results['deepseek_no_claim_wins'] + results['deepseek_no_claim_losses']
if no_claim_total > 0:
    print(f"    - Win rate without claiming: {100*results['deepseek_no_claim_wins']/no_claim_total:.1f}%")

print("\n=== EXAMPLE GAMES WHERE DEEPSEEK CLAIMED DETECTIVE AS SECOND SPEAKER ===")
for i, example in enumerate(detailed_examples, 1):
    print(f"\nExample {i} - Game {example['game_id']} (Winner: {example['winner']})")
    print(f"Villager (first): {example['villager_statement'][:150]}...")
    print(f"DeepSeek (second): {example['deepseek_statement'][:150]}...")

conn.close()
