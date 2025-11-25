import sqlite3
import pandas as pd

conn = sqlite3.connect('database/mini_mafia.db')

print("=== CLARIFYING: WHO IS THE SECOND CLAIMANT? ===\n")

query = """
SELECT
    g.game_id,
    gp.character_name as villager_character
FROM games g
JOIN game_players gp ON g.game_id = gp.game_id
JOIN players p ON gp.player_id = p.player_id
WHERE p.model_name LIKE '%grok-3-mini%'
AND gp.role = 'villager'
LIMIT 500
"""

games = pd.read_sql_query(query, conn)

# When both claim and Grok votes for second claimant
second_is_detective = 0
second_is_mafioso = 0
total_both_claim = 0

for _, row in games.iterrows():
    game_id = row['game_id']
    villager_char = row['villager_character']

    # Get roles
    roles_query = f"""
    SELECT character_name, role
    FROM game_players
    WHERE game_id = '{game_id}'
    """
    roles = pd.read_sql_query(roles_query, conn)

    detective_char = roles[roles['role'] == 'detective']['character_name'].values[0]
    mafioso_char = roles[roles['role'] == 'mafioso']['character_name'].values[0]

    # Get discussion
    discuss_query = f"""
    SELECT step, actor, parsed_result
    FROM game_sequence
    WHERE game_id = '{game_id}'
    AND action = 'discuss'
    ORDER BY step
    """
    discussions = pd.read_sql_query(discuss_query, conn)

    # Find who claimed first
    detective_claim_step = None
    mafioso_claim_step = None

    for _, disc in discussions.iterrows():
        actor = disc['actor']
        statement = str(disc['parsed_result']).lower()
        step = disc['step']

        is_claim = ('detective' in statement or 'investigated' in statement)

        if is_claim:
            if actor == detective_char and detective_claim_step is None:
                detective_claim_step = step
            elif actor == mafioso_char and mafioso_claim_step is None:
                mafioso_claim_step = step

    # Skip if both didn't claim
    if detective_claim_step is None or mafioso_claim_step is None:
        continue

    total_both_claim += 1

    # Who claimed second?
    if detective_claim_step < mafioso_claim_step:
        # Detective first, mafioso second
        second_claimant = mafioso_char
    else:
        # Mafioso first, detective second
        second_claimant = detective_char

    # Get Grok's vote
    vote_query = f"""
    SELECT parsed_result
    FROM game_sequence
    WHERE game_id = '{game_id}'
    AND action = 'vote'
    AND actor = '{villager_char}'
    """
    vote = pd.read_sql_query(vote_query, conn)

    if len(vote) == 0:
        continue

    voted_for = vote.iloc[0]['parsed_result']

    # Check if Grok voted for second claimant
    if voted_for == second_claimant:
        if second_claimant == detective_char:
            second_is_detective += 1
        else:
            second_is_mafioso += 1

print(f"Out of {total_both_claim} games where both claimed and Grok voted for second claimant:")
print(f"  Second claimant was detective: {second_is_detective}")
print(f"  Second claimant was mafioso: {second_is_mafioso}")
print()

if second_is_detective + second_is_mafioso > 0:
    print("This means when Grok 'trusts the second claimant':")
    total_second = second_is_detective + second_is_mafioso
    print(f"  It votes for detective {second_is_detective}/{total_second} times ({100*second_is_detective/total_second:.1f}%)")
    print(f"  It votes for mafioso {second_is_mafioso}/{total_second} times ({100*second_is_mafioso/total_second:.1f}%)")

print("\n=== INTERPRETATION ===")
print("If detective usually claims FIRST (87.7% of the time),")
print("and Grok trusts the SECOND claimant,")
print("then usually the second claimant is the MAFIOSO (counter-claiming).")
print()
print("BUT if Grok is voting for mafioso correctly 85.7% of the time,")
print("this suggests Grok trusts the COUNTER-CLAIM, which is usually the MAFIOSO!")
print()
print("Wait... let me recalculate what 'second claimant' actually means in terms of detective vs mafioso...")

conn.close()
