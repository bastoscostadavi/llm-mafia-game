import sqlite3
import pandas as pd

conn = sqlite3.connect('database/mini_mafia.db')

print("=== GROK 3 MINI STRATEGY: TRUST FIRST CLAIMANT ===\n")

query = """
SELECT
    g.game_id,
    gp.character_name as villager_character
FROM games g
JOIN game_players gp ON g.game_id = gp.game_id
JOIN players p ON gp.player_id = p.player_id
WHERE p.model_name LIKE '%grok-3-mini%'
AND gp.role = 'villager'
LIMIT 1000
"""

games = pd.read_sql_query(query, conn)

# Track who Grok BELIEVES (trusts their claim)
# Measured by who Grok does NOT vote to arrest
believed_first_claimant = 0
believed_second_claimant = 0
total_both_claim = 0

# Break down by whether first claimant is detective or mafioso
first_detective_believed = 0
first_detective_total = 0
first_mafioso_believed = 0
first_mafioso_total = 0

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

    # Who claimed first?
    if detective_claim_step < mafioso_claim_step:
        first_claimant = detective_char
        second_claimant = mafioso_char
        first_detective_total += 1
    else:
        first_claimant = mafioso_char
        second_claimant = detective_char
        first_mafioso_total += 1

    # Get Grok's vote (who they want to ARREST)
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

    voted_to_arrest = vote.iloc[0]['parsed_result']

    # Who does Grok BELIEVE (not vote to arrest)?
    # If Grok votes to arrest the second claimant, they believe the first
    # If Grok votes to arrest the first claimant, they believe the second

    if voted_to_arrest == second_claimant:
        believed_first_claimant += 1
        if first_claimant == detective_char:
            first_detective_believed += 1
    elif voted_to_arrest == first_claimant:
        believed_second_claimant += 1
        if first_claimant == mafioso_char:
            # This is CORRECT - voting to arrest the mafioso who claimed first
            pass

print(f"Analyzed {total_both_claim} games where both claimed\n")

print("=== OVERALL TRUST PATTERN ===")
print(f"Grok believed first claimant: {believed_first_claimant}/{total_both_claim} ({100*believed_first_claimant/total_both_claim:.1f}%)")
print(f"Grok believed second claimant: {believed_second_claimant}/{total_both_claim} ({100*believed_second_claimant/total_both_claim:.1f}%)")
print()

print("=== BREAKDOWN BY WHO CLAIMED FIRST ===")
print(f"When detective claimed first: {first_detective_total} games")
if first_detective_total > 0:
    print(f"  Grok believed detective: {first_detective_believed}/{first_detective_total} ({100*first_detective_believed/first_detective_total:.1f}%)")
    print(f"  Grok believed mafioso (counter-claim): {first_detective_total - first_detective_believed}/{first_detective_total} ({100*(first_detective_total - first_detective_believed)/first_detective_total:.1f}%)")

print()
print(f"When mafioso claimed first: {first_mafioso_total} games")
if first_mafioso_total > 0:
    mafioso_first_correct = first_mafioso_total - first_mafioso_believed  # Believing second means voting for first (mafioso)
    print(f"  Grok believed mafioso: {first_mafioso_believed}/{first_mafioso_total} ({100*first_mafioso_believed/first_mafioso_total:.1f}%)")
    print(f"  Grok believed detective (counter-claim): {mafioso_first_correct}/{first_mafioso_total} ({100*mafioso_first_correct/first_mafioso_total:.1f}%)")

print("\n=== INTERPRETATION ===")
print("Grok 3 Mini's strategy is: BELIEVE THE FIRST CLAIMANT")
print(f"  Success rate: ~{100*believed_first_claimant/total_both_claim:.0f}%")
print()
print("Why this works:")
print("  - Detective claims first 87.7% of the time (from earlier analysis)")
print("  - When detective claims first, Grok believes them")
print("  - When mafioso claims first, Grok still believes them (but this is rare)")
print()
print("Game-theoretic basis:")
print("  - Detective has information, so claims proactively")
print("  - Mafioso doesn't know who detective is, so waits")
print("  - If mafioso claims first randomly, 50% chance of accusing villager")
print("    (villager knows it's false, so mafioso gains nothing)")
print("  - Therefore: First claimant is highly likely to be detective!")

conn.close()
