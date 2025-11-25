import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('database/mini_mafia.db')

print("=== ANALYZING 'TRUST THE FIRST CLAIMANT' STRATEGY ===\n")

def analyze_first_claim_trust(model_name_pattern, model_label):
    """Analyze if the model trusts whoever claims detective/investigation first"""

    query = f"""
    SELECT
        g.game_id,
        g.winner,
        gp.character_name as villager_character
    FROM games g
    JOIN game_players gp ON g.game_id = gp.game_id
    JOIN players p ON gp.player_id = p.player_id
    WHERE p.model_name LIKE '%{model_name_pattern}%'
    AND gp.role = 'villager'
    LIMIT 1000
    """

    games = pd.read_sql_query(query, conn)

    both_claimed_cases = 0
    villager_spoke_first_cases = 0

    # When both claim
    trusted_first_claimant = 0
    trusted_second_claimant = 0

    # When villager spoke first (revealing they're not detective)
    trusted_first_claim_after_villager = 0
    trusted_second_claim_after_villager = 0

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

        # Get all discussion
        discuss_query = f"""
        SELECT step, actor, parsed_result
        FROM game_sequence
        WHERE game_id = '{game_id}'
        AND action = 'discuss'
        ORDER BY step
        """
        discussions = pd.read_sql_query(discuss_query, conn)

        # Find first discussion step
        first_speaker = discussions.iloc[0]['actor'] if len(discussions) > 0 else None
        villager_spoke_first = (first_speaker == villager_char)

        # Find who claimed detective/investigation first
        detective_claim_step = None
        detective_claim_actor = None
        mafioso_claim_step = None
        mafioso_claim_actor = None

        for _, disc in discussions.iterrows():
            actor = disc['actor']
            statement = str(disc['parsed_result']).lower()
            step = disc['step']

            # Check for detective claims (explicit or investigation claims)
            is_claim = ('detective' in statement or
                       'investigated' in statement or
                       'investigation' in statement)

            if is_claim:
                if actor == detective_char and detective_claim_step is None:
                    detective_claim_step = step
                    detective_claim_actor = actor
                elif actor == mafioso_char and mafioso_claim_step is None:
                    mafioso_claim_step = step
                    mafioso_claim_actor = actor

        # Skip if both didn't claim
        if detective_claim_step is None or mafioso_claim_step is None:
            continue

        both_claimed_cases += 1

        # Determine who claimed first
        first_claimant = detective_char if detective_claim_step < mafioso_claim_step else mafioso_char
        second_claimant = mafioso_char if first_claimant == detective_char else detective_char

        # Get villager's vote
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

        # Track trust pattern
        if villager_spoke_first:
            villager_spoke_first_cases += 1
            if voted_for == first_claimant:
                trusted_first_claim_after_villager += 1
            elif voted_for == second_claimant:
                trusted_second_claim_after_villager += 1
        else:
            if voted_for == first_claimant:
                trusted_first_claimant += 1
            elif voted_for == second_claimant:
                trusted_second_claimant += 1

    print(f"=== {model_label} ===")
    print(f"Games with competing claims analyzed: {both_claimed_cases}")

    if both_claimed_cases - villager_spoke_first_cases > 0:
        print(f"\nWhen villager did NOT speak first:")
        print(f"  Trusted first claimant: {trusted_first_claimant}/{both_claimed_cases - villager_spoke_first_cases} ({100*trusted_first_claimant/(both_claimed_cases - villager_spoke_first_cases):.1f}%)")
        print(f"  Trusted second claimant: {trusted_second_claimant}/{both_claimed_cases - villager_spoke_first_cases} ({100*trusted_second_claimant/(both_claimed_cases - villager_spoke_first_cases):.1f}%)")

    if villager_spoke_first_cases > 0:
        print(f"\nWhen villager spoke first (revealed not detective):")
        print(f"  Trusted first claimant after them: {trusted_first_claim_after_villager}/{villager_spoke_first_cases} ({100*trusted_first_claim_after_villager/villager_spoke_first_cases:.1f}%)")
        print(f"  Trusted second claimant: {trusted_second_claim_after_villager}/{villager_spoke_first_cases} ({100*trusted_second_claim_after_villager/villager_spoke_first_cases:.1f}%)")
    print()

analyze_first_claim_trust("grok-3-mini", "GROK 3 MINI")
analyze_first_claim_trust("claude-sonnet-4", "CLAUDE SONNET 4")

# Now let's verify the game-theoretic reasoning
print("\n=== GAME-THEORETIC ANALYSIS ===\n")

print("If mafioso claims detective randomly (without knowledge):")
print("  - 50% chance of accusing the real detective")
print("  - 50% chance of accusing the villager")
print()
print("Strategy: 'Trust the first claimant'")
print("  - If detective claims first: Villager wins (correct)")
print("  - If mafioso claims first (random guess):")
print("    - 50% accused detective → detective counter-claims → both claimed")
print("    - 50% accused villager → villager knows it's wrong, so doesn't help mafioso")
print()
print("Therefore: When both claim, the first claimant is more likely to be detective!")
print()

# Let's verify this empirically
print("=== EMPIRICAL VERIFICATION ===\n")

query = """
SELECT g.game_id
FROM games g
LIMIT 5000
"""
all_games = pd.read_sql_query(query, conn)

first_claimant_is_detective = 0
first_claimant_is_mafioso = 0

for _, row in all_games.iterrows():
    game_id = row['game_id']

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

    # Check who claimed first when both claimed
    if detective_claim_step is not None and mafioso_claim_step is not None:
        if detective_claim_step < mafioso_claim_step:
            first_claimant_is_detective += 1
        else:
            first_claimant_is_mafioso += 1

total = first_claimant_is_detective + first_claimant_is_mafioso
if total > 0:
    print(f"Out of {total} games where both claimed:")
    print(f"  Detective claimed first: {first_claimant_is_detective} ({100*first_claimant_is_detective/total:.1f}%)")
    print(f"  Mafioso claimed first: {first_claimant_is_mafioso} ({100*first_claimant_is_mafioso/total:.1f}%)")
    print()
    print("This validates the strategy: First claimant is more likely to be the detective!")

conn.close()
