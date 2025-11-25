import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('database/mini_mafia.db')

print("=== DETECTION CAPABILITY COMPARISON ===\n")

# Analyze trust in detective claims
def analyze_detective_trust(model_name_pattern, model_label):
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
    """

    games = pd.read_sql_query(query, conn)

    # For each game, check if detective made an explicit accusation
    detective_accused_cases = 0
    villager_trusted_detective = 0
    villager_distrusted_detective = 0

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
        SELECT actor, parsed_result
        FROM game_sequence
        WHERE game_id = '{game_id}'
        AND action = 'discuss'
        ORDER BY step
        """
        discussions = pd.read_sql_query(discuss_query, conn)

        # Check if detective made explicit accusation
        detective_accused = False
        for _, disc in discussions[discussions['actor'] == detective_char].iterrows():
            statement = str(disc['parsed_result']).lower()
            if (mafioso_char.lower() in statement and
                ('mafioso' in statement or 'mafia' in statement or 'investigated' in statement)):
                detective_accused = True
                break

        if not detective_accused:
            continue

        detective_accused_cases += 1

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

        if voted_for == mafioso_char:
            villager_trusted_detective += 1
        elif voted_for == detective_char:
            villager_distrusted_detective += 1

    print(f"=== {model_label} ===")
    print(f"Games where detective made explicit accusation: {detective_accused_cases}")
    print(f"  Villager voted for mafioso (trusted detective): {villager_trusted_detective} ({100*villager_trusted_detective/detective_accused_cases:.1f}%)")
    print(f"  Villager voted for detective (distrusted detective): {villager_distrusted_detective} ({100*villager_distrusted_detective/detective_accused_cases:.1f}%)")
    print()

analyze_detective_trust("grok-3-mini", "GROK 3 MINI")
analyze_detective_trust("claude-sonnet-4", "CLAUDE SONNET 4")
analyze_detective_trust("deepseek", "DEEPSEEK V3.1")
analyze_detective_trust("gpt-5-mini", "GPT-5 MINI")

# Additional analysis: sensitivity to mafioso counter-arguments
print("\n=== RESPONSE TO MAFIOSO COUNTER-CLAIMS ===\n")

def analyze_counter_claim_response(model_name_pattern, model_label):
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
    LIMIT 500
    """

    games = pd.read_sql_query(query, conn)

    both_claimed_detective = 0
    voted_for_real_detective = 0
    voted_for_mafioso = 0

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
        SELECT actor, parsed_result
        FROM game_sequence
        WHERE game_id = '{game_id}'
        AND action = 'discuss'
        ORDER BY step
        """
        discussions = pd.read_sql_query(discuss_query, conn)

        # Check if both claimed to be detective or made investigation claims
        detective_claimed = False
        mafioso_claimed = False

        for _, disc in discussions[discussions['actor'] == detective_char].iterrows():
            statement = str(disc['parsed_result']).lower()
            if 'detective' in statement or 'investigated' in statement:
                detective_claimed = True
                break

        for _, disc in discussions[discussions['actor'] == mafioso_char].iterrows():
            statement = str(disc['parsed_result']).lower()
            if 'detective' in statement or 'investigated' in statement:
                mafioso_claimed = True
                break

        if not (detective_claimed and mafioso_claimed):
            continue

        both_claimed_detective += 1

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

        if voted_for == detective_char:
            voted_for_real_detective += 1
        elif voted_for == mafioso_char:
            voted_for_mafioso += 1

    if both_claimed_detective > 0:
        print(f"{model_label}:")
        print(f"  Games with competing detective claims: {both_claimed_detective}")
        print(f"  Voted for real detective: {voted_for_real_detective} ({100*voted_for_real_detective/both_claimed_detective:.1f}%)")
        print(f"  Voted for mafioso: {voted_for_mafioso} ({100*voted_for_mafioso/both_claimed_detective:.1f}%)")
        print()

analyze_counter_claim_response("grok-3-mini", "GROK 3 MINI")
analyze_counter_claim_response("claude-sonnet-4", "CLAUDE SONNET 4")

conn.close()
