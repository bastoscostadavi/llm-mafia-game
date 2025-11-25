import sqlite3
import pandas as pd
from collections import Counter

conn = sqlite3.connect('database/mini_mafia.db')

print("=== DETECTIVE DISCLOSURE STRATEGIES ANALYSIS ===\n")

# Get performance data for top and bottom detectives
models_to_analyze = [
    ('gpt-5-mini', 'GPT-5 MINI (TOP)'),
    ('grok-3-mini', 'GROK 3 MINI (TOP)'),
    ('claude-opus-4', 'CLAUDE OPUS 4.1 (TOP)'),
    ('deepseek-v3', 'DEEPSEEK V3.1 (MIDDLE)'),
    ('llama-3.1-8b', 'LLAMA 3.1 8B (BOTTOM)')
]

for model_pattern, model_label in models_to_analyze:
    query = f"""
    SELECT
        g.game_id,
        g.winner,
        gp.character_name as detective_character
    FROM games g
    JOIN game_players gp ON g.game_id = gp.game_id
    JOIN players p ON gp.player_id = p.player_id
    WHERE p.model_name LIKE '%{model_pattern}%'
    AND gp.role = 'detective'
    """

    games = pd.read_sql_query(query, conn)

    print(f"\n=== {model_label} ===")
    print(f"Total games as detective: {len(games)}")
    print(f"Town wins: {(games['winner'] == 'town').sum()} ({100*(games['winner'] == 'town').sum()/len(games):.1f}%)")
    print(f"Mafia wins: {(games['winner'] == 'mafia').sum()} ({100*(games['winner'] == 'mafia').sum()/len(games):.1f}%)")

print("\n\n=== ANALYZING DISCLOSURE TIMING AND CONTENT ===\n")

def analyze_disclosure_strategy(model_pattern, model_label, sample_size=500):
    """Analyze when and how detective discloses information"""

    query = f"""
    SELECT
        g.game_id,
        g.winner,
        gp.character_name as detective_character
    FROM games g
    JOIN game_players gp ON g.game_id = gp.game_id
    JOIN players p ON gp.player_id = p.player_id
    WHERE p.model_name LIKE '%{model_pattern}%'
    AND gp.role = 'detective'
    LIMIT {sample_size}
    """

    games = pd.read_sql_query(query, conn)

    # Analyze disclosure patterns
    disclosed_first_discussion = 0
    disclosed_second_discussion = 0
    never_disclosed = 0

    # Content analysis
    direct_accusations = 0
    hedged_accusations = 0
    explained_methodology = 0

    total_analyzed = 0

    for _, row in games.iterrows():
        game_id = row['game_id']
        detective_char = row['detective_character']

        # Get roles
        roles_query = f"""
        SELECT character_name, role
        FROM game_players
        WHERE game_id = '{game_id}'
        """
        roles = pd.read_sql_query(roles_query, conn)

        mafioso_char = roles[roles['role'] == 'mafioso']['character_name'].values[0]

        # Get detective's discussion statements
        discuss_query = f"""
        SELECT step, parsed_result
        FROM game_sequence
        WHERE game_id = '{game_id}'
        AND action = 'discuss'
        AND actor = '{detective_char}'
        ORDER BY step
        """

        discussions = pd.read_sql_query(discuss_query, conn)

        if len(discussions) == 0:
            continue

        total_analyzed += 1

        # Check when they disclosed
        disclosed = False
        disclosure_round = None

        for idx, disc in discussions.iterrows():
            statement = str(disc['parsed_result']).lower()

            # Check if they revealed their investigation
            if (mafioso_char.lower() in statement and
                ('mafioso' in statement or 'mafia' in statement or
                 'investigated' in statement or 'detective' in statement)):
                disclosed = True
                disclosure_round = idx + 1  # 1-indexed

                # Analyze disclosure content
                if 'i am the detective' in statement or "i'm the detective" in statement:
                    direct_accusations += 1
                elif 'believe' in statement or 'think' in statement or 'suspect' in statement:
                    hedged_accusations += 1
                else:
                    direct_accusations += 1

                if 'investigated' in statement or 'investigation' in statement:
                    explained_methodology += 1

                break

        if disclosed:
            if disclosure_round == 1:
                disclosed_first_discussion += 1
            elif disclosure_round == 2:
                disclosed_second_discussion += 1
        else:
            never_disclosed += 1

    print(f"\n=== {model_label} ===")
    print(f"Games analyzed: {total_analyzed}")
    print(f"\nDisclosure timing:")
    print(f"  First discussion: {disclosed_first_discussion}/{total_analyzed} ({100*disclosed_first_discussion/total_analyzed:.1f}%)")
    print(f"  Second discussion: {disclosed_second_discussion}/{total_analyzed} ({100*disclosed_second_discussion/total_analyzed:.1f}%)")
    print(f"  Never disclosed: {never_disclosed}/{total_analyzed} ({100*never_disclosed/total_analyzed:.1f}%)")

    total_disclosed = disclosed_first_discussion + disclosed_second_discussion
    if total_disclosed > 0:
        print(f"\nDisclosure style (of those who disclosed):")
        print(f"  Direct accusations: {direct_accusations}/{total_disclosed} ({100*direct_accusations/total_disclosed:.1f}%)")
        print(f"  Hedged accusations: {hedged_accusations}/{total_disclosed} ({100*hedged_accusations/total_disclosed:.1f}%)")
        print(f"  Explained methodology: {explained_methodology}/{total_disclosed} ({100*explained_methodology/total_disclosed:.1f}%)")

# Analyze top performers
analyze_disclosure_strategy('gpt-5-mini', 'GPT-5 MINI (TOP)')
analyze_disclosure_strategy('grok-3-mini', 'GROK 3 MINI (TOP)')

# Analyze bottom performer
analyze_disclosure_strategy('llama-3.1-8b', 'LLAMA 3.1 8B (BOTTOM)')

# Analyze middle performer
analyze_disclosure_strategy('deepseek-v3', 'DEEPSEEK V3.1 (MIDDLE)')

conn.close()
