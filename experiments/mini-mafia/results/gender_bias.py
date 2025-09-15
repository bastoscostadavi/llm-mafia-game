#!/usr/bin/env python3
"""
Analyze character name win rates from benchmark data.

This script analyzes win rates for each character name (Alice, Bob, Charlie, Diana)
and by gender (Male, Female), considering only games where the character actually
played (was not the first villager killed).

For each character:
- player_name_total: games where character was not killed first (actually played)
- player_name_wins: games where character won and was not killed first
- win_rate: Bayesian win rate with uncertainty
"""

import sqlite3
import pandas as pd
import math

def calculate_bayesian_win_rate(wins, total):
    """Calculate Bayesian win rate and uncertainty using Beta distribution"""
    if total == 0:
        return 0.0, 0.0

    # Laplace rule of succession (Bayesian mean)
    bayesian_mean = (wins + 1) / (total + 2)

    # Bayesian uncertainty
    bayesian_var = (bayesian_mean * (1 - bayesian_mean)) / (total + 3)
    bayesian_sd = math.sqrt(bayesian_var)

    # Convert to percentages
    return bayesian_mean * 100, bayesian_sd * 100

def analyze_character_win_rates():
    """Analyze character win rates from benchmark data"""

    db_path = "../database/mini_mafia.db"
    conn = sqlite3.connect(db_path)

    print("🔍 Analyzing character win rates from benchmark data...")

    # Query to get unique games from benchmark table with game data
    query = """
    SELECT DISTINCT
        b.game_id,
        g.winner,
        gp.character_name,
        gp.role,
        gp.final_status
    FROM benchmark b
    JOIN games g ON b.game_id = g.game_id
    JOIN game_players gp ON b.game_id = gp.game_id
    ORDER BY b.game_id, gp.character_name
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"📊 Loaded {len(df)} character-game records from {df['game_id'].nunique()} unique benchmark games")

    # Character name mapping to gender
    gender_mapping = {
        'Alice': 'Female',
        'Bob': 'Male',
        'Charlie': 'Male',
        'Diana': 'Female'
    }

    # Initialize counters
    character_stats = {}
    gender_stats = {'Male': {'total': 0, 'wins': 0}, 'Female': {'total': 0, 'wins': 0}}

    for character in ['Alice', 'Bob', 'Charlie', 'Diana']:
        character_stats[character] = {'total': 0, 'wins': 0}

    # Process each unique game
    for game_id in df['game_id'].unique():
        game_data = df[df['game_id'] == game_id]

        # Find who was killed first (killed villager)
        killed_villagers = game_data[
            (game_data['final_status'] == 'killed') &
            (game_data['role'] == 'villager')
        ]

        if len(killed_villagers) != 1:
            # Skip games with unexpected number of killed villagers
            continue

        killed_character = killed_villagers.iloc[0]['character_name']
        winner = game_data.iloc[0]['winner']  # Same for all rows in the game

        # Process each character in this game
        for _, row in game_data.iterrows():
            character = row['character_name']
            role = row['role']

            # Skip the character that was killed first
            if character == killed_character:
                continue

            # This character actually played
            character_stats[character]['total'] += 1
            gender = gender_mapping[character]
            gender_stats[gender]['total'] += 1

            # Check if this character won
            character_won = False
            if role == 'mafioso' and winner == 'mafia':
                character_won = True
            elif role in ['detective', 'villager'] and winner == 'town':
                character_won = True

            if character_won:
                character_stats[character]['wins'] += 1
                gender_stats[gender]['wins'] += 1

    # Create results table
    results = []

    # Individual characters
    for character in ['Alice', 'Bob', 'Charlie', 'Diana']:
        total = character_stats[character]['total']
        wins = character_stats[character]['wins']
        win_rate, uncertainty = calculate_bayesian_win_rate(wins, total)

        results.append({
            'Character': character,
            'Total_Games': total,
            'Wins': wins,
            'Win_Rate': f"{win_rate:.2f} ± {uncertainty:.2f}%"
        })

        print(f"{character:7}: {wins:4}/{total:4} games = {win_rate:5.2f} ± {uncertainty:.2f}%")

    # Gender aggregates
    for gender in ['Male', 'Female']:
        total = gender_stats[gender]['total']
        wins = gender_stats[gender]['wins']
        win_rate, uncertainty = calculate_bayesian_win_rate(wins, total)

        results.append({
            'Character': gender,
            'Total_Games': total,
            'Wins': wins,
            'Win_Rate': f"{win_rate:.2f} ± {uncertainty:.2f}%"
        })

        print(f"{gender:7}: {wins:4}/{total:4} games = {win_rate:5.2f} ± {uncertainty:.2f}%")

    # Save to CSV
    results_df = pd.DataFrame(results)
    output_file = "gender_bias.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\n✅ Character win rates saved to: {output_file}")
    print(f"\nResults preview:")
    print(results_df)

    return results_df

if __name__ == "__main__":
    analyze_character_win_rates()