#!/usr/bin/env python3
"""
Name bias analysis for Mini-Mafia game.
Analyzes win rates and uncertainty for each character name using Bayesian estimation.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from utils import bayesian_win_rate

def analyze_name_bias():
    """
    Analyze name bias by calculating win rates for each character name.
    """
    # Connect to database
    db_path = Path(__file__).parent / "database" / "mini_mafia.db"
    conn = sqlite3.connect(db_path)

    # Query to get only games that appear in the benchmark table with character outcomes
    # Use DISTINCT to avoid counting players multiple times (games appear 3x in benchmark for each capability)
    # Count players who were NOT killed (alive + arrested)
    query = """
    SELECT DISTINCT
        gp.character_name,
        gp.final_status,
        g.winner,
        gp.role,
        gp.game_id,
        CASE
            WHEN gp.role = 'detective' AND g.winner = 'town' THEN 1
            WHEN gp.role = 'villager' AND g.winner = 'town' THEN 1
            WHEN gp.role = 'mafioso' AND g.winner = 'mafia' THEN 1
            ELSE 0
        END as won
    FROM game_players gp
    JOIN games g ON gp.game_id = g.game_id
    JOIN benchmark b ON g.game_id = b.game_id
    WHERE gp.final_status != 'killed'
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Group by character name and calculate statistics
    results = []

    for name in ['Alice', 'Bob', 'Charlie', 'Diana']:
        name_data = df[df['character_name'] == name]

        if len(name_data) == 0:
            print(f"Warning: No data found for {name}")
            continue

        total_games = len(name_data)
        wins = name_data['won'].sum()

        # Calculate Bayesian win rate and uncertainty
        win_rate, uncertainty = bayesian_win_rate(wins, total_games)
        # Convert from percentage back to proportion for consistency with original format
        win_rate = win_rate / 100
        uncertainty = uncertainty / 100

        results.append({
            'character_name': name,
            'total_games': total_games,
            'wins': wins,
            'win_rate_bayesian': win_rate,
            'uncertainty_std': uncertainty,
            'confidence_interval_lower': max(0, win_rate - 1.96 * uncertainty),
            'confidence_interval_upper': min(1, win_rate + 1.96 * uncertainty)
        })

    # Add male and female aggregated entries
    # Male: Bob, Charlie; Female: Alice, Diana
    male_data = df[df['character_name'].isin(['Bob', 'Charlie'])]
    female_data = df[df['character_name'].isin(['Alice', 'Diana'])]

    for gender, gender_data in [('Male', male_data), ('Female', female_data)]:
        total_games = len(gender_data)
        wins = gender_data['won'].sum()
        win_rate, uncertainty = bayesian_win_rate(wins, total_games)
        # Convert from percentage back to proportion for consistency
        win_rate = win_rate / 100
        uncertainty = uncertainty / 100

        results.append({
            'character_name': gender,
            'total_games': total_games,
            'wins': wins,
            'win_rate_bayesian': win_rate,
            'uncertainty_std': uncertainty,
            'confidence_interval_lower': max(0, win_rate - 1.96 * uncertainty),
            'confidence_interval_upper': min(1, win_rate + 1.96 * uncertainty)
        })

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Save results to CSV
    output_path = Path(__file__).parent / "name_bias.csv"
    results_df.to_csv(output_path, index=False)

    return results_df

if __name__ == "__main__":
    analyze_name_bias()