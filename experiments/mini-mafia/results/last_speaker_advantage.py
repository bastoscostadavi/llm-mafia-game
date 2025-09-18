#!/usr/bin/env python3
"""
Create CSV table with role win rates (total and when last to speak) using Bayesian estimation.
Uses only the unique games from the benchmark table (14,000 games).
"""

import sqlite3
import math
import csv
from pathlib import Path


def bayesian_win_rate(wins, total_games):
    """Calculate Bayesian win rate estimate and uncertainty."""
    mean = (wins + 1) / (total_games + 2)
    variance = (mean * (1 - mean)) / (total_games + 3)
    std = math.sqrt(variance)
    return mean, std


def analyze_role_win_rates(db_path):
    """Analyze win rates by role (total and when last to speak) using benchmark games only."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get unique games from benchmark table with their winners
    cursor.execute("""
        SELECT DISTINCT b.game_id, g.winner
        FROM benchmark b
        JOIN games g ON b.game_id = g.game_id
        WHERE g.winner IS NOT NULL
    """)
    benchmark_games = cursor.fetchall()

    print(f"🔍 Analyzing {len(benchmark_games)} unique benchmark games...")

    # Initialize counters
    role_stats = {}
    for role in ['detective', 'mafioso', 'villager']:
        role_stats[role] = {
            'total_games': 0,
            'total_wins': 0,
            'last_speaker_games': 0,
            'last_speaker_wins': 0
        }

    for game_id, winner in benchmark_games:
        # Get all participating players (not killed)
        cursor.execute("""
            SELECT character_name, role, final_status
            FROM game_players
            WHERE game_id = ? AND final_status IN ('alive', 'arrested')
        """, (game_id,))
        game_players = cursor.fetchall()

        if not game_players:
            continue

        # Find the last person to speak in discussions
        cursor.execute("""
            SELECT actor FROM game_sequence
            WHERE game_id = ? AND action = 'discuss'
            ORDER BY step DESC LIMIT 1
        """, (game_id,))
        last_speaker_result = cursor.fetchone()
        last_speaker = last_speaker_result[0] if last_speaker_result else None

        # Map character names to roles for this game
        char_to_role = {}
        for character_name, role, final_status in game_players:
            char_to_role[character_name] = role

            # Count total games and wins for each role
            role_stats[role]['total_games'] += 1

            # Check if this role won
            role_won = False
            if winner == 'town' and role in ('detective', 'villager'):
                role_won = True
            elif winner == 'mafia' and role == 'mafioso':
                role_won = True

            if role_won:
                role_stats[role]['total_wins'] += 1

            # Check if this character was the last speaker
            if character_name == last_speaker:
                role_stats[role]['last_speaker_games'] += 1
                if role_won:
                    role_stats[role]['last_speaker_wins'] += 1

    conn.close()

    # Calculate win rates using Bayesian estimation
    results = []
    for role in ['detective', 'mafioso', 'villager']:
        stats = role_stats[role]

        # Total win rate
        if stats['total_games'] > 0:
            total_mean, total_std = bayesian_win_rate(stats['total_wins'], stats['total_games'])
            total_win_rate = total_mean * 100
            total_uncertainty = total_std * 100
        else:
            total_win_rate = 0
            total_uncertainty = 0

        # Last speaker win rate
        if stats['last_speaker_games'] > 0:
            last_mean, last_std = bayesian_win_rate(stats['last_speaker_wins'], stats['last_speaker_games'])
            last_win_rate = last_mean * 100
            last_uncertainty = last_std * 100
        else:
            last_win_rate = 0
            last_uncertainty = 0

        results.append({
            'role': role,
            'total_games': stats['total_games'],
            'total_wins': stats['total_wins'],
            'total_win_rate_pct': total_win_rate,
            'total_uncertainty_pct': total_uncertainty,
            'last_speaker_games': stats['last_speaker_games'],
            'last_speaker_wins': stats['last_speaker_wins'],
            'last_speaker_win_rate_pct': last_win_rate,
            'last_speaker_uncertainty_pct': last_uncertainty
        })

        # Print results
        print(f"{role.capitalize():9}: {stats['total_wins']:5}/{stats['total_games']:5} total = {total_win_rate:5.2f}±{total_uncertainty:.2f}%")
        if stats['last_speaker_games'] > 0:
            print(f"         : {stats['last_speaker_wins']:5}/{stats['last_speaker_games']:5} last  = {last_win_rate:5.2f}±{last_uncertainty:.2f}%")
        else:
            print(f"         : {'0':5}/{'0':5} last  = {'0.00':5}±{'0.00':.2f}%")

    return results


def main():
    """Main function."""
    db_path = "../database/mini_mafia.db"
    results = analyze_role_win_rates(db_path)

    # Write CSV
    output_file = "last_speark_advantage.csv"
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'role', 'total_games', 'total_wins', 'total_win_rate_pct', 'total_uncertainty_pct',
            'last_speaker_games', 'last_speaker_wins', 'last_speaker_win_rate_pct', 'last_speaker_uncertainty_pct'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Role win rates saved to: {output_file}")


if __name__ == "__main__":
    main()