#!/usr/bin/env python3
"""
Create CSV table with role win rates (total and when last to speak) using Bayesian estimation.
Uses only the unique games from the benchmark table (14,000 games).
"""

import sqlite3
import math
import csv
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import bayesian_win_rate

# Set up paths
RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "serendipitous_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(__file__).parent.parent.parent / "database" / "mini_mafia.db"


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
            total_win_rate, total_uncertainty = bayesian_win_rate(stats['total_wins'], stats['total_games'])
        else:
            total_win_rate = 0
            total_uncertainty = 0

        # Last speaker win rate
        if stats['last_speaker_games'] > 0:
            last_win_rate, last_uncertainty = bayesian_win_rate(stats['last_speaker_wins'], stats['last_speaker_games'])
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

    # Calculate aggregate statistics across all roles
    total_games_all = sum(role_stats[role]['total_games'] for role in role_stats)
    total_wins_all = sum(role_stats[role]['total_wins'] for role in role_stats)
    last_speaker_games_all = sum(role_stats[role]['last_speaker_games'] for role in role_stats)
    last_speaker_wins_all = sum(role_stats[role]['last_speaker_wins'] for role in role_stats)

    # Compute aggregate win rates using Bayesian estimation
    if total_games_all > 0:
        agg_total_win_rate, agg_total_uncertainty = bayesian_win_rate(total_wins_all, total_games_all)
    else:
        agg_total_win_rate = 0
        agg_total_uncertainty = 0

    if last_speaker_games_all > 0:
        agg_last_win_rate, agg_last_uncertainty = bayesian_win_rate(last_speaker_wins_all, last_speaker_games_all)
    else:
        agg_last_win_rate = 0
        agg_last_uncertainty = 0

    # Add aggregate results
    aggregate_result = {
        'role': 'aggregate',
        'total_games': total_games_all,
        'total_wins': total_wins_all,
        'total_win_rate_pct': agg_total_win_rate,
        'total_uncertainty_pct': agg_total_uncertainty,
        'last_speaker_games': last_speaker_games_all,
        'last_speaker_wins': last_speaker_wins_all,
        'last_speaker_win_rate_pct': agg_last_win_rate,
        'last_speaker_uncertainty_pct': agg_last_uncertainty
    }
    results.append(aggregate_result)

    # Print aggregate results
    print(f"\n{'Aggregate':9}: {total_wins_all:5}/{total_games_all:5} total = {agg_total_win_rate:5.2f}±{agg_total_uncertainty:.2f}%")
    if last_speaker_games_all > 0:
        print(f"         : {last_speaker_wins_all:5}/{last_speaker_games_all:5} last  = {agg_last_win_rate:5.2f}±{agg_last_uncertainty:.2f}%")
    else:
        print(f"         : {'0':5}/{'0':5} last  = {'0.00':5}±{'0.00':.2f}%")

    return results


def main():
    """Main function."""
    results = analyze_role_win_rates(DB_PATH)

    # Write CSV
    output_file = RESULTS_DIR / "last_speaker_advantage.csv"
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