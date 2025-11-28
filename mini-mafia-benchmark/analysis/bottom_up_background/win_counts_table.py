#!/usr/bin/env python3
"""
Create win_counts table from benchmark data.

This script analyzes games in the benchmark table to create a CSV with win counts
for each (capability, target_model, background) combination.

Output format:
- Capability: deceive, detect, disclose
- Model: target model being tested
- Columns for each background model: win counts out of 100 games
"""

import sqlite3
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import get_display_name

def create_win_counts_table():
    """Create win counts table from benchmark data."""

    db_path = "../../database/mini_mafia.db"
    conn = sqlite3.connect(db_path)

    # Query to get win counts for each combination
    query = """
    SELECT
        b.capability,
        b.target as model,
        b.background,
        COUNT(CASE WHEN g.winner =
            CASE b.capability
                WHEN 'deceive' THEN 'mafia'
                WHEN 'detect' THEN 'town'
                WHEN 'disclose' THEN 'town'
            END THEN 1 END) as wins,
        COUNT(*) as total_games
    FROM benchmark b
    JOIN games g ON b.game_id = g.game_id
    GROUP BY b.capability, b.target, b.background
    ORDER BY b.capability, b.target, b.background
    """

    # Execute query
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Pivot table to get backgrounds as columns
    pivot_df = df.pivot_table(
        index=['capability', 'model'],
        columns='background',
        values='wins',
        fill_value=0
    ).reset_index()

    # Use centralized display name mapping for columns
    pivot_df.columns = [get_display_name(col) for col in pivot_df.columns]

    # Use centralized display name mapping for models
    pivot_df['model'] = pivot_df['model'].map(get_display_name)

    # Capitalize capability names
    pivot_df['capability'] = pivot_df['capability'].str.capitalize()

    # Reorder columns - get background columns dynamically
    background_cols = [col for col in pivot_df.columns if col not in ['capability', 'model']]
    pivot_df = pivot_df[['capability', 'model'] + sorted(background_cols)]

    # Save to CSV
    output_dir = Path(__file__).parent.parent.parent / "results" / "bottom_up_background"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "win_counts.csv"
    pivot_df.to_csv(output_file, index=False, float_format='%.1f')

    print(f"Win counts table saved to {output_file}")
    print(f"Shape: {pivot_df.shape}")
    print("\nFirst few rows:")
    print(pivot_df.head())

    # Show sample for each capability
    for capability in ['Deceive', 'Detect', 'Disclose']:
        print(f"\n{capability} capability:")
        cap_data = pivot_df[pivot_df['capability'] == capability]
        print(cap_data.head())

if __name__ == "__main__":
    create_win_counts_table()