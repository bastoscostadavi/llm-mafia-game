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

def create_win_counts_table():
    """Create win counts table from benchmark data."""

    db_path = "../database/mini_mafia.db"
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

    # Rename columns to match expected format
    column_mapping = {
        'deepseek_v3_1': 'DeepSeek V3.1',
        'gpt_4_1_mini': 'GPT-4.1 Mini',
        'gpt_5_mini': 'GPT-5 Mini',
        'grok_3_mini': 'Grok 3 Mini',
        'mistral_7b_instruct': 'Mistral 7B Instruct'
    }

    pivot_df.columns = [column_mapping.get(col, col) for col in pivot_df.columns]

    # Rename model values to match expected format
    model_mapping = {
        'claude_opus_4_1': 'Claude Opus 4.1',
        'claude_sonnet_4': 'Claude Sonnet 4',
        'deepseek_v3_1': 'DeepSeek V3.1',
        'gemini_2_5_flash_lite': 'Gemini 2.5 Flash Lite',
        'gpt_4_1_mini': 'GPT-4.1 Mini',
        'gpt_5_mini': 'GPT-5 Mini',
        'grok_3_mini': 'Grok 3 Mini',
        'llama_3_1_8b_instruct': 'Llama 3.1 8B Instruct',
        'mistral_7b_instruct': 'Mistral 7B Instruct',
        'qwen2_5_7b_instruct': 'Qwen2.5 7B Instruct'
    }

    pivot_df['model'] = pivot_df['model'].map(model_mapping)

    # Capitalize capability names
    pivot_df['capability'] = pivot_df['capability'].str.capitalize()

    # Reorder columns
    background_cols = ['DeepSeek V3.1', 'GPT-4.1 Mini', 'GPT-5 Mini', 'Grok 3 Mini', 'Mistral 7B Instruct']
    pivot_df = pivot_df[['capability', 'model'] + background_cols]

    # Save to CSV
    output_file = "win_counts.csv"
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