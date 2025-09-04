#!/usr/bin/env python3
"""
Model Performance Analysis Script

Creates a table with rows being different models and columns showing:
1. Percentage of votes that failed to parse
2. Percentage of "remained silent" messages 
3. Percentage of times that, as detective voted for mafioso
4. Percentage of times that, as mafioso voted for detective
5. Percentage of times that, as villager voted for mafioso
"""

import sqlite3
import pandas as pd
from pathlib import Path

def get_db_connection():
    """Get database connection."""
    # Try multiple possible locations for the database relative to mini-mafia folder
    possible_paths = [
        Path("../../mini_mafia.db"),  # Database is in the root project folder
        Path("database/mini_mafia.db"),  # Database might be in database subfolder
        Path("mini_mafia.db"),  # Database might be in current folder
    ]
    
    for db_path in possible_paths:
        if db_path.exists():
            return sqlite3.connect(db_path)
    
    raise FileNotFoundError(f"Database not found in any of these locations: {possible_paths}")

def get_models_list(conn):
    """Get all unique model combinations."""
    query = """
    SELECT DISTINCT 
        p.model_name,
        p.model_provider,
        CASE 
            WHEN p.model_provider = 'local' THEN p.model_name
            ELSE p.model_name
        END as display_name
    FROM players p 
    JOIN game_players gp ON p.player_id = gp.player_id
    ORDER BY p.model_provider, p.model_name
    """
    return pd.read_sql_query(query, conn)

def calculate_vote_parsing_failures(conn, model_name, model_provider):
    """Calculate percentage of votes that failed to parse for a model."""
    query = """
    SELECT 
        COUNT(*) as total_votes,
        SUM(CASE WHEN parsed_successfully = 0 THEN 1 ELSE 0 END) as failed_votes
    FROM votes v
    JOIN game_players gp ON v.game_id = gp.game_id AND v.character_name = gp.character_name
    JOIN players p ON gp.player_id = p.player_id
    WHERE p.model_name = ? AND p.model_provider = ?
    """
    result = conn.execute(query, (model_name, model_provider)).fetchone()
    total, failed = result[0], result[1]
    return (failed / total * 100) if total > 0 else 0

def calculate_silent_messages(conn, model_name, model_provider):
    """Calculate percentage of discussion messages that were 'remained silent'."""
    query = """
    SELECT 
        COUNT(*) as total_discussions,
        SUM(CASE WHEN LOWER(parsed_result) = 'remained silent' THEN 1 ELSE 0 END) as silent_messages
    FROM game_sequence gs
    JOIN game_players gp ON gs.game_id = gp.game_id AND gs.actor = gp.character_name
    JOIN players p ON gp.player_id = p.player_id
    WHERE gs.action = 'discuss' 
    AND p.model_name = ? AND p.model_provider = ?
    """
    result = conn.execute(query, (model_name, model_provider)).fetchone()
    total, silent = result[0], result[1]
    return (silent / total * 100) if total > 0 else 0

def calculate_role_voting_accuracy(conn, model_name, model_provider, voter_role, target_role):
    """Calculate percentage of times a role voted for a specific target role."""
    # First, get all votes by the voter role
    query = """
    SELECT 
        v.voted_for,
        gp_voter.role as voter_role,
        gp_target.role as target_role
    FROM votes v
    JOIN game_players gp_voter ON v.game_id = gp_voter.game_id AND v.character_name = gp_voter.character_name
    JOIN players p ON gp_voter.player_id = p.player_id
    LEFT JOIN game_players gp_target ON v.game_id = gp_target.game_id AND v.voted_for = gp_target.character_name
    WHERE gp_voter.role = ?
    AND p.model_name = ? AND p.model_provider = ?
    AND v.voted_for IS NOT NULL
    """
    result = conn.execute(query, (voter_role, model_name, model_provider)).fetchall()
    
    if not result:
        return 0
    
    total_votes = len(result)
    target_votes = sum(1 for vote in result if vote[2] == target_role)
    
    return (target_votes / total_votes * 100) if total_votes > 0 else 0

def generate_performance_table():
    """Generate the complete performance analysis table."""
    conn = get_db_connection()
    
    try:
        models_df = get_models_list(conn)
        results = []
        
        for _, model_row in models_df.iterrows():
            model_name = model_row['model_name']
            model_provider = model_row['model_provider']
            display_name = model_row['display_name']
            
            # Calculate all metrics
            vote_parsing_failures = calculate_vote_parsing_failures(conn, model_name, model_provider)
            silent_messages = calculate_silent_messages(conn, model_name, model_provider)
            detective_vs_mafioso = calculate_role_voting_accuracy(conn, model_name, model_provider, 'detective', 'mafioso')
            mafioso_vs_detective = calculate_role_voting_accuracy(conn, model_name, model_provider, 'mafioso', 'detective')
            villager_vs_mafioso = calculate_role_voting_accuracy(conn, model_name, model_provider, 'villager', 'mafioso')
            
            results.append({
                'Model': f"{display_name} ({model_provider})",
                'Vote Parse Failures (%)': round(vote_parsing_failures, 2),
                'Silent Messages (%)': round(silent_messages, 2),
                'Detective → Mafioso (%)': round(detective_vs_mafioso, 2),
                'Mafioso → Detective (%)': round(mafioso_vs_detective, 2),
                'Villager → Mafioso (%)': round(villager_vs_mafioso, 2)
            })
        
        # Create DataFrame and sort by model provider then name
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Model')
        
        return results_df
        
    finally:
        conn.close()

def main():
    """Main function to run the analysis and display results."""
    try:
        print("Analyzing model performance data...")
        print("=" * 80)
        
        results_df = generate_performance_table()
        
        # Display the table
        print(results_df.to_string(index=False))
        
        # Save to CSV
        output_file = "model_performance_analysis.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n\nResults saved to: {output_file}")
        
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())