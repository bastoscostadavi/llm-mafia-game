#!/usr/bin/env python3
"""
Experiment Design Table Generator

Creates a comprehensive table showing the experimental design:
- Rows: Models being tested
- Columns: Experiment types (Deceive/Detect/Disclose) x Background models
- Values: Number of games for each combination

Usage: python experiment_design_table.py
"""

import sqlite3
import pandas as pd
from collections import defaultdict

def get_display_name(model_name):
    """Convert technical model names to display names"""
    display_mapping = {
        'gpt-4.1-mini': 'GPT-4.1 Mini',
        'gpt-4.1-nano': 'GPT-4.1 Nano',
        'gpt-4o': 'GPT-4o',
        'gpt-4o-mini': 'GPT-4o Mini',
        'gpt-5': 'GPT-5',
        'gpt-5-mini': 'GPT-5 Mini',
        'grok-3-mini': 'Grok 3 Mini',
        'grok-4': 'Grok-4',
        'claude-3-haiku-20240307': 'Claude 3 Haiku',
        'claude-3-5-haiku-latest': 'Claude 3.5 Haiku',
        'claude-sonnet-4-20250514': 'Claude Sonnet 4',
        'claude-opus-4-1-20250805': 'Claude Opus 4.1',
        'deepseek-chat': 'DeepSeek V3.1',
        'deepseek-reasoner': 'DeepSeek R1',
        'gemini-2.5-flash-lite': 'Gemini 2.5 Flash Lite',
        'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf': 'Mistral 7B Instruct',
        'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf': 'Llama 3.1 8B',
        'Qwen2.5-7B-Instruct-Q4_K_M.gguf': 'Qwen2.5 7B Instruct',
    }
    
    return display_mapping.get(model_name, model_name)

def analyze_experiment_design(db_path='database/mini_mafia.db'):
    """Analyze the experimental design from the database"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        # Get all game configurations with player models
        query = """
        SELECT 
            g.game_id,
            gp.role,
            p.model_name
        FROM games g
        JOIN game_players gp ON g.game_id = gp.game_id
        JOIN players p ON gp.player_id = p.player_id
        WHERE p.player_id IS NOT NULL  -- Exclude killed players
        ORDER BY g.game_id, gp.role
        """
        
        results = conn.execute(query).fetchall()
        
        # Group by game to get configurations
        games_by_id = defaultdict(lambda: {})
        
        for row in results:
            game_id = row['game_id']
            games_by_id[game_id][row['role']] = row['model_name']
        
        # Analyze experiment types
        experiment_data = defaultdict(lambda: defaultdict(int))
        
        for game_id, config in games_by_id.items():
            if len(config) != 3:  # Should have detective, mafioso, villager
                continue
                
            detective_model = get_display_name(config.get('detective'))
            mafioso_model = get_display_name(config.get('mafioso'))
            villager_model = get_display_name(config.get('villager'))
            
            # Deceive experiments (Mafioso varying, Detective==Villager background)
            if detective_model == villager_model:
                background = detective_model
                varying_model = mafioso_model
                experiment_key = f"Deceive_{background}"
                experiment_data[varying_model][experiment_key] += 1
            
            # Detect experiments (Detective varying, Mafioso==Villager background)  
            if mafioso_model == detective_model:
                background = mafioso_model
                varying_model = villager_model
                experiment_key = f"Detect_{background}"
                experiment_data[varying_model][experiment_key] += 1
            
            # Disclose experiments (Villager varying, Detective==Mafioso background)
            if villager_model == mafioso_model:
                background = villager_model
                varying_model = detective_model
                experiment_key = f"Disclose_{background}"
                experiment_data[varying_model][experiment_key] += 1
        
        return dict(experiment_data)
        
    finally:
        conn.close()

def create_experiment_design_table(experiment_data, output_file='experiment_design_table.csv'):
    """Create a comprehensive table showing the experimental design"""
    
    # Define the allowed models and backgrounds
    allowed_models = [
        'DeepSeek V3.1', 'DeepSeek R1', 'Claude Opus 4.1', 'Claude Sonnet 4', 'Gemini 2.5 Flash Lite',
        'Grok 3 Mini', 'GPT-4.1 Mini', 'GPT-5', 'GPT-5 Mini', 'Mistral 7B Instruct', 
        'Qwen2.5 7B Instruct', 'Llama 3.1 8B'
    ]
    
    allowed_backgrounds = [
        'Mistral 7B Instruct', 'GPT-4.1 Mini', 'GPT-5 Mini', 'Grok 3 Mini', 'DeepSeek V3.1'
    ]
    
    # Create column structure: Experiment_Background
    columns = []
    for experiment in ['Deceive', 'Detect', 'Disclose']:
        for background in allowed_backgrounds:
            columns.append(f"{experiment}_{background}")
    
    # Filter to only allowed models
    filtered_models = [model for model in allowed_models if model in experiment_data]
    
    # Create the data matrix
    data = []
    for model in filtered_models:
        row = {'Model': model}
        for col in columns:
            row[col] = experiment_data[model].get(col, 0)
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Reorder columns for better readability
    ordered_columns = ['Model']
    for experiment in ['Deceive', 'Detect', 'Disclose']:
        for background in allowed_backgrounds:
            ordered_columns.append(f"{experiment}_{background}")
    
    df = df[ordered_columns]
    
    # Add total columns
    for experiment in ['Deceive', 'Detect', 'Disclose']:
        exp_cols = [col for col in df.columns if col.startswith(f"{experiment}_")]
        df[f"{experiment}_Total"] = df[exp_cols].sum(axis=1)
    
    # Add grand total
    total_cols = [col for col in df.columns if col.endswith('_Total')]
    df['Grand_Total'] = df[total_cols].sum(axis=1)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Experiment design table saved to {output_file}")
    
    return df

def print_experiment_summary(df):
    """Print a summary of the experimental design"""
    print("EXPERIMENT DESIGN SUMMARY")
    print("=" * 120)
    print()
    
    # Print column headers with better formatting
    print(f"{'Model':<20}", end='')
    
    backgrounds = ['Mistral 7B Instruct', 'GPT-4.1 Mini', 'GPT-5 Mini', 'Grok 3 Mini', 'DeepSeek V3.1']
    
    # Print experiment headers
    for experiment in ['Deceive', 'Detect', 'Disclose']:
        print(f" | {experiment:^40}", end='')
    print(f" | {'Totals':^15}")
    
    # Print sub-headers for backgrounds
    print(f"{'':<20}", end='')
    for experiment in ['Deceive', 'Detect', 'Disclose']:
        for i, bg in enumerate(backgrounds):
            bg_short = bg.replace(' 7B Instruct', '').replace(' 3 Mini', '').replace('GPT-4.1 Mini', 'GPT-4.1').replace('GPT-5 Mini', 'GPT-5').replace('DeepSeek V3.1', 'DeepSeek')
            print(f" {bg_short:>9}", end='')
        print(" |", end='')
    print(f" {'Dec Det Dis Tot':>15}")
    
    print("-" * 120)
    
    # Print data rows
    for _, row in df.iterrows():
        model_short = row['Model'].replace(' 7B Instruct', '').replace(' 3 Mini', '').replace('GPT-4.1 Mini', 'GPT-4.1').replace('GPT-5 Mini', 'GPT-5').replace('DeepSeek V3.1', 'DeepSeek').replace('DeepSeek R1', 'DSR1')
        print(f"{model_short:<20}", end='')
        
        # Print experiment data
        for experiment in ['Deceive', 'Detect', 'Disclose']:
            for bg in backgrounds:
                col_name = f"{experiment}_{bg}"
                value = row[col_name] if col_name in row else 0
                print(f" {value:>9}", end='')
            print(" |", end='')
        
        # Print totals
        dec_total = row.get('Deceive_Total', 0)
        det_total = row.get('Detect_Total', 0)
        dis_total = row.get('Disclose_Total', 0)
        grand_total = row.get('Grand_Total', 0)
        print(f" {dec_total:>3} {det_total:>3} {dis_total:>3} {grand_total:>3}")
    
    print("-" * 120)
    
    # Print column totals
    print(f"{'TOTALS':<20}", end='')
    for experiment in ['Deceive', 'Detect', 'Disclose']:
        for bg in backgrounds:
            col_name = f"{experiment}_{bg}"
            if col_name in df.columns:
                total = df[col_name].sum()
                print(f" {total:>9}", end='')
            else:
                print(f" {0:>9}", end='')
        print(" |", end='')
    
    # Print grand totals
    dec_total = df['Deceive_Total'].sum() if 'Deceive_Total' in df.columns else 0
    det_total = df['Detect_Total'].sum() if 'Detect_Total' in df.columns else 0
    dis_total = df['Disclose_Total'].sum() if 'Disclose_Total' in df.columns else 0
    grand_total = df['Grand_Total'].sum() if 'Grand_Total' in df.columns else 0
    print(f" {dec_total:>3} {det_total:>3} {dis_total:>3} {grand_total:>3}")
    
    print("\n" + "=" * 120)

def main():
    """Main function to generate experiment design table"""
    print("🔄 Analyzing experimental design from database...")
    
    experiment_data = analyze_experiment_design()
    
    if not experiment_data:
        print("❌ No experimental data found in database")
        return
    
    print(f"📊 Found experimental data for {len(experiment_data)} models")
    
    df = create_experiment_design_table(experiment_data)
    print_experiment_summary(df)
    
    print(f"\n✅ Experiment design analysis complete!")

if __name__ == "__main__":
    main()