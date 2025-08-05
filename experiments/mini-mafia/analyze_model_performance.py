#!/usr/bin/env python3
"""
Analyze model performance by role and model type.
This script demonstrates the new model tracking capabilities.
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_model_performance():
    """Analyze win rates by model and role"""
    data_dir = Path(__file__).parent / "data"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Track statistics by model and role
    stats = {
        'by_model_role': defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'total': 0})),
        'by_model': defaultdict(lambda: {'wins': 0, 'total': 0}),
        'by_role': defaultdict(lambda: {'wins': 0, 'total': 0})
    }
    
    batch_info = {}
    
    # Process all batch folders
    for batch_folder in data_dir.iterdir():
        if not batch_folder.is_dir() or not batch_folder.name.startswith('batch_'):
            continue
        
        print(f"Processing batch: {batch_folder.name}")
        
        # Read batch summary if available
        summary_files = list(batch_folder.glob("*summary*.json"))
        if summary_files:
            with open(summary_files[0], 'r') as f:
                batch_summary = json.load(f)
                batch_info[batch_folder.name] = batch_summary.get('configuration', {})
        
        # Process game files
        game_count = 0
        for game_file in batch_folder.glob("*_game_*.json"):
            try:
                with open(game_file, 'r') as f:
                    game_data = json.load(f)
                
                winner = game_data.get('winner')
                if winner not in ['good', 'evil']:
                    continue
                
                # Analyze each player
                for player in game_data.get('final_state', []):
                    role = player.get('role')
                    model_info = player.get('model', {})
                    model_name = model_info.get('model_name', 'unknown')
                    
                    if not role or not model_name:
                        continue
                    
                    # Determine if this player's team won
                    player_won = False
                    if winner == 'good' and role in ['detective', 'villager']:
                        player_won = True
                    elif winner == 'evil' and role == 'mafioso':
                        player_won = True
                    
                    # Update statistics
                    stats['by_model_role'][model_name][role]['total'] += 1
                    stats['by_model'][model_name]['total'] += 1
                    stats['by_role'][role]['total'] += 1
                    
                    if player_won:
                        stats['by_model_role'][model_name][role]['wins'] += 1
                        stats['by_model'][model_name]['wins'] += 1
                        stats['by_role'][role]['wins'] += 1
                
                game_count += 1
                
            except Exception as e:
                print(f"Error processing {game_file}: {e}")
        
        print(f"  Processed {game_count} games")
    
    # Print results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    print("\nBATCH CONFIGURATIONS:")
    for batch_name, config in batch_info.items():
        print(f"  {batch_name}:")
        model_configs = config.get('model_configs', {})
        if model_configs:
            for role, model_config in model_configs.items():
                model_name = model_config.get('model_path', '').split('/')[-1] or model_config.get('model', 'unknown')
                print(f"    {role}: {model_name}")
        else:
            print(f"    Model: {config.get('model', 'unknown')}")
    
    print(f"\nOVERALL PERFORMANCE BY MODEL:")
    for model_name, model_stats in stats['by_model'].items():
        total = model_stats['total']
        wins = model_stats['wins']
        win_rate = (wins / total * 100) if total > 0 else 0
        print(f"  {model_name}: {wins}/{total} ({win_rate:.1f}%)")
    
    print(f"\nPERFORMANCE BY ROLE:")
    for role, role_stats in stats['by_role'].items():
        total = role_stats['total']
        wins = role_stats['wins']
        win_rate = (wins / total * 100) if total > 0 else 0
        print(f"  {role}: {wins}/{total} ({win_rate:.1f}%)")
    
    print(f"\nPERFORMANCE BY MODEL AND ROLE:")
    for model_name, roles in stats['by_model_role'].items():
        print(f"  {model_name}:")
        for role, role_stats in roles.items():
            total = role_stats['total']
            wins = role_stats['wins']
            win_rate = (wins / total * 100) if total > 0 else 0
            print(f"    {role}: {wins}/{total} ({win_rate:.1f}%)")
    
    print("="*60)
    
    return stats

if __name__ == "__main__":
    analyze_model_performance()