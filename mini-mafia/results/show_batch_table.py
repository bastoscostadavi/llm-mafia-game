#!/usr/bin/env python3
"""
Batch Configuration Table Generator
Shows all v4.1 batches with their model configurations in a clean table format.
"""

import os
import json
import sys
from pathlib import Path

# Import from local analyze_results module
from analyze_results import extract_model_name

def get_experiment_type(mafioso, detective, villager):
    """Determine the experiment type and background based on model configuration."""
    
    # Determine which role varies and which are fixed (background)
    if detective == villager and detective != mafioso:
        # Mafioso experiment: detective and villager are the same (background), mafioso varies
        return "Mafioso", detective, mafioso, detective
    elif mafioso == villager and mafioso != detective:
        # Detective experiment: mafioso and villager are the same (background), detective varies  
        return "Detective", mafioso, detective, mafioso
    elif mafioso == detective and mafioso != villager:
        # Villager experiment: mafioso and detective are the same (background), villager varies
        return "Villager", mafioso, villager, mafioso
    else:
        # Mixed or unknown configuration
        return "Mixed", "Mixed", f"M:{mafioso[:8]} D:{detective[:8]} V:{villager[:8]}", "ZZZ_Mixed"

def create_experiment_title(experiment_type, background, varying_model):
    """Create a descriptive experiment title."""
    if experiment_type == "Mixed":
        return f"Mixed: {varying_model}"
    else:
        return f"{varying_model} {experiment_type.lower()} vs {background} background"

def main():
    # Path to batch directory
    batch_dir = Path(__file__).parent.parent / 'data' / 'batch'
    
    # Collect batch information
    batches = []
    
    for batch_path in batch_dir.glob('batch_*_v4.1'):
        if batch_path.is_dir():
            batch_name = batch_path.name
            config_file = batch_path / 'batch_config.json'
            
            # Count game files
            game_count = len(list(batch_path.glob('game_*.json')))
            
            # Load configuration
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        config = json.load(f)
                    
                    models = config.get('model_configs', {})
                    mafioso = extract_model_name(models.get('mafioso', {}))
                    detective = extract_model_name(models.get('detective', {}))
                    villager = extract_model_name(models.get('villager', {}))
                    
                    # Determine experiment type
                    experiment_type, background, varying_model, sort_background = get_experiment_type(mafioso, detective, villager)
                    experiment_title = create_experiment_title(experiment_type, background, varying_model)
                    
                except Exception as e:
                    mafioso = detective = villager = f'ERROR: {e}'
                    experiment_title = f'ERROR: {e}'
                    sort_background = 'ZZZ_ERROR'
            else:
                mafioso = detective = villager = 'NO CONFIG'
                experiment_title = 'NO CONFIG'
                sort_background = 'ZZZ_NO_CONFIG'
            
            batches.append({
                'name': batch_name,
                'mafioso': mafioso,
                'detective': detective,
                'villager': villager,
                'experiment_title': experiment_title,
                'sort_background': sort_background,
                'games': game_count
            })
    
    # Sort batches by background, then experiment type, then by varying model name
    batches.sort(key=lambda x: (x['sort_background'], x['experiment_title'], x['name']))
    
    # Print table header with experiment titles
    print("Batch Configuration Table with Experiment Classification")
    print("=" * 150)
    print(f"{'Batch':<32} | {'Experiment Description':<60} | {'Games':>5}")
    print("-" * 32 + "+" + "-" * 62 + "+" + "-" * 7)
    
    # Print batch rows with experiment titles
    total_games = 0
    for batch in batches:
        print(f"{batch['name']:<32} | {batch['experiment_title']:<60} | {batch['games']:>5}")
        total_games += batch['games']
    
    print("-" * 150)
    print(f"{'TOTAL':<32} | {'':60} | {total_games:>5}")
    print()
    
    # Also print the detailed table
    print("Detailed Model Configuration Table")
    print("=" * 120)
    print(f"{'Batch':<32} | {'Mafioso':<18} | {'Detective':<18} | {'Villager':<18} | {'Games':>5}")
    print("-" * 32 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 7)
    
    # Print detailed batch rows
    for batch in batches:
        print(f"{batch['name']:<32} | {batch['mafioso']:<18} | {batch['detective']:<18} | {batch['villager']:<18} | {batch['games']:>5}")
    
    print("-" * 120)
    print(f"{'TOTAL':<32} | {'':18} | {'':18} | {'':18} | {total_games:>5}")
    print()
    
    # Show experiment summary
    print("Experiment Summary:")
    print("=" * 80)
    
    experiment_counts = {}
    experiment_games = {}
    
    for batch in batches:
        exp_title = batch['experiment_title']
        if exp_title not in experiment_counts:
            experiment_counts[exp_title] = 0
            experiment_games[exp_title] = 0
        experiment_counts[exp_title] += 1
        experiment_games[exp_title] += batch['games']
    
    # Sort by total games (descending)
    sorted_experiments = sorted(experiment_games.items(), key=lambda x: x[1], reverse=True)
    
    for exp_title, total_games_for_exp in sorted_experiments:
        batches_count = experiment_counts[exp_title]
        print(f"{exp_title}: {batches_count} batch{'es' if batches_count > 1 else ''}, {total_games_for_exp} games")

if __name__ == "__main__":
    main()