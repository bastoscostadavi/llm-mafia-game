#!/usr/bin/env python3
"""
Batch Configuration Table Generator
Shows all v4.1 batches with their model configurations in a clean table format.
"""

import os
import json
from pathlib import Path

def get_model_name(model_config):
    """Extract a clean model name from the config."""
    model = model_config.get('model', 'unknown')
    
    # Map model names to clean display names
    if 'grok-3-mini' in model:
        return 'Grok 3 Mini'
    elif 'gpt-4o-mini' in model:
        return 'GPT-4o Mini'
    elif 'gpt-4.1-mini' in model:
        return 'GPT-4.1 Mini'
    elif 'gpt-4.1-nano' in model:
        return 'GPT-4.1 Nano'
    elif 'gpt-5-mini' in model:
        return 'GPT-5 Mini'
    elif 'gpt-5' in model:
        return 'GPT-5'
    elif 'gpt-4o' in model:
        return 'GPT-4o'
    elif 'claude-opus-4' in model:
        return 'Claude Opus 4.1'
    elif 'Mistral-7B' in model:
        return 'Mistral 7B'
    elif 'Meta-Llama-3.1' in model:
        return 'Llama 3.1 8B'
    elif 'Qwen2.5-7B' in model:
        return 'Qwen 2.5 7B'
    else:
        # Truncate long model names
        return model[:15] + '...' if len(model) > 15 else model

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
                    mafioso = get_model_name(models.get('mafioso', {}))
                    detective = get_model_name(models.get('detective', {}))
                    villager = get_model_name(models.get('villager', {}))
                    
                except Exception as e:
                    mafioso = detective = villager = f'ERROR: {e}'
            else:
                mafioso = detective = villager = 'NO CONFIG'
            
            batches.append({
                'name': batch_name,
                'mafioso': mafioso,
                'detective': detective,
                'villager': villager,
                'games': game_count
            })
    
    # Sort batches by name
    batches.sort(key=lambda x: x['name'])
    
    # Print table header
    print("Batch Configuration Table")
    print("=" * 120)
    print(f"{'Batch':<32} | {'Mafioso':<18} | {'Detective':<18} | {'Villager':<18} | {'Games':>5}")
    print("-" * 32 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 20 + "+" + "-" * 7)
    
    # Print batch rows
    total_games = 0
    for batch in batches:
        print(f"{batch['name']:<32} | {batch['mafioso']:<18} | {batch['detective']:<18} | {batch['villager']:<18} | {batch['games']:>5}")
        total_games += batch['games']
    
    print("-" * 120)
    print(f"{'TOTAL':<32} | {'':18} | {'':18} | {'':18} | {total_games:>5}")
    print()
    
    # Show configuration summary
    print("Configuration Summary:")
    print("=" * 50)
    
    config_counts = {}
    config_games = {}
    
    for batch in batches:
        config_key = f"{batch['mafioso']} vs {batch['detective']}-{batch['villager']}"
        if config_key not in config_counts:
            config_counts[config_key] = 0
            config_games[config_key] = 0
        config_counts[config_key] += 1
        config_games[config_key] += batch['games']
    
    # Sort by total games (descending)
    sorted_configs = sorted(config_games.items(), key=lambda x: x[1], reverse=True)
    
    for config_key, total_games_for_config in sorted_configs:
        batches_count = config_counts[config_key]
        print(f"{config_key}: {batches_count} batch{'es' if batches_count > 1 else ''}, {total_games_for_config} games")

if __name__ == "__main__":
    main()