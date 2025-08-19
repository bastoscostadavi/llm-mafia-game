#!/usr/bin/env python3
"""
Analyze all game results and count evil vs good victories
"""
import json
import os
from collections import defaultdict

def determine_winner(players):
    """Determine who won the game from player data"""
    arrested_player = next((p for p in players if p.get('imprisoned')), None)
    if arrested_player:
        if arrested_player.get('role') == "mafioso":
            return "good"
        else:  # villager or detective arrested
            return "evil"
    return "unknown"

def get_batch_config(batch_path):
    """Extract configuration from batch folder"""
    config_path = os.path.join(batch_path, 'batch_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                prompt_version = config.get('prompt_config', {}).get('version', 'unknown')
                
                # Extract model info - assume all roles use same model config for grouping
                model_configs = config.get('model_configs', {})
                detective_config = model_configs.get('detective', {})
                
                # Extract model name from path or type
                model_name = 'unknown'
                if detective_config.get('type') == 'local':
                    model_path = detective_config.get('model_path', '')
                    if 'mistral' in model_path.lower():
                        model_name = 'mistral'
                    elif 'qwen' in model_path.lower():
                        model_name = 'qwen'
                    elif 'llama' in model_path.lower():
                        model_name = 'llama'
                    else:
                        model_name = 'local_' + os.path.basename(model_path).split('.')[0]
                else:
                    model_name = detective_config.get('model', 'api_model')
                
                temperature = detective_config.get('temperature', 0.7)
                
                return prompt_version, model_name, temperature
        except Exception as e:
            print(f"Error reading config from {config_path}: {e}")
    
    return 'unknown', 'unknown', 'unknown'

def create_config_key(prompt_version, model_name, temperature):
    """Create a readable key for configuration grouping"""
    return f"{prompt_version}_{model_name}_t{temperature}"

def analyze_games():
    data_dir = "../data/batch"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}. Run from mini-mafia/analysis/ directory.")
        return
    
    results = {
        'good': 0,
        'evil': 0,
        'total': 0,
        'by_config': defaultdict(lambda: {'good': 0, 'evil': 0, 'total': 0, 'batches': set()})
    }
    
    # Process all batch folders
    for batch_folder in os.listdir(data_dir):
        batch_path = os.path.join(data_dir, batch_folder)
        
        # Skip non-directories and system files
        if not os.path.isdir(batch_path) or batch_folder.startswith('.'):
            continue
        
        print(f"Processing batch folder: {batch_folder}")
        
        # Get configuration for this batch
        prompt_version, model_name, temperature = get_batch_config(batch_path)
        config_key = create_config_key(prompt_version, model_name, temperature)
        
        # Process all JSON game files in this batch folder
        for filename in os.listdir(batch_path):
            if not filename.endswith('.json') or filename.endswith('_summary.json') or filename == 'prompt_config.json' or filename == 'batch_config.json':
                continue
                
            filepath = os.path.join(batch_path, filename)
            try:
                with open(filepath, 'r') as f:
                    game_data = json.load(f)
                
                players = game_data.get('players', [])
                
                # Determine winner from player data
                winner = determine_winner(players)
                
                if winner in ['good', 'evil']:
                    results[winner] += 1
                    results['total'] += 1
                    results['by_config'][config_key][winner] += 1
                    results['by_config'][config_key]['total'] += 1
                    results['by_config'][config_key]['batches'].add(batch_folder)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Print results
    print("=" * 60)
    print("MAFIA GAME RESULTS ANALYSIS")
    print("=" * 60)
    
    print(f"\nBY CONFIGURATION (Prompt Version + Model + Temperature):")
    for config_key, config_results in sorted(results['by_config'].items()):
        batches_list = sorted(list(config_results['batches']))
        print(f"  {config_key}:")
        print(f"    Batches: {', '.join(batches_list)} ({len(batches_list)} batches)")
        print(f"    Total games: {config_results['total']}")
        
        if config_results['total'] > 0:
            # Calculate config uncertainties
            config_good_uncertainty = (config_results['good']**0.5) if config_results['good'] > 0 else 0
            config_evil_uncertainty = (config_results['evil']**0.5) if config_results['evil'] > 0 else 0
            
            # Calculate config win rate uncertainties
            config_good_rate = config_results['good']/config_results['total']*100
            config_evil_rate = config_results['evil']/config_results['total']*100
            config_good_rate_uncertainty = config_good_uncertainty/config_results['total']*100
            config_evil_rate_uncertainty = config_evil_uncertainty/config_results['total']*100
            
            print(f"    Good: {config_results['good']} ± {config_good_uncertainty:.0f} ({config_good_rate:.1f}% ± {config_good_rate_uncertainty:.1f}%)")
            print(f"    Evil: {config_results['evil']} ± {config_evil_uncertainty:.0f} ({config_evil_rate:.1f}% ± {config_evil_rate_uncertainty:.1f}%)")
    
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    analyze_games()