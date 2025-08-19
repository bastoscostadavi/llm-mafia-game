#!/usr/bin/env python3
"""
Model Performance Analysis for v4.0 Mini-Mafia Games

Analyzes evil win rates across different model combinations in v4.0 batches.
Groups by model configuration pattern (MmodelMDModelDVModelV) to show
how different models perform in each role against various backgrounds.

Ignores temperature variations and focuses on model performance patterns.
"""

import json
import os
import math
from collections import defaultdict
from pathlib import Path

def calculate_sem(successes, total):
    """Calculate standard error for binomial proportion"""
    if total == 0:
        return 0.0
    
    p = successes / total
    n = total
    
    # Standard Error of the Mean (SEM) for binomial proportion
    sem = math.sqrt(p * (1 - p) / n)
    
    return sem * 100

def extract_model_name(model_config):
    """Extract short model name from model configuration"""
    if not model_config:
        return "unknown"
    
    # Handle API models (OpenAI, Anthropic)
    if model_config.get('type') == 'openai':
        model = model_config.get('model', 'unknown')
        if model.startswith('gpt-5'):
            return 'GPT-5'
        elif model.startswith('gpt-4'):
            return 'GPT-4'
        elif model.startswith('gpt-3.5'):
            return 'GPT-3.5'
        else:
            return model.upper()
    
    elif model_config.get('type') == 'anthropic':
        model = model_config.get('model', 'unknown')
        if 'claude-3-haiku' in model:
            return 'Claude-3-Haiku'
        elif 'claude-3-sonnet' in model:
            return 'Claude-3-Sonnet'
        elif 'claude-3-opus' in model:
            return 'Claude-3-Opus'
        else:
            return f"Claude-{model}"
    
    # Handle local models
    elif model_config.get('type') == 'local':
        model_path = model_config.get('model_path', '')
        if not model_path:
            return "unknown"
        
        filename = os.path.basename(model_path)
        
        # Map model files to short names
        model_mapping = {
            'mistral.gguf': 'Mistral',
            'Qwen2.5-7B-Instruct-Q4_K_M.gguf': 'Qwen2.5',
            'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf': 'Llama3.1',
            'openai_gpt-oss-20b-Q4_K_M.gguf': 'GPT-OSS'
        }
        
        return model_mapping.get(filename, filename.replace('.gguf', ''))
    
    else:
        return "unknown"

def create_config_key(model_configs):
    """Create a configuration key from model configs (MafiosoDetectiveVillager)"""
    mafioso_model = extract_model_name(model_configs.get('mafioso', {}))
    detective_model = extract_model_name(model_configs.get('detective', {}))
    villager_model = extract_model_name(model_configs.get('villager', {}))
    
    return f"M{mafioso_model}D{detective_model}V{villager_model}"

def determine_winner(players):
    """Determine who won the game"""
    arrested = next((p for p in players if p.get('imprisoned', False)), None)
    if arrested:
        if arrested['role'] == "mafioso":
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
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None

def analyze_v4_batches():
    """Analyze all v4.0 batches for model performance"""
    data_dir = "../data/batch"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found. Run from mini-mafia/analysis/ directory.")
        return
    
    # Find all v4.0 batch directories
    v4_batches = []
    for batch_dir in os.listdir(data_dir):
        if batch_dir.endswith("_v4.0"):
            batch_path = os.path.join(data_dir, batch_dir)
            if os.path.isdir(batch_path):
                v4_batches.append(batch_path)
    
    if not v4_batches:
        print("No v4.0 batch directories found.")
        return
    
    print(f"Found {len(v4_batches)} v4.0 batch directories")
    print("=" * 60)
    
    # Track results by model configuration
    config_results = defaultdict(lambda: {'evil_wins': 0, 'total_games': 0})
    batch_details = []
    
    for batch_path in sorted(v4_batches):
        batch_name = os.path.basename(batch_path)
        print(f"\nAnalyzing {batch_name}...")
        
        # Get batch configuration
        config = get_batch_config(batch_path)
        if not config:
            print(f"  Warning: Could not read batch config for {batch_name}")
            continue
        
        model_configs = config.get('model_configs', {})
        config_key = create_config_key(model_configs)
        
        # Count games and wins
        evil_wins = 0
        total_games = 0
        
        # Process all game files in batch
        for game_file in os.listdir(batch_path):
            if game_file.startswith('game_') and game_file.endswith('.json'):
                game_path = os.path.join(batch_path, game_file)
                try:
                    with open(game_path, 'r') as f:
                        game_data = json.load(f)
                    
                    players = game_data.get('players', [])
                    winner = determine_winner(players)
                    
                    if winner != "unknown":
                        total_games += 1
                        if winner == "evil":
                            evil_wins += 1
                            
                except (json.JSONDecodeError, IOError, KeyError) as e:
                    print(f"  Warning: Could not process {game_file}: {e}")
                    continue
        
        if total_games > 0:
            evil_win_rate = (evil_wins / total_games) * 100
            print(f"  Configuration: {config_key}")
            print(f"  Games: {total_games}, Evil wins: {evil_wins} ({evil_win_rate:.1f}%)")
            
            # Add to overall results
            config_results[config_key]['evil_wins'] += evil_wins
            config_results[config_key]['total_games'] += total_games
            
            batch_details.append({
                'batch': batch_name,
                'config': config_key,
                'games': total_games,
                'evil_wins': evil_wins,
                'evil_win_rate': evil_win_rate
            })
        else:
            print(f"  No valid games found in {batch_name}")
    
    # Print summary by model configuration
    print("\n" + "=" * 60)
    print("SUMMARY BY MODEL CONFIGURATION")
    print("=" * 60)
    print(f"{'Configuration':<25} {'Games':<8} {'Evil Wins':<10} {'Win Rate':<12}")
    print("-" * 60)
    
    for config_key in sorted(config_results.keys()):
        results = config_results[config_key]
        evil_wins = results['evil_wins']
        total_games = results['total_games']
        win_rate = (evil_wins / total_games) * 100 if total_games > 0 else 0
        
        # Calculate SEM
        sem = calculate_sem(evil_wins, total_games)
        
        print(f"{config_key:<25} {total_games:<8} {evil_wins:<10} {win_rate:.1f}% ± {sem:.1f}%")
    
    # Print detailed breakdown
    print("\n" + "=" * 60)
    print("DETAILED BREAKDOWN BY BATCH")
    print("=" * 60)
    
    for detail in sorted(batch_details, key=lambda x: x['config']):
        sem = calculate_sem(detail['evil_wins'], detail['games'])
        print(f"{detail['batch']}: {detail['config']} - "
              f"{detail['games']} games, {detail['evil_wins']} evil wins "
              f"({detail['evil_win_rate']:.1f}% ± {sem:.1f}%)")
    
    print(f"\nTotal configurations analyzed: {len(config_results)}")
    print(f"Total v4.0 batches: {len(batch_details)}")
    
    # Add explanation of uncertainty calculation
    print("\n" + "=" * 60)
    print("STATISTICAL NOTES")
    print("=" * 60)
    print("• Standard Error (±): SEM = √[p(1-p)/n] where p = win rate, n = sample size")
    print("• Larger sample sizes → smaller standard errors → more precise estimates")

if __name__ == "__main__":
    analyze_v4_batches()