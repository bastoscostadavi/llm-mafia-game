#!/usr/bin/env python3
"""
Voting accuracy matrix: Detective/Villager performance by model vs different opponents
"""

import json
import os
import glob
from collections import defaultdict


def extract_model_name(model_config: dict) -> str:
    """Extract a clean model name from configuration."""
    if model_config.get('type') == 'local' or (model_config.get('model', '').endswith('.gguf')):
        model_filename = model_config.get('model', '')
        if 'mistral' in model_filename.lower():
            return 'Mistral'
        elif 'llama' in model_filename.lower():
            return 'Llama'
        elif 'qwen' in model_filename.lower():
            return 'Qwen'
        elif 'gemma' in model_filename.lower():
            return 'Gemma'
        else:
            return 'Unknown'
    elif model_config.get('type') == 'openai':
        model = model_config.get('model', '')
        if 'gpt-4o' in model:
            return 'GPT-4o'
        elif 'gpt-4' in model:
            return 'GPT-4'
        else:
            return f'OpenAI-{model}'
    elif model_config.get('type') == 'anthropic':
        model = model_config.get('model', '')
        if 'claude-sonnet-4' in model:
            return 'Claude'
        else:
            return f'Claude-{model}'
    else:
        return 'Unknown'


def parse_votes(vote_line: str):
    """Parse voting results from memory line."""
    votes = {}
    if not vote_line.startswith("Votes: "):
        return votes
    
    vote_str = vote_line[7:]  # Remove "Votes: "
    vote_pairs = vote_str.split(", ")
    
    for vote_pair in vote_pairs:
        try:
            parts = vote_pair.split(" voted for ")
            if len(parts) == 2:
                voter = parts[0].strip()
                target = parts[1].strip()
                votes[voter] = target
        except:
            continue
    
    return votes


def analyze_game(game_file: str):
    """Analyze voting accuracy for detective and villager in a single game."""
    try:
        with open(game_file, 'r') as f:
            game_data = json.load(f)
        
        # Get batch config to determine models
        batch_dir = os.path.dirname(game_file)
        config_file = os.path.join(batch_dir, 'batch_config.json')
        
        if not os.path.exists(config_file):
            return None
            
        with open(config_file, 'r') as f:
            batch_config = json.load(f)
        
        model_configs = batch_config.get('model_configs', {})
        detective_model = extract_model_name(model_configs.get('detective', {}))
        villager_model = extract_model_name(model_configs.get('villager', {}))
        mafioso_model = extract_model_name(model_configs.get('mafioso', {}))
        
        # Find players by role
        detective_name = None
        mafioso_name = None
        villager_names = []
        
        for player in game_data['players']:
            if player['role'] == 'detective':
                detective_name = player['name']
            elif player['role'] == 'mafioso':
                mafioso_name = player['name']
            elif player['role'] == 'villager':
                villager_names.append(player['name'])
        
        if not detective_name or not mafioso_name:
            return None
        
        # Find voting results
        votes = {}
        for player in game_data['players']:
            for memory_item in player['memory']:
                if 'Votes:' in memory_item:
                    votes = parse_votes(memory_item)
                    if votes:
                        break
            if votes:
                break
        
        if not votes:
            return None
        
        # Check votes
        detective_voted_mafioso = votes.get(detective_name) == mafioso_name
        villager_voted_mafioso = any(votes.get(v_name) == mafioso_name for v_name in villager_names)
        
        return {
            'detective_model': detective_model,
            'villager_model': villager_model,
            'mafioso_model': mafioso_model,
            'detective_voted_mafioso': detective_voted_mafioso,
            'villager_voted_mafioso': villager_voted_mafioso
        }
        
    except Exception as e:
        print(f"Error in {game_file}: {e}")
        return None


def main():
    print("🎯 Detective & Villager Voting Accuracy Matrix")
    print("=" * 60)
    
    # Find all batch directories
    batch_dirs = glob.glob("../data/batch/batch_*_v4.0")
    
    # Data structure: [detective_model][mafioso_model] = [total_games, correct_votes]
    detective_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    villager_stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    
    for batch_dir in sorted(batch_dirs):
        if not os.path.isdir(batch_dir):
            continue
            
        print(f"Processing {os.path.basename(batch_dir)}...")
        
        game_files = glob.glob(os.path.join(batch_dir, "game_*.json"))
        
        for game_file in game_files:
            result = analyze_game(game_file)
            if result:
                det_model = result['detective_model']
                vil_model = result['villager_model']
                maf_model = result['mafioso_model']
                
                # Update detective stats
                detective_stats[det_model][maf_model][0] += 1  # total games
                if result['detective_voted_mafioso']:
                    detective_stats[det_model][maf_model][1] += 1  # correct votes
                
                # Update villager stats
                villager_stats[vil_model][maf_model][0] += 1  # total games
                if result['villager_voted_mafioso']:
                    villager_stats[vil_model][maf_model][1] += 1  # correct votes
    
    # Get all unique models
    all_models = set()
    for det_model in detective_stats:
        all_models.add(det_model)
        for maf_model in detective_stats[det_model]:
            all_models.add(maf_model)
    
    for vil_model in villager_stats:
        all_models.add(vil_model)
        for maf_model in villager_stats[vil_model]:
            all_models.add(maf_model)
    
    mafioso_models = sorted(all_models)
    
    print(f"\n📊 DETECTIVE VOTING ACCURACY")
    print("-" * 60)
    
    # Header
    header = f"{'Detective Model':<15}"
    for maf_model in mafioso_models:
        header += f"{f'vs {maf_model}':<12}"
    print(header)
    print("-" * 60)
    
    # Detective rows
    for det_model in sorted(detective_stats.keys()):
        row = f"{det_model:<15}"
        for maf_model in mafioso_models:
            if maf_model in detective_stats[det_model]:
                total, correct = detective_stats[det_model][maf_model]
                if total > 0:
                    percentage = 100.0 * correct / total
                    row += f"{percentage:>8.1f}%   "
                else:
                    row += f"{'--':>8}%   "
            else:
                row += f"{'--':>8}%   "
        print(row)
    
    print(f"\n📊 VILLAGER VOTING ACCURACY")
    print("-" * 60)
    
    # Header
    header = f"{'Villager Model':<15}"
    for maf_model in mafioso_models:
        header += f"{f'vs {maf_model}':<12}"
    print(header)
    print("-" * 60)
    
    # Villager rows
    for vil_model in sorted(villager_stats.keys()):
        row = f"{vil_model:<15}"
        for maf_model in mafioso_models:
            if maf_model in villager_stats[vil_model]:
                total, correct = villager_stats[vil_model][maf_model]
                if total > 0:
                    percentage = 100.0 * correct / total
                    row += f"{percentage:>8.1f}%   "
                else:
                    row += f"{'--':>8}%   "
            else:
                row += f"{'--':>8}%   "
        print(row)


if __name__ == "__main__":
    main()