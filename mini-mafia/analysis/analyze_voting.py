#!/usr/bin/env python3
"""
Analyze voting patterns between detectives and mafiosos
"""
import json
import os
import re
from collections import defaultdict

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

def analyze_voting_patterns():
    """Analyze voting patterns between detectives and mafiosos"""
    data_dir = "../data/batch"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    voting_stats = {
        'detective_voted_mafioso': 0,
        'mafioso_voted_detective': 0,
        'villager_voted_mafioso': 0,
        'voting_ties': 0,
        'total_number_of_games': 0,
        'by_config': defaultdict(lambda: {
            'detective_voted_mafioso': 0,
            'mafioso_voted_detective': 0,
            'villager_voted_mafioso': 0,
            'voting_ties': 0,
            'total_number_of_games': 0,
            'batches': set()
        })
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
                
                # Find detective, mafioso, and surviving villager
                detective = None
                mafioso = None
                surviving_villager = None
                
                for player in players:
                    if player.get('role') == 'detective':
                        detective = player
                    elif player.get('role') == 'mafioso':
                        mafioso = player
                    elif player.get('role') == 'villager' and player.get('alive', False):
                        # Find the villager who survived
                        surviving_villager = player
                
                
                voting_stats['total_number_of_games'] += 1
                voting_stats['by_config'][config_key]['total_number_of_games'] += 1
                voting_stats['by_config'][config_key]['batches'].add(batch_folder)
                
                # Extract voting information from memory
                detective_voted_mafioso = False
                mafioso_voted_detective = False
                villager_voted_mafioso = False
                has_voting_tie = False
                
                # Check all players' memories for voting patterns
                for player in players:
                    memory = player.get('memory', [])
                    
                    for memory_entry in memory:
                        # Look for voting lines like "Day X votes: Player1 voted for Player2, ..."
                        if "votes:" in memory_entry.lower():
                            # Parse voting pattern
                            votes = parse_votes(memory_entry)
                            
                            # Check if detective voted for mafioso
                            if detective['name'] in votes and votes[detective['name']] == mafioso['name']:
                                detective_voted_mafioso = True
                            
                            # Check if mafioso voted for detective
                            if mafioso['name'] in votes and votes[mafioso['name']] == detective['name']:
                                mafioso_voted_detective = True
                            
                            # Check villager voting patterns (surviving villager only)
                            if surviving_villager and surviving_villager['name'] in votes:
                                villager_vote_target = votes[surviving_villager['name']]
                                if villager_vote_target == mafioso['name']:
                                    villager_voted_mafioso = True
                            
                            # Check for ties by analyzing vote distribution
                            if votes:
                                vote_counts = {}
                                for voter, target in votes.items():
                                    vote_counts[target] = vote_counts.get(target, 0) + 1
                                
                                # Check if there's a tie (multiple players with the same max vote count)
                                if vote_counts:
                                    max_votes = max(vote_counts.values())
                                    players_with_max_votes = [player for player, count in vote_counts.items() if count == max_votes]
                                    if len(players_with_max_votes) > 1:
                                        has_voting_tie = True
                
                # Update statistics
                if detective_voted_mafioso:
                    voting_stats['detective_voted_mafioso'] += 1
                    voting_stats['by_config'][config_key]['detective_voted_mafioso'] += 1
                
                if mafioso_voted_detective:
                    voting_stats['mafioso_voted_detective'] += 1
                    voting_stats['by_config'][config_key]['mafioso_voted_detective'] += 1
                
                if villager_voted_mafioso:
                    voting_stats['villager_voted_mafioso'] += 1
                    voting_stats['by_config'][config_key]['villager_voted_mafioso'] += 1
                
                if has_voting_tie:
                    voting_stats['voting_ties'] += 1
                    voting_stats['by_config'][config_key]['voting_ties'] += 1
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Print voting pattern results
    print("=" * 60)
    print("VOTING PATTERN ANALYSIS")
    print("=" * 60)
    
    # Remove overall voting patterns section since it doesn't make sense across different configs
    
    print(f"\nBY CONFIGURATION (Prompt Version + Model + Temperature):")
    for config_key, config_stats in sorted(voting_stats['by_config'].items()):
        config_total = config_stats['total_number_of_games']
        batches_list = sorted(list(config_stats['batches']))
        if config_total > 0:
            config_detective_pct = (config_stats['detective_voted_mafioso'] / config_total) * 100
            config_mafioso_pct = (config_stats['mafioso_voted_detective'] / config_total) * 100
            config_villager_mafioso_pct = (config_stats['villager_voted_mafioso'] / config_total) * 100
            config_ties_pct = (config_stats['voting_ties'] / config_total) * 100
            
            print(f"  {config_key}:")
            print(f"    Batches: {', '.join(batches_list)} ({len(batches_list)} batches)")
            print(f"    Total number of games: {config_total}")
            print(f"    Detective voted mafioso: {config_stats['detective_voted_mafioso']} ({config_detective_pct:.1f}%)")
            print(f"    Mafioso voted detective: {config_stats['mafioso_voted_detective']} ({config_mafioso_pct:.1f}%)")
            print(f"    Villager voted mafioso: {config_stats['villager_voted_mafioso']} ({config_villager_mafioso_pct:.1f}%)")
            print(f"    Games with ties: {config_stats['voting_ties']} ({config_ties_pct:.1f}%)")
    
    print("=" * 60)
    return voting_stats

def find_exceptional_games():
    """Find games where voting patterns deviate from the norm"""
    data_dir = "../data/batch"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    detective_not_voting_mafioso = []
    mafioso_not_voting_detective = []
    
    # Process all batch folders
    for batch_folder in os.listdir(data_dir):
        batch_path = os.path.join(data_dir, batch_folder)
        
        # Skip non-directories and system files
        if not os.path.isdir(batch_path) or batch_folder.startswith('.'):
            continue
        
        # Process all JSON game files in this batch folder
        for filename in os.listdir(batch_path):
            if not filename.endswith('.json') or filename.endswith('_summary.json') or filename == 'prompt_config.json' or filename == 'batch_config.json':
                continue
                
            filepath = os.path.join(batch_path, filename)
            try:
                with open(filepath, 'r') as f:
                    game_data = json.load(f)
                
                game_id = game_data.get('game_id', filename)
                players = game_data.get('players', [])
                
                # Find detective and mafioso players
                detective = None
                mafioso = None
                
                for player in players:
                    if player.get('role') == 'detective':
                        detective = player
                    elif player.get('role') == 'mafioso':
                        mafioso = player
                
                # Skip games without both roles
                if not detective or not mafioso:
                    continue
                
                # Extract voting information from memory
                detective_voted_mafioso = False
                mafioso_voted_detective = False
                votes_info = None
                
                # Check all players' memories for voting patterns
                for player in players:
                    memory = player.get('memory', [])
                    
                    for memory_entry in memory:
                        # Look for voting lines like "Day X votes: Player1 voted for Player2, ..."
                        if "votes:" in memory_entry.lower():
                            votes_info = memory_entry
                            # Parse voting pattern
                            votes = parse_votes(memory_entry)
                            
                            # Check if detective voted for mafioso
                            if detective['name'] in votes and votes[detective['name']] == mafioso['name']:
                                detective_voted_mafioso = True
                            
                            # Check if mafioso voted for detective
                            if mafioso['name'] in votes and votes[mafioso['name']] == detective['name']:
                                mafioso_voted_detective = True
                
                # Record exceptional cases
                if not detective_voted_mafioso:
                    detective_not_voting_mafioso.append({
                        'game_id': game_id,
                        'detective': detective['name'],
                        'mafioso': mafioso['name'],
                        'votes_info': votes_info
                    })
                
                if not mafioso_voted_detective:
                    mafioso_not_voting_detective.append({
                        'game_id': game_id,
                        'detective': detective['name'],
                        'mafioso': mafioso['name'],
                        'votes_info': votes_info
                    })
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Print exceptional cases
    print("\n" + "=" * 60)
    print("EXCEPTIONAL VOTING CASES")
    print("=" * 60)
    
    print(f"\nGAMES WHERE DETECTIVE DID NOT VOTE FOR MAFIOSO ({len(detective_not_voting_mafioso)} cases):")
    for i, case in enumerate(detective_not_voting_mafioso[:5]):  # Show first 5
        print(f"  {i+1}. {case['game_id']}")
        print(f"     Detective: {case['detective']}, Mafioso: {case['mafioso']}")
        print(f"     Votes: {case['votes_info']}")
    
    print(f"\nGAMES WHERE MAFIOSO DID NOT VOTE FOR DETECTIVE ({len(mafioso_not_voting_detective)} cases):")
    for i, case in enumerate(mafioso_not_voting_detective[:5]):  # Show first 5
        print(f"  {i+1}. {case['game_id']}")
        print(f"     Detective: {case['detective']}, Mafioso: {case['mafioso']}")
        print(f"     Votes: {case['votes_info']}")
    
    print("=" * 60)
    
    return detective_not_voting_mafioso, mafioso_not_voting_detective

def parse_votes(vote_line):
    """Parse a vote line and return a dictionary of voter -> target"""
    votes = {}
    
    # Pattern to match "Player voted for Target"
    vote_pattern = r'(\w+) voted for (\w+)'
    matches = re.findall(vote_pattern, vote_line)
    
    for voter, target in matches:
        votes[voter] = target
    
    return votes

if __name__ == "__main__":
    analyze_voting_patterns()
    find_exceptional_games()
