#!/usr/bin/env python3
"""
Individual Model Performance Analysis for v4.0 Mini-Mafia Games

Analyzes individual model behavior across all v4.0 games:
1. Percentage of "remained silent" messages
2. Percentage of random votes (failed format)
3. Detective accuracy: % times voted for mafioso when playing detective
4. Mafioso targeting: % times voted for detective when playing mafioso

Groups results by model type regardless of role configuration.
"""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

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

def analyze_player_messages(memory, player_name):
    """Analyze messages from a player's memory"""
    messages = []
    silent_count = 0
    
    for entry in memory:
        # Look for player's own messages (format: "You: message" or "PlayerName: message")
        if entry.startswith(f"You: "):
            message = entry[5:]  # Remove "You: " prefix
            messages.append(message)
            if "remained silent" in message:
                silent_count += 1
        elif entry.startswith(f"{player_name}: "):
            message = entry[len(player_name) + 2:]  # Remove "PlayerName: " prefix
            messages.append(message)
            if "remained silent" in message:
                silent_count += 1
    
    return len(messages), silent_count

def analyze_player_votes(memory, player_name):
    """Analyze votes from a player's memory"""
    votes = []
    random_votes = 0
    
    for entry in memory:
        # Look for voting results (format: "Votes: ...")
        if entry.startswith("Votes: "):
            vote_line = entry[7:]  # Remove "Votes: " prefix
            
            # Extract individual votes using regex
            # Pattern: "PlayerName voted for Target" or "PlayerName voted for Target (explanation)"
            vote_pattern = r'(\w+) voted for (\w+)(?:\s*\([^)]+\))?'
            matches = re.findall(vote_pattern, vote_line)
            
            for voter, target in matches:
                if voter == player_name or (voter == "You" and player_name):
                    votes.append(target)
                    # Check if this was a random vote
                    if f"{voter} voted for {target} (the vote was cast randomly because of a failed format)" in vote_line:
                        random_votes += 1
    
    return len(votes), random_votes

def get_mafioso_and_detective(players):
    """Find the mafioso and detective in the game"""
    mafioso = None
    detective = None
    
    for player in players:
        if player['role'] == 'mafioso':
            mafioso = player['name']
        elif player['role'] == 'detective':
            detective = player['name']
    
    return mafioso, detective

def analyze_strategic_voting(memory, player_name, player_role, mafioso_name, detective_name):
    """Analyze strategic voting patterns for detective and mafioso"""
    votes_for_target = 0
    total_votes = 0
    
    target_name = None
    if player_role == "detective":
        target_name = mafioso_name
    elif player_role == "mafioso":
        target_name = detective_name
    else:
        return 0, 0  # Not applicable for villagers
    
    if not target_name:
        return 0, 0
    
    for entry in memory:
        if entry.startswith("Votes: "):
            vote_line = entry[7:]
            vote_pattern = r'(\w+) voted for (\w+)(?:\s*\([^)]+\))?'
            matches = re.findall(vote_pattern, vote_line)
            
            for voter, voted_for in matches:
                if voter == player_name or voter == "You":
                    total_votes += 1
                    if voted_for == target_name:
                        votes_for_target += 1
                    break  # Only count one vote per voting round
    
    return votes_for_target, total_votes

def analyze_model_performance():
    """Analyze model performance across all v4.0 batches"""
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
    
    print(f"Analyzing model performance across {len(v4_batches)} v4.0 batches...")
    print("=" * 70)
    
    # Track performance by model
    model_stats = defaultdict(lambda: {
        'total_messages': 0,
        'silent_messages': 0,
        'total_votes': 0,
        'random_votes': 0,
        'detective_votes_for_mafioso': 0,
        'detective_total_votes': 0,
        'mafioso_votes_for_detective': 0,
        'mafioso_total_votes': 0
    })
    
    for batch_path in sorted(v4_batches):
        batch_name = os.path.basename(batch_path)
        
        # Get batch configuration to know model assignments
        config = get_batch_config(batch_path)
        if not config:
            print(f"Warning: Could not read batch config for {batch_name}")
            continue
        
        model_configs = config.get('model_configs', {})
        
        # Create role-to-model mapping
        role_to_model = {}
        for role, role_config in model_configs.items():
            model_name = extract_model_name(role_config)
            role_to_model[role] = model_name
        
        # Process all games in this batch
        for game_file in os.listdir(batch_path):
            if not (game_file.startswith('game_') and game_file.endswith('.json')):
                continue
                
            game_path = os.path.join(batch_path, game_file)
            try:
                with open(game_path, 'r') as f:
                    game_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            
            players = game_data.get('players', [])
            if not players:
                continue
            
            # Get mafioso and detective names
            mafioso_name, detective_name = get_mafioso_and_detective(players)
            
            # Analyze each player
            for player in players:
                player_name = player['name']
                player_role = player['role']
                memory = player.get('memory', [])
                
                # Get the model for this player's role
                model_name = role_to_model.get(player_role, 'unknown')
                if model_name == 'unknown':
                    continue
                
                stats = model_stats[model_name]
                
                # Analyze messages
                total_messages, silent_messages = analyze_player_messages(memory, player_name)
                stats['total_messages'] += total_messages
                stats['silent_messages'] += silent_messages
                
                # Analyze votes
                total_votes, random_votes = analyze_player_votes(memory, player_name)
                stats['total_votes'] += total_votes
                stats['random_votes'] += random_votes
                
                # Analyze strategic voting
                strategic_votes, strategic_total = analyze_strategic_voting(
                    memory, player_name, player_role, mafioso_name, detective_name)
                
                if player_role == "detective":
                    stats['detective_votes_for_mafioso'] += strategic_votes
                    stats['detective_total_votes'] += strategic_total
                elif player_role == "mafioso":
                    stats['mafioso_votes_for_detective'] += strategic_votes
                    stats['mafioso_total_votes'] += strategic_total
    
    # Print results
    print(f"{'Model':<12} {'Silent%':<8} {'Random%':<9} {'DetAcc%':<9} {'MafTarg%':<10}")
    print("-" * 70)
    
    for model_name in sorted(model_stats.keys()):
        stats = model_stats[model_name]
        
        # Calculate percentages
        silent_pct = (stats['silent_messages'] / stats['total_messages'] * 100) if stats['total_messages'] > 0 else 0
        random_pct = (stats['random_votes'] / stats['total_votes'] * 100) if stats['total_votes'] > 0 else 0
        det_acc_pct = (stats['detective_votes_for_mafioso'] / stats['detective_total_votes'] * 100) if stats['detective_total_votes'] > 0 else 0
        maf_targ_pct = (stats['mafioso_votes_for_detective'] / stats['mafioso_total_votes'] * 100) if stats['mafioso_total_votes'] > 0 else 0
        
        print(f"{model_name:<12} {silent_pct:<8.1f} {random_pct:<9.1f} {det_acc_pct:<9.1f} {maf_targ_pct:<10.1f}")
    
    # Print detailed statistics
    print("\n" + "=" * 70)
    print("DETAILED STATISTICS")
    print("=" * 70)
    
    for model_name in sorted(model_stats.keys()):
        stats = model_stats[model_name]
        print(f"\n{model_name}:")
        print(f"  Messages: {stats['silent_messages']}/{stats['total_messages']} silent "
              f"({(stats['silent_messages']/stats['total_messages']*100):.1f}%)")
        print(f"  Votes: {stats['random_votes']}/{stats['total_votes']} random "
              f"({(stats['random_votes']/stats['total_votes']*100):.1f}%)")
        # Detective accuracy (only if model played as detective)
        if stats['detective_total_votes'] > 0:
            detective_accuracy = (stats['detective_votes_for_mafioso']/stats['detective_total_votes']*100)
            print(f"  Detective accuracy: {stats['detective_votes_for_mafioso']}/{stats['detective_total_votes']} "
                  f"({detective_accuracy:.1f}%)")
        else:
            print(f"  Detective accuracy: N/A (model never played as detective)")
        
        # Mafioso targeting (only if model played as mafioso)
        if stats['mafioso_total_votes'] > 0:
            mafioso_targeting = (stats['mafioso_votes_for_detective']/stats['mafioso_total_votes']*100)
            print(f"  Mafioso targeting: {stats['mafioso_votes_for_detective']}/{stats['mafioso_total_votes']} "
                  f"({mafioso_targeting:.1f}%)")
        else:
            print(f"  Mafioso targeting: N/A (model never played as mafioso)")

if __name__ == "__main__":
    analyze_model_performance()