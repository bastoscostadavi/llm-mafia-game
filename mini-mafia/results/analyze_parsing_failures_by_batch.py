#!/usr/bin/env python3
"""
Parsing Failure Analysis by Batch

Analyzes voting format failures and discussion silence for each model 
across all v4.0 and v4.1 batches, showing the percentage of parsing failures.
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add analysis functions from analyze_results
sys.path.append('../analysis')
from analyze_results import extract_model_name, get_batch_config

def analyze_batch_parsing_failures(batch_path):
    """Analyze parsing failures for a single batch"""
    batch_name = os.path.basename(batch_path)
    
    # Get batch configuration
    config = get_batch_config(batch_path)
    if not config:
        return None
    
    model_configs = config.get('model_configs', {})
    
    # Track failures by model and role
    stats = defaultdict(lambda: {
        'total_votes': 0,
        'random_votes': 0,
        'total_discussions': 0,
        'silent_discussions': 0,
        'detective_votes_for_mafioso': 0,
        'detective_total_votes': 0,
        'mafioso_votes_for_detective': 0,
        'mafioso_total_votes': 0,
        'villager_votes_for_mafioso': 0,
        'villager_total_votes': 0
    })
    
    # Process all game files in batch
    game_files = [f for f in os.listdir(batch_path) 
                  if f.startswith('game_') and f.endswith('.json')]
    
    for game_file in game_files:
        game_path = os.path.join(batch_path, game_file)
        try:
            with open(game_path, 'r') as f:
                game_data = json.load(f)
            
            players = game_data.get('players', [])
            
            # Find player names by role for voting accuracy tracking
            mafioso_name = None
            detective_name = None
            villager_names = []
            for player in players:
                if player.get('role') == 'mafioso':
                    mafioso_name = player.get('name')
                elif player.get('role') == 'detective':
                    detective_name = player.get('name')
                elif player.get('role') == 'villager':
                    villager_names.append(player.get('name'))
            
            # First, extract all unique voting information from any surviving player's memory
            vote_entries = set()  # Use set to avoid duplicates
            for player in players:
                memory = player.get('memory', [])
                for entry in memory:
                    if entry.startswith('Votes:'):
                        vote_entries.add(entry)  # Set automatically deduplicates
            
            # Process each unique voting round
            for vote_entry in vote_entries:
                # Parse individual votes from the summary
                vote_parts = vote_entry[7:].split(', ')  # Remove "Votes: " prefix
                
                for vote_part in vote_parts:
                    # Extract voter name and target
                    vote_part = vote_part.strip()
                    if " voted for " in vote_part:
                        voter_name = vote_part.split(" voted for ")[0]
                        target_and_rest = vote_part.split(" voted for ")[1]
                        
                        # Find the voter in players list to get their role and model
                        voter_player = None
                        for player in players:
                            if player.get('name') == voter_name:
                                voter_player = player
                                break
                        
                        if voter_player:
                            voter_role = voter_player.get('role', 'unknown')
                            model_config = model_configs.get(voter_role, {})
                            model_name = extract_model_name(model_config)
                            model_key = f"{model_name}_{voter_role}"
                            
                            stats[model_key]['total_votes'] += 1
                            
                            # Check for random vote due to format failure
                            if "(the vote was cast randomly because of a failed format)" in vote_part:
                                stats[model_key]['random_votes'] += 1
                            
                            # Track role-specific strategic voting
                            if voter_role == 'detective' and mafioso_name:
                                stats[model_key]['detective_total_votes'] += 1
                                if f"voted for {mafioso_name}" in vote_part:
                                    stats[model_key]['detective_votes_for_mafioso'] += 1
                            
                            elif voter_role == 'mafioso' and detective_name:
                                stats[model_key]['mafioso_total_votes'] += 1
                                if f"voted for {detective_name}" in vote_part:
                                    stats[model_key]['mafioso_votes_for_detective'] += 1
                            
                            elif voter_role == 'villager' and mafioso_name:
                                stats[model_key]['villager_total_votes'] += 1
                                if f"voted for {mafioso_name}" in vote_part:
                                    stats[model_key]['villager_votes_for_mafioso'] += 1
            
            # Analyze each player's memory for discussion messages
            for player in players:
                role = player.get('role', 'unknown')
                memory = player.get('memory', [])
                
                # Get model name for this role
                model_config = model_configs.get(role, {})
                model_name = extract_model_name(model_config)
                model_key = f"{model_name}_{role}"
                
                # Count discussion messages
                for entry in memory:
                    if entry.startswith("You:"):
                        stats[model_key]['total_discussions'] += 1
                        
                        # Check if remained silent
                        if entry.strip() == "You: remained silent.":
                            stats[model_key]['silent_discussions'] += 1
                        
        except (json.JSONDecodeError, IOError, KeyError) as e:
            continue
    
    # Calculate percentages and return results
    results = {}
    for model_key, data in stats.items():
        total_votes = data['total_votes']
        random_votes = data['random_votes']
        total_discussions = data['total_discussions']
        silent_discussions = data['silent_discussions']
        
        vote_failure_rate = (random_votes / total_votes * 100) if total_votes > 0 else 0
        silence_rate = (silent_discussions / total_discussions * 100) if total_discussions > 0 else 0
        detective_accuracy = (data['detective_votes_for_mafioso'] / data['detective_total_votes'] * 100) if data['detective_total_votes'] > 0 else 0
        mafioso_accuracy = (data['mafioso_votes_for_detective'] / data['mafioso_total_votes'] * 100) if data['mafioso_total_votes'] > 0 else 0
        villager_accuracy = (data['villager_votes_for_mafioso'] / data['villager_total_votes'] * 100) if data['villager_total_votes'] > 0 else 0
        
        results[model_key] = {
            'total_votes': total_votes,
            'random_votes': random_votes,
            'vote_failure_rate': vote_failure_rate,
            'total_discussions': total_discussions,
            'silent_discussions': silent_discussions,
            'silence_rate': silence_rate,
            'detective_votes_for_mafioso': data['detective_votes_for_mafioso'],
            'detective_total_votes': data['detective_total_votes'],
            'detective_accuracy': detective_accuracy,
            'mafioso_votes_for_detective': data['mafioso_votes_for_detective'],
            'mafioso_total_votes': data['mafioso_total_votes'],
            'mafioso_accuracy': mafioso_accuracy,
            'villager_votes_for_mafioso': data['villager_votes_for_mafioso'],
            'villager_total_votes': data['villager_total_votes'],
            'villager_accuracy': villager_accuracy,
            'model_configs': model_configs
        }
    
    return {
        'batch_name': batch_name,
        'results': results
    }

def main():
    """Analyze parsing failures across all v4.0 and v4.1 batches"""
    
    print("📊 PARSING FAILURE ANALYSIS BY BATCH")
    print("=" * 80)
    
    data_dir = "../data/batch"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found. Run from mini-mafia/results/ directory.")
        return
    
    # Find all v4.0 and v4.1 batch directories
    v4_batches = []
    for batch_dir in os.listdir(data_dir):
        if batch_dir.endswith("_v4.0") or batch_dir.endswith("_v4.1"):
            batch_path = os.path.join(data_dir, batch_dir)
            if os.path.isdir(batch_path):
                v4_batches.append(batch_path)
    
    if not v4_batches:
        print("No v4.0 or v4.1 batch directories found.")
        return
    
    # Analyze each batch
    all_batch_results = []
    for batch_path in sorted(v4_batches):
        print(f"Processing {os.path.basename(batch_path)}...")
        result = analyze_batch_parsing_failures(batch_path)
        if result:
            all_batch_results.append(result)
    
    # Print results for each batch
    print(f"\n🎯 RESULTS BY BATCH")
    print("=" * 80)
    
    for batch_result in all_batch_results:
        batch_name = batch_result['batch_name']
        results = batch_result['results']
        
        print(f"\n📁 {batch_name}")
        print("-" * 60)
        
        # Show model configuration for this batch
        if results:
            model_configs = list(results.values())[0]['model_configs']
            print("Model Configuration:")
            for role, model_config in model_configs.items():
                model_name = extract_model_name(model_config)
                print(f"  {role.capitalize()}: {model_name}")
            print()
        
        # Show results for each model in this batch
        for model_key, data in sorted(results.items()):
            model_name, role = model_key.split('_', 1)
            
            print(f"{model_name} ({role}):")
            
            # Voting failures
            if data['total_votes'] > 0:
                print(f"  Voting: {data['random_votes']}/{data['total_votes']} failures "
                      f"({data['vote_failure_rate']:.1f}%)")
            else:
                print(f"  Voting: No votes recorded")
            
            # Discussion silence
            if data['total_discussions'] > 0:
                print(f"  Discussion: {data['silent_discussions']}/{data['total_discussions']} silent "
                      f"({data['silence_rate']:.1f}%)")
            else:
                print(f"  Discussion: No discussions recorded")
            
            # Role-specific strategic voting accuracy
            if role == 'detective' and data['detective_total_votes'] > 0:
                print(f"  Detective accuracy: {data['detective_votes_for_mafioso']}/{data['detective_total_votes']} "
                      f"voted for mafioso ({data['detective_accuracy']:.1f}%)")
            
            elif role == 'mafioso' and data['mafioso_total_votes'] > 0:
                print(f"  Mafioso accuracy: {data['mafioso_votes_for_detective']}/{data['mafioso_total_votes']} "
                      f"voted for detective ({data['mafioso_accuracy']:.1f}%)")
            
            elif role == 'villager' and data['villager_total_votes'] > 0:
                print(f"  Villager accuracy: {data['villager_votes_for_mafioso']}/{data['villager_total_votes']} "
                      f"voted for mafioso ({data['villager_accuracy']:.1f}%)")
            
            print()
    
    # Summary statistics
    print(f"\n📈 SUMMARY ACROSS ALL BATCHES")
    print("=" * 80)
    
    # Aggregate by model type and role
    model_aggregates = defaultdict(lambda: {
        'total_votes': 0, 'random_votes': 0,
        'total_discussions': 0, 'silent_discussions': 0,
        'detective_votes_for_mafioso': 0, 'detective_total_votes': 0,
        'mafioso_votes_for_detective': 0, 'mafioso_total_votes': 0,
        'villager_votes_for_mafioso': 0, 'villager_total_votes': 0
    })
    
    # Also aggregate by model+role for detailed breakdown
    model_role_aggregates = defaultdict(lambda: {
        'total_votes': 0, 'random_votes': 0,
        'total_discussions': 0, 'silent_discussions': 0,
        'detective_votes_for_mafioso': 0, 'detective_total_votes': 0,
        'mafioso_votes_for_detective': 0, 'mafioso_total_votes': 0,
        'villager_votes_for_mafioso': 0, 'villager_total_votes': 0
    })
    
    for batch_result in all_batch_results:
        for model_key, data in batch_result['results'].items():
            model_name = model_key.split('_')[0]
            role = '_'.join(model_key.split('_')[1:])
            
            # Aggregate by model type
            model_aggregates[model_name]['total_votes'] += data['total_votes']
            model_aggregates[model_name]['random_votes'] += data['random_votes']
            model_aggregates[model_name]['total_discussions'] += data['total_discussions']
            model_aggregates[model_name]['silent_discussions'] += data['silent_discussions']
            model_aggregates[model_name]['detective_votes_for_mafioso'] += data['detective_votes_for_mafioso']
            model_aggregates[model_name]['detective_total_votes'] += data['detective_total_votes']
            model_aggregates[model_name]['mafioso_votes_for_detective'] += data['mafioso_votes_for_detective']
            model_aggregates[model_name]['mafioso_total_votes'] += data['mafioso_total_votes']
            model_aggregates[model_name]['villager_votes_for_mafioso'] += data['villager_votes_for_mafioso']
            model_aggregates[model_name]['villager_total_votes'] += data['villager_total_votes']
            
            # Aggregate by model+role for detailed breakdown
            model_role_key = f"{model_name}_{role}"
            model_role_aggregates[model_role_key]['total_votes'] += data['total_votes']
            model_role_aggregates[model_role_key]['random_votes'] += data['random_votes']
            model_role_aggregates[model_role_key]['total_discussions'] += data['total_discussions']
            model_role_aggregates[model_role_key]['silent_discussions'] += data['silent_discussions']
            model_role_aggregates[model_role_key]['detective_votes_for_mafioso'] += data['detective_votes_for_mafioso']
            model_role_aggregates[model_role_key]['detective_total_votes'] += data['detective_total_votes']
            model_role_aggregates[model_role_key]['mafioso_votes_for_detective'] += data['mafioso_votes_for_detective']
            model_role_aggregates[model_role_key]['mafioso_total_votes'] += data['mafioso_total_votes']
            model_role_aggregates[model_role_key]['villager_votes_for_mafioso'] += data['villager_votes_for_mafioso']
            model_role_aggregates[model_role_key]['villager_total_votes'] += data['villager_total_votes']
    
    for model_name in sorted(model_aggregates.keys()):
        data = model_aggregates[model_name]
        
        print(f"\n{model_name} (All Batches):")
        
        # Overall stats
        if data['total_votes'] > 0:
            vote_failure_rate = (data['random_votes'] / data['total_votes']) * 100
            print(f"  Overall voting failures: {data['random_votes']}/{data['total_votes']} ({vote_failure_rate:.1f}%)")
        
        # Role-specific voting failures
        roles_for_model = ['detective', 'mafioso', 'villager']
        for role in roles_for_model:
            role_key = f"{model_name}_{role}"
            if role_key in model_role_aggregates:
                role_data = model_role_aggregates[role_key]
                if role_data['total_votes'] > 0:
                    role_failure_rate = (role_data['random_votes'] / role_data['total_votes']) * 100
                    print(f"    {role.capitalize()}: {role_data['random_votes']}/{role_data['total_votes']} ({role_failure_rate:.1f}%)")
        
        if data['total_discussions'] > 0:
            silence_rate = (data['silent_discussions'] / data['total_discussions']) * 100
            print(f"  Discussion silence: {data['silent_discussions']}/{data['total_discussions']} ({silence_rate:.1f}%)")
        
        # Strategic voting accuracies
        if data['detective_total_votes'] > 0:
            detective_accuracy = (data['detective_votes_for_mafioso'] / data['detective_total_votes']) * 100
            print(f"  Detective accuracy: {data['detective_votes_for_mafioso']}/{data['detective_total_votes']} ({detective_accuracy:.1f}%)")
        
        if data['mafioso_total_votes'] > 0:
            mafioso_accuracy = (data['mafioso_votes_for_detective'] / data['mafioso_total_votes']) * 100
            print(f"  Mafioso accuracy: {data['mafioso_votes_for_detective']}/{data['mafioso_total_votes']} ({mafioso_accuracy:.1f}%)")
        
        if data['villager_total_votes'] > 0:
            villager_accuracy = (data['villager_votes_for_mafioso'] / data['villager_total_votes']) * 100
            print(f"  Villager accuracy: {data['villager_votes_for_mafioso']}/{data['villager_total_votes']} ({villager_accuracy:.1f}%)")

if __name__ == "__main__":
    main()