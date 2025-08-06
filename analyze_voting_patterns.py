#!/usr/bin/env python3
"""
Analyze voting patterns in mini-mafia games to understand villager voting bias.
"""

import json
import os
import re
from collections import defaultdict, Counter

def extract_voting_info(game_data):
    """Extract voting information from a game."""
    # Find the surviving villager (not dead, not imprisoned)
    surviving_villager = None
    detective = None
    mafioso = None
    
    for player in game_data['final_state']:
        if player['role'] == 'villager' and player['alive'] and not player['imprisoned']:
            surviving_villager = player
        elif player['role'] == 'detective':
            detective = player
        elif player['role'] == 'mafioso':
            mafioso = player
    
    if not surviving_villager:
        return None
    
    # Look for voting information in the memory
    voting_pattern = None
    for memory_item in surviving_villager['memory']:
        if 'votes:' in memory_item and 'Day 1' in memory_item:
            voting_pattern = memory_item
            break
    
    if not voting_pattern:
        return None
    
    # Parse the voting pattern
    # Format: "Day 1 votes: Alice voted for Charlie, Charlie voted for Diana, Diana voted for Alice"
    villager_vote = None
    villager_name = surviving_villager['name']
    
    # Use regex to find who the villager voted for
    vote_match = re.search(f"{villager_name} voted for (\\w+)", voting_pattern)
    if vote_match:
        voted_for = vote_match.group(1)
        
        # Determine if they voted for detective or mafioso
        if detective and voted_for == detective['name']:
            villager_vote = 'detective'
        elif mafioso and voted_for == mafioso['name']:
            villager_vote = 'mafioso'
        else:
            villager_vote = 'unknown'
    
    return {
        'game_id': game_data['game_id'],
        'villager_name': villager_name,
        'villager_vote': villager_vote,
        'voted_for_name': vote_match.group(1) if vote_match else None,
        'detective_name': detective['name'] if detective else None,
        'mafioso_name': mafioso['name'] if mafioso else None,
        'voting_pattern': voting_pattern,
        'winner': game_data['winner']
    }

def analyze_batch(batch_dir):
    """Analyze all games in a batch directory."""
    voting_results = []
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(batch_dir) if f.endswith('.json')]
    json_files.sort()
    
    print(f"Analyzing {len(json_files)} games in {batch_dir}")
    
    for json_file in json_files:
        file_path = os.path.join(batch_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                game_data = json.load(f)
            
            voting_info = extract_voting_info(game_data)
            if voting_info:
                voting_results.append(voting_info)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    return voting_results

def main():
    batch_dir = "/Users/davicosta/Desktop/projects/llm-mafia-game/experiments/mini-mafia/data/batch_20250804_175125_v1.0"
    
    print("Analyzing mini-mafia voting patterns...")
    print("="*50)
    
    voting_results = analyze_batch(batch_dir)
    
    if not voting_results:
        print("No valid voting data found!")
        return
    
    print(f"\nAnalyzed {len(voting_results)} games with valid voting data")
    
    # Count voting patterns
    vote_counts = Counter([result['villager_vote'] for result in voting_results])
    
    print("\nVillager Voting Distribution:")
    print("-" * 30)
    total_votes = sum(vote_counts.values())
    
    for vote_type, count in vote_counts.most_common():
        percentage = (count / total_votes) * 100
        print(f"{vote_type:12}: {count:4} ({percentage:5.1f}%)")
    
    # Calculate expected vs actual
    if 'detective' in vote_counts and 'mafioso' in vote_counts:
        detective_votes = vote_counts['detective']
        mafioso_votes = vote_counts['mafioso']
        total_clear_votes = detective_votes + mafioso_votes
        
        if total_clear_votes > 0:
            mafioso_percentage = (mafioso_votes / total_clear_votes) * 100
            detective_percentage = (detective_votes / total_clear_votes) * 100
            
            print(f"\nClear Votes Only (Detective vs Mafioso):")
            print("-" * 40)
            print(f"Voted for Detective: {detective_votes:4} ({detective_percentage:5.1f}%)")
            print(f"Voted for Mafioso  : {mafioso_votes:4} ({mafioso_percentage:5.1f}%)")
            print(f"Expected (random)  : 50.0% each")
            print(f"Bias toward detective: {detective_percentage - 50.0:+5.1f}%")
    
    # Look at some examples of each type
    print(f"\nSample Games:")
    print("-" * 20)
    
    for vote_type in ['detective', 'mafioso', 'unknown']:
        examples = [r for r in voting_results if r['villager_vote'] == vote_type][:3]
        if examples:
            print(f"\n{vote_type.upper()} votes:")
            for ex in examples:
                print(f"  Game {ex['game_id'][-4:]}: {ex['villager_name']} voted for {ex['voted_for_name']}")
                print(f"    Detective: {ex['detective_name']}, Mafioso: {ex['mafioso_name']}")
    
    # Look for patterns in the voting text
    print(f"\nMemory Analysis:")
    print("-" * 20)
    
    # Analyze memories for strategic reasoning
    detective_voters = [r for r in voting_results if r['villager_vote'] == 'detective']
    mafioso_voters = [r for r in voting_results if r['villager_vote'] == 'mafioso']
    
    print(f"\nSample memories from villagers who voted for DETECTIVE:")
    for i, result in enumerate(detective_voters[:3]):
        print(f"\nGame {result['game_id'][-4:]}:")
        # Find the game file and read the villager's full memory
        game_file = os.path.join(batch_dir, f"batch_20250804_175125_game_{result['game_id'][-4:]}.json")
        try:
            with open(game_file, 'r') as f:
                game_data = json.load(f)
            
            for player in game_data['final_state']:
                if player['name'] == result['villager_name']:
                    print(f"  Memory snippets:")
                    for mem in player['memory'][-5:]:  # Last 5 memory items
                        if len(mem) < 200:  # Skip very long memories
                            print(f"    - {mem}")
        except:
            pass
    
    print(f"\nSample memories from villagers who voted for MAFIOSO:")
    for i, result in enumerate(mafioso_voters[:3]):
        print(f"\nGame {result['game_id'][-4:]}:")
        game_file = os.path.join(batch_dir, f"batch_20250804_175125_game_{result['game_id'][-4:]}.json")
        try:
            with open(game_file, 'r') as f:
                game_data = json.load(f)
            
            for player in game_data['final_state']:
                if player['name'] == result['villager_name']:
                    print(f"  Memory snippets:")
                    for mem in player['memory'][-5:]:  # Last 5 memory items
                        if len(mem) < 200:
                            print(f"    - {mem}")
        except:
            pass

if __name__ == "__main__":
    main()