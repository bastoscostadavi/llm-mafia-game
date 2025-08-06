#!/usr/bin/env python3
"""
Deep analysis of villager voting patterns across all available batches.
"""

import json
import os
import re
from collections import defaultdict, Counter

def analyze_game_voting(game_data):
    """Analyze voting patterns and memory content for a single game."""
    game_id = game_data['game_id']
    
    # Find all players and their roles
    players = {}
    for player in game_data['final_state']:
        players[player['name']] = {
            'role': player['role'],
            'alive': player['alive'],
            'imprisoned': player['imprisoned'],
            'memory': player['memory']
        }
    
    # Find the surviving villager (alive and not imprisoned)
    surviving_villagers = []
    for name, player in players.items():
        if player['role'] == 'villager' and player['alive'] and not player['imprisoned']:
            surviving_villagers.append((name, player))
    
    if len(surviving_villagers) != 1:
        return {
            'game_id': game_id,
            'status': 'invalid_villager_count',
            'surviving_villagers': len(surviving_villagers),
            'winner': game_data['winner']
        }
    
    villager_name, villager_data = surviving_villagers[0]
    
    # Find detective and mafioso
    detective_name = mafioso_name = None
    for name, player in players.items():
        if player['role'] == 'detective':
            detective_name = name
        elif player['role'] == 'mafioso':
            mafioso_name = name
    
    # Find voting information in any player's memory
    voting_line = None
    for name, player in players.items():
        for memory_item in player['memory']:
            if 'Day 1 votes:' in memory_item:
                voting_line = memory_item
                break
        if voting_line:
            break
    
    if not voting_line:
        return {
            'game_id': game_id,
            'status': 'no_voting_data',
            'winner': game_data['winner']
        }
    
    # Parse votes
    all_votes = {}
    vote_pattern = r'(\w+) voted for (\w+)'
    for match in re.finditer(vote_pattern, voting_line):
        voter = match.group(1)
        voted_for = match.group(2)
        all_votes[voter] = voted_for
    
    villager_voted_for = all_votes.get(villager_name)
    if not villager_voted_for:
        return {
            'game_id': game_id,
            'status': 'villager_vote_missing',
            'voting_line': voting_line,
            'villager_name': villager_name,
            'winner': game_data['winner']
        }
    
    # Determine vote type
    villager_vote_type = 'unknown'
    if villager_voted_for == detective_name:
        villager_vote_type = 'detective'
    elif villager_voted_for == mafioso_name:
        villager_vote_type = 'mafioso'
    
    # Analyze villager's memory for strategic reasoning
    memory_analysis = analyze_villager_memory(villager_data['memory'], detective_name, mafioso_name)
    
    return {
        'game_id': game_id,
        'status': 'valid',
        'villager_name': villager_name,
        'villager_voted_for': villager_voted_for,
        'villager_vote_type': villager_vote_type,
        'detective_name': detective_name,
        'mafioso_name': mafioso_name,
        'all_votes': all_votes,
        'winner': game_data['winner'],
        'voting_line': voting_line,
        'memory_analysis': memory_analysis
    }

def analyze_villager_memory(memory, detective_name, mafioso_name):
    """Analyze villager's memory for strategic patterns."""
    analysis = {
        'total_memories': len(memory),
        'mentions_detective': 0,
        'mentions_mafioso': 0,
        'accusations_received': 0,
        'accusations_made': 0,
        'strategic_keywords': 0,
        'sample_memories': memory[-3:] if len(memory) >= 3 else memory  # Last 3 memories
    }
    
    strategic_keywords = ['suspicious', 'evidence', 'guilty', 'innocent', 'mafioso', 'detective', 'investigate', 'accuse']
    
    for mem in memory:
        if detective_name and detective_name in mem:
            analysis['mentions_detective'] += 1
        if mafioso_name and mafioso_name in mem:
            analysis['mentions_mafioso'] += 1
        if 'I suspect' in mem or 'I think' in mem or "Let's vote" in mem:
            analysis['accusations_made'] += 1
        if any(keyword in mem.lower() for keyword in strategic_keywords):
            analysis['strategic_keywords'] += 1
    
    return analysis

def analyze_batch_directory(batch_dir, batch_name):
    """Analyze all games in a batch directory."""
    print(f"\nAnalyzing batch: {batch_name}")
    print("-" * 40)
    
    # Get all game files
    json_files = [f for f in os.listdir(batch_dir) if f.endswith('.json') and 'game_' in f]
    json_files.sort()
    
    print(f"Found {len(json_files)} game files")
    
    results = []
    status_counts = Counter()
    
    for json_file in json_files:
        file_path = os.path.join(batch_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                game_data = json.load(f)
            
            result = analyze_game_voting(game_data)
            results.append(result)
            status_counts[result['status']] += 1
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Print status summary
    print(f"Game processing status:")
    for status, count in status_counts.items():
        print(f"  {status:20}: {count}")
    
    # Analyze valid games
    valid_games = [r for r in results if r['status'] == 'valid']
    
    if not valid_games:
        print("No valid games found in this batch!")
        return results
    
    print(f"\nVoting Analysis ({len(valid_games)} valid games):")
    vote_counts = Counter([game['villager_vote_type'] for game in valid_games])
    
    for vote_type, count in vote_counts.most_common():
        percentage = (count / len(valid_games)) * 100
        print(f"  {vote_type:12}: {count:4} ({percentage:5.1f}%)")
    
    # Detective vs Mafioso specific analysis
    clear_votes = [g for g in valid_games if g['villager_vote_type'] in ['detective', 'mafioso']]
    if clear_votes:
        detective_votes = len([g for g in clear_votes if g['villager_vote_type'] == 'detective'])
        mafioso_votes = len([g for g in clear_votes if g['villager_vote_type'] == 'mafioso'])
        
        print(f"\nDetective vs Mafioso votes:")
        print(f"  Detective: {detective_votes} ({detective_votes/len(clear_votes)*100:.1f}%)")
        print(f"  Mafioso  : {mafioso_votes} ({mafioso_votes/len(clear_votes)*100:.1f}%)")
        print(f"  Bias toward detective: {(detective_votes/len(clear_votes) - 0.5)*100:+.1f}%")
    
    # Memory analysis
    print(f"\nMemory Pattern Analysis:")
    detective_voters = [g for g in valid_games if g['villager_vote_type'] == 'detective']
    mafioso_voters = [g for g in valid_games if g['villager_vote_type'] == 'mafioso']
    
    if detective_voters:
        avg_detective_mentions = sum(g['memory_analysis']['mentions_detective'] for g in detective_voters) / len(detective_voters)
        avg_mafioso_mentions = sum(g['memory_analysis']['mentions_mafioso'] for g in detective_voters) / len(detective_voters)
        print(f"  When voting for detective (avg mentions): Detective={avg_detective_mentions:.1f}, Mafioso={avg_mafioso_mentions:.1f}")
    
    if mafioso_voters:
        avg_detective_mentions = sum(g['memory_analysis']['mentions_detective'] for g in mafioso_voters) / len(mafioso_voters)
        avg_mafioso_mentions = sum(g['memory_analysis']['mentions_mafioso'] for g in mafioso_voters) / len(mafioso_voters)
        print(f"  When voting for mafioso (avg mentions): Detective={avg_detective_mentions:.1f}, Mafioso={avg_mafioso_mentions:.1f}")
    
    return results

def main():
    base_dir = "/Users/davicosta/Desktop/projects/llm-mafia-game/experiments/mini-mafia/data"
    
    print("COMPREHENSIVE VILLAGER VOTING ANALYSIS")
    print("=" * 50)
    
    # Find all batch directories
    batch_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('batch_'):
            batch_dirs.append((item_path, item))
    
    batch_dirs.sort(key=lambda x: x[1])  # Sort by batch name
    
    all_results = {}
    
    for batch_dir, batch_name in batch_dirs:
        results = analyze_batch_directory(batch_dir, batch_name)
        all_results[batch_name] = results
    
    # Compare across batches
    print(f"\n" + "="*50)
    print("CROSS-BATCH COMPARISON")
    print("="*50)
    
    for batch_name, results in all_results.items():
        valid_games = [r for r in results if r['status'] == 'valid']
        if not valid_games:
            continue
        
        vote_counts = Counter([game['villager_vote_type'] for game in valid_games])
        clear_votes = [g for g in valid_games if g['villager_vote_type'] in ['detective', 'mafioso']]
        
        if clear_votes:
            mafioso_votes = len([g for g in clear_votes if g['villager_vote_type'] == 'mafioso'])
            mafioso_pct = mafioso_votes / len(clear_votes) * 100
            
            print(f"{batch_name:30}: {len(valid_games):3} games, mafioso votes: {mafioso_votes:3}/{len(clear_votes):3} ({mafioso_pct:5.1f}%)")
    
    # Look for the 40.3% figure specifically
    print(f"\nLooking for 40.3% pattern...")
    for batch_name, results in all_results.items():
        valid_games = [r for r in results if r['status'] == 'valid']
        if not valid_games:
            continue
        
        clear_votes = [g for g in valid_games if g['villager_vote_type'] in ['detective', 'mafioso']]
        if clear_votes:
            mafioso_votes = len([g for g in clear_votes if g['villager_vote_type'] == 'mafioso'])
            mafioso_pct = mafioso_votes / len(clear_votes) * 100
            
            if abs(mafioso_pct - 40.3) < 2.0:  # Within 2% of 40.3%
                print(f"  MATCH: {batch_name} has {mafioso_pct:.1f}% mafioso votes (close to 40.3%)")
    
    # Show sample strategic reasoning
    print(f"\nSAMPLE STRATEGIC REASONING:")
    print("-" * 30)
    
    for batch_name, results in all_results.items():
        valid_games = [r for r in results if r['status'] == 'valid']
        detective_voters = [g for g in valid_games if g['villager_vote_type'] == 'detective'][:2]
        mafioso_voters = [g for g in valid_games if g['villager_vote_type'] == 'mafioso'][:2]
        
        if detective_voters:
            print(f"\n{batch_name} - Villagers voting for DETECTIVE:")
            for game in detective_voters:
                print(f"  Game {game['game_id'][-4:]}: {game['villager_name']} -> {game['villager_voted_for']}")
                for mem in game['memory_analysis']['sample_memories']:
                    if len(mem) < 150:
                        print(f"    Memory: {mem}")
        
        if mafioso_voters:
            print(f"\n{batch_name} - Villagers voting for MAFIOSO:")
            for game in mafioso_voters:
                print(f"  Game {game['game_id'][-4:]}: {game['villager_name']} -> {game['villager_voted_for']}")
                for mem in game['memory_analysis']['sample_memories']:
                    if len(mem) < 150:
                        print(f"    Memory: {mem}")

if __name__ == "__main__":
    main()