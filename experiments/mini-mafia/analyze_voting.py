#!/usr/bin/env python3
"""
Analyze voting patterns between detectives and mafiosos
"""
import json
import os
import re
from collections import defaultdict

def analyze_voting_patterns():
    """Analyze voting patterns between detectives and mafiosos"""
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    voting_stats = {
        'detective_voted_mafioso': 0,
        'mafioso_voted_detective': 0,
        'villager_voted_mafioso': 0,
        'voting_ties': 0,
        'total_games_with_both_roles': 0,
        'by_batch': defaultdict(lambda: {
            'detective_voted_mafioso': 0,
            'mafioso_voted_detective': 0,
            'villager_voted_mafioso': 0,
            'voting_ties': 0,
            'total_games_with_both_roles': 0
        })
    }
    
    # Process all batch folders
    for batch_folder in os.listdir(data_dir):
        batch_path = os.path.join(data_dir, batch_folder)
        
        # Skip non-directories and system files
        if not os.path.isdir(batch_path) or batch_folder.startswith('.'):
            continue
        
        print(f"Processing batch folder: {batch_folder}")
        
        # Process all JSON game files in this batch folder
        for filename in os.listdir(batch_path):
            if not filename.endswith('.json') or filename.endswith('_summary.json') or filename == 'prompt_config.json':
                continue
                
            filepath = os.path.join(batch_path, filename)
            try:
                with open(filepath, 'r') as f:
                    game_data = json.load(f)
                
                batch_id = batch_folder
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
                
                
                voting_stats['total_games_with_both_roles'] += 1
                voting_stats['by_batch'][batch_id]['total_games_with_both_roles'] += 1
                
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
                    voting_stats['by_batch'][batch_id]['detective_voted_mafioso'] += 1
                
                if mafioso_voted_detective:
                    voting_stats['mafioso_voted_detective'] += 1
                    voting_stats['by_batch'][batch_id]['mafioso_voted_detective'] += 1
                
                if villager_voted_mafioso:
                    voting_stats['villager_voted_mafioso'] += 1
                    voting_stats['by_batch'][batch_id]['villager_voted_mafioso'] += 1
                
                if has_voting_tie:
                    voting_stats['voting_ties'] += 1
                    voting_stats['by_batch'][batch_id]['voting_ties'] += 1
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Print voting pattern results
    print("=" * 60)
    print("VOTING PATTERN ANALYSIS")
    print("=" * 60)
    
    total_games = voting_stats['total_games_with_both_roles']
    if total_games > 0:
        detective_voted_pct = (voting_stats['detective_voted_mafioso'] / total_games) * 100
        mafioso_voted_pct = (voting_stats['mafioso_voted_detective'] / total_games) * 100
        villager_voted_mafioso_pct = (voting_stats['villager_voted_mafioso'] / total_games) * 100
        ties_pct = (voting_stats['voting_ties'] / total_games) * 100
        
        print(f"\nOVERALL VOTING PATTERNS:")
        print(f"  Total games with both detective and mafioso: {total_games}")
        print(f"  Detective voted for mafioso: {voting_stats['detective_voted_mafioso']} ({detective_voted_pct:.1f}%)")
        print(f"  Mafioso voted for detective: {voting_stats['mafioso_voted_detective']} ({mafioso_voted_pct:.1f}%)")
        print(f"  Villager voted for mafioso: {voting_stats['villager_voted_mafioso']} ({villager_voted_mafioso_pct:.1f}%)")
        print(f"  Games with voting ties: {voting_stats['voting_ties']} ({ties_pct:.1f}%)")
        
        print(f"\nBY BATCH:")
        for batch_id, batch_stats in sorted(voting_stats['by_batch'].items()):
            batch_total = batch_stats['total_games_with_both_roles']
            if batch_total > 0:
                batch_detective_pct = (batch_stats['detective_voted_mafioso'] / batch_total) * 100
                batch_mafioso_pct = (batch_stats['mafioso_voted_detective'] / batch_total) * 100
                batch_villager_mafioso_pct = (batch_stats['villager_voted_mafioso'] / batch_total) * 100
                batch_ties_pct = (batch_stats['voting_ties'] / batch_total) * 100
                
                print(f"  {batch_id}:")
                print(f"    Games with both roles: {batch_total}")
                print(f"    Detective voted mafioso: {batch_stats['detective_voted_mafioso']} ({batch_detective_pct:.1f}%)")
                print(f"    Mafioso voted detective: {batch_stats['mafioso_voted_detective']} ({batch_mafioso_pct:.1f}%)")
                print(f"    Villager voted mafioso: {batch_stats['villager_voted_mafioso']} ({batch_villager_mafioso_pct:.1f}%)")
                print(f"    Games with ties: {batch_stats['voting_ties']} ({batch_ties_pct:.1f}%)")
    else:
        print("No games found with both detective and mafioso roles.")
    
    print("=" * 60)
    return voting_stats

def find_exceptional_games():
    """Find games where voting patterns deviate from the norm"""
    data_dir = "data"
    
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
            if not filename.endswith('.json') or filename.endswith('_summary.json') or filename == 'prompt_config.json':
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