#!/usr/bin/env python3
"""
Comprehensive analysis of voting patterns in mini-mafia games.
"""

import json
import os
import re
from collections import defaultdict, Counter

def analyze_game_voting(game_data):
    """Comprehensive analysis of a single game's voting pattern."""
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
    
    # Find the surviving villager
    surviving_villager = None
    surviving_villager_name = None
    for name, player in players.items():
        if player['role'] == 'villager' and player['alive'] and not player['imprisoned']:
            surviving_villager = player
            surviving_villager_name = name
            break
    
    if not surviving_villager:
        return {
            'game_id': game_id,
            'status': 'no_surviving_villager',
            'winner': game_data['winner']
        }
    
    # Find detective and mafioso
    detective_name = None
    mafioso_name = None
    for name, player in players.items():
        if player['role'] == 'detective':
            detective_name = name
        elif player['role'] == 'mafioso':
            mafioso_name = name
    
    # Look for voting information in ANY player's memory (they all should have it)
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
    
    # Parse all votes from the voting line
    # Format: "Day 1 votes: Alice voted for Charlie, Charlie voted for Diana, Diana voted for Alice"
    all_votes = {}
    vote_pattern = r'(\w+) voted for (\w+)'
    for match in re.finditer(vote_pattern, voting_line):
        voter = match.group(1)
        voted_for = match.group(2)
        all_votes[voter] = voted_for
    
    # Get villager's vote
    villager_voted_for = all_votes.get(surviving_villager_name)
    if not villager_voted_for:
        return {
            'game_id': game_id,
            'status': 'villager_vote_not_found',
            'voting_line': voting_line,
            'all_votes': all_votes,
            'villager_name': surviving_villager_name,
            'winner': game_data['winner']
        }
    
    # Determine what the villager voted for
    villager_vote_type = 'unknown'
    if villager_voted_for == detective_name:
        villager_vote_type = 'detective'
    elif villager_voted_for == mafioso_name:
        villager_vote_type = 'mafioso'
    
    # Check for ties
    vote_counts = Counter(all_votes.values())
    most_votes = max(vote_counts.values())
    tied_players = [player for player, votes in vote_counts.items() if votes == most_votes]
    is_tie = len(tied_players) > 1
    
    return {
        'game_id': game_id,
        'status': 'valid',
        'villager_name': surviving_villager_name,
        'villager_voted_for': villager_voted_for,
        'villager_vote_type': villager_vote_type,
        'detective_name': detective_name,
        'mafioso_name': mafioso_name,
        'all_votes': all_votes,
        'vote_counts': dict(vote_counts),
        'is_tie': is_tie,
        'tied_players': tied_players,
        'arrested_player': game_data.get('arrested_player', {}).get('name'),
        'winner': game_data['winner'],
        'voting_line': voting_line
    }

def main():
    batch_dir = "/Users/davicosta/Desktop/projects/llm-mafia-game/experiments/mini-mafia/data/batch_20250805_123454_v1.0"
    
    print("Comprehensive Mini-Mafia Voting Analysis")
    print("=" * 50)
    
    # Get all JSON game files (handle both naming conventions)
    json_files = [f for f in os.listdir(batch_dir) if f.endswith('.json') and 'game_' in f]
    json_files.sort()
    
    print(f"Found {len(json_files)} game files")
    
    all_results = []
    status_counts = Counter()
    
    for json_file in json_files:
        file_path = os.path.join(batch_dir, json_file)
        try:
            with open(file_path, 'r') as f:
                game_data = json.load(f)
            
            result = analyze_game_voting(game_data)
            all_results.append(result)
            status_counts[result['status']] += 1
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print(f"\nGame Analysis Status:")
    print("-" * 25)
    for status, count in status_counts.items():
        print(f"{status:25}: {count}")
    
    # Focus on valid games
    valid_games = [r for r in all_results if r['status'] == 'valid']
    print(f"\nAnalyzing {len(valid_games)} valid games")
    
    if not valid_games:
        print("No valid games found!")
        return
    
    # Basic voting distribution
    vote_type_counts = Counter([game['villager_vote_type'] for game in valid_games])
    
    print(f"\nVillager Vote Distribution:")
    print("-" * 30)
    total_valid = len(valid_games)
    for vote_type, count in vote_type_counts.most_common():
        percentage = (count / total_valid) * 100
        print(f"{vote_type:12}: {count:4} ({percentage:5.1f}%)")
    
    # Analyze ties
    tied_games = [g for g in valid_games if g['is_tie']]
    print(f"\nTie Analysis:")
    print("-" * 15)
    print(f"Games with ties: {len(tied_games)} ({len(tied_games)/len(valid_games)*100:.1f}%)")
    
    if tied_games:
        print("Sample tied games:")
        for game in tied_games[:5]:
            print(f"  {game['game_id'][-4:]}: Votes = {game['vote_counts']}, Arrested = {game['arrested_player']}")
    
    # Check detective vs mafioso specifically
    clear_votes = [g for g in valid_games if g['villager_vote_type'] in ['detective', 'mafioso']]
    if clear_votes:
        detective_votes = len([g for g in clear_votes if g['villager_vote_type'] == 'detective'])
        mafioso_votes = len([g for g in clear_votes if g['villager_vote_type'] == 'mafioso'])
        
        print(f"\nDetective vs Mafioso Votes:")
        print("-" * 30)
        print(f"Detective votes: {detective_votes} ({detective_votes/len(clear_votes)*100:.1f}%)")
        print(f"Mafioso votes  : {mafioso_votes} ({mafioso_votes/len(clear_votes)*100:.1f}%)")
        print(f"Expected (50/50): {len(clear_votes)/2:.1f} each")
        print(f"Bias toward detective: {(detective_votes/len(clear_votes) - 0.5)*100:+.1f}%")
    
    # Analyze relationship between voting and winning
    print(f"\nVoting vs Outcome Analysis:")
    print("-" * 30)
    
    outcomes = defaultdict(lambda: defaultdict(int))
    for game in valid_games:
        outcomes[game['villager_vote_type']][game['winner']] += 1
    
    for vote_type in ['detective', 'mafioso', 'unknown']:
        if vote_type in outcomes:
            good_wins = outcomes[vote_type]['good']
            evil_wins = outcomes[vote_type]['evil']
            total = good_wins + evil_wins
            if total > 0:
                print(f"When villager votes {vote_type:9}: Good wins {good_wins:3}/{total:3} ({good_wins/total*100:4.1f}%)")
    
    # Look for patterns in problematic cases
    unknown_votes = [g for g in valid_games if g['villager_vote_type'] == 'unknown']
    if unknown_votes:
        print(f"\nUnknown Vote Analysis ({len(unknown_votes)} games):")
        print("-" * 35)
        print("Sample cases where villager didn't vote for detective or mafioso:")
        for game in unknown_votes[:5]:
            print(f"  Game {game['game_id'][-4:]}: {game['villager_name']} voted for {game['villager_voted_for']}")
            print(f"    Detective: {game['detective_name']}, Mafioso: {game['mafioso_name']}")
            print(f"    All votes: {game['all_votes']}")
    
    # Check if your 40.3% figure might come from a different calculation
    print(f"\nAlternative Calculations:")
    print("-" * 25)
    
    # Include ties as a separate category
    detective_count = vote_type_counts['detective']
    mafioso_count = vote_type_counts['mafioso']  
    unknown_count = vote_type_counts['unknown']
    tie_count = len(tied_games)
    
    print(f"Including all outcomes:")
    print(f"  Detective: {detective_count} ({detective_count/total_valid*100:.1f}%)")
    print(f"  Mafioso  : {mafioso_count} ({mafioso_count/total_valid*100:.1f}%)")
    print(f"  Unknown  : {unknown_count} ({unknown_count/total_valid*100:.1f}%)")
    print(f"  Ties     : {tie_count} ({tie_count/total_valid*100:.1f}%)")
    
    # Show some actual vote patterns
    print(f"\nSample Voting Patterns:")
    print("-" * 25)
    
    sample_detective = [g for g in valid_games if g['villager_vote_type'] == 'detective'][:3]
    sample_mafioso = [g for g in valid_games if g['villager_vote_type'] == 'mafioso'][:3]
    
    print("Villager voted for DETECTIVE:")
    for game in sample_detective:
        print(f"  Game {game['game_id'][-4:]}: {game['voting_line']}")
    
    print("Villager voted for MAFIOSO:")
    for game in sample_mafioso:
        print(f"  Game {game['game_id'][-4:]}: {game['voting_line']}")

if __name__ == "__main__":
    main()