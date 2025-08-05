#!/usr/bin/env python3
"""
Analyze all game results and count evil vs good victories
"""
import json
import os
from collections import defaultdict

def analyze_games():
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    results = {
        'good': 0,
        'evil': 0,
        'total': 0,
        'by_batch': defaultdict(lambda: {'good': 0, 'evil': 0, 'total': 0})
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
                
                winner = game_data.get('winner')
                batch_id = game_data.get('batch_id', batch_folder)
                
                if winner in ['good', 'evil']:
                    results[winner] += 1
                    results['total'] += 1
                    results['by_batch'][batch_id][winner] += 1
                    results['by_batch'][batch_id]['total'] += 1
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Print results
    print("=" * 60)
    print("MAFIA GAME RESULTS ANALYSIS")
    print("=" * 60)
    
    print(f"\nOVERALL RESULTS:")
    print(f"  Total games: {results['total']}")
    
    # Calculate uncertainties (sqrt(n))
    good_uncertainty = (results['good']**0.5) if results['good'] > 0 else 0
    evil_uncertainty = (results['evil']**0.5) if results['evil'] > 0 else 0
    
    # Calculate win rate uncertainties
    good_rate = results['good']/results['total']*100
    evil_rate = results['evil']/results['total']*100
    good_rate_uncertainty = good_uncertainty/results['total']*100
    evil_rate_uncertainty = evil_uncertainty/results['total']*100
    
    print(f"  Good victories: {results['good']} ± {good_uncertainty:.0f} ({good_rate:.1f}% ± {good_rate_uncertainty:.1f}%)")
    print(f"  Evil victories: {results['evil']} ± {evil_uncertainty:.0f} ({evil_rate:.1f}% ± {evil_rate_uncertainty:.1f}%)")
    
    print(f"\nBY BATCH:")
    for batch_id, batch_results in sorted(results['by_batch'].items()):
        print(f"  {batch_id}:")
        print(f"    Total: {batch_results['total']}")
        
        # Calculate batch uncertainties
        batch_good_uncertainty = (batch_results['good']**0.5) if batch_results['good'] > 0 else 0
        batch_evil_uncertainty = (batch_results['evil']**0.5) if batch_results['evil'] > 0 else 0
        
        # Calculate batch win rate uncertainties
        batch_good_rate = batch_results['good']/batch_results['total']*100
        batch_evil_rate = batch_results['evil']/batch_results['total']*100
        batch_good_rate_uncertainty = batch_good_uncertainty/batch_results['total']*100
        batch_evil_rate_uncertainty = batch_evil_uncertainty/batch_results['total']*100
        
        print(f"    Good: {batch_results['good']} ± {batch_good_uncertainty:.0f} ({batch_good_rate:.1f}% ± {batch_good_rate_uncertainty:.1f}%)")
        print(f"    Evil: {batch_results['evil']} ± {batch_evil_uncertainty:.0f} ({batch_evil_rate:.1f}% ± {batch_evil_rate_uncertainty:.1f}%)")
    
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    analyze_games()