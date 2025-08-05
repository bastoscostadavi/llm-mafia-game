#!/usr/bin/env python3
"""
Update existing batch summaries and game files with model information.
All existing batches used Mistral for all roles.
"""

import json
import os
from pathlib import Path

def get_mistral_model_configs():
    """Model configs used in existing batches (all Mistral)"""
    return {
        'detective': {'type': 'local', 'model_path': 'models/mistral.gguf'},
        'mafioso': {'type': 'local', 'model_path': 'models/mistral.gguf'},
        'villager': {'type': 'local', 'model_path': 'models/mistral.gguf'}
    }

def get_model_info_from_config(role, model_configs):
    """Get model info dict for a specific role"""
    config = model_configs[role]
    return {
        "type": config['type'],
        "model_path": config['model_path'],
        "model_name": config['model_path'].split('/')[-1]
    }

def update_batch_summary(batch_folder):
    """Update batch summary file with model configuration"""
    # Look for any summary file in the batch folder
    summary_files = list(batch_folder.glob("*summary*.json"))
    
    if not summary_files:
        print(f"No summary file found in {batch_folder}")
        return False
    
    summary_file = summary_files[0]  # Use the first one found
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    model_configs = get_mistral_model_configs()
    
    # Update configuration section
    if 'configuration' in summary:
        summary['configuration']['model_configs'] = model_configs
        # Remove old 'model' field if it exists
        if 'model' in summary['configuration']:
            del summary['configuration']['model']
    
    # Save updated summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Updated summary: {summary_file}")
    return True

def update_game_files(batch_folder):
    """Update all game files in a batch with model information"""
    model_configs = get_mistral_model_configs()
    updated_count = 0
    
    # Process all game JSON files in the batch folder
    for game_file in batch_folder.glob("*_game_*.json"):
        try:
            with open(game_file, 'r') as f:
                game_data = json.load(f)
            
            # Add model info to initial_players
            if 'initial_players' in game_data:
                for player in game_data['initial_players']:
                    if 'model' not in player and 'role' in player:
                        player['model'] = get_model_info_from_config(player['role'], model_configs)
            
            # Add model info to final_state
            if 'final_state' in game_data:
                for player in game_data['final_state']:
                    if 'model' not in player and 'role' in player:
                        player['model'] = get_model_info_from_config(player['role'], model_configs)
            
            # Add model_configs to game metadata
            if 'model_configs' not in game_data:
                game_data['model_configs'] = model_configs
            
            # Save updated game file
            with open(game_file, 'w') as f:
                json.dump(game_data, f, indent=2)
            
            updated_count += 1
            
        except Exception as e:
            print(f"Error updating {game_file}: {e}")
    
    print(f"Updated {updated_count} game files in {batch_folder}")
    return updated_count

def main():
    """Update all existing batch folders"""
    data_dir = Path(__file__).parent / "data"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    batch_folders = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('batch_')]
    
    if not batch_folders:
        print("No batch folders found")
        return
    
    print("Updating existing batches with model information...")
    print("All existing batches used Mistral for all roles.")
    print("="*60)
    
    for batch_folder in sorted(batch_folders):
        print(f"\nProcessing {batch_folder.name}:")
        
        # Update batch summary
        summary_updated = update_batch_summary(batch_folder)
        
        # Update game files
        games_updated = update_game_files(batch_folder)
        
        print(f"  Summary updated: {summary_updated}")
        print(f"  Games updated: {games_updated}")
    
    print("\n" + "="*60)
    print("Model information update complete!")
    print("All batches now include per-player model tracking.")

if __name__ == "__main__":
    main()