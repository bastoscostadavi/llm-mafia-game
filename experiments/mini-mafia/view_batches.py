#!/usr/bin/env python3
"""
View Batch Information

Simple utility to view batch information and progress.
"""

import sys
import json
from database.db_utils import MiniMafiaDB

def get_role_model(config_json, role):
    """Extract model name for a specific role."""
    if not config_json:
        return "Unknown"
    
    try:
        config = json.loads(config_json)
        if role in config:
            return config[role].get('model', 'unknown')
        return "-"
    except:
        return "Invalid"

def main():
    """Main function to display batch information."""
    db = MiniMafiaDB()
    db.connect()
    
    try:
        print("RECENT BATCHES")
        print("=" * 120)
        
        batches = db.list_batches(100)
        
        if not batches:
            print("No batches found in database.")
            return
        
        # Header
        print(f"{'Batch ID':<20} {'Detective':<25} {'Mafioso':<25} {'Villager':<25} {'Games':<12}")
        print("-" * 120)
        
        # Get detailed info for each batch
        for batch in batches:
            batch_id = batch[0]
            completed = batch[3] or 0
            planned = batch[2] or 0
            
            batch_info = db.get_batch_info(batch_id)
            if batch_info:
                _, _, _, _, model_configs = batch_info
                
                detective_model = get_role_model(model_configs, 'detective')
                mafioso_model = get_role_model(model_configs, 'mafioso')
                villager_model = get_role_model(model_configs, 'villager')
                games_progress = f"{completed}/{planned}"
                
                print(f"{batch_id:<20} {detective_model:<25} {mafioso_model:<25} {villager_model:<25} {games_progress:<12}")
            else:
                print(f"{batch_id:<20} {'Unknown':<25} {'Unknown':<25} {'Unknown':<25} {f'{completed}/{planned}':<12}")
        
        print("=" * 120)
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        db.close()

if __name__ == "__main__":
    main()