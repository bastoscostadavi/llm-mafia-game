#!/usr/bin/env python3
"""
View Batch Information

Simple utility to view batch information and progress.
"""

import sys
import json
from database.db_utils import MiniMafiaDB

def format_model_config(config_json):
    """Format model configuration for display."""
    if not config_json:
        return "Unknown"
    
    try:
        config = json.loads(config_json)
        parts = []
        for role, model_info in config.items():
            model_name = model_info.get('model', 'unknown')
            provider = model_info.get('type', 'unknown')
            parts.append(f"{role}: {model_name}({provider})")
        return ", ".join(parts)
    except:
        return "Invalid JSON"

def main():
    """Main function to display batch information."""
    db = MiniMafiaDB()
    db.connect()
    
    try:
        print("RECENT BATCHES")
        print("=" * 80)
        
        batches = db.list_batches(100)
        
        if not batches:
            print("No batches found in database.")
            return
        
        print(f"{'Batch ID':<20} {'Timestamp':<20} {'Progress':<12}")
        print("-" * 50)
        
        for batch in batches:
            batch_id, timestamp, planned, completed = batch
            
            # Format timestamp
            timestamp_str = timestamp[:16] if timestamp else "Unknown"
            
            # Progress
            progress = f"{completed or 0}/{planned or 0}"
            
            print(f"{batch_id:<20} {timestamp_str:<20} {progress:<12}")
        
        print("\n" + "=" * 80)
        
        # Show detailed info for most recent batch
        if batches:
            recent_batch_id = batches[0][0]
            print(f"LATEST BATCH DETAILS: {recent_batch_id}")
            print("-" * 40)
            
            batch_info = db.get_batch_info(recent_batch_id)
            if batch_info:
                batch_id, timestamp, planned, completed, model_configs = batch_info
                
                print(f"Timestamp: {timestamp}")
                print(f"Games Planned: {planned or 0}")
                print(f"Games Completed: {completed or 0}")
                print(f"Model Configuration:")
                print(f"  {format_model_config(model_configs)}")
                
                # Show completion percentage
                if planned and planned > 0:
                    percentage = (completed or 0) / planned * 100
                    print(f"Progress: {percentage:.1f}% complete")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        db.close()

if __name__ == "__main__":
    main()