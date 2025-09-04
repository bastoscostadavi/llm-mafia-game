#!/usr/bin/env python3
"""
Create table with aggregated Deceive/Detect/Disclose scores

Extracts z-scores from the aggregated plotting data and creates a clean table:
- Rows: Models (alphabetically ordered)
- Columns: Deceive, Detect, Disclose
- Values: Z-score ± error
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from create_aggregated_score_plots import load_data_files, compute_z_scores, aggregate_z_scores

def extract_aggregated_scores():
    """Extract z-scores for all three experiment types"""
    scores_data = {}
    
    # Load and compute scores for each experiment type
    for behavior_type, display_name in [('mafioso', 'Deceive'), ('detective', 'Detect'), ('villager', 'Disclose')]:
        print(f"📊 Processing {display_name} ({behavior_type}) data...")
        
        try:
            # Load data files for this behavior type
            datasets = load_data_files(behavior_type)
            
            if not datasets:
                print(f"   ⚠️  No data files found for {behavior_type}")
                continue
            
            print(f"   Loaded {len(datasets)} background datasets")
            
            # Compute z-scores and aggregate them
            model_z_scores, model_companies = compute_z_scores(datasets)
            aggregated_results = aggregate_z_scores(model_z_scores)
            
            if not aggregated_results:
                print(f"   ⚠️  No results computed for {behavior_type}")
                continue
            
            print(f"   Computed z-scores for {len(aggregated_results)} models")
            
            # Store results
            scores_data[display_name] = {}
            for model_name, result_data in aggregated_results.items():
                z_score = result_data['mean_z_score']
                z_error = result_data['sem_z_score']
                scores_data[display_name][model_name] = (z_score, z_error)
                
        except Exception as e:
            print(f"   ❌ Error processing {behavior_type}: {e}")
            continue
    
    return scores_data

def create_scores_table(scores_data, output_file='aggregated_scores_table.csv'):
    """Create a table with models and their Deceive/Detect/Disclose scores"""
    
    # Get all unique models across all experiment types
    all_models = set()
    for experiment_scores in scores_data.values():
        all_models.update(experiment_scores.keys())
    
    # Sort models alphabetically
    sorted_models = sorted(all_models)
    
    print(f"📊 Creating table for {len(sorted_models)} models")
    
    # Create table data
    table_data = []
    for model in sorted_models:
        row = {'Model': model}
        
        for experiment in ['Deceive', 'Detect', 'Disclose']:
            if experiment in scores_data and model in scores_data[experiment]:
                z_score, z_error = scores_data[experiment][model]
                # Format as "score ± error"
                row[experiment] = f"{z_score:.2f} ± {z_error:.2f}"
            else:
                row[experiment] = "N/A"
        
        table_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(table_data)
    df.to_csv(output_file, index=False)
    
    print(f"📋 Table saved to {output_file}")
    return df

def print_scores_table(df):
    """Print the scores table in a formatted way"""
    print("\nAGGREGATED Z-SCORES TABLE")
    print("=" * 80)
    print()
    
    # Print header
    print(f"{'Model':<25} {'Deceive':<18} {'Detect':<18} {'Disclose':<18}")
    print("-" * 80)
    
    # Print data rows
    for _, row in df.iterrows():
        print(f"{row['Model']:<25} {row['Deceive']:<18} {row['Detect']:<18} {row['Disclose']:<18}")
    
    print("-" * 80)
    print()
    print("Note: Z-scores represent performance relative to background models")
    print("      Negative values = Better performance (lower win rate for opponents)")
    print("      Positive values = Worse performance (higher win rate for opponents)")

def main():
    """Main function to create aggregated scores table"""
    print("🔄 Extracting aggregated z-scores from plot data...")
    
    # Extract scores from aggregated plotting functions
    scores_data = extract_aggregated_scores()
    
    if not scores_data:
        print("❌ No scores data found")
        return
    
    print(f"✅ Extracted scores for {len(scores_data)} experiment types")
    
    # Create and save table
    df = create_scores_table(scores_data)
    
    # Print formatted table
    print_scores_table(df)
    
    print("✅ Aggregated scores table created successfully!")

if __name__ == "__main__":
    main()