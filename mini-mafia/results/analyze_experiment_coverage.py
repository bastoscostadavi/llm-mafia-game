#!/usr/bin/env python3
"""
Experiment Coverage Analyzer

Analyzes all v4.1 batch data to show what experiments have been run and how many games
for each model combination. Organizes by behavior type (Deceive/Detect/Disclose) and
background conditions.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

def extract_model_name(model_config):
    """Extract short model name from model configuration"""
    if not model_config:
        return "unknown"
    
    # Get model filename from either new 'model' field or old 'model_path' field
    model_filename = model_config.get('model', '')
    if not model_filename and 'model_path' in model_config:
        import os
        model_filename = os.path.basename(model_config['model_path'])
    
    # Check if this is a local model file
    if model_filename.endswith('.gguf'):
        model_mapping = {
            'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf': 'Mistral 7B Instruct',
            'Mistral-7B-Instruct-v0.3-Q4_K_M.gguf': 'Mistral 7B Instruct v0.3',
            'Qwen2.5-7B-Instruct-Q4_K_M.gguf': 'Qwen2.5 7B Instruct',
            'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf': 'Llama3.1 8B Instruct',
            'mistral.gguf': 'Mistral 7B Instruct'
        }
        return model_mapping.get(model_filename, model_filename.replace('.gguf', ''))
    
    # Handle API models
    if model_config.get('type') == 'openai':
        model = model_config.get('model', 'unknown')
        if model.startswith('gpt-4.1-mini'):
            return 'GPT-4.1 Mini'
        elif 'gpt-4o-mini' in model.lower():
            return 'GPT-4o Mini'
        else:
            return model.upper()
    
    elif model_config.get('type') == 'xai':
        model = model_config.get('model', 'unknown')
        if model.startswith('grok-3-mini'):
            return 'Grok 3 Mini'
        else:
            return model.upper()
    
    elif model_config.get('type') == 'deepseek':
        model = model_config.get('model', 'unknown')
        if model.startswith('deepseek-chat') or model.startswith('deepseek-v3'):
            return 'DeepSeek V3'
        else:
            return model.upper()
    
    elif model_config.get('type') == 'google':
        model = model_config.get('model', 'unknown')
        if model.startswith('gemini-2.5-flash-lite'):
            return 'Gemini 2.5 Flash Lite'
        else:
            return model.upper()
    
    elif model_config.get('type') == 'anthropic':
        model = model_config.get('model', 'unknown')
        if 'claude-sonnet-4' in model or 'sonnet-4' in model:
            return 'Claude Sonnet 4'
        elif 'claude-opus-4' in model or 'opus-4' in model:
            return 'Claude Opus 4.1'
        else:
            return f"Claude {model.replace('claude-', '')}"
    
    else:
        return "unknown"

def get_batch_config(batch_path):
    """Extract configuration from batch folder"""
    config_path = os.path.join(batch_path, 'batch_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None

def count_games_in_batch(batch_path):
    """Count the number of game files in a batch"""
    game_count = 0
    for file in os.listdir(batch_path):
        if file.startswith('game_') and file.endswith('.json'):
            game_count += 1
    return game_count

def get_standard_background_name(model_name):
    """Map model name to one of the 4 standard backgrounds"""
    standard_backgrounds = {
        'DeepSeek V3': 'DeepSeek V3',
        'GPT-4.1 Mini': 'GPT-4.1 Mini', 
        'Grok 3 Mini': 'Grok 3 Mini',
        'Mistral 7B Instruct': 'Mistral 7B Instruct'
    }
    return standard_backgrounds.get(model_name, None)

def analyze_experiment_coverage():
    """Analyze experiment coverage across all v4.1 batches"""
    data_dir = "../data/batch"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found. Run from mini-mafia/results/ directory.")
        return
    
    # Find all v4.1 batch directories
    v4_batches = []
    for batch_dir in os.listdir(data_dir):
        if batch_dir.endswith("_v4.1"):
            batch_path = os.path.join(data_dir, batch_dir)
            if os.path.isdir(batch_path):
                v4_batches.append(batch_path)
    
    if not v4_batches:
        print("No v4.1 batch directories found.")
        return
    
    print(f"🔍 Analyzing experiment coverage across {len(v4_batches)} v4.1 batches...")
    print("=" * 80)
    
    # Standard backgrounds used in benchmark plots
    standard_backgrounds = ['DeepSeek V3', 'GPT-4.1 Mini', 'Grok 3 Mini', 'Mistral 7B Instruct']
    
    # Track experiments by type and standard background
    experiments = {
        'deceive': {bg: defaultdict(int) for bg in standard_backgrounds},
        'detect': {bg: defaultdict(int) for bg in standard_backgrounds}, 
        'disclose': {bg: defaultdict(int) for bg in standard_backgrounds}
    }
    
    for batch_path in sorted(v4_batches):
        batch_name = os.path.basename(batch_path)
        
        # Get batch configuration
        config = get_batch_config(batch_path)
        if not config:
            print(f"⚠️  Warning: Could not read batch config for {batch_name}")
            continue
        
        model_configs = config.get('model_configs', {})
        if not model_configs:
            continue
        
        # Extract model names
        detective_model = extract_model_name(model_configs.get('detective', {}))
        mafioso_model = extract_model_name(model_configs.get('mafioso', {}))
        villager_model = extract_model_name(model_configs.get('villager', {}))
        
        # Count games in this batch
        game_count = count_games_in_batch(batch_path)
        if game_count == 0:
            continue
        
        # Skip if any model is unknown
        if detective_model == "unknown" or mafioso_model == "unknown" or villager_model == "unknown":
            continue
        
        # Determine which standard background this batch contributes to
        # A batch contributes to a background if 2 of its 3 models match that background
        
        for background in standard_backgrounds:
            # Count how many roles match this background (the other 2 roles form the background)
            background_roles = []
            tested_role = None
            
            # For deceive experiments: mafioso is tested, detective+villager form background
            if detective_model == background and villager_model == background:
                experiments['deceive'][background][mafioso_model] += game_count
            
            # For detect experiments: villager is tested, detective+mafioso form background  
            if detective_model == background and mafioso_model == background:
                experiments['detect'][background][villager_model] += game_count
                
            # For disclose experiments: detective is tested, mafioso+villager form background
            if mafioso_model == background and villager_model == background:
                experiments['disclose'][background][detective_model] += game_count
    
    # Print results
    expected_models = [
        'GPT-4.1 Mini', 'GPT-4o Mini', 'Grok 3 Mini', 'DeepSeek V3', 
        'Gemini 2.5 Flash Lite', 'Claude Sonnet 4', 'Claude Opus 4.1', 
        'Mistral 7B Instruct', 'Llama3.1 8B Instruct', 'Qwen2.5 7B Instruct'
    ]
    
    for experiment_type, experiment_data in experiments.items():
        print(f"\n🎯 {experiment_type.upper()} EXPERIMENTS")
        print("=" * 50)
        
        for background in standard_backgrounds:
            models = experiment_data[background]
            print(f"\n📋 Background: {background}")
            print("-" * 40)
            
            if not models:
                print("  ❌ No experiments found for this background")
                continue
            
            # Show what we have
            total_games = 0
            for model, count in sorted(models.items()):
                if count > 0:
                    print(f"  ✅ {model:<25} {count:>4} games")
                    total_games += count
            
            if total_games > 0:
                print(f"  {'TOTAL':<25} {total_games:>4} games")
                
                # Show what might be missing
                tested_models = set(model for model, count in models.items() if count > 0)
                missing_models = set(expected_models) - tested_models
                if missing_models:
                    print(f"  ⚠️  Missing models: {', '.join(sorted(missing_models))}")
            else:
                print("  ❌ No experiments found for this background")
    
    print(f"\n" + "=" * 80)
    print("✅ Analysis complete!")
    
    # Summary statistics
    total_experiments = sum(len(bg_data) for exp_data in experiments.values() for bg_data in exp_data.values())
    total_games = sum(count for exp_data in experiments.values() 
                     for bg_data in exp_data.values() 
                     for count in bg_data.values())
    
    print(f"📊 Summary: {total_experiments} model configurations, {total_games} total games")

if __name__ == "__main__":
    analyze_experiment_coverage()