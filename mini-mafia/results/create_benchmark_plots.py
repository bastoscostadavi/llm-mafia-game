#!/usr/bin/env python3
"""
Dynamic LLM Mafia Benchmark Plot Generator for v4.1 batches

Creates horizontal bar plots for LLM Mafia benchmark results using v4.1 batch data.
This script processes ONLY v4.1 batches to maintain version separation.
Groups by background models for comparative analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import requests
from PIL import Image
import io
import os
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Import analysis functions from local analyze_results module
from analyze_results import (
    extract_model_name, create_config_key, determine_winner, 
    get_batch_config, calculate_sem
)

def was_tie_vote(players):
    """Determine if the final vote was a tie by checking if all living players received votes"""
    # Find players who were alive at the time of voting (not killed in night phase)
    living_players = [p for p in players if p.get('alive', True)]
    
    # Look for voting information in memory
    voting_info = None
    for player in players:
        memory = player.get('memory', [])
        for entry in reversed(memory):  # Check from most recent
            if entry.startswith("Votes:"):
                voting_info = entry
                break
        if voting_info:
            break
    
    if not voting_info:
        return False
    
    # Extract vote targets from the voting string
    # Format: "Votes: Alice voted for Bob, Charlie voted for Diana, Diana voted for Alice"
    vote_targets = set()
    votes_part = voting_info.replace("Votes: ", "")
    vote_entries = votes_part.split(", ")
    
    for vote_entry in vote_entries:
        # Handle cases like "Alice voted for Bob (the vote was cast randomly because of a failed format)"
        if " voted for " in vote_entry:
            target = vote_entry.split(" voted for ")[1]
            # Remove any parenthetical information
            if " (" in target:
                target = target.split(" (")[0]
            vote_targets.add(target.strip())
    
    # Get names of living players
    living_names = {p['name'] for p in living_players}
    
    # A tie occurred if all living players received at least one vote
    return vote_targets == living_names

def get_model_company(model_name):
    """Map model names to their companies"""
    company_mapping = {
        'GPT-4o': 'OpenAI',
        'GPT-4o Mini': 'OpenAI', 
        'GPT-5': 'OpenAI',
        'GPT-5 Mini': 'OpenAI',
        'GPT-3.5': 'OpenAI',
        'GPT-4.1 Mini': 'OpenAI',
        'Grok-4': 'X',
        'Grok 3 Mini': 'X',
        'Claude 3 Haiku': 'Anthropic',
        'Claude 3.5 Haiku': 'Anthropic',
        'Claude 3 Sonnet': 'Anthropic',
        'Claude 3.5 Sonnet': 'Anthropic',
        'Claude 3 Opus': 'Anthropic',
        'Claude Sonnet 4': 'Anthropic',
        'Claude Opus 4.1': 'Anthropic',
        'DeepSeek V3': 'DeepSeek',
        'DeepSeek R1': 'DeepSeek',
        'Gemini 2.5 Flash Lite': 'Google',
        'Gemini 2.5 Flash': 'Google',
        'Mistral 7B Instruct': 'Mistral AI',
        'Mistral 7B Instruct v0.2': 'Mistral AI',
        'Mistral 7B Instruct v0.3': 'Mistral AI',
        'Llama3.1 8B Instruct': 'Meta',
        'Qwen2.5 7B Instruct': 'Alibaba',
        'Qwen3 14B Instruct': 'Alibaba',
        'GPT-OSS': 'OpenAI',
        'Gemma 2 27B Instruct': 'DeepMind'
    }
    
    # Handle Claude variations and fallbacks
    if 'claude' in model_name.lower():
        return 'Anthropic'
    elif 'gpt' in model_name.lower():
        return 'OpenAI'
    elif 'grok' in model_name.lower():
        return 'X'
    elif 'mistral' in model_name.lower():
        return 'Mistral AI'
    elif 'llama' in model_name.lower():
        return 'Meta'
    elif 'qwen' in model_name.lower():
        return 'Alibaba'
    elif 'gemma' in model_name.lower():
        return 'DeepMind'
    
    return company_mapping.get(model_name, 'Unknown')

def get_company_color(company):
    """Get brand colors for companies"""
    colors = {
        'OpenAI': '#10A37F',
        'X': '#000000',
        'Anthropic': '#DE7C3A', 
        'Mistral AI': '#FF6B35',
        'Meta': '#1877F2',
        'Alibaba': '#FF6A00',
        'DeepMind': '#4285F4',
        'DeepSeek': '#007ACC',  # Blue color for DeepSeek
        'Google': '#4285F4',    # Google blue
        'Unknown': '#666666'
    }
    return colors.get(company, '#666666')

def load_company_logo(company, size=(40, 40)):
    """Load actual company logos from the logos folder"""
    # Map company names to logo filenames  
    logo_files = {
        "OpenAI": "openai.png",
        "X": "xai.png",
        "Mistral AI": "mistral.png", 
        "Meta": "meta.png",
        "Alibaba": "baba.png",
        "Anthropic": "anthropic.png",
        "DeepMind": "deepmind.png",
        "DeepSeek": "deepseek.png",
        "Google": "deepmind.png"  # Use DeepMind logo for Google since they're the same company
    }
    
    try:
        logo_path = f"logos/{logo_files[company]}"
        if os.path.exists(logo_path):
            # Load and resize the actual logo
            img = Image.open(logo_path)
            
            # Convert to RGBA if needed
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Resize maintaining aspect ratio
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Create new image with transparent background
            new_img = Image.new('RGBA', size, (255, 255, 255, 0))
            
            # Center the logo
            x = (size[0] - img.width) // 2
            y = (size[1] - img.height) // 2
            new_img.paste(img, (x, y), img if img.mode == 'RGBA' else None)
            
            return np.array(new_img)
        else:
            print(f"Logo file not found: {logo_path}")
            return None
            
    except Exception as e:
        print(f"Error loading logo for {company}: {e}")
        return None


def analyze_v4_1_batch_data():
    """Analyze ONLY v4.1 batch data and return organized results"""
    data_dir = "../data/batch"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found. Run from mini-mafia/results/ directory.")
        return {}
    
    # Find all v4.1 batch directories (excluding the special GPT-4o vs GPT-5 batch handled in v4.0 script)
    v4_1_batches = []
    for batch_dir in os.listdir(data_dir):
        if batch_dir.endswith("_v4.1"):
            batch_path = os.path.join(data_dir, batch_dir)
            if os.path.isdir(batch_path):
                v4_1_batches.append(batch_path)
    
    if not v4_1_batches:
        print("No v4.1 batch directories found (excluding special GPT-4o vs GPT-5 batch).")
        return {}
    
    print(f"Found {len(v4_1_batches)} v4.1 batch directories")
    
    # Track results by model configuration
    config_results = defaultdict(lambda: {'evil_wins': 0, 'evil_wins_after_tie': 0, 'good_wins_after_tie': 0, 'total_games': 0, 'model_configs': None})
    
    for batch_path in sorted(v4_1_batches):
        batch_name = os.path.basename(batch_path)
        print(f"Analyzing {batch_name}...")
        
        # Get batch configuration
        config = get_batch_config(batch_path)
        if not config:
            continue
        
        model_configs = config.get('model_configs', {})
        config_key = create_config_key(model_configs)
        
        # Store model configs for later use
        if config_results[config_key]['model_configs'] is None:
            config_results[config_key]['model_configs'] = model_configs
        
        # Count games and wins
        evil_wins = 0
        evil_wins_after_tie = 0
        good_wins_after_tie = 0
        total_games = 0
        
        # Process all game files in batch
        for game_file in os.listdir(batch_path):
            if game_file.startswith('game_') and game_file.endswith('.json'):
                game_path = os.path.join(batch_path, game_file)
                try:
                    with open(game_path, 'r') as f:
                        game_data = json.load(f)
                    
                    players = game_data.get('players', [])
                    winner = determine_winner(players)
                    
                    if winner != "unknown":
                        total_games += 1
                        was_tie = was_tie_vote(players)
                        
                        if winner == "evil":
                            evil_wins += 1
                            if was_tie:
                                evil_wins_after_tie += 1
                        elif winner == "good" and was_tie:
                            good_wins_after_tie += 1
                            
                except (json.JSONDecodeError, IOError, KeyError):
                    continue
        
        # Add to overall results
        config_results[config_key]['evil_wins'] += evil_wins
        config_results[config_key]['evil_wins_after_tie'] += evil_wins_after_tie
        config_results[config_key]['good_wins_after_tie'] += good_wins_after_tie
        config_results[config_key]['total_games'] += total_games
    
    return config_results

def sample_games_from_config(batch_path, sample_size=100, random_seed=2):
    """Sample a specified number of games from a batch directory"""
    random.seed(random_seed)
    
    # Get all game files
    game_files = [f for f in os.listdir(batch_path) if f.startswith('game_') and f.endswith('.json')]
    
    if len(game_files) <= sample_size:
        return game_files  # Return all if we have fewer than requested
    
    # Sample without replacement
    sampled_files = random.sample(game_files, sample_size)
    return sampled_files

def analyze_v4_1_batch_data_with_sampling():
    """Analyze v4.1 batch data with sampling for Mistral configurations"""
    data_dir = "../data/batch"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found. Run from mini-mafia/results/ directory.")
        return {}
    
    # Find all v4.1 batch directories
    v4_1_batches = []
    for batch_dir in os.listdir(data_dir):
        if batch_dir.endswith("_v4.1"):
            batch_path = os.path.join(data_dir, batch_dir)
            if os.path.isdir(batch_path):
                v4_1_batches.append(batch_path)
    
    if not v4_1_batches:
        print("No v4.1 batch directories found.")
        return {}
    
    print(f"Found {len(v4_1_batches)} v4.1 batch directories")
    
    # Track results by model configuration
    config_results = defaultdict(lambda: {'evil_wins': 0, 'evil_wins_after_tie': 0, 'good_wins_after_tie': 0, 'total_games': 0, 'model_configs': None})
    
    for batch_path in sorted(v4_1_batches):
        batch_name = os.path.basename(batch_path)
        print(f"Analyzing {batch_name}...")
        
        # Get batch configuration
        config = get_batch_config(batch_path)
        if not config:
            continue
        
        model_configs = config.get('model_configs', {})
        config_key = create_config_key(model_configs)
        
        # Store model configs for later use
        if config_results[config_key]['model_configs'] is None:
            config_results[config_key]['model_configs'] = model_configs
        
        # Check if this is a Mistral configuration (all players are Mistral)
        detective_model = extract_model_name(model_configs.get('detective', {}))
        villager_model = extract_model_name(model_configs.get('villager', {}))
        mafioso_model = extract_model_name(model_configs.get('mafioso', {}))
        
        is_mistral_config = (
            'mistral' in detective_model.lower() and 
            'mistral' in villager_model.lower() and 
            'mistral' in mafioso_model.lower()
        )
        
        # Sample games if this is a Mistral configuration
        if is_mistral_config:
            game_files = sample_games_from_config(batch_path, sample_size=100)
            print(f"   Sampling 100 games from {len([f for f in os.listdir(batch_path) if f.startswith('game_') and f.endswith('.json')])} available games")
        else:
            # Use all games for non-Mistral configurations
            game_files = [f for f in os.listdir(batch_path) if f.startswith('game_') and f.endswith('.json')]
        
        # Count games and wins
        evil_wins = 0
        evil_wins_after_tie = 0
        good_wins_after_tie = 0
        total_games = 0
        
        # Process selected game files
        for game_file in game_files:
            game_path = os.path.join(batch_path, game_file)
            try:
                with open(game_path, 'r') as f:
                    game_data = json.load(f)
                
                players = game_data.get('players', [])
                winner = determine_winner(players)
                
                if winner != "unknown":
                    total_games += 1
                    was_tie = was_tie_vote(players)
                    
                    if winner == "evil":
                        evil_wins += 1
                        if was_tie:
                            evil_wins_after_tie += 1
                    elif winner == "good" and was_tie:
                        good_wins_after_tie += 1
                        
            except (json.JSONDecodeError, IOError, KeyError):
                continue
        
        # Add to overall results
        config_results[config_key]['evil_wins'] += evil_wins
        config_results[config_key]['evil_wins_after_tie'] += evil_wins_after_tie
        config_results[config_key]['good_wins_after_tie'] += good_wins_after_tie
        config_results[config_key]['total_games'] += total_games
        
        print(f"   Processed {total_games} games from {batch_name}")
    
    return config_results

def group_results_by_mafioso_experiments(config_results):
    """Group results for mafioso-changing experiments by background model"""
    background_groups = defaultdict(list)
    
    for config_key, results in config_results.items():
        model_configs = results['model_configs']
        if not model_configs:
            continue
        
        detective_model = extract_model_name(model_configs.get('detective', {}))
        villager_model = extract_model_name(model_configs.get('villager', {}))
        mafioso_model = extract_model_name(model_configs.get('mafioso', {}))
        
        # Determine background model for MAFIOSO experiments
        # Only group configurations where detective == villager (pure backgrounds)
        if detective_model == villager_model:
            background_model = detective_model  # Pure background (same detective and villager)
        else:
            # Skip mixed backgrounds - we only want pure background experiments
            continue
        
        evil_wins = results['evil_wins']
        evil_wins_after_tie = results['evil_wins_after_tie']
        total_games = results['total_games']
        if total_games > 0:
            win_rate = (evil_wins / total_games) * 100
            tie_win_rate = (evil_wins_after_tie / total_games) * 100
            sem = calculate_sem(evil_wins, total_games)
            
            # Check if this varying model already exists in this background group
            existing_entry = None
            for entry in background_groups[background_model]:
                if entry['varying_model'] == mafioso_model:
                    existing_entry = entry
                    break
            
            if existing_entry:
                # Combine with existing entry
                total_evil_wins = existing_entry['evil_wins'] + evil_wins
                total_evil_wins_after_tie = existing_entry['evil_wins_after_tie'] + evil_wins_after_tie
                total_games_combined = existing_entry['games'] + total_games
                combined_win_rate = (total_evil_wins / total_games_combined) * 100
                combined_tie_win_rate = (total_evil_wins_after_tie / total_games_combined) * 100
                combined_sem = calculate_sem(total_evil_wins, total_games_combined)
                
                existing_entry.update({
                    'win_rate': combined_win_rate,
                    'tie_win_rate': combined_tie_win_rate,
                    'sem': combined_sem,
                    'games': total_games_combined,
                    'evil_wins': total_evil_wins,
                    'evil_wins_after_tie': total_evil_wins_after_tie
                })
            else:
                # Add new entry
                background_groups[background_model].append({
                    'varying_model': mafioso_model,
                    'background_config': f"{detective_model}_{villager_model}",  # Keep original config for reference
                    'win_rate': win_rate,
                    'tie_win_rate': tie_win_rate,
                    'sem': sem,
                    'games': total_games,
                    'evil_wins': evil_wins,
                    'evil_wins_after_tie': evil_wins_after_tie
                })
    
    return background_groups

def group_results_by_detective_experiments(config_results):
    """Group results for detective-changing experiments by background model"""
    background_groups = defaultdict(list)
    
    for config_key, results in config_results.items():
        model_configs = results['model_configs']
        if not model_configs:
            continue
        
        detective_model = extract_model_name(model_configs.get('detective', {}))
        villager_model = extract_model_name(model_configs.get('villager', {}))
        mafioso_model = extract_model_name(model_configs.get('mafioso', {}))
        
        # Determine background model for DETECTIVE experiments
        # Only group configurations where mafioso == villager (pure backgrounds)
        if mafioso_model == villager_model:
            background_model = mafioso_model  # Pure background (same mafioso and villager)
        else:
            # Skip mixed backgrounds - we only want pure background experiments
            continue
        
        evil_wins = results['evil_wins']
        good_wins_after_tie = results['good_wins_after_tie']
        total_games = results['total_games']
        if total_games > 0:
            good_wins = total_games - evil_wins  # Calculate actual good wins
            win_rate = (good_wins / total_games) * 100  # Good win rate for detective experiments
            tie_win_rate = (good_wins_after_tie / total_games) * 100
            sem = calculate_sem(good_wins, total_games)
            
            # Check if this varying model already exists in this background group
            existing_entry = None
            for entry in background_groups[background_model]:
                if entry['varying_model'] == detective_model:
                    existing_entry = entry
                    break
            
            if existing_entry:
                # Combine with existing entry
                total_good_wins = existing_entry['good_wins'] + good_wins
                total_good_wins_after_tie = existing_entry['good_wins_after_tie'] + good_wins_after_tie
                total_games_combined = existing_entry['games'] + total_games
                combined_win_rate = (total_good_wins / total_games_combined) * 100
                combined_tie_win_rate = (total_good_wins_after_tie / total_games_combined) * 100
                combined_sem = calculate_sem(total_good_wins, total_games_combined)
                
                existing_entry.update({
                    'win_rate': combined_win_rate,
                    'tie_win_rate': combined_tie_win_rate,
                    'sem': combined_sem,
                    'games': total_games_combined,
                    'good_wins': total_good_wins,
                    'good_wins_after_tie': total_good_wins_after_tie
                })
            else:
                # Add new entry
                background_groups[background_model].append({
                    'varying_model': detective_model,
                    'background_config': f"{mafioso_model}_{villager_model}",  # Keep original config for reference
                    'win_rate': win_rate,
                    'tie_win_rate': tie_win_rate,
                    'sem': sem,
                    'games': total_games,
                    'good_wins': good_wins,
                    'good_wins_after_tie': good_wins_after_tie
                })
    
    return background_groups

def group_results_by_villager_experiments(config_results):
    """Group results for villager-changing experiments by background model"""
    background_groups = defaultdict(list)
    
    for config_key, results in config_results.items():
        model_configs = results['model_configs']
        if not model_configs:
            continue
        
        detective_model = extract_model_name(model_configs.get('detective', {}))
        villager_model = extract_model_name(model_configs.get('villager', {}))
        mafioso_model = extract_model_name(model_configs.get('mafioso', {}))
        
        # Determine background model for VILLAGER experiments
        # Only group configurations where mafioso == detective (pure backgrounds)
        if mafioso_model == detective_model:
            background_model = mafioso_model  # Pure background (same mafioso and detective)
        else:
            # Skip mixed backgrounds - we only want pure background experiments
            continue
        
        evil_wins = results['evil_wins']
        good_wins_after_tie = results['good_wins_after_tie']
        total_games = results['total_games']
        if total_games > 0:
            good_wins = total_games - evil_wins  # Calculate actual good wins
            win_rate = (good_wins / total_games) * 100  # Good win rate for villager experiments
            tie_win_rate = (good_wins_after_tie / total_games) * 100
            sem = calculate_sem(good_wins, total_games)
            
            # Check if this varying model already exists in this background group
            existing_entry = None
            for entry in background_groups[background_model]:
                if entry['varying_model'] == villager_model:
                    existing_entry = entry
                    break
            
            if existing_entry:
                # Combine with existing entry
                total_good_wins = existing_entry['good_wins'] + good_wins
                total_good_wins_after_tie = existing_entry['good_wins_after_tie'] + good_wins_after_tie
                total_games_combined = existing_entry['games'] + total_games
                combined_win_rate = (total_good_wins / total_games_combined) * 100
                combined_tie_win_rate = (total_good_wins_after_tie / total_games_combined) * 100
                combined_sem = calculate_sem(total_good_wins, total_games_combined)
                
                existing_entry.update({
                    'win_rate': combined_win_rate,
                    'tie_win_rate': combined_tie_win_rate,
                    'sem': combined_sem,
                    'games': total_games_combined,
                    'good_wins': total_good_wins,
                    'good_wins_after_tie': total_good_wins_after_tie
                })
            else:
                # Add new entry
                background_groups[background_model].append({
                    'varying_model': villager_model,
                    'background_config': f"{mafioso_model}_{detective_model}",  # Keep original config for reference
                    'win_rate': win_rate,
                    'tie_win_rate': tie_win_rate,
                    'sem': sem,
                    'games': total_games,
                    'good_wins': good_wins,
                    'good_wins_after_tie': good_wins_after_tie
                })
    
    return background_groups

def get_background_color(background_key):
    """Get color based on background model type"""
    if 'mistral' in background_key.lower():
        return '#FF6600'  # Mistral orange
    elif 'gpt-4o mini' in background_key.lower():
        return '#00D4AA'  # GPT-4o Mini lighter green
    elif 'gpt-4o' in background_key.lower():
        return '#00A67E'  # GPT-4o green  
    elif 'gpt-4.1' in background_key.lower() and 'mini' in background_key.lower():
        return '#10A37F'  # GPT-4.1 Mini green
    elif 'grok' in background_key.lower() and 'mini' in background_key.lower():
        return '#8A2BE2'  # Grok Mini purple
    elif 'grok' in background_key.lower():
        return '#FF6B35'  # Grok orange/red
    elif 'llama' in background_key.lower():
        return '#4A90E2'  # Llama blue
    elif 'qwen' in background_key.lower():
        return '#7B68EE'  # Qwen purple
    elif 'gemma' in background_key.lower():
        return '#FF4081'  # Gemma pink
    elif 'deepseek' in background_key.lower():
        return '#007ACC'  # DeepSeek blue
    elif 'gemini' in background_key.lower():
        return '#4285F4'  # Google blue
    else:
        return '#666666'  # Default gray

def create_benchmark_plot(benchmark_data, title, filename, background_key="", use_good_wins=False):
    """Create a horizontal bar plot with logos and formatting"""
    # Use non-interactive backend
    plt.ioff()
    
    # Set font size to match LaTeX document (even larger for better readability)
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
        'figure.titlesize': 24
    })
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    models = benchmark_data['models']
    values = benchmark_data['values'] 
    tie_values = benchmark_data.get('tie_values', [0] * len(values))
    errors = benchmark_data['errors']
    companies = benchmark_data['companies']
    
    y_positions = range(len(models))
    
    # Get color based on background
    bar_color = get_background_color(background_key)
    
    # Create main bars
    bars = ax.barh(y_positions, values, xerr=errors, 
                   color=bar_color, alpha=0.8, height=0.6,
                   error_kw={'capsize': 5, 'capthick': 2})
    
    # Create hatched overlay bars for tie-based wins
    tie_bars = ax.barh(y_positions, tie_values, 
                       color=bar_color, alpha=0.8, height=0.6,
                       hatch='///', edgecolor='white', linewidth=1)
    
    # Add model names and values on the right side of bars
    for i, (model, value, error) in enumerate(zip(models, values, errors)):
        ax.text(value + error + 1.5, i, f'{model}: {value:.0f}% ± {error:.1f}%', 
                ha='left', va='center', fontweight='bold', fontsize=24)
    
    # Add company logos on the left
    for i, company in enumerate(companies):
        logo_img = load_company_logo(company, size=(40, 40))
        if logo_img is not None:
            try:
                imagebox = OffsetImage(logo_img, zoom=0.8)
                ab = AnnotationBbox(imagebox, (-6, i), frameon=False, 
                                  xycoords='data', boxcoords="data")
                ax.add_artist(ab)
            except Exception as e:
                print(f"Failed to add logo for {company}: {e}")
                # Fallback to company initial
                ax.text(-6, i, company[0], ha='center', va='center', 
                        fontweight='bold', fontsize=24, color='black',
                        bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgray'))
        else:
            # Fallback to company initial
            ax.text(-6, i, company[0], ha='center', va='center', 
                    fontweight='bold', fontsize=24, color='black',
                    bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgray'))
    
    # Set axis labels based on metric type
    if use_good_wins:
        xlabel = 'Town Win Rate (%)'
    else:
        xlabel = 'Mafia Win Rate (%)'
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold')
    # Remove title - will be handled by LaTeX captions
    ax.set_yticks([])  # Remove y-axis labels
    ax.set_xlim(-10, 100)  # Set range from -10 (for logos) to 100 (full domain)
    
    # Customize x-axis to only show positive values
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    
    # Add custom grid lines only for positive values
    for x in [0, 20, 40, 60, 80, 100]:
        ax.axvline(x=x, color='gray', alpha=0.3, linewidth=0.5)
    
    # Hide all spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Add custom bottom axis line only from 0 to 100
    ax.plot([0, 100], [ax.get_ylim()[0], ax.get_ylim()[0]], color='black', linewidth=0.8)
    
    # Remove legend - will be explained in LaTeX caption
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Plot saved as {filename}")

def main():
    """Create dynamic benchmark plots from v4.1 batch data"""
    
    print("🔄 Analyzing v4.1 batch data with sampling...")
    config_results = analyze_v4_1_batch_data_with_sampling()
    
    if not config_results:
        print("❌ No v4.1 data found to create plots")
        return
    
    print("📊 Grouping v4.1 results by experiment type...")
    
    # Group results for all three experiment types
    mafioso_groups = group_results_by_mafioso_experiments(config_results)
    detective_groups = group_results_by_detective_experiments(config_results)
    villager_groups = group_results_by_villager_experiments(config_results)
    
    plots_created = 0
    
    # Define the main 4 backgrounds we want to plot
    main_backgrounds = ['Mistral 7B Instruct', 'GPT-4.1 Mini', 'Grok 3 Mini', 'DeepSeek V3']
    
    # Models to exclude from plots due to high failure rates or poor format compliance
    excluded_models = [
        'Claude 3 Haiku',      # High message format failures
        'Claude 3.5 Haiku',   # High message format failures
        'GPT-4o Mini',
        'DeepSeek R1'
    ]
    
    # Create mafioso-changing experiment plots (evil win rate)
    print("\n🟡 Creating v4.1 mafioso-changing experiment plots...")
    for background_key, results_list in mafioso_groups.items():
        if background_key not in main_backgrounds:  # Only plot main backgrounds
            continue
        
        # Filter out excluded models
        results_list = [r for r in results_list if r['varying_model'] not in excluded_models]
        
        if len(results_list) < 1:  # Need at least 1 model
            continue
        
        # Sort by win rate descending, then alphabetically for ties
        results_list.sort(key=lambda x: (-x['win_rate'], x['varying_model']))
        
        # Extract data for plotting (reverse order so highest scores appear at top)
        models = [r['varying_model'] for r in reversed(results_list)]
        values = [r['win_rate'] for r in reversed(results_list)]
        tie_values = [r['tie_win_rate'] for r in reversed(results_list)]
        errors = [r['sem'] for r in reversed(results_list)]
        companies = [get_model_company(model) for model in models]
        
        # Create benchmark data structure
        benchmark_data = {
            'models': models,
            'values': values,
            'tie_values': tie_values,
            'errors': errors,
            'companies': companies
        }
        
        # Create descriptive title and filename using background model
        background_model = background_key  # background_key is now the background model name
        title = f"Mafioso vs {background_model} Town"
        filename = f"mafioso_{background_model.lower().replace(' ', '_')}_v4_1_benchmark.png"
        
        print(f"📈 Creating v4.1 mafioso plot: {title}")
        print(f"   Models: {', '.join(models)}")
        print(f"   Sample sizes: {[r['games'] for r in results_list]}")
        
        create_benchmark_plot(benchmark_data, title, filename, background_key, use_good_wins=False)
        plots_created += 1
    
    # Create detective-changing experiment plots (good win rate)
    print("\n🔵 Creating v4.1 detective-changing experiment plots...")
    for background_key, results_list in detective_groups.items():
        if background_key not in main_backgrounds:  # Only plot main backgrounds
            continue
        
        # Filter out excluded models
        results_list = [r for r in results_list if r['varying_model'] not in excluded_models]
        
        if len(results_list) < 1:  # Need at least 1 model
            continue
        
        # Sort by win rate descending, then alphabetically for ties
        results_list.sort(key=lambda x: (-x['win_rate'], x['varying_model']))
        
        # Extract data for plotting (reverse order so highest scores appear at top)
        models = [r['varying_model'] for r in reversed(results_list)]
        values = [r['win_rate'] for r in reversed(results_list)]
        tie_values = [r['tie_win_rate'] for r in reversed(results_list)]
        errors = [r['sem'] for r in reversed(results_list)]
        companies = [get_model_company(model) for model in models]
        
        # Create benchmark data structure
        benchmark_data = {
            'models': models,
            'values': values,
            'tie_values': tie_values,
            'errors': errors,
            'companies': companies
        }
        
        # Create descriptive title and filename using background model
        background_model = background_key  # background_key is now the background model name
        title = f"Detective vs {background_model} Background"
        filename = f"detective_{background_model.lower().replace(' ', '_')}_v4_1_benchmark.png"
        
        print(f"📈 Creating v4.1 detective plot: {title}")
        print(f"   Models: {', '.join(models)}")
        print(f"   Sample sizes: {[r['games'] for r in results_list]}")
        
        create_benchmark_plot(benchmark_data, title, filename, background_key, use_good_wins=True)
        plots_created += 1
    
    # Create villager-changing experiment plots (good win rate)
    print("\n🟢 Creating v4.1 villager-changing experiment plots...")
    for background_key, results_list in villager_groups.items():
        if background_key not in main_backgrounds:  # Only plot main backgrounds
            continue
        
        # Filter out excluded models
        results_list = [r for r in results_list if r['varying_model'] not in excluded_models]
        
        if len(results_list) < 1:  # Need at least 1 model
            continue
        
        # Sort by win rate descending, then alphabetically for ties
        results_list.sort(key=lambda x: (-x['win_rate'], x['varying_model']))
        
        # Extract data for plotting (reverse order so highest scores appear at top)
        models = [r['varying_model'] for r in reversed(results_list)]
        values = [r['win_rate'] for r in reversed(results_list)]
        tie_values = [r['tie_win_rate'] for r in reversed(results_list)]
        errors = [r['sem'] for r in reversed(results_list)]
        companies = [get_model_company(model) for model in models]
        
        # Create benchmark data structure
        benchmark_data = {
            'models': models,
            'values': values,
            'tie_values': tie_values,
            'errors': errors,
            'companies': companies
        }
        
        # Create descriptive title and filename using background model
        background_model = background_key  # background_key is now the background model name
        title = f"Villager vs {background_model} Background"
        filename = f"villager_{background_model.lower().replace(' ', '_')}_v4_1_benchmark.png"
        
        print(f"📈 Creating v4.1 villager plot: {title}")
        print(f"   Models: {', '.join(models)}")
        print(f"   Sample sizes: {[r['games'] for r in results_list]}")
        
        create_benchmark_plot(benchmark_data, title, filename, background_key, use_good_wins=True)
        plots_created += 1
    
    # Count plots per experiment type for main backgrounds
    mafioso_main_plots = len([bg for bg in mafioso_groups.keys() if bg in main_backgrounds and len(mafioso_groups[bg]) >= 1])
    detective_main_plots = len([bg for bg in detective_groups.keys() if bg in main_backgrounds and len(detective_groups[bg]) >= 1])
    villager_main_plots = len([bg for bg in villager_groups.keys() if bg in main_backgrounds and len(villager_groups[bg]) >= 1])
    
    print(f"\n✅ Created {plots_created} main v4.1 benchmark plots!")
    print(f"   📊 Mafioso experiments: {mafioso_main_plots} plots (Mistral, GPT-4.1 Mini, Grok 3 Mini, DeepSeek V3 backgrounds)")
    print(f"   📊 Detective experiments: {detective_main_plots} plots (Mistral, GPT-4.1 Mini, Grok 3 Mini, DeepSeek V3 backgrounds)")
    print(f"   📊 Villager experiments: {villager_main_plots} plots (Mistral, GPT-4.1 Mini, Grok 3 Mini, DeepSeek V3 backgrounds)")
    print(f"\n🎯 Target: 12 plots total (4 backgrounds × 3 experiments)")
    print("\nNote: Only plotting main backgrounds. Additional experimental data available but filtered out.")

if __name__ == "__main__":
    main()