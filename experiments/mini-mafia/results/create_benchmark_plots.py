#!/usr/bin/env python3
"""
Database-based LLM Mafia Benchmark Plot Generator

Creates horizontal bar plots for LLM Mafia benchmark results using SQLite database.
Replaces the original JSON-based script with direct database queries.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import sqlite3
from PIL import Image
import os
import json
import math
from collections import defaultdict
from pathlib import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def get_model_company(model_name):
    """Map model names to their companies"""
    company_mapping = {
        'gpt-4o': 'OpenAI',
        'gpt-4o-mini': 'OpenAI', 
        'gpt-5': 'OpenAI',
        'gpt-5-mini': 'OpenAI',
        'gpt-4.1-mini': 'OpenAI',
        'gpt-4.1-nano': 'OpenAI',
        'grok-4': 'X',
        'grok-3-mini': 'X',
        'claude-3-haiku-20240307': 'Anthropic',
        'claude-3-5-haiku-latest': 'Anthropic',
        'claude-sonnet-4-20250514': 'Anthropic',
        'claude-opus-4-1-20250805': 'Anthropic',
        'deepseek-chat': 'DeepSeek',
        'deepseek-reasoner': 'DeepSeek',
        'gemini-2.5-flash-lite': 'Google',
        'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf': 'Mistral AI',
        'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf': 'Meta',
        'Qwen2.5-7B-Instruct-Q4_K_M.gguf': 'Alibaba',
    }
    
    # Handle variations and fallbacks
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
    elif 'deepseek' in model_name.lower():
        return 'DeepSeek'
    elif 'gemini' in model_name.lower():
        return 'Google'
    
    return company_mapping.get(model_name, 'Unknown')

def get_display_name(model_name):
    """Convert technical model names to display names"""
    display_mapping = {
        'gpt-4.1-mini': 'GPT-4.1 Mini',
        'gpt-4.1-nano': 'GPT-4.1 Nano',
        'gpt-4o': 'GPT-4o',
        'gpt-4o-mini': 'GPT-4o Mini',
        'gpt-5': 'GPT-5',
        'gpt-5-mini': 'GPT-5 Mini',
        'grok-3-mini': 'Grok 3 Mini',
        'grok-4': 'Grok-4',
        'claude-3-haiku-20240307': 'Claude 3 Haiku',
        'claude-3-5-haiku-latest': 'Claude 3.5 Haiku',
        'claude-sonnet-4-20250514': 'Claude Sonnet 4',
        'claude-opus-4-1-20250805': 'Claude Opus 4.1',
        'deepseek-chat': 'DeepSeek V3.1',
        'deepseek-reasoner': 'DeepSeek R1',
        'gemini-2.5-flash-lite': 'Gemini 2.5 Flash Lite',
        'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf': 'Mistral 7B Instruct',
        'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf': 'Llama 3.1 8B',
        'Qwen2.5-7B-Instruct-Q4_K_M.gguf': 'Qwen2.5 7B Instruct',
    }
    
    return display_mapping.get(model_name, model_name)

def get_background_color(background_key):
    """Get color based on background model type"""
    if 'mistral' in background_key.lower():
        return '#FF6600'  # Mistral orange
    elif 'gpt-5' in background_key.lower() and 'mini' in background_key.lower():
        return '#006B3C'  # GPT-5 Mini dark green
    elif 'gpt-4.1' in background_key.lower() and 'mini' in background_key.lower():
        return '#10A37F'  # GPT-4.1 Mini green
    elif 'grok' in background_key.lower() and 'mini' in background_key.lower():
        return '#8A2BE2'  # Grok Mini purple
    elif 'deepseek' in background_key.lower():
        return '#007ACC'  # DeepSeek blue
    elif 'llama' in background_key.lower():
        return '#4A90E2'  # Llama blue
    elif 'qwen' in background_key.lower():
        return '#7B68EE'  # Qwen purple
    else:
        return '#666666'  # Default gray

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
        # Try relative path first (from database/ directory to results/logos/)
        logo_path = f"../results/logos/{logo_files[company]}"
        if not os.path.exists(logo_path):
            # Try direct path in current directory
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

def calculate_bayesian_stats(wins, total):
    """Calculate Bayesian mean and standard deviation using Beta posterior"""
    if total == 0:
        return 0, 0
    
    # Bayesian posterior: Beta(wins + 1, total - wins + 1)
    # E = (wins + 1) / (total + 2)
    bayesian_mean = (wins + 1) / (total + 2)
    
    # SD = sqrt(E(1-E) / (total + 3))
    bayesian_sd = math.sqrt(bayesian_mean * (1 - bayesian_mean) / (total + 3))
    
    # Convert to percentages
    return bayesian_mean * 100, bayesian_sd * 100

def analyze_database_results(db_path='../database/mini_mafia.db'):
    """Analyze game results from SQLite database"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        # Get all game results with player models and tie information
        query = """
        SELECT 
            g.game_id,
            g.winner,
            g.was_tie,
            gp.role,
            p.model_name,
            p.model_provider
        FROM games g
        JOIN game_players gp ON g.game_id = gp.game_id
        JOIN players p ON gp.player_id = p.player_id
        WHERE p.player_id IS NOT NULL  -- Exclude killed players with NULL models
        ORDER BY g.game_id, gp.role
        """
        
        results = conn.execute(query).fetchall()
        
        # Group by game to reconstruct configurations
        games_by_id = defaultdict(lambda: {'winner': None, 'roles': {}, 'was_tie': False})
        
        for row in results:
            game_id = row['game_id']
            games_by_id[game_id]['winner'] = row['winner']
            games_by_id[game_id]['roles'][row['role']] = row['model_name']
            games_by_id[game_id]['was_tie'] = bool(row['was_tie'])
        
        # Group by model configuration
        config_results = defaultdict(lambda: {
            'mafia_wins': 0, 
            'town_wins': 0, 
            'mafia_tie_wins': 0,
            'town_tie_wins': 0,
            'total_games': 0,
            'detective_model': None,
            'mafioso_model': None, 
            'villager_model': None
        })
        
        for game_id, game_data in games_by_id.items():
            if len(game_data['roles']) != 3:  # Should have detective, mafioso, villager
                continue
                
            detective_model = game_data['roles'].get('detective')
            mafioso_model = game_data['roles'].get('mafioso') 
            villager_model = game_data['roles'].get('villager')
            winner = game_data['winner']
            was_tie = game_data['was_tie']
            
            if not all([detective_model, mafioso_model, villager_model, winner]):
                continue
                
            # Create configuration key
            config_key = f"{detective_model}|{mafioso_model}|{villager_model}"
            
            # Store model names
            config_results[config_key]['detective_model'] = detective_model
            config_results[config_key]['mafioso_model'] = mafioso_model
            config_results[config_key]['villager_model'] = villager_model
            
            # Count wins
            config_results[config_key]['total_games'] += 1
            if winner == 'mafia':
                config_results[config_key]['mafia_wins'] += 1
                if was_tie:
                    config_results[config_key]['mafia_tie_wins'] += 1
            elif winner == 'town':
                config_results[config_key]['town_wins'] += 1
                if was_tie:
                    config_results[config_key]['town_tie_wins'] += 1
        
        return dict(config_results)
        
    finally:
        conn.close()

def group_results_by_mafioso_experiments(config_results):
    """Group results for mafioso-changing experiments by background model"""
    background_groups = defaultdict(list)
    
    for config_key, results in config_results.items():
        detective_model = results['detective_model']
        villager_model = results['villager_model']
        mafioso_model = results['mafioso_model']
        
        # Only group configurations where detective == villager (pure backgrounds)
        if detective_model == villager_model:
            background_model = get_display_name(detective_model)
        else:
            continue  # Skip mixed backgrounds
        
        mafia_wins = results['mafia_wins']
        mafia_tie_wins = results['mafia_tie_wins']
        total_games = results['total_games']
        
        if total_games > 0:
            win_rate, sd = calculate_bayesian_stats(mafia_wins, total_games)
            tie_win_rate = (mafia_tie_wins / total_games) * 100
            
            # Check if this varying model already exists in this background group
            existing_entry = None
            for entry in background_groups[background_model]:
                if entry['varying_model'] == get_display_name(mafioso_model):
                    existing_entry = entry
                    break
            
            if existing_entry:
                # Combine with existing entry
                total_mafia_wins = existing_entry['mafia_wins'] + mafia_wins
                total_mafia_tie_wins = existing_entry['mafia_tie_wins'] + mafia_tie_wins
                total_games_combined = existing_entry['games'] + total_games
                combined_win_rate, combined_sd = calculate_bayesian_stats(total_mafia_wins, total_games_combined)
                combined_tie_win_rate = (total_mafia_tie_wins / total_games_combined) * 100
                
                existing_entry.update({
                    'win_rate': combined_win_rate,
                    'tie_win_rate': combined_tie_win_rate,
                    'sem': combined_sd,
                    'games': total_games_combined,
                    'mafia_wins': total_mafia_wins,
                    'mafia_tie_wins': total_mafia_tie_wins,
                })
            else:
                # Add new entry
                background_groups[background_model].append({
                    'varying_model': get_display_name(mafioso_model),
                    'win_rate': win_rate,
                    'tie_win_rate': tie_win_rate,
                    'sem': sd,
                    'games': total_games,
                    'mafia_wins': mafia_wins,
                    'mafia_tie_wins': mafia_tie_wins,
                })
    
    return background_groups

def group_results_by_detective_experiments(config_results):
    """Group results for detective-changing experiments by background model"""
    background_groups = defaultdict(list)
    
    for config_key, results in config_results.items():
        detective_model = results['detective_model']
        villager_model = results['villager_model']
        mafioso_model = results['mafioso_model']
        
        # Only group configurations where mafioso == villager (pure backgrounds)
        if mafioso_model == villager_model:
            background_model = get_display_name(mafioso_model)
        else:
            continue  # Skip mixed backgrounds
        
        town_wins = results['town_wins']
        town_tie_wins = results['town_tie_wins']
        total_games = results['total_games']
        
        if total_games > 0:
            win_rate, sd = calculate_bayesian_stats(town_wins, total_games)
            tie_win_rate = (town_tie_wins / total_games) * 100
            
            # Check if this varying model already exists in this background group
            existing_entry = None
            for entry in background_groups[background_model]:
                if entry['varying_model'] == get_display_name(detective_model):
                    existing_entry = entry
                    break
            
            if existing_entry:
                # Combine with existing entry
                total_town_wins = existing_entry['town_wins'] + town_wins
                total_town_tie_wins = existing_entry['town_tie_wins'] + town_tie_wins
                total_games_combined = existing_entry['games'] + total_games
                combined_win_rate, combined_sd = calculate_bayesian_stats(total_town_wins, total_games_combined)
                combined_tie_win_rate = (total_town_tie_wins / total_games_combined) * 100
                
                existing_entry.update({
                    'win_rate': combined_win_rate,
                    'tie_win_rate': combined_tie_win_rate,
                    'sem': combined_sd,
                    'games': total_games_combined,
                    'town_wins': total_town_wins,
                    'town_tie_wins': total_town_tie_wins,
                })
            else:
                # Add new entry
                background_groups[background_model].append({
                    'varying_model': get_display_name(detective_model),
                    'win_rate': win_rate,
                    'tie_win_rate': tie_win_rate,
                    'sem': sd,
                    'games': total_games,
                    'town_wins': town_wins,
                    'town_tie_wins': town_tie_wins,
                })
    
    return background_groups

def group_results_by_villager_experiments(config_results):
    """Group results for villager-changing experiments by background model"""
    background_groups = defaultdict(list)
    
    for config_key, results in config_results.items():
        detective_model = results['detective_model']
        villager_model = results['villager_model']
        mafioso_model = results['mafioso_model']
        
        # Only group configurations where mafioso == detective (pure backgrounds)
        if mafioso_model == detective_model:
            background_model = get_display_name(mafioso_model)
        else:
            continue  # Skip mixed backgrounds
        
        town_wins = results['town_wins']
        town_tie_wins = results['town_tie_wins']
        total_games = results['total_games']
        
        if total_games > 0:
            win_rate, sd = calculate_bayesian_stats(town_wins, total_games)
            tie_win_rate = (town_tie_wins / total_games) * 100
            
            # Check if this varying model already exists in this background group
            existing_entry = None
            for entry in background_groups[background_model]:
                if entry['varying_model'] == get_display_name(villager_model):
                    existing_entry = entry
                    break
            
            if existing_entry:
                # Combine with existing entry
                total_town_wins = existing_entry['town_wins'] + town_wins
                total_town_tie_wins = existing_entry['town_tie_wins'] + town_tie_wins
                total_games_combined = existing_entry['games'] + total_games
                combined_win_rate, combined_sd = calculate_bayesian_stats(total_town_wins, total_games_combined)
                combined_tie_win_rate = (total_town_tie_wins / total_games_combined) * 100
                
                existing_entry.update({
                    'win_rate': combined_win_rate,
                    'tie_win_rate': combined_tie_win_rate,
                    'sem': combined_sd,
                    'games': total_games_combined,
                    'town_wins': total_town_wins,
                    'town_tie_wins': total_town_tie_wins,
                })
            else:
                # Add new entry
                background_groups[background_model].append({
                    'varying_model': get_display_name(villager_model),
                    'win_rate': win_rate,
                    'tie_win_rate': tie_win_rate,
                    'sem': sd,
                    'games': total_games,
                    'town_wins': town_wins,
                    'town_tie_wins': town_tie_wins,
                })
    
    return background_groups

def create_benchmark_plot(benchmark_data, title, filename, background_key="", use_good_wins=False):
    """Create a horizontal bar plot with logos and formatting, and export numerical data"""
    # Export numerical data alongside the plot
    export_data = {
        'models': benchmark_data['models'],
        'win_rates': benchmark_data['values'],
        'tie_win_rates': benchmark_data.get('tie_values', [0] * len(benchmark_data['values'])),
        'errors': benchmark_data['errors'],
        'companies': benchmark_data['companies'],
        'background': background_key,
        'use_good_wins': use_good_wins
    }
    
    # Create data filename by replacing .png with .json
    data_filename = filename.replace('.png', '_data.json')
    with open(data_filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"Data exported as {data_filename}")
    
    # Use non-interactive backend
    plt.ioff()
    
    # Set font size to match LaTeX document
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
    
    # Add model names on the right side of bars (without values)
    for i, model in enumerate(models):
        ax.text(values[i] + errors[i] + 1.5, i, f'{model}', 
                ha='left', va='center', fontweight='bold', fontsize=24)
    
    # Add company logos on the left
    # for i, company in enumerate(companies):
    #     logo_img = load_company_logo(company, size=(40, 40))
    #     if logo_img is not None:
    #         try:
    #             imagebox = OffsetImage(logo_img, zoom=0.8)
    #             ab = AnnotationBbox(imagebox, (-6, i), frameon=False, 
    #                               xycoords='data', boxcoords="data")
    #             ax.add_artist(ab)
    #         except Exception as e:
    #             print(f"Failed to add logo for {company}: {e}")
    #             # Fallback to company initial
    #             ax.text(-6, i, company[0], ha='center', va='center', 
    #                     fontweight='bold', fontsize=24, color='black',
    #                     bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgray'))
    #     else:
    #         # Fallback to company initial
    #         ax.text(-6, i, company[0], ha='center', va='center', 
    #                 fontweight='bold', fontsize=24, color='black',
    #                 bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgray'))
    
    # Set axis labels based on metric type
    if use_good_wins:
        xlabel = 'Town Win Rate (%)'
    else:
        xlabel = 'Mafia Win Rate (%)'
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold')
    ax.set_yticks([])  # Remove y-axis labels
    ax.set_xlim(0, 100)  # Set range from 0 to 100 (full domain)
    
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
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Plot saved as {filename}")

def main():
    """Create benchmark plots from database"""
    
    print("🔄 Analyzing database results...")
    config_results = analyze_database_results()
    
    if not config_results:
        print("❌ No data found in database")
        return
    
    print(f"📊 Found {len(config_results)} unique model configurations")
    
    # Define allowed models and backgrounds for experiments
    allowed_models = [
        'DeepSeek V3.1', 'Claude Opus 4.1', 'Claude Sonnet 4', 'Gemini 2.5 Flash Lite',
        'Grok 3 Mini', 'GPT-4.1 Mini', 'GPT-5', 'GPT-5 Mini', 'Mistral 7B Instruct', 
        'Qwen2.5 7B Instruct', 'Llama 3.1 8B'
    ]
    
    allowed_backgrounds = [
        'Mistral 7B Instruct', 'GPT-4.1 Mini', 'GPT-5 Mini', 'Grok 3 Mini', 'DeepSeek V3.1',
    ]
    
    # Group results for all three experiment types
    mafioso_groups = group_results_by_mafioso_experiments(config_results)
    detective_groups = group_results_by_detective_experiments(config_results)
    villager_groups = group_results_by_villager_experiments(config_results)
    
    plots_created = 0
    
    # Create mafioso-changing experiment plots (mafia win rate)
    print("\n🟡 Creating mafioso-changing experiment plots...")
    for background_key, results_list in mafioso_groups.items():
        # Only include allowed backgrounds
        if background_key not in allowed_backgrounds:
            continue
            
        # Filter to only allowed models
        results_list = [r for r in results_list if r['varying_model'] in allowed_models]
        
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
        
        # Create descriptive filename using background model
        filename = f"mafioso_{background_key.lower().replace(' ', '_')}_db_benchmark.png"
        
        print(f"📈 Creating mafioso plot: {background_key} background")
        print(f"   Models: {', '.join(models)}")
        print(f"   Sample sizes: {[r['games'] for r in results_list]}")
        
        create_benchmark_plot(benchmark_data, f"Mafioso vs {background_key}", filename, background_key, use_good_wins=False)
        plots_created += 1
    
    # Create detective-changing experiment plots (town win rate)
    print("\n🔵 Creating detective-changing experiment plots...")
    for background_key, results_list in detective_groups.items():
        # Only include allowed backgrounds
        if background_key not in allowed_backgrounds:
            continue
            
        # Filter to only allowed models
        results_list = [r for r in results_list if r['varying_model'] in allowed_models]
        
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
        
        # Create descriptive filename using background model
        filename = f"detective_{background_key.lower().replace(' ', '_')}_db_benchmark.png"
        
        print(f"📈 Creating detective plot: {background_key} background")
        print(f"   Models: {', '.join(models)}")
        print(f"   Sample sizes: {[r['games'] for r in results_list]}")
        
        create_benchmark_plot(benchmark_data, f"Detective vs {background_key}", filename, background_key, use_good_wins=True)
        plots_created += 1
    
    # Create villager-changing experiment plots (town win rate)
    print("\n🟢 Creating villager-changing experiment plots...")
    for background_key, results_list in villager_groups.items():
        # Only include allowed backgrounds
        if background_key not in allowed_backgrounds:
            continue
            
        # Filter to only allowed models
        results_list = [r for r in results_list if r['varying_model'] in allowed_models]
        
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
        
        # Create descriptive filename using background model
        filename = f"villager_{background_key.lower().replace(' ', '_')}_db_benchmark.png"
        
        print(f"📈 Creating villager plot: {background_key} background")
        print(f"   Models: {', '.join(models)}")
        print(f"   Sample sizes: {[r['games'] for r in results_list]}")
        
        create_benchmark_plot(benchmark_data, f"Villager vs {background_key}", filename, background_key, use_good_wins=True)
        plots_created += 1
    
    print(f"\n✅ Created {plots_created} database-based benchmark plots!")
    print(f"   📊 Mafioso experiments: {len(mafioso_groups)} plots")
    print(f"   📊 Detective experiments: {len(detective_groups)} plots")
    print(f"   📊 Villager experiments: {len(villager_groups)} plots")

if __name__ == "__main__":
    main()