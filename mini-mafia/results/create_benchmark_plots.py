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
from collections import defaultdict
from pathlib import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Import analysis functions from analyze_results
import sys
sys.path.append('../analysis')
from analyze_results import (
    extract_model_name, create_config_key, determine_winner, 
    get_batch_config, calculate_sem
)

def get_model_company(model_name):
    """Map model names to their companies"""
    company_mapping = {
        'GPT-4o': 'OpenAI', 
        'GPT-5': 'OpenAI',
        'GPT-3.5': 'OpenAI',
        'GPT 4.1 Mini': 'OpenAI',
        'Grok-4': 'X',
        'Grok 3 Mini': 'X',
        'Claude-3-Haiku': 'Anthropic',
        'Claude-3-Sonnet': 'Anthropic',
        'Claude-3-Opus': 'Anthropic',
        'Claude-Sonnet-4': 'Anthropic',
        'Claude Opus 4.1': 'Anthropic',
        'Mistral 7B Instruct': 'Mistral AI',
        'Mistral 7B Instruct v0.2': 'Mistral AI',
        'Mistral 7B Instruct v0.3': 'Mistral AI',
        'Llama3.1 8B Instruct': 'Meta',
        'Qwen2.5 7B Instruct': 'Alibaba',
        'Qwen3 14B Instruct': 'Alibaba',
        'GPT-OSS': 'OpenAI',
        'Gemma 2 27B Instruct': 'DeepMind'
    }
    
    # Handle Claude variations
    if 'claude-sonnet-4' in model_name.lower() or 'sonnet-4' in model_name.lower():
        return 'Anthropic'
    elif 'claude-opus-4' in model_name.lower() or 'opus-4' in model_name.lower():
        return 'Anthropic'
    
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
        "DeepMind": "deepmind.png"
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
        if batch_dir.endswith("_v4.1") and batch_dir != "batch_20250821_150540_v4.1":
            batch_path = os.path.join(data_dir, batch_dir)
            if os.path.isdir(batch_path):
                v4_1_batches.append(batch_path)
    
    if not v4_1_batches:
        print("No v4.1 batch directories found (excluding special GPT-4o vs GPT-5 batch).")
        return {}
    
    print(f"Found {len(v4_1_batches)} v4.1 batch directories")
    
    # Track results by model configuration
    config_results = defaultdict(lambda: {'evil_wins': 0, 'total_games': 0, 'model_configs': None})
    
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
                        if winner == "evil":
                            evil_wins += 1
                            
                except (json.JSONDecodeError, IOError, KeyError):
                    continue
        
        # Add to overall results
        config_results[config_key]['evil_wins'] += evil_wins
        config_results[config_key]['total_games'] += total_games
    
    return config_results

def group_results_by_mafioso_experiments(config_results):
    """Group results for mafioso-changing experiments (detective + villager background)"""
    background_groups = defaultdict(list)
    
    for config_key, results in config_results.items():
        model_configs = results['model_configs']
        if not model_configs:
            continue
        
        detective_model = extract_model_name(model_configs.get('detective', {}))
        villager_model = extract_model_name(model_configs.get('villager', {}))
        mafioso_model = extract_model_name(model_configs.get('mafioso', {}))
        
        # Create background key (detective + villager)
        background_key = f"{detective_model}_{villager_model}"
        
        evil_wins = results['evil_wins']
        total_games = results['total_games']
        if total_games > 0:
            win_rate = (evil_wins / total_games) * 100
            sem = calculate_sem(evil_wins, total_games)
            
            background_groups[background_key].append({
                'varying_model': mafioso_model,
                'win_rate': win_rate,
                'sem': sem,
                'games': total_games,
                'evil_wins': evil_wins
            })
    
    return background_groups

def group_results_by_detective_experiments(config_results):
    """Group results for detective-changing experiments (mafioso + villager background)"""
    background_groups = defaultdict(list)
    
    for config_key, results in config_results.items():
        model_configs = results['model_configs']
        if not model_configs:
            continue
        
        detective_model = extract_model_name(model_configs.get('detective', {}))
        villager_model = extract_model_name(model_configs.get('villager', {}))
        mafioso_model = extract_model_name(model_configs.get('mafioso', {}))
        
        # Create background key (mafioso + villager)
        background_key = f"{mafioso_model}_{villager_model}"
        
        evil_wins = results['evil_wins']
        total_games = results['total_games']
        if total_games > 0:
            good_wins = total_games - evil_wins
            win_rate = (good_wins / total_games) * 100  # Good win rate for detective experiments
            sem = calculate_sem(good_wins, total_games)
            
            background_groups[background_key].append({
                'varying_model': detective_model,
                'win_rate': win_rate,
                'sem': sem,
                'games': total_games,
                'good_wins': good_wins
            })
    
    return background_groups

def get_background_color(background_key):
    """Get color based on background model type"""
    if 'mistral' in background_key.lower():
        return '#FF6600'  # Mistral orange
    elif 'gpt-4o' in background_key.lower():
        return '#00A67E'  # GPT-4o green  
    elif 'gpt-4.1-mini' in background_key.lower() or 'gpt 4.1 mini' in background_key.lower():
        return '#10A37F'  # GPT-4.1-mini green
    elif 'grok' in background_key.lower():
        return '#FF6B35'  # Grok orange/red
    elif 'llama' in background_key.lower():
        return '#4A90E2'  # Llama blue
    elif 'qwen' in background_key.lower():
        return '#7B68EE'  # Qwen purple
    elif 'gemma' in background_key.lower():
        return '#FF4081'  # Gemma pink
    else:
        return '#666666'  # Default gray

def create_benchmark_plot(benchmark_data, title, filename, background_key="", use_good_wins=False):
    """Create a horizontal bar plot with logos and formatting"""
    # Use non-interactive backend
    plt.ioff()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    models = benchmark_data['models']
    values = benchmark_data['values'] 
    errors = benchmark_data['errors']
    companies = benchmark_data['companies']
    
    y_positions = range(len(models))
    
    # Get color based on background
    bar_color = get_background_color(background_key)
    bars = ax.barh(y_positions, values, xerr=errors, 
                   color=bar_color, alpha=0.8, height=0.6,
                   error_kw={'capsize': 5, 'capthick': 2})
    
    # Add model names inside the bars (white text)
    for i, (model, value) in enumerate(zip(models, values)):
        ax.text(value/2, i, model, ha='center', va='center', 
                fontweight='bold', fontsize=11, color='white')
    
    # Add values on the right side of bars
    for i, (value, error) in enumerate(zip(values, errors)):
        ax.text(value + error + 1.5, i, f'{value:.1f}% ± {error:.1f}%', 
                ha='left', va='center', fontweight='bold', fontsize=10)
    
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
                        fontweight='bold', fontsize=12, color='black',
                        bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgray'))
        else:
            # Fallback to company initial
            ax.text(-6, i, company[0], ha='center', va='center', 
                    fontweight='bold', fontsize=12, color='black',
                    bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgray'))
    
    # Set axis labels based on metric type
    if use_good_wins:
        xlabel = 'Good Win Rate (%)'
    else:
        xlabel = 'Mafia Win Rate (%)'
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
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
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Plot saved as {filename}")

def main():
    """Create dynamic benchmark plots from v4.1 batch data"""
    
    print("🔄 Analyzing v4.1 batch data...")
    config_results = analyze_v4_1_batch_data()
    
    if not config_results:
        print("❌ No v4.1 data found to create plots")
        return
    
    print("📊 Grouping v4.1 results by experiment type...")
    
    # Group results for both experiment types
    mafioso_groups = group_results_by_mafioso_experiments(config_results)
    detective_groups = group_results_by_detective_experiments(config_results)
    
    plots_created = 0
    
    # Create mafioso-changing experiment plots (evil win rate)
    print("\n🟡 Creating v4.1 mafioso-changing experiment plots...")
    for background_key, results_list in mafioso_groups.items():
        if len(results_list) < 2:  # Need at least 2 models to compare
            continue
        
        # Sort by win rate (descending)
        results_list.sort(key=lambda x: x['win_rate'], reverse=True)
        
        # Extract data for plotting
        models = [r['varying_model'] for r in results_list]
        values = [r['win_rate'] for r in results_list]
        errors = [r['sem'] for r in results_list]
        companies = [get_model_company(model) for model in models]
        
        # Create benchmark data structure
        benchmark_data = {
            'models': models,
            'values': values,
            'errors': errors,
            'companies': companies
        }
        
        # Create descriptive title and filename
        background_parts = background_key.split('_')
        detective_model = background_parts[0] if len(background_parts) > 0 else "Unknown"
        villager_model = background_parts[1] if len(background_parts) > 1 else detective_model
        
        if detective_model == villager_model:
            background_desc = f"{detective_model} Town"
        else:
            background_desc = f"{detective_model} Detective + {villager_model} Villager Town"
        
        title = f"Mafioso vs {background_desc}"
        filename = f"mafioso_{background_key.lower()}_v4_1_benchmark.png"
        
        print(f"📈 Creating v4.1 mafioso plot: {title}")
        print(f"   Models: {', '.join(models)}")
        print(f"   Sample sizes: {[r['games'] for r in results_list]}")
        
        create_benchmark_plot(benchmark_data, title, filename, background_key, use_good_wins=False)
        plots_created += 1
    
    # Create detective-changing experiment plots (good win rate)
    print("\n🔵 Creating v4.1 detective-changing experiment plots...")
    for background_key, results_list in detective_groups.items():
        if len(results_list) < 2:  # Need at least 2 models to compare
            continue
        
        # Sort by win rate (descending)
        results_list.sort(key=lambda x: x['win_rate'], reverse=True)
        
        # Extract data for plotting
        models = [r['varying_model'] for r in results_list]
        values = [r['win_rate'] for r in results_list]
        errors = [r['sem'] for r in results_list]
        companies = [get_model_company(model) for model in models]
        
        # Create benchmark data structure
        benchmark_data = {
            'models': models,
            'values': values,
            'errors': errors,
            'companies': companies
        }
        
        # Create descriptive title and filename
        background_parts = background_key.split('_')
        mafioso_model = background_parts[0] if len(background_parts) > 0 else "Unknown"
        villager_model = background_parts[1] if len(background_parts) > 1 else mafioso_model
        
        if mafioso_model == villager_model:
            background_desc = f"{mafioso_model} Mafia and Villager"
        else:
            background_desc = f"{mafioso_model} Mafia and {villager_model} Villager"
        
        title = f"Detective vs {background_desc}"
        filename = f"detective_{background_key.lower()}_v4_1_benchmark.png"
        
        print(f"📈 Creating v4.1 detective plot: {title}")
        print(f"   Models: {', '.join(models)}")
        print(f"   Sample sizes: {[r['games'] for r in results_list]}")
        
        create_benchmark_plot(benchmark_data, title, filename, background_key, use_good_wins=True)
        plots_created += 1
    
    print(f"\n✅ Created {plots_created} v4.1 benchmark plots!")
    print(f"   📊 Mafioso experiments: {len(mafioso_groups)} plots")
    print(f"   📊 Detective experiments: {len(detective_groups)} plots")
    print("\nNote: v4.1 plots are separate from v4.0 plots to maintain version isolation.")
    print("The special GPT-4o vs GPT-5 v4.1 batch is included in the v4.0 script for comparison.")

if __name__ == "__main__":
    main()