#!/usr/bin/env python3
"""
Model Performance Plots Generator

Creates 3 horizontal bar plots for model performance metrics:
1. Detective voting accuracy (% times detective voted for mafioso)
2. Remained silent rate (% messages that were "remained silent")
3. Random vote rate (% votes cast randomly due to format failures)

Uses the same visual style as benchmark plots with company logos.
"""

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# Import analysis functions from model_performance
from model_performance import (
    extract_model_name, get_batch_config, analyze_player_messages,
    analyze_player_votes, get_mafioso_and_detective, analyze_strategic_voting
)

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
    elif 'gemma' in model_name.lower():
        return 'DeepMind'
    elif 'deepseek' in model_name.lower():
        return 'DeepSeek'
    elif 'gemini' in model_name.lower():
        return 'Google'
    
    return company_mapping.get(model_name, 'Unknown')

def load_company_logo(company, size=(40, 40)):
    """Load actual company logos from the logos folder"""
    logo_files = {
        "OpenAI": "openai.png",
        "X": "xai.png",
        "Mistral AI": "mistral.png", 
        "Meta": "meta.png",
        "Alibaba": "baba.png",
        "Anthropic": "anthropic.png",
        "DeepMind": "deepmind.png",
        "DeepSeek": "deepseek.png",
        "Google": "deepmind.png"
    }
    
    try:
        logo_path = f"logos/{logo_files[company]}"
        if os.path.exists(logo_path):
            img = Image.open(logo_path)
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            img.thumbnail(size, Image.Resampling.LANCZOS)
            new_img = Image.new('RGBA', size, (255, 255, 255, 0))
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

def collect_model_performance_data():
    """Collect performance data for all benchmark models"""
    data_dir = "../data/batch"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found. Run from mini-mafia/results/ directory.")
        return {}
    
    # Find all v4.1 batch directories
    v4_batches = []
    for batch_dir in os.listdir(data_dir):
        if batch_dir.endswith("_v4.1"):
            batch_path = os.path.join(data_dir, batch_dir)
            if os.path.isdir(batch_path):
                v4_batches.append(batch_path)
    
    if not v4_batches:
        print("No v4.1 batch directories found.")
        return {}
    
    print(f"Collecting performance data from {len(v4_batches)} v4.1 batches...")
    
    # Track performance by model
    model_stats = defaultdict(lambda: {
        'total_messages': 0,
        'silent_messages': 0,
        'total_votes': 0,
        'random_votes': 0,
        'detective_votes_for_mafioso': 0,
        'detective_total_votes': 0,
        'mafioso_votes_for_detective': 0,
        'mafioso_total_votes': 0
    })
    
    for batch_path in sorted(v4_batches):
        config = get_batch_config(batch_path)
        if not config:
            continue
        
        model_configs = config.get('model_configs', {})
        role_to_model = {}
        for role, role_config in model_configs.items():
            model_name = extract_model_name(role_config)
            role_to_model[role] = model_name
        
        # Process all games in this batch
        for game_file in os.listdir(batch_path):
            if not (game_file.startswith('game_') and game_file.endswith('.json')):
                continue
                
            game_path = os.path.join(batch_path, game_file)
            try:
                with open(game_path, 'r') as f:
                    game_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            
            players = game_data.get('players', [])
            if not players:
                continue
            
            mafioso_name, detective_name = get_mafioso_and_detective(players)
            
            # Analyze each player
            for player in players:
                player_name = player['name']
                player_role = player['role']
                memory = player.get('memory', [])
                
                model_name = role_to_model.get(player_role, 'unknown')
                if model_name == 'unknown':
                    continue
                
                stats = model_stats[model_name]
                
                # Analyze messages
                total_messages, silent_messages = analyze_player_messages(memory, player_name)
                stats['total_messages'] += total_messages
                stats['silent_messages'] += silent_messages
                
                # Analyze votes
                total_votes, random_votes = analyze_player_votes(memory, player_name)
                stats['total_votes'] += total_votes
                stats['random_votes'] += random_votes
                
                # Analyze strategic voting
                strategic_votes, strategic_total = analyze_strategic_voting(
                    memory, player_name, player_role, mafioso_name, detective_name)
                
                if player_role == "detective":
                    stats['detective_votes_for_mafioso'] += strategic_votes
                    stats['detective_total_votes'] += strategic_total
                elif player_role == "mafioso":
                    stats['mafioso_votes_for_detective'] += strategic_votes
                    stats['mafioso_total_votes'] += strategic_total
    
    return model_stats

def create_performance_plot(models, values, errors, companies, title, xlabel, filename, x_max=100):
    """Create horizontal bar plot for performance metrics"""
    plt.ioff()
    
    # Set font size to match benchmark plots
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
    
    y_positions = range(len(models))
    
    # Create bars with neutral color
    bars = ax.barh(y_positions, values, xerr=errors, 
                   color='#4A90E2', alpha=0.8, height=0.6,
                   error_kw={'capsize': 5, 'capthick': 2})
    
    # Add model names and values on the right side of bars
    for i, (model, value, error) in enumerate(zip(models, values, errors)):
        ax.text(max(value + error + 1.5, 5), i, f'{model}: {value:.1f}%', 
                ha='left', va='center', fontweight='bold', fontsize=24)
    
    # Add company logos on the left
    logo_x_pos = -8
    for i, company in enumerate(companies):
        logo_img = load_company_logo(company, size=(40, 40))
        if logo_img is not None:
            try:
                imagebox = OffsetImage(logo_img, zoom=0.8)
                ab = AnnotationBbox(imagebox, (logo_x_pos, i), frameon=False, 
                                  xycoords='data', boxcoords="data")
                ax.add_artist(ab)
            except Exception as e:
                print(f"Failed to add logo for {company}: {e}")
                ax.text(logo_x_pos, i, company[0], ha='center', va='center', 
                        fontweight='bold', fontsize=24, color='black',
                        bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgray'))
        else:
            ax.text(logo_x_pos, i, company[0], ha='center', va='center', 
                    fontweight='bold', fontsize=24, color='black',
                    bbox=dict(boxstyle="circle,pad=0.3", facecolor='lightgray'))
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(-12, x_max)
    ax.set_xticks(np.arange(0, x_max+1, 20))
    
    # Add custom grid lines
    for x in np.arange(0, x_max+1, 20):
        ax.axvline(x=x, color='gray', alpha=0.3, linewidth=0.5)
    
    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Add custom bottom axis line
    ax.plot([0, x_max], [ax.get_ylim()[0], ax.get_ylim()[0]], color='black', linewidth=0.8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Plot saved as {filename}")

def main():
    """Generate model performance plots"""
    print("🔄 Collecting model performance data...")
    
    # Collect raw data
    model_stats = collect_model_performance_data()
    
    # Filter to benchmark models
    benchmark_models = {
        'Mistral 7B Instruct', 'GPT-4.1 Mini', 'Grok 3 Mini', 'DeepSeek V3',
        'Llama3.1 8B Instruct', 'Qwen2.5 7B Instruct', 'Gemini 2.5 Flash Lite',
        'Claude Opus 4.1', 'Claude Sonnet 4'
    }
    
    filtered_stats = {name: stats for name, stats in model_stats.items() 
                     if name in benchmark_models and stats['total_messages'] > 0}
    
    if not filtered_stats:
        print("❌ No benchmark model data found!")
        return
    
    print(f"📊 Found data for {len(filtered_stats)} benchmark models")
    
    # Calculate metrics for each model
    model_metrics = {}
    for model_name, stats in filtered_stats.items():
        silent_pct = (stats['silent_messages'] / stats['total_messages'] * 100) if stats['total_messages'] > 0 else 0
        random_pct = (stats['random_votes'] / stats['total_votes'] * 100) if stats['total_votes'] > 0 else 0
        det_acc_pct = (stats['detective_votes_for_mafioso'] / stats['detective_total_votes'] * 100) if stats['detective_total_votes'] > 0 else None
        
        model_metrics[model_name] = {
            'silent_pct': silent_pct,
            'random_pct': random_pct,
            'detective_acc_pct': det_acc_pct,
            'company': get_model_company(model_name)
        }
    
    # Create Plot 1: Detective Voting Accuracy (only models that played detective)
    detective_models = [(name, metrics) for name, metrics in model_metrics.items() 
                       if metrics['detective_acc_pct'] is not None]
    
    if detective_models:
        detective_models.sort(key=lambda x: x[1]['detective_acc_pct'], reverse=False)  # Ascending for top-at-top
        
        models = [x[0] for x in detective_models]
        values = [x[1]['detective_acc_pct'] for x in detective_models]
        errors = [0] * len(values)  # No error bars for now
        companies = [x[1]['company'] for x in detective_models]
        
        create_performance_plot(models, values, errors, companies,
                              "Detective Voting Accuracy",
                              "Detective Accuracy (%)",
                              "detective_accuracy_performance.png",
                              x_max=100)
    
    # Create Plot 2: Remained Silent Rate
    all_models = list(model_metrics.items())
    all_models.sort(key=lambda x: x[1]['silent_pct'], reverse=False)  # Ascending for top-at-top
    
    models = [x[0] for x in all_models]
    values = [x[1]['silent_pct'] for x in all_models]
    errors = [0] * len(values)
    companies = [x[1]['company'] for x in all_models]
    
    create_performance_plot(models, values, errors, companies,
                          "Remained Silent Rate",
                          "Remained Silent (%)",
                          "remained_silent_performance.png",
                          x_max=60)
    
    # Create Plot 3: Random Vote Rate
    all_models.sort(key=lambda x: x[1]['random_pct'], reverse=False)  # Ascending for top-at-top
    
    models = [x[0] for x in all_models]
    values = [x[1]['random_pct'] for x in all_models]
    errors = [0] * len(values)
    companies = [x[1]['company'] for x in all_models]
    
    create_performance_plot(models, values, errors, companies,
                          "Random Vote Rate",
                          "Random Votes (%)",
                          "random_vote_performance.png",
                          x_max=60)
    
    print("\n✅ Model performance plots created!")
    print("📁 Generated files:")
    print("   - detective_accuracy_performance.png")
    print("   - remained_silent_performance.png") 
    print("   - random_vote_performance.png")

if __name__ == "__main__":
    main()