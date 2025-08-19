#!/usr/bin/env python3
"""
Dynamic LLM Mafia Benchmark Plot Generator

Creates horizontal bar plots for LLM Mafia benchmark results using the latest batch data.
Automatically reads from v4.0 batch results and groups by background models.
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

# Import analysis functions from analyze_results2
import sys
sys.path.append('../analysis')
from analyze_results2 import (
    extract_model_name, create_config_key, determine_winner, 
    get_batch_config, calculate_sem
)

def get_model_company(model_name):
    """Map model names to their companies"""
    company_mapping = {
        'GPT-4o': 'OpenAI', 
        'GPT-5': 'OpenAI',
        'GPT-3.5': 'OpenAI',
        'Claude-3-Haiku': 'Anthropic',
        'Claude-3-Sonnet': 'Anthropic',
        'Claude-3-Opus': 'Anthropic',
        'Claude-Sonnet-4': 'Anthropic',
        'Mistral 7B Instruct': 'Mistral AI',
        'Llama3.1 8B Instruct': 'Meta',
        'Qwen2.5 7B Instruct': 'Alibaba',
        'GPT-OSS': 'OpenAI'
    }
    
    # Handle Claude Sonnet-4 variations
    if 'claude-sonnet-4' in model_name.lower() or 'sonnet-4' in model_name.lower():
        return 'Anthropic'
    
    return company_mapping.get(model_name, 'Unknown')

def get_company_color(company):
    """Get brand colors for companies"""
    colors = {
        'OpenAI': '#10A37F',
        'Anthropic': '#DE7C3A', 
        'Mistral AI': '#FF6B35',
        'Meta': '#1877F2',
        'Alibaba': '#FF6A00',
        'Unknown': '#666666'
    }
    return colors.get(company, '#666666')

def load_company_logo(company, size=(40, 40)):
    """Load actual company logos from the logos folder"""
    # Map company names to logo filenames  
    logo_files = {
        "OpenAI": "openai.png",
        "Mistral AI": "mistral-color.png", 
        "Meta": "meta-logo-6760788.png",
        "Alibaba": "BABA.png",
        "Anthropic": "Anthropic.png"  # Correct filename with capital A
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

def analyze_latest_batch_data():
    """Analyze the latest v4.0 batch data and return organized results"""
    data_dir = "../data/batch"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found. Run from mini-mafia/results/ directory.")
        return {}
    
    # Find all v4.0 batch directories
    v4_batches = []
    for batch_dir in os.listdir(data_dir):
        if batch_dir.endswith("_v4.0"):
            batch_path = os.path.join(data_dir, batch_dir)
            if os.path.isdir(batch_path):
                v4_batches.append(batch_path)
    
    if not v4_batches:
        print("No v4.0 batch directories found.")
        return {}
    
    print(f"Found {len(v4_batches)} v4.0 batch directories")
    
    # Track results by model configuration
    config_results = defaultdict(lambda: {'evil_wins': 0, 'total_games': 0, 'model_configs': None})
    
    for batch_path in sorted(v4_batches):
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

def group_results_by_background(config_results):
    """Group results by background models (detective + villager)"""
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
                'mafioso_model': mafioso_model,
                'win_rate': win_rate,
                'sem': sem,
                'games': total_games,
                'evil_wins': evil_wins
            })
    
    return background_groups

def create_benchmark_plot(benchmark_data, title, filename):
    """Create a horizontal bar plot with logos and formatting"""
    # Use non-interactive backend
    plt.ioff()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    models = benchmark_data['models']
    values = benchmark_data['values'] 
    errors = benchmark_data['errors']
    companies = benchmark_data['companies']
    
    y_positions = range(len(models))
    
    # Create horizontal bars (all same color)
    bar_color = '#4A90E2'  # Nice blue color for all bars
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
    
    # Add reference lines
    baseline_1 = 5/12 * 100  # No-information exchange protocol ≈ 41.67%
    baseline_2 = 2/3 * 100   # Random voting ≈ 66.67%
    
    ax.axvline(x=baseline_1, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(x=baseline_2, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add legends with dotted lines
    legend_x = 65  # Position for legends
    
    # First legend (red line)
    legend_y1 = len(models) - 0.3
    ax.plot([legend_x - 8, legend_x - 2], [legend_y1, legend_y1], 
            color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(legend_x, legend_y1, 
            f'Random voting but detective votes mafioso ({baseline_1:.1f}%)', 
            ha='left', va='center', color='red', fontweight='bold', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='red'))
    
    # Second legend (blue line)  
    legend_y2 = len(models) - 0.5
    ax.plot([legend_x - 8, legend_x - 2], [legend_y2, legend_y2], 
            color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(legend_x, legend_y2, 
            f'Random voting ({baseline_2:.1f}%)', 
            ha='left', va='center', color='blue', fontweight='bold', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='blue'))
    
    # Formatting
    ax.set_xlabel('Evil Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_yticks([])  # Remove y-axis labels
    ax.set_xlim(-10, 100)  # Set range from -10 (for logos) to 100 (full domain)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Plot saved as {filename}")

def main():
    """Create dynamic benchmark plots from latest batch data"""
    
    print("🔄 Analyzing latest v4.0 batch data...")
    config_results = analyze_latest_batch_data()
    
    if not config_results:
        print("❌ No data found to create plots")
        return
    
    print("📊 Grouping results by background models...")
    background_groups = group_results_by_background(config_results)
    
    plots_created = 0
    
    for background_key, results_list in background_groups.items():
        if len(results_list) < 2:  # Need at least 2 models to compare
            continue
        
        # Sort by win rate (descending)
        results_list.sort(key=lambda x: x['win_rate'], reverse=True)
        
        # Extract data for plotting
        models = [r['mafioso_model'] for r in results_list]
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
            background_desc = f"{detective_model} Background"
        else:
            background_desc = f"{detective_model} Detective + {villager_model} Villager Background"
        
        title = f"Models as Mafioso vs {background_desc}"
        filename = f"{background_key.lower()}_benchmark_dynamic.png"
        
        print(f"📈 Creating plot: {title}")
        print(f"   Models: {', '.join(models)}")
        print(f"   Sample sizes: {[r['games'] for r in results_list]}")
        
        create_benchmark_plot(benchmark_data, title, filename)
        plots_created += 1
    
    print(f"\n✅ Created {plots_created} dynamic benchmark plots!")
    print("\nNote: Plots are automatically updated with the latest batch data.")
    print("Including Claude Sonnet-4 support when data becomes available.")

if __name__ == "__main__":
    main()