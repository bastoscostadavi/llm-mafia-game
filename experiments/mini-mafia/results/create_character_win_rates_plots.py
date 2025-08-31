#!/usr/bin/env python3
"""
Character Win Rate Analysis

Analyzes win rates by character name (Alice, Bob, Charlie, Diana) using SQL queries
and creates a bar plot showing individual character performance.
"""

import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import math
import random

def analyze_character_win_rates(db_path='../database/mini_mafia.db'):
    """Analyze win rates by character name using SQL queries with filtering to match benchmark plots"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    try:
        # First, get all games with their configurations to apply filtering
        query = """
        SELECT 
            g.game_id,
            g.winner,
            gp.character_name,
            gp.role,
            gp.final_status,
            p.model_name,
            detective_p.model_name as detective_model,
            mafioso_p.model_name as mafioso_model,
            villager_p.model_name as villager_model
        FROM games g
        JOIN game_players gp ON g.game_id = gp.game_id
        LEFT JOIN players p ON gp.player_id = p.player_id
        LEFT JOIN (
            SELECT gp2.game_id, p2.model_name 
            FROM game_players gp2 
            JOIN players p2 ON gp2.player_id = p2.player_id 
            WHERE gp2.role = 'detective'
        ) detective_p ON g.game_id = detective_p.game_id
        LEFT JOIN (
            SELECT gp2.game_id, p2.model_name 
            FROM game_players gp2 
            JOIN players p2 ON gp2.player_id = p2.player_id 
            WHERE gp2.role = 'mafioso'
        ) mafioso_p ON g.game_id = mafioso_p.game_id
        LEFT JOIN (
            SELECT gp2.game_id, p2.model_name 
            FROM game_players gp2 
            JOIN players p2 ON gp2.player_id = p2.player_id 
            WHERE gp2.role = 'villager'
        ) villager_p ON g.game_id = villager_p.game_id
        WHERE gp.player_id IS NOT NULL  -- Exclude killed players
        ORDER BY g.game_id, gp.character_name
        """
        
        results = conn.execute(query).fetchall()
        
        # Group games by configuration and apply sampling for Mistral configurations
        games_by_config = {}
        character_data = []
        
        for row in results:
            game_id = row['game_id']
            detective_model = row['detective_model'] 
            mafioso_model = row['mafioso_model']
            villager_model = row['villager_model']
            
            if not all([detective_model, mafioso_model, villager_model]):
                continue
                
            # Create configuration key
            config_key = f"{detective_model}|{mafioso_model}|{villager_model}"
            
            if config_key not in games_by_config:
                games_by_config[config_key] = []
            
            games_by_config[config_key].append({
                'game_id': game_id,
                'winner': row['winner'],
                'character_name': row['character_name'],
                'role': row['role'],
                'final_status': row['final_status'],
                'player_model': row['model_name'],
                'detective_model': detective_model,
                'mafioso_model': mafioso_model,
                'villager_model': villager_model
            })
        
        # Apply sampling for Mistral configurations (limit to 100 games)
        random.seed(42)  # For reproducible sampling
        filtered_character_data = []
        excluded_games_info = []
        
        for config_key, config_games in games_by_config.items():
            detective_model, mafioso_model, villager_model = config_key.split('|')
            
            # Check if this is a Mistral configuration that needs sampling
            is_mistral_config = (
                'Mistral-7B-Instruct' in detective_model or
                'Mistral-7B-Instruct' in mafioso_model or  
                'Mistral-7B-Instruct' in villager_model
            )
            
            # Group by game_id to sample games (not individual character records)
            games_by_id = {}
            for record in config_games:
                game_id = record['game_id']
                if game_id not in games_by_id:
                    games_by_id[game_id] = []
                games_by_id[game_id].append(record)
            
            game_ids = list(games_by_id.keys())
            
            # Sample 100 games for Mistral configs, use all games for others
            if is_mistral_config and len(game_ids) > 100:
                sampled_game_ids = random.sample(game_ids, 100)
                excluded_count = len(game_ids) - len(sampled_game_ids)
                excluded_games_info.append({
                    'config': config_key,
                    'total_games': len(game_ids),
                    'used_games': len(sampled_game_ids),
                    'excluded_games': excluded_count
                })
                print(f"⚠️  Sampling {len(sampled_game_ids)} games from {len(game_ids)} for Mistral config: {config_key}")
            else:
                sampled_game_ids = game_ids
            
            # Add sampled games to filtered data
            for game_id in sampled_game_ids:
                filtered_character_data.extend(games_by_id[game_id])
        
        # Store excluded games info for reporting
        excluded_info = excluded_games_info
        
        # Now analyze the filtered data - only count players who actually played (not killed)
        character_stats = {}
        
        for record in filtered_character_data:
            char = record['character_name']
            winner = record['winner']
            role = record['role']
            final_status = record['final_status']
            player_model = record['player_model']
            
            # Skip killed players (they have player_model = None and never actually played)
            if player_model is None or final_status == 'killed':
                continue
                
            if char not in character_stats:
                character_stats[char] = {
                    'total_games': 0,
                    'total_wins': 0,
                    'times_killed': 0,
                    'times_arrested': 0,
                    'times_survived': 0,
                    'roles': {}
                }
            
            # Count total games (only for players who actually played)
            character_stats[char]['total_games'] += 1
            
            # Count wins (town wins when they're town, mafia wins when they're mafia)
            if (winner == 'town' and role in ['detective', 'villager']) or (winner == 'mafia' and role == 'mafioso'):
                character_stats[char]['total_wins'] += 1
            
            # Count final status (only for actual players)
            if final_status == 'arrested':
                character_stats[char]['times_arrested'] += 1
            elif final_status == 'alive':
                character_stats[char]['times_survived'] += 1
            # Note: times_killed should always be 0 since we're excluding killed players
            
            # Count roles (only for actual players)
            if role not in character_stats[char]['roles']:
                character_stats[char]['roles'][role] = 0
            character_stats[char]['roles'][role] += 1
        
        # Calculate win rates and standard errors
        for char in character_stats:
            stats = character_stats[char]
            total_games = stats['total_games']
            total_wins = stats['total_wins']
            
            if total_games > 0:
                win_rate = (total_wins / total_games) * 100
                p = total_wins / total_games
                sem = math.sqrt(p * (1 - p) / total_games) * 100  # Convert to percentage
            else:
                win_rate = 0
                sem = 0
                
            stats['win_rate'] = win_rate
            stats['sem'] = sem
        
        return character_stats, excluded_info
        
    finally:
        conn.close()

def create_gender_bias_plot(character_stats, filename='character_gender_bias.png'):
    """Create a horizontal bar plot showing gender bias using benchmark template"""
    
    # Use non-interactive backend
    plt.ioff()
    
    # Set font sizes to match benchmark plots
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 14,
        'figure.titlesize': 24
    })
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract data and organize by gender
    male_chars = ['Bob', 'Charlie']
    female_chars = ['Alice', 'Diana']
    
    # Calculate overall mean and pooled standard deviation
    all_win_rates = [character_stats[char]['win_rate'] for char in character_stats.keys()]
    all_games = [character_stats[char]['total_games'] for char in character_stats.keys()]
    all_wins = [character_stats[char]['total_wins'] for char in character_stats.keys()]
    
    # Calculate pooled statistics for standardization
    total_games = sum(all_games)
    total_wins = sum(all_wins)
    overall_win_rate = (total_wins / total_games) * 100
    
    # Calculate pooled standard error (approximate population standard deviation)
    p_pooled = total_wins / total_games
    pooled_se = math.sqrt(p_pooled * (1 - p_pooled) / (total_games / len(all_games))) * 100
    
    # Calculate deviations in terms of standard errors
    char_data = []
    for char in ['Alice', 'Bob', 'Charlie', 'Diana']:
        win_rate = character_stats[char]['win_rate']
        deviation_se = (win_rate - overall_win_rate) / pooled_se
        gender = 'Male' if char in male_chars else 'Female'
        char_data.append({
            'name': char,
            'win_rate': win_rate,
            'deviation_se': deviation_se,
            'gender': gender,
            'sem': character_stats[char]['sem'],
            'total_games': character_stats[char]['total_games']
        })
    
    # Sort by win rate (lowest first, same order as other plot) - Bob at TOP  
    char_data.sort(key=lambda x: x['win_rate'], reverse=False)
    
    # Prepare plot data
    names = [d['name'] for d in char_data]
    deviations = [d['deviation_se'] for d in char_data]
    colors = ['#FF6B6B' if d['gender'] == 'Female' else '#45B7D1' for d in char_data]
    win_rates = [d['win_rate'] for d in char_data]
    errors_normalized = [d['sem'] / pooled_se for d in char_data]
    total_games = [d['total_games'] for d in char_data]
    
    # Create horizontal bars with error bars (match benchmark style)
    y_positions = range(len(names))
    bars = ax.barh(y_positions, deviations, xerr=errors_normalized, color=colors, alpha=0.8, height=0.6,
                   error_kw={'capsize': 5, 'capthick': 2, 'ecolor': 'black'})
    
    # No symbols needed for character names (unlike company logos)
    
    # Add character names and deviation values to the right of ERROR BARS (benchmark style)  
    for i, (name, dev, rate, error_norm) in enumerate(zip(names, deviations, win_rates, errors_normalized)):
        # Use the same positioning logic as create_aggregated_zscore_plots
        # Position text at max(bar_end + error + gap, minimum_position)
        x_pos = max(dev + error_norm + 0.1, 0.1)
        # Show deviation in standard deviations, not percentages
        ax.text(x_pos, i, f'{name}: {dev:+.1f} ± {error_norm:.1f}', 
                ha='left', va='center', fontweight='bold', fontsize=24)
    
    # Set y-axis labels (but make them invisible since we show symbols)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([''] * len(names))  # Empty labels
    
    # Formatting to match benchmark plots
    ax.set_xlabel('Trust Score', fontsize=24, fontweight='bold')
    ax.set_xlim(-2.5, 2.5)  # Show range from -2.5σ to +2.5σ
    
    # Set x-axis ticks
    ax.set_xticks([-2, -1, 0, 1, 2])
    
    # Add custom grid lines (like benchmark)
    for x in [-2, -1, 0, 1, 2]:
        ax.axvline(x=x, color='gray', alpha=0.3, linewidth=0.5)
    
    # Remove spines to match benchmark style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Add custom bottom axis line (like benchmark)  
    ax.plot([-2.5, 2.5], [ax.get_ylim()[0], ax.get_ylim()[0]], color='black', linewidth=0.8)
    
    # No legend - game counts will be in LaTeX caption
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Gender bias analysis plot saved as {filename}")

def create_character_win_rate_plot(character_stats, filename='character_win_rates.png'):
    """Create a horizontal bar plot showing win rates by character name using benchmark template"""
    
    # Use non-interactive backend
    plt.ioff()
    
    # Set font sizes to match benchmark plots
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 14,
        'figure.titlesize': 24
    })
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort characters by performance (best to worst)
    char_data = []
    for char in ['Alice', 'Bob', 'Charlie', 'Diana']:
        char_data.append({
            'name': char,
            'win_rate': character_stats[char]['win_rate'],
            'sem': character_stats[char]['sem'],
            'total_games': character_stats[char]['total_games'],
            'gender': 'Male' if char in ['Bob', 'Charlie'] else 'Female'
        })
    
    # Sort by win rate (lowest first) - because matplotlib puts LAST item at TOP
    char_data.sort(key=lambda x: x['win_rate'], reverse=False)
    
    # Extract sorted data
    characters = [d['name'] for d in char_data]
    win_rates = [d['win_rate'] for d in char_data]
    errors = [d['sem'] for d in char_data]
    total_games = [d['total_games'] for d in char_data]
    genders = [d['gender'] for d in char_data]
    
    # Character colors - red for women, blue for men
    colors = ['#FF6B6B' if gender == 'Female' else '#45B7D1' for gender in genders]
    
    # Create horizontal bars with error bars (match benchmark style)
    y_positions = range(len(characters))
    bars = ax.barh(y_positions, win_rates, xerr=errors, color=colors, alpha=0.8, height=0.6,
                   error_kw={'capsize': 5, 'capthick': 2, 'ecolor': 'black'})
    
    # No symbols needed for character names (unlike company logos)
    
    # Add character names and values to the right of error bars (benchmark style)
    for i, (char, rate, error, games) in enumerate(zip(characters, win_rates, errors, total_games)):
        ax.text(rate + error + 2, i, f'{char}: {rate:.1f}% ± {error:.1f}%', 
                ha='left', va='center', fontweight='bold', fontsize=24)
    
    # Set y-axis labels (but make them invisible since we show symbols)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([''] * len(characters))  # Empty labels
    
    # Formatting to match benchmark plots
    ax.set_xlabel('Win Rate (%)', fontsize=24, fontweight='bold')
    ax.set_xlim(0, 100)  # Set range from 0 to 100
    
    # Set x-axis ticks to match benchmark style
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    
    # Add custom grid lines only for positive values (like benchmark)
    for x in [0, 20, 40, 60, 80, 100]:
        ax.axvline(x=x, color='gray', alpha=0.3, linewidth=0.5)
    
    # Remove spines to match benchmark style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Add custom bottom axis line only from 0 to 100 (like benchmark)
    ax.plot([0, 100], [ax.get_ylim()[0], ax.get_ylim()[0]], color='black', linewidth=0.8)
    
    # No legend - game counts will be in LaTeX caption
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Character win rate plot saved as {filename}")

def print_character_summary(character_stats, excluded_info):
    """Print detailed character statistics"""
    print("CHARACTER WIN RATE ANALYSIS")
    print("=" * 80)
    
    # Print excluded models information
    if excluded_info:
        print("EXCLUDED GAMES FOR CONSISTENCY WITH BENCHMARK PLOTS:")
        print("-" * 80)
        for info in excluded_info:
            config_parts = info['config'].split('|')
            detective_model = config_parts[0].replace('Mistral-7B-Instruct-v0.2-Q4_K_M.gguf', 'Mistral 7B Instruct')
            mafioso_model = config_parts[1].replace('Mistral-7B-Instruct-v0.2-Q4_K_M.gguf', 'Mistral 7B Instruct') 
            villager_model = config_parts[2].replace('Mistral-7B-Instruct-v0.2-Q4_K_M.gguf', 'Mistral 7B Instruct')
            print(f"  Configuration: Detective={detective_model}, Mafioso={mafioso_model}, Villager={villager_model}")
            print(f"  Excluded {info['excluded_games']} games (used {info['used_games']}/{info['total_games']})")
        print("=" * 80)
    
    for char in sorted(character_stats.keys()):
        stats = character_stats[char]
        print(f"\n{char.upper()}:")
        print(f"  Total games: {stats['total_games']:,}")
        print(f"  Total wins: {stats['total_wins']:,}")
        print(f"  Win rate: {stats['win_rate']:.1f}% ± {stats['sem']:.1f}%")
        print(f"  Final status distribution:")
        print(f"    Killed: {stats['times_killed']:,} ({stats['times_killed']/stats['total_games']*100:.1f}%)")
        print(f"    Arrested: {stats['times_arrested']:,} ({stats['times_arrested']/stats['total_games']*100:.1f}%)")
        print(f"    Survived: {stats['times_survived']:,} ({stats['times_survived']/stats['total_games']*100:.1f}%)")
        
        if 'roles' in stats:
            print(f"  Role distribution:")
            for role, count in sorted(stats['roles'].items()):
                print(f"    {role}: {count:,} ({count/stats['total_games']*100:.1f}%)")
    
    print("\n" + "=" * 80)

def main():
    """Main function to run character win rate analysis"""
    print("🔄 Analyzing character win rates from database...")
    
    character_stats, excluded_info = analyze_character_win_rates()
    
    if not character_stats:
        print("❌ No character data found in database")
        return
    
    print_character_summary(character_stats, excluded_info)
    create_character_win_rate_plot(character_stats)
    create_gender_bias_plot(character_stats)
    
    print(f"\n✅ Character analysis complete!")

if __name__ == "__main__":
    main()