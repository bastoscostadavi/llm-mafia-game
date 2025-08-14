#!/usr/bin/env python3
"""
Mini-Mafia Batch Runner

Runs N mini-mafia games and saves the results for analysis.
Each game follows the same format as preset_games.mini_mafia_game():
- 4 players initially (Alice, Bob, Charlie, Diana with random roles)
- 1 villager killed randomly before day phase
- Detective and mafioso know each other
- Single day phase to determine winner
"""

import sys
import os
import json
import random
import io
import contextlib
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from preset_games import mini_mafia_game
from src.agents import MafiaAgent
from src.prompts import PromptConfig
from src.config import DEFAULT_PROMPT_VERSION, get_default_model_configs, get_default_prompt_config

# Import centralized model configs - no local definition needed

#{
#        'detective': {'type': 'openai', 'model': 'gpt-4o', 'temperature': 0.7},
#        'mafioso': {'type': 'openai', 'model': 'gpt-4o', 'temperature': 0.7},
#        'villager': {'type': 'openai', 'model': 'gpt-4o', 'temperature': 0.7}
#    }

def save_game_data(game, game_num, batch_id, batch_dir, prompt_config, model_configs=None):
    """Save minimal game data to JSON file in batch folder"""
    
    # Minimal game data - just memories and essential info
    game_data = {
        "game_id": f"{batch_id}_game_{game_num:04d}",
        "game_number": game_num,
        "timestamp": datetime.now().isoformat(),
        
        # Only save player memories and final status
        "players": [
            {
                "name": agent.name,
                "role": agent.role,
                "alive": agent.alive,
                "imprisoned": agent.imprisoned,
                "memory": agent.memory
            }
            for agent in game.state.agents
        ]
    }
    
    # Save to batch folder
    filename = f"game_{game_num:04d}.json"
    filepath = batch_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(game_data, f, indent=2)
    
    return game_data

def determine_winner(agents):
    """Determine who won the game"""
    arrested = next((a for a in agents if a.imprisoned), None)
    if arrested:
        if arrested.role == "mafioso":
            return "good"
        else:  # villager or detective arrested
            return "evil"
    return "unknown"

def get_arrested_player(agents):
    """Get the name and role of arrested player"""
    arrested = next((a for a in agents if a.imprisoned), None)
    if arrested:
        return {"name": arrested.name, "role": arrested.role}
    return None

def get_dead_player(agents):
    """Get the name and role of dead player"""
    dead = next((a for a in agents if not a.alive), None)
    if dead:
        return {"name": dead.name, "role": dead.role}
    return None

def create_batch_folder(batch_id):
    """Create batch folder and return paths"""
    base_dir = Path(__file__).parent / "data"
    batch_dir = base_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    return batch_dir

def save_batch_config(prompt_config, model_configs, batch_dir, batch_id):
    """Save batch configuration (prompt + model configs) to batch folder"""
    config_data = {
        "batch_id": batch_id,
        "timestamp": datetime.now().isoformat(),
        "prompt_config": prompt_config.get_config_dict(),
        "model_configs": model_configs or get_default_model_configs(),
        "game_type": "mini_mafia"
    }
    
    config_file = batch_dir / "batch_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    return config_file

def run_batch(n_games, debug_prompts=False, prompt_config=None, model_configs=None, temperature=None):
    """Run N mini-mafia games and save results"""
    
    # Use default model configs if none provided
    if model_configs is None:
        model_configs = get_default_model_configs()
    
    # Include temperature in batch ID if specified
    temp_suffix = f"_temp{temperature}" if temperature is not None else ""
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{prompt_config.version}{temp_suffix}"
    print(f"Starting batch: {batch_id}")
    print(f"Running {n_games} mini-mafia games with prompt version {prompt_config.version}...")
    
    # Create batch folder
    batch_dir = create_batch_folder(batch_id)
    print(f"Batch folder created: {batch_dir}")
    
    results = []
    stats = {"good_wins": 0, "evil_wins": 0, "unknown": 0}
    
    # Save batch configuration (prompt + model configs) once
    config_file = save_batch_config(prompt_config, model_configs, batch_dir, batch_id)
    print(f"Batch config saved to: {config_file}")
    
    for i in range(n_games):
        print(f"\nGame {i+1}/{n_games}")
        
        # Create and run game with specific prompt config and model configs
        game = mini_mafia_game(debug_prompts=debug_prompts, model_configs=model_configs, prompt_config=prompt_config)
        
        # Capture stdout to avoid cluttering output
        if not debug_prompts:
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            try:
                game.play()
            finally:
                sys.stdout = old_stdout
        else:
            game.play()
        
        # Save minimal game data
        game_data = save_game_data(game, i, batch_id, batch_dir, prompt_config, model_configs)
        results.append(game_data)
        
        # Update stats (determine winner from game state)
        winner = determine_winner(game.state.agents)
        if winner == "good":
            stats["good_wins"] += 1
        elif winner == "evil":
            stats["evil_wins"] += 1
        else:
            stats["unknown"] += 1
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1} games. Current stats: {stats}")
    
    
    print(f"\n{'='*50}")
    print(f"BATCH COMPLETE: {batch_id}")
    print(f"{'='*50}")
    print(f"Total games: {n_games}")
    print(f"Good wins: {stats['good_wins']} ({stats['good_wins']/n_games:.1%})")
    print(f"Evil wins: {stats['evil_wins']} ({stats['evil_wins']/n_games:.1%})")
    print(f"Unknown: {stats['unknown']} ({stats['unknown']/n_games:.1%})")
    print(f"\nResults saved to: {batch_dir}")
    print(f"Use: python analyze_voting.py to analyze results")
    
    return batch_id

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Run N mini-mafia games and save results')
    parser.add_argument('n_games', type=int, help='Number of games to run')
    parser.add_argument('--debug', action='store_true', help='Show LLM prompts')
    parser.add_argument('--debug-responses', action='store_true', help='Show raw model responses when parsing fails')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode with prompts')
    parser.add_argument('--prompt-version', default=DEFAULT_PROMPT_VERSION, help=f'Prompt version to use (default: {DEFAULT_PROMPT_VERSION} with caching)')
    parser.add_argument('--temperature', type=float, help='Temperature for all models (default: 0.7)')
    
    args = parser.parse_args()
    
    # Create prompt config using system default
    prompt_config = get_default_prompt_config()
    
    print("Mini-Mafia Batch Runner")
    print("="*40)
    
    if args.interactive:
        try:
            n_games = int(input("Number of games to run: "))
            debug = input("Show LLM prompts? (y/n) [n]: ").strip().lower() == 'y'
            
            if n_games <= 0:
                print("Number of games must be positive")
                return
            
            print(f"\nConfiguration:")
            print(f"  Games: {n_games}")
            print(f"  Debug prompts: {debug}")
            print(f"  Model: GPT-5 (with prompt caching)")
            
            confirm = input("\nProceed? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Cancelled.")
                return
            
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            return
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            return
    else:
        n_games = args.n_games
        debug = args.debug
    
    if n_games <= 0:
        print("Number of games must be positive")
        return
    
    print(f"\nConfiguration:")
    print(f"  Games: {n_games}")
    print(f"  Debug prompts: {debug}")
    print(f"  Model: GPT-5 (with prompt caching)")
    
    try:
        # Run the batch
        batch_id = run_batch(n_games, debug_prompts=debug, prompt_config=prompt_config, temperature=args.temperature)
        
    except KeyboardInterrupt:
        print("\n\nBatch interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()