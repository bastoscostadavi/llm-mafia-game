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
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preset_games import mini_mafia_game
from src.agents.agent import MafiaAgent
from src.prompts import PromptConfig, get_default_prompt_config

def save_game_data(game, game_num, batch_id, batch_dir, prompt_config):
    """Save game data to JSON file in batch folder"""
    
    # Extract game state and results
    game_data = {
        "game_id": f"{batch_id}_game_{game_num:04d}",
        "batch_id": batch_id,
        "game_number": game_num,
        "timestamp": datetime.now().isoformat(),
        
        # Initial setup
        "initial_players": [
            {
                "name": agent.name,
                "role": agent.role,
                "alive": agent.alive,
                "imprisoned": agent.imprisoned
            }
            for agent in game.state.agents
        ],
        
        # Game outcome
        "final_state": [
            {
                "name": agent.name,
                "role": agent.role,
                "alive": agent.alive,
                "imprisoned": agent.imprisoned,
                "memory": agent.memory
            }
            for agent in game.state.agents
        ],
        
        # Winner analysis
        "winner": determine_winner(game.state.agents),
        "arrested_player": get_arrested_player(game.state.agents),
        "dead_player": get_dead_player(game.state.agents),
        
        # Game metadata
        "total_rounds": game.state.round,
        "discussion_rounds": game.state.discussion_rounds,
        
        # Prompt configuration used for this game
        "prompt_config": prompt_config.get_config_dict()
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
    base_dir = Path(__file__).parent / "data" / "mini_mafia"
    batch_dir = base_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    return batch_dir

def save_prompt_config(prompt_config, batch_dir):
    """Save prompt configuration to batch folder"""
    prompt_config_file = batch_dir / "prompt_config.json"
    prompt_config.save_to_file(prompt_config_file)
    
    return prompt_config_file

def run_batch(n_games, debug_prompts=False, prompt_config=None):
    """Run N mini-mafia games and save results"""
    
    # Use default prompt config if none provided
    if prompt_config is None:
        prompt_config = get_default_prompt_config()
    
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{prompt_config.version}"
    print(f"Starting batch: {batch_id}")
    print(f"Running {n_games} mini-mafia games with prompt version {prompt_config.version}...")
    
    # Create batch folder
    batch_dir = create_batch_folder(batch_id)
    print(f"Batch folder created: {batch_dir}")
    
    results = []
    stats = {"good_wins": 0, "evil_wins": 0, "unknown": 0}
    
    # Save prompt configuration for this batch
    prompt_config_file = save_prompt_config(prompt_config, batch_dir)
    print(f"Prompt config saved to: {prompt_config_file}")
    
    for i in range(n_games):
        print(f"\nGame {i+1}/{n_games}")
        
        # Create and run game with specific prompt config
        game = mini_mafia_game(debug_prompts=debug_prompts, prompt_config=prompt_config)
        
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
        
        # Save game data
        game_data = save_game_data(game, i, batch_id, batch_dir, prompt_config)
        results.append(game_data)
        
        # Update stats
        winner = game_data["winner"]
        if winner == "good":
            stats["good_wins"] += 1
        elif winner == "evil":
            stats["evil_wins"] += 1
        else:
            stats["unknown"] += 1
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1} games. Current stats: {stats}")
    
    # Save batch summary
    summary = {
        "batch_id": batch_id,
        "timestamp": datetime.now().isoformat(),
        "total_games": n_games,
        "statistics": stats,
        "win_rates": {
            "good": stats["good_wins"] / n_games,
            "evil": stats["evil_wins"] / n_games,
            "unknown": stats["unknown"] / n_games
        },
        "configuration": {
            "game_type": "mini_mafia",
            "debug_prompts": debug_prompts,
            "model": "local mistral.gguf",
            "prompt_version": prompt_config.version
        },
        "batch_folder": str(batch_dir),
        "prompt_config_file": str(prompt_config_file)
    }
    
    # Save summary file in batch folder
    summary_file = batch_dir / "batch_summary.json"
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"BATCH COMPLETE: {batch_id}")
    print(f"{'='*50}")
    print(f"Total games: {n_games}")
    print(f"Good wins: {stats['good_wins']} ({stats['good_wins']/n_games:.1%})")
    print(f"Evil wins: {stats['evil_wins']} ({stats['evil_wins']/n_games:.1%})")
    print(f"Unknown: {stats['unknown']} ({stats['unknown']/n_games:.1%})")
    print(f"\nResults saved to: {batch_dir}")
    print(f"Use: python game_viewer.py to analyze results")
    
    return batch_id, summary

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Run N mini-mafia games and save results')
    parser.add_argument('n_games', type=int, help='Number of games to run')
    parser.add_argument('--debug', action='store_true', help='Show LLM prompts')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode with prompts')
    parser.add_argument('--prompt-version', default='v1.0', help='Prompt version to use (default: v1.0)')
    
    args = parser.parse_args()
    
    # Create prompt config
    prompt_config = PromptConfig(version=args.prompt_version)
    
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
            print(f"  Model: Local Qwen2.5 7B")
            
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
    print(f"  Model: Local Qwen2.5 7B")
    
    try:
        # Run the batch
        batch_id, summary = run_batch(n_games, debug_prompts=debug, prompt_config=prompt_config)
        
    except KeyboardInterrupt:
        print("\n\nBatch interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()