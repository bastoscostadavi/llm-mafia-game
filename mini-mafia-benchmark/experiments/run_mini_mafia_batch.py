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
import io
import argparse
from datetime import datetime
from pathlib import Path

# Add repo root and script directory to path before site-packages.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (REPO_ROOT, SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from mini_mafia import create_mini_mafia_game
from src.config import get_default_model_configs
from database.db_utils import MiniMafiaDB


def extract_model_info(model_config):
    """Extract standardized model information from config."""
    if not model_config:
        return 'llm', 'unknown', 'unknown', 0.7
        
    model_type = model_config.get('type', 'unknown')
    temperature = model_config.get('temperature', 0.7)
    
    # Map different config types to standardized names
    if model_type == 'openai':
        model = model_config.get('model', 'unknown')
        return 'llm', model, 'openai', temperature
    elif model_type == 'anthropic':
        model = model_config.get('model', 'unknown')
        return 'llm', model, 'anthropic', temperature
    elif model_type == 'xai':
        model = model_config.get('model', 'unknown')
        return 'llm', model, 'xai', temperature
    elif model_type == 'deepseek':
        model = model_config.get('model', 'deepseek-chat')
        return 'llm', model, 'deepseek', temperature
    elif model_type == 'google':
        model = model_config.get('model', 'unknown')
        return 'llm', model, 'google', temperature
    elif model_type == 'local':
        model_name = model_config.get('model')
        if not model_name:
            raise ValueError(f"Local model config missing 'model' field: {model_config}")
        return 'llm', model_name, 'local', temperature
    else:
        # Fallback
        model = model_config.get('model', model_type)
        return 'llm', model, model_type, temperature

def save_game_data_to_db(game, game_num, batch_id, db, player_ids):
    """Save game data directly to SQLite database"""
    
    # Generate game_id using batch_id as prefix: BATCH_ID_NNNN
    game_id = f"{batch_id}_{game_num:04d}"
    timestamp = datetime.now().isoformat()
    
    # Determine winner and final status
    winner = determine_winner(game.state.agents)
    
    # Insert game record
    db.insert_game(game_id, batch_id, game_num, timestamp, winner)
    
    # Insert game-player assignments and track characters
    character_roles = {}
    for agent in game.state.agents:
        character_name = agent.name
        role = agent.role
        final_status = 'arrested' if agent.imprisoned else ('killed' if not agent.alive else 'alive')
        
        player_id = player_ids.get(role)
        if player_id:
            db.insert_game_player(game_id, player_id, character_name, role, final_status)
            character_roles[character_name] = role
    
    # Save game sequence directly to database
    for log_entry in game.state.game_sequence:
        db.insert_game_sequence(
            game_id=game_id,
            step=log_entry['step'],
            action=log_entry['action'],
            actor=log_entry['actor'],
            raw_response=log_entry.get('raw_response'),
            parsed_result=log_entry.get('parsed_result')
        )
    
    return game_id


def determine_winner(agents):
    """Determine who won the game"""
    arrested = next((a for a in agents if a.imprisoned), None)
    if arrested:
        if arrested.role == "mafioso":
            return "good"
        else:  # villager or detective arrested
            return "evil"
    return "unknown"

MAX_GAME_RETRIES = 3


def run_batch(n_games, debug_prompts=False, model_configs=None, temperature=None, db_path=None):
    """Run N mini-mafia games and save results to SQLite database."""
    
    # Use default model configs if none provided
    if model_configs is None:
        model_configs = get_default_model_configs()
    
    # Use same format as game_id: YYYYMMDD_HHMMSS
    batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Include temperature suffix if specified
    if temperature is not None:
        batch_id = f"{batch_id}_temp{temperature}"
    print(f"Starting batch: {batch_id}")
    print(f"Running {n_games} mini-mafia games...")
    
    # Initialize database connection
    db = MiniMafiaDB(db_path=db_path) if db_path else MiniMafiaDB()
    db.connect()
    
    # Insert batch record
    timestamp = datetime.now().isoformat()
    
    db.insert_batch(
        batch_id=batch_id,
        timestamp=timestamp, 
        model_configs=model_configs,
        games_planned=n_games
    )
    print(f"Batch record created: {batch_id}")
    print(f"Games planned: {n_games}")
    
    # Create player records for this batch's model configurations
    player_ids = {}
    for role, config in model_configs.items():
        player_type, model_name, model_provider, temp = extract_model_info(config)
        player_id = db.get_or_create_player(player_type, model_name, model_provider, temp)
        player_ids[role] = player_id
    
    stats = {"good_wins": 0, "evil_wins": 0, "unknown": 0}
    
    try:
        for i in range(n_games):
            print(f"\nGame {i+1}/{n_games}")

            game = None
            for attempt in range(1, MAX_GAME_RETRIES + 1):
                try:
                    # Create and run game with specific model configs
                    game = create_mini_mafia_game(model_configs=model_configs, debug_prompts=debug_prompts)

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

                    break  # success
                except Exception as exc:
                    print(f"Error during game {i+1} attempt {attempt}/{MAX_GAME_RETRIES}: {exc}")
                    if attempt >= MAX_GAME_RETRIES:
                        raise
                    print("Retrying...")

            # Save game data to database
            game_id = save_game_data_to_db(game, i, batch_id, db, player_ids)
            
            # Update stats (determine winner from game state)
            winner = determine_winner(game.state.agents)
            if winner == "good":
                stats["good_wins"] += 1
            elif winner == "evil":
                stats["evil_wins"] += 1
            else:
                stats["unknown"] += 1
            
            # Update batch progress in database
            db.update_batch_progress(batch_id, i + 1)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1} games. Current stats: {stats}")
        
        print(f"\n{'='*50}")
        print(f"BATCH COMPLETE: {batch_id}")
        print(f"{'='*50}")
        print(f"Total games: {n_games}")
        print(f"Town wins: {stats['good_wins']} ({stats['good_wins']/n_games:.1%})")
        print(f"Mafia wins: {stats['evil_wins']} ({stats['evil_wins']/n_games:.1%})")
        print(f"Unknown: {stats['unknown']} ({stats['unknown']/n_games:.1%})")
        target_db = db.db_path if hasattr(db, 'db_path') else 'database/mini_mafia.db'
        print(f"\nResults saved to SQLite database: {target_db}")
        print(f"Batch ID: {batch_id}")
        
    finally:
        # Always close database connection
        db.close()
        
    return batch_id

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Run N mini-mafia games and save results')
    parser.add_argument('n_games', type=int, help='Number of games to run')
    parser.add_argument('--debug', action='store_true', help='Show LLM prompts')
    parser.add_argument('--debug-responses', action='store_true', help='Show raw model responses when parsing fails')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode with prompts')
    parser.add_argument('--temperature', type=float, help='Temperature for all models (default: 0.7)')
    parser.add_argument('--model-configs', type=str, help='JSON string with model configurations')
    parser.add_argument('--db-path', type=str, help='SQLite database path (default: database/mini_mafia.db)')
    
    args = parser.parse_args()
    
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
    
    try:
        # Parse model configs if provided
        model_configs = None
        if args.model_configs:
            import json
            model_configs = json.loads(args.model_configs)
        
        # Run the batch
        batch_id = run_batch(
            n_games,
            debug_prompts=debug,
            model_configs=model_configs,
            temperature=args.temperature,
            db_path=args.db_path
        )
        
    except KeyboardInterrupt:
        print("\n\nBatch interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
