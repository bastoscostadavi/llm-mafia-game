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
import argparse
from datetime import datetime
from pathlib import Path

# Add project root and current directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.dirname(__file__))

from mini_mafia import create_mini_mafia_game
import io
from src.agents import MafiaAgent
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
        model_path = model_config.get('model_path', '')
        model_name = os.path.basename(model_path) if model_path else 'unknown'
        return 'llm', model_name, 'local', temperature
    else:
        # Fallback
        model = model_config.get('model', model_type)
        return 'llm', model, model_type, temperature

def save_game_data_to_db(game, game_num, batch_id, db, player_ids):
    """Save game data directly to SQLite database"""
    
    game_id = f"{batch_id}_game_{game_num:04d}"
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
    
    # Save events from actual game sequence (much more accurate than reconstruction!)
    events = convert_game_sequence_to_events(game.state.game_sequence, game.state.agents)
    for event in events:
        db.insert_event(
            game_id=game_id,
            sequence_number=event['sequence_number'],
            event_type=event['event_type'],
            actor_character=event.get('actor_character'),
            target_character=event.get('target_character'),
            content=event.get('content'),
            round_number=event.get('round_number'),
            metadata=event.get('metadata')
        )
    
    return game_id

def convert_game_sequence_to_events(game_sequence, agents):
    """Convert the game's action log into database events"""
    
    events = []
    sequence_num = 1
    
    # Add game start event
    events.append({
        'sequence_number': sequence_num,
        'event_type': 'game_start',
        'actor_character': None,
        'target_character': None,
        'content': None,
        'round_number': None
    })
    sequence_num += 1
    
    # Process each logged action from the game sequence
    for log_entry in game_sequence:
        action = log_entry.get('action')
        actor = log_entry.get('actor')
        parsed_result = log_entry.get('parsed_result')
        raw_response = log_entry.get('raw_response')
        
        # Map game actions to our event types
        if action == 'discuss':
            if parsed_result == "remained silent":
                event_type = 'discussion_silent'
                content = None
            else:
                event_type = 'discussion_message'  
                content = parsed_result
                
            events.append({
                'sequence_number': sequence_num,
                'event_type': event_type,
                'actor_character': actor,
                'target_character': None,
                'content': content,
                'round_number': 1,  # Mini-mafia is always 1 round
                'metadata': json.dumps({
                    'raw_response': raw_response,
                    'step': log_entry.get('step')
                }) if raw_response else None
            })
            sequence_num += 1
            
        elif action == 'vote':
            events.append({
                'sequence_number': sequence_num,
                'event_type': 'vote_cast',
                'actor_character': actor,
                'target_character': parsed_result,
                'content': None,
                'round_number': 1,
                'metadata': json.dumps({
                    'raw_response': raw_response,
                    'step': log_entry.get('step')
                }) if raw_response else None
            })
            sequence_num += 1
            
        elif action == 'kill':
            events.append({
                'sequence_number': sequence_num,
                'event_type': 'kill_action',
                'actor_character': actor,
                'target_character': parsed_result,
                'content': None,
                'round_number': 1,
                'metadata': json.dumps({
                    'raw_response': raw_response,
                    'step': log_entry.get('step')
                }) if raw_response else None
            })
            sequence_num += 1
            
        elif action == 'investigate':
            # The parsed_result is the target, content should be the discovered role
            target_agent = next((a for a in agents if a.name == parsed_result), None)
            discovered_role = target_agent.role if target_agent else 'unknown'
            
            events.append({
                'sequence_number': sequence_num,
                'event_type': 'investigate_action',
                'actor_character': actor,
                'target_character': parsed_result,
                'content': discovered_role,
                'round_number': 1,
                'metadata': json.dumps({
                    'raw_response': raw_response,
                    'step': log_entry.get('step')
                }) if raw_response else None
            })
            sequence_num += 1
    
    # Add game end event
    winner = determine_winner(agents)
    events.append({
        'sequence_number': sequence_num,
        'event_type': 'game_end',
        'actor_character': None,
        'target_character': None,
        'content': winner,
        'round_number': 1
    })
    
    return events

def determine_winner(agents):
    """Determine who won the game"""
    arrested = next((a for a in agents if a.imprisoned), None)
    if arrested:
        if arrested.role == "mafioso":
            return "good"
        else:  # villager or detective arrested
            return "evil"
    return "unknown"

# JSON-based functions removed - now using SQLite directly

def run_batch(n_games, debug_prompts=False, model_configs=None, temperature=None):
    """Run N mini-mafia games and save results to SQLite database"""
    
    # Use default model configs if none provided
    if model_configs is None:
        model_configs = get_default_model_configs()
    
    # Include temperature in batch ID if specified
    temp_suffix = f"_temp{temperature}" if temperature is not None else ""
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}{temp_suffix}"
    print(f"Starting batch: {batch_id}")
    print(f"Running {n_games} mini-mafia games...")
    
    # Initialize database connection
    db = MiniMafiaDB()
    db.connect()
    
    # Insert batch record
    timestamp = datetime.now().isoformat()
    db.insert_batch(batch_id, timestamp, model_configs)
    print(f"Batch record created in database")
    
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
        print(f"\nResults saved to SQLite database")
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
        # Run the batch
        batch_id = run_batch(n_games, debug_prompts=debug, temperature=args.temperature)
        
    except KeyboardInterrupt:
        print("\n\nBatch interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()