#!/usr/bin/env python3
"""
Human vs AI Mini-Mafia Game

Interactive script where a human player chooses a role and plays against AI agents
using one of the four background models (Mistral, Grok, GPT, DeepSeek).
"""

import sys
import json
from datetime import datetime
sys.path.append('../..')

from mini_mafia import create_mini_mafia_game
from database.db_utils import MiniMafiaDB

def get_human_role():
    """Get the human player's role choice"""
    print("\n" + "="*50)
    print("ROLE SELECTION")
    print("="*50)
    print("Choose your role:")
    print("1. Detective (Town) - Investigate one player each night")
    print("2. Mafioso (Mafia) - Eliminate one town member each night") 
    print("3. Villager (Town) - No special powers, use reasoning")
    print("-" * 50)
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        if choice == '1':
            return 'detective'
        elif choice == '2':
            return 'mafioso'
        elif choice == '3':
            return 'villager'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def get_background_model():
    """Get the background model choice for AI agents"""
    print("\n" + "="*50)
    print("BACKGROUND MODEL SELECTION")
    print("="*50)
    print("Choose the AI background model:")
    print("1. Mistral 7B Instruct")
    print("2. Grok 3 Mini")
    print("3. GPT-4.1 Mini")
    print("4. DeepSeek V3.1")
    print("-" * 50)
    
    model_configs = {
        '1': {'type': 'local', 'model': 'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf'},
        '2': {'type': 'xai', 'model': 'grok-3-mini'},
        '3': {'type': 'openai', 'model': 'gpt-4.1-mini'},
        '4': {'type': 'deepseek', 'model': 'deepseek-chat'}
    }
    
    model_names = {
        '1': 'Mistral 7B Instruct',
        '2': 'Grok 3 Mini', 
        '3': 'GPT-4.1 Mini',
        '4': 'DeepSeek V3.1'
    }
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        if choice in model_configs:
            return model_configs[choice], model_names[choice]
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

def create_model_configs(human_role, ai_config):
    """Create model configuration for mini-mafia game"""
    
    # Define human config
    human_config = {'type': 'human', 'player_name': 'You'}
    
    # Create config for each role
    model_configs = {}
    
    for role in ['detective', 'mafioso', 'villager']:
        if role == human_role:
            model_configs[role] = human_config
        else:
            model_configs[role] = ai_config
    
    return model_configs

def determine_winner(agents):
    """Determine who won the game"""
    arrested = next((a for a in agents if a.imprisoned), None)
    if arrested:
        if arrested.role == "mafioso":
            return "good"
        else:  # villager or detective arrested
            return "evil"
    return "unknown"

def save_game_to_database(game, human_role, ai_model_name):
    """Save the game results to database"""
    try:
        db = MiniMafiaDB()
        db.connect()
        
        # Create unique game ID for human vs AI games
        timestamp = datetime.now()
        game_id = f"human_vs_ai_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        batch_id = "human_vs_ai"
        
        # Determine winner
        winner = determine_winner(game.state.agents)
        
        # Insert game record
        db.insert_game(game_id, batch_id, 0, timestamp.isoformat(), winner)
        
        # Create player IDs and insert players
        for agent in game.state.agents:
            character_name = agent.name
            role = agent.role
            final_status = 'arrested' if agent.imprisoned else ('killed' if not agent.alive else 'alive')
            
            # Determine model name based on role
            if role == human_role:
                model_name = "human"
                model_provider = "human"
            else:
                model_name = ai_model_name.lower().replace(' ', '_')
                # Map provider names
                if 'mistral' in ai_model_name.lower():
                    model_provider = "mistral"
                elif 'grok' in ai_model_name.lower():
                    model_provider = "xai"
                elif 'gpt' in ai_model_name.lower():
                    model_provider = "openai"
                elif 'deepseek' in ai_model_name.lower():
                    model_provider = "deepseek"
                else:
                    model_provider = "unknown"
            
            # Insert or get player
            player_id = db.insert_player_if_not_exists(model_name, model_provider)
            db.insert_game_player(game_id, player_id, character_name, role, final_status)
        
        # Save events if available (simplified)
        if hasattr(game.state, 'game_sequence') and game.state.game_sequence:
            from run_mini_mafia_batch import convert_game_sequence_to_events
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
                    metadata=json.dumps(event.get('metadata', {}))
                )
        
        db.close()
        print(f"\n💾 Game saved to database with ID: {game_id}")
        
        # Show winner info
        if winner == "good":
            print("🏆 TOWN VICTORY! Mafioso was arrested.")
        elif winner == "evil":
            print("🏆 MAFIA VICTORY! Town failed to arrest the Mafioso.")
        else:
            print("⚖️  UNCLEAR RESULT")
            
    except Exception as e:
        print(f"\n⚠️  Failed to save game to database: {e}")

def play_human_game():
    """Main game loop for human vs AI"""
    
    print("🎭 MINI-MAFIA: Human vs AI")
    print("="*50)
    print("Welcome to Mini-Mafia!")
    print("You'll play against 3 AI agents in a 4-player game.")
    print("Your goal depends on your role and faction alignment.")
    
    # Get human choices
    human_role = get_human_role()
    ai_config, ai_model_name = get_background_model()
    
    print(f"\n🎯 Game Setup:")
    print(f"Your role: {human_role.title()}")
    print(f"AI background: {ai_model_name}")
    
    # Create model configuration
    model_configs = create_model_configs(human_role, ai_config)
    
    # Show role assignments
    print(f"\n👥 Role Assignments:")
    for role in ['detective', 'mafioso', 'villager']:
        if model_configs[role].get('type') == 'human':
            print(f"  {role.title()}: YOU")
        else:
            print(f"  {role.title()}: AI ({ai_model_name})")
    
    print(f"\n🚀 Starting Mini-Mafia game...")
    print("-" * 50)
    
    # Create and play the mini-mafia game
    try:
        game = create_mini_mafia_game(model_configs, debug_prompts=False)
        game.play()  # This handles all game flow and victory display
        
        # Save game to database
        save_game_to_database(game, human_role, ai_model_name)
        
    except KeyboardInterrupt:
        print("\n\nGame interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Please try again or report this issue.")

def main():
    """Main entry point"""
    try:
        play_human_game()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()