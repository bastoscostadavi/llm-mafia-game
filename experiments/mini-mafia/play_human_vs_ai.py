#!/usr/bin/env python3
"""
Human vs AI Mini-Mafia Game

Interactive script where a human player chooses a role and plays against AI agents
using one of the four background models (Mistral, Grok, GPT, DeepSeek).
"""

import sys
sys.path.append('../..')

from mini_mafia import create_mini_mafia_game

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
    print("4. DeepSeek V3")
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
        '4': 'DeepSeek V3'
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