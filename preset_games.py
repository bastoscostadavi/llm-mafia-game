#!/usr/bin/env python3
"""
Preset Mafia Games

This file contains predefined game configurations.
Uses the simple create_game() function from src/main.py
"""
import sys
import random
sys.path.append('.')

from src.main import create_game

def get_model_options():
    """Get available model options"""
    return {
        '1': ('Local Mistral', {'type': 'local', 'model_path': 'models/mistral.gguf'}),
        '2': ('Local Llama 3.1 8B', {'type': 'local', 'model_path': 'models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'}),
        '3': ('Local Qwen2.5 7B', {'type': 'local', 'model_path': 'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf'}),
        '4': ('OpenAI GPT-3.5 (cheapest)', {'type': 'openai', 'model': 'gpt-3.5-turbo'}),
        '5': ('Claude Haiku (cheapest)', {'type': 'anthropic', 'model': 'claude-3-haiku-20240307'})
    }

def select_model_for_role(role):
    """Interactive model selection for a specific role"""
    options = get_model_options()
    
    print(f"\nSelect model for {role.upper()}:")
    for key, (name, _) in options.items():
        print(f"  {key}. {name}")
    
    while True:
        choice = input(f"Choice for {role} (1-5): ").strip()
        if choice in options:
            return options[choice][1]
        print("Invalid choice. Please select 1, 2, 3, 4, or 5.")

def get_model_configs_interactive():
    """Get model configurations through interactive selection"""
    print("\nConfigure models for each role:")
    print("=" * 40)
    
    model_configs = {}
    roles = ['detective', 'mafioso', 'villager']
    
    for role in roles:
        model_configs[role] = select_model_for_role(role)
    
    print("\nConfiguration summary:")
    options = get_model_options()
    for role, config in model_configs.items():
        # Find the model name for display
        model_name = "Unknown"
        for _, (name, cfg) in options.items():
            if cfg == config:
                model_name = name
                break
        print(f"  {role.capitalize()}: {model_name}")
    
    return model_configs

def get_preset_configs():
    """Get preset model configurations"""
    return {
        '1': ('All Local (Qwen2.5 7B)', {
            'detective': {'type': 'local', 'model_path': 'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf'},
            'mafioso': {'type': 'local', 'model_path': 'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf'},
            'villager': {'type': 'local', 'model_path': 'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf'}
        }),
        '2': ('All Local (Mistral)', {
            'detective': {'type': 'local', 'model_path': 'models/mistral.gguf'},
            'mafioso': {'type': 'local', 'model_path': 'models/mistral.gguf'},
            'villager': {'type': 'local', 'model_path': 'models/mistral.gguf'}
        }),
        '3': ('All Local (Llama 3.1 8B)', {
            'detective': {'type': 'local', 'model_path': 'models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'},
            'mafioso': {'type': 'local', 'model_path': 'models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'},
            'villager': {'type': 'local', 'model_path': 'models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'}
        }),
        '4': ('ChatGPT as Mafioso', {
            'detective': {'type': 'local', 'model_path': 'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf'},
            'mafioso': {'type': 'openai', 'model': 'gpt-3.5-turbo'},
            'villager': {'type': 'local', 'model_path': 'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf'}
        }),
        '5': ('Claude as Mafioso', {
            'detective': {'type': 'local', 'model_path': 'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf'},
            'mafioso': {'type': 'anthropic', 'model': 'claude-3-haiku-20240307'},
            'villager': {'type': 'local', 'model_path': 'models/Qwen2.5-7B-Instruct-Q4_K_M.gguf'}
        })
    }

def single_day_play(game):
    """Single day phase play function for mini-mafia"""
    game.display.show_game_start()
    game.display.show_roles(game.state.agents)
    
    # Start day 1 directly (no night phase)
    game.state.round = 1
    game.display.show_day_start(game.state.round)
    game.display.show_status(game.state)
    game.day_phase.run()
    
    # Determine winner based on who was arrested
    arrested_agent = next((a for a in game.state.agents if a.imprisoned), None)
    if arrested_agent:
        if arrested_agent.role == "mafioso":
            result = "GOOD WINS! The mafioso was arrested!"
        elif arrested_agent.role == "detective":
            result = "EVIL WINS! The detective was eliminated!"
        else:  # villager arrested
            result = "EVIL WINS! An innocent was arrested, allowing evil to continue!"
    else:
        result = "No one was arrested! Game incomplete."
        
    game.display.show_game_end(result)
    game.display.show_final_roles(game.state.agents)

def classic_game(debug_prompts=False):
    """Classic 6-player Mafia game: 2 mafiosos, 1 detective, 3 villagers"""
    # Fixed names
    roles = ['mafioso', 'mafioso', 'detective', 'villager', 'villager', 'villager']
    random.shuffle(roles)
    
    players = [
        {'name': 'Alice', 'role': roles[0], 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Bob', 'role': roles[1], 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Charlie', 'role': roles[2], 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Diana', 'role': roles[3], 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Eve', 'role': roles[4], 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Frank', 'role': roles[5], 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}}
    ]
    
    return create_game(players, discussion_rounds=1, debug_prompts=debug_prompts)

def mini_mafia_game(debug_prompts=False, model_configs=None):
    """Mini-mafia game with 4 specific players: Alice, Bob, Charlie, Diana.
    Roles assigned randomly: 1 detective, 1 mafioso, 2 villagers.
    One villager is killed, leaving detective + mafioso + 1 villager for day phase.
    
    Args:
        debug_prompts: Whether to show LLM prompts
        model_configs: Dict with 'detective', 'mafioso', 'villager' keys mapping to LLM configs
    """
    
    # Default to Qwen2.5 7B if no configs provided
    if model_configs is None:
        model_configs = {
            'detective': {'type': 'local', 'model_path': 'models/mistral.gguf'},
            'mafioso': {'type': 'local', 'model_path': 'models/mistral.gguf'},
            'villager': {'type': 'local', 'model_path': 'models/mistral.gguf'}
        }
    
    # Fixed names
    names = ['Alice', 'Bob', 'Charlie', 'Diana']
    roles = ['detective', 'mafioso', 'villager', 'villager']
    random.shuffle(roles)
    
    # Create all 4 players initially with assigned models
    players = [
        {'name': 'Alice', 'role': roles[0], 'llm': model_configs[roles[0]]},
        {'name': 'Bob', 'role': roles[1], 'llm': model_configs[roles[1]]},
        {'name': 'Charlie', 'role': roles[2], 'llm': model_configs[roles[2]]},
        {'name': 'Diana', 'role': roles[3], 'llm': model_configs[roles[3]]}
    ]
    
    # Create the game
    game = create_game(players, discussion_rounds=2, debug_prompts=debug_prompts)
    
    # Find and kill one of the villagers
    villagers = [a for a in game.state.agents if a.role == "villager"]
    victim = random.choice(villagers)
    victim.alive = False
    
    # Get survivors (should be detective + mafioso + 1 villager)
    alive_agents = game.state.get_alive_players()
    detective = next(a for a in alive_agents if a.role == "detective")
    mafioso = next(a for a in alive_agents if a.role == "mafioso")
    
    
    

    mafioso.remember(f"You killed {victim.name}.")
    # Everyone knows who was found dead, including role information for mini-mafia
    for agent in alive_agents:
        agent.remember(f"Night 1: The villager {victim.name} was found dead.")
    
   # Detective and mafioso know each other
    detective.remember(f"You investigated {mafioso.name} and discovered that they are the mafioso.")
    detective.remember(f"Beware, {mafioso.name} knows you're the detective.")
    mafioso.remember(f"Beware, {detective.name} is the detective and discovered you're the mafioso.")

    # Override play method for single day phase
    game.play = lambda: single_day_play(game)
    return game

def main():
    """Run preset games menu"""
    print("MAFIA GAME - Preset Games")
    print("="*40)
    print("1. Classic (6 players: 2 mafiosos, 1 detective, 3 villagers)")
    print("2. Mini-mafia (4 players: 1 mafioso, 1 detective, 2 villagers. But a villager is killed at night.)")
    
    choice = input("\nSelect game (1-2): ").strip()
    debug = input("Show LLM prompts? (y/n): ").strip().lower() == 'y'
    
    if choice == "1":
        print("\nStarting Classic Game...")
        game = classic_game(debug_prompts=debug)
        game.play()
        
    elif choice == "2":
        print("\nMini-mafia Game Setup")
        print("="*40)
        
        # Show preset options
        presets = get_preset_configs()
        print("\nSelect model configuration:")
        for key, (name, _) in presets.items():
            print(f"  {key}. {name}")
        print("  6. Custom (choose models for each role)")
        
        while True:
            preset_choice = input("\nChoice (1-6): ").strip()
            if preset_choice in ['1', '2', '3', '4', '5', '6']:
                break
            print("Invalid choice. Please select 1, 2, 3, 4, 5, or 6.")
        
        if preset_choice == '6':
            # Custom configuration
            model_configs = get_model_configs_interactive()
        else:
            # Use preset
            _, model_configs = presets[preset_choice]
            print(f"\nUsing preset: {presets[preset_choice][0]}")
        
        # Show API key reminder if using API models
        uses_api = any(config['type'] in ['openai', 'anthropic'] 
                      for config in model_configs.values())
        
        if uses_api:
            print(f"\nNote: Make sure you have your API keys set in environment variables:")
            print("  export OPENAI_API_KEY='your-key-here'")
            print("  export ANTHROPIC_API_KEY='your-key-here'")
            
            confirm = input("\nProceed with game? (y/n): ").strip().lower()
            if confirm != 'y':
                print("Game cancelled.")
                return
        
        print("\nStarting Mini-mafia Game...")
        game = mini_mafia_game(debug_prompts=debug, model_configs=model_configs)
        game.play()
        
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()