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

def mini_mafia_game(debug_prompts=False):
    """Mini-mafia game with 4 specific players: Alice, Bob, Charlie, Diana.
    Roles assigned randomly: 1 detective, 1 mafioso, 2 villagers.
    One villager is killed, leaving detective + mafioso + 1 villager for day phase."""
    
    # Fixed names
    names = ['Alice', 'Bob', 'Charlie', 'Diana']
    roles = ['detective', 'mafioso', 'villager', 'villager']
    random.shuffle(roles)
    
    # Create all 4 players initially
    players = [
        {'name': names[0], 'role': roles[0], 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': names[1], 'role': roles[1], 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': names[2], 'role': roles[2], 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': names[3], 'role': roles[3], 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}}
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
    
    # Detective and mafioso know each other
    detective.remember(f"You investigated {mafioso.name}: evil")
    detective.remember(f"{mafioso.name} knows you're the detective")
    mafioso.remember(f"You killed {victim.name}")
    mafioso.remember(f"{detective.name} investigated you and learned you're evil.")
    
    # Everyone knows who was found dead, including role information for mini-mafia
    death_message = f"{victim.name} ({victim.role}) was found dead this morning."
    
    for agent in alive_agents:
        # In mini-mafia, players learn the role of the dead person
        agent.remember(death_message)
    
    # Override play method for single day phase
    def single_day_play():
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
    
    game.play = single_day_play
    return game

def main():
    """Run preset games menu"""
    print("MAFIA GAME - Preset Games")
    print("="*40)
    print("1. Classic (6 players: 2 mafiosos, 1 detective, 3 villagers)")
    print("2. Mini-mafia (4 players, 1 killed randomly, single day phase)")
    
    choice = input("\nSelect game (1-2): ").strip()
    debug = input("Show LLM prompts? (y/n): ").strip().lower() == 'y'
    
    if choice == "1":
        print("\nStarting Classic Game...")
        game = classic_game(debug_prompts=debug)
        game.play()
        
    elif choice == "2":
        print("\nStarting Mini-mafia Game...")
        game = mini_mafia_game(debug_prompts=debug)
        game.play()
        
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()