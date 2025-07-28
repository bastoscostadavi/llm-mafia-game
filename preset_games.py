#!/usr/bin/env python3
"""
Preset Mafia Games

This file contains predefined game configurations.
Uses the simple create_game() function from src/main.py
"""
import sys
sys.path.append('.')

from src.main import create_game

def classic_game():
    """Classic 8-player Mafia game"""
    players = [
        {'name': 'Alice', 'role': 'villager', 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Bob', 'role': 'villager', 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Charlie', 'role': 'villager', 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Diana', 'role': 'detective', 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Eve', 'role': 'angel', 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Frank', 'role': 'assassin', 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Grace', 'role': 'assassin', 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Henry', 'role': 'psychopath', 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}}
    ]
    
    return create_game(players, discussion_rounds=2)

def mini_mafia_game():
    """Mini-mafia 4-player game with detective advantage"""
    players = [
        {'name': 'Alice', 'role': 'detective', 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Bob', 'role': 'assassin', 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Charlie', 'role': 'villager', 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Diana', 'role': 'villager', 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}}
    ]
    
    # Create the game
    game = create_game(players, discussion_rounds=2)
    
    # Special setup: Detective knows assassin from start
    detective = next(a for a in game.state.agents if a.role == "detective")
    assassin = next(a for a in game.state.agents if a.role == "assassin")
    detective.remember(f"You investigated {assassin.name}: they are the assassin")
    
    return game

def main():
    """Run preset games menu"""
    print("MAFIA GAME - Preset Games")
    print("="*40)
    print("1. Classic (8 players)")
    print("2. Mini-mafia (4 players, detective knows assassin)")
    
    choice = input("\nSelect game (1-2): ").strip()
    
    if choice == "1":
        print("\nStarting Classic Game...")
        game = classic_game()
        game.play()
        
    elif choice == "2":
        print("\nStarting Simple Game...")
        game = simple_game()
        game.play()
        
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()