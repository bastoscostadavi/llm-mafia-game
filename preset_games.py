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
from src.prompts import PromptConfig
from src.config import get_default_prompt_config


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

    return create_game(players, discussion_rounds=1, debug_prompts=debug_prompts, prompt_config=get_default_prompt_config())

def mini_mafia_game(debug_prompts=False):
    """Mini-mafia game: 4 players (1 detective, 1 mafioso, 2 villagers)"""
    # Fixed names and roles
    roles = ['detective', 'mafioso', 'villager', 'villager']
    random.shuffle(roles)
    
    players = [
        {'name': 'Alice', 'role': roles[0], 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Bob', 'role': roles[1], 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Charlie', 'role': roles[2], 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
        {'name': 'Diana', 'role': roles[3], 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}}
    ]

    return create_game(players, discussion_rounds=2, debug_prompts=debug_prompts, prompt_config=get_default_prompt_config())


def main():
    """Run preset games menu"""
    print("MAFIA GAME - Preset Games")
    print("="*40)
    print("1. Classic (6 players: 2 mafiosos, 1 detective, 3 villagers)")
    print("2. Mini-mafia (4 players: 1 detective, 1 mafioso, 2 villagers)")
    
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