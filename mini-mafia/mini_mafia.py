#!/usr/bin/env python3
"""
Mini-Mafia Game - 4-Player Mafia Variant

A compact version of the classic Mafia game:
- 4 players: 1 Detective, 1 Mafioso, 2 Villagers
- One villager is eliminated during the night phase
- Detective can investigate one player to learn their role
- Single day phase with discussion and voting
- Win by arresting the mafioso or eliminating innocents
"""

import sys
import random
sys.path.append('.')

from src.main import create_game
from src.config import get_default_prompt_config

def single_day_play(game):
    """Single day phase play function for mini-mafia"""
    print("Initializing Mini-Mafia Game...")
    game.show_roles()
    
    # Start day 1 directly (after night phase setup)
    game.state.round = 1
    print(f"\n{'='*50}")
    print(f"DAY {game.state.round}")
    print(f"{'='*50}")
    game.show_status()
    game.run_day_phase()
    
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
        
    game.show_game_end(result)

def create_mini_mafia_game(model_configs=None, debug_prompts=False, prompt_config=None):
    """Create a 4-player mini-mafia game.
    
    Game setup:
    - 4 players: 1 detective, 1 mafioso, 2 villagers
    - One villager is killed during the night phase
    - Detective investigates one player and learns their role
    - Single day phase with discussion and voting
    - Win condition: arrest the mafioso (good wins) or eliminate innocent (evil wins)
    
    Args:
        model_configs: Dict with 'detective', 'mafioso', 'villager' keys mapping to LLM configs
        debug_prompts: Whether to show LLM prompts
        prompt_config: PromptConfig instance for versioned prompts
    
    Returns:
        Game instance ready to play with game.play()
    """
    
    # Default to Mistral if no configs provided
    if model_configs is None:
        model_configs = {
            'detective': {'type': 'local', 'model_path': 'models/mistral.gguf'},
            'mafioso': {'type': 'local', 'model_path': 'models/mistral.gguf'},
            'villager': {'type': 'local', 'model_path': 'models/mistral.gguf'}
        }
    
    # Player names
    names = ['Alice', 'Bob', 'Charlie', 'Diana']
    roles = ['detective', 'mafioso', 'villager', 'villager']
    random.shuffle(roles)
    
    # Create all 4 players initially
    players = [
        {'name': 'Alice', 'role': roles[0], 'llm': model_configs[roles[0]]},
        {'name': 'Bob', 'role': roles[1], 'llm': model_configs[roles[1]]},
        {'name': 'Charlie', 'role': roles[2], 'llm': model_configs[roles[2]]},
        {'name': 'Diana', 'role': roles[3], 'llm': model_configs[roles[3]]}
    ]
    
    # Create the game
    if prompt_config is None:
        prompt_config = get_default_prompt_config()
    game = create_game(players, discussion_rounds=2, debug_prompts=debug_prompts, prompt_config=prompt_config)
    
    # Night phase setup
    villagers = [a for a in game.state.agents if a.role == "villager"]
    detective = next(a for a in game.state.agents if a.role == "detective")
    mafioso = next(a for a in game.state.agents if a.role == "mafioso")
    
    # One villager is killed during the night
    victim = random.choice(villagers)
    victim.alive = False
    
    # Night begins
    alive_agents = game.state.get_alive_players()
    for agent in alive_agents:
        agent.remember(f"Night 1 begins.")
    
    # Mafioso knows who they killed
    mafioso.remember(f"You killed {victim.name}.")
    
    # Everyone discovers the death
    for agent in alive_agents:
        agent.remember(f"{victim.name} was found dead.")
    
    # Detective investigates the mafioso
    detective.remember(f"You investigated {mafioso.name} and discovered that they are the mafioso.")
    
    # Override play method for single day phase
    game.play = lambda: single_day_play(game)
    return game

if __name__ == "__main__":
    debug = input("Show LLM prompts? (y/n): ").strip().lower() == 'y'
    
    print("\nStarting Mini-Mafia Game...")
    game = create_mini_mafia_game(debug_prompts=debug)
    game.play()
