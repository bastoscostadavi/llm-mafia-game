#!/usr/bin/env python3
"""
Mini-Mafia Plus - 5-Player Variant (1 Mafioso, 4 Villagers).

- Roles: 1 mafioso, 4 villagers
- Before day start, one villager is randomly eliminated (simulates night kill)
- No detective or night investigations
- Day phase uses 2 discussion rounds by default
"""

import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.main import create_game


def single_day_play(game, discussion_rounds):
    """Run a single day for the plus variant."""
    print("Initializing Mini-Mafia Plus Game...")
    game.show_roles()

    game.state.round = 1
    print(f"\n{'='*50}")
    print(f"DAY {game.state.round}")
    print(f"{'='*50}")
    game.show_status()
    game.run_day_phase()

    arrested_agent = next((a for a in game.state.agents if a.imprisoned), None)
    if arrested_agent:
        if arrested_agent.role == "mafioso":
            result = "GOOD WINS! The mafioso was arrested!"
        else:
            result = "EVIL WINS! An innocent was arrested!"
    else:
        result = "No arrest was made! Game incomplete."

    game.show_game_end(result)


def create_mini_mafia_plus_game(model_configs=None, discussion_rounds=2, debug_prompts=False):
    """Create a plus variant game."""
    if model_configs is None:
        model_configs = {
            'mafioso': {'type': 'local', 'model_path': 'models/mistral.gguf'},
            'villager': {'type': 'local', 'model_path': 'models/mistral.gguf'}
        }

    names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve']
    roles = ['mafioso'] + ['villager'] * 4
    random.shuffle(roles)

    players = [
        {'name': name, 'role': role, 'llm': model_configs[role]}
        for name, role in zip(names, roles)
    ]

    game = create_game(players, discussion_rounds=discussion_rounds, debug_prompts=debug_prompts)

    mafioso = next(a for a in game.state.agents if a.role == "mafioso")
    villagers = [a for a in game.state.agents if a.role == "villager"]

    victim = random.choice(villagers)
    victim.alive = False

    alive_agents = game.state.get_alive_players()
    for agent in alive_agents:
        agent.remember("Night 1 begins.")

    game.state.log_action('kill', mafioso.name, None, victim.name)

    mafioso.remember(f"You killed {victim.name}.")
    for agent in alive_agents:
        agent.remember(f"{victim.name} was found dead.")

    game.play = lambda: single_day_play(game, discussion_rounds)
    return game


if __name__ == '__main__':
    debug = input("Show LLM prompts? (y/n): ").strip().lower() == 'y'
    game = create_mini_mafia_plus_game(debug_prompts=debug)
    game.play()
