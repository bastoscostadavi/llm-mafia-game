#!/usr/bin/env python3
"""
Setup script for Mini-Mafia Web Game
Allows admin to select background configuration before starting server
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / 'web_game_config.json'

# Available model configurations
MODEL_CONFIGS = {
    'GPT-4.1 Mini': {'type': 'openai', 'model': 'gpt-4o-mini'},
    'GPT-5 Mini': {'type': 'openai', 'model': 'gpt-5-mini'},
    'DeepSeek V3.1': {'type': 'openai', 'model': 'deepseek-chat', 'base_url': 'https://api.deepseek.com'},
    'Grok 3 Mini': {'type': 'openai', 'model': 'grok-3-mini', 'base_url': 'https://api.x.ai/v1'},
    'Mistral 7B (Local)': {'type': 'local', 'model_path': 'models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf'},
}

def print_header():
    print("=" * 60)
    print("       Mini-Mafia Web Game - Background Configuration")
    print("=" * 60)
    print()

def select_model(role):
    print(f"\nSelect model for {role.upper()}:")
    print()

    models = list(MODEL_CONFIGS.keys())
    for i, name in enumerate(models, 1):
        print(f"  {i}. {name}")

    print()
    while True:
        try:
            choice = input(f"Enter number (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            print(f"Please enter a number between 1 and {len(models)}")
        except (ValueError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            exit(0)

def main():
    print_header()

    print("You will configure which AI models the students will play against.")
    print("Students will play as MAFIOSO.")
    print("You need to select models for DETECTIVE and VILLAGER roles.")
    print()

    # Select detective
    detective_name = select_model("Detective")
    detective_config = MODEL_CONFIGS[detective_name]

    # Select villager
    villager_name = select_model("Villager")
    villager_config = MODEL_CONFIGS[villager_name]

    # Create background name
    background_name = f"{detective_name}_{villager_name}".replace(" ", "_").lower()

    # Save configuration
    config = {
        'background_name': background_name,
        'detective_model': detective_config,
        'villager_model': villager_config
    }

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("✓ Configuration saved!")
    print("=" * 60)
    print(f"Background name: {background_name}")
    print(f"Detective: {detective_name}")
    print(f"Villager: {villager_name}")
    print()
    print("Next steps:")
    print("  1. Make sure your API keys are set in .env file")
    print("  2. Run: python mini-mafia-benchmark/web_game.py")
    print("  3. Share the link with students")
    print()

if __name__ == '__main__':
    main()
