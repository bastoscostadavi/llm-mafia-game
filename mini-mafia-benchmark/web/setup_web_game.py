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
# Both detective and villager will use the same model
MODEL_CONFIGS = {
    'gpt-5-mini': {'type': 'openai', 'model': 'gpt-5-mini'},
    'gpt-4.1-mini': {'type': 'openai', 'model': 'gpt-4o-mini'},
    'grok-3-mini': {'type': 'openai', 'model': 'grok-3-mini', 'base_url': 'https://api.x.ai/v1'},
    'deepseek': {'type': 'openai', 'model': 'deepseek-chat', 'base_url': 'https://api.deepseek.com'},
    'mistral': {'type': 'local', 'model_path': 'models/Mistral-7B-Instruct-v0.2-Q4_K_M.gguf'},
}

def print_header():
    print("=" * 60)
    print("       Mini-Mafia Web Game - Background Configuration")
    print("=" * 60)
    print()

def select_model():
    print("Select AI model for both Detective and Villager:")
    print("(Students will play as Mafioso)")
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

    # Select single model for both roles
    model_name = select_model()
    model_config = MODEL_CONFIGS[model_name]

    # Save configuration
    config = {
        'background_name': model_name,
        'detective_model': model_config,
        'villager_model': model_config  # Same model for both
    }

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("✓ Configuration saved!")
    print("=" * 60)
    print(f"Background: {model_name}")
    print(f"Detective: {model_name}")
    print(f"Villager: {model_name}")
    print()
    print("Next steps:")
    print("  1. Make sure your API keys are set in .env file")
    print("  2. Run: python mini-mafia-benchmark/web/web_interface.py")
    print("  3. Open browser to: http://localhost:5002")
    print("  4. Share the link with students")
    print()

if __name__ == '__main__':
    main()
