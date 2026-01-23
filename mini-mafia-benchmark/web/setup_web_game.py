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

def select_human_role():
    print("Select human player role:")
    print()
    print("  1. Mafioso (Deceive) - Human tries to deceive the villager")
    print("  2. Detective (Disclose) - Human reveals they know who the mafioso is")
    print("  3. Villager (Detect) - Human tries to detect who the mafioso is")
    print()

    while True:
        try:
            choice = input("Enter number (1-3): ").strip()
            if choice == '1':
                return 'mafioso'
            elif choice == '2':
                return 'detective'
            elif choice == '3':
                return 'villager'
            print("Please enter 1, 2, or 3")
        except (ValueError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            exit(0)

def select_model(role_name):
    print(f"\nSelect AI model for {role_name}:")
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

    # Select human role
    human_role = select_human_role()

    # Select AI models for the other two roles
    if human_role == 'mafioso':
        # Human is mafioso, need AI detective and villager
        detective_name = select_model("Detective")
        villager_name = select_model("Villager")
        mafioso_model = None  # Human
        detective_model = MODEL_CONFIGS[detective_name]
        villager_model = MODEL_CONFIGS[villager_name]
        background_name = f"{detective_name}+{villager_name}"

    elif human_role == 'detective':
        # Human is detective, need AI mafioso and villager
        mafioso_name = select_model("Mafioso")
        villager_name = select_model("Villager")
        mafioso_model = MODEL_CONFIGS[mafioso_name]
        detective_model = None  # Human
        villager_model = MODEL_CONFIGS[villager_name]
        background_name = f"{mafioso_name}+{villager_name}"

    elif human_role == 'villager':
        # Human is villager, need AI mafioso and detective
        mafioso_name = select_model("Mafioso")
        detective_name = select_model("Detective")
        mafioso_model = MODEL_CONFIGS[mafioso_name]
        detective_model = MODEL_CONFIGS[detective_name]
        villager_model = None  # Human
        background_name = f"{mafioso_name}+{detective_name}"

    # Save configuration
    config = {
        'human_role': human_role,
        'background_name': background_name,
        'mafioso_model': mafioso_model,
        'detective_model': detective_model,
        'villager_model': villager_model
    }

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("✓ Configuration saved!")
    print("=" * 60)
    print(f"Human role: {human_role.upper()}")
    print(f"Background: {background_name}")
    if human_role == 'mafioso':
        print(f"  Mafioso: HUMAN")
        print(f"  Detective: {detective_name}")
        print(f"  Villager: {villager_name}")
    elif human_role == 'detective':
        print(f"  Mafioso: {mafioso_name}")
        print(f"  Detective: HUMAN")
        print(f"  Villager: {villager_name}")
    elif human_role == 'villager':
        print(f"  Mafioso: {mafioso_name}")
        print(f"  Detective: {detective_name}")
        print(f"  Villager: HUMAN")
    print()
    print("Next steps:")
    print("  1. Make sure your API keys are set in .env file")
    print("  2. Run: python web/web_interface.py")
    print("  3. Open browser to: http://localhost:5000")
    print("  4. Share the link with participants")
    print()

if __name__ == '__main__':
    main()
