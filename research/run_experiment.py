import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from batch_runner import BatchRunner

def main():
    print("🔬 MAFIA RESEARCH EXPERIMENT")
    print("1. Quick test (10 games)")
    print("2. Small batch (100 games)")
    print("3. Medium batch (1000 games)")
    print("4. Large batch (10000 games)")
    print("5. Custom")
    
    choice = input("\nSelect option: ")
    
    game_counts = {
        "1": 10,
        "2": 100,
        "3": 1000,
        "4": 10000
    }
    
    if choice in game_counts:
        num_games = game_counts[choice]
    else:
        num_games = int(input("Number of games: ") or "100")
    
    save_individual = input("Save individual games? (y/n): ").lower() == 'y'
    
    # Update path to models since it's now in research folder
    runner = BatchRunner("models/mistral.gguf")
    runner.run_batch(num_games, save_games=save_individual)

if __name__ == "__main__":
    main()