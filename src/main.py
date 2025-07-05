# src/flexible_main.py
from src.game_launcher import GameLauncher

def main():
    model_path = "models/mistral.gguf"
    
    print("🎮 MAFIA GAME - Choose Game Type")
    print("1. Classic (8 players)")
    print("2. Simple (4 players)")
    print("3. Large (12+ players)")
    print("4. Custom (you choose)")
    
    choice = input("\nSelect game type (1-4): ").strip()
    
    if choice == "1":
        game = GameLauncher.create_classic_game(model_path)
    elif choice == "2":
        game = GameLauncher.create_simple_game(model_path)
    elif choice == "3":
        player_count = int(input("How many players? (12-18): ") or "12")
        game = GameLauncher.create_large_game(model_path, player_count)
    elif choice == "4":
        game = GameLauncher.create_custom_game(model_path)
    else:
        print("Invalid choice, using classic game")
        game = GameLauncher.create_classic_game(model_path)
    
    game.play()

if __name__ == "__main__":
    main()
