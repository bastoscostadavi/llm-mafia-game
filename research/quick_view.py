# research/quick_view.py
import sys
from view_games import GameViewer

if __name__ == "__main__":
    viewer = GameViewer()
    
    if len(sys.argv) > 1:
        # View specific file
        filename = sys.argv[1]
        if not filename.endswith('.json'):
            filename += '.json'
        
        game = viewer.load_game(filename)
        viewer.display_game(game)
    else:
        # View most recent
        files = sorted([f for f in os.listdir("data/games") if f.endswith('.json')])
        if files:
            game = viewer.load_game(files[-1])
            viewer.display_game(game)
            print(f"\nShowing most recent game: {files[-1]}")
        else:
            print("No games found!")
