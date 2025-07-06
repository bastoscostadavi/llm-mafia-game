# research/view_games.py
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class GameViewer:
    """View saved game files"""
    
    def __init__(self, data_dir: str = "data/games"):
        self.data_dir = data_dir
        
    def list_games(self, limit: int = 20):
        """List available game files"""
        if not os.path.exists(self.data_dir):
            print("No games found!")
            return []
        
        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.json')])
        
        print(f"\n📁 Found {len(files)} games")
        print("Recent games:")
        
        for i, filename in enumerate(files[-limit:]):
            # Extract info from filename
            parts = filename.replace('.json', '').split('_')
            if len(parts) >= 4:
                batch = f"{parts[1]}_{parts[2]}"
                game_num = parts[4]
                print(f"{i+1}. Game #{game_num} from batch {batch}")
            else:
                print(f"{i+1}. {filename}")
        
        return files
    
    def load_game(self, filename: str) -> dict:
        """Load a game file"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def display_game(self, game_data: dict):
        """Display a game nicely"""
        print("\n" + "="*60)
        print("🎮 GAME DETAILS")
        print("="*60)
        
        # Basic info
        print(f"\n📊 Setup:")
        print(f"  Game state: {game_data['game_state']}")
        print(f"  Victim: {game_data['victim']} (killed Night 1)")
        
        # Show roles
        print(f"\n🎭 Players:")
        for name, role in game_data['agents']:
            print(f"  {name}: {role}")
        
        # Discussion
        print(f"\n💬 DISCUSSION:")
        print("-"*40)
        
        current_round = 0
        for entry in game_data['dialogue']:
            if entry['round'] != current_round:
                current_round = entry['round']
                print(f"\n📢 Round {current_round}:")
            
            role_hint = f"[{entry['role'][0].upper()}]"  # A, D, V
            print(f"  {entry['speaker']} {role_hint}: {entry['message']}")
        
        # Voting result
        print(f"\n🗳️ VOTE RESULT:")
        print(f"  Accused: {game_data['accused']}")
        accused_role = next(role for name, role in game_data['agents'] if name == game_data['accused'])
        print(f"  They were: {accused_role}")
        
        # Outcome
        if game_data['outcome'] == 0:
            print(f"\n✅ GOOD WINS! (Assassin was caught)")
        else:
            print(f"\n❌ EVIL WINS! (Assassin escaped)")
        
        print("\n" + "="*60)
    
    def search_games(self, criteria: dict):
        """Search for games matching criteria"""
        matches = []
        
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        
        for filename in files:
            game = self.load_game(filename)
            
            # Check criteria
            match = True
            if 'game_state' in criteria and game['game_state'] != criteria['game_state']:
                match = False
            if 'outcome' in criteria and game['outcome'] != criteria['outcome']:
                match = False
            if 'detective_alive' in criteria:
                detective_alive = any(name != game['victim'] for name, role in game['agents'] if role == 'detective')
                if detective_alive != criteria['detective_alive']:
                    match = False
            
            if match:
                matches.append((filename, game))
        
        return matches
    
    def show_statistics(self):
        """Show overall statistics"""
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        
        stats = {
            'total': len(files),
            'ADV': {'good': 0, 'evil': 0},
            'AVV': {'good': 0, 'evil': 0}
        }
        
        for filename in files:
            game = self.load_game(filename)
            state = game['game_state']
            outcome = 'good' if game['outcome'] == 0 else 'evil'
            stats[state][outcome] += 1
        
        print("\n📊 OVERALL STATISTICS")
        print("="*40)
        print(f"Total games: {stats['total']}")
        
        for state in ['ADV', 'AVV']:
            total = stats[state]['good'] + stats[state]['evil']
            if total > 0:
                good_pct = stats[state]['good'] / total * 100
                print(f"\n{state} games ({total}):")
                print(f"  Good wins: {stats[state]['good']} ({good_pct:.1f}%)")
                print(f"  Evil wins: {stats[state]['evil']} ({100-good_pct:.1f}%)")

def main():
    viewer = GameViewer()
    
    while True:
        print("\n🔍 GAME VIEWER")
        print("1. List recent games")
        print("2. View specific game")
        print("3. Search games")
        print("4. Show statistics")
        print("5. Exit")
        
        choice = input("\nChoice: ")
        
        if choice == '1':
            files = viewer.list_games()
            
        elif choice == '2':
            files = viewer.list_games()
            if files:
                idx = input("\nEnter game number (or filename): ")
                
                if idx.isdigit() and 1 <= int(idx) <= len(files):
                    filename = files[-int(idx)]
                elif idx.endswith('.json'):
                    filename = idx
                else:
                    filename = files[-1]  # Default to most recent
                
                game = viewer.load_game(filename)
                viewer.display_game(game)
        
        elif choice == '3':
            print("\nSearch criteria:")
            print("1. ADV games where good wins")
            print("2. AVV games where evil wins")
            print("3. Games where detective survives")
            print("4. Custom")
            
            search_choice = input("\nChoice: ")
            
            criteria = {}
            if search_choice == '1':
                criteria = {'game_state': 'ADV', 'outcome': 0}
            elif search_choice == '2':
                criteria = {'game_state': 'AVV', 'outcome': 1}
            elif search_choice == '3':
                criteria = {'detective_alive': True}
            
            matches = viewer.search_games(criteria)
            print(f"\nFound {len(matches)} matching games")
            
            if matches and input("View them? (y/n): ").lower() == 'y':
                for filename, game in matches[:5]:  # Show first 5
                    viewer.display_game(game)
                    if input("\nContinue? (y/n): ").lower() != 'y':
                        break
        
        elif choice == '4':
            viewer.show_statistics()
            
        elif choice == '5':
            break

if __name__ == "__main__":
    main()
