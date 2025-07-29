#!/usr/bin/env python3
"""
Game Viewer for Mini-Mafia Batch Results

Views saved games from run_mini_mafia_batch.py experiments.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

class BatchViewer:
    """Viewer for batch experiment results"""
    
    def __init__(self, data_dir: str = None):
        # Auto-detect data directory
        if data_dir is None:
            if os.path.exists("data/mini_mafia"):
                self.data_dir = "data/mini_mafia"
            elif os.path.exists("../data/mini_mafia"):
                self.data_dir = "../data/mini_mafia"
            else:
                self.data_dir = "data/mini_mafia"  # fallback
        else:
            self.data_dir = data_dir
        
    def list_batches(self) -> List[str]:
        """List available batch experiments"""
        if not os.path.exists(self.data_dir):
            return []
        
        # First try to find batches via summary files
        files = [f for f in os.listdir(self.data_dir) if f.endswith('_summary.json')]
        batches = [f.replace('_summary.json', '') for f in files]
        
        # If no summary files, extract batch IDs from game files
        if not batches:
            game_files = [f for f in os.listdir(self.data_dir) if f.endswith('_game_0000.json')]
            batches = [f.replace('_game_0000.json', '') for f in game_files]
        
        return sorted(batches)
    
    def load_batch_summary(self, batch_id: str) -> Optional[Dict]:
        """Load batch summary"""
        summary_path = os.path.join(self.data_dir, f"{batch_id}_summary.json")
        if not os.path.exists(summary_path):
            return None
        
        with open(summary_path, 'r') as f:
            return json.load(f)
    
    def load_batch_games(self, batch_id: str) -> List[Dict]:
        """Load all games from a batch"""
        games = []
        i = 0
        while True:
            game_path = os.path.join(self.data_dir, f"{batch_id}_game_{i:04d}.json")
            if not os.path.exists(game_path):
                break
            
            with open(game_path, 'r') as f:
                games.append(json.load(f))
            i += 1
        
        return games
    
    def show_batch_summary(self, batch_id: str):
        """Show summary statistics for a batch"""
        summary = self.load_batch_summary(batch_id)
        if summary:
            # Use existing summary file
            print(f"\n📊 BATCH SUMMARY: {batch_id}")
            print("=" * 50)
            print(f"Total games: {summary['total_games']}")
            print(f"Good wins: {summary['statistics']['good_wins']} ({summary['win_rates']['good']:.1%})")
            print(f"Evil wins: {summary['statistics']['evil_wins']} ({summary['win_rates']['evil']:.1%})")
            print(f"Unknown: {summary['statistics']['unknown']} ({summary['win_rates']['unknown']:.1%})")
            print(f"Timestamp: {summary['timestamp']}")
            print(f"Debug prompts: {summary['configuration']['debug_prompts']}")
        else:
            # Generate summary from game files
            games = self.load_batch_games(batch_id)
            if not games:
                print(f"❌ Batch {batch_id} not found")
                return
                
            # Calculate statistics from games
            total_games = len(games)
            good_wins = sum(1 for g in games if g.get('winner') == 'good')
            evil_wins = sum(1 for g in games if g.get('winner') == 'evil')
            unknown = total_games - good_wins - evil_wins
            
            print(f"\n📊 BATCH SUMMARY: {batch_id} (generated)")
            print("=" * 50)
            print(f"Total games: {total_games}")
            print(f"Good wins: {good_wins} ({good_wins/total_games:.1%})")
            print(f"Evil wins: {evil_wins} ({evil_wins/total_games:.1%})")
            print(f"Unknown: {unknown} ({unknown/total_games:.1%})")
            if games:
                print(f"Timestamp: {games[0].get('timestamp', 'Unknown')}")
            print(f"Debug prompts: Unknown")
    
    def show_game_detail(self, batch_id: str, game_num: int):
        """Show detailed view of a specific game"""
        games = self.load_batch_games(batch_id)
        if game_num >= len(games):
            print(f"❌ Game {game_num} not found in batch {batch_id}")
            return
        
        game = games[game_num]
        
        print(f"\n🎮 GAME DETAIL: {game['game_id']}")
        print("=" * 50)
        print(f"Winner: {game['winner'].upper()}")
        print(f"Arrested: {game['arrested_player']['name']} ({game['arrested_player']['role']})")
        print(f"Dead: {game['dead_player']['name']} ({game['dead_player']['role']})")
        print(f"Rounds: {game['total_rounds']}")
        
        print("\n📝 PLAYER MEMORIES:")
        for player in game['final_state']:
            if player['alive'] or player['imprisoned']:
                print(f"\n{player['name']} ({player['role']}):")
                for i, memory in enumerate(player['memory'], 1):  # All memories
                    print(f"  {i}. {memory}")
    
    def interactive_menu(self):
        """Interactive menu for browsing batches"""
        while True:
            print("\n" + "=" * 40)
            print("🎮 BATCH VIEWER")
            print("=" * 40)
            
            batches = self.list_batches()
            if not batches:
                print("❌ No batch experiments found.")
                print("Run: python run_mini_mafia_batch.py N")
                return
            
            print("Available batches:")
            for i, batch in enumerate(batches, 1):
                summary = self.load_batch_summary(batch)
                if summary:
                    games_count = summary['total_games']
                else:
                    # Count games manually
                    games = self.load_batch_games(batch)
                    games_count = len(games)
                print(f"  {i}. {batch} ({games_count} games)")
            
            print(f"\nOptions:")
            print(f"  1-{len(batches)}: View batch summary")
            print(f"  g<batch_num> <game_num>: View specific game (e.g., 'g1 0')")
            print(f"  q: Quit")
            
            choice = input("\nSelect option: ").strip().lower()
            
            if choice == 'q':
                break
            elif choice.startswith('g'):
                # Parse game viewing command
                parts = choice.split()
                if len(parts) == 2 and parts[0][1:].isdigit() and parts[1].isdigit():
                    batch_idx = int(parts[0][1:]) - 1  # Remove 'g' and convert to int
                    game_num = int(parts[1])
                    if 0 <= batch_idx < len(batches):
                        self.show_game_detail(batches[batch_idx], game_num)
                    else:
                        print("❌ Invalid batch number")
                else:
                    print("❌ Format: g<batch_num> <game_num> (e.g., 'g1 0')")
            elif choice.isdigit():
                batch_idx = int(choice) - 1
                if 0 <= batch_idx < len(batches):
                    self.show_batch_summary(batches[batch_idx])
                else:
                    print("❌ Invalid batch number")
            else:
                print("❌ Invalid option")

def main():
    """Main entry point"""
    viewer = BatchViewer()
    
    if len(sys.argv) > 1:
        # Command line mode
        batch_id = sys.argv[1]
        if len(sys.argv) > 2:
            # Show specific game
            game_num = int(sys.argv[2])
            viewer.show_game_detail(batch_id, game_num)
        else:
            # Show batch summary
            viewer.show_batch_summary(batch_id)
    else:
        # Interactive mode
        try:
            viewer.interactive_menu()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")

if __name__ == "__main__":
    main()