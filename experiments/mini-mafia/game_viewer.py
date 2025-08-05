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
            if os.path.exists("data"):
                self.data_dir = "data"
            elif os.path.exists("../data"):
                self.data_dir = "../data"
            else:
                self.data_dir = "data"  # fallback
        else:
            self.data_dir = data_dir
        
    def list_batches(self) -> List[str]:
        """List available batch experiments"""
        if not os.path.exists(self.data_dir):
            return []
        
        # Look for batch folders instead of individual files
        batches = []
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path) and item.startswith('batch_'):
                batches.append(item)
        
        return sorted(batches)
    
    def load_batch_summary(self, batch_id: str) -> Optional[Dict]:
        """Load batch summary"""
        batch_dir = os.path.join(self.data_dir, batch_id)
        if not os.path.exists(batch_dir):
            return None
        
        # Look for summary files in batch folder
        for filename in os.listdir(batch_dir):
            if filename.endswith('_summary.json') or filename == 'batch_summary.json':
                summary_path = os.path.join(batch_dir, filename)
                with open(summary_path, 'r') as f:
                    return json.load(f)
        
        return None
    
    def load_batch_games(self, batch_id: str) -> List[Dict]:
        """Load all games from a batch"""
        games = []
        batch_dir = os.path.join(self.data_dir, batch_id)
        if not os.path.exists(batch_dir):
            return games
        
        # Load all game files from batch folder
        for filename in sorted(os.listdir(batch_dir)):
            if filename.endswith('.json') and '_game_' in filename:
                game_path = os.path.join(batch_dir, filename)
                with open(game_path, 'r') as f:
                    games.append(json.load(f))
        
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
        """Show detailed view of a specific game - matches runtime display format"""
        games = self.load_batch_games(batch_id)
        if game_num >= len(games):
            print(f"❌ Game {game_num} not found in batch {batch_id}")
            return
        
        game = games[game_num]
        
        # Game start
        print("Initializing Mafia Game...")
        print("\nSecret roles:")
        for player in game['initial_players']:
            print(f"  {player['name']}: {player['role']}")
        
        # Simulate game flow based on memories
        self._simulate_game_flow(game)
        
        # Game end
        winner_text = "GOOD WINS! All mafiosos arrested!" if game['winner'] == 'good' else "EVIL WINS! All good players killed!"
        print(f"\n==================================================")
        print(winner_text)
        print(f"==================================================")
        
        # Final roles
        print(f"\nFINAL ROLES:")
        for player in game['final_state']:
            if player['imprisoned']:
                status = "[IMPRISONED] "
            elif not player['alive']:
                status = "[DEAD] "
            else:
                status = "[ALIVE] "
            print(f"{status}{player['name']}: {player['role'].upper()}")
    
    def _simulate_game_flow(self, game):
        """Simulate the game flow based on stored memories"""
        # Extract game events from memories
        for round_num in range(1, game['total_rounds'] + 1):
            print(f"\nNIGHT {round_num}")
            
            # Show night actions (detective investigation, mafioso coordination)
            print("\n[SPECTATOR] Resolving night actions...")
            
            # Find who died this round
            dead_player = game['dead_player']['name']
            if round_num == 1:  # In mini-mafia, death happens in round 1
                print(f"\nFound dead: {dead_player}")
            
            print(f"\n==================================================")
            print(f"DAY {round_num}")
            print(f"==================================================")
            
            # Show current status
            alive_players = [p['name'] for p in game['final_state'] if p['alive'] and not p['imprisoned']]
            imprisoned_players = [p['name'] for p in game['final_state'] if p['imprisoned']]
            dead_players = [p['name'] for p in game['final_state'] if not p['alive']]
            
            print(f"\nCurrent Status:")
            print(f"  Alive: {', '.join(alive_players) if alive_players else 'None'}")
            print(f"  Imprisoned: {', '.join(imprisoned_players) if imprisoned_players else 'None'}")
            print(f"  Dead: {', '.join(dead_players) if dead_players else 'None'}")
            
            # Show discussion
            print(f"\nDISCUSSION - Day {round_num}")
            
            # Extract discussion from memories
            self._show_discussion_from_memories(game, round_num)
            
            # Show voting
            print(f"\nVOTING:")
            self._show_voting_from_memories(game, round_num)
    
    def _show_discussion_from_memories(self, game, round_num):
        """Extract and show discussion from player memories"""
        discussions = []
        
        # Collect all discussion messages from memories
        for player in game['final_state']:
            for memory in player['memory']:
                # Look for player messages (not system messages)
                if ':' in memory and not memory.startswith('You'):
                    if not any(memory.startswith(prefix) for prefix in ['Night', 'Day', 'Found dead', 'You investigated']):
                        discussions.append(memory)
                elif memory.startswith('You said:'):
                    # Convert "You said:" to player name
                    message = memory.replace('You said:', f"{player['name']}:")
                    discussions.append(message)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_discussions = []
        for msg in discussions:
            if msg not in seen:
                seen.add(msg)
                unique_discussions.append(msg)
        
        # Show discussion rounds
        if unique_discussions:
            messages_per_round = len(unique_discussions) // game['discussion_rounds']
            for round_idx in range(game['discussion_rounds']):
                print(f"\nRound {round_idx + 1}:")
                start_idx = round_idx * messages_per_round
                end_idx = start_idx + messages_per_round if round_idx < game['discussion_rounds'] - 1 else len(unique_discussions)
                
                for msg in unique_discussions[start_idx:end_idx]:
                    print(msg)
    
    def _show_voting_from_memories(self, game, round_num):
        """Show voting results from game data"""
        # In mini-mafia, show who was arrested
        arrested_player = game['arrested_player']
        
        # Simulate voting (we don't have individual votes stored, so show result)
        print(f"\n{arrested_player['name']} has been arrested!")
        print(f"They were {arrested_player['role'].upper()}!")
    
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