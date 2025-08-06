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
    
    def load_batch_config(self, batch_id: str) -> Optional[Dict]:
        """Load batch configuration"""
        batch_dir = os.path.join(self.data_dir, batch_id)
        config_path = os.path.join(batch_dir, 'batch_config.json')
        
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def load_batch_games(self, batch_id: str) -> List[Dict]:
        """Load all games from a batch"""
        games = []
        batch_dir = os.path.join(self.data_dir, batch_id)
        if not os.path.exists(batch_dir):
            return games
        
        # Load all game files from batch folder (new format uses game_XXXX.json)
        for filename in sorted(os.listdir(batch_dir)):
            if filename.startswith('game_') and filename.endswith('.json'):
                game_path = os.path.join(batch_dir, filename)
                with open(game_path, 'r') as f:
                    games.append(json.load(f))
        
        return games
    
    def show_batch_summary(self, batch_id: str):
        """Show summary statistics for a batch"""
        config = self.load_batch_config(batch_id)
        games = self.load_batch_games(batch_id)
        
        if not games:
            print(f"No games found for batch: {batch_id}")
            return
        
        # Calculate statistics from games
        stats = {"good_wins": 0, "evil_wins": 0, "unknown": 0}
        for game in games:
            winner = self.determine_winner(game.get('players', []))
            if winner == "good":
                stats["good_wins"] += 1
            elif winner == "evil":
                stats["evil_wins"] += 1
            else:
                stats["unknown"] += 1
        
        total_games = len(games)
        print(f"\n📊 BATCH SUMMARY: {batch_id}")
        print("=" * 50)
        print(f"Total games: {total_games}")
        print(f"Good wins: {stats['good_wins']} ({stats['good_wins']/total_games:.1%})")
        print(f"Evil wins: {stats['evil_wins']} ({stats['evil_wins']/total_games:.1%})")
        print(f"Unknown: {stats['unknown']} ({stats['unknown']/total_games:.1%})")
        
        if config:
            print(f"\nConfiguration:")
            print(f"  Prompt version: {config.get('prompt_config', {}).get('version', 'unknown')}")
            print(f"  Model: {config.get('model_configs', {}).get('detective', {}).get('model_name', 'unknown')}")
            print(f"  Temperature: {config.get('model_configs', {}).get('detective', {}).get('temperature', 'unknown')}")
        else:
            print("\nNo configuration found - this appears to be an old format batch")
    
    def determine_winner(self, players: List[Dict]) -> str:
        """Determine who won the game from player data"""
        arrested_player = next((p for p in players if p.get('imprisoned')), None)
        if arrested_player:
            if arrested_player.get('role') == "mafioso":
                return "good"
            else:  # villager or detective arrested
                return "evil"
        return "unknown"
    
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
        for player in game['players']:
            print(f"  {player['name']}: {player['role']}")
        
        # Simulate game flow based on memories
        self._simulate_game_flow(game)
        
        # Game end
        winner = self.determine_winner(game['players'])
        winner_text = "GOOD WINS! All mafiosos arrested!" if winner == 'good' else "EVIL WINS! All good players killed!"
        print(f"\n==================================================")
        print(winner_text)
        print(f"==================================================")
        
        # Final roles
        print(f"\nFINAL ROLES:")
        for player in game['players']:
            if player['imprisoned']:
                status = "[IMPRISONED] "
            elif not player['alive']:
                status = "[DEAD] "
            else:
                status = "[ALIVE] "
            print(f"{status}{player['name']}: {player['role'].upper()}")
    
    def _simulate_game_flow(self, game):
        """Simulate the game flow based on stored memories"""
        # In mini-mafia, there's typically only 1 round
        print(f"\nNIGHT 1")
        
        # Find who died (from memories)
        dead_player = next((p['name'] for p in game['players'] if not p['alive']), None)
        if dead_player:
            print(f"{dead_player} was found dead.")
        
        print(f"\n==================================================")
        print(f"DAY 1")
        print(f"==================================================")
        
        # Show current status
        alive_players = [p['name'] for p in game['players'] if p['alive'] and not p['imprisoned']]
        imprisoned_players = [p['name'] for p in game['players'] if p['imprisoned']]
        dead_players = [p['name'] for p in game['players'] if not p['alive']]
        
        print(f"\nCurrent Status:")
        print(f"  Active: {', '.join(alive_players) if alive_players else 'None'}")
        print(f"  Imprisoned: {', '.join(imprisoned_players) if imprisoned_players else 'None'}")
        print(f"  Dead: {', '.join(dead_players) if dead_players else 'None'}")
        
        # Show discussion from memories
        print(f"\nDISCUSSION - Day 1")
        self._show_discussion_from_memories(game)
        
        # Show voting from memories
        print(f"\nVOTING:")
        self._show_voting_from_memories(game)
    
    def _show_discussion_from_memories(self, game):
        """Extract and show discussion from player memories"""
        discussions = []
        
        # Collect all discussion messages from memories
        for player in game['players']:
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
        
        # Show discussion - assume 2 rounds for mini-mafia
        if unique_discussions:
            discussion_rounds = 2  # Mini-mafia typically has 2 discussion rounds
            messages_per_round = len(unique_discussions) // discussion_rounds
            for round_idx in range(discussion_rounds):
                print(f"\nRound {round_idx + 1}:")
                start_idx = round_idx * messages_per_round
                end_idx = start_idx + messages_per_round if round_idx < discussion_rounds - 1 else len(unique_discussions)
                
                for msg in unique_discussions[start_idx:end_idx]:
                    print(msg)
    
    def _show_voting_from_memories(self, game):
        """Show voting results from game data"""
        # Find voting information from memories
        voting_line = None
        arrested_player = None
        
        # Look for voting information in memories
        for player in game['players']:
            for memory in player['memory']:
                if 'votes:' in memory:
                    voting_line = memory
                    break
            if voting_line:
                break
        
        # Find arrested player
        arrested_player = next((p for p in game['players'] if p.get('imprisoned')), None)
        
        if voting_line:
            print(f"\n{voting_line}")
        
        if arrested_player:
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
                # Count games manually from batch folder
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