# src/game_launcher.py
from src.configurable_game import ConfigurableGame
import random

class GameLauncher:
    """Helper to create different game configurations"""
    
    @staticmethod
    def create_classic_game(model_path: str):
        """Classic 8-player game"""
        config = {
            'players': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'],
            'roles': {
                'assassin': 2,
                'psychopath': 1,
                'detective': 1,
                'angel': 1,
                'villager': 3
            }
        }
        return ConfigurableGame(model_path, config)
    
    @staticmethod
    def create_simple_game(model_path: str):
        """Simple 4-player game"""
        config = {
            'players': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'roles': {
                'assassin': 1,
                'detective': 1,
                'villager': 2
            }
        }
        return ConfigurableGame(model_path, config)
    
    @staticmethod
    def create_large_game(model_path: str, player_count: int = 12):
        """Large game with many players"""
        # Generate names
        base_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 
                      'Grace', 'Henry', 'Ivan', 'Julia', 'Kevin', 'Luna',
                      'Max', 'Nina', 'Oscar', 'Petra', 'Quinn', 'Rosa']
        
        players = base_names[:player_count]
        
        # Distribute roles proportionally
        assassin_count = max(1, player_count // 4)  # 25% assassins
        detective_count = max(1, player_count // 8)  # 12.5% detectives
        angel_count = max(1, player_count // 8)     # 12.5% angels
        psychopath_count = 1 if player_count >= 6 else 0
        
        villager_count = player_count - (assassin_count + detective_count + 
                                         angel_count + psychopath_count)
        
        config = {
            'players': players,
            'roles': {
                'assassin': assassin_count,
                'detective': detective_count,
                'angel': angel_count,
                'psychopath': psychopath_count,
                'villager': villager_count
            }
        }
        
        return ConfigurableGame(model_path, config)
    
    @staticmethod
    def create_custom_game(model_path: str):
        """Interactive game setup"""
        print("🎮 CUSTOM GAME SETUP")
        
        # Get player count
        player_count = int(input("How many players? (4-18): ") or "6")
        player_count = max(4, min(18, player_count))
        
        # Get player names
        print(f"\nEnter {player_count} player names (or press Enter for defaults):")
        players = []
        default_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 
                        'Grace', 'Henry', 'Ivan', 'Julia', 'Kevin', 'Luna']
        
        for i in range(player_count):
            name = input(f"Player {i+1} name: ").strip()
            if not name:
                name = default_names[i] if i < len(default_names) else f"Player{i+1}"
            players.append(name)
        
        # Get role distribution
        print("\nRole distribution (must sum to", player_count, "):")
        
        roles = {}
        remaining = player_count
        
        for role in ['assassin', 'detective', 'angel', 'psychopath', 'villager']:
            if remaining <= 0:
                break
            
            if role == 'villager':
                # Villagers get whatever's left
                count = remaining
            else:
                max_allowed = remaining - 1 if role != 'villager' else remaining
                count = int(input(f"Number of {role}s (0-{max_allowed}): ") or "0")
                count = max(0, min(max_allowed, count))
            
            if count > 0:
                roles[role] = count
                remaining -= count
        
        config = {
            'players': players,
            'roles': roles
        }
        
        return ConfigurableGame(model_path, config)
