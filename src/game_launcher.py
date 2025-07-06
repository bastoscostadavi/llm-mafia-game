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
        """Simple 4-player game with special rule: detective always investigates assassin first"""
        config = {
            'players': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'roles': {
                'assassin': 1,
                'detective': 1,
                'villager': 2
            }
        }
        
        # Create the game with special first-night rule
        game = ConfigurableGame(model_path, config)
        
        # Override the first night to ensure detective investigates assassin
        original_play = game.play
        
        def play_with_special_rule():
            game.setup_game()
            
            # Special Night 1: Detective automatically investigates the assassin
            print("\n" + "="*50)
            print("🌙 NIGHT 1 - The game begins...")
            print("="*50)
            
            # Find detective and assassin
            detective = next(a for a in game.state.agents if a.role == "detective")
            assassin = next(a for a in game.state.agents if a.role == "assassin")
            
            # Let assassin kill someone
            alive = game.state.get_alive_players()
            alive_names = [a.name for a in alive if a.name != assassin.name]
            
            # Get assassin's kill choice
            kill_prompt = f"""[INST] You are {assassin.name}, the assassin. Choose your first victim.
    Available targets: {', '.join(alive_names)}
    Reply with just the name: [/INST]"""
            
            response = game.state.llm.generate(kill_prompt, max_tokens=15)
            
            # Parse target
            victim_name = None
            for name in alive_names:
                if name.lower() in response.lower():
                    victim_name = name
                    break
            
            if not victim_name:
                import random
                victim_name = random.choice(alive_names)
            
            # Kill the victim
            victim = game.state.get_agent_by_name(victim_name)
            victim.alive = False
            
            print(f"\n[SPECTATOR] Assassin {assassin.name} kills {victim_name}")
            
            # Detective automatically investigates assassin
            detective.remember(f"You investigated {assassin.name}: they are the assassin")
            print(f"[SPECTATOR] Detective {detective.name} investigates {assassin.name} (learns: assassin)")
            
            # Announce death
            print(f"\n☠️ {victim_name} was found dead!")
            
            # Everyone learns about the death
            for agent in game.state.get_alive_players():
                agent.remember(f"Night 1: {victim_name} was killed")
            
            # Reset ready_to_vote
            for agent in game.state.agents:
                agent.ready_to_vote = False
            
            # Now continue with normal day/night cycle
            game.state.round = 1  # We're starting Day 1
            
            while game.state.round < 10:
                # Day Phase
                game.display.show_day_start(game.state.round)
                game.display.show_status(game.state)
                game.day_phase.run()
                
                result = game.check_game_over()
                if result:
                    game.display.show_game_end(result)
                    game.display.show_final_roles(game.state.agents)
                    return
                
                # Night Phase
                game.state.round += 1
                game.display.show_night_start(game.state.round)
                game.night_phase.run()
                
                result = game.check_game_over()
                if result:
                    game.display.show_game_end(result)
                    game.display.show_final_roles(game.state.agents)
                    return
            
            game.display.show_timeout()
            game.display.show_final_roles(game.state.agents)
        
        # Replace play method
        game.play = play_with_special_rule
        
        print("\n⚠️  SPECIAL RULE: Detective will investigate the assassin on Night 1!")
        
        return game
    
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
