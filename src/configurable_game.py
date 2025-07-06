import random
from typing import List, Dict, Optional
from src.game_state import GameState
from src.phases.day_phase import DayPhase
from src.phases.night_phase import NightPhase
from src.display import Display
from src.agent import MafiaAgent

class ConfigurableGame:
    """Mafia game with configurable player count and roles"""
    
    def __init__(self, model_path: str, game_config: Dict):
        """
        game_config = {
            'players': ['Alice', 'Bob', 'Charlie', ...],  # Names
            'roles': {
                'assassin': 2,
                'psychopath': 1,
                'detective': 1,
                'angel': 1,
                'villager': 3
            }
        }
        """
        self.display = Display()
        self.display.show_game_start()
        
        # Validate configuration
        self._validate_config(game_config)
        
        # Initialize with custom configuration
        self.player_names = game_config['players']
        self.role_distribution = game_config['roles']
        
        # Create game components
        self.state = GameState(model_path)
        self.day_phase = DayPhase(self.state, self.display)
        self.night_phase = NightPhase(self.state, self.display)
        
    def _validate_config(self, config: Dict):
        """Ensure configuration is valid"""
        total_roles = sum(config['roles'].values())
        total_players = len(config['players'])
        
        if total_roles != total_players:
            raise ValueError(f"Mismatch: {total_players} players but {total_roles} roles")
        
        # Check for at least one evil and one good
        evil_count = config['roles'].get('assassin', 0) + config['roles'].get('psychopath', 0)
        if evil_count == 0:
            raise ValueError("Need at least one evil player")
        
        if evil_count >= total_players:
            raise ValueError("Need at least one good player")
    
    def setup_game(self):
        """Create agents based on configuration"""
        # Use the new setup method that handles structure
        self.state.setup_agents(self.player_names, self.role_distribution)
        self.display.show_roles(self.state.agents)
    
    def play(self):
        """Main game loop - Night/Day cycle"""
        self.setup_game()
        
        while self.state.round < 10:  # Max 10 rounds
            self.state.round += 1
            
            # Night Phase - murders and investigations
            self.display.show_night_start(self.state.round)
            self.night_phase.run()
            
            # Check if game ended during night
            result = self.check_game_over()
            if result:
                self.display.show_game_end(result)
                self.display.show_final_roles(self.state.agents)
                return
            
            # Day Phase - discuss the murder and vote
            self.display.show_day_start(self.state.round)
            self.display.show_status(self.state)
            self.day_phase.run()
            
            # Check if game ended during day
            result = self.check_game_over()
            if result:
                self.display.show_game_end(result)
                self.display.show_final_roles(self.state.agents)
                return
        
        self.display.show_timeout()
        self.display.show_final_roles(self.state.agents)
    
    def check_game_over(self) -> Optional[str]:
        """Check win conditions"""
        alive = self.state.get_alive_players()
        
        if len(alive) == 0:
            return "💀 Everyone is dead! Nobody wins!"
        
        if len(alive) == 1:
            winner = alive[0]
            if winner.role == "psychopath":
                return f"🔪 PSYCHOPATH WINS! {winner.name} is the last one standing!"
            else:
                return f"🎉 {winner.name} survives! But at what cost..."
        
        assassins = sum(1 for a in alive if a.role == "assassin")
        psychopath = sum(1 for a in alive if a.role == "psychopath")
        good = sum(1 for a in alive if a.role not in ["assassin", "psychopath"])
        
        # If only one good vs one evil, evil wins (they kill at night)
        if good == 1 and (assassins + psychopath) == 1:
            if assassins == 1:
                return "💀 ASSASSINS WIN! The last good player can't survive the night!"
            else:
                return "🔪 PSYCHOPATH WINS! The last good player can't survive the night!"
        
        if assassins == 0 and psychopath == 0:
            return "🎉 GOOD WINS! All evil eliminated!"
        elif good == 0:
            if assassins > 0 and psychopath == 0:
                return "💀 ASSASSINS WIN!"
            elif psychopath > 0 and assassins == 0:
                return "🔪 PSYCHOPATH WINS!"
        
        return None