import random
from typing import Optional
from src.game_state import GameState
from src.phases.day_phase import DayPhase
from src.phases.night_phase import NightPhase
from src.display import Display

class MafiaGame:
    """Main game orchestrator - coordinates all components"""
    
    def __init__(self, model_path: str):
        self.display = Display()
        self.display.show_game_start()
        
        # Initialize components
        self.state = GameState(model_path)
        self.day_phase = DayPhase(self.state, self.display)
        self.night_phase = NightPhase(self.state, self.display)
        
    def setup_game(self):
        """Initialize the game"""
        self.state.setup_agents()
        self.display.show_roles(self.state.agents)
        
    def play(self):
        """Main game loop"""
        self.setup_game()
        
        while self.state.round < 10:  # Max 10 rounds
            self.state.round += 1
            self.display.show_day_start(self.state.round)
            self.display.show_status(self.state)
            
            # Day Phase
            self.day_phase.run()
            
            # Check win condition
            result = self.check_game_over()
            if result:
                self.display.show_game_end(result)
                self.display.show_final_roles(self.state.agents)
                return
            
            # Night Phase
            self.night_phase.run()
            
            # Check win condition
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
        
        # If only one good vs one evil, evil wins
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
