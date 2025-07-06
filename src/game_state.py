import random
from typing import List, Dict
from src.agent import MafiaAgent
from src.llm_interface import LlamaCppInterface

class GameState:
    """Manages all game state and agents"""
    
    def __init__(self, model_path: str):
        self.llm = LlamaCppInterface(model_path)
        self.agents: List[MafiaAgent] = []
        self.round = 0
        self.message_log = []
        self.game_config = None  # Will be set by ConfigurableGame
    
    def build_game_structure_text(self) -> str:
        """Build game structure text based on actual configuration"""
        if not self.game_config:
            return ""
        
        total_players = len(self.game_config['players'])
        roles = self.game_config['roles']
        
        structure = f"""Game Structure (public knowledge):
- {total_players} players total"""
        
        # List roles
        for role, count in roles.items():
            if count > 0:
                structure += f"\n- {count} {role}{'s' if count > 1 else ''}"
                
                # Add role description
                if role == "assassin":
                    structure += " (work together, kill at night)"
                elif role == "psychopath":
                    structure += " (works alone, kills at night)"
                elif role == "detective":
                    structure += " (investigates at night)"
                elif role == "angel":
                    structure += " (protects at night)"
                elif role == "villager":
                    structure += " (no night action)"
        
        structure += """

Win conditions:
- Good wins: All evil (assassins + psychopath) eliminated
- Assassins win: Good players ≤ assassins
- Psychopath wins: Be the last one alive"""
        
        return structure
    
    def setup_agents(self, player_names: List[str], role_distribution: Dict[str, int]):
        """Create agents based on configuration"""
        # Store config for building structure
        self.game_config = {
            'players': player_names,
            'roles': role_distribution
        }
        
        # Build the game structure text
        game_structure = self.build_game_structure_text()
        
        # Build role list
        roles = []
        for role, count in role_distribution.items():
            roles.extend([role] * count)
        
        # Shuffle both lists
        names = player_names.copy()
        random.shuffle(names)
        random.shuffle(roles)
        
        # Create agents with game structure
        self.agents = []
        for name, role in zip(names, roles):
            agent = MafiaAgent(name, role, self.llm)
            # Give each agent the actual game structure
            agent.game_structure = game_structure
            self.agents.append(agent)
    
    def get_alive_players(self) -> List[MafiaAgent]:
        """Get all alive, non-imprisoned players"""
        return [a for a in self.agents if a.alive and not a.imprisoned]
    
    def get_agent_by_name(self, name: str) -> MafiaAgent:
        """Find agent by name"""
        return next(a for a in self.agents if a.name == name)
    
    def get_alive_names(self) -> List[str]:
        """Get names of alive players"""
        return [a.name for a in self.get_alive_players()]
