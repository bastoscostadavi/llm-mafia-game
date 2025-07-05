import random
from typing import List
from src.agent import MafiaAgent
from src.llm_interface import LlamaCppInterface

class GameState:
    """Manages all game state and agents"""
    
    def __init__(self, model_path: str):
        self.llm = LlamaCppInterface(model_path)
        self.agents: List[MafiaAgent] = []
        self.round = 0
        self.message_log = []
        
    def setup_agents(self):
        """Create 8 agents with roles"""
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
        roles = ["assassin", "assassin", "psychopath", "angel", "detective", 
                "villager", "villager", "villager"]
        
        random.shuffle(names)
        random.shuffle(roles)
        
        for name, role in zip(names, roles):
            agent = MafiaAgent(name, role, self.llm)
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
