# src/agent.py
from typing import List, Dict, Optional, Tuple
import json

class MafiaAgent:
    def __init__(self, name: str, role: str, llm_interface):
        self.name = name
        self.role = role
        self.llm = llm_interface
        self.alive = True
        self.imprisoned = False
        self.memory = []  # List of all messages and events
        
    def get_role_rules(self) -> str:
        rules = {
            "assassin": "You're an assassin. Night: vote to kill. Day: vote to arrest. Win: kill all good players.",
            "detective": "You're a detective. Night: investigate someone (learn if good/evil). Day: vote to arrest. Win: arrest all assassins.",
            "angel": "You're an angel. Night: protect someone from assassination. Day: vote to arrest. Win: arrest all assassins.",
            "villager": "You're a villager. Day: vote to arrest. Win: arrest all assassins.",
            "psychopath": "You're a psychopath. Night: kill someone. Day: vote to arrest. Win: be the last one alive. You work alone."
        }
        return rules.get(self.role, rules["villager"])
    
    def remember(self, event: str):
        """Add event to memory"""
        self.memory.append(event)
    
    def get_memory_context(self) -> str:
        """Format recent memory for prompt"""
        if not self.memory:
            return "No previous events."
        # Last 10 events
        recent = self.memory[-10:]
        return "\n".join(recent)
    
    def get_game_structure(self) -> str:
        """Full game structure that everyone knows"""
        # This needs to be set by the game when agents are created
        if hasattr(self, 'game_structure'):
            return self.game_structure
        
        # Fallback to generic description
        return """Game Structure (public knowledge):
    - Multiple players with secret roles
    - Evil roles (assassins/psychopath) kill at night
    - Detective investigates at night
    - Angel protects at night
    - Villagers have no night action

    Win conditions:
    - Good wins: All evil eliminated
    - Assassins win: Good players ≤ assassins
    - Psychopath wins: Be the last one alive"""

    def decide_action(self, alive_players: List[str], round_num: int) -> Dict:
        """Get public message from agent"""
        prompt = f"""[INST] You are {self.name}. {self.get_role_rules()}

    {self.get_game_structure()}

    Memory:
    {self.get_memory_context()}

    Alive players: {', '.join(alive_players)}
    Discussion round: {round_num}/2

    What do you want to say publicly to everyone? Start your response with "MESSAGE: " followed by what you want to say.

    Example: MESSAGE: I think we should be careful about our decision.

    IMPORTANT: If your response doesn't start with "MESSAGE: ", the system will assume you remained silent.

    [/INST]"""

        response = self.llm.generate(prompt, max_tokens=100)
        
        # Parse simple MESSAGE: format
        if response and response.strip().startswith("MESSAGE:"):
            # Extract everything after "MESSAGE:"
            message = response.strip()[8:].strip()  # Remove "MESSAGE:" and whitespace
            return {
                "action": "message", 
                "message": message
            }
        
        # If format doesn't match, remain silent
        return {
            "action": "message", 
            "message": ""
        }
    
    def vote(self, candidates: List[str]) -> str:
        """Vote for someone based on memory"""
        prompt = f"""[INST] You are {self.name}. {self.get_role_rules()}

Memory:
{self.get_memory_context()}

Vote to arrest ONE person from: {', '.join(candidates)}

IMPORTANT: Reply with just the name. If you don't follow this format, a random vote will be cast for you.

Consider what you've learned. Reply with just the name: [/INST]"""
        
        response = self.llm.generate(prompt, max_tokens=10)
        
        # Extract name
        for candidate in candidates:
            if candidate.lower() in response.lower():
                return candidate
        
        # If no valid name found, cast random vote
        import random
        return random.choice(candidates)