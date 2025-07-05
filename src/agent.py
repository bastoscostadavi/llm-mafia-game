# src/agent.py
from typing import List, Dict, Optional, Tuple
import json

class MafiaAgent:
    def __init__(self, name: str, role: str, llm):
        self.name = name
        self.role = role
        self.llm = llm
        self.alive = True
        self.imprisoned = False
        self.memory = []  # List of all messages and events
        self.ready_to_vote = False
        
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
        return """Game Structure (public knowledge):
    - 8 players total
    - 2 assassins (work together, kill at night)
    - 1 psychopath (works alone, kills at night)  
    - 1 detective (investigates at night)
    - 1 angel (protects at night)
    - 3 villagers (no night action)

    Win conditions:
    - Good wins: All evil (assassins + psychopath) eliminated
    - Assassins win: Good players ≤ assassins
    - Psychopath wins: Be the last one alive"""

    def decide_action(self, alive_players: List[str], round_num: int) -> Dict:
        """Decide whether to send message or vote"""
        prompt = f"""[INST] You are {self.name}. {self.get_role_rules()}

    {self.get_game_structure()}

    Memory:
    {self.get_memory_context()}

    Alive players: {', '.join(alive_players)}
    Discussion round: {round_num}/2

    You can either:
    1. Send a message (public or private)
    2. Declare ready to vote (if round 2)

    Respond in JSON format:
    {{"action": "message", "to": ["name1", "name2"] or "all", "message": "your message"}}
    OR
    {{"action": "vote"}}

    Be strategic. As {self.role}, what should you do? [/INST]"""

        response = self.llm.generate(prompt, max_tokens=100)
        
        # Parse response
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except:
            pass
        
        # Fallback
        if round_num == 2:
            return {"action": "vote"}
        else:
            return {
                "action": "message",
                "to": "all",
                "message": f"I think we should be careful about our decision."
            }
    
    def vote(self, candidates: List[str]) -> str:
        """Vote for someone based on memory"""
        prompt = f"""[INST] You are {self.name}. {self.get_role_rules()}

Memory:
{self.get_memory_context()}

Vote to arrest ONE person from: {', '.join(candidates)}

Consider what you've learned. Reply with just the name: [/INST]"""
        
        response = self.llm.generate(prompt, max_tokens=10)
        
        # Extract name
        for candidate in candidates:
            if candidate.lower() in response.lower():
                return candidate
        return candidates[0]