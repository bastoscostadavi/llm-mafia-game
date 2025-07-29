# src/agent.py
from typing import List, Dict, Optional, Tuple
import json

# Configuration constants
MESSAGE_LIMIT = 200  # Character limit for discussion messages

class MafiaAgent:
    def __init__(self, name: str, role: str, llm_interface, debug_prompts: bool = False):
        self.name = name
        self.role = role
        self.llm = llm_interface
        self.alive = True
        self.imprisoned = False
        self.memory = []  # List of all messages and events
        self.debug_prompts = debug_prompts
        
    def get_role_rules(self, discussion_rounds=2) -> str:
        rules = {
            "assassin": f"You're an assassin. Night: assassins vote to kill. Day: discuss for {discussion_rounds} rounds and then vote to arrest. Win: kill all good players.",
            "detective": f"You're a detective. Night: investigate someone (learn if good/evil). Day: discuss for {discussion_rounds} rounds and then vote to arrest. Win: arrest all assassins.",
            "villager": f"You're a villager. Day: discuss for {discussion_rounds} rounds and then vote to arrest. Win: arrest all assassins."
        }
        return rules.get(self.role, rules["villager"])
    
    def remember(self, event: str):
        """Add event to memory"""
        self.memory.append(event)
    
    def get_memory_context(self) -> str:
        """Format memory for prompt"""
        if not self.memory:
            return "No previous events."
        return "\n".join(self.memory)
    
    def get_general_game_explanation(self) -> str:
        """Detailed general explanation of Mafia game"""
        return """Mafia is a social deduction game where players have secret roles and assymetric information. Detectives and villagers are the good players; assassins are the evil players. The game alternates between night phases and day phases until one side wins.

Night Phase: 
- Assassination: Each assassin independently casts a vote to kill one player. The player who receives the most assassin votes is killed. If assassin votes tie, a tied player is killed at random.
- Investigation: Each detective independetly investigates one player to learn if they are an assassin. The detective receives this information privately.

Day Phase:
- Discussion: Players are prompted to talk in an order that is set at random each round. Each player can send messages or remain silent. Messages are visible to everyone along with the player's name.
- Voting: All living players vote to arrest one player. Each player casts one vote. The player who receives the most votes is arrested. If there is a tie, a tied player is arrested at random.

Game Mechanics:
- Night actions are secret.
- All information disclosed to agents appears under Memory: in chronological order.
- At the beginning of the game, the assassins learn each other's identities.

Win Conditions:
- Good players win by arresting all assassins.
- Evil players win by killing all good players."""
    
    def get_game_composition(self, all_players) -> str:
        """Specific composition for this game instance"""
        if hasattr(self, 'game_composition'):
            return self.game_composition
        
        # Count roles from all players (including dead ones)
        from collections import Counter
        
        # We need to get the actual agents to count roles
        # This is a bit of a hack, but we'll count based on typical game setups
        total_players = len(all_players)
        
        if total_players == 3:  # Mini game after 1 death
            return "In the game you're playing there are: 1 assassin, 1 detective, and 1 villager."
        elif total_players == 4:  # Mini game before death
            return "In the game you're playing there are: 1 assassin, 1 detective, and 2 villagers."
        elif total_players == 6:  # Classic game
            return "In the game you're playing there are: 2 assassins, 1 detective, and 3 villagers."
        else:
            # Generic fallback
            return f"In the game you're playing there are {total_players} players with various roles."

    def decide_action(self, alive_players: List[str], round_num: int, all_players: List[str] = None, discussion_rounds: int = 2) -> Dict:
        """Get public message from agent"""
        if all_players is None:
            all_players = alive_players
            
        other_players = [name for name in all_players if name != self.name]
        
        prompt = f"""[INST] You're {self.name} playing Mafia with {', '.join(other_players)}.

{self.get_general_game_explanation()}

{self.get_game_composition(all_players)}

Memory:
{self.get_memory_context()}

Discussion round: {round_num}/{discussion_rounds}

What do you want to say publicly to everyone? Start your response with "MESSAGE: " followed by what you want to say.

IMPORTANT: 
- Your message must be {MESSAGE_LIMIT} characters or less (excluding "MESSAGE: ")
- If your response doesn't start with "MESSAGE: ", the message "{self.name} remained silent" will be shared with other players.
- Messages longer than {MESSAGE_LIMIT} characters will be truncated

[/INST]"""

        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"PROMPT FOR {self.name} (DISCUSSION):")
            print(f"{'='*60}")
            print(prompt)
            print(f"{'='*60}")
        
        response = self.llm.generate(prompt, max_tokens=100)
        
        # Parse simple MESSAGE: format
        if response and response.strip().startswith("MESSAGE:"):
            # Extract everything after "MESSAGE:"
            message = response.strip()[8:].strip()  # Remove "MESSAGE:" and whitespace
            
            # Enforce message length limit
            if len(message) > MESSAGE_LIMIT:
                message = message[:MESSAGE_LIMIT].rstrip()  # Truncate and remove trailing whitespace
                # Try to truncate at word boundary if possible
                if not message.endswith('.') and not message.endswith('!') and not message.endswith('?'):
                    last_space = message.rfind(' ')
                    if last_space > MESSAGE_LIMIT * 0.8:  # Only truncate at word if we don't lose too much
                        message = message[:last_space]
                message += "..."  # Indicate truncation
            
            return {
                "action": "message", 
                "message": message
            }
        
        # If format doesn't match, remain silent
        return {
            "action": "message", 
            "message": ""
        }
    
    def vote(self, candidates: List[str], all_players: List[str] = None, discussion_rounds: int = 2) -> str:
        """Vote for someone based on memory"""
        if all_players is None:
            all_players = candidates
            
        other_players = [name for name in all_players if name != self.name]
        
        prompt = f"""[INST] You're {self.name} playing Mafia with {', '.join(other_players)}.

{self.get_general_game_explanation()}

{self.get_game_composition(all_players)}

Memory:
{self.get_memory_context()}

VOTING TIME: Vote to arrest ONE person from: {', '.join(candidates)}

IMPORTANT: Reply with just the name. If you don't follow this format, a random vote will be cast for you.

Consider what you've learned. Reply with just the name: [/INST]"""
        
        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"PROMPT FOR {self.name} (VOTING):")
            print(f"{'='*60}")
            print(prompt)
            print(f"{'='*60}")
        
        response = self.llm.generate(prompt, max_tokens=10)
        
        # Extract name
        for candidate in candidates:
            if candidate.lower() in response.lower():
                return candidate
        
        # If no valid name found, cast random vote
        import random
        return random.choice(candidates)