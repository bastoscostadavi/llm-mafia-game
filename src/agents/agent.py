# src/agent.py
from typing import List, Dict, Optional, Tuple
import json
import random

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
        return """Mafia is a social deduction game with secret roles and asymmetric information. The evil mafia competes against the good villagers and detectives. The game alternates between night and day until one side wins.

Night: 
- Assassination: Each night, one mafioso is randomly selected to choose a player to kill. The chosen player is killed.
- Investigation: Each detective independently investigates one player to learn if they are a mafioso. This information is received privately.

Day:
- Discussion: Players speak in a randomly determined order. Each may send a message or remain silent. All messages are public and attributed by name.
- Voting: All players vote to arrest someone. The player with the most votes is arrested. In case of a tie, a tied player is chosen at random.

Game Mechanics:
- Night actions are secret.
- The arrested and killed players do not participate in the game anymore.
- All information disclosed to players appears under "Memory:" in chronological order.
- Memory is an individual atribute of each players.
- At the beginning of the game, the mafiosos learn each other's identities.  
- Night deaths and daytime votes are publicly announced.
- The only communication channel is the public discussion during the day.

Win Conditions:
- Good players: arrest all mafiosos.
- Evil players: kill all good players."""
    
    def get_game_composition(self, game_state) -> str:
        """Get the game composition formatted for prompts"""
        role_counts = game_state.composition
        
        # Build description from stored data
        parts = []
        for role in ['mafioso', 'detective', 'villager']:
            count = role_counts.get(role, 0)
            if count > 0:
                role_word = role + ('s' if count > 1 else '')
                parts.append(f"{count} {role_word}")
        
        return f"In the game you're playing there are: {', and '.join(parts)}."

    def get_discussion_message(self, alive_players: List[str], round_num: int, all_players: List[str] = None, discussion_rounds: int = 2, game_state=None) -> Dict:
        """Get public discussion message from players"""
        if all_players is None:
            all_players = alive_players
            
        other_players = [name for name in all_players if name != self.name]
        random.shuffle(other_players)
        
        prompt = f"""[INST] You're {self.name} playing Mafia with {', '.join(other_players)}.

{self.get_general_game_explanation()}

{self.get_game_composition(game_state)}

Memory:
{self.get_memory_context()}

DISCUSSION TIME {round_num}/{discussion_rounds}: What message do you want to say to everyone? 

Be strategic and consider what you've learned. Your response must contain "MESSAGE: " followed by the message you want to say.

IMPORTANT: 
- Your message must be {MESSAGE_LIMIT} characters or less, and it will be truncated if longer.
- If your response doesn't contain "MESSAGE: ", the message "{self.name} remained silent." will be shared with other players.

[/INST]"""

        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"PROMPT FOR {self.name} (DISCUSSION):")
            print(f"{'='*60}")
            print(prompt)
            print(f"{'='*60}")
        
        response = self.llm.generate(prompt, max_tokens=100)
        
        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"RESPONSE FROM {self.name} (DISCUSSION):")
            print(f"{'='*60}")
            print(repr(response))
            print(f"{'='*60}")
        
        # Parse MESSAGE: format (can appear anywhere in response)
        if response and "MESSAGE:" in response.upper():
            # Find the part after "MESSAGE:"
            message_index = response.upper().find("MESSAGE:")
            message = response[message_index + 8:].strip()  # Get everything after "MESSAGE:"
            
            # Enforce message length limit
            if len(message) > MESSAGE_LIMIT:
                message = message[:MESSAGE_LIMIT].rstrip()  # Truncate and remove trailing whitespace
                message += "..."
            
            return {
                "type": "message", 
                "content": message
            }
        
        # If format doesn't match, remain silent
        if self.debug_prompts:
            print(f"[DEBUG] {self.name} failed to parse MESSAGE: format, remaining silent")
        return {
            "type": "message", 
            "content": f"{self.name} remained silent."
        }
    
    def vote(self, candidates: List[str], all_players: List[str] = None, discussion_rounds: int = 2, game_state=None) -> str:
        """Vote for someone based on memory"""
        if all_players is None:
            all_players = candidates
            
        other_players = [name for name in all_players if name != self.name]
        random.shuffle(other_players)
        
        prompt = f"""[INST] You're {self.name} playing Mafia with {', '.join(other_players)}.

{self.get_general_game_explanation()}

{self.get_game_composition(game_state)}

Memory:
{self.get_memory_context()}

VOTING TIME: Vote to arrest one person from: {', '.join(candidates)}

Be strategic and consider what you've learned. Your response must contain "VOTE: " followed by the name of the person you want to vote for.

IMPORTANT: 
- If your response doesn't contain "VOTE: " followed by the name of the person you want to vote for, the vote will be cast for a random person.

[/INST]"""
        
        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"PROMPT FOR {self.name} (VOTING):")
            print(f"{'='*60}")
            print(prompt)
            print(f"{'='*60}")
        
        response = self.llm.generate(prompt, max_tokens=100)
        
        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"RESPONSE FROM {self.name} (VOTING):")
            print(f"{'='*60}")
            print(repr(response))
            print(f"{'='*60}")
        
        # Extract vote using "VOTE: player_name" format
        for candidate in candidates:
            vote_pattern = f"VOTE: {candidate}"
            if vote_pattern.upper() in response.upper():
                if self.debug_prompts:
                    print(f"[DEBUG] {self.name} voted for {candidate} using pattern '{vote_pattern}'")
                return candidate
        
        # If no valid "VOTE: name" found, cast random vote
        fallback_vote = random.choice(candidates)
        if self.debug_prompts:
            print(f"[DEBUG] {self.name} failed to parse VOTE: format, random vote: {fallback_vote}")
        return fallback_vote