# src/agent.py
from typing import List, Dict, Optional, Tuple
import json
import random
import re
import ast

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
    
    def get_memory(self) -> str:
        """Format memory for prompt"""
        if not self.memory:
            return "No previous events."
        return "\n".join(self.memory)
    
    def get_general_game_explanation(self) -> str:
        """Detailed general explanation of Mafia game"""
        return """Mafia is a social deduction game where players are secretly assigned roles. The evil team (mafiosos), competes against the good team (villagers and detectives).
GAME DYNAMICS: The game alternates between night and day until one team wins.
1. Night:
    - Assassination: Each night, one mafioso is randomly selected to choose an active player (not killed or arrested) to kill. The chosen player is killed.
    - Investigation: Each detective independently investigates one active player to learn if they are a mafioso. This information is received privately.
2. Day:
    - Discussion: In each round, all active players speak in a randomized order. Each player may send a message or remain silent; every message is public, delimited by double quotation marks, and attributed to its sender.
    - Voting: all active players cast a vote to arrest one player. The player with the most votes is arrested. In case of a tie, a tied player is chosen at random.
GAME MECHANICS AND INFORMATION FLOW:
- The arrested and killed players do not participate in the game anymore.
- All information disclosed to players appears under "MEMORY:" in chronological order.
- Memory is an individual atribute of each players, creating an assymetry of information.
- All players know the total number of each role in the game.
- Detectives learn the exact roles of players they investigate.
- At the beginning of the game, the mafiosos learn each other's identities.
- Night deaths and daytime votes are publicly announced.
- Voting results are public - all players see who voted for whom. 
- There is no private messaging, the only communication channel is the public discussion during the day. 
WIN CONDITIONS:
- Good team: arrest all mafiosos.
- Evil team: kill all good players.""" 
    
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
        
        return f"In the game you're playing there are: {', '.join(parts)}."


    def get_discussion_message(self, active_players: List[str], round_num: int, all_players: List[str] = None, discussion_rounds: int = 2, game_state=None) -> Dict:
        """Get public discussion message from players"""
        if all_players is None:
            all_players = active_players
            
        other_players = [name for name in all_players if name != self.name]
        other_active_players = [name for name in active_players if name != self.name]
        random.shuffle(other_players)
        
        discussion_prompt = f"""*CONTEXT*:        
You're {self.name} playing Mafia with: {', '.join(other_players)}. You're the {self.role} and you want to win the game.


*GENERAL GAME EXPLANATION*:
{self.get_general_game_explanation()}


*GAME COMPOSITION*:
{self.get_game_composition(game_state)} 


*MEMORY*:
{self.get_memory()}


*DISCUSSION TIME {round_num}/{discussion_rounds}*: 
What message do you want to say to everyone? 
Be strategic and consider what you've learned. Do your best to win the game. 
Your response must contain 'MESSAGE: ' followed by the message you want to say delimited by double quotation marks. Example: response = '... MESSAGE: "the message you want to say" ...'.
Important:
- Your message must be {MESSAGE_LIMIT} characters or fewer. Longer messages will be truncated.
- If your response does not contain MESSAGE: followed by a quoted message, the message = "{self.name}: remained silent." will be shared with the other players.""" 

        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"PROMPT FOR {self.name} (DISCUSSION):")
            print(f"{'='*60}")
            print(discussion_prompt)
            print(f"{'='*60}")
        
        response = self.llm.generate(discussion_prompt, max_tokens=100)
        
        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"RESPONSE FROM {self.name} (DISCUSSION):")
            print(f"{'='*60}")
            print(repr(response))
            print(f"{'='*60}")
            

        #parse the message    
        message = re.search(r'MESSAGE:\s*("((?:\\.|[^"\\])*)")', response)
        if not message:
            return "remained silent"
        
        message = message.group(2)
        
        if len(message) > MESSAGE_LIMIT:
            message = message[:MESSAGE_LIMIT]
        
        return f'"{message}"'


    def vote(self, candidates: List[str], all_players: List[str] = None, discussion_rounds: int = 2, game_state=None) -> str:
        """Vote for someone based on memory"""
        if all_players is None:
            all_players = candidates
            
        other_players = [name for name in all_players if name != self.name]
        random.shuffle(other_players)
        
        prompt = f"""[INST]*CONTEXT*:
You're {self.name} playing Mafia with: {','.join(other_players)}. You're the {self.role} and you want to win the game.


*GENERAL GAME EXPLANATION*:
{self.get_general_game_explanation()}


*GAME COMPOSITION*:
{self.get_game_composition(game_state)}


*MEMORY*:
{self.get_memory()}


*VOTING TIME*: 
Vote to arrest one person from: {', '.join(candidates)}.
Be strategic and consider what you've learned. Do your best to win the game.
Your response must contain "VOTE: " followed by the name of the person you want to vote for.
Important:
- If your response does not contain "VOTE: " followed by a name, the vote will be cast for a random person.[/INST]"""
        
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