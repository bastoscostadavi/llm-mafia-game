# src/agent.py
from typing import List, Dict, Optional, Tuple
import json
import random
import re
import ast
from src.prompts import PromptConfig, MESSAGE_LIMIT

class MafiaAgent:
    def __init__(self, name: str, role: str, llm_interface, debug_prompts: bool = False, prompt_config: PromptConfig = None):
        self.name = name
        self.role = role
        self.llm = llm_interface
        self.alive = True
        self.imprisoned = False
        self.memory = []  # List of all messages and events
        self.debug_prompts = debug_prompts
        self.prompt_config = prompt_config or PromptConfig()
        
    
    def remember(self, event: str):
        """Add event to memory"""
        self.memory.append(event)
    
    def get_memory(self) -> str:
        """Format memory for prompt"""
        if not self.memory:
            return "No previous events."
        return "\n".join(self.memory)

    def get_discussion_message(self, active_players: List[str], round_num: int, all_players: List[str] = None, discussion_rounds: int = 2, game_state=None) -> Dict:
        """Get public discussion message from players"""
        if all_players is None:
            all_players = active_players
            
        other_players = [name for name in all_players if name != self.name]
        other_active_players = [name for name in active_players if name != self.name]
        random.shuffle(other_players)
        
        discussion_prompt = self.prompt_config.format_discussion_prompt(
            name=self.name,
            role=self.role,
            other_players=', '.join(other_players),
            composition=game_state.get_composition_string(),
            memory=self.get_memory(),
            round_num=round_num,
            discussion_rounds=discussion_rounds
        ) 

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
        match = re.search(r'MESSAGE:\s*("((?:\\.|[^"\\])*)")', response)
        if not match:
            return "remained silent."
        
        message = match.group(2)
        
        if len(message) > MESSAGE_LIMIT:
            message = message[:MESSAGE_LIMIT]
        
        return f'"{message}"'

    def vote(self, candidates: List[str], all_players: List[str] = None, discussion_rounds: int = 2, game_state=None) -> str:
        """Vote for someone based on memory"""
        if all_players is None:
            all_players = candidates
            
        other_players = [name for name in all_players if name != self.name]
        random.shuffle(other_players)
        
        voting_prompt = self.prompt_config.format_voting_prompt(
            name=self.name,
            role=self.role,
            other_players=', '.join(other_players),
            composition=game_state.get_composition_string(),
            memory=self.get_memory(),
            candidates=', '.join(candidates)
        )
        
        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"PROMPT FOR {self.name} (VOTING):")
            print(f"{'='*60}")
            print(prompt)
            print(f"{'='*60}")
        
        response = self.llm.generate(voting_prompt, max_tokens=100)
        
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