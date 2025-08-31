# src/agent.py
from typing import List, Dict, Optional
import json
import random
import re
from src.prompt_utils import (
    format_discussion_prompt, format_voting_prompt, format_night_action_prompt,
    parse_discussion_response, parse_voting_response, parse_night_action_response,
    MESSAGE_LIMIT
)
from src.config import TOKEN_LIMITS, get_token_limits_for_model

class MafiaAgent:
    def __init__(self, name: str, role: str, llm_interface, debug_prompts: bool = False, model_config: dict = None):
        self.name = name
        self.role = role
        self.llm = llm_interface
        self.alive = True
        self.imprisoned = False
        self.memory = []  # List of all messages and events
        self.debug_prompts = debug_prompts
        self.model_config = model_config or {}
        self.token_limits = get_token_limits_for_model(self.model_config)
        
    
    def remember(self, event: str):
        """Add event to memory"""
        self.memory.append(event)
    
    def get_memory(self) -> str:
        """Format memory for prompt"""
        if not self.memory:
            return "No previous events."
        return "\n".join(self.memory)
    
    
    def _get_token_limit(self, action_type: str) -> int:
        """Get token limit for action type"""
        return self.token_limits.get(action_type, self.token_limits['discussion'])
    
    def _generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Generate response from LLM (all wrappers have generate method)"""
        return self.llm.generate(prompt, max_tokens)

    def message(self, active_players: List[str], round_num: int, all_players: List[str] = None, discussion_rounds: int = 2, game_state=None) -> Dict:
        """Get public discussion message from players"""
        if all_players is None:
            all_players = active_players
            
        other_players = [name for name in all_players if name != self.name]
        other_active_players = [name for name in active_players if name != self.name]
        random.shuffle(other_players)
        
        discussion_prompt = format_discussion_prompt(
            name=self.name,
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
        
        response = self._generate(discussion_prompt, max_tokens=self._get_token_limit('discussion'))
        
        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"RESPONSE FROM {self.name} (DISCUSSION):")
            print(f"{'='*60}")
            print(repr(response))
            print(f"{'='*60}")
            

        # Parse the message using centralized parsing
        message = parse_discussion_response(response)
        if not message:
            # Log failed discussion
            game_state.log_action("discuss", self.name, response, "remained silent")
            # Debug: always show what the model actually returned when parsing fails
            print(f"[DEBUG] {self.name} failed to parse discussion format. Raw response:")
            print(f"[DEBUG] {repr(response)}")
            return "remained silent."
        
        # Log successful discussion
        game_state.log_action("discuss", self.name, response, message)
        return f'"{message}"'

    def vote(self, candidates: List[str], all_players: List[str] = None, discussion_rounds: int = 2, game_state=None) -> tuple[str, bool]:
        """Vote for someone based on memory
        
        Returns:
            tuple: (vote_target, was_successful_parse)
        """
        if all_players is None:
            all_players = candidates
            
        other_players = [name for name in all_players if name != self.name]
        random.shuffle(other_players)
        
        voting_prompt = format_voting_prompt(
            name=self.name,
            other_players=', '.join(other_players),
            composition=game_state.get_composition_string(),
            memory=self.get_memory(),
            candidates=', '.join(candidates)
        )
        
        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"PROMPT FOR {self.name} (VOTING):")
            print(f"{'='*60}")
            print(voting_prompt)
            print(f"{'='*60}")
        
        response = self._generate(voting_prompt, max_tokens=self._get_token_limit('voting'))
        
        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"RESPONSE FROM {self.name} (VOTING):")
            print(f"{'='*60}")
            print(repr(response))
            print(f"{'='*60}")
        
        # Parse the vote using centralized parsing
        vote_target = parse_voting_response(response, candidates)
        if vote_target:
            # Log successful vote
            game_state.log_action("vote", self.name, response, vote_target)
            if self.debug_prompts:
                print(f"[DEBUG] {self.name} voted for {vote_target}")
            return vote_target, True
        
        # If no valid vote found, cast random vote
        fallback_vote = random.choice(candidates)
        parsed_result = f"{fallback_vote} (random)"
        game_state.log_action("vote", self.name, response, parsed_result)
        if self.debug_prompts:
            print(f"[DEBUG] {self.name} failed to parse vote format, random vote: {fallback_vote}")
        return fallback_vote, False
    
    def kill(self, candidates: List[str], game_state=None) -> str:
        """Choose target to kill"""
        all_players = [a.name for a in game_state.agents]
        other_players = [name for name in all_players if name != self.name]
        random.shuffle(other_players)
        
        killing_prompt = format_night_action_prompt(
            name=self.name,
            other_players=', '.join(other_players),
            composition=game_state.get_composition_string(),
            memory=self.get_memory(),
            round_num=game_state.round,
            action="kill",
            candidates=', '.join(candidates)
        )
        
        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"PROMPT FOR {self.name} (NIGHT KILL):")
            print(f"{'='*60}")
            print(killing_prompt)
            print(f"{'='*60}")
        
        response = self._generate(killing_prompt, max_tokens=self._get_token_limit('night_action'))
        
        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"RESPONSE FROM {self.name} (NIGHT KILL):")
            print(f"{'='*60}")
            print(repr(response))
            print(f"{'='*60}")
        
        # Parse the target using centralized parsing
        target = parse_night_action_response(response, candidates)
        if not target:
            target = random.choice(candidates)
            parsed_result = f"{target} (random)"
            game_state.log_action("kill", self.name, response, parsed_result)
            if self.debug_prompts:
                print(f"[DEBUG] {self.name} failed to parse response, random choice: {target}")
        else:
            game_state.log_action("kill", self.name, response, target)
            if self.debug_prompts:
                print(f"[DEBUG] {self.name} chose {target}")
        
        # Remember the action
        self.remember(f"You killed {target}")
        return target
    
    def investigate(self, candidates: List[str], game_state=None) -> str:
        """Choose target to investigate"""
        all_players = [a.name for a in game_state.agents]
        other_players = [name for name in all_players if name != self.name]
        random.shuffle(other_players)
        
        investigating_prompt = format_night_action_prompt(
            name=self.name,
            other_players=', '.join(other_players),
            composition=game_state.get_composition_string(),
            memory=self.get_memory(),
            round_num=game_state.round,
            action="investigate",
            candidates=', '.join(candidates)
        )
        
        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"PROMPT FOR {self.name} (NIGHT INVESTIGATE):")
            print(f"{'='*60}")
            print(investigating_prompt)
            print(f"{'='*60}")
        
        response = self._generate(investigating_prompt, max_tokens=self._get_token_limit('night_action'))
        
        if self.debug_prompts:
            print(f"\n{'='*60}")
            print(f"RESPONSE FROM {self.name} (NIGHT INVESTIGATE):")
            print(f"{'='*60}")
            print(repr(response))
            print(f"{'='*60}")
        
        # Parse the target using centralized parsing
        target = parse_night_action_response(response, candidates)
        if not target:
            target = random.choice(candidates)
            parsed_result = f"{target} (random)"
            game_state.log_action("investigate", self.name, response, parsed_result)
            if self.debug_prompts:
                print(f"[DEBUG] {self.name} failed to parse response, random choice: {target}")
        else:
            game_state.log_action("investigate", self.name, response, target)
            if self.debug_prompts:
                print(f"[DEBUG] {self.name} chose {target}")
        
        # Remember the investigation result
        target_agent = game_state.get_agent_by_name(target)
        self.remember(f"You investigated {target} and discovered that they are a {target_agent.role}")
        return target