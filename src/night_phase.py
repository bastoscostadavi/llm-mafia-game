import random
from typing import List, Optional
from collections import Counter
from src.agents.agent import MafiaAgent

class NightPhase:
    """Handles night phase: kills, protections, investigations"""
    
    def __init__(self, game_state, display):
        self.state = game_state
        self.display = display
        
    def run(self):
        """Execute the night phase"""
        active_players = self.state.get_active_players()
        if len(active_players) <= 1:
            return
        
        # Process all night actions
        self._process_night_actions()
    
    def _process_night_actions(self):
        """Process all night actions: kills and investigations"""
        active_players = self.state.get_active_players()
        active_names = self.state.get_active_names()
        
        # Detective acts independently
        active_detectives = [a for a in active_players if a.role == "detective"]
        for active_detective in active_detectives:
            target = self._get_night_action(active_detective, active_names, "investigate")
            if target:
                self._investigate(active_detective, target)
                
        # One random mafioso chooses kill target
        mafiosos_active = [a for a in active_players if a.role == "mafioso"]
        if mafiosos_active:
            chosen_mafioso = random.choice(mafiosos_active)
            target = self._get_night_action(chosen_mafioso, active_names, "kill")
            if target:
                self._kill(chosen_mafioso, target)
        

    
    
    def _get_night_action(self, agent, active_names: List[str], action: str) -> Optional[str]:
        """Get night action"""
        candidates = [n for n in active_names if n != agent.name]
        
        if not candidates:
            return None
        
        # Get all players for context
        all_players = [a.name for a in self.state.agents]
        other_players = [name for name in all_players if name != agent.name]
        
        prompt = agent.prompt_config.format_night_action_prompt(
            name=agent.name,
            role=agent.role,
            other_players=', '.join(other_players),
            composition=self.state.get_composition_string(),
            memory=agent.get_memory(),
            round_num=self.state.round,
            action=action,
            candidates=', '.join(candidates)
        )
        
        if agent.debug_prompts:
            print(f"\n{'='*60}")
            print(f"PROMPT FOR {agent.name} (NIGHT {action.upper()}):")
            print(f"{'='*60}")
            print(prompt)
            print(f"{'='*60}")
        
        response = self.state.llm.generate(prompt, max_tokens=15)
        
        if agent.debug_prompts:
            print(f"\n{'='*60}")
            print(f"RESPONSE FROM {agent.name} (NIGHT {action.upper()}):")
            print(f"{'='*60}")
            print(repr(response))
            print(f"{'='*60}")
        
        # Extract name
        for name in candidates:
            if name.lower() in response.lower():
                if agent.debug_prompts:
                    print(f"[DEBUG] {agent.name} chose {name} from response parsing")
                return name
        
        # If no valid name found, make random choice
        fallback_choice = random.choice(candidates)
        if agent.debug_prompts:
            print(f"[DEBUG] {agent.name} failed to parse response, random choice: {fallback_choice}")
        return fallback_choice
    
    def _kill(self, chosen_mafioso, target_name: str):
        """Mafioso kills target and updates memories"""
        # Kill the target
        victim = self.state.get_agent_by_name(target_name)
        victim.alive = False
        
        print(f"[SPECTATOR] {chosen_mafioso.name} (chosen mafioso) kills {target_name}")
        print(f"{target_name} was found dead.")
        
        # All mafiosos learn about the kill
        mafiosos_alive = [a for a in self.state.agents if a.role == "mafioso" and a.alive]
        for mafioso in mafiosos_alive:
            if mafioso == chosen_mafioso:
                kill_memory = f"You killed {target_name}"
            else:
                kill_memory = f"The Mafia member {chosen_mafioso.name} killed {target_name}"
            mafioso.remember(kill_memory)
        
        # Everyone learns who died
        for agent in self.state.get_active_players():
            agent.remember(f"Night {self.state.round}: {target_name} was found dead.")
    
    def _investigate(self, detective, target_name: str):
        """Detective learns if target is good or evil"""
        target = self.state.get_agent_by_name(target_name)
        detective.remember(f"You investigated {target_name} and discovered that they are a {target.role}")
        print(f"[SPECTATOR] Detective {detective.name} investigates {target_name} and learns that they are a {target.role})")
    
