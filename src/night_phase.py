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
        alive = self.state.get_alive_players()
        if len(alive) <= 1:
            return
        
        # Collect all night actions
        actions = self._collect_night_actions()
        
        # Resolve actions and announce results
        self._resolve_night_actions(actions)
    
    def _collect_night_actions(self) -> dict:
        """Collect actions from all night-active roles"""
        alive = self.state.get_alive_players()
        alive_names = self.state.get_alive_names()
        
        actions = {
            'mafioso_kill': None,
            'detective_investigate': None
        }
        
        # One random mafioso chooses kill target
        mafiosos_alive = [a for a in alive if a.role == "mafioso"]
        if mafiosos_alive:
            chosen_mafioso = random.choice(mafiosos_alive)
            target = self._get_night_action(chosen_mafioso, alive_names, "kill")
            if target:
                actions['mafioso_kill'] = (chosen_mafioso, target)
                print(f"[SPECTATOR] {chosen_mafioso.name} (chosen mafioso) targets {target}")
        
        # Detective acts independently
        detective_actors = [a for a in alive if a.role == "detective"]
        
        for agent in detective_actors:
            if agent.role == "detective":
                target = self._get_night_action(agent, alive_names, "investigate")
                if target:
                    actions['detective_investigate'] = (agent, target)
                    self._investigate(agent, target)
        
        return actions
    
    
    def _get_night_action(self, agent, alive_names: List[str], action: str) -> Optional[str]:
        """Get night action from non-mafioso roles"""
        candidates = [n for n in alive_names if n != agent.name]
        
        if not candidates:
            return None
        
        # Get all players for context
        all_players = [a.name for a in self.state.agents]
        other_players = [name for name in all_players if name != agent.name]
        
        prompt = f"""[INST] You're {agent.name} playing Mafia with {', '.join(other_players)}.

{agent.get_general_game_explanation()}

{agent.get_game_composition(self.state)}

Memory:
{agent.get_memory_context()}

Night {self.state.round}. Choose a player to {action}: {', '.join(candidates)}

Reply with just the name (otherwise a random choice will be made for you): [/INST]"""
        
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
        random_choice = random.choice(candidates)
        if agent.debug_prompts:
            print(f"[DEBUG] {agent.name} failed to parse response, random choice: {random_choice}")
        return random_choice
    
    def _investigate(self, detective, target_name: str):
        """Detective learns if target is good or evil"""
        target = self.state.get_agent_by_name(target_name)
        if target.role in ["mafioso"]:
            result = "evil"
        else:
            result = "good"
        
        detective.remember(f"You investigated {target_name}: {result}")
        print(f"[SPECTATOR] Detective {detective.name} investigates {target_name} (learns: {result})")
    
    def _resolve_night_actions(self, actions: dict):
        """Resolve all night actions and announce deaths"""
        print("\n[SPECTATOR] Resolving night actions...")
        killed = []
        
        # Resolve mafioso kill
        if actions['mafioso_kill']:
            chosen_mafioso, target = actions['mafioso_kill']
            killed.append(target)
        
        
        # Announce deaths
        if killed:
            for victim_name in killed:
                victim = self.state.get_agent_by_name(victim_name)
                victim.alive = False
            
            death_msg = f"Found dead: {', '.join(killed)}"
            print(f"\n{death_msg}")
            
            # Mafiosi remember the kill action
            if actions['mafioso_kill']:
                chosen_mafioso, target = actions['mafioso_kill']
                # All mafiosos learn about the kill, but only the chosen one remembers making it
                mafiosos_alive = [a for a in self.state.agents if a.role == "mafioso" and a.alive]
                for mafioso in mafiosos_alive:
                    if mafioso == chosen_mafioso:
                        kill_memory = f"You killed {target}"
                    else:
                        kill_memory = f"The Mafia member {chosen_mafioso.name} killed {target}"
                    mafioso.remember(kill_memory)
            
            # Everyone learns who died (including mafiosos, but this comes after their kill memory)
            for agent in self.state.get_alive_players():
                agent.remember(f"Night {self.state.round}: {', '.join(killed)} was found dead")
        else:
            print("\nNobody died tonight!")
            for agent in self.state.get_alive_players():
                agent.remember(f"Night {self.state.round}: Nobody died")
