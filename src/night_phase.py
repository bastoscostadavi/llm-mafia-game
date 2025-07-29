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
            'assassin_votes': {},
            'detective_investigate': None
        }
        
        # Assassins coordinate
        assassins_alive = [a for a in alive if a.role == "assassin"]
        if assassins_alive:
            actions['assassin_votes'] = self._get_assassin_votes(assassins_alive, alive_names)
        
        # Detective acts independently
        detective_actors = [a for a in alive if a.role == "detective"]
        
        for agent in detective_actors:
            if agent.role == "detective":
                target = self._get_night_action(agent, alive_names, "investigate")
                if target:
                    actions['detective_investigate'] = (agent, target)
                    self._investigate(agent, target)
        
        return actions
    
    def _get_assassin_votes(self, assassins: List, alive_names: List[str]) -> dict:
        """Assassins coordinate their kill"""
        votes = {}
        
        if len(assassins) > 1:
            print("\n[SPECTATOR] Assassins are coordinating...")
            for assassin in assassins:
                target = self._get_assassin_target(assassin, alive_names, assassins)
                if target:
                    votes[assassin.name] = target
                    print(f"[SPECTATOR] {assassin.name} wants to kill {target}")
        else:
            # Single assassin
            assassin = assassins[0]
            target = self._get_night_action(assassin, alive_names, "kill")
            if target:
                votes[assassin.name] = target
                print(f"[SPECTATOR] {assassin.name} (alone) targets {target}")
        
        return votes
    
    def _get_assassin_target(self, assassin, alive_names: List[str], all_assassins: List) -> Optional[str]:
        """Individual assassin chooses target"""
        candidates = [n for n in alive_names if n != assassin.name and 
                      n not in [a.name for a in all_assassins]]
        
        if not candidates:
            return None
        
        # Let assassin reason based on their memory
        if len(all_assassins) > 1:
            other_assassins = [a.name for a in all_assassins if a.name != assassin.name]
            coordination_note = f"You need to coordinate with {', '.join(other_assassins)}."
        else:
            coordination_note = "You are the only assassin left."
        
        prompt = f"""[INST] You are {assassin.name}. {assassin.get_role_rules()}

Memory:
{assassin.get_memory_context()}

Night {self.state.round}. {coordination_note}
Who can you kill: {', '.join(candidates)}

IMPORTANT: Reply with just the name. If you don't follow this format, a random target will be chosen for you.

Choose your target. Reply with just the name: [/INST]"""
        
        if assassin.debug_prompts:
            print(f"\n{'='*60}")
            print(f"PROMPT FOR {assassin.name} (NIGHT KILL):")
            print(f"{'='*60}")
            print(prompt)
            print(f"{'='*60}")
        
        response = self.state.llm.generate(prompt, max_tokens=15)
        
        # Extract name
        for name in candidates:
            if name.lower() in response.lower():
                return name
        
        return random.choice(candidates)
    
    def _get_night_action(self, agent, alive_names: List[str], action: str) -> Optional[str]:
        """Get night action from non-assassin roles"""
        candidates = [n for n in alive_names if n != agent.name]
        
        if not candidates:
            return None
        
        prompt = f"""[INST] You are {agent.name}. {agent.get_role_rules()}

Memory:
{agent.get_memory_context()}

Night {self.state.round}. Choose someone to {action}.
Options: {', '.join(candidates)}

IMPORTANT: Reply with just the name. If you don't follow this format, a random choice will be made for you.

Reply with just the name: [/INST]"""
        
        if agent.debug_prompts:
            print(f"\n{'='*60}")
            print(f"PROMPT FOR {agent.name} (NIGHT {action.upper()}):")
            print(f"{'='*60}")
            print(prompt)
            print(f"{'='*60}")
        
        response = self.state.llm.generate(prompt, max_tokens=15)
        
        # Extract name
        for name in candidates:
            if name.lower() in response.lower():
                return name
        
        return random.choice(candidates)
    
    def _investigate(self, detective, target_name: str):
        """Detective learns if target is good or evil"""
        target = self.state.get_agent_by_name(target_name)
        if target.role in ["assassin"]:
            result = "evil"
        else:
            result = "good"
        
        detective.remember(f"You investigated {target_name}: {result}")
        print(f"[SPECTATOR] Detective {detective.name} investigates {target_name} (learns: {result})")
    
    def _resolve_night_actions(self, actions: dict):
        """Resolve all night actions and announce deaths"""
        print("\n[SPECTATOR] Resolving night actions...")
        killed = []
        
        # Resolve assassin kill
        if actions['assassin_votes']:
            vote_counts = Counter(actions['assassin_votes'].values())
            assassin_target = vote_counts.most_common(1)[0][0]
            print(f"[SPECTATOR] Assassins agreed on: {assassin_target}")
            killed.append(assassin_target)
        
        
        # Announce deaths
        if killed:
            for victim_name in killed:
                victim = self.state.get_agent_by_name(victim_name)
                victim.alive = False
            
            death_msg = f"Found dead: {', '.join(killed)}"
            print(f"\n{death_msg}")
            
            # Assassins remember their kill action first
            assassins_alive = [a for a in self.state.agents if a.role == "assassin" and a.alive]
            if assassins_alive and actions['assassin_votes']:
                for assassin in assassins_alive:
                    # Get other assassins (exclude self)
                    other_assassin_names = [a.name for a in assassins_alive if a != assassin]
                    if other_assassin_names:
                        kill_memory = f"You, {', '.join(other_assassin_names)}, killed {', '.join(killed)}."
                    else:
                        kill_memory = f"You killed {', '.join(killed)}."
                    assassin.remember(kill_memory)
            
            # Everyone learns who died (including assassins, but this comes after their kill memory)
            for agent in self.state.get_alive_players():
                agent.remember(f"Night {self.state.round}: {', '.join(killed)} was found dead")
        else:
            print("\nNobody died tonight!")
            for agent in self.state.get_alive_players():
                agent.remember(f"Night {self.state.round}: Nobody died")
