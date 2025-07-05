import random
from typing import List, Optional
from collections import Counter
from src.agent import MafiaAgent

class NightPhase:
    """Handles night phase: kills, protections, investigations"""
    
    def __init__(self, game_state, display):
        self.state = game_state
        self.display = display
        
    def run(self):
        """Execute the night phase"""
        self.display.show_night_start(self.state.round)
        
        alive = self.state.get_alive_players()
        if len(alive) <= 1:
            return
        
        # Collect all night actions
        actions = self._collect_night_actions()
        
        # Resolve actions and announce results
        self._resolve_night_actions(actions)
        
        # Reset ready_to_vote status
        for agent in self.state.agents:
            agent.ready_to_vote = False
    
    def _collect_night_actions(self) -> dict:
        """Collect actions from all night-active roles"""
        alive = self.state.get_alive_players()
        alive_names = self.state.get_alive_names()
        
        actions = {
            'assassin_votes': {},
            'psychopath_kill': None,
            'angel_protect': None,
            'detective_investigate': None
        }
        
        # Assassins coordinate
        assassins_alive = [a for a in alive if a.role == "assassin"]
        if assassins_alive:
            actions['assassin_votes'] = self._get_assassin_votes(assassins_alive, alive_names)
        
        # Other roles act independently
        other_actors = [a for a in alive if a.role in ["psychopath", "angel", "detective"]]
        random.shuffle(other_actors)
        
        for agent in other_actors:
            if agent.role == "psychopath":
                target = self._get_night_action(agent, alive_names, "kill")
                if target:
                    actions['psychopath_kill'] = target
                    agent.remember(f"You targeted {target}")
                    print(f"[SPECTATOR] Psychopath {agent.name} targets {target}")
            
            elif agent.role == "angel":
                target = self._get_night_action(agent, alive_names, "protect")
                if target:
                    actions['angel_protect'] = target
                    agent.remember(f"You protected {target}")
                    print(f"[SPECTATOR] Angel {agent.name} protects {target}")
            
            elif agent.role == "detective":
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

{assassin.get_game_structure()}

Memory:
{assassin.get_memory_context()}

Night {self.state.round}. {coordination_note}
Who can you kill: {', '.join(candidates)}

Choose your target. Reply with just the name: [/INST]"""
        
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

Reply with just the name: [/INST]"""
        
        response = self.state.llm.generate(prompt, max_tokens=15)
        
        # Extract name
        for name in candidates:
            if name.lower() in response.lower():
                return name
        
        return random.choice(candidates)
    
    def _investigate(self, detective, target_name: str):
        """Detective learns if target is good or evil"""
        target = self.state.get_agent_by_name(target_name)
        if target.role in ["assassin", "psychopath"]:
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
            
            if assassin_target == actions['angel_protect']:
                print(f"[SPECTATOR] But {assassin_target} was protected!")
                print(f"\n✨ Someone was protected by the angel!")
            else:
                killed.append(assassin_target)
        
        # Resolve psychopath kill
        if actions['psychopath_kill']:
            if actions['psychopath_kill'] == actions['angel_protect']:
                print(f"[SPECTATOR] Psychopath's target was protected!")
                if actions['psychopath_kill'] not in killed:
                    print(f"\n✨ Someone was protected by the angel!")
            elif actions['psychopath_kill'] not in killed:
                killed.append(actions['psychopath_kill'])
        
        # Announce deaths
        if killed:
            for victim_name in killed:
                victim = self.state.get_agent_by_name(victim_name)
                victim.alive = False
            
            death_msg = f"☠️ Found dead: {', '.join(killed)}"
            print(f"\n{death_msg}")
            
            # Everyone learns who died
            for agent in self.state.get_alive_players():
                agent.remember(f"Night {self.state.round}: {', '.join(killed)} killed")
        else:
            print("\n✨ Nobody died tonight!")
            for agent in self.state.get_alive_players():
                agent.remember(f"Night {self.state.round}: Nobody died")
