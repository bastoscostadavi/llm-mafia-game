# src/main.py - Simple Mafia Game System
import sys
import random
from collections import Counter
from src.agents import MafiaAgent
from src.prompts import PromptConfig
from src.llm_utils import create_llm

sys.path.append('.')

class GameState:
    """Manages the state of a Mafia game"""
    
    def __init__(self, agents, discussion_rounds):
        self.agents = agents
        self.round = 0
        self.discussion_rounds = discussion_rounds
        self.message_log = []
        # Use first agent's LLM as fallback for phases that expect state.llm
        self.llm = agents[0].llm if agents else None
        # Calculate and store game composition once  
        self.composition = Counter(agent.role for agent in agents)
    
    def get_alive_players(self):
        return [a for a in self.agents if a.alive]
    
    def get_active_players(self):
        return [a for a in self.agents if a.alive and not a.imprisoned]
    
    def get_agent_by_name(self, name):
        return next(a for a in self.agents if a.name == name)
    
    def get_alive_names(self):
        return [a.name for a in self.get_alive_players()]
    
    def get_active_names(self):
        return [a.name for a in self.get_active_players()]
    
    def get_composition_string(self) -> str:
        """Get the game composition formatted for prompts"""
        parts = []
        for role in ['mafioso', 'detective', 'villager']:
            count = self.composition.get(role, 0)
            if count > 0:
                role_word = role + ('s' if count > 1 else '')
                parts.append(f"{count} {role_word}")
        
        return ', '.join(parts)

class Game:
    """Main game controller that manages the game flow"""
    
    def __init__(self, state):
        self.state = state
    
    def play(self):
        print("Initializing Mafia Game...")
        self.show_roles()
        
        while self.state.round < 10:  # Max rounds
            self.state.round += 1
            
            # Night Phase
            print(f"\nNIGHT {self.state.round}")
            self.run_night_phase()
            
            # Check win conditions
            result = self.check_game_over()
            if result:
                self.show_game_end(result)
                return
            
            # Day Phase
            print(f"\n{'='*50}")
            print(f"DAY {self.state.round}")
            print(f"{'='*50}")
            self.show_status()
            self.run_day_phase()
            
            # Check win conditions
            result = self.check_game_over()
            if result:
                self.show_game_end(result)
                return
        
        print(f"\nGame ended due to timeout!")
        self.show_final_roles()
    
    def run_night_phase(self):
        """Execute night phase"""
        active_players = self.state.get_active_players()
        if len(active_players) <= 1:
            return
        
        # Announce night phase start to all players
        for agent in self.state.agents:
            agent.remember(f"Night {self.state.round} begins.")
        
        killed_player = self.night_actions()
        
        # Announce deaths before day starts
        if killed_player:
            print(f"{killed_player} was found dead.")
            # Everyone learns who died
            for agent in self.state.agents:
                agent.remember(f"{killed_player} was found dead.")
    
    def night_actions(self):
        """Process all night actions: kills and investigations"""
        active_players = self.state.get_active_players()
        active_names = self.state.get_active_names()
        candidates = [n for n in active_names]
        killed_player = None
        
        # Detective acts independently
        active_detectives = [a for a in active_players if a.role == "detective"]
        for active_detective in active_detectives:
            detective_candidates = [n for n in candidates if n != active_detective.name]
            if detective_candidates:
                target = active_detective.investigate(detective_candidates, self.state)
                print(f"[SPECTATOR] Detective {active_detective.name} investigates {target} and learns their role")
                
        # One random mafioso chooses kill target
        mafiosos_active = [a for a in active_players if a.role == "mafioso"]
        if mafiosos_active:
            chosen_mafioso = random.choice(mafiosos_active)
            # Exclude all mafiosos from kill candidates (mafiosos cannot kill other mafiosos)
            mafioso_names = [a.name for a in active_players if a.role == "mafioso"]
            kill_candidates = [n for n in candidates if n not in mafioso_names]
            if kill_candidates:
                target = chosen_mafioso.kill(kill_candidates, self.state)
                
                # Kill the target
                victim = self.state.get_agent_by_name(target)
                victim.alive = False
                killed_player = target
                
                print(f"[SPECTATOR] {chosen_mafioso.name} (chosen mafioso) kills {target}")
                
                # All mafiosos learn about the kill (except the killer who already remembers)
                mafiosos_alive = [a for a in self.state.agents if a.role == "mafioso" and a.alive]
                for mafioso in mafiosos_alive:
                    if mafioso != chosen_mafioso:
                        mafioso.remember(f"The Mafia member {chosen_mafioso.name} killed {target}")
        
        return killed_player
    
    def run_day_phase(self):
        """Execute day phase"""
        # Announce day phase start to all players
        for agent in self.state.agents:
            agent.remember(f"Day {self.state.round} begins.")
        
        self.discussion_rounds()
        self.voting_round()
    
    def discussion_rounds(self):
        """Two rounds of strategic communication"""
        print(f"\nDISCUSSION - Day {self.state.round}")
        active_players = self.state.get_active_players()
        active_names = self.state.get_active_names()
        
        for round_num in range(1, self.state.discussion_rounds + 1):
            print(f"\nRound {round_num}:")
            
            # Random order each round
            agents_order = active_players.copy()
            random.shuffle(agents_order)
            
            for agent in agents_order:
                # Agent sends a public message
                all_player_names = [a.name for a in self.state.agents]
                message = agent.message(active_names, round_num, all_player_names, self.state.discussion_rounds, self.state)
                print(f'{agent.name}: {message}')
                
                for a in active_players:
                    if a != agent:
                        a.remember(f'{agent.name}: {message}')
                    else: 
                        agent.remember(f'You: {message}')
    
    def voting_round(self):
        """Everyone votes to arrest someone"""
        print(f"\nVOTING:")
        active_players = self.state.get_active_players()
        votes = {}
        
        for agent in active_players:
            candidates = [a.name for a in active_players if a != agent]
            all_player_names = [a.name for a in self.state.agents]
            vote = agent.vote(candidates, all_player_names, self.state.discussion_rounds, self.state)
            votes[agent.name] = vote
            print(f"{agent.name} votes for {vote}")
        
        # Count votes and arrest
        self.resolve_votes(votes)
    
    def resolve_votes(self, votes):
        """Count votes and arrest the chosen player"""
        if not votes:
            return
            
        vote_counts = Counter(votes.values())
        
        # Handle ties
        max_votes = max(vote_counts.values())
        tied = [name for name, count in vote_counts.items() if count == max_votes]
        
        if len(tied) > 1:
            print(f"\nTie between: {', '.join(tied)}")
            arrested = random.choice(tied)
            print(f"Random tiebreaker: {arrested}")
        else:
            arrested = tied[0]
        
        # Arrest the player
        arrested_agent = self.state.get_agent_by_name(arrested)
        arrested_agent.imprisoned = True
        
        print(f"\n{arrested} has been arrested!")
        
        # Create vote summary for memory (show all votes to everyone)
        votes_announcement = f"Votes: " + ", ".join([f"{voter} voted for {target}" for voter, target in votes.items()])
        arrest_announcement = f"{arrested} was arrested"
        
        # Everyone remembers both the votes and the arrest
        for agent in self.state.get_active_players():
            agent.remember(votes_announcement)
            agent.remember(arrest_announcement)
    
    def check_game_over(self):
        """Check win conditions"""
        active = self.state.get_active_players()
        mafiosos = sum(1 for a in active if a.role == "mafioso")
        good = sum(1 for a in active if a.role not in ["mafioso"])
        
        if mafiosos == 0:
            return "TOWN WINS! All mafiosos eliminated!"
        elif good == 0:
            return "MAFIA WINS! All non-mafiosos eliminated!"

        return None
    
    def show_roles(self):
        print("\nSecret roles:")
        for agent in self.state.agents:
            print(f"  {agent.name}: {agent.role}")
    
    def show_status(self):
        active = [a.name for a in self.state.get_active_players()]
        imprisoned = [a.name for a in self.state.agents if a.imprisoned]
        dead = [a.name for a in self.state.agents if not a.alive]
        
        print(f"\nCurrent Status:")
        print(f"  Active: {', '.join(active) if active else 'None'}")
        print(f"  Imprisoned: {', '.join(imprisoned) if imprisoned else 'None'}")
        print(f"  Dead: {', '.join(dead) if dead else 'None'}")
    
    def show_game_end(self, result):
        print(f"\n{'='*50}")
        print(result)
        print(f"{'='*50}")
        self.show_final_roles()
    
    def show_final_roles(self):
        print(f"\nFINAL ROLES:")
        for agent in self.state.agents:
            if agent.imprisoned:
                status = "[IMPRISONED] "
            elif not agent.alive:
                status = "[DEAD] "
            else:
                status = "[ALIVE] "
            print(f"{status}{agent.name}: {agent.role.upper()}")



def create_game(players, discussion_rounds=2, debug_prompts=False, prompt_config=PromptConfig(version="v1.0")):
    """Create a Mafia game with specified players"""
    # Create agents
    agents = []
    for player in players:
        llm = create_llm(player['llm'])
        agent = MafiaAgent(player['name'], player['role'], llm, debug_prompts, prompt_config)
        agent.remember(f"You're {agent.name}, the {agent.role}.")
        agents.append(agent)
    
    # Create game state
    state = GameState(agents, discussion_rounds)
    
    # Game setup: Mafiosos know each other
    mafiosos = [a for a in agents if a.role == "mafioso"]
    if len(mafiosos) > 1:
        for mafioso in mafiosos:
            other_mafiosos = [a.name for a in mafiosos if a != mafioso]
            mafioso.remember(f"Your fellow mafiosos are: {', '.join(other_mafiosos)}.")
    
    return Game(state)

if __name__ == "__main__":
    print("Use preset_games.py to run predefined games, or use create_game() function directly.")