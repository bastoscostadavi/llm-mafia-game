import random
from typing import List
from collections import Counter
from src.agents.agent import MafiaAgent

class DayPhase:
    """Handles day phase: discussion and voting"""
    
    def __init__(self, game_state, display):
        self.state = game_state
        self.display = display
        
    def run(self):
        """Execute the day phase"""
        self.discussion_rounds()
        self.voting_round()
        
    def discussion_rounds(self):
        """Two rounds of strategic communication"""
        self.display.show_discussion_start(self.state.round)
        alive = self.state.get_alive_players()
        alive_names = self.state.get_alive_names()
        
        for round_num in range(1, self.state.discussion_rounds + 1):
            print(f"\nRound {round_num}:")
            
            # Random order each round
            agents_order = alive.copy()
            random.shuffle(agents_order)
            
            for agent in agents_order:
                print(f"\n{agent.name}'s turn:")
                
                # Agent sends a public message
                action = agent.decide_action(alive_names, round_num)
                self._process_message(agent, action, alive)
    
    def _process_message(self, sender, action, alive_players):
        """Process and distribute a message (public only)"""
        message = action.get("message", "")
        
        # Only allow public messages
        if message.strip():
            print(f"{sender.name}: {message}")
            
            # Add to everyone's memory
            for a in alive_players:
                if a != sender:
                    a.remember(f"{sender.name}: {message}")
            
            # Sender remembers their own message
            sender.remember(f"You said: {message}")
        else:
            # If no valid message, agent remains silent
            print(f"{sender.name} remained silent.")
    
    def voting_round(self):
        """Everyone votes to arrest someone"""
        self.display.show_voting_start()
        alive = self.state.get_alive_players()
        votes = {}
        
        # Random order for voting
        voting_order = alive.copy()
        random.shuffle(voting_order)
        
        for agent in voting_order:
            candidates = [a.name for a in alive if a != agent]
            vote = agent.vote(candidates)
            votes[agent.name] = vote
            print(f"{agent.name} votes for {vote}")
        
        # Count votes and arrest
        self._resolve_votes(votes)
    
    def _resolve_votes(self, votes):
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
        
        self.display.show_arrest(arrested, arrested_agent.role)
        
        # Create announcement for memory
        if arrested_agent.role == "assassin":
            announcement = f"Day {self.state.round}: {arrested} was arrested (assassin)"
        elif arrested_agent.role == "psychopath":
            announcement = f"Day {self.state.round}: {arrested} was arrested (psychopath)"
        else:
            announcement = f"Day {self.state.round}: {arrested} was arrested (innocent)"
        
        # Everyone remembers the arrest
        for agent in self.state.get_alive_players():
            agent.remember(announcement)
