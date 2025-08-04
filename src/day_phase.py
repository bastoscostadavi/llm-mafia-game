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
                message = agent.get_discussion_message(active_names, round_num, all_player_names, self.state.discussion_rounds, self.state)
                print(f'{agent.name}: {message}')

                for a in active_players:
                    if a != agent:
                        a.remember(f'{agent.name}: {message}')
                    else: 
                        agent.remember(f'You: {message}')
    
    def voting_round(self):
        """Everyone votes to arrest someone"""
        self.display.show_voting_start()
        active_players = self.state.get_active_players()
        votes = {}
        
        for agent in active_players:
            candidates = [a.name for a in active_players if a != agent]
            all_player_names = [a.name for a in self.state.agents]
            vote = agent.vote(candidates, all_player_names, self.state.discussion_rounds, self.state)
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
        
        print(f"\n{arrested} has been arrested!")
        
        # Create vote summary for memory (show all votes to everyone)
        vote_summary = f"Day {self.state.round} votes: " + ", ".join([f"{voter} voted for {target}" for voter, target in votes.items()])
        
        # Create arrest announcement for memory (without role)
        arrest_announcement = f"Day {self.state.round}: {arrested} was arrested"
        
        # Everyone remembers both the votes and the arrest
        for agent in self.state.get_active_players():
            agent.remember(vote_summary)
            agent.remember(arrest_announcement)