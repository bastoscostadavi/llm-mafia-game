import random
from typing import List
from collections import Counter
from src.agent import MafiaAgent

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
        
        for round_num in [1, 2]:
            print(f"\nRound {round_num}:")
            
            # Random order each round
            agents_order = alive.copy()
            random.shuffle(agents_order)
            
            for agent in agents_order:
                # Skip if ready to vote
                if agent.ready_to_vote and round_num == 2:
                    print(f"{agent.name} is ready to vote.")
                    continue
                
                print(f"\n{agent.name}'s turn:")
                
                # Agent decides action
                action = agent.decide_action(alive_names, round_num)
                
                if action["action"] == "vote":
                    agent.ready_to_vote = True
                    print(f"{agent.name} is ready to vote.")
                    
                elif action["action"] == "message":
                    self._process_message(agent, action, alive)
    
    def _process_message(self, sender, action, alive_players):
        """Process and distribute a message"""
        recipients = action.get("to", "all")
        message = action.get("message", "...")
        
        if recipients == "all":
            recipient_names = [a.name for a in alive_players if a != sender]
            print(f"{sender.name} to All: {message}")
            
            # Add to everyone's memory
            for a in alive_players:
                if a != sender:
                    a.remember(f"{sender.name} to All: {message}")
                    
        else:
            # Private message(s)
            if isinstance(recipients, str):
                recipients = [recipients]
            
            alive_names = [a.name for a in alive_players]
            valid_recipients = [r for r in recipients if r in alive_names and r != sender.name]
            
            if valid_recipients:
                recipient_str = ', '.join(valid_recipients)
                print(f"{sender.name} to {recipient_str}: {message}")
                
                # Add to recipients' memory
                for a in alive_players:
                    if a.name in valid_recipients:
                        a.remember(f"{sender.name} to you: {message}")
        
        # Sender remembers their own message
        sender.remember(f"You said to {recipients}: {message}")
    
    def voting_round(self):
        """Everyone votes to arrest someone"""
        print("\n🗳️  VOTING:")
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
            print(f"\n⚖️  Tie between: {', '.join(tied)}")
            arrested = random.choice(tied)
            print(f"Random tiebreaker: {arrested}")
        else:
            arrested = tied[0]
        
        # Arrest the player
        arrested_agent = self.state.get_agent_by_name(arrested)
        arrested_agent.imprisoned = True
        
        print(f"\n⛓️  {arrested} has been arrested!")
        
        # Reveal role and inform everyone
        if arrested_agent.role == "assassin":
            print(f"🎭 They were an ASSASSIN!")
            announcement = f"Day {self.state.round}: {arrested} was arrested (assassin)"
        elif arrested_agent.role == "psychopath":
            print(f"🔪 They were the PSYCHOPATH!")
            announcement = f"Day {self.state.round}: {arrested} was arrested (psychopath)"
        else:
            print(f"😇 They were innocent!")
            announcement = f"Day {self.state.round}: {arrested} was arrested (innocent)"
        
        # Everyone remembers the arrest
        for agent in self.state.get_alive_players():
            agent.remember(announcement)
