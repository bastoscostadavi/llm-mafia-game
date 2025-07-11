# research/src/fixed_order_game.py
import random
from typing import List, Tuple
from research_agent import ResearchAgent

class FixedOrderGame:
    """ADV game with fixed speaking order and mutual knowledge"""
    
    def __init__(self, llm, speaking_order: List[str], discussion_rounds: int = 3):
        self.llm = llm
        self.speaking_order = speaking_order  # e.g., ['A', 'D', 'V']
        self.discussion_rounds = discussion_rounds
        
    def setup(self) -> List[ResearchAgent]:
        """Setup ADV game where assassin knows detective"""
        names = ["Alice", "Bob", "Charlie"]
        roles = ["assassin", "detective", "villager"]
        
        random.shuffle(names)
        
        # Create agents
        agents = []
        role_to_name = {}
        
        for name, role in zip(names, roles):
            context = "3 players: 1 assassin, 1 detective, 1 villager. Everyone is alive."
            agent = ResearchAgent(name, role, self.llm, context)
            agents.append(agent)
            role_to_name[role[0].upper()] = name  # A, D, V
        
        # Set all players for each agent
        all_names = [a.name for a in agents]
        for agent in agents:
            agent.all_players = all_names
        
        # Give initial knowledge
        assassin = next(a for a in agents if a.role == "assassin")
        detective = next(a for a in agents if a.role == "detective")
        
        # Assassin knows detective
        assassin.remember(f"You know {detective.name} is the detective")
        
        # Detective knows assassin
        detective.remember(f"You investigated {assassin.name}: they are the assassin")
        
        # Store role mapping for fixed order
        self.role_to_agent = {
            'A': assassin,
            'D': detective, 
            'V': next(a for a in agents if a.role == "villager")
        }
        
        return agents
    
    def run_discussion(self, agents: List[ResearchAgent]) -> List[dict]:
        """Run discussion with FIXED speaking order"""
        dialogue = []
        alive_names = [a.name for a in agents]
        
        for round_num in range(1, self.discussion_rounds + 1):
            # Use fixed order instead of random
            for role_letter in self.speaking_order:
                agent = self.role_to_agent[role_letter]
                
                response = agent.speak(round_num, self.discussion_rounds, alive_names)
                
                if response:
                    # Update memories
                    for other in agents:
                        if other != agent:
                            other.remember(f"{agent.name}: {response}")
                    
                    agent.remember(f"You said: {response}")
                    
                    dialogue.append({
                        'round': round_num,
                        'speaker': agent.name,
                        'role': agent.role,
                        'order': self.speaking_order,
                        'message': response
                    })
        
        return dialogue
    
    def run_voting(self, agents: List[ResearchAgent]) -> Tuple[str, int]:
        """Voting phase"""
        votes = {}
        
        for agent in agents:
            candidates = [a.name for a in agents if a != agent]
            response = agent.vote(candidates)
            
            # Parse vote
            vote = None
            for name in candidates:
                if name.lower() in response.lower():
                    vote = name
                    break
            
            vote = vote or random.choice(candidates)
            votes[agent.name] = vote
            
            agent.remember(f"You voted for: {vote}")
        
        # Count votes
        from collections import Counter
        vote_counts = Counter(votes.values())
        accused = vote_counts.most_common(1)[0][0]
        
        # Everyone learns result
        for agent in agents:
            agent.remember(f"Vote result: {accused} was accused by majority")
        
        # Outcome
        accused_agent = next(a for a in agents if a.name == accused)
        outcome = 0 if accused_agent.role == "assassin" else 1
        
        return accused, outcome
