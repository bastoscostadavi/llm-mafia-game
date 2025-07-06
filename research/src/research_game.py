import random
from typing import List, Tuple
from src.research_agent import ResearchAgent

class ResearchGame:
    """Single 3-player game instance"""
    
    def __init__(self, llm, discussion_rounds: int = 5):
        self.llm = llm
        self.discussion_rounds = discussion_rounds
        self.context = "4 players: 1 assassin, 1 detective, 2 villagers. Night 1: Detective investigated the assassin. Assassin killed someone. Now 3 players remain."
        
    def setup(self) -> Tuple[List[ResearchAgent], str, str]:
        """Setup game and return (agents, game_type, victim_name)"""
        names = ["Alice", "Bob", "Charlie", "Diana"]
        roles = ["assassin", "detective", "villager", "villager"]
        
        random.shuffle(names)
        random.shuffle(roles)
        
        # Create all agents with simple context
        all_agents = []
        assassin_name = None
        
        for name, role in zip(names, roles):
            agent = ResearchAgent(name, role, self.llm, self.context)
            all_agents.append(agent)
            if role == "assassin":
                assassin_name = name
        
        # Kill someone (not assassin)
        victims = [a for a in all_agents if a.role != "assassin"]
        victim = random.choice(victims)
        victim.alive = False
        
        # Setup survivors
        survivors = [a for a in all_agents if a.alive]
        
        # Give critical memories
        for agent in survivors:
            if agent.role == "detective":
                agent.remember(f"You investigated {assassin_name}: they are the assassin!")
            agent.remember(f"{victim.name} was killed last night")
        
        # Determine state
        game_type = "AVV" if victim.role == "detective" else "ADV"
        
        return survivors, game_type, victim.name
    
    def run_discussion(self, agents: List[ResearchAgent]) -> List[dict]:
        """Run discussion rounds"""
        dialogue = []
        alive_names = [a.name for a in agents]
        
        for round_num in range(1, self.discussion_rounds + 1):
            speakers = agents.copy()
            random.shuffle(speakers)
            
            for agent in speakers:
                response = agent.speak(round_num, self.discussion_rounds, alive_names)
                
                if response:
                    # Update memories for others
                    for other in agents:
                        if other != agent:
                            other.remember(f"{agent.name}: {response}")
                    
                    # IMPORTANT: Agent remembers what they said!
                    agent.remember(f"You said: {response}")
                    
                    dialogue.append({
                        'round': round_num,
                        'speaker': agent.name,
                        'role': agent.role,
                        'message': response
                    })
        
        return dialogue
    
    def run_voting(self, agents: List[ResearchAgent]) -> Tuple[str, int]:
        """Voting phase - returns (accused_name, outcome)"""
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
            
            # Agent remembers their vote
            agent.remember(f"You voted for: {vote}")
        
        # Count votes
        from collections import Counter
        vote_counts = Counter(votes.values())
        accused = vote_counts.most_common(1)[0][0]
        
        # Everyone learns the result
        for agent in agents:
            agent.remember(f"Vote result: {accused} was accused by majority")
        
        # Outcome: 0 = good wins, 1 = evil wins
        accused_agent = next(a for a in agents if a.name == accused)
        outcome = 0 if accused_agent.role == "assassin" else 1
        
        return accused, outcome