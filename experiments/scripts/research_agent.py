class ResearchAgent:
    """Minimal agent for research - just what we need"""
    
    def __init__(self, name: str, role: str, llm, context: str):
        self.name = name
        self.role = role
        self.llm = llm
        self.alive = True
        self.memory = []
        self.context = context  # Simple game context
        self.debug = False  # Set to True to see prompts
        
    def remember(self, event: str):
        """Add to memory"""
        self.memory.append(event)
        
    def get_prompt_base(self) -> str:
        """Get base prompt with role and context"""
        role_desc = {
            "assassin": "You are the assassin trying to avoid detection. You win if you're not voted for by the other two agents.",
            "detective": "You are the detective who knows who the assassin is. You win if the assassin is voted for by you and the villager.",
            "villager": "You are a villager trying to find the assassin. You win if the assassin is voted for by you and the other agent."
        }
        
        return f"""Game context: {self.context}

    Role: {role_desc.get(self.role, self.role)}

    Your memories:
    {self._format_memory()}"""
    
    def _format_memory(self) -> str:
        """Format ALL memory for prompt"""
        if not self.memory:
            return "None yet"
        # Return ALL memories, not just last 10
        return "\n".join(f"- {m}" for m in self.memory)
    
    def speak(self, round_num: int, total_rounds: int, players: list) -> str:
        """Generate speech for discussion"""
        # Get other players for the intro
        others = [p for p in players if p != self.name]
        
        prompt = f"""You are {self.name} in a Mafia game with {' and '.join(others)}.

    {self.get_prompt_base()}

    Day Discussion - Turn {round_num} of {total_rounds}.
    Alive players: {', '.join(players)}

    What do you say? (1-2 sentences):"""
        
        full_prompt = f"[INST] {prompt} [/INST]"
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"PROMPT FOR {self.name} (Turn {round_num}):")
            print(f"{'='*60}")
            print(full_prompt)
            print(f"{'='*60}\n")
        
        return self.llm.generate(full_prompt, max_tokens=50).strip()
    
    def vote(self, candidates: list) -> str:
        """Vote for who to eliminate"""
        # Get other players for the intro
        others = [p for p in candidates]  # candidates already excludes self
        
        prompt = f"""You are {self.name} in a Mafia game with {', '.join(others)}.

    {self.get_prompt_base()}

    Day 1 - Final Vote
    Time to vote. Who is the assassin?
    Candidates: {', '.join(candidates)}

    Reply with just a name:"""
        
        full_prompt = f"[INST] {prompt} [/INST]"
        
        if self.debug:
            print(f"\n{'='*60}")
            print(f"VOTING PROMPT FOR {self.name}:")
            print(f"{'='*60}")
            print(full_prompt)
            print(f"{'='*60}\n")
        
        return self.llm.generate(full_prompt, max_tokens=15).strip()