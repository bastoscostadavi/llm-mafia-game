# src/main.py - Simple Mafia Game System
import sys
import os
import random
from collections import Counter
from src.agents.agent import MafiaAgent
from src.day_phase import DayPhase
from src.night_phase import NightPhase
from src.agents.llm_interface import LlamaCppInterface

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from dotenv import load_dotenv
    # Load .env file if it exists
    load_dotenv()
except ImportError:
    # dotenv not installed, skip loading
    pass

sys.path.append('.')

# Simple model cache to avoid conflicts
_model_cache = {}

class SimpleOpenAI:
    """OpenAI API wrapper"""
    def __init__(self, client, model):
        self.client = client
        self.model = model
    
    def generate(self, prompt, max_tokens=50):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

class SimpleAnthropic:
    """Anthropic API wrapper"""
    def __init__(self, client, model):
        self.client = client
        self.model = model
    
    def generate(self, prompt, max_tokens=50):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

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

class Game:
    """Main game controller that manages the game flow"""
    
    def __init__(self, state, display, day_phase, night_phase):
        self.state = state
        self.display = display
        self.day_phase = day_phase
        self.night_phase = night_phase
    
    def play(self):
        self.display.show_game_start()
        self.display.show_roles(self.state.agents)
        
        while self.state.round < 10:  # Max rounds
            self.state.round += 1
            
            # Night Phase
            self.display.show_night_start(self.state.round)
            self.night_phase.run()
            
            # Check win conditions
            result = self.check_game_over()
            if result:
                self.display.show_game_end(result)
                self.display.show_final_roles(self.state.agents)
                return
            
            # Day Phase
            self.display.show_day_start(self.state.round)
            self.display.show_status(self.state)
            self.day_phase.run()
            
            # Check win conditions
            result = self.check_game_over()
            if result:
                self.display.show_game_end(result)
                self.display.show_final_roles(self.state.agents)
                return
        
        self.display.show_timeout()
        self.display.show_final_roles(self.state.agents)
    
    def check_game_over(self):
        """Check win conditions"""
        active = self.state.get_active_players()
        mafiosos = sum(1 for a in active if a.role == "mafioso")
        good = sum(1 for a in active if a.role not in ["mafioso"])
        
        if mafiosos == 0:
            return "GOOD WINS! All mafiosos arrested!"
        elif good == 0:
            return "EVIL WINS! All good players killed!"
        
        return None

class Display:
    """Simple display class for game output"""
    
    def show_game_start(self):
        print("Initializing Mafia Game...")
    
    def show_roles(self, agents):
        print("\nSecret roles:")
        for agent in agents:
            print(f"  {agent.name}: {agent.role}")
    
    def show_night_start(self, round_num):
        print(f"\nNIGHT {round_num}")
    
    def show_day_start(self, round_num):
        print(f"\n==================================================")
        print(f"DAY {round_num}")
        print(f"==================================================")
    
    def show_status(self, state):
        active = [a.name for a in state.get_active_players()]
        imprisoned = [a.name for a in state.agents if a.imprisoned]
        dead = [a.name for a in state.agents if not a.alive]
        
        print(f"\nCurrent Status:")
        print(f"  Active: {', '.join(active) if active else 'None'}")
        print(f"  Imprisoned: {', '.join(imprisoned) if imprisoned else 'None'}")
        print(f"  Dead: {', '.join(dead) if dead else 'None'}")
    
    def show_game_end(self, result):
        print(f"\n==================================================")
        print(result)
        print(f"==================================================")
    
    def show_final_roles(self, agents):
        print(f"\nFINAL ROLES:")
        for agent in agents:
            if agent.imprisoned:
                status = "[IMPRISONED] "
            elif not agent.alive:
                status = "[DEAD] "
            else:
                status = "[ALIVE] "
            print(f"{status}{agent.name}: {agent.role.upper()}")
    
    def show_timeout(self):
        print(f"\nGame ended due to timeout!")
    
    def show_discussion_start(self, round_num):
        print(f"\nDISCUSSION - Day {round_num}")
    
    def show_voting_start(self):
        print(f"\nVOTING:")
    
    def show_arrest(self, player_name):
        print(f"\n{player_name} has been arrested!")
    
    def show_no_consensus(self):
        print(f"\nNo consensus reached. No one was arrested.")

def create_llm(llm_config):
    """Simple LLM creator - supports local, openai, anthropic"""
    llm_type = llm_config.get('type', 'local')
    
    if llm_type == 'local':
        # Use shared model to avoid conflicts
        model_path = llm_config.get('model_path', 'models/mistral.gguf')
        if model_path not in _model_cache:
            _model_cache[model_path] = LlamaCppInterface(model_path)
        return _model_cache[model_path]
    
    elif llm_type == 'openai':
        if openai is None:
            print("ERROR: OpenAI package not installed. Using local model instead.")
            return create_llm({'type': 'local'})
        
        client = openai.OpenAI(api_key=llm_config.get('api_key') or os.getenv('OPENAI_API_KEY'))
        model = llm_config.get('model', 'gpt-3.5-turbo')
        return SimpleOpenAI(client, model)
    
    elif llm_type == 'anthropic':
        if anthropic is None:
            print("ERROR: Anthropic package not installed. Using local model instead.")
            return create_llm({'type': 'local'})
        
        client = anthropic.Anthropic(api_key=llm_config.get('api_key') or os.getenv('ANTHROPIC_API_KEY'))
        model = llm_config.get('model', 'claude-3-haiku-20240307')
        return SimpleAnthropic(client, model)
    
    else:
        # Default to local
        return create_llm({'type': 'local'})

def create_game(players, discussion_rounds=2, debug_prompts=False):
    """
    Create a Mafia game with specified players
    
    Args:
        players: List of player dicts with 'name', 'role', and 'llm' keys
        discussion_rounds: Number of discussion rounds per day (default: 2)
        debug_prompts: Whether to print prompts sent to LLMs (default: False)
        
    Example:
        players = [
            {'name': 'Alice', 'role': 'detective', 'llm': {'type': 'local', 'model_path': 'models/mistral.gguf'}},
            {'name': 'Bob', 'role': 'mafioso', 'llm': {'type': 'openai', 'model': 'gpt-3.5-turbo'}},
            {'name': 'Charlie', 'role': 'villager', 'llm': {'type': 'anthropic', 'model': 'claude-3-haiku'}}
        ]
    """
    # Create agents
    agents = []
    for player in players:
        llm = create_llm(player['llm'])
        agent = MafiaAgent(player['name'], player['role'], llm, debug_prompts)
        # Add role identity as first memory entry
        agent.remember(f"You're {agent.name}, the {agent.role}.")
        agents.append(agent)
    
    # Create game components
    state = GameState(agents, discussion_rounds)
    display = Display()
    day_phase = DayPhase(state, display)
    night_phase = NightPhase(state, display)
    
    # Game setup: Mafiosos know each other
    mafiosos = [a for a in agents if a.role == "mafioso"]
    if len(mafiosos) > 1:
        for mafioso in mafiosos:
            other_mafiosos = [a.name for a in mafiosos if a != mafioso]
            mafioso.remember(f"Your fellow mafiosos are: {', '.join(other_mafiosos)}.")
    
    return Game(state, display, day_phase, night_phase)

if __name__ == "__main__":
    print("Use preset_games.py to run predefined games, or use create_game() function directly.")