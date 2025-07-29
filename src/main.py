# src/main.py - Simple Mafia Game System
import sys
import os
import random
sys.path.append('.')

# Simple model cache to avoid conflicts
_model_cache = {}

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
        alive = [a.name for a in state.get_alive_players()]
        imprisoned = [a.name for a in state.agents if a.imprisoned]
        dead = [a.name for a in state.agents if not a.alive]
        
        print(f"\nCurrent Status:")
        print(f"  Alive: {', '.join(alive) if alive else 'None'}")
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
    
    def show_arrest(self, player_name, role):
        print(f"\n{player_name} has been arrested!")
        print(f"They were {role.upper()}!")
    
    def show_no_consensus(self):
        print(f"\nNo consensus reached. No one was arrested.")

def create_llm(llm_config):
    """Simple LLM creator - supports local, openai, anthropic"""
    llm_type = llm_config.get('type', 'local')
    
    if llm_type == 'local':
        # Use shared model to avoid conflicts
        model_path = llm_config.get('model_path', 'models/mistral.gguf')
        if model_path not in _model_cache:
            from src.agents.llm_interface import LlamaCppInterface
            _model_cache[model_path] = LlamaCppInterface(model_path)
        return _model_cache[model_path]
    
    elif llm_type == 'openai':
        try:
            import openai
            client = openai.OpenAI(api_key=llm_config.get('api_key') or os.getenv('OPENAI_API_KEY'))
            
            class SimpleOpenAI:
                def generate(self, prompt, max_tokens=50):
                    response = client.chat.completions.create(
                        model=llm_config.get('model', 'gpt-3.5-turbo'),
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=0.7
                    )
                    return response.choices[0].message.content.strip()
            
            return SimpleOpenAI()
        except ImportError:
            print("ERROR: OpenAI package not installed. Using local model instead.")
            return create_llm({'type': 'local'})
    
    elif llm_type == 'anthropic':
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=llm_config.get('api_key') or os.getenv('ANTHROPIC_API_KEY'))
            
            class SimpleAnthropic:
                def generate(self, prompt, max_tokens=50):
                    response = client.messages.create(
                        model=llm_config.get('model', 'claude-3-haiku-20240307'),
                        max_tokens=max_tokens,
                        temperature=0.7,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.content[0].text.strip()
            
            return SimpleAnthropic()
        except ImportError:
            print("ERROR: Anthropic package not installed. Using local model instead.")
            return create_llm({'type': 'local'})
    
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
            {'name': 'Bob', 'role': 'assassin', 'llm': {'type': 'openai', 'model': 'gpt-3.5-turbo'}},
            {'name': 'Charlie', 'role': 'villager', 'llm': {'type': 'anthropic', 'model': 'claude-3-haiku'}}
        ]
    """
    from src.agents.agent import MafiaAgent
    from src.day_phase import DayPhase
    from src.night_phase import NightPhase
    
    # Create agents
    agents = []
    for player in players:
        llm = create_llm(player['llm'])
        agent = MafiaAgent(player['name'], player['role'], llm, debug_prompts)
        # Add role identity as first memory entry
        agent.remember(f"You're {agent.name}, the {agent.role}.")
        agents.append(agent)
    
    # Game state
    class GameState:
        def __init__(self, agents, discussion_rounds):
            self.agents = agents
            self.round = 0
            self.discussion_rounds = discussion_rounds
            self.message_log = []
            # Use first agent's LLM as fallback for phases that expect state.llm
            self.llm = agents[0].llm if agents else None
        
        def get_alive_players(self):
            return [a for a in self.agents if a.alive and not a.imprisoned]
        
        def get_agent_by_name(self, name):
            return next(a for a in self.agents if a.name == name)
        
        def get_alive_names(self):
            return [a.name for a in self.get_alive_players()]
    
    # Create game components
    state = GameState(agents, discussion_rounds)
    display = Display()
    day_phase = DayPhase(state, display)
    night_phase = NightPhase(state, display)
    
    # Game setup: Assassins know each other
    assassins = [a for a in agents if a.role == "assassin"]
    if len(assassins) > 1:
        for assassin in assassins:
            other_assassins = [a.name for a in assassins if a != assassin]
            assassin.remember(f"Your fellow assassins are: {', '.join(other_assassins)}.")
    
    # Game object
    class Game:
        def __init__(self):
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
            alive = self.state.get_alive_players()
            
            if len(alive) == 0:
                return "Everyone is dead! Nobody wins!"
            
            if len(alive) == 1:
                winner = alive[0]
                return f"{winner.name} survives! But at what cost..."
            
            assassins = sum(1 for a in alive if a.role == "assassin")
            good = sum(1 for a in alive if a.role not in ["assassin"])
            
            if assassins == 0:
                return "GOOD WINS! All assassins arrested!"
            elif good == 0:
                return "EVIL WINS! All good players killed!"
            
            return None
    
    return Game()

if __name__ == "__main__":
    print("Use preset_games.py to run predefined games, or use create_game() function directly.")