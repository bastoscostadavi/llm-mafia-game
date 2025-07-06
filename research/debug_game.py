# research/debug_game.py
import sys
sys.path.insert(0, 'src')

from llm_interface import LlamaCppInterface
from research_game import ResearchGame

def run_debug_game():
    """Run a single game with debug output"""
    print("🔍 DEBUG MODE - Single Game")
    print("This will show all prompts sent to players\n")
    
    # Initialize
    llm = LlamaCppInterface("models/mistral.gguf")
    game = ResearchGame(llm, discussion_rounds=3)  # Fewer rounds for clarity
    
    # Setup
    agents, game_state, victim = game.setup()
    
    # Enable debug mode for all agents
    for agent in agents:
        agent.debug = True
    
    print(f"\n{'='*60}")
    print("GAME SETUP")
    print(f"{'='*60}")
    print(f"Game state: {game_state}")
    print(f"Victim: {victim}")
    print("\nPlayers:")
    for agent in agents:
        print(f"  {agent.name}: {agent.role}")
        if agent.memory:
            print(f"    Initial memory: {agent.memory}")
    
    print(f"\n{'='*60}")
    print("DISCUSSION PHASE")
    print(f"{'='*60}")
    
    # Run discussion
    dialogue = game.run_discussion(agents)
    
    print(f"\n{'='*60}")
    print("VOTING PHASE")
    print(f"{'='*60}")
    
    # Run voting
    accused, outcome = game.run_voting(agents)
    
    print(f"\n{'='*60}")
    print("GAME RESULT")
    print(f"{'='*60}")
    print(f"Accused: {accused}")
    print(f"Outcome: {'Good wins' if outcome == 0 else 'Evil wins'}")
    
    # Show final memory state
    print(f"\n{'='*60}")
    print("FINAL MEMORY STATE")
    print(f"{'='*60}")
    for agent in agents:
        print(f"\n{agent.name} ({agent.role}) final memories:")
        for i, mem in enumerate(agent.memory):
            print(f"  {i+1}. {mem}")

if __name__ == "__main__":
    run_debug_game()
