import json
import os
import sys
sys.path.insert(0, 'src')

from llm_interface import LlamaCppInterface
from fixed_order_game import FixedOrderGame
from research_agent import ResearchAgent

class FixedOrderViewer:
    def __init__(self):
        self.base_dir = "data/fixed_order"
        
    def list_games(self):
        """List available games by order"""
        if not os.path.exists(self.base_dir):
            print("No fixed order games found!")
            return
        
        orders = sorted([d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))])
        
        print("\nAvailable games:")
        for order in orders:
            order_dir = os.path.join(self.base_dir, order)
            games = [f for f in os.listdir(order_dir) if f.endswith('.json')]
            print(f"\n{order}: {len(games)} games")
            for game in games[:3]:  # Show first 3
                print(f"  - {game}")
    
    def view_game(self, order: str, game_num: int):
        """View a specific game summary"""
        filename = f"{self.base_dir}/{order}/game_{game_num:03d}.json"
        
        if not os.path.exists(filename):
            print(f"Game not found: {filename}")
            return
        
        with open(filename, 'r') as f:
            game_data = json.load(f)
        
        print("\n" + "="*60)
        print(f"GAME: {order} - Game #{game_num}")
        print("="*60)
        
        print("\nPlayers:")
        for name, role in game_data['agents']:
            print(f"  {name}: {role}")
        
        print(f"\nSpeaking order: {order}")
        
        print("\nDISCUSSION:")
        print("-"*40)
        
        current_round = 0
        for entry in game_data['dialogue']:
            if entry['round'] != current_round:
                current_round = entry['round']
                print(f"\nRound {current_round}:")
            
            role_tag = entry['role'][0].upper()
            print(f"  {entry['speaker']} [{role_tag}]: {entry['message']}")
        
        print(f"\nVOTE RESULT: {game_data['accused']} was accused")
        print(f"OUTCOME: {'Good wins' if game_data['outcome'] == 0 else 'Evil wins'}")
    
    def replay_with_prompts(self, order: str, game_num: int):
        """Replay a game showing all prompts"""
        filename = f"{self.base_dir}/{order}/game_{game_num:03d}.json"
        
        if not os.path.exists(filename):
            print(f"Game not found: {filename}")
            return
        
        with open(filename, 'r') as f:
            game_data = json.load(f)
        
        print("\n" + "="*60)
        print("REPLAYING GAME WITH PROMPTS AND MEMORY UPDATES")
        print("="*60)
        
        # Create mock agents
        agents = []
        role_to_agent = {}
        
        for name, role in game_data['agents']:
            agent = ResearchAgent(name, role, None, "3 players: 1 assassin, 1 detective, 1 villager. Everyone is alive.")
            agent.all_players = [n for n, _ in game_data['agents']]
            agents.append(agent)
            role_to_agent[role[0].upper()] = agent
            
            # Set initial knowledge
            if role == "assassin":
                detective_name = next(n for n, r in game_data['agents'] if r == "detective")
                agent.remember(f"You know {detective_name} is the detective")
            elif role == "detective":
                assassin_name = next(n for n, r in game_data['agents'] if r == "assassin")
                agent.remember(f"You investigated {assassin_name}: they are the assassin")
        
        print("\nINITIAL KNOWLEDGE:")
        for agent in agents:
            print(f"\n{agent.name} ({agent.role}):")
            if agent.memory:
                for mem in agent.memory:
                    print(f"  - {mem}")
            else:
                print("  - None")
        
        # Replay discussion with immediate updates
        print("\n" + "="*60)
        print("DISCUSSION PHASE")
        print("="*60)
        
        for entry in game_data['dialogue']:
            round_num = entry['round']
            speaker_name = entry['speaker']
            message = entry['message']
            
            # Find the speaking agent
            speaker = next(a for a in agents if a.name == speaker_name)
            
            print(f"\n{'='*50}")
            print(f"Round {round_num} - {speaker.name}'s turn ({speaker.role})")
            print(f"{'='*50}")
            
            # Show current memory state
            print("\nCURRENT MEMORY:")
            if speaker.memory:
                for mem in speaker.memory:
                    print(f"  - {mem}")
            else:
                print("  - None")
            
            # Show the prompt
            others = [p for p in speaker.all_players if p != speaker.name]
            alive_names = [a.name for a in agents]
            
            print("\nPROMPT:")
            print(f"[INST] You are {speaker.name} in a Mafia game with {', '.join(others)}.")
            print("\nGame context: 3 players: 1 assassin, 1 detective, 1 villager. Everyone is alive.")
            print(f"\nRole: {self._get_role_desc(speaker.role)}")
            print("\nYour memories:")
            if speaker.memory:
                for mem in speaker.memory:
                    print(f"- {mem}")
            else:
            print("None yet")
            discussion_rounds = game_data.get('discussion_rounds', 3)
            print(f"\nDay 1 Discussion - Turn {round_num} of {discussion_rounds}.")
            print(f"\nDay 1 Discussion - Turn {round_num} of {entry.get('total_rounds',2)}.")
            print(f"Alive players: {', '.join(alive_names)}")
            print("\nWhat do you say? (1-2 sentences): [/INST]")
            
            print(f"\nGENERATED RESPONSE: {message}")
            
            # Speaker remembers what they said
            speaker.remember(f"You said: {message}")
            
            # Others hear it immediately
            for other in agents:
                if other.name != speaker.name:
                    other.remember(f"{speaker.name}: {message}")

        
        # Show final memories before voting
        print("\n" + "="*60)
        print("MEMORIES BEFORE VOTING")
        print("="*60)
        
        for agent in agents:
            print(f"\n{agent.name} ({agent.role}):")
            for mem in agent.memory:
                print(f"  - {mem}")
        
        # Show voting phase
        print("\n" + "="*60)
        print("VOTING PHASE")
        print("="*60)
        
        print("\nEach player now votes based on their accumulated memories.")
        print(f"\nFinal result: {game_data['accused']} was accused")
        print(f"Outcome: {'Good wins' if game_data['outcome'] == 0 else 'Evil wins'}")
    
    def _get_role_desc(self, role):
        descs = {
            "assassin": "You are the assassin trying to avoid detection.",
            "detective": "You are the detective who knows who the assassin is.",
            "villager": "You are a villager trying to find the assassin."
        }
        return descs.get(role, role)

def main():
    viewer = FixedOrderViewer()
    
    while True:
        print("\nFIXED ORDER GAME VIEWER")
        print("1. List available games")
        print("2. View game summary")
        print("3. Replay game with prompts")
        print("4. Exit")
        
        choice = input("\nChoice: ")
        
        if choice == '1':
            viewer.list_games()
        
        elif choice == '2':
            order = input("Order (e.g., ADV): ").upper()
            game_num = int(input("Game number (e.g., 0): "))
            viewer.view_game(order, game_num)
        
        elif choice == '3':
            order = input("Order (e.g., ADV): ").upper()
            game_num = int(input("Game number (e.g., 0): "))
            viewer.replay_with_prompts(order, game_num)
        
        elif choice == '4':
            break

if __name__ == "__main__":
    main()