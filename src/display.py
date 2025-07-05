# src/display.py
from typing import List

class Display:
    """Handles all game output/UI"""
    
    def show_game_start(self):
        print("🎮 Initializing Mafia Game...")
    
    def show_roles(self, agents):
        print("\n🎭 Secret roles:")
        for agent in agents:
            print(f"  {agent.name}: {agent.role}")
    
    def show_day_start(self, round_num: int):
        print(f"\n{'='*50}")
        print(f"DAY {round_num}")
        print(f"{'='*50}")
    
    def show_night_start(self, round_num: int):
        print(f"\n🌙 NIGHT {round_num}")
    
    def show_discussion_start(self, round_num: int):
        print(f"\n💬 DISCUSSION - Day {round_num}")
    
    def show_status(self, game_state):
        """Show current game status"""
        print("\n📊 Current Status:")
        alive = [a.name for a in game_state.agents if a.alive and not a.imprisoned]
        imprisoned = [a.name for a in game_state.agents if a.imprisoned]
        dead = [a.name for a in game_state.agents if not a.alive]
        
        print(f"  Alive: {', '.join(alive) if alive else 'None'}")
        print(f"  Imprisoned: {', '.join(imprisoned) if imprisoned else 'None'}")
        print(f"  Dead: {', '.join(dead) if dead else 'None'}")
    
    def show_game_end(self, result: str):
        print(f"\n{'='*50}")
        print(f"{result}")
        print(f"{'='*50}")
    
    def show_timeout(self):
        print("\n⏰ Game ended - round limit reached!")
    
    def show_final_roles(self, agents):
        """Reveal all roles at game end"""
        print("\n📋 FINAL ROLES:")
        for agent in agents:
            status = "☠️" if not agent.alive else "⛓️" if agent.imprisoned else "✅"
            print(f"{status} {agent.name}: {agent.role.upper()}")
