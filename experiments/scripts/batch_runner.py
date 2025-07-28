import json
import os
from datetime import datetime
from typing import Dict

# Now these are local imports since everything is in research folder
from llm_interface import LlamaCppInterface
from research_game import ResearchGame

class BatchRunner:
    """Run and analyze multiple games"""
    
    def __init__(self, model_path: str):
        self.llm = LlamaCppInterface(model_path)
        self.results = {"ADV": {"good": 0, "evil": 0},
                       "AVV": {"good": 0, "evil": 0}}
        
    def run_batch(self, num_games: int, save_games: bool = True):
        """Run batch of games"""
        print(f"🔬 Running {num_games} games...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for i in range(num_games):
            if i % 50 == 0 and i > 0:
                print(f"Progress: {i}/{num_games}")
                self._print_current_stats()
            
            # Run game
            game = ResearchGame(self.llm)
            agents, game_type, victim = game.setup()
            dialogue = game.run_discussion(agents)
            accused, outcome = game.run_voting(agents)
            
            # Track results
            if outcome == 0:
                self.results[game_type]["good"] += 1
            else:
                self.results[game_type]["evil"] += 1
            
            # Save individual game
            if save_games:
                self._save_game(i, timestamp, agents, game_type, 
                               victim, dialogue, accused, outcome)
        
        # Final results
        self._save_summary(timestamp, num_games)
        self._print_final_stats(num_games)
    
    def _save_game(self, game_num, batch_id, agents, game_type, 
                   victim, dialogue, accused, outcome):
        """Save individual game data"""
        game_data = {
            'game_num': game_num,
            'batch_id': batch_id,
            'game_type': game_type,
            'victim': victim,
            'outcome': outcome,
            'agents': [(a.name, a.role) for a in agents],
            'accused': accused,
            'dialogue': dialogue
        }
        
        os.makedirs("data/games", exist_ok=True)
        filename = f"data/games/batch_{batch_id}_game_{game_num:04d}.json"
        
        with open(filename, 'w') as f:
            json.dump(game_data, f)
    
    def _save_summary(self, batch_id, num_games):
        """Save batch summary"""
        summary = {
            'batch_id': batch_id,
            'num_games': num_games,
            'results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs("data/results", exist_ok=True)
        filename = f"data/results/summary_{batch_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _print_current_stats(self):
        """Print current statistics"""
        total = sum(self.results[s]["good"] + self.results[s]["evil"] 
                   for s in ["ADV", "AVV"])
        
        if total > 0:
            good_total = self.results["ADV"]["good"] + self.results["AVV"]["good"]
            print(f"  Current: {good_total}/{total} good wins ({good_total/total*100:.1f}%)")
    
    def _print_final_stats(self, num_games):
        """Print final statistics"""
        print("\n" + "="*50)
        print("📊 FINAL RESULTS")
        print("="*50)
        
        # By game type
        for state in ["ADV", "AVV"]:
            total = self.results[state]["good"] + self.results[state]["evil"]
            if total > 0:
                good_pct = self.results[state]["good"] / total * 100
                print(f"\n{state} games ({total} total):")
                print(f"  Good wins: {self.results[state]['good']} ({good_pct:.1f}%)")
                print(f"  Evil wins: {self.results[state]['evil']} ({100-good_pct:.1f}%)")
        
        # Overall
        total_good = self.results["ADV"]["good"] + self.results["AVV"]["good"]
        total_evil = self.results["ADV"]["evil"] + self.results["AVV"]["evil"]
        
        print(f"\nOVERALL ({num_games} games):")
        print(f"  Good wins: {total_good} ({total_good/num_games*100:.1f}%)")
        print(f"  Evil wins: {total_evil} ({total_evil/num_games*100:.1f}%)")


if __name__ == "__main__":
    runner = BatchRunner("../../models/mistral.gguf")
    runner.run_batch(100, save_games=True)
