# research/src/permutation_runner.py
import itertools
import json
import os
from datetime import datetime
from typing import Dict, List
import sys
sys.path.insert(0, os.path.abspath('../..'))

from src.llm_interface import LlamaCppInterface
from fixed_order_game import FixedOrderGame

class PermutationRunner:
    """Run games with all permutations of speaking order"""
    
    def __init__(self, model_path: str):
        self.llm = LlamaCppInterface(model_path)
        self.permutations = list(itertools.permutations(['A', 'D', 'V']))
        
    def run_single_permutation(self, order: List[str], num_games: int) -> Dict:
        """Run games with a specific speaking order"""
        results = {"good": 0, "evil": 0}
        order_str = ''.join(order)
        
        print(f"\nTesting order: {order_str}")
        
        for i in range(num_games):
            if i % 100 == 0 and i > 0:
                print(f"  Progress: {i}/{num_games}")
            
            game = FixedOrderGame(self.llm, order)
            agents = game.setup()
            dialogue = game.run_discussion(agents)
            accused, outcome = game.run_voting(agents)
            
            if outcome == 0:
                results["good"] += 1
            else:
                results["evil"] += 1
            
            # Save game if needed
            self._save_game(order_str, i, agents, dialogue, accused, outcome)
        
        return results
    
    def run_all_permutations(self, num_games_per_order: int):
        """Run all 6 permutations"""
        all_results = {}
        
        print(f"Running {num_games_per_order} games for each of 6 speaking orders")
        print("Game: ADV with mutual knowledge")
        
        for order in self.permutations:
            order_str = ''.join(order)
            results = self.run_single_permutation(order, num_games_per_order)
            all_results[order_str] = results
            
            # Print results for this order
            total = results["good"] + results["evil"]
            good_pct = results["good"] / total * 100
            print(f"  {order_str}: {results['good']}/{total} good wins ({good_pct:.1f}%)")
        
        # Save and display summary
        self._save_summary(all_results, num_games_per_order)
        self._display_summary(all_results, num_games_per_order)
        
    def _save_game(self, order_str: str, game_num: int, agents, dialogue, accused, outcome):
        """Save individual game"""
        game_data = {
            'order': order_str,
            'game_num': game_num,
            'agents': [(a.name, a.role) for a in agents],
            'dialogue': dialogue,
            'accused': accused,
            'outcome': outcome,
            'discussion_rounds': len(set(entry['round'] for entry in dialogue))  # Don't forget comma if more items follow
        }
        
        os.makedirs(f"data/fixed_order/{order_str}", exist_ok=True)
        filename = f"data/fixed_order/{order_str}/game_{game_num:03d}.json"
        
        with open(filename, 'w') as f:
            json.dump(game_data, f, indent=2)
    
    def _save_summary(self, results: Dict, num_games: int):
        """Save summary of all permutations"""
        summary = {
            'experiment': 'fixed_order_ADV_mutual_knowledge',
            'num_games_per_order': num_games,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs("data/results", exist_ok=True)
        filename = f"data/results/fixed_order_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary saved to: {filename}")
    
    def _display_summary(self, results: Dict, num_games: int):
        """Display analysis of results"""
        print("\n" + "="*50)
        print("SUMMARY: Effect of Speaking Order")
        print("="*50)
        
        # Sort by good win rate
        sorted_orders = sorted(results.items(), 
                             key=lambda x: x[1]["good"] / (x[1]["good"] + x[1]["evil"]), 
                             reverse=True)
        
        print("\nRanked by good win rate:")
        for order, res in sorted_orders:
            total = res["good"] + res["evil"]
            good_pct = res["good"] / total * 100
            print(f"  {order}: {good_pct:.1f}% good wins")
        
        # Analyze patterns
        print("\nAnalysis:")
        
        # Who speaks first matters?
        first_speaker_stats = {'A': [], 'D': [], 'V': []}
        for order, res in results.items():
            first = order[0]
            total = res["good"] + res["evil"]
            good_rate = res["good"] / total
            first_speaker_stats[first].append(good_rate)
        
        print("\nAverage good win rate by first speaker:")
        for role, rates in first_speaker_stats.items():
            avg = sum(rates) / len(rates) * 100
            print(f"  {role} speaks first: {avg:.1f}%")
        
        # Who speaks last matters?
        last_speaker_stats = {'A': [], 'D': [], 'V': []}
        for order, res in results.items():
            last = order[2]
            total = res["good"] + res["evil"]
            good_rate = res["good"] / total
            last_speaker_stats[last].append(good_rate)
        
        print("\nAverage good win rate by last speaker:")
        for role, rates in last_speaker_stats.items():
            avg = sum(rates) / len(rates) * 100
            print(f"  {role} speaks last: {avg:.1f}%")
