import random
from collections import Counter
import json
import os
from datetime import datetime

class StrategicNoExchangeGame:
    def run(self):
        # Setup 4 players
        names = ["Alice", "Bob", "Charlie", "Diana"]
        roles = ["assassin", "detective", "villager", "villager"]
        
        random.shuffle(names)
        random.shuffle(roles)
        
        players = {name: role for name, role in zip(names, roles)}
        
        # Find key players
        assassin = next(name for name, role in players.items() if role == "assassin")
        detective = next(name for name, role in players.items() if role == "detective")
        
        # Night 1: Kill someone (not assassin)
        victims = [name for name, role in players.items() if role != "assassin"]
        victim = random.choice(victims)
        
        # Game type
        game_type = "AVV" if victim == detective else "ADV"
        
        # Day 1: Strategic voting
        survivors = [name for name in players if name != victim]
        votes = {}
        
        for voter in survivors:
            voter_role = players[voter]
            candidates = [s for s in survivors if s != voter]
            
            if voter_role == "detective":
                # Detective always votes for assassin
                votes[voter] = assassin
            elif voter_role == "assassin":
                # Assassin votes randomly (doesn't know who's detective)
                votes[voter] = random.choice(candidates)
            else:  # villager
                # Villagers vote randomly
                votes[voter] = random.choice(candidates)
        
        # Count votes
        vote_counts = Counter(votes.values())
        max_votes = max(vote_counts.values())
        tied = [p for p, v in vote_counts.items() if v == max_votes]
        accused = random.choice(tied)
        
        # Outcome
        outcome = 0 if accused == assassin else 1
        
        return game_type, outcome

def run_strategic_baseline(num_games=10000):
    results = {"ADV": {"good": 0, "evil": 0},
               "AVV": {"good": 0, "evil": 0}}
    
    print(f"Running {num_games} games with strategic voting (no communication)")
    print("Rules: Detective always votes for assassin if alive")
    
    for i in range(num_games):
        if i % 1000 == 0 and i > 0:
            print(f"Progress: {i}/{num_games}")
        
        game = StrategicNoExchangeGame()
        game_type, outcome = game.run()
        
        if outcome == 0:
            results[game_type]["good"] += 1
        else:
            results[game_type]["evil"] += 1
    
    # Results
    print(f"\nResults for {num_games} strategic games (no exchange):")
    
    for game_type in ["ADV", "AVV"]:
        total = results[game_type]["good"] + results[game_type]["evil"]
        if total > 0:
            good_pct = results[game_type]["good"] / total * 100
            print(f"\n{game_type} games ({total} total):")
            print(f"  Good wins: {results[game_type]['good']} ({good_pct:.1f}%)")
            print(f"  Evil wins: {results[game_type]['evil']} ({100-good_pct:.1f}%)")
    
    # Overall
    total_good = results["ADV"]["good"] + results["AVV"]["good"]
    total_evil = results["ADV"]["evil"] + results["AVV"]["evil"]
    
    print(f"\nOverall:")
    print(f"  Good wins: {total_good} ({total_good/num_games*100:.1f}%)")
    print(f"  Evil wins: {total_evil} ({total_evil/num_games*100:.1f}%)")
    
    print("\nExpected differences:")
    print("- ADV should have higher good win rate (detective guarantees 1 vote)")
    print("- AVV should be close to random (33.3% good wins)")
    
    # Save
    os.makedirs("data/results", exist_ok=True)
    summary = {
        'type': 'strategic_no_exchange',
        'num_games': num_games,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"data/results/strategic_no_exchange_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    run_strategic_baseline(10000)