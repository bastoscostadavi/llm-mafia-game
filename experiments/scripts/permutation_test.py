# research/run_permutation_experiment.py
import sys
sys.path.insert(0, 'src')

from permutation_runner import PermutationRunner

def main():
    print("FIXED ORDER EXPERIMENT")
    print("ADV game with mutual knowledge")
    print("Testing all 6 permutations of speaking order")
    
    # Get number of games
    num_games = int(input("\nGames per permutation (default 1000): ") or "1000")
    
    # Run experiment
    runner = PermutationRunner("models/mistral.gguf")
    runner.run_all_permutations(num_games)

if __name__ == "__main__":
    main()
