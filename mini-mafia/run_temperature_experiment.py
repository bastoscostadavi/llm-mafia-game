#!/usr/bin/env python3
"""
Temperature Experiment Runner for Mini-Mafia

Runs 100 games for each temperature from 0.0 to 1.0 in 0.1 increments.
Each temperature gets its own folder with batch configuration.
Uses GPT-OSS-20B model for all players.
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from run_mini_mafia_batch import run_batch
from src.prompts import PromptConfig

def create_model_config_with_temperature(temperature):
    """Create model configuration with specified temperature"""
    # Use absolute path to avoid issues when running from different directories
    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / 'models' / 'openai_gpt-oss-20b-Q4_K_M.gguf'
    
    return {
        'detective': {'type': 'local', 'model_path': str(model_path), 'temperature': temperature, 'n_ctx': 8192},
        'mafioso': {'type': 'local', 'model_path': str(model_path), 'temperature': temperature, 'n_ctx': 8192},
        'villager': {'type': 'local', 'model_path': str(model_path), 'temperature': temperature, 'n_ctx': 8192}
    }

def run_temperature_experiment(n_games=100, temperatures=None, debug_prompts=False):
    """Run temperature experiment with different temperature values"""
    
    if temperatures is None:
        # Default: 0.0 to 1.0 in 0.1 increments
        temperatures = [round(i * 0.1, 1) for i in range(11)]
    
    experiment_id = f"temp_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Starting temperature experiment: {experiment_id}")
    print(f"Temperatures: {temperatures}")
    print(f"Games per temperature: {n_games}")
    print(f"Total games: {len(temperatures) * n_games}")
    
    # Create experiment folder
    experiment_dir = Path(__file__).parent / "data" / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment metadata
    experiment_metadata = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "experiment_type": "temperature_variation",
        "model": "openai_gpt-oss-20b-Q4_K_M.gguf",
        "prompt_version": "v2.0",
        "temperatures": temperatures,
        "games_per_temperature": n_games,
        "total_games": len(temperatures) * n_games
    }
    
    with open(experiment_dir / "experiment_metadata.json", 'w') as f:
        json.dump(experiment_metadata, f, indent=2)
    
    results = []
    
    for temp in temperatures:
        print(f"\n{'='*60}")
        print(f"Running temperature {temp} ({n_games} games)")
        print(f"{'='*60}")
        
        # Create model config with this temperature
        model_configs = create_model_config_with_temperature(temp)
        
        # Create prompt config
        prompt_config = PromptConfig(version="v2.0")
        
        # Custom batch ID with temperature
        batch_id = f"temp_{temp:0.1f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_v2.0"
        
        # Create temperature-specific folder
        temp_dir = experiment_dir / f"temp_{temp:0.1f}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Run batch with custom folder
            batch_result = run_batch_with_custom_folder(
                n_games=n_games, 
                debug_prompts=debug_prompts, 
                prompt_config=prompt_config, 
                model_configs=model_configs,
                batch_id=batch_id,
                batch_dir=temp_dir
            )
            
            results.append({
                "temperature": temp,
                "batch_id": batch_result,
                "folder": str(temp_dir)
            })
            
        except Exception as e:
            print(f"Error running temperature {temp}: {e}")
            results.append({
                "temperature": temp,
                "batch_id": None,
                "error": str(e)
            })
    
    # Save experiment results summary
    with open(experiment_dir / "experiment_results.json", 'w') as f:
        json.dump({
            "experiment_metadata": experiment_metadata,
            "results": results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"TEMPERATURE EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Experiment ID: {experiment_id}")
    print(f"Results saved to: {experiment_dir}")
    print(f"Temperatures tested: {temperatures}")
    print(f"Total games run: {len([r for r in results if r.get('batch_id')])}")
    
    return experiment_id

def run_batch_with_custom_folder(n_games, debug_prompts=False, prompt_config=None, model_configs=None, batch_id=None, batch_dir=None):
    """Modified version of run_batch that uses a custom folder"""
    from run_mini_mafia_batch import save_game_data, determine_winner, save_batch_config
    from preset_games import mini_mafia_game
    import io
    
    print(f"Starting batch: {batch_id}")
    print(f"Running {n_games} mini-mafia games with temperature {model_configs['detective']['temperature']}...")
    
    # Save batch configuration
    config_file = save_batch_config(prompt_config, model_configs, batch_dir, batch_id)
    print(f"Batch config saved to: {config_file}")
    
    results = []
    stats = {"good_wins": 0, "evil_wins": 0}
    
    for i in range(n_games):
        if (i + 1) % 20 == 0:  # Less frequent updates for cleaner output
            print(f"Game {i+1}/{n_games}")
        
        # Create and run game
        game = mini_mafia_game(debug_prompts=debug_prompts, model_configs=model_configs, prompt_config=prompt_config)
        
        # Capture stdout to reduce clutter
        if not debug_prompts:
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            try:
                game.play()
            finally:
                sys.stdout = old_stdout
        else:
            game.play()
        
        # Save game data
        game_data = save_game_data(game, i, batch_id, batch_dir, prompt_config, model_configs)
        results.append(game_data)
        
        # Update stats
        winner = determine_winner(game.state.agents)
        if winner == "good":
            stats["good_wins"] += 1
        else:  # winner == "evil"
            stats["evil_wins"] += 1
    
    temperature = model_configs['detective']['temperature']
    print(f"\nBatch complete for temperature {temperature}")
    print(f"Good wins: {stats['good_wins']} ({stats['good_wins']/n_games:.1%})")
    print(f"Evil wins: {stats['evil_wins']} ({stats['evil_wins']/n_games:.1%})")
    
    return batch_id

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run temperature experiment for mini-mafia')
    parser.add_argument('--games', '-g', type=int, default=100, help='Games per temperature (default: 100)')
    parser.add_argument('--debug', action='store_true', help='Show LLM prompts')
    parser.add_argument('--temperatures', '-t', nargs='+', type=float, 
                       help='Specific temperatures to test (default: 0.0 to 1.0 in 0.1 increments)')
    
    args = parser.parse_args()
    
    temperatures = args.temperatures
    if temperatures is None:
        temperatures = [round(i * 0.1, 1) for i in range(11)]
    
    print("Mini-Mafia Temperature Experiment")
    print("=" * 40)
    print(f"Model: GPT-OSS-20B")
    print(f"Prompt version: v2.0")
    print(f"Temperatures: {temperatures}")
    print(f"Games per temperature: {args.games}")
    print(f"Total games: {len(temperatures) * args.games}")
    
    if not args.debug and len(temperatures) * args.games > 10:
        confirm = input("\nProceed with experiment? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Experiment cancelled.")
            return
    
    try:
        experiment_id = run_temperature_experiment(
            n_games=args.games, 
            temperatures=temperatures, 
            debug_prompts=args.debug
        )
        print(f"\nExperiment complete: {experiment_id}")
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()