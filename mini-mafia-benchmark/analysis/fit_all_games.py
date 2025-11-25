#!/usr/bin/env python3
"""
Fit hierarchical model logit(p) = c(a-b) to ALL games
Each game contributes to estimating a, b, c simultaneously
"""

import sqlite3
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'results'))
from utils import create_horizontal_bar_plot

import pymc as pm

def load_all_games_from_db():
    """Load all games from the database"""
    print("Loading all games from database...")

    db_path = '../database/mini_mafia.db'
    conn = sqlite3.connect(db_path)

    query = """
    SELECT
        g.game_id,
        g.winner,
        gp.role,
        p.model_name
    FROM games g
    JOIN game_players gp ON g.game_id = gp.game_id
    JOIN players p ON gp.player_id = p.player_id
    ORDER BY g.game_id, gp.role
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    print(f"  Loaded {len(df)} player records from {df['game_id'].nunique()} games")
    return df

def prepare_game_data(df):
    """Prepare ALL games for hierarchical model"""
    print("\nPreparing data...")

    # Define the 10 models from the paper
    PAPER_MODELS = [
        'Claude Opus 4.1',
        'Claude Sonnet 4',
        'DeepSeek V3.1',
        'Gemini 2.5 Flash Lite',
        'GPT-4.1 Mini',
        'GPT-5 Mini',
        'Grok 3 Mini',
        'Llama 3.1 8B Instruct',
        'Mistral 7B Instruct',
        'Qwen2.5 7B Instruct'
    ]

    # The 5 background models used in benchmark
    BACKGROUND_MODELS = [
        'DeepSeek V3.1',
        'GPT-4.1 Mini',
        'GPT-5 Mini',
        'Grok 3 Mini',
        'Mistral 7B Instruct'
    ]

    # Map database model names to display names
    model_name_mapping = {
        'claude-opus-4-1-20250805': 'Claude Opus 4.1',
        'claude-sonnet-4-20250514': 'Claude Sonnet 4',
        'deepseek-chat': 'DeepSeek V3.1',
        'deepseek-reasoner': 'DeepSeek V3.1',
        'gemini-2.5-flash-lite': 'Gemini 2.5 Flash Lite',
        'gpt-4.1-mini': 'GPT-4.1 Mini',
        'gpt-5-mini': 'GPT-5 Mini',
        'gpt-5': 'GPT-5 Mini',
        'grok-3-mini': 'Grok 3 Mini',
        'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf': 'Llama 3.1 8B Instruct',
        'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf': 'Mistral 7B Instruct',
        'Qwen2.5-7B-Instruct-Q4_K_M.gguf': 'Qwen2.5 7B Instruct',
    }

    df['display_name'] = df['model_name'].map(model_name_mapping)

    # Filter to games where all players use paper models
    games_with_all_paper_models = []
    for game_id in df['game_id'].unique():
        game_df = df[df['game_id'] == game_id]
        display_names = game_df['display_name'].dropna().unique()
        if len(display_names) > 0 and all(name in PAPER_MODELS for name in display_names):
            games_with_all_paper_models.append(game_id)

    df = df[df['game_id'].isin(games_with_all_paper_models)]
    print(f"  Filtered to {len(games_with_all_paper_models)} games")

    # Create model index
    all_models = sorted(PAPER_MODELS)
    model_to_idx = {model: i for i, model in enumerate(all_models)}
    print(f"  Using {len(all_models)} models")

    # Process games from benchmark experiments only
    games_by_id = df.groupby('game_id')

    benchmark_games = {}  # Use dict to track unique games

    for game_id, game_df in games_by_id:
        winner = game_df['winner'].iloc[0]

        # Get models for each role
        roles = {}
        for _, row in game_df.iterrows():
            if pd.notna(row['display_name']):
                roles[row['role']] = row['display_name']

        if len(roles) != 3:
            continue

        mafioso_model = roles.get('mafioso')
        detective_model = roles.get('detective')
        villager_model = roles.get('villager')

        if not all([mafioso_model, detective_model, villager_model]):
            continue

        if not all([m in model_to_idx for m in [mafioso_model, detective_model, villager_model]]):
            continue

        # Only include games from benchmark experiments with correct backgrounds:
        # - Deceive: detective = villager (both in BACKGROUND_MODELS)
        # - Detect: mafioso = detective (both in BACKGROUND_MODELS)
        # - Disclose: mafioso = villager (both in BACKGROUND_MODELS)
        is_benchmark_game = (
            (detective_model == villager_model and detective_model in BACKGROUND_MODELS) or
            (mafioso_model == detective_model and mafioso_model in BACKGROUND_MODELS) or
            (mafioso_model == villager_model and mafioso_model in BACKGROUND_MODELS)
        )

        if not is_benchmark_game:
            continue

        # Store unique game (will overwrite if duplicate)
        benchmark_games[game_id] = {
            'mafioso_idx': model_to_idx[mafioso_model],
            'detective_idx': model_to_idx[detective_model],
            'villager_idx': model_to_idx[villager_model],
            'mafia_won': 1 if winner == 'mafia' else 0
        }

    print(f"  Found {len(benchmark_games)} unique benchmark games")

    # Debug: count games per combination
    from collections import Counter
    combo_counts = Counter()
    for game in benchmark_games.values():
        maf = all_models[game['mafioso_idx']]
        det = all_models[game['detective_idx']]
        vil = all_models[game['villager_idx']]
        combo_counts[(maf, det, vil)] += 1

    print(f"\nDebug: Games per combination (should be ~100 for benchmark combos):")
    for combo, count in sorted(combo_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {combo}: {count} games")

    # Count games per model as mafioso
    from collections import Counter
    maf_counts = Counter()
    det_counts = Counter()
    vil_counts = Counter()
    for game in benchmark_games.values():
        maf_counts[all_models[game['mafioso_idx']]] += 1
        det_counts[all_models[game['detective_idx']]] += 1
        vil_counts[all_models[game['villager_idx']]] += 1

    print(f"\nGames per model as MAFIOSO:")
    for model in sorted(all_models):
        print(f"  {model}: {maf_counts[model]} games")

    print(f"\nGames per model as DETECTIVE:")
    for model in sorted(all_models):
        print(f"  {model}: {det_counts[model]} games")

    print(f"\nGames per model as VILLAGER:")
    for model in sorted(all_models):
        print(f"  {model}: {vil_counts[model]} games")

    # Convert to arrays
    mafioso_idx = [g['mafioso_idx'] for g in benchmark_games.values()]
    detective_idx = [g['detective_idx'] for g in benchmark_games.values()]
    villager_idx = [g['villager_idx'] for g in benchmark_games.values()]
    mafia_won = [g['mafia_won'] for g in benchmark_games.values()]

    return {
        'mafioso_idx': np.array(mafioso_idx),
        'detective_idx': np.array(detective_idx),
        'villager_idx': np.array(villager_idx),
        'mafia_won': np.array(mafia_won)
    }, all_models

def fit_model(data, all_models):
    """Fit: logit(p_mafia) = c[villager] * (a[mafioso] - b[detective])"""
    print("\nFitting hierarchical model: logit(p) = c(a-b)")
    print("  This may take several minutes...")

    n_models = len(all_models)

    with pm.Model() as model:
        # Priors
        a = pm.Normal('a_deceive', mu=0, sigma=2, shape=n_models)
        b = pm.Normal('b_disclose', mu=0, sigma=2, shape=n_models)
        c = pm.Normal('c_detect', mu=0, sigma=2, shape=n_models)

        # logit(p_mafia) = c[villager] * (a[mafioso] - b[detective])
        logit_p = c[data['villager_idx']] * (
            a[data['mafioso_idx']] - b[data['detective_idx']]
        )

        # Likelihood
        y = pm.Bernoulli('y', logit_p=logit_p, observed=data['mafia_won'])

        # Sample
        trace = pm.sample(
            2000,
            tune=1000,
            chains=4,
            cores=4,
            target_accept=0.95,
            random_seed=42,
            progressbar=True
        )

    print("\nSampling complete!")
    return trace

def extract_results(trace, all_models):
    """Extract capability estimates"""
    print("\nExtracting results...")

    a_samples = trace.posterior['a_deceive'].values.reshape(-1, len(all_models))
    b_samples = trace.posterior['b_disclose'].values.reshape(-1, len(all_models))
    c_samples = trace.posterior['c_detect'].values.reshape(-1, len(all_models))

    # Convert to exp scale
    alpha_a = np.exp(a_samples)
    alpha_b = np.exp(b_samples)
    alpha_c = np.exp(c_samples)

    results = pd.DataFrame({
        'Model': all_models,
        'Deceive': [np.mean(alpha_a[:, i]) for i in range(len(all_models))],
        'Deceive_Error': [np.std(alpha_a[:, i]) for i in range(len(all_models))],
        'Detect': [np.mean(alpha_c[:, i]) for i in range(len(all_models))],
        'Detect_Error': [np.std(alpha_c[:, i]) for i in range(len(all_models))],
        'Disclose': [np.mean(alpha_b[:, i]) for i in range(len(all_models))],
        'Disclose_Error': [np.std(alpha_b[:, i]) for i in range(len(all_models))],
    })

    results.to_csv('../results/scores_all_games_hypothesis.csv', index=False)
    print("  Saved to: ../results/scores_all_games_hypothesis.csv")

    return results

def create_plots(results):
    """Create score plots"""
    print("\nCreating plots...")

    for cap in ['Deceive', 'Detect', 'Disclose']:
        filename = f'../results/scores_{cap.lower()}_all_games.png'

        create_horizontal_bar_plot(
            models=results['Model'].tolist(),
            values=results[cap].tolist(),
            errors=results[f'{cap}_Error'].tolist(),
            xlabel=f'{cap} Score',
            filename=filename,
            color='#E74C3C',
            sort_ascending=True,
            show_reference_line=True
        )

        print(f"  {cap}: {filename}")

def main():
    print("="*60)
    print("FIT MODEL TO ALL GAMES")
    print("="*60)

    df = load_all_games_from_db()
    data, all_models = prepare_game_data(df)
    trace = fit_model(data, all_models)
    results = extract_results(trace, all_models)
    create_plots(results)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

if __name__ == '__main__':
    main()
