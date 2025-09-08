#!/usr/bin/env python3
"""
Hierarchical Bayesian Analysis for Mini-Mafia Results

Implements the logistic mixed-effects model described in the paper:
- z_ij ~ Binomial(n_ij, p_ij)  
- logit(p_ij) = α_i + β_j
- α_i ~ Normal(μ_α, σ_α²)  # Model abilities
- β_j ~ Normal(0, σ_β²)    # Background effects

Generates:
- 15 win rate plots showing p̂_ij = logit^(-1)(α_i + β_j)  
- 3 aggregated score plots showing α_i (model abilities)
"""

import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
from collections import defaultdict
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import os

# For Bayesian inference (required)
import pymc as pm
import arviz as az

warnings.filterwarnings('ignore')

def get_display_name(model_name):
    """Convert internal model names to display names"""
    display_names = {
        'claude-opus-4-1-20250805': 'Claude Opus 4.1',
        'claude-sonnet-4-20250514': 'Claude Sonnet 4', 
        'deepseek-chat': 'DeepSeek V3.1',
        'deepseek-reasoner': 'DeepSeek R1',
        'gemini-2.5-flash-lite': 'Gemini 2.5 Flash Lite',
        'grok-3-mini': 'Grok 3 Mini',
        'gpt-4.1-mini': 'GPT-4.1 Mini',
        'gpt-5': 'GPT-5',
        'gpt-5-mini': 'GPT-5 Mini',
        'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf': 'Mistral 7B Instruct',
        'Qwen2.5-7B-Instruct-Q4_K_M.gguf': 'Qwen2.5 7B Instruct',
        'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf': 'Llama 3.1 8B'
    }
    return display_names.get(model_name, model_name)

def get_company_from_model(model_name):
    """Get company from model name"""
    company_map = {
        'Claude Opus 4.1': 'Anthropic',
        'Claude Sonnet 4': 'Anthropic',
        'DeepSeek V3.1': 'DeepSeek', 
        'DeepSeek R1': 'DeepSeek',
        'Gemini 2.5 Flash Lite': 'Google',
        'Grok 3 Mini': 'X',
        'GPT-4.1 Mini': 'OpenAI',
        'GPT-5': 'OpenAI',
        'GPT-5 Mini': 'OpenAI',
        'Mistral 7B Instruct': 'Mistral AI',
        'Qwen2.5 7B Instruct': 'Alibaba',
        'Llama 3.1 8B': 'Meta'
    }
    return company_map.get(model_name, 'Unknown')

def analyze_database_results(db_path='../database/mini_mafia.db'):
    """Extract data for hierarchical Bayesian analysis"""
    conn = sqlite3.connect(db_path)
    
    # SQL query to get all game results with model assignments
    query = """
    SELECT 
        g.game_id,
        g.winner,
        MAX(CASE WHEN gp.role='mafioso' THEN p.model_name END) as mafioso_model,
        MAX(CASE WHEN gp.role='detective' THEN p.model_name END) as detective_model,
        MAX(CASE WHEN gp.role='villager' THEN p.model_name END) as villager_model
    FROM games g
    JOIN game_players gp ON g.game_id = gp.game_id
    JOIN players p ON gp.player_id = p.player_id
    WHERE p.player_id IS NOT NULL
    GROUP BY g.game_id
    HAVING mafioso_model IS NOT NULL 
       AND detective_model IS NOT NULL 
       AND villager_model IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def prepare_hierarchical_data(df, experiment_type):
    """Prepare data for hierarchical Bayesian analysis"""
    
    if experiment_type == 'mafioso':
        # Mafioso experiments: vary mafioso, fix detective+villager as background
        df_exp = df[df['detective_model'] == df['villager_model']].copy()
        df_exp['varying_model'] = df_exp['mafioso_model'] 
        df_exp['background'] = df_exp['detective_model']
        df_exp['success'] = (df_exp['winner'] == 'mafia').astype(int)
        
    elif experiment_type == 'detective':
        # Detective experiments: vary detective, fix mafioso+villager as background  
        df_exp = df[df['mafioso_model'] == df['villager_model']].copy()
        df_exp['varying_model'] = df_exp['detective_model']
        df_exp['background'] = df_exp['mafioso_model'] 
        df_exp['success'] = (df_exp['winner'] == 'town').astype(int)
        
    elif experiment_type == 'villager':
        # Villager experiments: vary villager, fix mafioso+detective as background
        df_exp = df[df['mafioso_model'] == df['detective_model']].copy()
        df_exp['varying_model'] = df_exp['villager_model']
        df_exp['background'] = df_exp['mafioso_model']
        df_exp['success'] = (df_exp['winner'] == 'town').astype(int)
        
    else:
        raise ValueError("experiment_type must be 'mafioso', 'detective', or 'villager'")
    
    # Convert to display names
    df_exp['model_display'] = df_exp['varying_model'].apply(get_display_name)
    df_exp['background_display'] = df_exp['background'].apply(get_display_name)
    
    return df_exp


def fit_hierarchical_model(df_exp):
    """Fit hierarchical Bayesian model using PyMC"""
    
    # Prepare data
    models = sorted(df_exp['model_display'].unique())
    backgrounds = sorted(df_exp['background_display'].unique())
    
    # Create index mappings
    model_idx = {model: i for i, model in enumerate(models)}
    bg_idx = {bg: i for i, bg in enumerate(backgrounds)}
    
    df_exp['model_i'] = df_exp['model_display'].map(model_idx)
    df_exp['bg_j'] = df_exp['background_display'].map(bg_idx)
    
    # Aggregate data for binomial likelihood
    agg_data = df_exp.groupby(['model_i', 'bg_j']).agg({
        'success': ['sum', 'count']
    }).reset_index()
    agg_data.columns = ['model_i', 'bg_j', 'successes', 'trials']
    
    with pm.Model() as model:
        # Hyperpriors
        mu_alpha = pm.Normal('mu_alpha', 0, 2)
        sigma_alpha = pm.HalfNormal('sigma_alpha', 1)
        sigma_beta = pm.HalfNormal('sigma_beta', 1)
        
        # Model abilities (α_i)
        alpha = pm.Normal('alpha', mu_alpha, sigma_alpha, shape=len(models))
        
        # Background effects (β_j) with sum-to-zero constraint
        beta_raw = pm.Normal('beta_raw', 0, sigma_beta, shape=len(backgrounds)-1)
        beta = pm.Deterministic('beta', 
                               pm.math.concatenate([beta_raw, [-pm.math.sum(beta_raw)]]))
        
        # Linear predictor
        logit_p = alpha[agg_data['model_i'].values] + beta[agg_data['bg_j'].values]
        
        # Likelihood
        obs = pm.Binomial('obs', n=agg_data['trials'].values, 
                         logit_p=logit_p, observed=agg_data['successes'].values)
        
        # Sample posterior
        trace = pm.sample(2000, tune=1000, chains=2, 
                         target_accept=0.95, random_seed=42,
                         return_inferencedata=True,
                         compute_convergence_checks=False)
    
    # Extract posterior summaries directly from trace
    alpha_samples = trace.posterior.alpha  # Shape: (chain, draw, model)
    beta_samples = trace.posterior.beta    # Shape: (chain, draw, background)
    
    # Flatten across chains and compute statistics
    alpha_flat = alpha_samples.values.reshape(-1, len(models))  # (total_draws, models)
    beta_flat = beta_samples.values.reshape(-1, len(backgrounds))  # (total_draws, backgrounds)
    
    alpha_post = {models[i]: float(np.mean(alpha_flat[:, i])) 
                  for i in range(len(models))}
    alpha_std = {models[i]: float(np.std(alpha_flat[:, i])) 
                 for i in range(len(models))}
    
    beta_post = {backgrounds[i]: float(np.mean(beta_flat[:, i])) 
                 for i in range(len(backgrounds))}
    beta_std = {backgrounds[i]: float(np.std(beta_flat[:, i])) 
                for i in range(len(backgrounds))}
    
    return {
        'alpha': alpha_post,
        'beta': beta_post,
        'alpha_std': alpha_std, 
        'beta_std': beta_std,
        'models': models,
        'backgrounds': backgrounds,
        'trace': trace
    }

def create_win_rate_plot(results, df_exp, experiment_type, background_name, filename):
    """Create win rate plot showing p̂_ij = logit^(-1)(α_i + β_j)"""
    
    # Use non-interactive backend
    plt.ioff()
    
    # Set font size
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
        'figure.titlesize': 24
    })
    
    # Calculate predicted win rates: p̂_ij = logit^(-1)(α_i + β_j)
    plot_data = []
    for model in results['models']:
        if model in results['alpha'] and background_name in results['beta']:
            logit_p = results['alpha'][model] + results['beta'][background_name]
            win_rate = 100 / (1 + np.exp(-logit_p))  # Convert to percentage
            
            # Uncertainty via delta method: Var(logit^(-1)(x)) ≈ (dlogit^(-1)/dx)² * Var(x)
            # d/dx logit^(-1)(x) = logit^(-1)(x) * (1 - logit^(-1)(x))
            p = win_rate / 100
            derivative = p * (1 - p)
            var_logit = results['alpha_std'][model]**2 + results['beta_std'][background_name]**2
            std_p = derivative * np.sqrt(var_logit) * 100  # Convert to percentage
            
            plot_data.append({
                'model': model,
                'win_rate': win_rate,
                'std_err': std_p,
                'company': get_company_from_model(model)
            })
    
    # Sort by win rate
    plot_data.sort(key=lambda x: x['win_rate'])
    
    models = [item['model'] for item in plot_data]
    win_rates = [item['win_rate'] for item in plot_data]
    errors = [item['std_err'] for item in plot_data]
    companies = [item['company'] for item in plot_data]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    y_positions = range(len(models))
    
    # Create bars
    bars = ax.barh(y_positions, win_rates, xerr=errors,
                   color='#4A90E2', alpha=0.8, height=0.6,
                   error_kw={'capsize': 5, 'capthick': 2})
    
    # Add model names
    for i, (model, win_rate, error) in enumerate(zip(models, win_rates, errors)):
        ax.text(max(win_rate + error + 2, 5), i,
                f'{model}',
                ha='left', va='center', fontweight='bold', fontsize=24)
    
    # Formatting
    role_labels = {
        'mafioso': 'Mafia Victory %',
        'detective': 'Town Victory %', 
        'villager': 'Town Victory %'
    }
    ax.set_xlabel(role_labels.get(experiment_type, 'Victory %'), fontsize=24, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Hierarchical Bayesian win rate plot saved as {filename}")

def create_ability_plot(all_results, experiment_type, filename):
    """Create plot showing model abilities (α_i) across all backgrounds"""
    
    # Use non-interactive backend  
    plt.ioff()
    
    # Set font size
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
        'figure.titlesize': 24
    })
    
    # Aggregate α_i estimates across all backgrounds (should be consistent)
    # Take the first background's results since α_i should be the same
    first_bg = list(all_results.keys())[0]
    results = all_results[first_bg]
    
    # Sort by ability
    models = list(results['alpha'].keys())
    abilities = [results['alpha'][model] for model in models]
    ability_stds = [results['alpha_std'][model] for model in models]
    
    # Sort by ability (ascending, so best performers at top)
    sorted_indices = np.argsort(abilities)
    models = [models[i] for i in sorted_indices]
    abilities = [abilities[i] for i in sorted_indices]  
    ability_stds = [ability_stds[i] for i in sorted_indices]
    companies = [get_company_from_model(model) for model in models]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    y_positions = range(len(models))
    
    # Create bars
    bars = ax.barh(y_positions, abilities, xerr=ability_stds,
                   color='#E74C3C', alpha=0.8, height=0.6,
                   error_kw={'capsize': 5, 'capthick': 2})
    
    # Add model names
    for i, (model, ability, ability_std) in enumerate(zip(models, abilities, ability_stds)):
        x_pos = max(ability + ability_std + 0.1, 0.1)
        ax.text(x_pos, i, f'{model}',
                ha='left', va='center', fontweight='bold', fontsize=24)
    
    # Set axis labels
    behavior_labels = {
        'mafioso': 'Deceive Ability (α_i)',
        'detective': 'Disclose Ability (α_i)',
        'villager': 'Detect Ability (α_i)'
    }
    xlabel = behavior_labels.get(experiment_type, 'Model Ability (α_i)')
    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold')
    ax.set_yticks([])
    
    # Set x-axis limits
    min_val = min([a - s for a, s in zip(abilities, ability_stds)])
    max_val = max([a + s for a, s in zip(abilities, ability_stds)])
    margin = (max_val - min_val) * 0.1
    ax.set_xlim(min_val - margin, max_val + margin)
    
    # Add vertical line at α=0
    ax.axvline(x=0, color='gray', alpha=0.5, linewidth=1, linestyle='--')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Hierarchical Bayesian ability plot saved as {filename}")

def create_ability_plot_exponential(all_results, experiment_type, filename):
    """Create plot showing exponentiated model abilities exp(α_i) across all backgrounds"""
    
    # Use non-interactive backend  
    plt.ioff()
    
    # Set font size
    plt.rcParams.update({
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24,
        'figure.titlesize': 24
    })
    
    # Aggregate α_i estimates across all backgrounds (should be consistent)
    # Take the first background's results since α_i should be the same
    first_bg = list(all_results.keys())[0]
    results = all_results[first_bg]
    
    # Sort by ability
    models = list(results['alpha'].keys())
    abilities = [results['alpha'][model] for model in models]
    ability_stds = [results['alpha_std'][model] for model in models]
    
    # Exponentiate: exp(α_i) with error propagation
    exp_abilities = [np.exp(alpha) for alpha in abilities]
    exp_ability_stds = [np.exp(alpha) * std for alpha, std in zip(abilities, ability_stds)]
    
    # Sort by exponentiated ability (ascending, so best performers at top)
    sorted_indices = np.argsort(exp_abilities)
    models = [models[i] for i in sorted_indices]
    exp_abilities = [exp_abilities[i] for i in sorted_indices]  
    exp_ability_stds = [exp_ability_stds[i] for i in sorted_indices]
    companies = [get_company_from_model(model) for model in models]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    y_positions = range(len(models))
    
    # Create bars
    bars = ax.barh(y_positions, exp_abilities, xerr=exp_ability_stds,
                   color='#E74C3C', alpha=0.8, height=0.6,
                   error_kw={'capsize': 5, 'capthick': 2})
    
    # Add model names
    for i, (model, exp_ability, exp_ability_std) in enumerate(zip(models, exp_abilities, exp_ability_stds)):
        x_pos = max(exp_ability + exp_ability_std + 0.05, 0.05)
        ax.text(x_pos, i, f'{model}',
                ha='left', va='center', fontweight='bold', fontsize=24)
    
    # Set axis labels
    behavior_labels = {
        'mafioso': 'Deceive Score',
        'detective': 'Disclose Score',
        'villager': 'Detect Score'
    }
    xlabel = behavior_labels.get(experiment_type, 'Model Score')
    ax.set_xlabel(xlabel, fontsize=24, fontweight='bold')
    ax.set_yticks([])
    
    # Set data-driven x-axis limits with some padding
    max_val = max([s + e for s, e in zip(exp_abilities, exp_ability_stds)])
    min_val = min([s - e for s, e in zip(exp_abilities, exp_ability_stds)])
    padding = (max_val - min_val) * 0.1
    x_min = max(0, min_val - padding)
    x_max = max_val + padding
    ax.set_xlim(x_min, x_max)
    
    # Add vertical line at exp(0) = 1 - the key reference point
    if x_min <= 1 <= x_max:  # Only show if 1 is in the visible range
        ax.axvline(x=1, color='gray', alpha=0.7, linewidth=2, linestyle='--')
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Hide spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Hierarchical Bayesian exponentiated ability plot saved as {filename}")

def main():
    """Main function to generate hierarchical Bayesian plots"""
    
    print("🔄 Running Hierarchical Bayesian Analysis...")
    
    # Analyze database
    print("📊 Loading data from database...")
    df = analyze_database_results()
    print(f"   Found {len(df)} games")
    
    # Define experiment types
    experiment_types = {
        'mafioso': 'Deceive',
        'detective': 'Disclose', 
        'villager': 'Detect'
    }
    
    # Define allowed models and backgrounds
    allowed_models = [
        'DeepSeek V3.1', 'Claude Sonnet 4', 'Gemini 2.5 Flash Lite',
        'Grok 3 Mini', 'GPT-4.1 Mini',  'GPT-5 Mini', 'Mistral 7B Instruct', 
        'Qwen2.5 7B Instruct', 'Llama 3.1 8B', 'Claude Opus 4.1'#,'GPT-5'
    ]
    
    allowed_backgrounds = [
        'GPT-4.1 Mini', 'GPT-5 Mini', 'Grok 3 Mini', 'DeepSeek V3.1', 'Mistral 7B Instruct', 
    ]
    
    # Process each experiment type
    for exp_type, behavior_name in experiment_types.items():
        print(f"\n📊 Processing {behavior_name} ({exp_type}) experiments...")
        
        # Prepare data
        df_exp = prepare_hierarchical_data(df, exp_type)
        
        # Filter to allowed models and backgrounds
        df_exp = df_exp[df_exp['model_display'].isin(allowed_models)]
        df_exp = df_exp[df_exp['background_display'].isin(allowed_backgrounds)]
        
        print(f"   Filtered to {len(df_exp)} games")
        print(f"   Models: {sorted(df_exp['model_display'].unique())}")
        print(f"   Backgrounds: {sorted(df_exp['background_display'].unique())}")
        
        # Fit hierarchical model for each background
        all_results = {}
        for background in allowed_backgrounds:
            if background in df_exp['background_display'].values:
                print(f"   Fitting model for {background} background...")
                df_bg = df_exp[df_exp['background_display'] == background]
                
                try:
                    results = fit_hierarchical_model(df_bg)
                    print(f"     Used Bayesian inference")
                    all_results[background] = results
                    
                    # Create win rate plot
                    filename = f"{exp_type}_{background.replace(' ', '_').lower()}_hierarchical_bayesian.png"
                    create_win_rate_plot(results, df_bg, exp_type, background, filename)
                    
                except Exception as e:
                    print(f"     Error fitting model for {background}: {e}")
                    continue
        
        # Create aggregated ability plot (both regular and exponentiated)
        if all_results:
            ability_filename = f"{exp_type}_ability_hierarchical_bayesian.png"  
            create_ability_plot(all_results, exp_type, ability_filename)
            
            # Create exponentiated ability plot
            exp_ability_filename = f"{exp_type}_ability_hierarchical_bayesian_exponential.png"
            create_ability_plot_exponential(all_results, exp_type, exp_ability_filename)
    
    print("\n✅ Hierarchical Bayesian analysis complete!")
    print(f"📁 Generated plots in current directory")

if __name__ == "__main__":
    main()