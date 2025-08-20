# src/config.py
"""
Centralized configuration for the LLM Mafia Game system.

This module provides a single source of truth for system-wide settings,
preventing version mismatches and configuration inconsistencies.
"""


from src.prompts import PromptConfig

# CURRENT SYSTEM VERSION
# This is the single source of truth for prompt version across the entire system
DEFAULT_PROMPT_VERSION = "v4.0"

# MODEL CONFIGURATIONS
# Available local models in models/ directory:
# - Mistral-7B-Instruct-v0.2-Q4_K_M.gguf  # Mistral 7B v0.2 (Dec 2023, 4.1GB) 
# - Mistral-7B-Instruct-v0.3-Q4_K_M.gguf  # Mistral 7B v0.3 (Aug 2024, 4.1GB)
# - Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf # Llama 3.1 8B (4.6GB)
# - Llama-3.2-3B-Instruct-Q4_K_M.gguf     # Llama 3.2 3B (1.9GB, largest text-only 3.2)
# - Qwen2.5-7B-Instruct-Q4_K_M.gguf       # Qwen2.5 7B (4.4GB)
# - qwen2.5-32b-instruct-q4_k_m-*-of-00005.gguf # Qwen2.5 32B (5 parts, ~19GB total)
# - openai_gpt-oss-20b-Q4_K_M.gguf        # GPT-OSS 20B (11GB)

DEFAULT_MODEL_CONFIGS = {
    'detective': {'type': 'local', 'model_path': '/Users/davicosta/Desktop/projects/llm-mafia-game/models/openai_gpt-oss-20b-Q4_K_M.gguf', 'n_ctx': 2048},
    'mafioso': {'type': 'local', 'model_path': '/Users/davicosta/Desktop/projects/llm-mafia-game/models/openai_gpt-oss-20b-Q4_K_M.gguf', 'n_ctx': 2048},
    'villager': {'type': 'local', 'model_path': '/Users/davicosta/Desktop/projects/llm-mafia-game/models/openai_gpt-oss-20b-Q4_K_M.gguf', 'n_ctx': 2048}
}

# GAME SETTINGS
DEFAULT_MESSAGE_LIMIT = 200

# TOKEN LIMITS by model type
# Standard limits for most models
STANDARD_TOKEN_LIMITS = {
    'discussion': 50,   # Balanced between quality and cost
    'voting': 5,        # A name
    'night_action': 5   # A name
}

# Extended limits for reasoning models (GPT-OSS, etc.)
REASONING_TOKEN_LIMITS = {
    'discussion': 250,
    'voting': 50,       
    'night_action': 50  
}

# Model-specific token limit mapping
MODEL_TOKEN_LIMITS = {
    'openai_gpt-oss-20b-Q4_K_M.gguf': REASONING_TOKEN_LIMITS,
    'gpt-oss': REASONING_TOKEN_LIMITS,  # For any GPT-OSS variants
}

# Default token limits (used if model not in specific mapping)
TOKEN_LIMITS = STANDARD_TOKEN_LIMITS


def get_token_limits_for_model(model_config):
    """Get appropriate token limits based on model configuration."""
    if model_config.get('type') == 'local':
        model_path = model_config.get('model_path', '')
        model_filename = model_path.split('/')[-1] if model_path else ''
        
        # Check if this is a reasoning model
        if model_filename in MODEL_TOKEN_LIMITS:
            return MODEL_TOKEN_LIMITS[model_filename]
        elif 'gpt-oss' in model_filename.lower():
            return REASONING_TOKEN_LIMITS
    
    # Default to standard limits
    return STANDARD_TOKEN_LIMITS

def get_default_prompt_config():
    """Get the default prompt configuration for the system"""
    return PromptConfig(version=DEFAULT_PROMPT_VERSION)

def get_default_model_configs():
    """Get the default model configuration for the system"""
    return DEFAULT_MODEL_CONFIGS.copy()