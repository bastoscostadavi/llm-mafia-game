# src/config.py
"""
Centralized configuration for the LLM Mafia Game system.

This module provides a single source of truth for system-wide settings,
preventing version mismatches and configuration inconsistencies.
"""


from src.prompts import PromptConfig

# CURRENT SYSTEM VERSION
# This is the single source of truth for prompt version across the entire system
DEFAULT_PROMPT_VERSION = "v4.1"

# MODEL CONFIGURATIONS
# Available local models in models/ directory:
# - Mistral-7B-Instruct-v0.2-Q4_K_M.gguf  # Mistral 7B v0.2 (Dec 2023, 4.1GB) 
# - Mistral-7B-Instruct-v0.3-Q4_K_M.gguf  # Mistral 7B v0.3 (Aug 2024, 4.1GB)
# - Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf # Llama 3.1 8B (4.6GB)
# - Llama-3.2-3B-Instruct-Q4_K_M.gguf     # Llama 3.2 3B (1.9GB, largest text-only 3.2)
# - Qwen2.5-7B-Instruct-Q4_K_M.gguf       # Qwen2.5 7B (4.4GB)
# - qwen2.5-32b-instruct-q4_k_m-*-of-00005.gguf # Qwen2.5 32B (5 parts, ~19GB total)
# - openai_gpt-oss-20b-Q4_K_M.gguf        # GPT-OSS 20B (11GB)
# - gemma-2-27b-it-Q4_K_M.gguf            # Gemma 2 27B (16GB)
# - Qwen3-14B-Q4_K_M.gguf        # Qwen3 14B (~8GB)
# EX: 'mafioso': {'type': 'local', 'model': 'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf', 'temperature': 0.7},

# Available OpenAI/Anthropic/xAI models:
# - gpt-4o                      # OpenAI GPT-4o
# - gpt-5                       # OpenAI GPT-5 (reasoning model)
# - gpt-4.1-mini                  # OpenAI GPT-4.1 mini
# - grok-4                      # xAI Grok-4 (reasoning model)
# - grok-3-mini                   # xAI Grok-3 mini
# - claude-sonnet-4-20250514    #Claude Sonnet
# - claude-opus-4-1-20250805    #Claude Opus
# EX: 'mafioso': {'type': 'openai', 'model': 'gpt-4o', 'temperature': 0.7},
# EX: 'mafioso': {'type': 'openai', 'model': 'gpt-5', 'temperature': 0.7},
# EX: 'mafioso': {'type': 'xai', 'model': 'grok-4', 'temperature': 0.7},

DEFAULT_MODEL_CONFIGS = {
    'detective': {'type': 'openai', 'model': 'gpt-4.1-mini', 'temperature': 0.7},
    'mafioso': {'type': 'local', 'model': 'Mistral-7B-Instruct-v0.2-Q4_K_M.gguf', 'temperature': 0.7},
    'villager': {'type': 'openai', 'model': 'gpt-4.1-mini', 'temperature': 0.7}
}

# GAME SETTINGS
DEFAULT_MESSAGE_LIMIT = 200

# TOKEN LIMITS by model type
# Standard limits for most models
STANDARD_TOKEN_LIMITS = {
    'discussion': 55,   # Balanced between quality and cost
    'voting': 5,        # A name
    'night_action': 5   # A name
}

# Extended limits for reasoning models (GPT-OSS, etc.)
#For reasoning_effort = "miminum" , 1000,100,100 is enough
#For reasoning_effort = "medium" , 2000,1000,1000 is enough
REASONING_TOKEN_LIMITS = {
    'discussion': 4000,   # Increased for Grok-4 which may need more tokens
    'voting': 4000,       # Increased from 1000
    'night_action': 4000  # Increased from 1000
}

# Model-specific token limit mapping
MODEL_TOKEN_LIMITS = {
    'openai_gpt-oss-20b-Q4_K_M.gguf': REASONING_TOKEN_LIMITS,
    'gpt-oss': REASONING_TOKEN_LIMITS,  # For any GPT-OSS variants
    'gpt-5': REASONING_TOKEN_LIMITS,    # OpenAI GPT-5 reasoning model
    'grok-4': REASONING_TOKEN_LIMITS,   # xAI Grok-4 reasoning model
}

# Default token limits (used if model not in specific mapping)
TOKEN_LIMITS = STANDARD_TOKEN_LIMITS


def get_token_limits_for_model(model_config):
    """Get appropriate token limits based on model configuration."""
    if model_config.get('type') == 'local':
        model_filename = model_config.get('model', '')
        
        # Check if this is a reasoning model
        if model_filename in MODEL_TOKEN_LIMITS:
            return MODEL_TOKEN_LIMITS[model_filename]
        elif 'gpt-oss' in model_filename.lower():
            return REASONING_TOKEN_LIMITS
    
    elif model_config.get('type') == 'openai':
        model_name = model_config.get('model', '')
        
        # Check if this is a reasoning model
        if model_name in MODEL_TOKEN_LIMITS:
            return MODEL_TOKEN_LIMITS[model_name]
        elif model_name.startswith('gpt-5'):
            return REASONING_TOKEN_LIMITS
    
    elif model_config.get('type') == 'xai':
        model_name = model_config.get('model', '')
        
        # Check if this is a reasoning model
        if model_name in MODEL_TOKEN_LIMITS:
            return MODEL_TOKEN_LIMITS[model_name]
        elif model_name.startswith('grok'):
            return REASONING_TOKEN_LIMITS
    
    # Default to standard limits
    return STANDARD_TOKEN_LIMITS

def get_default_prompt_config():
    """Get the default prompt configuration for the system"""
    return PromptConfig(version=DEFAULT_PROMPT_VERSION)

def get_default_model_configs():
    """Get the default model configuration for the system"""
    return DEFAULT_MODEL_CONFIGS.copy()