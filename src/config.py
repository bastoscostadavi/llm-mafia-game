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
DEFAULT_MODEL_CONFIGS = {
    'detective': {'type': 'local', 'model_path': '/Users/davicosta/Desktop/projects/llm-mafia-game/models/mistral.gguf', 'n_ctx': 2048},
    'mafioso': {'type': 'openai', 'model': 'gpt-4o'},
    'villager': {'type': 'local', 'model_path': '/Users/davicosta/Desktop/projects/llm-mafia-game/models/mistral.gguf', 'n_ctx': 2048}
}

# GAME SETTINGS
DEFAULT_MESSAGE_LIMIT = 200

# TOKEN LIMITS (optimized for cost efficiency)
TOKEN_LIMITS = {
    'discussion': 60,   # ~200 chars message + format
    'voting': 7,        # A name
    'night_action': 7  # A name
}

# GPT-4o specific token limits (cost-optimized)
GPT4O_TOKEN_LIMITS = {
    'discussion': 40,    # GPT-4o is more concise than GPT-5
    'voting': 5,         # Just a name
    'night_action': 5    # Just a name
}

# GPT-5 specific token limits (needs higher limits even with minimal reasoning)
GPT5_TOKEN_LIMITS = {
    'discussion': 120,   # GPT-5 needs more tokens even with minimal reasoning
    'voting': 25,        # Same as night actions
    'night_action': 25   # Same as voting
}

def get_default_prompt_config():
    """Get the default prompt configuration for the system"""
    return PromptConfig(version=DEFAULT_PROMPT_VERSION)

def get_default_model_configs():
    """Get the default model configuration for the system"""
    return DEFAULT_MODEL_CONFIGS.copy()