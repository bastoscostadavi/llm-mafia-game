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
    'mafioso': {'type': 'anthropic', 'model': 'claude-sonnet-4-20250514', 'temperature': 0.7, 'use_cache': True},
    'villager': {'type': 'local', 'model_path': '/Users/davicosta/Desktop/projects/llm-mafia-game/models/mistral.gguf', 'n_ctx': 2048}
}

# GAME SETTINGS
DEFAULT_MESSAGE_LIMIT = 200

# TOKEN LIMITS (standard for all models)
TOKEN_LIMITS = {
    'discussion': 50,   # Balanced between quality and cost
    'voting': 5,        # A name
    'night_action': 5   # A name
}

def get_default_prompt_config():
    """Get the default prompt configuration for the system"""
    return PromptConfig(version=DEFAULT_PROMPT_VERSION)

def get_default_model_configs():
    """Get the default model configuration for the system"""
    return DEFAULT_MODEL_CONFIGS.copy()