# src/prompt_utils.py
"""
Simple prompt utilities that load from prompt.txt file.
"""

import re
from pathlib import Path
from typing import List

# Configuration constants
MESSAGE_LIMIT = 200

def load_base_prompt() -> str:
    """Load the base prompt template from prompt.txt"""
    prompt_file = Path(__file__).parent / "prompt.txt"
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

# Cache the base prompt
_BASE_PROMPT = load_base_prompt()

def format_discussion_prompt(name: str, other_players: str, composition: str, 
                           memory: str, round_num: int, discussion_rounds: int) -> str:
    """Format discussion prompt"""
    action_suffix = f"""#DISCUSSION ROUND {round_num}/{discussion_rounds}: 
What message do you want to say to everyone?
Your response must start with your message in double quotation marks, followed by optional reasoning:
"""
    
    return _BASE_PROMPT.format(
        name=name,
        other_players=other_players,
        composition=composition,
        memory=memory,
        action_specific_content=action_suffix
    )

def format_voting_prompt(name: str, other_players: str, composition: str, 
                        memory: str, candidates: str) -> str:
    """Format voting prompt"""
    action_suffix = f"""#VOTING TIME: 
Vote to arrest one player from: {candidates}.
Reply with just a name:
"""
    
    return _BASE_PROMPT.format(
        name=name,
        other_players=other_players,
        composition=composition,
        memory=memory,
        action_specific_content=action_suffix
    )

def format_night_action_prompt(name: str, other_players: str, composition: str,
                              memory: str, round_num: int, action: str, candidates: str) -> str:
    """Format night action prompt"""
    action_suffix = f"""#NIGHT {round_num}: 
Choose a player to {action} from: {candidates}.
Reply with just a name:
"""
    
    return _BASE_PROMPT.format(
        name=name,
        other_players=other_players,
        composition=composition,
        memory=memory,
        action_specific_content=action_suffix
    )

def parse_discussion_response(response: str) -> str:
    """Parse discussion response - expects format: "message" \\n reasoning..."""
    # Handle GPT-OSS Harmony format: extract content after final channel
    if '<|channel|>final<|message|>' in response:
        response = response.split('<|channel|>final<|message|>', 1)[1]
        response = response.split('<|')[0].strip()
    
    # Handle GPT-OSS reasoning format: extract content after final<|message|>
    if 'final<|message|>' in response:
        response = response.split('final<|message|>')[-1].strip()
    
    # Split on newline and take only the first line to avoid capturing reasoning
    first_line = response.split('\n')[0].strip()
    
    # First try to match complete quoted message on first line
    match = re.search(r'^\s*"([^"]+)"', first_line)
    if match:
        return match.group(1)
    
    # If no complete quote found, check for truncated message on first line
    truncated_match = re.search(r'^\s*"([^"]*)', first_line)
    if truncated_match and len(truncated_match.group(1)) > 0:
        return f'{truncated_match.group(1)}...'
    
    return None

def parse_voting_response(response: str, candidates: List[str]) -> str:
    """Parse voting response - expects format: player_name \\n reasoning..."""
    # Handle GPT-OSS formats
    if '<|channel|>final<|message|>' in response:
        response = response.split('<|channel|>final<|message|>', 1)[1]
        response = response.split('<|')[0].strip()
    
    if 'final<|message|>' in response:
        response = response.split('final<|message|>')[-1].strip()
    
    response_lines = response.strip().split('\n')
    first_line = response_lines[0].strip()
    
    # First try exact match on first line
    for candidate in candidates:
        if candidate.lower() == first_line.lower():
            return candidate
    
    # If no exact match, look for candidate names anywhere in the response
    for candidate in candidates:
        if candidate.lower() in response.lower():
            return candidate
    
    return None

def parse_night_action_response(response: str, candidates: List[str]) -> str:
    """Parse night action response - expects format: player_name \\n reasoning..."""
    # Handle GPT-OSS formats
    if '<|channel|>final<|message|>' in response:
        response = response.split('<|channel|>final<|message|>', 1)[1]
        response = response.split('<|')[0].strip()
    
    if 'final<|message|>' in response:
        response = response.split('final<|message|>')[-1].strip()
    
    response_lines = response.strip().split('\n')
    first_line = response_lines[0].strip()
    
    # First try exact match on first line
    for candidate in candidates:
        if candidate.lower() == first_line.lower():
            return candidate
    
    # If no exact match, look for candidate names anywhere in the response
    for candidate in candidates:
        if candidate.lower() in response.lower():
            return candidate
    
    return None