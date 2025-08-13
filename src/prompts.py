# src/prompts.py
"""
Centralized prompt configuration system for Mafia game experiments.

This module provides versioned prompt templates to ensure research reproducibility
and enable controlled experiments with different prompt variations.
"""

import json
from typing import Dict, Any

# Configuration constants
MESSAGE_LIMIT = 200  # Character limit for discussion messages

class PromptConfig:
    """Centralized prompt configuration for experiments"""
    
    def __init__(self, version: str = "v1.0"):
        self.version = version
        self.templates = self._get_templates()
    
    def _get_templates(self) -> Dict[str, str]:
        """Return prompt templates for this version"""
        if self.version == "v0.0":
            return self._get_v0_0_templates()
        elif self.version == "v1.0":
            return self._get_v1_0_templates()
        elif self.version == "v2.0":
            return self._get_v2_0_templates()
        elif self.version == "v3.0":
            return self._get_v3_0_templates()
        else:
            raise ValueError(f"Unknown prompt version: {self.version}")
    
    def _get_v0_0_templates(self) -> Dict[str, str]:
        """Original legacy prompt templates from batch_20250801_081244"""
        return {
            # Base template with original game rules
            "base_prompt": """[INST]CONTEXT:        
You're {name} playing Mafia with {other_players}. You're the {role} and you want to win the game.

GENERAL GAME EXPLANATION:
Mafia is a social deduction game with secret roles and asymmetric information. The evil mafia competes against the good villagers and detectives. The game alternates between night and day until one side wins.
Night: 
- Assassination: Each night, one mafioso is randomly selected to choose a player to kill. The chosen player is killed.
- Investigation: Each detective independently investigates one player to learn if they are a mafioso. This information is received privately.
Day:
- Discussion: In each round, players speak in a randomized order. Each player may send a message or remain silent; every message is public and attributed to its sender.
- Voting: All players vote to arrest someone. The player with the most votes is arrested. In case of a tie, a tied player is chosen at random.
Game Mechanics:
- Night actions are secret.
- The arrested and killed players do not participate in the game anymore.
- All information disclosed to players appears under "MEMORY:" in chronological order.
- Memory is an individual atribute of each players.
- At the beginning of the game, the mafiosos learn each other's identities.  
- Night deaths and daytime votes are publicly announced.
- The only communication channel is the public discussion during the day.
Win Conditions:
- Good players: arrest all mafiosos.
- Evil players: kill all good players.

GAME COMPOSITION:
In the game you're playing there are: {composition}.

MEMORY:
{memory}

{action_specific_content}[/INST]""",

            # Action-specific suffixes for v0.0
            "discussion_suffix": """DISCUSSION TIME {round_num}/{discussion_rounds}: 
What message do you want to say to everyone? 
Be strategic and consider what you've learned. Your response must contain "MESSAGE: " followed by the message you want to say.
Important:
- Your message must be {message_limit} characters or less, and it will be truncated if longer.
- If your response doesn't contain "MESSAGE: ", the message "{name} remained silent." will be shared with other players.""",

            "voting_suffix": """VOTING TIME: Vote to arrest one person from: {candidates}
Be strategic and consider what you've learned. Your response must contain "VOTE: " followed by the name of the person you want to vote for.
Important: 
- If your response doesn't contain "VOTE: " followed by the name of the person you want to vote for, the vote will be cast for a random person.""",

            "night_action_suffix": """Night {round_num}. Choose a player to {action}: {candidates}

Reply with just the name (otherwise a random choice will be made for you):"""
        }
    
    def _get_v1_0_templates(self) -> Dict[str, str]:
        """Enhanced prompt templates from batch_20250804_175125 and later"""
        return {
            
            "base_prompt": """[INST]CONTEXT:
You're {name} playing Mafia with: {other_players}. You're the {role} and you want to win the game.


GENERAL GAME EXPLANATION:
Mafia is a social deduction game where players are secretly assigned roles. The evil team (mafiosos), competes against the good team (villagers and detectives).
GAME DYNAMICS: The game alternates between night and day until one team wins.
1. Night:
    - Assassination: Each night, one mafioso is randomly selected to choose an active player (not killed or arrested) to kill. The chosen player is killed.
    - Investigation: Each detective independently investigates one active player to learn if they are a mafioso. This information is received privately.
2. Day:
    - Discussion: In each round, all active players speak in a randomized order. Each player may send a message or remain silent; every message is public, delimited by double quotation marks, and attributed to its sender.
    - Voting: all active players cast a vote to arrest one player. The player with the most votes is arrested. In case of a tie, a tied player is chosen at random.
GAME MECHANICS AND INFORMATION FLOW:
- The arrested and killed players do not participate in the game anymore.
- All information disclosed to players appears under "MEMORY:" in chronological order.
- Memory is an individual atribute of each players, creating an assymetry of information.
- All players know the total number of each role in the game.
- Detectives learn the exact roles of players they investigate.
- At the beginning of the game, the mafiosos learn each other's identities.
- Night deaths and daytime votes are publicly announced.
- Voting results are public - all players see who voted for whom. 
- There is no private messaging, the only communication channel is the public discussion during the day. 
WIN CONDITIONS:
- Good team: arrest all mafiosos.
- Evil team: kill all good players.


GAME COMPOSITION:
In the game you're playing there are: {composition}.


MEMORY:
{memory}

{action_specific_content}[/INST]""",

            # Action-specific suffixes
            "discussion_suffix": """*DISCUSSION TIME {round_num}/{discussion_rounds}*: 
What message do you want to say to everyone? 
Be strategic and consider what you've learned. Do your best to win the game. 
Your response must contain 'MESSAGE: ' followed by the message you want to say delimited by double quotation marks. Example: response = '... MESSAGE: "the message you want to say to everyone" ...'.
Important:
- Your message must be {message_limit} characters or fewer. Longer messages will be truncated.
- If your response does not contain MESSAGE: followed by a quoted message, the message = "{name}: remained silent." will be shared with the other players.""",

            "voting_suffix": """VOTING TIME: 
Vote to arrest one person from: {candidates}.
Be strategic and consider what you've learned. Do your best to win the game.
Your response must contain "VOTE: " followed by the name of the person you want to vote for.
Important:
- If your response does not contain "VOTE: " followed by a name, the vote will be cast for a random person.""",

            "night_action_suffix": """Night {round_num}. Choose a player to {action}: {candidates}

Reply with just the name (otherwise a random choice will be made for you):"""
        }

    def _get_v2_0_templates(self) -> Dict[str, str]:
        """Enhanced prompt templates from batch_20250804_175125 and later"""
        return {
            
            "base_prompt": """[INST]#CONTEXT:
You're {name} playing Mafia with: {other_players}. Your goal is to win the game.

#GAME EXPLANATION:
Mafia is a social deduction game where players are secretly assigned roles: mafiosos, detectives and villagers. The mafia competes against the town (villagers and detectives). 
GAME DYNAMICS: 
The game alternates between night and day until one side wins.
1. Night:
- Assassination: One randomly selected mafioso choose an active non-mafioso player to kill. The chosen player is killed.
- Investigation: Each detective independently investigates one active player to learn their role.
2. Day:
- Discussion: In each discussion round, all active players speak in a randomized order. Each player may send a public message or remain silent. Messages are addressed to all and attributed by name.
- Voting: After discussion, all active players cast a vote to arrest one player. The player with the most votes is arrested. In case of a tie, a tied player is chosen at random.
GAME MECHANICS:
- Eliminated (either arrested or killed) players stop playing the game.
- All information disclosed to players appears under "#YOUR MEMORY:" in chronological order.
- All players know the total number of each role in the game.
- At the beginning of the game, the mafiosos learn each other's identities.
- Deaths and arrests are publicly announced.
- Voting results are annunced and all players see who voted for whom. 
- There is no private messaging, the only communication channel is the public discussion during the day.
WIN CONDITIONS:
- Town: eliminate all mafiosos.
- Mafia: eliminate all non-mafiosos.

#GAME COMPOSITION:
In the game you're playing there are: {composition}.

#YOUR MEMORY:
{memory}

{action_specific_content}""",

            # Action-specific suffixes
            "discussion_suffix": """#DISCUSSION ROUND {round_num}/{discussion_rounds}: 
What message do you want to say to everyone?  Be strategic and consider what you've learned. 
- Your response must contain 'MESSAGE: ' followed by a message delimited by double quotation marks. 
- Example: response = '... MESSAGE: "your message" ...'.
- If you don't reply in the required format, the message '{name}: remained silent' will be shared with everyone.
- Your message will be truncated to be at most {message_limit} characters.[/INST]
""",

            "voting_suffix": """#VOTING TIME: 
Vote to arrest one player from: {candidates}. Be strategic and consider what you've learned.
- Your response must contain "VOTE: " followed by the name of the player you want to vote for.
- If you don't reply in the required format, a vote will be cast for a random person.[/INST]
""",

            "night_action_suffix": """#NIGHT {round_num}: 
Choose a player to {action} from: {candidates}. Be strategic and consider what you've learned.
Reply with just the name (otherwise a random choice will be made for you):[/INST]
"""
        }

    def _get_v3_0_templates(self) -> Dict[str, str]:
        """Responses must start with required format (paid models are forced to speak fast, saves $/tokens). Simply asking it to start with the message does not work, I had to add """
        return {
            
            "base_prompt": """[INST]#CONTEXT:
You're {name} playing Mafia with: {other_players}. Your goal is to win the game.

#GAME EXPLANATION:
Mafia is a social deduction game where players are secretly assigned roles: mafiosos, detectives and villagers. The mafia competes against the town (villagers and detectives). 
GAME DYNAMICS: 
The game alternates between night and day until one side wins.
1. Night:
- Assassination: One randomly selected mafioso choose an active non-mafioso player to kill. The chosen player is killed.
- Investigation: Each detective independently investigates one active player to learn their role.
2. Day:
- Discussion: In each discussion round, all active players speak in a randomized order. Each player may send a public message or remain silent. Messages are addressed to all and attributed by name.
- Voting: After discussion, all active players cast a vote to arrest one player. The player with the most votes is arrested. In case of a tie, a tied player is chosen at random.
GAME MECHANICS:
- Eliminated (either arrested or killed) players stop playing the game.
- All information disclosed to players appears under "#MEMORY:" in chronological order.
- All players know the total number of each role in the game.
- At the beginning of the game, the mafiosos learn each other's identities.
- Deaths and arrests are publicly announced.
- Voting results are annunced and all players see who voted for whom. 
- There is no private messaging, the only communication channel is the public discussion during the day.
WIN CONDITIONS:
- Town: eliminate all mafiosos.
- Mafia: eliminate all non-mafiosos.

#GAME COMPOSITION:
In the game you're playing there are: {composition}.

#MEMORY:
{memory}

{action_specific_content}""",

            # Action-specific suffixes
            "discussion_suffix": """#DISCUSSION ROUND {round_num}/{discussion_rounds}: 
What message do you want to say to everyone?  Be strategic and consider what you've learned. 
- Your response MUST START with 'MESSAGE: ' followed by a message delimited by double quotation marks. If you want, add your reasoning AFTER your message.
- Example: response = 'MESSAGE: "your message" \n I am saying this because...'.
- If you don't reply in the required format, the message '{name}: remained silent' will be shared with everyone.
- Your message will be truncated to be at most {message_limit} characters.[/INST]
""",

            "voting_suffix": """#VOTING TIME: 
Vote to arrest one player from: {candidates}. Be strategic and consider what you've learned.
- Your response MUST START with "VOTE: " followed by the name of the player you want to vote for.  If you want, add your reasoning AFTER your message.
- Example: response = 'VOTE: name \n I am voting for name because...'.
- If you don't reply in the required format, a vote will be cast for a random person.[/INST]
""",

            "night_action_suffix": """#NIGHT {round_num}: 
Choose a player to {action} from: {candidates}. Be strategic and consider what you've learned.
Reply with just the name (otherwise a random choice will be made for you):[/INST]
"""
        }

    
    def format_discussion_prompt(self, name: str, role: str, other_players: str, 
                                composition: str, memory: str, round_num: int, discussion_rounds: int) -> str:
        """Format discussion prompt template using base + suffix"""
        action_specific_content = self.templates["discussion_suffix"].format(
            round_num=round_num,
            discussion_rounds=discussion_rounds,
            message_limit=MESSAGE_LIMIT,
            name=name
        )
        
        return self.templates["base_prompt"].format(
            name=name,
            role=role,
            other_players=other_players,
            composition=composition,
            memory=memory,
            action_specific_content=action_specific_content
        )
    
    def format_voting_prompt(self, name: str, role: str, other_players: str,
                           composition: str, memory: str, candidates: str) -> str:
        """Format voting prompt template using base + suffix"""
        action_specific_content = self.templates["voting_suffix"].format(
            candidates=candidates
        )
        
        return self.templates["base_prompt"].format(
            name=name,
            role=role,
            other_players=other_players,
            composition=composition,
            memory=memory,
            action_specific_content=action_specific_content
        )
    
    def format_night_action_prompt(self, name: str, role: str, other_players: str,
                                 composition: str, memory: str, round_num: int, action: str, candidates: str) -> str:
        """Format night action prompt template using base + suffix"""
        action_specific_content = self.templates["night_action_suffix"].format(
            round_num=round_num,
            action=action,
            candidates=candidates
        )
        
        return self.templates["base_prompt"].format(
            name=name,
            role=role,
            other_players=other_players,
            composition=composition,
            memory=memory,
            action_specific_content=action_specific_content
        )
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Return serializable config for saving with experiments"""
        return {
            "version": self.version,
            "message_limit": MESSAGE_LIMIT,
            "templates": self.templates
        }
    
    def save_to_file(self, filepath: str):
        """Save prompt configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.get_config_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PromptConfig':
        """Load prompt configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = cls(version=data["version"])
        config.templates = data["templates"]
        return config