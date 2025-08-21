# src/prompts.py
"""
Centralized prompt configuration system for Mafia game experiments.

This module provides versioned prompt templates to ensure research reproducibility
and enable controlled experiments with different prompt variations.
"""

import json
import re
from typing import Dict, Any, List

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
        elif self.version == "v4.0":
            return self._get_v4_0_templates()
        elif self.version == "v4.1":
            return self._get_v4_1_templates()
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

    def _get_v4_0_templates(self) -> Dict[str, str]:
        """
        Optimized for prompt caching (Anthropic + OpenAI).
        
        Key changes from v3.0:
        - Reordered prompt structure to maximize cache efficiency
        - Static game rules and explanations moved to beginning (cacheable)
        - Dynamic content (name, memory, actions) moved to end (non-cacheable)
        - Large static content (~500+ tokens) at start enables automatic caching
        - Benefits: ~50% cost reduction + faster response times for repeated games
        - Caching needs >1024 tokens => increase the cached part.
        
        Cache Strategy:
        - Anthropic: Can use cache_control on game explanation section
        - OpenAI: Automatic caching for 1024+ token prompts with common prefixes
        """
        return {
            
            "base_prompt": """[INST]
#CONTEXT:
You're playing the following instance of the Mafia game. Your ultimate objective is to ensure your team wins, whether through persuasion, deduction, deception, or careful timing.


#MAFIA GAME EXPLANATION

OVERVIEW:
Mafia is a social deduction game of deception, logic, and hidden identities. Each player is secretly assigned to one of two opposing teams, but only the mafia know exactly who is on their side. The town must work together, using discussion, observation, and voting to identify and eliminate the mafia. Meanwhile, the mafia attempt to blend in, spread doubt, and mislead the town while secretly eliminating opponents at night. Every conversation, vote, and pause can carry meaning, and both teams must carefully balance truth and lies to secure victory. Over the course of alternating night and day phases, alliances form, suspicions grow, and the game builds toward a decisive end where only one side will prevail.

TEAMS AND ROLES:
The game features two opposing teams: the informed mafia and the uninformed town.
- Mafia: Mafiosos act together at night to eliminate members of the town. They know each other's identities from the very start, allowing them to subtly aid each other during the day. While they cannot speak privately during the day, they may attempt to coordinate through hints and voting behavior.
- Detectives: Town members with the ability to investigate at night. A detective learns the role of the player they investigate, but only they receive this information. They must decide carefully when and how to reveal their findings to avoid becoming a mafia target.
- Villagers: Town members with no special abilities. Though they lack private information, their power lies in collective reasoning, pattern recognition, and their ability to influence voting outcomes.

GAME STRUCTURE:
The game alternates between night and day phases until one side wins.
1. Night Phase:
- Investigation: Each detective chooses one player to investigate and learns their role privately. Investigations are secret and can only be shared by choice in the public discussion.
- Assassination: One randomly selected mafioso makes the final kill decision for the team. The target must be an active non-mafioso.
- Outcome: Killed players become inactive immediately and take no further part in discussions, votes, or actions.

2. Day Phase:
- Discussion: In each discussion round, all active players speak in a randomized order. Each player may send a single public message or remain silent. Messages are addressed to all and attributed by name. No private communication is allowed.
- Voting: Following discussion, all active players cast a vote to arrest one player. The player with the most votes is arrested. If there is a tie, a tied player is selected randomly.
- Outcome: Arrested players became inactive, and stop playing the game.

WIN CONDITIONS:
- Town: eliminate all mafiosos through coordinated voting during the day.
- Mafia: eliminate all non-mafiosos through a combination of night kills and daytime voting deception.

INFORMATION FLOW:
- All information revealead to players appears in their #YOUR MEMORY: log, in strict chronological order.
- Memory is unique to each player, creating asymmetries in what individuals know.
- Detective investigation results are visible only to that specific detective.
- All mafiosos know which mafioso was assigned the kill decision at night, and they are all informed of the outcome.
- There is no private messaging at any point; all communication occurs in public during the day.
- The total number of each role is known to all players from the start.
- Deaths and arrests are publicly announced.
- Voting results are announced and all players see who voted for whom.
- Voting results are revealed once all active players have cast their vote.


#REQUIRED RESPONSE FORMATS

NIGHT ACTION RESPONSE FORMAT:
- Your night action response MUST START with the name of the player you want to choose.
- Optional reasoning may follow AFTER a line break.
- Example: response = 'player_name \n I choose player_name because...'.
- If you fail to respond in the required format, a random valid choice will be made for you.

DISCUSSION RESPONSE FORMAT:
- Your discussion response MUST START with your message, enclosed in double quotation marks. 
- Optional reasoning may follow AFTER a line break.
- Example: response = '"your message" \n I am saying this because...'.
- If you fail to respond in the required format, a message stating that you remained silent will be shared with everyone.
- Your message will be truncated to a maximum of 200 characters.

VOTING RESPONSE FORMAT: 
- Your voting response MUST START with the name of the player you want to vote for. 
- Optional reasoning may follow AFTER a line break.
- Example: response = 'player_name \n I am voting for player_name because...'.
- If you fail to respond in the required format, a random valid vote will be cast for you.


#GAME PLAYERS AND COMPOSITION
- In the game you're playing there are: {composition}.
- You're {name} and the other players are: {other_players}.


#YOUR MEMORY:
{memory}


{action_specific_content}""",

            # Action-specific suffixes
            "discussion_suffix": """#DISCUSSION ROUND {round_num}/{discussion_rounds}: 
What message do you want to say to everyone?
Your response must start with your message in double quotation marks, followed by optional reasoning:[/INST]
""",

            "voting_suffix": """#VOTING TIME: 
Vote to arrest one player from: {candidates}.
Reply with just a name:[/INST]
""",

            "night_action_suffix": """#NIGHT {round_num}: 
Choose a player to {action} from: {candidates}.
Reply with just a name:[/INST]
"""
        }
    
    def _get_v4_1_templates(self) -> Dict[str, str]:
        """
        Enhanced v4.0 with improved truncation handling.
        
        Key changes from v4.0:
        - Clarified discussion response format to explain truncation vs silence
        - Better instructions for models about token limit behavior
        - Explains that no name prefixing is needed for discussio messages
        - Maintains all caching optimizations from v4.0
        """
        return {
            
            "base_prompt": """[INST]
#CONTEXT:
You're playing the following instance of the Mafia game. Your ultimate objective is to ensure your team wins, whether through persuasion, deduction, deception, or careful timing.


#MAFIA GAME EXPLANATION

OVERVIEW:
Mafia is a social deduction game of deception, logic, and hidden identities. Each player is secretly assigned to one of two opposing teams, but only the mafia know exactly who is on their side. The town must work together, using discussion, observation, and voting to identify and eliminate the mafia. Meanwhile, the mafia attempt to blend in, spread doubt, and mislead the town while secretly eliminating opponents at night. Every conversation, vote, and pause can carry meaning, and both teams must carefully balance truth and lies to secure victory. Over the course of alternating night and day phases, alliances form, suspicions grow, and the game builds toward a decisive end where only one side will prevail.

TEAMS AND ROLES:
The game features two opposing teams: the informed mafia and the uninformed town.
- Mafia: Mafiosos act together at night to eliminate members of the town. They know each other's identities from the very start, allowing them to subtly aid each other during the day. While they cannot speak privately during the day, they may attempt to coordinate through hints and voting behavior.
- Detectives: Town members with the ability to investigate at night. A detective learns the role of the player they investigate, but only they receive this information. They must decide carefully when and how to reveal their findings to avoid becoming a mafia target.
- Villagers: Town members with no special abilities. Though they lack private information, their power lies in collective reasoning, pattern recognition, and their ability to influence voting outcomes.

GAME STRUCTURE:
The game alternates between night and day phases until one side wins.
1. Night Phase:
- Investigation: Each detective chooses one player to investigate and learns their role privately. Investigations are secret and can only be shared by choice in the public discussion.
- Assassination: One randomly selected mafioso makes the final kill decision for the team. The target must be an active non-mafioso.
- Outcome: Killed players become inactive immediately and take no further part in discussions, votes, or actions.

2. Day Phase:
- Discussion: In each discussion round, all active players speak in a randomized order. Each player may send a single public message or remain silent. Messages are addressed to all and attributed by name. No private communication is allowed.
- Voting: Following discussion, all active players cast a vote to arrest one player. The player with the most votes is arrested. If there is a tie, a tied player is selected randomly.
- Outcome: Arrested players became inactive, and stop playing the game.

WIN CONDITIONS:
- Town: eliminate all mafiosos through coordinated voting during the day.
- Mafia: eliminate all non-mafiosos through a combination of night kills and daytime voting deception.

INFORMATION FLOW:
- All information revealead to players appears in their #YOUR MEMORY: log, in strict chronological order.
- Memory is unique to each player, creating asymmetries in what individuals know.
- Detective investigation results are visible only to that specific detective.
- All mafiosos know which mafioso was assigned the kill decision at night, and they are all informed of the outcome.
- There is no private messaging at any point; all communication occurs in public during the day.
- The total number of each role is known to all players from the start.
- Deaths and arrests are publicly announced.
- Voting results are announced and all players see who voted for whom.
- Voting results are revealed once all active players have cast their vote.


#REQUIRED RESPONSE FORMATS

NIGHT ACTION RESPONSE FORMAT:
- Your night action response MUST START with the name of the player you want to choose.
- Optional reasoning may follow AFTER a line break.
- Example: response = 'player_name \n I choose player_name because...'.
- If your response does not start with a player name, a random valid choice will be made for you.

DISCUSSION RESPONSE FORMAT:
- Your discussion response MUST START with the message you want to say to everyone, enclosed in double quotation marks. 
- Optional reasoning may follow AFTER a line break.
- No name prefixing needed, the system already attributes the message to the speaker.
- Example: response = '"I believe that..." \n I am saying this because...'.
- Your message content will be truncated to a maximum of 200 characters.
- If your response does not start with an opening quote, you will be marked as remaining silent.

VOTING RESPONSE FORMAT: 
- Your voting response MUST START with the name of the player you want to vote for. 
- Optional reasoning may follow AFTER a line break.
- Example: response = 'player_name \n I am voting for player_name because...'.
- If your response does not start with a player name, a random valid choice will be made for you.


#GAME PLAYERS AND COMPOSITION
- In the game you're playing there are: {composition}.
- You're {name} and the other players are: {other_players}.


#YOUR MEMORY:
{memory}


{action_specific_content}""",

            # Action-specific suffixes
            "discussion_suffix": """#DISCUSSION ROUND {round_num}/{discussion_rounds}: 
What message do you want to say to everyone?
Your response must start with your message in double quotation marks, followed by optional reasoning:[/INST]
""",

            "voting_suffix": """#VOTING TIME: 
Vote to arrest one player from: {candidates}.
Reply with just a name:[/INST]
""",

            "night_action_suffix": """#NIGHT {round_num}: 
Choose a player to {action} from: {candidates}.
Reply with just a name:[/INST]
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
    
    def parse_discussion_response(self, response: str) -> str:
        """Parse discussion response based on prompt version"""
        # Handle GPT-OSS Harmony format: extract content after final channel
        if '<|channel|>final<|message|>' in response:
            response = response.split('<|channel|>final<|message|>', 1)[1]
            response = response.split('<|')[0].strip()  # Remove trailing harmony tokens
        
        # Handle GPT-OSS reasoning format: extract content after final<|message|>
        if 'final<|message|>' in response:
            response = response.split('final<|message|>')[-1].strip()
        
        if self.version == "v4.0":
            # v4.0 format: "message" \n reasoning...
            match = re.search(r'^\s*"([^"]+)"', response)
            if not match:
                return None
            return match.group(1)
        elif self.version == "v4.1":
            # v4.1 format: "message" \n reasoning... (with truncation handling)
            # Split on newline and take only the first line to avoid capturing reasoning
            first_line = response.split('\n')[0].strip()
            
            # First try to match complete quoted message on first line
            match = re.search(r'^\s*"([^"]+)"', first_line)
            if match:
                return match.group(1)
            
            # If no complete quote found, check for truncated message on first line
            # Look for opening quote followed by content (potentially truncated)
            truncated_match = re.search(r'^\s*"([^"]*)', first_line)
            if truncated_match and len(truncated_match.group(1)) > 0:
                # Found truncated message - return the partial content with proper closing
                return f'{truncated_match.group(1)}...'
            
            # No valid message format found
            return None
        else:
            # v3.0 and earlier format: MESSAGE: "message"
            # First try to match complete quoted message
            match = re.search(r'MESSAGE:\s*"((?:\\.|[^"\\])*)"', response)
            if match:
                return match.group(1)
            
            # If no complete quote found, check for truncated message
            # Look for MESSAGE: followed by opening quote and content (potentially truncated)
            truncated_match = re.search(r'MESSAGE:\s*"([^"]*)', response.strip())
            if truncated_match and len(truncated_match.group(1)) > 0:
                # Found truncated message - return the partial content
                return f'{truncated_match.group(1)}..."'
            
            # No valid message format found
            return None
    
    def parse_voting_response(self, response: str, candidates: List[str]) -> str:
        """Parse voting response based on prompt version"""
        # Handle GPT-OSS Harmony format: extract content after final channel
        if '<|channel|>final<|message|>' in response:
            response = response.split('<|channel|>final<|message|>', 1)[1]
            response = response.split('<|')[0].strip()  # Remove trailing harmony tokens
        
        # Handle GPT-OSS reasoning format: extract content after final<|message|>
        if 'final<|message|>' in response:
            response = response.split('final<|message|>')[-1].strip()
        
        if self.version == "v4.1":
            # v4.1 format: strict parsing - only check first word
            # response.strip().split()[0] gets the first word after removing whitespace
            first_word = response.strip().split()[0] if response.strip().split() else ""
            
            # Check if first word matches any candidate (case insensitive)
            for candidate in candidates:
                if candidate.lower() == first_word.lower():
                    return candidate
            
            # No match found - return None for random vote
            return None
        elif self.version == "v4.0":
            # v4.0 format: player_name \n reasoning... (with fallback)
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
        else:
            # v3.0 and earlier format: VOTE: player_name
            for candidate in candidates:
                vote_pattern = f"VOTE: {candidate}"
                if vote_pattern.upper() in response.upper():
                    return candidate
        
        return None
    
    def parse_night_action_response(self, response: str, candidates: List[str]) -> str:
        """Parse night action response based on prompt version"""
        # Handle GPT-OSS Harmony format: extract content after final channel
        if '<|channel|>final<|message|>' in response:
            response = response.split('<|channel|>final<|message|>', 1)[1]
            response = response.split('<|')[0].strip()  # Remove trailing harmony tokens
        
        # Handle GPT-OSS reasoning format: extract content after final<|message|>
        if 'final<|message|>' in response:
            response = response.split('final<|message|>')[-1].strip()
        
        if self.version == "v4.1":
            # v4.1 format: strict parsing - only check first word
            # response.strip().split()[0] gets the first word after removing whitespace
            first_word = response.strip().split()[0] if response.strip().split() else ""
            
            # Check if first word matches any candidate (case insensitive)
            for candidate in candidates:
                if candidate.lower() == first_word.lower():
                    return candidate
            
            # No match found - return None for random choice
            return None
        elif self.version == "v4.0":
            # v4.0 format: player_name \n reasoning... (with fallback)
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
        else:
            # v3.0 and earlier format: look for name anywhere in response
            for candidate in candidates:
                if candidate.lower() in response.lower():
                    return candidate
        
        return None

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