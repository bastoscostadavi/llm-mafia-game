"""
Database utilities for Mini-Mafia game data storage.
"""

import sqlite3
import json
from pathlib import Path

class MiniMafiaDB:
    """Database utility for storing Mini-Mafia game data."""
    
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = Path(__file__).parent / "mini_mafia.db"
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def insert_batch(self, batch_id, timestamp, model_configs):
        """Insert a batch record - simplified for mini-mafia."""
        # For mini-mafia, we don't have a separate batches table
        # This is just for compatibility with the existing code
        pass
    
    def get_or_create_player(self, player_type, model_name, model_provider, temperature):
        """Get or create a player record and return the player_id."""
        cursor = self.conn.cursor()
        
        # Try to find existing player
        cursor.execute("""
            SELECT player_id FROM players 
            WHERE player_type = ? AND model_name = ? AND model_provider = ? AND temperature = ?
        """, (player_type, model_name, model_provider, temperature))
        
        result = cursor.fetchone()
        if result:
            return result[0]
        
        # Create new player
        cursor.execute("""
            INSERT INTO players (player_type, model_name, model_provider, temperature)
            VALUES (?, ?, ?, ?)
        """, (player_type, model_name, model_provider, temperature))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_game(self, game_id, batch_id, game_num, timestamp, winner):
        """Insert a game record."""
        cursor = self.conn.cursor()
        
        # Map winner format
        winner_mapped = "town" if winner == "good" else "mafia" if winner == "evil" else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO games (game_id, timestamp, winner)
            VALUES (?, ?, ?)
        """, (game_id, timestamp, winner_mapped))
        
        self.conn.commit()
    
    def insert_game_player(self, game_id, player_id, character_name, role, final_status):
        """Insert a game player assignment."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO game_players (game_id, player_id, character_name, role, final_status)
            VALUES (?, ?, ?, ?, ?)
        """, (game_id, player_id, character_name, role, final_status))
        
        self.conn.commit()
    
    def insert_event(self, game_id, sequence_number, event_type, actor_character, 
                    target_character=None, content=None, round_number=None, metadata=None):
        """Insert an event/action record."""
        cursor = self.conn.cursor()
        
        # Map event types to actions
        action_map = {
            'discussion_message': 'discuss',
            'discussion_silent': 'discuss',
            'vote_cast': 'vote',
            'kill_action': 'kill',
            'investigate_action': 'investigate',
            'game_start': 'discuss',  # Map to discuss as a placeholder
            'game_end': 'discuss'     # Map to discuss as a placeholder
        }
        
        action = action_map.get(event_type, event_type)
        if action not in ['discuss', 'vote', 'kill', 'investigate']:
            return  # Skip unsupported event types
        
        # Parse metadata if it's a JSON string
        raw_response = None
        parsed_result = content or target_character
        
        if metadata:
            try:
                metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                raw_response = metadata_dict.get('raw_response')
            except:
                pass
        
        cursor.execute("""
            INSERT INTO game_sequence (game_id, step, action, actor, raw_response, parsed_result)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (game_id, sequence_number, action, actor_character, raw_response, parsed_result))
        
        self.conn.commit()