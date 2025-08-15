#!/usr/bin/env python3
"""
Web Mafia Game - Human vs LLMs
Flask + WebSockets interface for the mafia game engine
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import uuid
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

from src.main import create_game
from src.agents import MafiaAgent  
from src.llm_utils import create_llm
from src.config import get_default_prompt_config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mafia-game-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global game storage
active_games: Dict[str, 'WebGame'] = {}

class WebGame:
    """Web wrapper around the existing game engine"""
    
    def __init__(self, game_id: str, human_name: str, human_role: str):
        self.game_id = game_id
        self.human_name = human_name
        self.human_role = human_role
        self.human_sid = None  # Socket ID of human player
        self.game_engine = None
        self.current_phase = "waiting"
        self.game_log = []
        self.created_at = datetime.now()
        
        # Game state
        self.waiting_for_human = False
        self.human_response = None
        self.response_event = threading.Event()
        
        self._setup_game()
    
    def _setup_game(self):
        """Initialize the game with human + 3 LLMs"""
        # Create player configurations
        players = self._create_players()
        
        # Create game engine
        self.game_engine = create_game(
            players=players,
            discussion_rounds=2,
            debug_prompts=False,
            prompt_config=get_default_prompt_config()
        )
        
        self.log_event("game_created", {
            "players": [{"name": p.name, "role": p.role} for p in self.game_engine.state.agents],
            "human_player": self.human_name,
            "human_role": self.human_role
        })
    
    def _create_players(self):
        """Create player list with human + 3 LLMs"""
        # Default LLM config (Mistral for speed)
        llm_config = {
            'type': 'local', 
            'model_path': '/Users/davicosta/Desktop/projects/llm-mafia-game/models/mistral.gguf',
            'n_ctx': 2048
        }
        
        # Role assignments (human gets chosen role, others random)
        all_roles = ["detective", "mafioso", "villager", "villager"]
        roles = all_roles.copy()
        
        # Ensure human gets their chosen role
        roles.remove(self.human_role)
        
        players = [
            # Human player
            {"name": self.human_name, "role": self.human_role, "llm": {"type": "web_human"}},
            # 3 LLM players
            {"name": "Alice", "role": roles[0], "llm": llm_config},
            {"name": "Bob", "role": roles[1], "llm": llm_config},
            {"name": "Charlie", "role": roles[2], "llm": llm_config}
        ]
        
        return players
    
    def log_event(self, event_type: str, data: dict):
        """Log game events for data collection"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        self.game_log.append(entry)
        
        # Emit to client
        socketio.emit('game_log', entry, room=self.game_id)
    
    def start_game(self):
        """Start the game in a separate thread"""
        def game_thread():
            try:
                # Replace human agent with web interface
                self._setup_human_agent()
                
                # Start game
                self.current_phase = "playing"
                self.emit_game_update("Game starting!")
                
                # Run game
                self.game_engine.play()
                
                # Game finished
                self.current_phase = "finished"
                self.emit_game_update("Game finished!")
                self._save_game_to_json()
                
            except Exception as e:
                print(f"Game error: {e}")
                self.emit_game_update(f"Game error: {e}")
        
        threading.Thread(target=game_thread, daemon=True).start()
    
    def _setup_human_agent(self):
        """Replace human agent with web interface"""
        for agent in self.game_engine.state.agents:
            if agent.name == self.human_name:
                # Replace LLM with web interface
                agent.llm = WebHumanInterface(self)
                break
    
    def emit_game_update(self, message: str):
        """Send game update to client"""
        socketio.emit('game_update', {
            'message': message,
            'phase': self.current_phase,
            'timestamp': datetime.now().isoformat()
        }, room=self.game_id)
    
    def wait_for_human_response(self, prompt: str, response_type: str) -> str:
        """Wait for human player response via WebSocket"""
        self.waiting_for_human = True
        self.human_response = None
        self.response_event.clear()
        
        # Send prompt to human
        socketio.emit('human_turn', {
            'prompt': prompt,
            'type': response_type,
            'timestamp': datetime.now().isoformat()
        }, room=self.game_id)
        
        # Wait for response (with timeout)
        if self.response_event.wait(timeout=120):  # 2 minute timeout
            response = self.human_response
            self.waiting_for_human = False
            return response
        else:
            # Timeout - return default
            self.waiting_for_human = False
            return "Timed out"
    
    def submit_human_response(self, response: str):
        """Submit human player response"""
        if self.waiting_for_human:
            self.human_response = response
            self.response_event.set()
            self.log_event("human_response", {"response": response})
    
    def _save_game_to_json(self):
        """Save completed game using same format as batch experiments"""
        save_web_game_to_json(self)

class WebHumanInterface:
    """Human interface for web games"""
    
    def __init__(self, web_game: WebGame):
        self.web_game = web_game
        self.display_name = f"Human Player ({web_game.human_name})"
    
    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """Get response from human via web interface"""
        # Determine response type from prompt content
        if "DISCUSSION" in prompt:
            response_type = "discussion"
        elif "VOTING" in prompt:
            response_type = "voting"  
        elif "NIGHT" in prompt:
            response_type = "night_action"
        else:
            response_type = "general"
        
        return self.web_game.wait_for_human_response(prompt, response_type)

def create_web_games_folder():
    """Create web games folder and return path"""
    base_dir = Path(__file__).parent / "web_games"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def determine_winner(agents):
    """Determine who won the game (same logic as batch experiments)"""
    arrested = next((a for a in agents if a.imprisoned), None)
    if arrested:
        if arrested.role == "mafioso":
            return "good"
        else:  # villager or detective arrested
            return "evil"
    return "unknown"

def save_web_game_to_json(web_game: WebGame):
    """Save web game using same format as batch experiments"""
    web_games_dir = create_web_games_folder()
    
    # Use same minimal format as batch experiments - just memories and essential info
    game_data = {
        "game_id": f"web_{web_game.game_id}",
        "human_player": web_game.human_name,
        "human_role": web_game.human_role,
        "timestamp": web_game.created_at.isoformat(),
        "finished_at": datetime.now().isoformat(),
        "winner": determine_winner(web_game.game_engine.state.agents),
        
        # Same player format as batch experiments
        "players": [
            {
                "name": agent.name,
                "role": agent.role,
                "alive": agent.alive,
                "imprisoned": agent.imprisoned,
                "memory": agent.memory
            }
            for agent in web_game.game_engine.state.agents
        ]
    }
    
    # Save with timestamp and human name for easy identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"web_game_{timestamp}_{web_game.human_name}_{web_game.human_role}.json"
    filepath = web_games_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(game_data, f, indent=2)
    
    print(f"Web game saved to: {filepath}")
    return game_data

# Flask Routes
@app.route('/')
def index():
    """Main game page"""
    return render_template('index.html')

@app.route('/api/create_game', methods=['POST'])
def create_game_endpoint():
    """Create a new game"""
    data = request.json
    human_name = data.get('name', 'Human')
    human_role = data.get('role', 'detective')
    
    # Generate game ID
    game_id = str(uuid.uuid4())
    
    # Create game
    game = WebGame(game_id, human_name, human_role)
    active_games[game_id] = game
    
    return jsonify({
        'game_id': game_id,
        'message': 'Game created successfully!'
    })

# WebSocket Events
@socketio.on('join_game')
def on_join_game(data):
    """Player joins a game"""
    game_id = data['game_id']
    
    if game_id not in active_games:
        emit('error', {'message': 'Game not found'})
        return
    
    game = active_games[game_id]
    game.human_sid = request.sid
    join_room(game_id)
    
    emit('joined_game', {
        'game_id': game_id,
        'human_name': game.human_name,
        'human_role': game.human_role
    })
    
    # Start game
    game.start_game()

@socketio.on('submit_response')
def on_submit_response(data):
    """Human player submits response"""
    game_id = data['game_id']
    response = data['response']
    
    if game_id in active_games:
        active_games[game_id].submit_human_response(response)

@socketio.on('disconnect')
def on_disconnect():
    """Handle player disconnect"""
    print(f"Client disconnected: {request.sid}")

if __name__ == '__main__':
    # Create web games folder
    create_web_games_folder()
    
    # Start server
    print("Starting Mafia Web Game Server...")
    print("Open http://localhost:8080 to play!")
    print("Games will be saved to: web_games/ folder")
    
    socketio.run(app, debug=True, host='0.0.0.0', port=8080, allow_unsafe_werkzeug=True)