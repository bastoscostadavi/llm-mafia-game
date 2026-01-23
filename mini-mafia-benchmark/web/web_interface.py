#!/usr/bin/env python3
"""
Web interface for Mini-Mafia using existing mini_mafia.py and agent_interfaces.py
Students play as Mafioso against AI opponents
"""

import sys
import json
import sqlite3
import os
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, jsonify, session, request
from flask_session import Session
import threading
import queue
from contextlib import redirect_stdout

# Setup path
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARK_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add benchmark dir to path for mini_mafia import
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from mini_mafia import create_mini_mafia_game, single_day_play
from src.prompt_utils import load_base_prompt
from src.agent_interfaces import create_agent_interface

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mini-mafia-secret'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = str(SCRIPT_DIR / 'flask_session')
Session(app)

CONFIG_FILE = SCRIPT_DIR / 'web_game_config.json'
DB_FILE = BENCHMARK_DIR / 'database' / 'mini_mafia_human.db'

# Queue system for web-human communication
prompt_queues = {}  # session_id -> queue of prompts waiting for human response
response_queues = {}  # session_id -> queue of human responses
event_queues = {}  # session_id -> queue of game events (AI messages, etc)

# Game results storage (since we can't access Flask session from background thread)
game_results = {}  # session_id -> {game_done, winner, result_message, game_id}

# Game sequence tracking (for 10-game sessions)
game_sequences = {}  # session_id -> {current_game: int, total_games: int, game_ids: list, wins: int}

# Get game explanation from prompt.txt
GAME_EXPLANATION = load_base_prompt().split('#REQUIRED RESPONSE FORMATS')[0].strip()

def load_config():
    with open(CONFIG_FILE) as f:
        return json.load(f)

class WebHuman:
    """Web-compatible human interface - communicates via queues with Flask"""
    def __init__(self, session_id):
        self.session_id = session_id
        self.display_name = "Human Player"
        # Queues should already be created by start_game endpoint

    def generate(self, prompt, max_tokens=50):
        """Send prompt to web frontend and wait for response"""
        # Extract just the relevant part from the prompt for display
        # The prompt has #YOUR MEMORY: section with conversation history
        # We want to show the last few messages before the action prompt

        display_prompt = self._extract_display_prompt(prompt)

        # Put display prompt in queue for web frontend to retrieve
        prompt_queues[self.session_id].put(display_prompt)

        # Wait for response from web frontend
        response = response_queues[self.session_id].get(timeout=300)  # 5 min timeout

        # For discussion: enforce character limit and format for parsing
        if '#DISCUSSION ROUND' in prompt:
            
            # (LLMs need complex parsing, humans just need the quote wrapper)
            return f'"{response}"'

        # For voting: just return the name (no special formatting needed)
        return response

    def _extract_display_prompt(self, full_prompt):
        """Extract the relevant display info from the full prompt"""
        # Find the action-specific section (what we're being asked to do)
        if '#DISCUSSION ROUND' in full_prompt:
            # Extract the discussion round info
            action_part = full_prompt.split('#DISCUSSION ROUND')[1].strip()
            # Get the round info line
            lines = action_part.split('\n')
            round_line = lines[0] if lines else ''
            return f"DISCUSSION ROUND{round_line}\nWhat message do you want to say to everyone?"
        elif '#VOTING TIME:' in full_prompt:
            # Extract voting info
            action_section = full_prompt.split('#VOTING TIME:')[1].strip()
            lines = action_section.split('\n')[:2]
            return 'VOTING TIME:\n' + '\n'.join(lines)
        else:
            # Fallback: try to get last few lines
            lines = full_prompt.strip().split('\n')
            return '\n'.join(lines[-3:]) if len(lines) > 3 else full_prompt

def create_web_mini_mafia_game(config, session_id):
    """Create mini-mafia game with web-human as mafioso, detective, or villager"""
    human_role = config.get('human_role', 'mafioso')

    # Create model configs for all roles
    if human_role == 'detective':
        model_configs = {
            'detective': config['mafioso_model'],  # Temporary, will be replaced
            'villager': config.get('villager_model', config['mafioso_model']),
            'mafioso': config['mafioso_model']
        }
    elif human_role == 'villager':
        model_configs = {
            'detective': config['detective_model'],
            'villager': config['detective_model'],  # Temporary, will be replaced
            'mafioso': config['mafioso_model']
        }
    else:  # mafioso
        model_configs = {
            'detective': config.get('detective_model', config.get('mafioso_model', {})),
            'villager': config.get('villager_model', config.get('detective_model', {})),
            'mafioso': config.get('detective_model', config.get('mafioso_model', {}))  # Temporary, will be replaced
        }

    game = create_mini_mafia_game(model_configs, debug_prompts=False)

    # Get game information for initial messages
    victim = next(a for a in game.state.agents if not a.alive)
    # For villager, make sure we get the ALIVE one (since one villager dies at night)
    if human_role == 'villager':
        human_agent = next(a for a in game.state.agents if a.role == human_role and a.alive)
    else:
        human_agent = next(a for a in game.state.agents if a.role == human_role)
    all_names = [a.name for a in game.state.agents]
    other_names = [n for n in all_names if n != human_agent.name]

    # Send initial game setup messages
    if session_id in event_queues:
        event_queues[session_id].put({
            'type': 'system',
            'message': f"You're {human_agent.name} and you're playing Mafia with: {', '.join(other_names)}."
        })
        event_queues[session_id].put({
            'type': 'system',
            'message': f"You're {human_agent.name}, the {human_role}."
        })
        event_queues[session_id].put({
            'type': 'system',
            'message': "Night 1 begins."
        })
        if human_role == 'mafioso':
            event_queues[session_id].put({
                'type': 'system',
                'message': f"You killed {victim.name}."
            })
        elif human_role == 'detective':
            # Detective investigates and learns who the mafioso is
            mafioso = next(a for a in game.state.agents if a.role == 'mafioso')
            event_queues[session_id].put({
                'type': 'system',
                'message': f"You investigated {mafioso.name} and discovered they are the MAFIOSO!"
            })
        event_queues[session_id].put({
            'type': 'system',
            'message': f"{victim.name} was found dead."
        })
        event_queues[session_id].put({
            'type': 'system',
            'message': "Day 1 begins."
        })

    # Replace the human role's LLM with WebHuman
    human_agent.llm = WebHuman(session_id)
    human_agent.model_config = {'type': 'web_human'}

    # Wrap AI agents' message method to capture their responses
    for agent in game.state.agents:
        if agent.role != human_role:
            original_message = agent.message
            def wrapped_message(active_players, round_num, all_players, discussion_rounds, game_state, agent=agent, orig=original_message):
                result = orig(active_players, round_num, all_players, discussion_rounds, game_state)
                # Send AI message to event queue
                if session_id in event_queues:
                    # Clean up the message format
                    clean_msg = result.strip('"\'')
                    event_queues[session_id].put({'type': 'ai_message', 'speaker': agent.name, 'message': clean_msg})
                return result
            agent.message = wrapped_message

    return game

def init_database():
    DB_FILE.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            game_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            winner TEXT,
            mafioso_name TEXT,
            detective_name TEXT,
            villager_name TEXT,
            arrested_name TEXT,
            background_name TEXT,
            human_role TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_actions (
            action_id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id INTEGER,
            step INTEGER,
            action_type TEXT,
            actor TEXT,
            raw_response TEXT,
            parsed_result TEXT,
            FOREIGN KEY (game_id) REFERENCES games (game_id)
        )
    ''')
    conn.commit()
    conn.close()

def save_game(game_state, background_name, human_role='mafioso'):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    mafioso = next(a for a in game_state.agents if a.role == 'mafioso')
    detective = next(a for a in game_state.agents if a.role == 'detective')
    villager = next(a for a in game_state.agents if a.role == 'villager' and a.alive)
    arrested = next((a for a in game_state.agents if a.imprisoned), None)

    winner = 'GOOD' if arrested and arrested.role == 'mafioso' else 'EVIL'

    cursor.execute('''
        INSERT INTO games (timestamp, winner, mafioso_name, detective_name, villager_name,
                          arrested_name, background_name, human_role)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), winner, mafioso.name, detective.name,
          villager.name, arrested.name if arrested else None, background_name, human_role))

    game_id = cursor.lastrowid

    for action in game_state.game_sequence:
        cursor.execute('''
            INSERT INTO game_actions (game_id, step, action_type, actor, raw_response, parsed_result)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (game_id, action['step'], action['action'], action['actor'],
              action['raw_response'], str(action['parsed_result'])))

    conn.commit()
    conn.close()
    return game_id

# HTML Template - Event feed style
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Mini-Mafia</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 20px auto; padding: 20px; background: #fafafa; }
        h1 { color: #333; text-align: center; font-weight: normal; font-size: 28px; margin-bottom: 30px; }
        .game-area { background: white; padding: 25px; border: 1px solid #ddd; }
        .event-feed { min-height: 400px; max-height: 600px; overflow-y: auto; margin-bottom: 20px; border: 1px solid #eee; padding: 15px; }
        .event { padding: 10px; margin: 6px 0; border-left: 3px solid #ddd; }
        .event.system { color: #666; font-style: italic; }
        .event.ai { background: #e5e5ea; color: #000; }
        .event.you { background: #dcf8c6; font-weight: bold; border-left-color: #4caf50; }
        .event.prompt { background: #f5f5f5; border-left-color: #999; }
        .input-area { margin-top: 20px; }
        textarea { width: 100%; padding: 10px; font-size: 14px; border: 1px solid #ddd; font-family: Arial, sans-serif; box-sizing: border-box; }
        button { padding: 10px 25px; font-size: 14px; background: #333; color: white; border: none; cursor: pointer; margin: 10px 5px 0 0; }
        button:hover { background: #555; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .result { text-align: center; padding: 30px; font-size: 20px; font-weight: normal; }
        .result.win { color: #2e7d32; }
        .result.lose { color: #c62828; }
        .progress-header { text-align: center; color: #666; font-size: 16px; margin: -10px 0 20px 0; }
    </style>
</head>
<body>
    <h1>Mini-Mafia Game</h1>
    <div class="progress-header" id="progress-header">10-Game Sequence</div>

    <div class="game-area">
        <div class="event-feed" id="event-feed"></div>

        <div class="input-area" id="input-area" style="display:none;">
            <textarea id="response-input" rows="3" placeholder="" ></textarea>
            <button onclick="submitResponse()">Submit Response</button>
        </div>

        <div id="result-area" style="display:none;"></div>
    </div>

    <script>
        let gameThread = null;
        let isWaitingForInput = false;

        function addEvent(text, type='system') {
            const feed = document.getElementById('event-feed');
            const event = document.createElement('div');
            event.className = 'event ' + type;
            event.textContent = text;
            feed.appendChild(event);
            feed.scrollTop = feed.scrollHeight;
        }

        async function startGame() {
            const res = await fetch('/api/start', {method: 'POST'});
            const data = await res.json();

            if (data.success) {
                gameThread = data.thread_id;
                // Update progress header
                document.getElementById('progress-header').textContent =
                    `Game ${data.current_game} of ${data.total_games}`;
                // Show game progress
                addEvent(`Starting Game ${data.current_game} of ${data.total_games}`, 'system');
                pollForEvents();
            }
        }

        async function pollForEvents() {
            while (true) {
                const res = await fetch('/api/get_prompt');
                const data = await res.json();

                if (data.has_event && data.event_type === 'system') {
                    // Show system message
                    addEvent(data.message, 'system');
                    // Continue polling without delay
                    continue;
                } else if (data.has_event && data.event_type === 'ai_message') {
                    // Show AI message
                    addEvent(data.speaker + ': "' + data.message + '"', 'ai');
                    // Continue polling without delay
                    continue;
                } else if (data.has_prompt) {
                    // Use prompt as textarea placeholder instead of showing it
                    const textarea = document.getElementById('response-input');
                    textarea.placeholder = data.prompt;
                    textarea.value = '';

                    // Show input area
                    document.getElementById('input-area').style.display = 'block';
                    textarea.focus();
                    isWaitingForInput = true;
                    break;
                } else if (data.game_done) {
                    // Game is over, show results
                    showResults(data);
                    break;
                }

                await new Promise(resolve => setTimeout(resolve, 200));
            }
        }

        async function submitResponse() {
            if (!isWaitingForInput) return;

            const response = document.getElementById('response-input').value.trim();
            if (!response) return;

            // Show user's response as an event
            addEvent('You: ' + response, 'you');

            // Hide input area
            document.getElementById('input-area').style.display = 'none';
            isWaitingForInput = false;

            await fetch('/api/submit_response', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({response})
            });

            // Continue polling for next events
            pollForEvents();
        }

        function showResults(data) {
            const resultDiv = document.getElementById('result-area');

            // Determine if human won based on role
            // Mafioso wins when EVIL wins, Detective/Villager win when GOOD wins
            const humanRole = data.human_role || 'mafioso';  // default to mafioso for backward compat
            const youWon = (humanRole === 'mafioso' && data.winner === 'EVIL') ||
                          ((humanRole === 'detective' || humanRole === 'villager') && data.winner === 'GOOD');

            if (data.continue_sequence) {
                // Game finished, but more to go
                addEvent('Game Over!', 'system');
                addEvent(data.result_message, youWon ? 'you' : 'ai');
                addEvent(`Progress: ${data.wins_so_far} wins out of ${data.current_game} games`, 'system');
                addEvent(`Starting next game in 3 seconds...`, 'system');

                // Auto-start next game after 3 seconds
                setTimeout(() => {
                    startGame();
                }, 3000);

            } else if (data.sequence_complete) {
                // All 10 games complete - show final summary
                addEvent('🎉 All games complete!', 'system');

                resultDiv.className = 'result';
                resultDiv.innerHTML =
                    `<div style="font-size: 24px; margin-bottom: 20px;">Sequence Complete!</div>` +
                    `<div style="font-size: 36px; font-weight: bold; color: #2ECC71; margin: 20px 0;">${data.total_wins}/${data.total_games} Wins</div>` +
                    `<div style="font-size: 20px; margin-bottom: 20px;">Win Rate: ${data.win_rate.toFixed(1)}%</div>` +
                    `<p style="font-size: 12px; color: #999; margin-top: 20px;">Game IDs: ${data.game_ids.join(', ')}</p>`;
                resultDiv.style.display = 'block';

            } else {
                // Single game (shouldn't happen in sequence mode, but handle it)
                addEvent('Game Over!', 'system');
                addEvent(data.result_message, youWon ? 'you' : 'ai');

                resultDiv.className = 'result ' + (youWon ? 'win' : 'lose');
                resultDiv.innerHTML =
                    `<div>${youWon ? 'YOU WON' : 'YOU LOST'}</div>` +
                    `<p style="font-size: 12px; color: #999; margin-top: 10px;">Game ID: ${data.game_id}</p>`;
                resultDiv.style.display = 'block';
            }
        }

        // Character counter
        function updateCharCount() {
            const input = document.getElementById('response-input');
            const counter = document.getElementById('char-count');
            if (input && counter) {
                counter.textContent = input.value.length + '/200';
            }
        }

        // Enter to submit
        document.addEventListener('DOMContentLoaded', () => {
            const input = document.getElementById('response-input');
            if (input) {
                input.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        submitResponse();
                    }
                });
                input.addEventListener('input', updateCharCount);
            }
        });

        window.onload = () => setTimeout(startGame, 500);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/start', methods=['POST'])
def start_game():
    session_id = session.sid if hasattr(session, 'sid') else id(session)
    config = load_config()

    # Initialize game sequence if first game
    if session_id not in game_sequences:
        game_sequences[session_id] = {
            'current_game': 1,
            'total_games': 100,
            'game_ids': [],
            'wins': 0
        }

    # Get current game number
    current_game = game_sequences[session_id]['current_game']
    total_games = game_sequences[session_id]['total_games']

    # Clear any previous game results
    game_results.pop(session_id, None)

    # Initialize queues before creating game
    prompt_queues[session_id] = queue.Queue()
    response_queues[session_id] = queue.Queue()
    event_queues[session_id] = queue.Queue()

    # Start game in background thread
    def run_game():
        try:
            # Suppress terminal output during web game
            with open(os.devnull, 'w') as devnull:
                with redirect_stdout(devnull):
                    game = create_web_mini_mafia_game(config, session_id)
                    game.play()  # This runs single_day_play which handles the full game

            # Game finished, save results (outside stdout redirect)
            human_role = config.get('human_role', 'mafioso')
            game_id = save_game(game.state, config['background_name'], human_role)

            # Determine results
            arrested = next((a for a in game.state.agents if a.imprisoned), None)
            winner = 'GOOD' if arrested and arrested.role == 'mafioso' else 'EVIL'
            result_message = "Town wins! Mafioso arrested!" if winner == 'GOOD' else "Mafia wins! Town lost!"

            # Update sequence tracking
            human_role = config.get('human_role', 'mafioso')
            game_sequences[session_id]['game_ids'].append(game_id)
            # Human wins if: (mafioso and EVIL wins) OR (detective/villager and GOOD wins)
            if (human_role == 'mafioso' and winner == 'EVIL') or (human_role in ['detective', 'villager'] and winner == 'GOOD'):
                game_sequences[session_id]['wins'] += 1

            # Check if we should continue to next game
            current = game_sequences[session_id]['current_game']
            total = game_sequences[session_id]['total_games']

            if current < total:
                # More games to play
                game_sequences[session_id]['current_game'] += 1
                game_results[session_id] = {
                    'game_done': True,
                    'continue_sequence': True,
                    'current_game': current,
                    'total_games': total,
                    'winner': winner,
                    'result_message': result_message,
                    'game_id': game_id,
                    'wins_so_far': game_sequences[session_id]['wins'],
                    'human_role': human_role
                }
            else:
                # Sequence complete
                game_results[session_id] = {
                    'game_done': True,
                    'sequence_complete': True,
                    'total_games': total,
                    'total_wins': game_sequences[session_id]['wins'],
                    'game_ids': game_sequences[session_id]['game_ids'],
                    'win_rate': (game_sequences[session_id]['wins'] / total) * 100,
                    'human_role': human_role
                }

        except Exception as e:
            print(f"Game error: {e}")
            import traceback
            traceback.print_exc()
            game_results[session_id] = {
                'game_done': True,
                'error': str(e)
            }

    thread = threading.Thread(target=run_game, daemon=True)
    thread.start()

    session['session_id'] = session_id
    session['game_done'] = False

    return jsonify({
        'success': True,
        'thread_id': session_id,
        'current_game': current_game,
        'total_games': total_games
    })

@app.route('/api/get_prompt', methods=['GET'])
def get_prompt():
    session_id = session.get('session_id')

    # Check if game is done (from global dict)
    if session_id in game_results:
        result = game_results[session_id]
        if result.get('game_done'):
            return jsonify({
                'game_done': True,
                'winner': result.get('winner'),
                'result_message': result.get('result_message'),
                'game_id': result.get('game_id'),
                'error': result.get('error'),
                'human_role': result.get('human_role'),
                'continue_sequence': result.get('continue_sequence'),
                'sequence_complete': result.get('sequence_complete'),
                'current_game': result.get('current_game'),
                'total_games': result.get('total_games'),
                'wins_so_far': result.get('wins_so_far'),
                'total_wins': result.get('total_wins'),
                'game_ids': result.get('game_ids'),
                'win_rate': result.get('win_rate')
            })

    # Check if there's an event waiting (AI message, system message, etc)
    if session_id in event_queues and not event_queues[session_id].empty():
        event = event_queues[session_id].get()
        if event['type'] == 'ai_message':
            return jsonify({
                'has_event': True,
                'event_type': 'ai_message',
                'speaker': event['speaker'],
                'message': event['message']
            })
        elif event['type'] == 'system':
            return jsonify({
                'has_event': True,
                'event_type': 'system',
                'message': event['message']
            })

    # Check if there's a prompt waiting (human's turn)
    if session_id in prompt_queues and not prompt_queues[session_id].empty():
        prompt = prompt_queues[session_id].get()
        return jsonify({'has_prompt': True, 'prompt': prompt})

    return jsonify({'has_prompt': False, 'has_event': False})

@app.route('/api/submit_response', methods=['POST'])
def submit_response():
    session_id = session.get('session_id')
    data = request.json
    response = data['response']

    # Put response in queue for game thread to retrieve
    if session_id in response_queues:
        response_queues[session_id].put(response)

    return jsonify({'success': True})

def kill_process_on_port(port):
    """Kill any process using the specified port"""
    import subprocess
    import time
    try:
        # Find process using the port
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    print(f"⚠️  Killing existing process on port {port} (PID: {pid})")
                    subprocess.run(['kill', '-9', pid], capture_output=True)

            # Wait for port to be released
            print(f"⏳ Waiting for port {port} to be released...")
            time.sleep(1)
            return True
        return False
    except Exception as e:
        print(f"⚠️  Could not check port {port}: {e}")
        return False

if __name__ == '__main__':
    print("Initializing Mini-Mafia Web Interface...")
    init_database()

    config = load_config()
    human_role = config.get('human_role', 'mafioso')
    print(f"\nCurrent background: {config['background_name']}")
    print(f"Human role: {human_role}")
    if human_role == 'detective':
        print(f"  Mafioso: {config['mafioso_model']}")
        print(f"  Villager: {config.get('villager_model', config['mafioso_model'])}")
    elif human_role == 'villager':
        print(f"  Mafioso: {config['mafioso_model']}")
        print(f"  Detective: {config['detective_model']}")
    else:  # mafioso
        print(f"  Detective: {config.get('detective_model', config.get('mafioso_model', 'N/A'))}")
        print(f"  Villager: {config.get('villager_model', config.get('detective_model', 'N/A'))}")

    # Kill any existing process on port 5002
    kill_process_on_port(5002)

    print("\nStarting server at: http://localhost:5002")
    print("For external access: ngrok http 5002\n")

    app.run(debug=False, host='0.0.0.0', port=5002, threaded=True)
