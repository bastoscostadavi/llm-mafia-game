#!/usr/bin/env python3
"""
Simple web interface for Mini-Mafia where humans play as Mafioso
Uses the existing mini_mafia.py game structure
"""

import sys
import json
import random
import sqlite3
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, session
from flask_session import Session

# Setup path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.main import create_game, GameState
from src.agents import MafiaAgent
from src.agent_interfaces import create_agent_interface
from src.prompt_utils import load_base_prompt

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mini-mafia-secret'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

CONFIG_FILE = SCRIPT_DIR / 'web_game_config.json'
DB_FILE = SCRIPT_DIR / 'database' / 'mini_mafia_human.db'

# Load game explanation from prompt.txt
GAME_EXPLANATION = load_base_prompt().split('#REQUIRED RESPONSE FORMATS')[0].strip()

def load_config():
    with open(CONFIG_FILE) as f:
        return json.load(f)

def init_database():
    """Initialize database"""
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
            background_name TEXT
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

def create_human_mini_mafia_game(config):
    """Create mini-mafia game with human as mafioso"""

    # Randomly assign roles to names
    names = ['Alice', 'Bob', 'Charlie', 'Diana']
    roles = ['mafioso', 'detective', 'villager', 'villager']
    random.shuffle(roles)

    # Find which name got mafioso
    mafioso_index = roles.index('mafioso')
    mafioso_name = names[mafioso_index]

    # Create players (mafioso gets None for LLM - will be human-controlled)
    players = []
    for i, name in enumerate(names):
        role = roles[i]
        if role == 'mafioso':
            # Human player
            players.append({'name': name, 'role': role, 'llm': None})
        else:
            # AI player
            model_config = config['detective_model'] if role == 'detective' else config['villager_model']
            players.append({'name': name, 'role': role, 'llm': model_config})

    # Create agents
    agents = []
    for player in players:
        if player['role'] == 'mafioso':
            agent = MafiaAgent(player['name'], 'mafioso', None, False, None)
        else:
            llm = create_agent_interface(player['llm'])
            agent = MafiaAgent(player['name'], player['role'], llm, False, player['llm'])
        agents.append(agent)

    # Initialize memories
    for agent in agents:
        agent.remember(f"You're {agent.name}, the {agent.role}.")

    # Create game state
    state = GameState(agents, discussion_rounds=2)

    # Night phase setup (same as mini_mafia.py)
    villagers = [a for a in agents if a.role == "villager"]
    victim = random.choice(villagers)
    victim.alive = False

    detective = next(a for a in agents if a.role == "detective")
    mafioso = next(a for a in agents if a.role == "mafioso")

    # Night begins
    alive_agents = [a for a in agents if a.alive]
    for agent in alive_agents:
        agent.remember(f"Night 1 begins.")

    # Log night actions
    state.log_action('kill', mafioso.name, None, victim.name)
    state.log_action('investigate', detective.name, None, mafioso.name)

    # Mafioso knows who they killed
    mafioso.remember(f"You killed {victim.name}.")

    # Everyone discovers the death
    for agent in alive_agents:
        agent.remember(f"{victim.name} was found dead.")

    # Detective investigates the mafioso
    detective.remember(f"You investigated {mafioso.name} and discovered that they are the mafioso.")

    state.round = 1

    return state

def save_game(game_state, background_name):
    """Save completed game to database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    mafioso = next(a for a in game_state.agents if a.role == 'mafioso')
    detective = next(a for a in game_state.agents if a.role == 'detective')
    villager = next(a for a in game_state.agents if a.role == 'villager' and a.alive)
    arrested = next((a for a in game_state.agents if a.imprisoned), None)

    winner = 'GOOD' if arrested and arrested.role == 'mafioso' else 'EVIL'

    cursor.execute('''
        INSERT INTO games (timestamp, winner, mafioso_name, detective_name, villager_name,
                          arrested_name, background_name)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        winner,
        mafioso.name,
        detective.name,
        villager.name,
        arrested.name if arrested else None,
        background_name
    ))

    game_id = cursor.lastrowid

    # Save actions
    for action in game_state.game_sequence:
        cursor.execute('''
            INSERT INTO game_actions (game_id, step, action_type, actor, raw_response, parsed_result)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (game_id, action['step'], action['action'], action['actor'],
              action['raw_response'], str(action['parsed_result'])))

    conn.commit()
    conn.close()
    return game_id

# Simple single-page app HTML
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Mini-Mafia</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial; max-width: 800px; margin: 20px auto; padding: 20px; }
        .rules { background: #f5f5f5; padding: 20px; border-radius: 10px; white-space: pre-wrap; font-size: 14px; line-height: 1.6; max-height: 400px; overflow-y: auto; }
        .status { background: #e3f2fd; padding: 15px; border-radius: 10px; margin: 20px 0; }
        .chat { border: 1px solid #ddd; padding: 15px; min-height: 300px; max-height: 400px; overflow-y: auto; margin: 20px 0; border-radius: 10px; }
        .msg { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .msg.system { background: #f0f0f0; font-style: italic; }
        .msg.ai { background: #e8f5e9; }
        .msg.you { background: #fff3e0; font-weight: bold; }
        .input-area { margin: 20px 0; }
        textarea { width: 100%; padding: 10px; font-size: 14px; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; margin: 5px; }
        .vote-btn { padding: 15px 30px; margin: 10px; font-size: 16px; }
        h1 { color: #1976d2; }
        h2 { color: #424242; }
        .result { text-align: center; padding: 30px; font-size: 20px; }
        .result.win { color: #2e7d32; }
        .result.lose { color: #c62828; }
    </style>
</head>
<body>
    <h1>🎭 Mini-Mafia</h1>

    <div id="rules" class="rules">{{ game_explanation }}</div>

    <div class="status" id="status"></div>
    <div class="chat" id="chat"></div>

    <div class="input-area" id="input-area" style="display:none;">
        <textarea id="message-input" rows="3" placeholder="Type your message..."></textarea>
        <button onclick="sendMessage()">Send Message</button>
    </div>

    <div id="voting-area" style="display:none;">
        <h3>Vote to Arrest:</h3>
        <div id="vote-buttons"></div>
    </div>

    <div id="result-area" style="display:none;"></div>

    <script>
        let gameStarted = false;

        function addMsg(text, type='system') {
            const chat = document.getElementById('chat');
            const msg = document.createElement('div');
            msg.className = 'msg ' + type;
            msg.textContent = text;
            chat.appendChild(msg);
            chat.scrollTop = chat.scrollHeight;
        }

        async function startGame() {
            if (gameStarted) return;
            gameStarted = true;

            addMsg('Initializing game...');
            const res = await fetch('/api/start', {method: 'POST'});
            const data = await res.json();

            document.getElementById('status').innerHTML =
                `<strong>You are: ${data.your_name} (${data.your_role})</strong><br>` +
                `Alive: ${data.alive.join(', ')}<br>` +
                `Dead: ${data.dead}`;

            addMsg('Night 1 completed. ' + data.dead + ' was killed.');
            addMsg('Day 1 begins. Discussion starting...');

            setTimeout(getNextMessage, 1000);
        }

        async function getNextMessage() {
            const res = await fetch('/api/next');
            const data = await res.json();

            if (data.done) {
                startVoting(data.alive);
            } else if (data.your_turn) {
                addMsg('Your turn to speak!', 'system');
                document.getElementById('input-area').style.display = 'block';
                document.getElementById('message-input').focus();
            } else {
                addMsg(data.speaker + ': ' + data.message, 'ai');
                setTimeout(getNextMessage, 500);
            }
        }

        async function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            if (!message) return;

            addMsg('You: ' + message, 'you');
            document.getElementById('input-area').style.display = 'none';
            input.value = '';

            await fetch('/api/message', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message})
            });

            setTimeout(getNextMessage, 500);
        }

        function startVoting(alive) {
            addMsg('Discussion complete. Time to vote!', 'system');
            document.getElementById('voting-area').style.display = 'block';

            const buttons = document.getElementById('vote-buttons');
            alive.forEach(name => {
                const btn = document.createElement('button');
                btn.className = 'vote-btn';
                btn.textContent = 'Arrest ' + name;
                btn.onclick = () => vote(name);
                buttons.appendChild(btn);
            });
        }

        async function vote(target) {
            addMsg('You voted to arrest ' + target, 'you');
            document.getElementById('voting-area').style.display = 'none';
            addMsg('Waiting for other votes...', 'system');

            const res = await fetch('/api/vote', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({vote: target})
            });
            const data = await res.json();

            // Show voting results
            addMsg('Voting results:', 'system');
            Object.entries(data.votes).forEach(([voter, voted]) => {
                addMsg(`  ${voter} → ${voted}`, 'system');
            });

            addMsg(`${data.arrested} was arrested! (${data.arrested_role})`, 'system');

            // Show result
            const resultDiv = document.getElementById('result-area');
            const youWon = data.winner === 'EVIL';
            resultDiv.className = 'result ' + (youWon ? 'win' : 'lose');
            resultDiv.innerHTML =
                `<h2>${youWon ? '🎉 YOU WON!' : '😞 YOU LOST'}</h2>` +
                `<p>${data.result_message}</p>` +
                `<p>Game saved (ID: ${data.game_id})</p>`;
            resultDiv.style.display = 'block';
        }

        // Start game on load
        window.onload = () => {
            setTimeout(startGame, 1000);
        };

        // Enter to send message
        document.addEventListener('DOMContentLoaded', () => {
            const input = document.getElementById('message-input');
            if (input) {
                input.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                    }
                });
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, game_explanation=GAME_EXPLANATION)

@app.route('/api/start', methods=['POST'])
def start_game():
    config = load_config()
    game_state = create_human_mini_mafia_game(config)

    # Store in session
    session['game_state'] = serialize_game_state(game_state)
    session['discussion_history'] = []
    session['background_name'] = config['background_name']

    mafioso = next(a for a in game_state.agents if a.role == 'mafioso')
    alive = [a.name for a in game_state.agents if a.alive]
    dead = next(a.name for a in game_state.agents if not a.alive)

    return jsonify({
        'your_name': mafioso.name,
        'your_role': 'Mafioso',
        'alive': alive,
        'dead': dead
    })

@app.route('/api/next', methods=['GET'])
def next_message():
    game_state = deserialize_game_state(session['game_state'])
    discussion_history = session['discussion_history']

    active = [a for a in game_state.agents if a.alive and not a.imprisoned]
    ai_players = [a for a in active if a.role != 'mafioso']

    # Pattern: AI speaks, Human speaks, AI speaks, Human speaks (2 rounds)
    total_msgs = len(discussion_history)

    if total_msgs >= 8:  # 2 rounds * 4 messages (2 AI + 2 Human) = too many, should be 6 total (2 AI, 1 Human per round)
        return jsonify({'done': True})

    # Simpler: AI, Human, AI, Human, AI, Human (3 turns each over 2 rounds)
    if total_msgs % 2 == 0:  # AI's turn
        if total_msgs // 2 >= len(ai_players) * 2:
            return jsonify({'done': True})

        ai_index = (total_msgs // 2) % len(ai_players)
        ai_agent = ai_players[ai_index]

        active_names = [a.name for a in active]
        all_names = [a.name for a in game_state.agents]
        round_num = (total_msgs // 4) + 1

        message = ai_agent.message(active_names, round_num, all_names, 2, game_state)

        # Update memories
        for a in active:
            if a != ai_agent:
                a.remember(f'{ai_agent.name}: {message}')
            else:
                a.remember(f'You: {message}')

        discussion_history.append({'speaker': ai_agent.name, 'message': message})
        session['game_state'] = serialize_game_state(game_state)
        session['discussion_history'] = discussion_history

        return jsonify({'done': False, 'your_turn': False, 'speaker': ai_agent.name, 'message': message})
    else:  # Human's turn
        return jsonify({'done': False, 'your_turn': True})

@app.route('/api/message', methods=['POST'])
def human_message():
    data = request.json
    message = data['message']

    game_state = deserialize_game_state(session['game_state'])
    discussion_history = session['discussion_history']

    mafioso = next(a for a in game_state.agents if a.role == 'mafioso')
    active = [a for a in game_state.agents if a.alive and not a.imprisoned]

    # Update memories
    for a in active:
        if a.role != 'mafioso':
            a.remember(f'{mafioso.name}: "{message}"')
        else:
            a.remember(f'You: "{message}"')

    discussion_history.append({'speaker': mafioso.name, 'message': message})
    game_state.log_action('discuss', mafioso.name, message, message)

    session['game_state'] = serialize_game_state(game_state)
    session['discussion_history'] = discussion_history

    return jsonify({'success': True})

@app.route('/api/vote', methods=['POST'])
def vote():
    data = request.json
    human_vote = data['vote']

    game_state = deserialize_game_state(session['game_state'])
    active = [a for a in game_state.agents if a.alive and not a.imprisoned]
    mafioso = next(a for a in active if a.role == 'mafioso')

    votes = {mafioso.name: human_vote}

    # Get AI votes
    for agent in active:
        if agent.role != 'mafioso':
            candidates = [a.name for a in active if a.name != agent.name]
            all_names = [a.name for a in game_state.agents]
            vote_target, _ = agent.vote(candidates, all_names, 2, game_state)
            votes[agent.name] = vote_target

    # Count votes
    from collections import Counter
    vote_counts = Counter(votes.values())
    max_votes = max(vote_counts.values())
    tied = [name for name, count in vote_counts.items() if count == max_votes]
    arrested_name = random.choice(tied) if len(tied) > 1 else tied[0]

    # Arrest
    arrested = next(a for a in game_state.agents if a.name == arrested_name)
    arrested.imprisoned = True

    # Determine winner
    winner = 'GOOD' if arrested.role == 'mafioso' else 'EVIL'
    result_message = "Town wins! Mafioso arrested!" if winner == 'GOOD' else "Mafia wins! Innocent arrested!"

    # Save to database
    game_id = save_game(game_state, session['background_name'])

    return jsonify({
        'votes': votes,
        'arrested': arrested_name,
        'arrested_role': arrested.role,
        'winner': winner,
        'result_message': result_message,
        'game_id': game_id
    })

def serialize_game_state(state):
    return {
        'agents': [{
            'name': a.name, 'role': a.role, 'alive': a.alive,
            'imprisoned': a.imprisoned, 'memory': a.memory,
            'model_config': a.model_config
        } for a in state.agents],
        'round': state.round,
        'discussion_rounds': state.discussion_rounds,
        'game_sequence': state.game_sequence
    }

def deserialize_game_state(data):
    config = load_config()
    agents = []
    for agent_data in data['agents']:
        if agent_data['role'] == 'mafioso':
            agent = MafiaAgent(agent_data['name'], 'mafioso', None, False, None)
        else:
            model_config = agent_data['model_config']
            llm = create_agent_interface(model_config)
            agent = MafiaAgent(agent_data['name'], agent_data['role'], llm, False, model_config)
        agent.alive = agent_data['alive']
        agent.imprisoned = agent_data['imprisoned']
        agent.memory = agent_data['memory']
        agents.append(agent)

    state = GameState(agents, data['discussion_rounds'])
    state.round = data['round']
    state.game_sequence = data['game_sequence']
    return state

if __name__ == '__main__':
    print("Initializing Mini-Mafia Web Interface...")
    init_database()

    config = load_config()
    print(f"\nCurrent background: {config['background_name']}")
    print(f"  Detective: {config['detective_model']}")
    print(f"  Villager: {config['villager_model']}")

    print("\nStarting server...")
    print("Access at: http://localhost:5001")
    print("For external access: ngrok http 5001")

    app.run(debug=True, host='0.0.0.0', port=5001)
