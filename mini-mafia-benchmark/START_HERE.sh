#!/bin/bash

# Quick Start Script for Mini-Mafia Web Game
# Run this to start the server for your class

cd "$(dirname "$0")/.."

echo "================================================"
echo "    Mini-Mafia Web Game - Quick Start"
echo "================================================"
echo ""

# Check if config exists
if [ ! -f "mini-mafia-benchmark/web_game_config.json" ]; then
    echo "⚠️  No configuration found. Let's set it up first!"
    echo ""
    python3 mini-mafia-benchmark/setup_web_game.py
    echo ""
fi

# Show current config
echo "Current Configuration:"
echo "====================="
python3 -c "
import json
with open('mini-mafia-benchmark/web_game_config.json') as f:
    config = json.load(f)
print(f\"Background: {config['background_name']}\")
print(f\"Detective: {config['detective_model']}\")
print(f\"Villager: {config['villager_model']}\")
"
echo ""

# Ask if they want to change it
read -p "Do you want to change the background? (y/N): " change_config
if [[ $change_config =~ ^[Yy]$ ]]; then
    python3 mini-mafia-benchmark/setup_web_game.py
    echo ""
fi

# Get local IP for convenience
echo "Network Information:"
echo "==================="
echo "Students on the same network can access at:"

if [[ "$OSTYPE" == "darwin"* ]]; then
    LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
    echo "  http://$LOCAL_IP:5001"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    echo "  http://$LOCAL_IP:5001"
else
    echo "  http://<your-ip-address>:5001"
fi

echo ""
echo "For external access (students not on same network):"
echo "  Run in another terminal: ngrok http 5001"
echo "  Then share the HTTPS URL ngrok provides"
echo ""

read -p "Press Enter to start the server..."

echo ""
echo "🚀 Starting Mini-Mafia Web Server..."
echo "================================================"
echo ""

# Start the server
python3 mini-mafia-benchmark/web_game.py
