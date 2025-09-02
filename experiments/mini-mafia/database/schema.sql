-- Mini-Mafia Game Database Schema

-- Enable foreign key constraints
PRAGMA foreign_keys = ON;

-- Reusable player/model definitions  
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_type TEXT NOT NULL CHECK(player_type IN ('llm', 'human')),
    model_name TEXT, -- 'gpt-4.1-mini', 'claude-sonnet-4', 'human', etc.
    model_provider TEXT, -- 'openai', 'anthropic', 'human', etc.
    temperature REAL, -- model temperature setting
    UNIQUE(player_type, model_name, model_provider, temperature)
);

-- Batch/Session tracking
CREATE TABLE batches (
    batch_id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    model_configs TEXT, -- JSON string of model configurations
    games_planned INTEGER,
    games_completed INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Individual games metadata
CREATE TABLE games (
    game_id TEXT PRIMARY KEY, -- Format: BATCH_ID_NNNN (e.g., 20250821_111911_0001)
    timestamp DATETIME NOT NULL,
    winner TEXT CHECK(winner IN ('town', 'mafia')),
    was_tie BOOLEAN DEFAULT FALSE, -- True if all 3 active players received votes (tie situation)
    batch_id TEXT, -- Links to batches table for experiment organization
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
);

-- Game sequence - stores actions directly from game_sequence
CREATE TABLE game_sequence (
    game_id TEXT NOT NULL,
    step INTEGER NOT NULL, -- From game_sequence step counter
    action TEXT NOT NULL CHECK(action IN (
        'discuss', 'vote', 'kill', 'investigate'
    )),
    actor TEXT NOT NULL, -- Player name who performed the action
    raw_response TEXT, -- Raw LLM response  
    parsed_result TEXT, -- Parsed/cleaned result (target, message, etc.)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, step),
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- Voting information per game
CREATE TABLE votes (
    game_id TEXT NOT NULL,
    character_name TEXT NOT NULL CHECK(character_name IN ('Alice', 'Bob', 'Charlie', 'Diana')),
    role TEXT NOT NULL CHECK(role IN ('detective', 'mafioso', 'villager')),
    voted_for TEXT CHECK(voted_for IN ('Alice', 'Bob', 'Charlie', 'Diana')),
    parsed_successfully BOOLEAN NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, character_name),
    FOREIGN KEY (game_id) REFERENCES games(game_id)
);

-- Player assignments per game (character names and roles)
CREATE TABLE game_players (
    game_id TEXT NOT NULL,
    player_id INTEGER, -- NULL for killed players who never actually played
    character_name TEXT NOT NULL CHECK(character_name IN ('Alice', 'Bob', 'Charlie', 'Diana')),
    role TEXT NOT NULL CHECK(role IN ('detective', 'mafioso', 'villager')),
    final_status TEXT NOT NULL CHECK(final_status IN ('alive', 'killed', 'arrested')),
    PRIMARY KEY (game_id, character_name),
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
    -- Note: Same player (model) can play multiple characters in a game (e.g., two villagers)
    -- Note: Killed players have NULL player_id as they never interacted with AI models
);

-- Performance indexes for common query patterns
CREATE INDEX idx_batches_timestamp ON batches(timestamp);
CREATE INDEX idx_games_winner ON games(winner);
CREATE INDEX idx_games_timestamp ON games(timestamp);
CREATE INDEX idx_games_was_tie ON games(was_tie);
CREATE INDEX idx_games_batch ON games(batch_id);
CREATE INDEX idx_game_sequence_game ON game_sequence(game_id);
CREATE INDEX idx_game_sequence_action ON game_sequence(action);
CREATE INDEX idx_game_sequence_actor ON game_sequence(actor);
CREATE INDEX idx_votes_game ON votes(game_id);
CREATE INDEX idx_votes_role ON votes(role);
CREATE INDEX idx_votes_target ON votes(voted_for);
CREATE INDEX idx_game_players_game ON game_players(game_id);
CREATE INDEX idx_game_players_role ON game_players(role);
CREATE INDEX idx_game_players_character ON game_players(character_name);