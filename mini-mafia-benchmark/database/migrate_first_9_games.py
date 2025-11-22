#!/usr/bin/env python3
"""
Migrate first 9 games to a fresh database.
- Renames old db to mini_mafia_human_old.db
- Creates new mini_mafia_human.db
- Copies first 9 games
- Next game will start at ID 10
"""

import sqlite3
import shutil
from pathlib import Path

DB_DIR = Path(__file__).parent
OLD_DB = DB_DIR / 'mini_mafia_human.db'
BACKUP_DB = DB_DIR / 'mini_mafia_human_old.db'
NEW_DB = DB_DIR / 'mini_mafia_human_new.db'

def create_schema(conn):
    """Create database schema"""
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
    print("✅ Schema created")

def migrate():
    # Check old db exists
    if not OLD_DB.exists():
        print(f"❌ Database not found: {OLD_DB}")
        return

    # Backup old db
    if BACKUP_DB.exists():
        print(f"⚠️  Backup already exists: {BACKUP_DB}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    print(f"📦 Backing up old database...")
    shutil.copy2(OLD_DB, BACKUP_DB)
    print(f"   Created: {BACKUP_DB}")

    # Create new database
    print(f"\n🆕 Creating new database...")
    new_conn = sqlite3.connect(NEW_DB)
    create_schema(new_conn)

    # Connect to old database
    old_conn = sqlite3.connect(BACKUP_DB)
    old_cursor = old_conn.cursor()
    new_cursor = new_conn.cursor()

    # Copy first 9 games
    print(f"\n📋 Copying first 9 games...")

    old_cursor.execute("""
        SELECT game_id, timestamp, winner, mafioso_name, detective_name,
               villager_name, arrested_name, background_name
        FROM games
        WHERE game_id <= 9
        ORDER BY game_id
    """)

    games = old_cursor.fetchall()

    if len(games) == 0:
        print("❌ No games found with game_id <= 9")
        old_conn.close()
        new_conn.close()
        return

    for game in games:
        new_cursor.execute("""
            INSERT INTO games (game_id, timestamp, winner, mafioso_name, detective_name,
                             villager_name, arrested_name, background_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, game)

    print(f"   Copied {len(games)} games (IDs: {games[0][0]} to {games[-1][0]})")

    # Copy corresponding game_actions
    print(f"\n📋 Copying game actions...")

    old_cursor.execute("""
        SELECT action_id, game_id, step, action_type, actor, raw_response, parsed_result
        FROM game_actions
        WHERE game_id <= 9
        ORDER BY action_id
    """)

    actions = old_cursor.fetchall()

    for action in actions:
        new_cursor.execute("""
            INSERT INTO game_actions (action_id, game_id, step, action_type, actor,
                                    raw_response, parsed_result)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, action)

    print(f"   Copied {len(actions)} game actions")

    new_conn.commit()

    # Verify
    new_cursor.execute("SELECT COUNT(*), MAX(game_id) FROM games")
    total_games, max_id = new_cursor.fetchone()

    print(f"\n✅ Migration complete!")
    print(f"   New database: {NEW_DB}")
    print(f"   Games copied: {total_games}")
    print(f"   Highest game_id: {max_id}")
    print(f"   Next game will be ID: {max_id + 1}")

    old_conn.close()
    new_conn.close()

    # Replace old db with new one
    print(f"\n🔄 Replacing old database with new one...")
    OLD_DB.unlink()  # Delete old
    NEW_DB.rename(OLD_DB)  # Rename new to old name

    print(f"\n✅ Done!")
    print(f"   Active database: {OLD_DB}")
    print(f"   Backup (all games): {BACKUP_DB}")
    print(f"   Next game will save with ID 10")

if __name__ == "__main__":
    migrate()
