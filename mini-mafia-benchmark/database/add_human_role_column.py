#!/usr/bin/env python3
"""
Add human_role column to existing mini_mafia_human.db
Sets existing games to 'mafioso' (deceive games)
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / 'mini_mafia_human.db'

def migrate():
    if not DB_PATH.exists():
        print(f"❌ Database not found: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if column already exists
    cursor.execute("PRAGMA table_info(games)")
    columns = [row[1] for row in cursor.fetchall()]

    if 'human_role' in columns:
        print("✅ Column 'human_role' already exists")
        conn.close()
        return

    # Add column
    print("📝 Adding 'human_role' column...")
    cursor.execute("ALTER TABLE games ADD COLUMN human_role TEXT")

    # Set all existing games to 'mafioso' (deceive games)
    cursor.execute("UPDATE games SET human_role = 'mafioso'")

    conn.commit()

    # Verify
    cursor.execute("SELECT COUNT(*) FROM games WHERE human_role = 'mafioso'")
    count = cursor.fetchone()[0]

    print(f"✅ Migration complete!")
    print(f"   Updated {count} existing games to human_role='mafioso'")

    conn.close()

if __name__ == "__main__":
    migrate()
