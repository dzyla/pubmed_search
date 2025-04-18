#!/usr/bin/env python3
import sqlite3
from datetime import datetime

try:
    from tabulate import tabulate
except ImportError:
    print("The 'tabulate' package is not installed. Install it with 'pip install tabulate'.")
    exit(1)

def fetch_session_history(db_path: str = "sessions_history.db"):
    """
    Connects to the SQLite database and fetches all records from the session_history table.
    Returns a list of tuples (id, session_id, last_seen_formatted).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Ensure the table exists.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            last_seen INTEGER
        )
    """)
    conn.commit()
    
    cursor.execute("SELECT id, session_id, last_seen FROM session_history ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    
    # Convert last_seen timestamp to formatted string.
    formatted_rows = []
    for row in rows:
        rec_id, session_id, last_seen = row
        formatted_time = datetime.fromtimestamp(last_seen).strftime("%Y-%m-%d-%H-%M-%S")
        formatted_rows.append((rec_id, session_id, formatted_time))
    return formatted_rows

def main():
    data = fetch_session_history()
    if not data:
        print("No session history found.")
    else:
        headers = ["ID", "Session ID", "Last Seen (YYYY-MM-DD-HH-mm-ss)"]
        table = tabulate(data, headers=headers, tablefmt="pretty")
        print(table)

if __name__ == "__main__":
    main()
