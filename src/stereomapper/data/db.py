"""Module to assign the settings for the databases created in stereomapper"""
import sqlite3

def connect(db_path: str, use_wal: bool = False) -> sqlite3.Connection:
    """
    Connect to a database and set the following paramaters.

    Args:
    - db_path: str, full path to the database in question.
    - use_wal: bool, use WAL settings or not.

    Returns:
    - conn: sqlite3.Connection
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 120000;")

    if use_wal:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
    else:
        conn.execute("PRAGMA journal_mode = DELETE;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        conn.execute("PRAGMA mmap_size = 30000000000;")  # 30 GB mapping (no-op if not supported)
        conn.execute("PRAGMA cache_size = -200000;")      # ~200MB page cache
        conn.execute("PRAGMA locking_mode = EXCLUSIVE;")

    return conn
