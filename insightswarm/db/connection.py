from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

_local = threading.local()


def get_db_connection(db_path: str | Path) -> sqlite3.Connection:
    db_path = str(Path(db_path))

    if not hasattr(_local, "connections"):
        _local.connections = {}

    if db_path not in _local.connections:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        _local.connections[db_path] = conn

    return _local.connections[db_path]

