from __future__ import annotations

from pathlib import Path

from insightswarm.db.connection import get_db_connection


def init_db(db_path: str | Path) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    conn = get_db_connection(db_path)
    conn.executescript(schema_path.read_text(encoding="utf-8"))

