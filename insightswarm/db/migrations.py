from __future__ import annotations

from pathlib import Path

from insightswarm.db.connection import get_db_connection


def init_db(db_path: str | Path) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    conn = get_db_connection(db_path)
    _drop_legacy_runtime_tables(conn)
    conn.executescript(schema_path.read_text(encoding="utf-8"))


def _drop_legacy_runtime_tables(conn) -> None:
    """Remove pre-swarm runtime tables from local databases.

    The active runtime stores all coordination data in ``swarm_*`` tables.
    Keeping the old table family around lets stale foreign keys and fallback
    reads silently re-enter the system, so startup prunes them deliberately.
    """
    model_calls_is_legacy = any(
        row["table"] in {"runs", "tasks"}
        for row in conn.execute("PRAGMA foreign_key_list(model_calls)").fetchall()
    )
    conn.execute("PRAGMA foreign_keys=OFF;")
    try:
        if model_calls_is_legacy:
            conn.execute("DROP TABLE IF EXISTS model_calls;")
        for table in (
            "agent_events",
            "messages",
            "citations",
            "artifacts",
            "tasks",
            "phases",
            "runs",
        ):
            conn.execute(f"DROP TABLE IF EXISTS {table};")
    finally:
        conn.execute("PRAGMA foreign_keys=ON;")
