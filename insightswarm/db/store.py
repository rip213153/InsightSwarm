from __future__ import annotations

import hashlib
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from insightswarm.db.connection import get_db_connection
from insightswarm.schemas.citation import ImageBBox, TextSpan
from insightswarm.util import dumps, loads, new_id, now_iso


class Store:
    def __init__(self, db_path: str | Path, artifact_dir: str | Path):
        self.db_path = Path(db_path)
        self.artifact_dir = Path(artifact_dir)
        self._write_lock = threading.RLock()

    @property
    def conn(self) -> sqlite3.Connection:
        return get_db_connection(self.db_path)

    @contextmanager
    def transaction(self):
        with self._write_lock:
            self.conn.execute("BEGIN IMMEDIATE;")
            try:
                yield self.conn
            except Exception:
                self.conn.execute("ROLLBACK;")
                raise
            else:
                self.conn.execute("COMMIT;")

    def create_run(self, name: str, metadata: dict | None = None) -> str:
        run_id = new_id("run")
        now = now_iso()
        phases = ["Discovery", "Extract", "Synthesize", "QA", "Deliver"]
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, name, "created", dumps(metadata or {}), now, now),
            )
            for index, phase in enumerate(phases):
                conn.execute(
                    "INSERT INTO phases VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (new_id("phase"), run_id, phase, "pending", index, now, now),
                )
        return run_id

    def create_task(
        self,
        run_id: str,
        phase: str,
        agent_name: str,
        depends_on: Iterable[str] = (),
        metadata: dict | None = None,
    ) -> str:
        task_id = new_id("task")
        now = now_iso()
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    run_id,
                    phase,
                    agent_name,
                    "pending",
                    dumps(list(depends_on)),
                    0,
                    dumps(metadata or {}),
                    now,
                    now,
                ),
            )
        return task_id

    def update_run_status(self, run_id: str, status: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE run_id = ?",
                (status, now_iso(), run_id),
            )

    def set_task_status(
        self,
        task_id: str,
        status: str,
        metadata_update: dict | None = None,
        retry_delta: int = 0,
    ) -> None:
        task = self.get_task(task_id)
        metadata = loads(task["metadata_json"], {})
        metadata.update(metadata_update or {})
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, retry_count = retry_count + ?, metadata_json = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (status, retry_delta, dumps(metadata), now_iso(), task_id),
            )

    def get_run(self, run_id: str) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run not found: {run_id}")
        return row

    def get_run_metadata(self, run_id: str) -> dict:
        return loads(self.get_run(run_id)["metadata_json"], {})

    def get_task(self, task_id: str) -> sqlite3.Row:
        row = self.conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            raise KeyError(f"task not found: {task_id}")
        return row

    def list_tasks(self, run_id: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM tasks WHERE run_id = ? ORDER BY created_at", (run_id,)
            )
        )

    def list_artifacts(self, run_id: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at", (run_id,)
            )
        )

    def list_citations(self, run_id: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM citations WHERE run_id = ? ORDER BY created_at", (run_id,)
            )
        )

    def list_events(self, run_id: str, limit: int = 50) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT * FROM agent_events
                WHERE run_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (run_id, limit),
            )
        )

    def write_artifact(
        self,
        run_id: str,
        task_id: str | None,
        artifact_type: str,
        mime_type: str,
        content: bytes | str,
        source_url: str | None = None,
        metadata: dict | None = None,
        suffix: str = ".txt",
    ) -> str:
        artifact_id = new_id("artifact")
        raw = content.encode("utf-8") if isinstance(content, str) else content
        digest = hashlib.sha256(raw).hexdigest()
        run_dir = self.artifact_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{artifact_id}{suffix}"
        path.write_bytes(raw)
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    artifact_id,
                    run_id,
                    task_id,
                    artifact_type,
                    mime_type,
                    str(path),
                    digest,
                    source_url,
                    dumps(metadata or {}),
                    now_iso(),
                ),
            )
        return artifact_id

    def create_document_citation(
        self,
        run_id: str,
        task_id: str,
        artifact_id: str,
        source_url: str,
        quote: str,
        text_span: TextSpan,
        confidence: float,
    ) -> str:
        text_span.validate()
        citation_id = new_id("doc")
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO citations
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    citation_id,
                    run_id,
                    task_id,
                    "document",
                    artifact_id,
                    source_url,
                    quote,
                    dumps({"start": text_span.start, "end": text_span.end}),
                    None,
                    dumps([]),
                    None,
                    confidence,
                    now_iso(),
                ),
            )
        return citation_id

    def create_image_citation(
        self,
        run_id: str,
        task_id: str,
        artifact_id: str,
        source_url: str,
        image_bbox: ImageBBox,
        confidence: float,
    ) -> str:
        citation_id = new_id("img")
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO citations
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    citation_id,
                    run_id,
                    task_id,
                    "image",
                    artifact_id,
                    source_url,
                    None,
                    None,
                    dumps(image_bbox.to_dict()),
                    dumps([]),
                    None,
                    confidence,
                    now_iso(),
                ),
            )
        return citation_id

    def create_inference_citation(
        self,
        run_id: str,
        task_id: str,
        evidence_ids: list[str],
        claim: str,
        confidence: float,
    ) -> str:
        if not evidence_ids:
            raise ValueError("inference citation requires at least one evidence id")
        citation_id = new_id("inf")
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO citations
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    citation_id,
                    run_id,
                    task_id,
                    "inference",
                    None,
                    None,
                    None,
                    None,
                    None,
                    dumps(evidence_ids),
                    claim,
                    confidence,
                    now_iso(),
                ),
            )
        return citation_id

    def emit_event(
        self,
        run_id: str,
        task_id: str | None,
        agent_name: str,
        event_type: str,
        message: str,
        metadata: dict | None = None,
    ) -> str:
        event_id = new_id("event")
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO agent_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    run_id,
                    task_id,
                    agent_name,
                    event_type,
                    message,
                    dumps(metadata or {}),
                    now_iso(),
                ),
            )
        return event_id

    def record_model_call(
        self,
        run_id: str,
        task_id: str | None,
        provider: str,
        model: str,
        request: dict,
        response: dict,
        usage: dict | None,
        latency_ms: int,
        status: str,
        error: str | None = None,
    ) -> str:
        call_id = new_id("model")
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO model_calls (
                    model_call_id, run_id, task_id, provider, model,
                    request_json, response_json, usage_json, latency_ms,
                    status, error, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call_id,
                    run_id,
                    task_id,
                    provider,
                    model,
                    dumps(request),
                    dumps(response),
                    dumps(usage or {}),
                    latency_ms,
                    status,
                    error,
                    now_iso(),
                ),
            )
        return call_id

    def create_message(
        self,
        run_id: str,
        task_id: str | None,
        sender: str,
        recipient: str,
        payload: dict,
        idempotency_key: str,
    ) -> str:
        existing = self.conn.execute(
            "SELECT message_id FROM messages WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        if existing:
            return existing["message_id"]
        message_id = new_id("msg")
        now = now_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    message_id, run_id, task_id, sender, recipient, status,
                    lease_owner, leased_at, lease_expires_at, acked_at,
                    idempotency_key, payload_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    run_id,
                    task_id,
                    sender,
                    recipient,
                    "pending",
                    None,
                    None,
                    None,
                    None,
                    idempotency_key,
                    dumps(payload),
                    now,
                    now,
                ),
            )
        return message_id

    def lease_messages(
        self,
        run_id: str,
        recipient: str,
        lease_owner: str,
        expires_at: str,
        limit: int = 10,
    ) -> list[sqlite3.Row]:
        now = now_iso()
        with self.transaction() as conn:
            rows = list(
                conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE run_id = ? AND recipient = ? AND status = 'pending'
                    ORDER BY created_at
                    LIMIT ?
                    """,
                    (run_id, recipient, limit),
                )
            )
            for row in rows:
                conn.execute(
                    """
                    UPDATE messages
                    SET status = 'leased', lease_owner = ?, leased_at = ?,
                        lease_expires_at = ?, updated_at = ?
                    WHERE message_id = ?
                    """,
                    (lease_owner, now, expires_at, now, row["message_id"]),
                )
        return [
            self.conn.execute("SELECT * FROM messages WHERE message_id = ?", (row["message_id"],)).fetchone()
            for row in rows
        ]

    def ack_messages(self, message_ids: list[str]) -> None:
        if not message_ids:
            return
        now = now_iso()
        with self.transaction() as conn:
            conn.executemany(
                """
                UPDATE messages
                SET status = 'acked', acked_at = ?, updated_at = ?
                WHERE message_id = ?
                """,
                [(now, now, message_id) for message_id in message_ids],
            )

    def recover_expired_leases(self, now: str | None = None) -> int:
        now = now or now_iso()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE messages
                SET status = 'pending', lease_owner = NULL, leased_at = NULL,
                    lease_expires_at = NULL, updated_at = ?
                WHERE status = 'leased' AND lease_expires_at < ?
                """,
                (now, now),
            )
        return cursor.rowcount

    def get_artifact(self, artifact_id: str) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"artifact not found: {artifact_id}")
        return row

    def get_citation(self, citation_id: str) -> sqlite3.Row:
        row = self.conn.execute(
            "SELECT * FROM citations WHERE citation_id = ?", (citation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"citation not found: {citation_id}")
        return row
