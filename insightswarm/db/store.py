from __future__ import annotations

import hashlib
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from insightswarm.db.connection import get_db_connection
from insightswarm.schemas.swarm import Artifact as SwarmArtifact
from insightswarm.schemas.swarm import BoardItem as SwarmBoardItem
from insightswarm.schemas.swarm import Evidence as SwarmEvidence
from insightswarm.schemas.swarm import Message as SwarmMessage
from insightswarm.schemas.swarm import RunState as SwarmRunState
from insightswarm.schemas.swarm import Task as SwarmTask
from insightswarm.schemas.citation import ImageBBox, TextSpan
from insightswarm.util import dumps, loads, new_id, now_iso


_MISSING = object()


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

    def create_swarm_run_state(
        self,
        *,
        objective: str,
        budget: dict[str, Any] | None = None,
        phase: str = "discovery",
        stop_reason: str | None = None,
        delivery_gate: bool = False,
        run_id: str | None = None,
    ) -> SwarmRunState:
        run_id = run_id or new_id("run")
        now = now_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO swarm_run_states (
                    run_id, objective, phase, budget_json, stop_reason,
                    delivery_gate, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    objective,
                    phase,
                    dumps(budget or {}),
                    stop_reason,
                    int(delivery_gate),
                    now,
                    now,
                ),
            )
        return self.get_swarm_run_state(run_id)

    def get_swarm_run_state(self, run_id: str) -> SwarmRunState:
        row = self.conn.execute(
            "SELECT * FROM swarm_run_states WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"swarm run state not found: {run_id}")
        return SwarmRunState.from_row(row)

    def update_swarm_run_state(
        self,
        run_id: str,
        *,
        phase: str | None = None,
        budget: dict[str, Any] | None = None,
        stop_reason: str | None | object = _MISSING,
        delivery_gate: bool | None = None,
    ) -> SwarmRunState:
        current = self.get_swarm_run_state(run_id)
        next_state = SwarmRunState(
            run_id=current.run_id,
            objective=current.objective,
            phase=phase or current.phase,
            budget=current.budget if budget is None else dict(budget),
            stop_reason=current.stop_reason if stop_reason is _MISSING else stop_reason,
            delivery_gate=current.delivery_gate if delivery_gate is None else delivery_gate,
            created_at=current.created_at,
            updated_at=current.updated_at,
        )
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE swarm_run_states
                SET phase = ?, budget_json = ?, stop_reason = ?, delivery_gate = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (
                    next_state.phase,
                    dumps(next_state.budget),
                    next_state.stop_reason,
                    int(next_state.delivery_gate),
                    now_iso(),
                    run_id,
                ),
            )
        return self.get_swarm_run_state(run_id)

    def create_swarm_task(
        self,
        run_id: str,
        *,
        kind: str,
        status: str,
        owner_role: str,
        inputs: dict[str, Any] | None = None,
        depends_on: Iterable[str] = (),
        priority: int = 0,
        lease_until: str | None = None,
        created_by: str = "system",
    ) -> SwarmTask:
        task = SwarmTask(
            run_id=run_id,
            kind=kind,
            status=status,
            owner_role=owner_role,
            inputs=dict(inputs or {}),
            depends_on=list(depends_on),
            priority=priority,
            lease_until=lease_until,
            created_by=created_by,
        )
        task_id = new_id("task")
        now = now_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO swarm_tasks (
                    task_id, run_id, kind, status, owner_role, inputs_json,
                    depends_on_json, priority, lease_until, created_by,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    run_id,
                    task.kind,
                    task.status,
                    task.owner_role,
                    dumps(task.inputs),
                    dumps(task.depends_on),
                    task.priority,
                    task.lease_until,
                    task.created_by,
                    now,
                    now,
                ),
            )
        return self.get_swarm_task(task_id)

    def get_swarm_task(self, task_id: str) -> SwarmTask:
        row = self.conn.execute(
            "SELECT * FROM swarm_tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"swarm task not found: {task_id}")
        return SwarmTask.from_row(row)

    def list_swarm_tasks(
        self,
        run_id: str,
        *,
        owner_role: str | None = None,
        status: str | None = None,
    ) -> list[SwarmTask]:
        clauses = ["run_id = ?"]
        values: list[Any] = [run_id]
        if owner_role is not None:
            clauses.append("owner_role = ?")
            values.append(owner_role)
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        rows = self.conn.execute(
            f"SELECT * FROM swarm_tasks WHERE {' AND '.join(clauses)} ORDER BY priority DESC, created_at",
            values,
        )
        return [SwarmTask.from_row(row) for row in rows]

    def create_swarm_message(
        self,
        run_id: str,
        *,
        from_role: str,
        message_type: str,
        payload: dict[str, Any] | None = None,
        to_role: str | None = None,
        broadcast: bool = False,
        related_task_id: str | None = None,
    ) -> SwarmMessage:
        message = SwarmMessage(
            run_id=run_id,
            from_role=from_role,
            to_role=to_role,
            broadcast=broadcast,
            type=message_type,
            payload=dict(payload or {}),
            related_task_id=related_task_id,
        )
        message_id = new_id("msg")
        now = now_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO swarm_messages (
                    message_id, run_id, from_role, to_role, broadcast, intent,
                    payload_json, related_task_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    run_id,
                    message.from_role,
                    message.to_role,
                    int(message.broadcast),
                    message.type,
                    dumps(message.payload),
                    message.related_task_id,
                    now,
                ),
            )
        return self.get_swarm_message(message_id)

    def get_swarm_message(self, message_id: str) -> SwarmMessage:
        row = self.conn.execute(
            "SELECT * FROM swarm_messages WHERE message_id = ?",
            (message_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"swarm message not found: {message_id}")
        return SwarmMessage.from_row(row)

    def list_swarm_messages(
        self,
        run_id: str,
        *,
        to_role: str | None = None,
        include_broadcast: bool = True,
    ) -> list[SwarmMessage]:
        clauses = ["run_id = ?"]
        values: list[Any] = [run_id]
        if to_role is not None:
            if include_broadcast:
                clauses.append("(to_role = ? OR broadcast = 1)")
            else:
                clauses.append("to_role = ?")
            values.append(to_role)
        rows = self.conn.execute(
            f"SELECT * FROM swarm_messages WHERE {' AND '.join(clauses)} ORDER BY created_at",
            values,
        )
        return [SwarmMessage.from_row(row) for row in rows]

    def create_swarm_artifact(
        self,
        run_id: str,
        *,
        type: str,
        status: str,
        payload_ref: str,
        summary: str,
        source_task_id: str | None = None,
    ) -> SwarmArtifact:
        artifact_id = new_id("artifact")
        now = now_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO swarm_artifacts (
                    artifact_id, run_id, type, status, source_task_id,
                    payload_ref, summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    run_id,
                    type,
                    status,
                    source_task_id,
                    payload_ref,
                    summary,
                    now,
                ),
            )
        return self.get_swarm_artifact(artifact_id)

    def get_swarm_artifact(self, artifact_id: str) -> SwarmArtifact:
        row = self.conn.execute(
            "SELECT * FROM swarm_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"swarm artifact not found: {artifact_id}")
        return SwarmArtifact.from_row(row)

    def list_swarm_artifacts(
        self,
        run_id: str,
        *,
        source_task_id: str | None = None,
    ) -> list[SwarmArtifact]:
        if source_task_id is None:
            rows = self.conn.execute(
                "SELECT * FROM swarm_artifacts WHERE run_id = ? ORDER BY created_at",
                (run_id,),
            )
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM swarm_artifacts
                WHERE run_id = ? AND source_task_id = ?
                ORDER BY created_at
                """,
                (run_id, source_task_id),
            )
        return [SwarmArtifact.from_row(row) for row in rows]

    def create_swarm_evidence(
        self,
        run_id: str,
        *,
        artifact_id: str,
        source_url: str,
        quote: str,
        freshness: str | None,
        confidence: float,
        qa_state: str,
    ) -> SwarmEvidence:
        evidence_id = new_id("evidence")
        now = now_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO swarm_evidence (
                    evidence_id, run_id, artifact_id, source_url, quote,
                    freshness, confidence, qa_state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    run_id,
                    artifact_id,
                    source_url,
                    quote,
                    freshness,
                    confidence,
                    qa_state,
                    now,
                ),
            )
        return self.get_swarm_evidence(evidence_id)

    def get_swarm_evidence(self, evidence_id: str) -> SwarmEvidence:
        row = self.conn.execute(
            "SELECT * FROM swarm_evidence WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"swarm evidence not found: {evidence_id}")
        return SwarmEvidence.from_row(row)

    def list_swarm_evidence(
        self,
        run_id: str,
        *,
        artifact_id: str | None = None,
        qa_state: str | None = None,
    ) -> list[SwarmEvidence]:
        clauses = ["run_id = ?"]
        values: list[Any] = [run_id]
        if artifact_id is not None:
            clauses.append("artifact_id = ?")
            values.append(artifact_id)
        if qa_state is not None:
            clauses.append("qa_state = ?")
            values.append(qa_state)
        rows = self.conn.execute(
            f"SELECT * FROM swarm_evidence WHERE {' AND '.join(clauses)} ORDER BY created_at",
            values,
        )
        return [SwarmEvidence.from_row(row) for row in rows]

    def create_swarm_board_item(
        self,
        run_id: str,
        *,
        kind: str,
        status: str,
        title: str,
        payload: dict[str, Any] | None = None,
        parent_id: str | None = None,
        evidence_id: str | None = None,
        artifact_id: str | None = None,
        source_task_id: str | None = None,
        dedupe_key: str | None = None,
        priority: int = 0,
        created_by: str = "system",
    ) -> SwarmBoardItem:
        if dedupe_key:
            existing = self.conn.execute(
                "SELECT * FROM swarm_board_items WHERE run_id = ? AND dedupe_key = ?",
                (run_id, dedupe_key),
            ).fetchone()
            if existing is not None:
                return SwarmBoardItem.from_row(existing)

        item_id = new_id("board")
        now = now_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO swarm_board_items (
                    item_id, run_id, kind, status, title, payload_json,
                    parent_id, evidence_id, artifact_id, source_task_id,
                    dedupe_key, priority, created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    run_id,
                    kind,
                    status,
                    title,
                    dumps(payload or {}),
                    parent_id,
                    evidence_id,
                    artifact_id,
                    source_task_id,
                    dedupe_key,
                    priority,
                    created_by,
                    now,
                    now,
                ),
            )
        return self.get_swarm_board_item(item_id)

    def get_swarm_board_item(self, item_id: str) -> SwarmBoardItem:
        row = self.conn.execute(
            "SELECT * FROM swarm_board_items WHERE item_id = ?",
            (item_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"swarm board item not found: {item_id}")
        return SwarmBoardItem.from_row(row)

    def update_swarm_board_item(
        self,
        item_id: str,
        *,
        status: str | None = None,
        title: str | None = None,
        payload: dict[str, Any] | None = None,
        priority: int | None = None,
    ) -> SwarmBoardItem:
        current = self.get_swarm_board_item(item_id)
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE swarm_board_items
                SET status = ?, title = ?, payload_json = ?, priority = ?, updated_at = ?
                WHERE item_id = ?
                """,
                (
                    status or current.status,
                    title or current.title,
                    dumps(current.payload if payload is None else payload),
                    current.priority if priority is None else priority,
                    now_iso(),
                    item_id,
                ),
            )
        return self.get_swarm_board_item(item_id)

    def list_swarm_board_items(
        self,
        run_id: str,
        *,
        kind: str | None = None,
        status: str | None = None,
        parent_id: str | None = None,
    ) -> list[SwarmBoardItem]:
        clauses = ["run_id = ?"]
        values: list[Any] = [run_id]
        if kind is not None:
            clauses.append("kind = ?")
            values.append(kind)
        if status is not None:
            clauses.append("status = ?")
            values.append(status)
        if parent_id is not None:
            clauses.append("parent_id = ?")
            values.append(parent_id)
        rows = self.conn.execute(
            f"SELECT * FROM swarm_board_items WHERE {' AND '.join(clauses)} ORDER BY priority DESC, created_at",
            values,
        )
        return [SwarmBoardItem.from_row(row) for row in rows]

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
