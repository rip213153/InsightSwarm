from __future__ import annotations

from datetime import datetime, UTC, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Any

from insightswarm.db.store import Store
from insightswarm.event_bus import EventBus
from insightswarm.message_protocol import validate_message
from insightswarm.schemas.swarm import Artifact, BoardItem, Evidence, Message, Task
from insightswarm.util import dumps, new_id

ACTIVE_TASK_STATUSES = {"pending", "leased"}


class TaskStore:
    def __init__(self, store: Store, *, event_bus: EventBus | None = None):
        self.store = store
        self._event_bus = event_bus

    def create(
        self,
        run_id: str,
        **task_kwargs: Any,
    ) -> Task:
        with self.store.transaction() as conn:
            existing = self._find_existing_task_in_conn(conn, run_id, task_kwargs)
            if existing is not None:
                return existing

            task = Task(
                run_id=run_id,
                kind=str(task_kwargs["kind"]),
                status=str(task_kwargs["status"]),
                owner_role=str(task_kwargs["owner_role"]),
                inputs=dict(task_kwargs.get("inputs") or {}),
                depends_on=list(task_kwargs.get("depends_on") or []),
                priority=int(task_kwargs.get("priority") or 0),
                lease_until=task_kwargs.get("lease_until"),
                created_by=str(task_kwargs.get("created_by") or "system"),
            )
            task_id = new_id("task")
            now = _utc_now()
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
        # Transaction has committed above (the `with` block exited). Notify only
        # after the commit so a woken worker can actually see the new row.
        bus = self._event_bus
        if bus is not None:
            target_role = str(task_kwargs.get("owner_role") or "")
            if target_role:
                bus.notify_role(target_role)
        return self.store.get_swarm_task(task_id)

    def list_pending(self, run_id: str, owner_role: str | None = None) -> list[Task]:
        return self.store.list_swarm_tasks(run_id, owner_role=owner_role, status="pending")

    def claim_next(
        self,
        run_id: str,
        *,
        owner_role: str,
        lease_seconds: int = 900,
    ) -> Task | None:
        lease_until = _utc_now_plus(lease_seconds)
        with self.store.transaction() as conn:
            rows = list(
                conn.execute(
                    """
                    SELECT * FROM swarm_tasks
                    WHERE run_id = ? AND owner_role = ? AND status = 'pending'
                    ORDER BY priority DESC, created_at
                    """,
                    (run_id, owner_role),
                )
            )
            for row in rows:
                task = Task.from_row(row)
                if not self._dependencies_satisfied_in_conn(conn, task):
                    continue
                cursor = conn.execute(
                    """
                    UPDATE swarm_tasks
                    SET status = ?, lease_until = ?, updated_at = ?
                    WHERE task_id = ? AND status = 'pending'
                    """,
                    ("leased", lease_until, _utc_now(), task.task_id),
                )
                if cursor.rowcount == 1:
                    return self.store.get_swarm_task(task.task_id)
        return None

    def recover_expired_leases(self, run_id: str) -> int:
        now = _utc_now()
        with self.store.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE swarm_tasks
                SET status = 'pending', lease_until = NULL, updated_at = ?
                WHERE run_id = ? AND status = 'leased' AND lease_until IS NOT NULL AND lease_until < ?
                """,
                (now, run_id, now),
            )
            return int(cursor.rowcount or 0)

    def renew_lease_if_leased(self, task_id: str, *, lease_seconds: int = 900) -> bool:
        # Conditional lease renewal: extends lease_until ONLY when the task is
        # still leased. This never resurrects a task the body already
        # completed/blocked/needs_repaired — the WHERE status='leased' guard
        # prevents a late LeaseGuard heartbeat from undoing the release. Used
        # by LeaseGuard's heartbeat loop.
        lease_until = _utc_now_plus(lease_seconds)
        with self.store.transaction() as conn:
            cursor = conn.execute(
                "UPDATE swarm_tasks SET lease_until = ?, updated_at = ? WHERE task_id = ? AND status = 'leased'",
                (lease_until, _utc_now(), task_id),
            )
            return int(cursor.rowcount or 0) > 0

    def complete(self, task_id: str) -> Task:
        return self.update_status(task_id, status="done", lease_until=None)

    def block(self, task_id: str) -> Task:
        return self.update_status(task_id, status="blocked", lease_until=None)

    def needs_repair(self, task_id: str) -> Task:
        return self.update_status(task_id, status="needs_repair", lease_until=None)

    def update_status(
        self,
        task_id: str,
        *,
        status: str,
        lease_until: str | None,
    ) -> Task:
        with self.store.transaction() as conn:
            conn.execute(
                """
                UPDATE swarm_tasks
                SET status = ?, lease_until = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (status, lease_until, _utc_now(), task_id),
            )
        return self.store.get_swarm_task(task_id)

    def list_active(self, run_id: str, owner_role: str | None = None) -> list[Task]:
        return [
            task
            for task in self.store.list_swarm_tasks(run_id, owner_role=owner_role)
            if task.status in ACTIVE_TASK_STATUSES
        ]

    def _dependencies_satisfied(self, task: Task) -> bool:
        return self._dependencies_satisfied_in_conn(self.store.conn, task)

    def _dependencies_satisfied_in_conn(self, conn: sqlite3.Connection, task: Task) -> bool:
        for dependency_id in task.depends_on:
            dependency = conn.execute(
                "SELECT status FROM swarm_tasks WHERE task_id = ?",
                (dependency_id,),
            ).fetchone()
            if dependency is None or dependency["status"] != "done":
                return False
        return True

    def _find_existing_task_in_conn(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        task_kwargs: dict[str, Any],
    ) -> Task | None:
        kind = str(task_kwargs.get("kind") or "")
        owner_role = str(task_kwargs.get("owner_role") or "")
        inputs = dict(task_kwargs.get("inputs") or {})
        tasks = self._list_tasks_in_conn(conn, run_id, owner_role=owner_role or None)

        if kind == "raw_document" and owner_role == "extractor":
            if str(inputs.get("retry_of_task_id") or ""):
                return None
            artifact_id = str(inputs.get("artifact_id") or "")
            if artifact_id:
                for task in tasks:
                    if task.kind == kind and str(task.inputs.get("artifact_id") or "") == artifact_id:
                        if task.status in ACTIVE_TASK_STATUSES:
                            return task

        if kind == "delivery_request" and owner_role == "writer":
            for task in tasks:
                if task.kind == kind and task.status in ACTIVE_TASK_STATUSES:
                    return task

        if kind == "hard_acquisition" and owner_role == "browser_agent":
            issue_key = str(inputs.get("issue_key") or "")
            target_url = str(inputs.get("target_url") or "")
            goal = _normalize_board_text(str(inputs.get("goal") or ""))
            for task in tasks:
                if task.kind != kind or task.status not in ACTIVE_TASK_STATUSES:
                    continue
                if issue_key and str(task.inputs.get("issue_key") or "") == issue_key:
                    return task
                if target_url and str(task.inputs.get("target_url") or "") == target_url:
                    return task
                if goal and _normalize_board_text(str(task.inputs.get("goal") or "")) == goal:
                    return task

        if kind == "repair_request":
            issue_key = str(inputs.get("issue_key") or "")
            if issue_key:
                for task in tasks:
                    if (
                        task.kind == kind
                        and task.status in ACTIVE_TASK_STATUSES
                        and str(task.inputs.get("issue_key") or "") == issue_key
                    ):
                        return task

        if kind == "evidence_review" and owner_role == "critic":
            bundle_key = _evidence_bundle_key(inputs)
            if bundle_key:
                for task in tasks:
                    if task.kind != kind:
                        continue
                    if task.status not in {"pending", "leased", "done"}:
                        continue
                    if _evidence_bundle_key(task.inputs) == bundle_key:
                        return task

        if kind == "extraction_failure_review" and owner_role == "critic":
            source_artifact_id = str(inputs.get("source_artifact_id") or "")
            extractor_task_id = str(inputs.get("extractor_task_id") or "")
            if source_artifact_id or extractor_task_id:
                for task in tasks:
                    if task.kind != kind or task.status not in {"pending", "leased", "done"}:
                        continue
                    if source_artifact_id and str(task.inputs.get("source_artifact_id") or "") == source_artifact_id:
                        return task
                    if extractor_task_id and str(task.inputs.get("extractor_task_id") or "") == extractor_task_id:
                        return task

        return None

    def _list_tasks_in_conn(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        *,
        owner_role: str | None = None,
    ) -> list[Task]:
        clauses = ["run_id = ?"]
        values: list[Any] = [run_id]
        if owner_role is not None:
            clauses.append("owner_role = ?")
            values.append(owner_role)
        rows = conn.execute(
            f"SELECT * FROM swarm_tasks WHERE {' AND '.join(clauses)} ORDER BY priority DESC, created_at",
            values,
        )
        return [Task.from_row(row) for row in rows]


class Mailbox:
    def __init__(self, store: Store, *, event_bus: EventBus | None = None):
        self.store = store
        self._event_bus = event_bus

    def send(
        self,
        run_id: str,
        *,
        from_role: str,
        to_role: str | None = None,
        broadcast: bool = False,
        message_type: str,
        payload: dict[str, Any] | None = None,
        related_task_id: str | None = None,
    ) -> Message:
        validate_message(
            message_type=message_type,
            payload=payload,
            related_task_id=related_task_id,
        )
        message = self.store.create_swarm_message(
            run_id,
            from_role=from_role,
            to_role=to_role,
            broadcast=broadcast,
            message_type=message_type,
            payload=payload,
            related_task_id=related_task_id,
        )
        # create_swarm_message commits its own transaction before returning, so
        # the row is durable before we wake any recipient worker.
        bus = self._event_bus
        if bus is not None:
            if broadcast:
                # Broadcasts reach every role's inbox; wake them all so each
                # role's worker observes the message on its next claim cycle.
                bus.notify_all_roles()
            elif to_role:
                bus.notify_role(to_role)
        return message

    def inbox(self, run_id: str, *, role: str) -> list[Message]:
        return self.store.list_swarm_messages(run_id, to_role=role, include_broadcast=True)

    def broadcasts(self, run_id: str) -> list[Message]:
        return [message for message in self.store.list_swarm_messages(run_id) if message.broadcast]


class BoardStore:
    def __init__(self, store: Store):
        self.store = store

    def create_question(
        self,
        run_id: str,
        *,
        title: str,
        question_type: str,
        status: str = "open",
        parent_id: str | None = None,
        owner_role: str = "researcher",
        priority: int = 0,
        created_by: str = "system",
        payload: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> BoardItem:
        data = {
            "question_type": question_type,
            "owner_role": owner_role,
            **dict(payload or {}),
        }
        return self.store.create_swarm_board_item(
            run_id,
            kind="question",
            status=status,
            title=title,
            payload=data,
            parent_id=parent_id,
            priority=priority,
            created_by=created_by,
            dedupe_key=dedupe_key or _board_dedupe("question", title, parent_id),
        )

    def create_claim(
        self,
        run_id: str,
        *,
        title: str,
        question_id: str | None,
        claim_type: str,
        status: str = "proposed",
        priority: int = 0,
        created_by: str = "system",
        payload: dict[str, Any] | None = None,
        source_task_id: str | None = None,
        dedupe_key: str | None = None,
    ) -> BoardItem:
        data = {"claim_type": claim_type, "question_id": question_id, **dict(payload or {})}
        return self.store.create_swarm_board_item(
            run_id,
            kind="claim",
            status=status,
            title=title,
            payload=data,
            parent_id=question_id,
            source_task_id=source_task_id,
            priority=priority,
            created_by=created_by,
            dedupe_key=dedupe_key,
        )

    def create_conflict(
        self,
        run_id: str,
        *,
        title: str,
        question_id: str | None,
        status: str = "open",
        priority: int = 0,
        created_by: str = "system",
        payload: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> BoardItem:
        data = {"question_id": question_id, **dict(payload or {})}
        return self.store.create_swarm_board_item(
            run_id,
            kind="conflict",
            status=status,
            title=title,
            payload=data,
            parent_id=question_id,
            priority=priority,
            created_by=created_by,
            dedupe_key=dedupe_key,
        )

    def write_plan(
        self,
        run_id: str,
        *,
        title: str,
        plan_kind: str,
        status: str = "active",
        parent_id: str | None = None,
        priority: int = 0,
        created_by: str = "system",
        payload: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> BoardItem:
        data = {"plan_kind": plan_kind, **dict(payload or {})}
        return self.store.create_swarm_board_item(
            run_id,
            kind="plan",
            status=status,
            title=title,
            payload=data,
            parent_id=parent_id,
            priority=priority,
            created_by=created_by,
            dedupe_key=dedupe_key,
        )

    def record_evidence(
        self,
        run_id: str,
        *,
        evidence: Evidence,
        question_id: str | None,
        artifact_id: str | None = None,
        source_task_id: str | None = None,
        issue_key: str | None = None,
    ) -> BoardItem:
        return self.store.create_swarm_board_item(
            run_id,
            kind="evidence",
            status=evidence.qa_state,
            title=evidence.quote[:160],
            payload={
                "question_id": question_id,
                "source_url": evidence.source_url,
                "confidence": evidence.confidence,
                "freshness": evidence.freshness,
                "issue_key": issue_key,
            },
            parent_id=question_id,
            evidence_id=evidence.evidence_id,
            artifact_id=artifact_id or evidence.artifact_id,
            source_task_id=source_task_id,
            priority=0,
            created_by="extractor",
            dedupe_key=f"evidence:{evidence.evidence_id}",
        )

    def issue_keys_for_evidence(self, run_id: str, evidence_ids: list[str]) -> list[str]:
        evidence_id_set = {str(value) for value in evidence_ids if str(value)}
        if not evidence_id_set:
            return []

        issue_keys: set[str] = set()
        for item in self.store.list_swarm_board_items(run_id, kind="evidence"):
            if item.evidence_id not in evidence_id_set:
                continue
            issue_key = str(item.payload.get("issue_key") or "").strip()
            if issue_key:
                issue_keys.add(issue_key)
                continue
            if item.source_task_id:
                task = self.store.get_swarm_task(item.source_task_id)
                task_issue_key = str(task.inputs.get("issue_key") or "").strip()
                if task_issue_key:
                    issue_keys.add(task_issue_key)
        return sorted(issue_keys)

    def scoped_snapshot(
        self,
        run_id: str,
        *,
        focus_question_id: str | None = None,
        question_text: str | None = None,
    ) -> dict[str, list[BoardItem]]:
        items = self.store.list_swarm_board_items(run_id)
        if focus_question_id is None and question_text:
            focus_question_id = self._find_question_id(run_id, question_text)

        if focus_question_id:
            related_question_ids = {
                focus_question_id,
                *[
                    item.item_id
                    for item in items
                    if item.kind == "question" and item.parent_id == focus_question_id
                ],
            }
            filtered = [
                item
                for item in items
                if item.kind == "plan"
                or item.item_id in related_question_ids
                or item.parent_id in related_question_ids
                or str(item.payload.get("question_id") or "") in related_question_ids
            ]
        else:
            filtered = items

        snapshot = {"question": [], "claim": [], "evidence": [], "conflict": [], "plan": []}
        for item in filtered:
            snapshot.setdefault(item.kind, []).append(item)
        return snapshot

    def update_status(self, item_id: str, *, status: str) -> BoardItem:
        return self.store.update_swarm_board_item(item_id, status=status)

    def update_payload(self, item_id: str, *, payload: dict[str, Any]) -> BoardItem:
        current = self.store.get_swarm_board_item(item_id)
        merged = {**current.payload, **payload}
        return self.store.update_swarm_board_item(item_id, payload=merged)

    def resolve_conflicts(
        self,
        run_id: str,
        *,
        issue_keys: list[str] | None = None,
        evidence_ids: list[str] | None = None,
        resolved_by: str,
        reason: str,
        resolution_event_at: str | None = None,
    ) -> list[BoardItem]:
        issue_key_set = {str(item) for item in (issue_keys or []) if str(item)}
        evidence_id_set = {str(item) for item in (evidence_ids or []) if str(item)}
        if not issue_key_set and not evidence_id_set:
            return []

        resolved: list[BoardItem] = []
        for item in self.store.list_swarm_board_items(run_id, kind="conflict"):
            if item.status not in {"open", "active"}:
                continue
            if resolution_event_at and item.created_at and item.created_at > resolution_event_at:
                continue
            payload = dict(item.payload or {})
            conflict_issue_key = str(payload.get("issue_key") or "")
            conflict_evidence_ids = {str(value) for value in list(payload.get("evidence_ids") or payload.get("conflicting_evidence_ids") or []) if str(value)}
            if issue_key_set and conflict_issue_key in issue_key_set:
                matched = True
            elif evidence_id_set and conflict_evidence_ids and evidence_id_set.intersection(conflict_evidence_ids):
                matched = True
            else:
                matched = False
            if not matched:
                continue
            updated_payload = {
                **payload,
                "resolved_by": resolved_by,
                "resolution_reason": reason,
                "resolved_at": _utc_now(),
            }
            resolved.append(self.store.update_swarm_board_item(item.item_id or "", status="resolved", payload=updated_payload))
        return resolved

    def mark_claims_for_question(self, run_id: str, *, question_id: str | None, status: str) -> list[BoardItem]:
        if not question_id:
            return []
        updated: list[BoardItem] = []
        for item in self.store.list_swarm_board_items(run_id, kind="claim"):
            if item.parent_id != question_id and str(item.payload.get("question_id") or "") != question_id:
                continue
            updated.append(self.store.update_swarm_board_item(item.item_id or "", status=status))
        return updated

    def _find_question_id(self, run_id: str, title: str) -> str | None:
        normalized = _normalize_board_text(title)
        for item in self.store.list_swarm_board_items(run_id, kind="question"):
            if _normalize_board_text(item.title) == normalized:
                return item.item_id
        return None


class ArtifactStore:
    def __init__(self, store: Store):
        self.store = store

    def write_raw_document(
        self,
        run_id: str,
        *,
        source_task_id: str | None,
        document: dict[str, Any],
        summary: str,
    ) -> Artifact:
        artifact = self.store.create_swarm_artifact(
            run_id,
            type="raw_document",
            status="ready",
            source_task_id=source_task_id,
            payload_ref="pending",
            summary=summary,
        )
        return self._persist_payload(artifact, document)

    def write_citation(
        self,
        run_id: str,
        *,
        source_task_id: str | None,
        citation: dict[str, Any],
        summary: str,
    ) -> Artifact:
        artifact = self.store.create_swarm_artifact(
            run_id,
            type="citation",
            status="ready",
            source_task_id=source_task_id,
            payload_ref="pending",
            summary=summary,
        )
        return self._persist_payload(artifact, citation)

    def write_report(
        self,
        run_id: str,
        *,
        source_task_id: str | None,
        report_kind: str,
        body: str,
        summary: str,
    ) -> Artifact:
        artifact = self.store.create_swarm_artifact(
            run_id,
            type=report_kind,
            status="ready",
            source_task_id=source_task_id,
            payload_ref="pending",
            summary=summary,
        )
        payload = {"body": body, "summary": summary, "report_kind": report_kind}
        return self._persist_text(artifact, payload, suffix=".md")

    def read_payload(self, artifact_id: str) -> dict[str, Any]:
        artifact = self.store.get_swarm_artifact(artifact_id)
        return json.loads(Path(artifact.payload_ref).read_text(encoding="utf-8"))

    def create_evidence(
        self,
        run_id: str,
        *,
        artifact_id: str,
        source_url: str,
        quote: str,
        freshness: str | None,
        confidence: float,
        qa_state: str,
    ) -> Evidence:
        return self.store.create_swarm_evidence(
            run_id,
            artifact_id=artifact_id,
            source_url=source_url,
            quote=quote,
            freshness=freshness,
            confidence=confidence,
            qa_state=qa_state,
        )

    def _persist_payload(self, artifact: Artifact, payload: dict[str, Any]) -> Artifact:
        return self._persist_text(artifact, payload, suffix=".json")

    def _persist_text(self, artifact: Artifact, payload: dict[str, Any], *, suffix: str) -> Artifact:
        payload_dir = self.store.artifact_dir / artifact.run_id / "swarm"
        payload_dir.mkdir(parents=True, exist_ok=True)
        payload_path = payload_dir / f"{artifact.artifact_id}{suffix}"
        if suffix == ".md":
            payload_path.write_text(str(payload["body"]), encoding="utf-8")
        else:
            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE swarm_artifacts SET payload_ref = ? WHERE artifact_id = ?",
                (str(payload_path), artifact.artifact_id),
            )
        return self.store.get_swarm_artifact(artifact.artifact_id)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _utc_now_plus(seconds: int) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def _evidence_bundle_key(inputs: dict[str, Any]) -> str:
    explicit = str(inputs.get("evidence_bundle_key") or "")
    if explicit:
        return explicit
    evidence_ids = sorted(str(item) for item in (inputs.get("evidence_ids") or []) if str(item))
    return "|".join(evidence_ids)


def _normalize_board_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _board_dedupe(kind: str, title: str, parent_id: str | None) -> str:
    return f"{kind}:{parent_id or 'root'}:{_normalize_board_text(title)}"
