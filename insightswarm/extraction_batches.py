from __future__ import annotations

from datetime import datetime, UTC
from typing import Any

from insightswarm.db.store import Store
from insightswarm.swarm_store import BoardStore, Mailbox, TaskStore
from insightswarm.util import now_iso


EXTRACTION_BATCH_PLAN_KIND = "extraction_batch"
EXTRACTION_BATCH_TIMEOUT_SECONDS = 900
TERMINAL_EXTRACTION_STATUSES = {"done", "blocked", "needs_repair", "technical_failed"}
EXISTING_RUN_REVIEW_STATUSES = {"pending", "leased", "done", "blocked", "needs_repair", "technical_failed"}


def create_extraction_batch(
    *,
    board_store: BoardStore,
    run_id: str,
    batch_id: str,
    source_task_id: str,
    raw_artifact_ids: list[str],
    extractor_task_ids: list[str],
    purpose: str,
    issue_key: str,
    priority: int,
) -> None:
    board_store.write_plan(
        run_id,
        title=f"Extraction batch {batch_id}",
        plan_kind=EXTRACTION_BATCH_PLAN_KIND,
        status="collecting",
        priority=priority,
        created_by="researcher",
        payload={
            "batch_id": batch_id,
            "source_task_id": source_task_id,
            "raw_artifact_ids": list(raw_artifact_ids),
            "extractor_task_ids": list(extractor_task_ids),
            "purpose": purpose,
            "issue_key": issue_key,
            "created_at": now_iso(),
            "expected_count": len(extractor_task_ids),
        },
        dedupe_key=f"extraction_batch:{batch_id}",
    )


def synchronize_extraction_batches(
    store: Store,
    run_id: str,
    *,
    timeout_seconds: int = EXTRACTION_BATCH_TIMEOUT_SECONDS,
) -> int:
    board_store = BoardStore(store)
    updated_batches = 0
    for item in store.list_swarm_board_items(run_id, kind="plan"):
        payload = dict(item.payload or {})
        if payload.get("plan_kind") != EXTRACTION_BATCH_PLAN_KIND:
            continue
        if item.status in {"ready_for_review", "partial_ready", "reviewing", "reviewed", "exhausted"}:
            continue
        batch_id = str(payload.get("batch_id") or "")
        extractor_task_ids = [str(value) for value in list(payload.get("extractor_task_ids") or []) if str(value)]
        if not batch_id or not extractor_task_ids:
            continue

        statuses = _task_statuses(store, extractor_task_ids)
        evidence_ids = _evidence_ids_for_tasks(store, run_id, extractor_task_ids)
        all_terminal = bool(statuses) and all(status in TERMINAL_EXTRACTION_STATUSES for status in statuses.values())
        timed_out = _batch_age_seconds(payload.get("created_at")) >= timeout_seconds
        partial_ready = timed_out and bool(evidence_ids)
        exhausted = timed_out and not evidence_ids

        if not all_terminal and not partial_ready and not exhausted:
            continue
        if exhausted:
            board_store.update_status(item.item_id or "", status="exhausted")
            updated_batches += 1
            continue
        next_status = "partial_ready" if partial_ready and not all_terminal else "ready_for_review"
        board_store.update_status(item.item_id or "", status=next_status)
        updated_batches += 1
    return updated_batches


def synchronize_run_evidence_review(store: Store, run_id: str) -> int:
    batches = _extraction_batch_items(store, run_id)
    reviewable_batches = [item for item in batches if item.status in {"ready_for_review", "partial_ready"}]
    if not reviewable_batches:
        return 0
    if not _run_review_gate_open(store, run_id, batches):
        return 0

    evidence_ids = sorted(
        str(item.evidence_id or "")
        for item in store.list_swarm_evidence(run_id, qa_state="ready")
        if item.evidence_id
    )
    if not evidence_ids:
        return 0

    bundle_key = f"run:root:{'|'.join(evidence_ids)}"
    if _run_review_exists(store, run_id, bundle_key):
        return 0

    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    batch_statuses = {
        str(item.payload.get("batch_id") or item.item_id): item.status
        for item in batches
        if dict(item.payload or {}).get("plan_kind") == EXTRACTION_BATCH_PLAN_KIND
    }
    review_task = task_store.create(
        run_id,
        kind="evidence_review",
        status="pending",
        owner_role="critic",
        inputs={
            "evidence_scope": "run",
            "batch_ids": [str(item.payload.get("batch_id") or "") for item in reviewable_batches],
            "batch_statuses": batch_statuses,
            "partial_bundle": any(item.status == "partial_ready" for item in reviewable_batches),
            "evidence_ids": evidence_ids,
            "evidence_bundle_key": bundle_key,
            "question": store.get_swarm_run_state(run_id).objective,
            "issue_key": "",
        },
        depends_on=[],
        priority=max((item.priority for item in reviewable_batches), default=0),
        created_by="evidence_review_gate",
    )
    mailbox.send(
        run_id,
        from_role="evidence_review_gate",
        to_role="critic",
        message_type="request",
        payload={
            "kind": "review_evidence",
            "task_id": review_task.task_id,
            "evidence_scope": "run",
            "batch_ids": [str(item.payload.get("batch_id") or "") for item in reviewable_batches],
            "evidence_ids": evidence_ids,
            "partial_bundle": any(item.status == "partial_ready" for item in reviewable_batches),
        },
        related_task_id=review_task.task_id,
    )
    return 1


def _extraction_batch_items(store: Store, run_id: str) -> list[Any]:
    return [
        item
        for item in store.list_swarm_board_items(run_id, kind="plan")
        if dict(item.payload or {}).get("plan_kind") == EXTRACTION_BATCH_PLAN_KIND
    ]


def _run_review_gate_open(store: Store, run_id: str, batches: list[Any]) -> bool:
    if batches and all(item.status in {"ready_for_review", "partial_ready", "reviewing", "reviewed", "exhausted"} for item in batches):
        return True
    return not any(
        task.status in {"pending", "leased"}
        for task in store.list_swarm_tasks(run_id, owner_role="extractor")
        if task.kind == "raw_document"
    )


def _run_review_exists(store: Store, run_id: str, bundle_key: str) -> bool:
    for task in store.list_swarm_tasks(run_id, owner_role="critic"):
        if task.kind != "evidence_review":
            continue
        if task.status not in EXISTING_RUN_REVIEW_STATUSES:
            continue
        if str(task.inputs.get("evidence_scope") or "") != "run":
            continue
        if str(task.inputs.get("evidence_bundle_key") or "") == bundle_key:
            return True
    return False


def _task_statuses(store: Store, task_ids: list[str]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for task_id in task_ids:
        try:
            statuses[task_id] = store.get_swarm_task(task_id).status
        except KeyError:
            statuses[task_id] = "missing"
    return statuses


def _evidence_ids_for_tasks(store: Store, run_id: str, extractor_task_ids: list[str]) -> list[str]:
    task_id_set = set(extractor_task_ids)
    evidence_ids = [
        item.evidence_id or ""
        for item in store.list_swarm_board_items(run_id, kind="evidence")
        if item.source_task_id in task_id_set and item.evidence_id
    ]
    return sorted(set(evidence_ids))


def _batch_age_seconds(created_at: Any) -> float:
    try:
        created = datetime.fromisoformat(str(created_at))
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        return (datetime.now(UTC) - created).total_seconds()
    except (TypeError, ValueError):
        return 0.0
