from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Literal

from insightswarm.authorization_flow import pending_authorization_requests
from insightswarm.db.store import Store
from insightswarm.swarm_store import BoardStore, Mailbox, TaskStore


GateStatus = Literal["open", "closed", "blocked"]
REPAIR_PRIORITY_THRESHOLD = 8
MATERIAL_RESEARCH_ROLES = {"researcher", "extractor", "browser_agent"}
MATERIAL_RESEARCH_KINDS = {
    "research_subquestion",
    "research_repair",
    "raw_document",
    "hard_acquisition",
}


@dataclass(frozen=True)
class DeliveryGateDecision:
    status: GateStatus
    reasons: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    critic_verdict: str | None = None
    frontier_hash: str = ""

    @property
    def is_open(self) -> bool:
        return self.status == "open"


def evaluate_delivery_gate(store: Store, run_id: str) -> DeliveryGateDecision:
    _reconcile_passed_conflicts(store, run_id)
    evidence_rows = store.list_swarm_evidence(run_id, qa_state="ready")
    tasks = store.list_swarm_tasks(run_id)
    board_items = store.list_swarm_board_items(run_id)
    pending_repairs = [
        task
        for task in tasks
        if task.kind == "repair_request" and task.status in {"pending", "leased"} and task.priority >= REPAIR_PRIORITY_THRESHOLD
    ]
    pending_reviews = [
        task
        for task in tasks
        if task.kind in {"evidence_review", "extraction_failure_review"} and task.status in {"pending", "leased"}
    ]
    active_research_tasks = [
        task
        for task in tasks
        if task.owner_role in MATERIAL_RESEARCH_ROLES
        and task.kind in MATERIAL_RESEARCH_KINDS
        and task.status in {"pending", "leased"}
    ]
    open_repair_questions = [
        item
        for item in board_items
        if item.kind == "question"
        and item.status in {"open", "active", "blocked"}
        and str(item.payload.get("question_type") or "") in {"repair", "source_request"}
    ]
    open_conflicts = [
        item
        for item in board_items
        if item.kind == "conflict" and item.status in {"open", "active"}
    ]
    authorization_messages = pending_authorization_requests(store, run_id)
    critic_pass = _latest_critic_pass(store, run_id)
    critic_verdict = _latest_critic_verdict(store, run_id)
    pending_reviews_block_delivery = bool(pending_reviews) and not (
        critic_pass is not None and not active_research_tasks and not pending_repairs
    )
    delivery_evidence_ids = _accepted_frontier_evidence_ids(evidence_rows, critic_pass)

    reasons: list[str] = []
    status: GateStatus = "open"
    if not evidence_rows:
        status = "closed"
        reasons.append("citation-backed evidence is missing")
    if pending_reviews_block_delivery or (evidence_rows and critic_verdict is None):
        status = "closed"
        reasons.append("critic review is pending")
    if pending_repairs:
        status = "closed"
        reasons.append("high-priority repair_request is unresolved")
    if active_research_tasks:
        status = "closed"
        reasons.append("research tasks are still running")
    if open_repair_questions:
        status = "closed"
        reasons.append("repair/source questions are still open")
    if open_conflicts:
        status = "closed"
        reasons.append("board conflicts are still open")
    if authorization_messages:
        status = "blocked"
        reasons.append("authorization_request is pending")
    if critic_verdict == "block":
        status = "blocked"
        reasons.append("latest critic verdict is block")

    return DeliveryGateDecision(
        status=status,
        reasons=reasons,
        evidence_ids=delivery_evidence_ids,
        critic_verdict=critic_verdict,
        frontier_hash=_frontier_hash(delivery_evidence_ids),
    )


def synchronize_delivery_gate(store: Store, run_id: str) -> DeliveryGateDecision:
    decision = evaluate_delivery_gate(store, run_id)
    store.update_swarm_run_state(
        run_id,
        phase="delivery" if decision.is_open else None,
        delivery_gate=decision.is_open,
    )
    if decision.is_open:
        _ensure_delivery_request(store, run_id, decision)
    return decision


def _ensure_delivery_request(store: Store, run_id: str, decision: DeliveryGateDecision) -> None:
    existing = [
        task
        for task in store.list_swarm_tasks(run_id)
        if task.owner_role == "writer" and task.kind == "delivery_request" and task.status in {"pending", "leased", "done"}
    ]
    task_store = TaskStore(store)
    for task in existing:
        task_frontier_hash = str(task.inputs.get("frontier_hash") or "")
        if task_frontier_hash == decision.frontier_hash:
            return
        if task.status in {"pending", "leased"}:
            task_store.block(task.task_id or "")
    run_state = store.get_swarm_run_state(run_id)
    mailbox = Mailbox(store)
    delivery_task = task_store.create(
        run_id,
        kind="delivery_request",
        status="pending",
        owner_role="writer",
        inputs={
            "question": run_state.objective,
            "evidence_ids": list(decision.evidence_ids),
            "frontier_hash": decision.frontier_hash,
            "report_kind": _report_kind_for_decision(store, run_id, decision),
        },
        priority=10,
        created_by="delivery_gate",
    )
    mailbox.send(
        run_id,
        from_role="delivery_gate",
        to_role="writer",
        message_type="request",
        payload={
            "kind": "delivery_request",
            "task_id": delivery_task.task_id,
            "evidence_ids": list(decision.evidence_ids),
            "frontier_hash": decision.frontier_hash,
            "report_kind": delivery_task.inputs.get("report_kind"),
        },
        related_task_id=delivery_task.task_id,
    )


def _reconcile_passed_conflicts(store: Store, run_id: str) -> None:
    board_store = BoardStore(store)
    for message in store.list_swarm_messages(run_id):
        if message.from_role != "critic":
            continue
        payload = dict(message.payload or {})
        if payload.get("verdict") not in {"pass", "pass_with_caveats"} and payload.get("kind") not in {"pass", "pass_with_caveats"}:
            continue
        evidence_ids = [str(value) for value in list(payload.get("evidence_ids") or []) if str(value)]
        issue_keys = [str(payload.get("issue_key") or "").strip()]
        issue_keys.extend(board_store.issue_keys_for_evidence(run_id, evidence_ids))
        board_store.resolve_conflicts(
            run_id,
            issue_keys=[key for key in issue_keys if key],
            evidence_ids=evidence_ids,
            resolved_by="critic",
            reason=str(payload.get("reason") or "Critic pass reconciled stale conflict state."),
            resolution_event_at=message.created_at,
        )


def _latest_critic_verdict(store: Store, run_id: str) -> str | None:
    payload = store.conn.execute(
        """
        SELECT payload_json
        FROM swarm_messages
        WHERE run_id = ? AND from_role = 'critic'
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    if payload is None:
        return None

    return str(json.loads(payload["payload_json"]).get("verdict") or "")


def _latest_critic_pass(store: Store, run_id: str) -> dict[str, object] | None:
    for message in reversed(store.list_swarm_messages(run_id)):
        if message.from_role != "critic":
            continue
        payload = dict(message.payload or {})
        verdict = str(payload.get("verdict") or payload.get("kind") or "")
        if verdict in {"pass", "pass_with_caveats"}:
            return {
                "created_at": message.created_at,
                "evidence_ids": [str(value) for value in list(payload.get("evidence_ids") or []) if str(value)],
            }
    return None


def _accepted_frontier_evidence_ids(evidence_rows: list[object], critic_pass: dict[str, object] | None) -> list[str]:
    ready_ids = [str(getattr(row, "evidence_id", "") or "") for row in evidence_rows]
    ready_id_set = set(ready_ids)
    if not critic_pass:
        return ready_ids
    passed_ids = [str(value) for value in list(critic_pass.get("evidence_ids") or []) if str(value)]
    frontier_ids = [evidence_id for evidence_id in passed_ids if evidence_id in ready_id_set]
    return frontier_ids or ready_ids


def _frontier_hash(evidence_ids: list[str]) -> str:
    normalized = "\n".join(sorted(str(evidence_id) for evidence_id in evidence_ids if str(evidence_id)))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _report_kind_for_decision(store: Store, run_id: str, decision: DeliveryGateDecision) -> str:
    has_delivery_gap = any(
        message.type == "observation" and str(message.payload.get("kind") or "") == "delivery_gap"
        for message in store.list_swarm_messages(run_id)
    )
    if has_delivery_gap or decision.critic_verdict != "pass":
        return "report_partial"
    return "report"
