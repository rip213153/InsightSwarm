from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from insightswarm.db.store import Store
from insightswarm.schemas.swarm import Message
from insightswarm.swarm_store import Mailbox, TaskStore


AuthorizationDecision = Literal["allow", "deny"]


@dataclass(frozen=True)
class PendingAuthorization:
    message_id: str
    task_id: str
    goal: str
    reason: str


def pending_authorization_requests(store: Store, run_id: str) -> list[PendingAuthorization]:
    decisions = _authorization_decisions_by_task(store, run_id)
    pending: list[PendingAuthorization] = []
    for message in _authorization_requests(store, run_id):
        task_id = str(message.payload.get("task_id") or message.related_task_id or "")
        if task_id and task_id in decisions:
            continue
        pending.append(
            PendingAuthorization(
                message_id=message.message_id or "",
                task_id=task_id,
                goal=str(message.payload.get("goal") or ""),
                reason=str(message.payload.get("reason") or ""),
            )
        )
    return pending


def authorization_decision_for_task(store: Store, run_id: str, task_id: str) -> AuthorizationDecision | None:
    return _authorization_decisions_by_task(store, run_id).get(task_id)


def write_authorization_decision(
    store: Store,
    run_id: str,
    *,
    task_id: str,
    decision: AuthorizationDecision,
    reason: str,
) -> None:
    Mailbox(store).send(
        run_id,
        from_role="operator",
        to_role="browser_agent",
        message_type="observation",
        payload={
            "kind": "authorization_decision",
            "task_id": task_id,
            "decision": decision,
            "reason": reason,
        },
        related_task_id=task_id,
    )
    if decision == "allow":
        TaskStore(store).update_status(task_id, status="pending", lease_until=None)


def _authorization_requests(store: Store, run_id: str) -> list[Message]:
    return [
        message
        for message in store.list_swarm_messages(run_id)
        if message.type == "observation" and str(message.payload.get("kind") or "") == "authorization_request"
    ]


def _authorization_decisions_by_task(store: Store, run_id: str) -> dict[str, AuthorizationDecision]:
    decisions: dict[str, AuthorizationDecision] = {}
    for message in store.list_swarm_messages(run_id):
        if message.type != "observation" or str(message.payload.get("kind") or "") != "authorization_decision":
            continue
        task_id = str(message.payload.get("task_id") or message.related_task_id or "")
        decision = str(message.payload.get("decision") or "")
        if task_id and decision in {"allow", "deny"}:
            decisions[task_id] = decision  # latest message wins because store order is chronological.
    return decisions
