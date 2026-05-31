from __future__ import annotations

from insightswarm.agents.failure_policy import normalize_agent_failure
from insightswarm.schemas.swarm import Task
from insightswarm.swarm_store import Mailbox


def record_agent_technical_failure(
    *,
    mailbox: Mailbox,
    role: str,
    task: Task,
    status: str | None,
    reason: str | None,
) -> str | None:
    failure = normalize_agent_failure(status=status, reason=reason)
    if failure.category != "technical":
        return None
    message = mailbox.send(
        task.run_id,
        from_role=role,
        broadcast=True,
        message_type="observation",
        payload={
            "kind": "technical_failure",
            "role": role,
            "task_id": task.task_id,
            "task_kind": task.kind,
            "status": status,
            "reason": failure.reason,
            "retryable": failure.retryable,
            "should_trigger_research_repair": False,
            "issue_key": str(task.inputs.get("issue_key") or "").strip(),
        },
        related_task_id=task.task_id,
    )
    return message.message_id
