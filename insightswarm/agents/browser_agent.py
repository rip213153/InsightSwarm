from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
from typing import Any

from insightswarm.schemas.swarm import Task
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore


class HumanAuthorizationRequired(RuntimeError):
    pass


HIGH_RISK_MARKERS = {
    "login",
    "log in",
    "log into",
    "pay",
    "payment",
    "upload",
    "download",
    "cookie",
    "storage",
    "header",
    "headers",
    "password",
    "token",
    "javascript",
    "js",
}


def execute_browser_goal(goal: str) -> dict:
    lowered = goal.lower()
    for marker in HIGH_RISK_MARKERS:
        if marker in lowered:
            raise HumanAuthorizationRequired(f"High-risk browser action requires approval: {marker}")
    return {"status": "ok", "goal": goal}


@dataclass(frozen=True)
class BrowserWorkResult:
    claimed_task_id: str
    created_task_ids: list[str] = field(default_factory=list)
    created_message_ids: list[str] = field(default_factory=list)
    created_artifact_ids: list[str] = field(default_factory=list)


class BrowserWorker:
    def __init__(self, task_store: TaskStore, mailbox: Mailbox, artifact_store: ArtifactStore, board_store: BoardStore | None = None):
        self.task_store = task_store
        self.mailbox = mailbox
        self.artifact_store = artifact_store
        self.board_store = board_store or BoardStore(task_store.store)

    def run_once(self, run_id: str) -> BrowserWorkResult | None:
        task = self.task_store.claim_next(run_id, owner_role="browser_agent")
        if task is None:
            return None
        result = self._process_task(task)
        current = self.task_store.store.get_swarm_task(task.task_id)
        if current.status in {"pending", "leased"}:
            self.task_store.complete(task.task_id)
        return result

    def run_forever(
        self,
        run_id: str,
        stop_event: Event,
        *,
        poll_interval: float = 0.2,
        max_iterations: int | None = None,
    ) -> list[BrowserWorkResult]:
        results: list[BrowserWorkResult] = []

        while not stop_event.is_set():
            result = self.run_once(run_id)
            if result is None:
                stop_event.wait(poll_interval)
                continue
            results.append(result)
            if max_iterations is not None and len(results) >= max_iterations:
                break

        return results

    def _process_task(self, task: Task) -> BrowserWorkResult:
        goal = str(task.inputs.get("goal") or "").strip()
        board_item_id = str(task.inputs.get("board_item_id") or "").strip() or None
        if _is_high_risk(goal):
            message = self.mailbox.send(
                task.run_id,
                from_role="browser_agent",
                to_role="lead",
                message_type="observation",
                payload={"kind": "authorization_request", "task_id": task.task_id, "goal": goal},
                related_task_id=task.task_id,
            )
            if board_item_id:
                self.board_store.update_status(board_item_id, status="blocked")
            self.task_store.block(task.task_id)
            return BrowserWorkResult(claimed_task_id=task.task_id, created_message_ids=[message.message_id])

        raw_document = {
            "url": task.inputs.get("url") or "browser://captured",
            "goal": goal,
            "text": f"Raw browser capture for: {goal}",
            "html": f"<html><body>{goal}</body></html>",
        }
        artifact = self.artifact_store.write_raw_document(
            task.run_id,
            source_task_id=task.task_id,
            document=raw_document,
            summary=f"Raw browser capture for {goal}",
        )
        extractor_task = self.task_store.create(
            task.run_id,
            kind="raw_document",
            status="pending",
            owner_role="extractor",
            inputs={"artifact_id": artifact.artifact_id, "source_task_id": task.task_id, "board_item_id": board_item_id},
            depends_on=[task.task_id],
            priority=task.priority,
            created_by="browser_agent",
        )
        handoff = self.mailbox.send(
            task.run_id,
            from_role="browser_agent",
            to_role="extractor",
            message_type="request",
            payload={
                "kind": "extract_evidence",
                "task_id": extractor_task.task_id,
                "artifact_id": artifact.artifact_id,
                "related_artifact_id": artifact.artifact_id,
            },
            related_task_id=extractor_task.task_id,
        )
        if board_item_id:
            self.board_store.update_status(board_item_id, status="active")
        return BrowserWorkResult(
            claimed_task_id=task.task_id,
            created_task_ids=[extractor_task.task_id],
            created_message_ids=[handoff.message_id],
            created_artifact_ids=[artifact.artifact_id],
        )


def _is_high_risk(goal: str) -> bool:
    lowered = goal.lower()
    return any(marker in lowered for marker in HIGH_RISK_MARKERS)
