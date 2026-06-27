from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any

from insightswarm.agents.agent_loop import AgentLoopState, run_agent_loop
from insightswarm.agents.execution_cell import run_in_cell, run_supervised_once
from insightswarm.agents.extractor_tools import CRITIC_ROLE, EXTRACTOR_ROLE, EXTRACTOR_TOOLS, ExtractorToolHandlers, ExtractorToolState
from insightswarm.agents.failure_policy import normalize_agent_failure
from insightswarm.agents.tool_executor import ToolExecutor
from insightswarm.agents.trace import build_tool_trace_callback
from insightswarm.event_bus import EventBus
from insightswarm.schemas.swarm import Task
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore


@dataclass(frozen=True)
class ExtractorResult:
    claimed_task_id: str
    created_artifact_ids: list[str] = field(default_factory=list)
    created_evidence_ids: list[str] = field(default_factory=list)
    created_message_ids: list[str] = field(default_factory=list)
    created_task_ids: list[str] = field(default_factory=list)
    terminal_status: str | None = None
    terminal_reason: str | None = None


class Extractor:
    def __init__(self, task_store: TaskStore, mailbox: Mailbox, artifact_store: ArtifactStore, board_store: BoardStore | None = None):
        self.task_store = task_store
        self.mailbox = mailbox
        self.artifact_store = artifact_store
        self.board_store = board_store or BoardStore(task_store.store)

    def run_once(
        self,
        run_id: str,
        *,
        model_client: object | None = None,
        trace_path: Path | None = None,
        run_root: Path | None = None,
    ) -> ExtractorResult | None:
        task = self.task_store.claim_next(run_id, owner_role=EXTRACTOR_ROLE)
        if task is None:
            return None
        return run_in_cell(
            task_store=self.task_store,
            mailbox=self.mailbox,
            task=task,
            role=EXTRACTOR_ROLE,
            run_root=run_root,
            body=lambda claimed: self.run_task(claimed, model_client=model_client, trace_path=trace_path),
            make_failure_result=lambda failed: ExtractorResult(claimed_task_id=failed.task_id or ""),
        )

    def run_forever(
        self,
        run_id: str,
        stop_event: Event,
        *,
        poll_interval: float = 0.2,
        model_client: object | None = None,
        max_iterations: int | None = None,
        trace_path: Path | None = None,
        run_root: Path | None = None,
        event_bus: EventBus | None = None,
    ) -> list[ExtractorResult]:
        results: list[ExtractorResult] = []
        while not stop_event.is_set():
            result = run_supervised_once(
                stop_event=stop_event,
                poll_interval=poll_interval,
                call_once=lambda: self.run_once(
                    run_id, model_client=model_client, trace_path=trace_path, run_root=run_root
                ),
            )
            if result is None:
                # Push-based wake: block on the role's condition until a notify
                # (task created / message sent) or the fallback timeout. The
                # runtime calls event_bus.notify_all_roles on teardown so a
                # blocked wait wakes within one notify of stop_event.
                if event_bus is not None:
                    event_bus.wait(EXTRACTOR_ROLE, timeout=poll_interval)
                else:
                    stop_event.wait(poll_interval)
                continue
            results.append(result)
            if max_iterations is not None and len(results) >= max_iterations:
                break
        return results

    def run_probe(
        self,
        task: Task,
        *,
        model_client: object | None = None,
        safety_cap: int = 20,
        on_tool_result: Any | None = None,
    ) -> list[dict[str, Any]]:
        tool_state = ExtractorToolState()
        loop_state = AgentLoopState()
        trace, _ = self._run_loop(
            task,
            model_client=model_client,
            tool_state=tool_state,
            loop_state=loop_state,
            safety_cap=safety_cap,
            on_tool_result=on_tool_result,
        )
        return trace

    def run_task(
        self,
        task: Task,
        *,
        model_client: object | None = None,
        safety_cap: int = 20,
        trace_path: Path | None = None,
    ) -> ExtractorResult:
        tool_state = ExtractorToolState()
        loop_state = AgentLoopState()
        self._run_loop(
            task,
            model_client=model_client,
            tool_state=tool_state,
            loop_state=loop_state,
            safety_cap=safety_cap,
            on_tool_result=build_tool_trace_callback(trace_path, role=EXTRACTOR_ROLE, task=task),
        )
        terminal_status = tool_state.terminal_status or loop_state.terminal_status
        if terminal_status and terminal_status != "done" and not tool_state.created_evidence_ids:
            self._handle_failed_extraction(task, tool_state, loop_state, terminal_status)
        if terminal_status and terminal_status != "done":
            self.task_store.block(task.task_id or "")
        else:
            self.task_store.complete(task.task_id or "")
        return ExtractorResult(
            claimed_task_id=task.task_id or "",
            created_artifact_ids=list(tool_state.created_artifact_ids),
            created_evidence_ids=list(tool_state.created_evidence_ids),
            created_message_ids=list(tool_state.created_message_ids),
            created_task_ids=list(tool_state.created_task_ids),
            terminal_status=terminal_status,
            terminal_reason=tool_state.terminal_reason or loop_state.terminal_reason,
        )

    def _run_loop(
        self,
        task: Task,
        *,
        model_client: object | None,
        tool_state: ExtractorToolState,
        loop_state: AgentLoopState,
        safety_cap: int,
        on_tool_result: Any | None = None,
    ) -> tuple[list[dict[str, Any]], AgentLoopState]:
        handlers = ExtractorToolHandlers(
            task=task,
            task_store=self.task_store,
            mailbox=self.mailbox,
            artifact_store=self.artifact_store,
            board_store=self.board_store,
            state=tool_state,
        )
        executor = ToolExecutor(EXTRACTOR_TOOLS, handlers.handlers())
        return run_agent_loop(
            model_client=model_client,
            system_prompt=_extractor_prompt(),
            tool_specs=EXTRACTOR_TOOLS,
            executor=executor,
            initial_user_payload={
                "assigned_task": {
                    "task_id": task.task_id,
                    "kind": task.kind,
                    "owner_role": task.owner_role,
                    "run_id": task.run_id,
                    "artifact_id": task.inputs.get("artifact_id"),
                },
                "instruction": "Extract exact quote-backed citations from the raw document. Use tools only.",
            },
            state=loop_state,
            safety_cap=safety_cap,
            max_tokens=3200,
            metadata_role="extractor_tool_loop",
            metadata={
                "run_id": task.run_id,
                "task_id": task.task_id,
                "operation": "extractor_tool_loop",
                "artifact_id": task.inputs.get("artifact_id"),
            },
            on_tool_result=on_tool_result,
        )


    def _handle_failed_extraction(
        self,
        task: Task,
        tool_state: ExtractorToolState,
        loop_state: AgentLoopState,
        terminal_status: str,
    ) -> None:
        failure = normalize_agent_failure(
            status=terminal_status,
            reason=tool_state.terminal_reason or loop_state.terminal_reason,
        )
        if failure.category == "technical":
            if failure.retryable and self._retry_extraction(task, tool_state, failure.reason):
                return
            self._record_technical_failure(task, tool_state, failure.reason)
            return
        if not failure.should_trigger_critic_review:
            self._record_technical_failure(task, tool_state, failure.reason)
            return

        artifact_id = tool_state.artifact_id or str(task.inputs.get("artifact_id") or "")
        if not artifact_id:
            return
        reason = failure.reason or "Extractor stopped before creating citation-backed evidence."
        issue_key = _issue_key_for_failed_extraction(task, tool_state.raw_document or {})
        review_task = self.task_store.create(
            task.run_id,
            kind="extraction_failure_review",
            status="pending",
            owner_role=CRITIC_ROLE,
            inputs={
                "targeted_query": reason,
                "source_artifact_id": artifact_id,
                "failure_reason": reason,
                "extractor_task_id": task.task_id,
                "question": self.task_store.store.get_swarm_run_state(task.run_id).objective,
                "issue_key": issue_key,
            },
            depends_on=[],
            priority=task.priority,
            created_by=EXTRACTOR_ROLE,
        )
        message = self.mailbox.send(
            task.run_id,
            from_role=EXTRACTOR_ROLE,
            to_role=CRITIC_ROLE,
            message_type="request",
            payload={
                "kind": "review_extraction_failure",
                "task_id": review_task.task_id,
                "artifact_id": artifact_id,
                "failure_reason": reason,
                "issue_key": issue_key,
            },
            related_task_id=review_task.task_id,
        )
        tool_state.created_task_ids.append(review_task.task_id or "")
        tool_state.created_message_ids.append(message.message_id or "")

    def _retry_extraction(self, task: Task, tool_state: ExtractorToolState, reason: str) -> bool:
        artifact_id = tool_state.artifact_id or str(task.inputs.get("artifact_id") or "")
        if not artifact_id:
            return False
        attempt = _safe_int(task.inputs.get("extraction_attempt"), default=1)
        max_attempts = _safe_int(task.inputs.get("max_extraction_attempts"), default=2)
        if attempt >= max_attempts:
            return False
        retry_task = self.task_store.create(
            task.run_id,
            kind="raw_document",
            status="pending",
            owner_role=EXTRACTOR_ROLE,
            inputs={
                **dict(task.inputs or {}),
                "artifact_id": artifact_id,
                "extraction_attempt": attempt + 1,
                "max_extraction_attempts": max_attempts,
                "retry_of_task_id": task.task_id,
                "retry_reason": reason,
            },
            depends_on=[],
            priority=task.priority,
            created_by=EXTRACTOR_ROLE,
        )
        message = self.mailbox.send(
            task.run_id,
            from_role=EXTRACTOR_ROLE,
            to_role=EXTRACTOR_ROLE,
            message_type="request",
            payload={
                "kind": "extract_evidence",
                "artifact_id": artifact_id,
                "task_id": retry_task.task_id,
                "technical_retry": True,
                "attempt": attempt + 1,
            },
            related_task_id=retry_task.task_id,
        )
        tool_state.created_task_ids.append(retry_task.task_id or "")
        tool_state.created_message_ids.append(message.message_id or "")
        return True

    def _record_technical_failure(self, task: Task, tool_state: ExtractorToolState, reason: str) -> None:
        artifact_id = tool_state.artifact_id or str(task.inputs.get("artifact_id") or "")
        issue_key = _issue_key_for_failed_extraction(task, tool_state.raw_document or {})
        message = self.mailbox.send(
            task.run_id,
            from_role=EXTRACTOR_ROLE,
            broadcast=True,
            message_type="observation",
            payload={
                "kind": "technical_failure",
                "role": EXTRACTOR_ROLE,
                "task_id": task.task_id,
                "task_kind": task.kind,
                "status": "technical_failure",
                "failure_category": "technical",
                "artifact_id": artifact_id,
                "issue_key": issue_key,
                "reason": reason or "technical extraction failure",
                "retryable": False,
                "should_trigger_research_repair": False,
            },
            related_task_id=task.task_id,
        )
        tool_state.created_message_ids.append(message.message_id or "")


def _extractor_prompt() -> str:
    return (Path(__file__).resolve().parent.parent / "prompts" / "extractor.md").read_text(encoding="utf-8")


def _issue_key_for_failed_extraction(task: Task, document: dict[str, Any]) -> str:
    task_issue_key = str(task.inputs.get("issue_key") or "").strip()
    if task_issue_key:
        return task_issue_key
    metadata = document.get("metadata") if isinstance(document, dict) else {}
    if isinstance(metadata, dict):
        metadata_issue_key = str(metadata.get("issue_key") or "").strip()
        if metadata_issue_key:
            return metadata_issue_key
    return str(document.get("issue_key") or "").strip()


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
