from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any

from insightswarm.agents.agent_loop import AgentLoopState, run_agent_loop
from insightswarm.agents.agent_failure_reporting import record_agent_technical_failure
from insightswarm.agents.browser_agent_tools import (
    BROWSER_AGENT_TOOLS,
    BROWSER_ROLE,
    BrowserAgentToolHandlers,
    BrowserAgentToolState,
    HumanAuthorizationRequired,
)
from insightswarm.agents.tool_executor import ToolExecutor
from insightswarm.agents.trace import build_tool_trace_callback
from insightswarm.browser_code_session import BrowserCodeSession
from insightswarm.schemas.swarm import Task
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore


@dataclass(frozen=True)
class BrowserWorkResult:
    claimed_task_id: str
    created_task_ids: list[str] = field(default_factory=list)
    created_message_ids: list[str] = field(default_factory=list)
    created_artifact_ids: list[str] = field(default_factory=list)
    terminal_status: str | None = None
    terminal_reason: str | None = None


class BrowserWorker:
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
    ) -> BrowserWorkResult | None:
        task = self.task_store.claim_next(run_id, owner_role=BROWSER_ROLE)
        if task is None:
            return None
        return self.run_task(task, model_client=model_client, trace_path=trace_path)

    def run_forever(
        self,
        run_id: str,
        stop_event: Event,
        *,
        poll_interval: float = 0.2,
        max_iterations: int | None = None,
        model_client: object | None = None,
        trace_path: Path | None = None,
    ) -> list[BrowserWorkResult]:
        results: list[BrowserWorkResult] = []

        while not stop_event.is_set():
            result = self.run_once(run_id, model_client=model_client, trace_path=trace_path)
            if result is None:
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
        tool_state = BrowserAgentToolState()
        loop_state = AgentLoopState()
        trace, _ = self._run_loop(
            task,
            model_client=model_client,
            tool_state=tool_state,
            loop_state=loop_state,
            safety_cap=safety_cap,
            on_tool_result=on_tool_result,
        )
        if tool_state.browser_session is not None:
            tool_state.browser_session.close()
        return trace

    def run_task(
        self,
        task: Task,
        *,
        model_client: object | None = None,
        safety_cap: int = 20,
        trace_path: Path | None = None,
    ) -> BrowserWorkResult:
        tool_state = BrowserAgentToolState()
        loop_state = AgentLoopState()
        try:
            if model_client is None:
                self._run_fallback(task, tool_state)
            else:
                self._run_code_session(
                    task,
                    model_client=model_client,
                    tool_state=tool_state,
                    safety_cap=safety_cap,
                    trace_path=trace_path,
                )
        finally:
            if tool_state.browser_session is not None:
                tool_state.browser_session.close()

        terminal_status = tool_state.terminal_status or loop_state.terminal_status
        if terminal_status and terminal_status != "done":
            terminal_reason = tool_state.terminal_reason or loop_state.terminal_reason
            if "authorization" not in str(terminal_reason or "").lower():
                message_id = record_agent_technical_failure(
                    mailbox=self.mailbox,
                    role=BROWSER_ROLE,
                    task=task,
                    status=terminal_status,
                    reason=terminal_reason,
                )
                if message_id:
                    tool_state.created_message_ids.append(message_id)
            self.task_store.block(task.task_id or "")
        else:
            current = self.task_store.store.get_swarm_task(task.task_id)
            if current.status in {"pending", "leased"}:
                self.task_store.complete(task.task_id or "")

        return BrowserWorkResult(
            claimed_task_id=task.task_id or "",
            created_task_ids=list(tool_state.created_task_ids),
            created_message_ids=list(tool_state.created_message_ids),
            created_artifact_ids=list(tool_state.created_artifact_ids),
            terminal_status=terminal_status,
            terminal_reason=tool_state.terminal_reason or loop_state.terminal_reason,
        )

    def _run_code_session(
        self,
        task: Task,
        *,
        model_client: object,
        tool_state: BrowserAgentToolState,
        safety_cap: int,
        trace_path: Path | None,
    ) -> None:
        handlers = BrowserAgentToolHandlers(
            task=task,
            task_store=self.task_store,
            mailbox=self.mailbox,
            artifact_store=self.artifact_store,
            board_store=self.board_store,
            state=tool_state,
            model_client=model_client,
        )
        BrowserCodeSession(
            task=task,
            handlers=handlers,
            tool_state=tool_state,
            model_client=model_client,
            trace_path=trace_path,
            max_cells=safety_cap,
        ).run()

    def _run_loop(
        self,
        task: Task,
        *,
        model_client: object | None,
        tool_state: BrowserAgentToolState,
        loop_state: AgentLoopState,
        safety_cap: int,
        on_tool_result: Any | None = None,
    ) -> tuple[list[dict[str, Any]], AgentLoopState]:
        handlers = BrowserAgentToolHandlers(
            task=task,
            task_store=self.task_store,
            mailbox=self.mailbox,
            artifact_store=self.artifact_store,
            board_store=self.board_store,
            state=tool_state,
            model_client=model_client,
        )
        executor = ToolExecutor(BROWSER_AGENT_TOOLS, handlers.handlers())
        return run_agent_loop(
            model_client=model_client,
            system_prompt=_browser_prompt(),
            tool_specs=BROWSER_AGENT_TOOLS,
            executor=executor,
            initial_user_payload={
                "assigned_task": {
                    "task_id": task.task_id,
                    "kind": task.kind,
                    "owner_role": task.owner_role,
                    "run_id": task.run_id,
                },
                "instruction": "Acquire the requested public browser-only source. Use restricted browser code, publish usable raw source, then finish.",
            },
            state=loop_state,
            safety_cap=safety_cap,
            max_tokens=2600,
            metadata_role="browser_agent_tool_loop",
            metadata={
                "run_id": task.run_id,
                "task_id": task.task_id,
                "operation": "browser_agent_tool_loop",
            },
            on_tool_result=on_tool_result,
        )

    def _run_fallback(self, task: Task, tool_state: BrowserAgentToolState) -> None:
        handlers = BrowserAgentToolHandlers(
            task=task,
            task_store=self.task_store,
            mailbox=self.mailbox,
            artifact_store=self.artifact_store,
            board_store=self.board_store,
            state=tool_state,
        )
        handlers.read_task({})
        goal = str(task.inputs.get("goal") or "").strip()
        target_url = str(task.inputs.get("target_url") or task.inputs.get("url") or "").strip()
        if _is_high_risk(goal):
            handlers.execute_browser_code(
                {
                    "why_this_code": "Fallback high-risk guard.",
                    "code": "request_authorization('high-risk browser goal requires approval')",
                }
            )
            return
        text = (
            f"Raw browser capture for: {goal}. "
            f"This fallback capture represents public visible text acquired for the browser-only task. "
            f"Target URL: {target_url or 'browser://captured'}."
        )
        handlers.execute_browser_code(
            {
                "why_this_code": "Fallback for tests without a model client.",
                "code": (
                    "publish_raw_source("
                    f"{text!r}, "
                    f"url={(target_url or 'browser://captured')!r}, "
                    f"title={'Browser capture'!r}, "
                    "why_ready='fallback no-model browser capture')\n"
                    "finish_browser('complete', 'fallback browser acquisition completed')"
                ),
            }
        )


def _browser_prompt() -> str:
    return (Path(__file__).resolve().parent.parent / "prompts" / "browser_agent.md").read_text(encoding="utf-8")


def _is_high_risk(goal: str) -> bool:
    lowered = goal.lower()
    return any(marker in lowered for marker in ("login", "log in", "pay", "payment", "upload", "download", "cookie", "storage", "header", "headers", "password", "token", "javascript", "js"))
