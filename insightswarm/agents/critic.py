from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any

from insightswarm.agents.agent_loop import AgentLoopState, run_agent_loop
from insightswarm.agents.agent_failure_reporting import record_agent_technical_failure
from insightswarm.agents.critic_tools import CRITIC_ROLE, CRITIC_TOOLS, CriticToolHandlers, CriticToolState
from insightswarm.agents.tool_executor import ToolExecutor
from insightswarm.agents.trace import build_tool_trace_callback
from insightswarm.schemas.swarm import Task
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore


@dataclass(frozen=True)
class CriticResult:
    claimed_task_id: str
    created_task_ids: list[str] = field(default_factory=list)
    created_message_ids: list[str] = field(default_factory=list)
    created_board_item_ids: list[str] = field(default_factory=list)
    terminal_status: str | None = None
    terminal_reason: str | None = None


class Critic:
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
    ) -> CriticResult | None:
        task = self.task_store.claim_next(run_id, owner_role=CRITIC_ROLE)
        if task is None:
            return None
        return self.run_task(task, model_client=model_client, trace_path=trace_path)

    def run_forever(
        self,
        run_id: str,
        stop_event: Event,
        *,
        poll_interval: float = 0.2,
        model_client: object | None = None,
        max_iterations: int | None = None,
        trace_path: Path | None = None,
    ) -> list[CriticResult]:
        results: list[CriticResult] = []
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
        safety_cap: int = 5,
        on_tool_result: Any | None = None,
    ) -> list[dict[str, Any]]:
        tool_state = CriticToolState()
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
        safety_cap: int = 5,
        trace_path: Path | None = None,
    ) -> CriticResult:
        tool_state = CriticToolState()
        loop_state = AgentLoopState()
        self._run_loop(
            task,
            model_client=model_client,
            tool_state=tool_state,
            loop_state=loop_state,
            safety_cap=safety_cap,
            on_tool_result=build_tool_trace_callback(trace_path, role=CRITIC_ROLE, task=task),
        )
        terminal_status = tool_state.terminal_status or loop_state.terminal_status
        if terminal_status and terminal_status != "done":
            message_id = record_agent_technical_failure(
                mailbox=self.mailbox,
                role=CRITIC_ROLE,
                task=task,
                status=terminal_status,
                reason=tool_state.terminal_reason or loop_state.terminal_reason,
            )
            if message_id:
                tool_state.created_message_ids.append(message_id)
            self.task_store.block(task.task_id or "")
        else:
            self.task_store.complete(task.task_id or "")
        return CriticResult(
            claimed_task_id=task.task_id or "",
            created_task_ids=list(tool_state.created_task_ids),
            created_message_ids=list(tool_state.created_message_ids),
            created_board_item_ids=list(tool_state.created_board_item_ids),
            terminal_status=terminal_status,
            terminal_reason=tool_state.terminal_reason or loop_state.terminal_reason,
        )

    def _run_loop(
        self,
        task: Task,
        *,
        model_client: object | None,
        tool_state: CriticToolState,
        loop_state: AgentLoopState,
        safety_cap: int,
        on_tool_result: Any | None = None,
    ) -> tuple[list[dict[str, Any]], AgentLoopState]:
        handlers = CriticToolHandlers(
            task=task,
            task_store=self.task_store,
            mailbox=self.mailbox,
            artifact_store=self.artifact_store,
            board_store=self.board_store,
            state=tool_state,
        )
        executor = ToolExecutor(CRITIC_TOOLS, handlers.handlers())
        return run_agent_loop(
            model_client=model_client,
            system_prompt=_critic_prompt(),
            tool_specs=CRITIC_TOOLS,
            executor=executor,
            initial_user_payload={
                "assigned_task": {
                    "task_id": task.task_id,
                    "kind": task.kind,
                    "owner_role": task.owner_role,
                    "run_id": task.run_id,
                    "evidence_ids": task.inputs.get("evidence_ids"),
                    "evidence_count": len(list(task.inputs.get("evidence_ids") or [])),
                },
                "instruction": "Review the scoped evidence bundle. Validate first, then pass or request targeted repair.",
            },
            state=loop_state,
            safety_cap=safety_cap,
            metadata_role="critic_tool_loop",
            metadata={
                "run_id": task.run_id,
                "task_id": task.task_id,
                "operation": "critic_tool_loop",
                "evidence_count": len(list(task.inputs.get("evidence_ids") or [])),
            },
            on_tool_result=on_tool_result,
        )


def _critic_prompt() -> str:
    return (Path(__file__).resolve().parent.parent / "prompts" / "critic.md").read_text(encoding="utf-8")
