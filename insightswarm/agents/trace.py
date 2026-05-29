from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from insightswarm.agents.agent_loop import AgentLoopState
from insightswarm.schemas.swarm import Task


def build_tool_trace_callback(trace_path: Path | None, *, role: str, task: Task):
    if trace_path is None:
        return None

    def _write(round_number: int, tool_call: dict[str, Any], tool_result: dict[str, Any], loop_state: AgentLoopState) -> None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "kind": "agent_tool_round",
            "role": role,
            "task_id": task.task_id,
            "task_kind": task.kind,
            "round": round_number,
            "tool_call": tool_call,
            "tool_result": _summarize_tool_result(tool_result),
            "private_state": loop_state.private_state,
            "event_memory": loop_state.event_memory,
        }
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return _write


def _summarize_tool_result(tool_result: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "ok",
        "tool_name",
        "error",
        "terminal",
        "status",
        "reason",
        "artifact_ids",
        "citation_artifact_ids",
        "evidence_ids",
        "repair_task_id",
        "message_id",
        "review_task_id",
        "unresolved_publishable_count",
    )
    summary = {key: tool_result.get(key) for key in keys if key in tool_result}
    if isinstance(tool_result.get("document"), dict):
        document = tool_result["document"]
        summary["document"] = {
            "url": document.get("url"),
            "title": document.get("title"),
            "usable": document.get("usable"),
            "usability_reason": document.get("usability_reason"),
            "page_type": document.get("page_type"),
            "information_density": document.get("information_density"),
            "fetcher": document.get("fetcher"),
        }
    if "candidates" in tool_result:
        summary["candidate_count"] = len(list(tool_result.get("candidates") or []))
    if "ranked_sources" in tool_result:
        summary["ranked_source_count"] = len(list(tool_result.get("ranked_sources") or []))
    return summary
