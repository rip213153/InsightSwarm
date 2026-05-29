from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable

from insightswarm.agents.tool_executor import ToolExecutor


@dataclass
class AgentLoopState:
    private_state: dict[str, Any] = field(default_factory=dict)
    event_memory: dict[str, Any] = field(default_factory=lambda: {"events": [], "key_decisions": [], "abandoned_paths": []})
    messages: list[dict[str, Any]] = field(default_factory=list)
    terminal_status: str | None = None
    terminal_reason: str | None = None


def run_agent_loop(
    *,
    model_client: object | None,
    system_prompt: str,
    tool_specs: list[dict[str, Any]],
    executor: ToolExecutor,
    initial_user_payload: dict[str, Any],
    state: AgentLoopState | None = None,
    safety_cap: int = 50,
    metadata_role: str = "agent_tool_loop",
    on_tool_result: Callable[[int, dict[str, Any], dict[str, Any], AgentLoopState], None] | None = None,
) -> tuple[list[dict[str, Any]], AgentLoopState]:
    loop_state = state or AgentLoopState()
    if not loop_state.messages:
        loop_state.messages.append({"role": "user", "content": initial_user_payload})

    trace: list[dict[str, Any]] = []
    for round_number in range(1, max(1, safety_cap) + 1):
        turn = _call_model(
            model_client=model_client,
            system_prompt=system_prompt,
            tool_specs=tool_specs,
            state=loop_state,
            metadata_role=metadata_role,
        )
        tool_call = _parse_tool_call(turn, tool_specs)
        loop_state.private_state = _next_private_state(turn)

        if tool_call is None:
            loop_state.terminal_status = _safe_text(turn.get("stop_reason")) or "done"
            loop_state.terminal_reason = _safe_text(turn.get("assistant_text")) or "model stopped without a tool call"
            trace.append(
                {
                    "round": round_number,
                    "assistant_text": turn.get("assistant_text"),
                    "tool_call": None,
                    "tool_result": None,
                    "stop_reason": loop_state.terminal_status,
                    "private_state": dict(loop_state.private_state),
                    "minimal_event_memory": dict(loop_state.event_memory),
                }
            )
            break

        execution = executor.execute(tool_call)
        tool_result = execution.to_message()
        _append_event(loop_state, round_number, tool_call, tool_result)
        loop_state.messages.append({"role": "assistant", "content": _compact_json(turn, limit=3000)})
        loop_state.messages.append({"role": "tool", "content": _compact_json(tool_result, limit=4000)})
        if on_tool_result:
            on_tool_result(round_number, tool_call, tool_result, loop_state)

        if execution.terminal:
            loop_state.terminal_status = str(tool_result.get("status") or "done")
            loop_state.terminal_reason = _safe_text(tool_result.get("reason")) or _safe_text(tool_result.get("error"))

        trace.append(
            {
                "round": round_number,
                "assistant_text": turn.get("assistant_text"),
                "tool_call": tool_call,
                "executed_tool": execution.tool_name,
                "tool_result": _compact_json(tool_result, limit=4000),
                "stop_reason": loop_state.terminal_status,
                "private_state": dict(loop_state.private_state),
                "minimal_event_memory": dict(loop_state.event_memory),
            }
        )
        if loop_state.terminal_status:
            break
    else:
        loop_state.terminal_status = "blocked"
        loop_state.terminal_reason = "safety cap reached"
        trace.append({"round": safety_cap, "stop_reason": "blocked", "reason": loop_state.terminal_reason})

    return trace, loop_state


def _call_model(
    *,
    model_client: object | None,
    system_prompt: str,
    tool_specs: list[dict[str, Any]],
    state: AgentLoopState,
    metadata_role: str,
) -> dict[str, Any]:
    if model_client is None:
        return {
            "assistant_text": "No model client was supplied.",
            "tool_call": {"name": "finish_research", "input": {"status": "blocked", "reason": "missing model client"}},
            "stop_reason": None,
        }

    payload = {
        "tool_specs": tool_specs,
        "private_state": state.private_state,
        "minimal_event_memory": state.event_memory,
        "recent_tool_transcript": state.messages[-10:],
    }
    result = model_client.complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, sort_keys=True)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=2200,
        metadata={"role": metadata_role},
    )
    if str(getattr(result, "status", "ok")) != "ok":
        return {
            "assistant_text": str(getattr(result, "error", "") or "model call failed")[:1000],
            "tool_call": None,
            "stop_reason": "model_error",
        }
    data = getattr(result, "json_data", None)
    if isinstance(data, dict):
        return data
    text = str(getattr(result, "text", "") or "")
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {"assistant_text": text[:1000], "tool_call": None, "stop_reason": "invalid_json"}


def _parse_tool_call(turn: dict[str, Any], tool_specs: list[dict[str, Any]]) -> dict[str, Any] | None:
    raw = turn.get("tool_call")
    if raw is None or not isinstance(raw, dict):
        return None
    name = _safe_text(raw.get("name"))
    if name not in {str(spec.get("name") or "") for spec in tool_specs}:
        return None
    tool_input = raw.get("input")
    return {"name": name, "input": dict(tool_input or {}) if isinstance(tool_input, dict) else {}}


def _next_private_state(turn: dict[str, Any]) -> dict[str, Any]:
    private_state = turn.get("private_state")
    if isinstance(private_state, dict):
        return _compact_json(private_state, limit=3500)
    return {
        "current_understanding": _safe_text(turn.get("current_understanding")) or _safe_text(turn.get("assistant_text")),
        "gap": _safe_text(turn.get("gap")),
        "situation_assessment": turn.get("situation_assessment") if isinstance(turn.get("situation_assessment"), dict) else None,
        "failure_reflection": _safe_text(turn.get("failure_reflection")),
        "source_priority_reasoning": turn.get("source_priority_reasoning") if isinstance(turn.get("source_priority_reasoning"), dict) else None,
        "plan": _safe_text(turn.get("plan")),
        "publish_check": turn.get("publish_check") if isinstance(turn.get("publish_check"), dict) else None,
    }


def _append_event(state: AgentLoopState, round_number: int, tool_call: dict[str, Any], tool_result: dict[str, Any]) -> None:
    event = {
        "round": round_number,
        "type": str(tool_result.get("tool_name") or tool_call.get("name")),
        "summary": _event_summary(tool_call, tool_result),
    }
    state.event_memory.setdefault("events", []).append(event)
    state.event_memory["events"] = list(state.event_memory["events"])[-24:]


def _event_summary(tool_call: dict[str, Any], tool_result: dict[str, Any]) -> str:
    name = str(tool_result.get("tool_name") or tool_call.get("name") or "")
    if tool_result.get("error"):
        return f"{name} failed: {tool_result.get('error')}"
    if name == "search_web":
        return f"searched {tool_result.get('query')} and found {len(list(tool_result.get('candidates') or []))} candidates"
    if name == "fetch_source":
        doc = dict(tool_result.get("document") or {})
        return f"fetched {doc.get('url')}; usable={doc.get('usable')}; reason={doc.get('usability_reason')}"
    if name == "publish_raw_source":
        return f"published {len(list(tool_result.get('artifact_ids') or []))} raw document(s)"
    if name == "finish_research":
        return f"finished: {tool_result.get('status')} {tool_result.get('reason')}"
    return f"{name} completed"


def _compact_json(value: Any, *, limit: int) -> Any:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    if len(text) <= limit:
        return value
    return {"truncated_json": text[:limit]}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
