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
    model_failure_count: int = 0
    tool_contract_recovery: dict[str, Any] | None = None
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
            failure_status = _safe_text(turn.get("stop_reason")) or "model_no_tool"
            failure_reason = _safe_text(turn.get("assistant_text")) or "model stopped without a tool call"
            loop_state.model_failure_count += 1
            tool_result = {
                "ok": False,
                "tool_name": failure_status,
                "error": failure_reason,
                "terminal": False,
                "failure_kind": _failure_kind(failure_status, failure_reason),
                "required_next_step": _required_next_step(tool_specs, loop_state),
                "model_failure_count": loop_state.model_failure_count,
            }
            loop_state.tool_contract_recovery = _tool_contract_recovery(
                tool_specs=tool_specs,
                state=loop_state,
                failure_status=failure_status,
                failure_reason=failure_reason,
            )
            loop_state.messages.append({"role": "assistant", "content": _compact_json(turn, limit=3000)})
            loop_state.messages.append({"role": "tool", "content": _compact_json(tool_result, limit=2000)})
            _append_event(loop_state, round_number, {"name": failure_status, "input": {}}, tool_result)
            if on_tool_result:
                on_tool_result(round_number, {"name": failure_status, "input": {}}, tool_result, loop_state)
            trace.append(
                {
                    "round": round_number,
                    "assistant_text": turn.get("assistant_text"),
                    "tool_call": None,
                    "tool_result": tool_result,
                    "stop_reason": None,
                    "failure_kind": tool_result["failure_kind"],
                    "private_state": dict(loop_state.private_state),
                    "minimal_event_memory": dict(loop_state.event_memory),
                }
            )
            if _recoverable_model_failure(failure_status) and loop_state.model_failure_count < 3:
                continue
            loop_state.terminal_status = failure_status
            loop_state.terminal_reason = failure_reason
            break

        execution = executor.execute(tool_call)
        loop_state.model_failure_count = 0
        loop_state.tool_contract_recovery = None
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
    if state.tool_contract_recovery:
        payload["tool_contract_recovery"] = state.tool_contract_recovery
    result = model_client.complete(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
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


def _failure_kind(status: str, reason: str) -> str:
    lowered_status = _safe_text(status).lower()
    lowered_reason = _safe_text(reason).lower()
    if lowered_status == "model_error" and ("timed out" in lowered_reason or "timeout" in lowered_reason):
        return "model_timeout"
    if lowered_status in {"model_error", "invalid_json"}:
        return lowered_status
    if lowered_status == "model_no_tool":
        return "model_no_tool"
    if lowered_status == "blocked" and "safety cap" in lowered_reason:
        return "safety_cap"
    return lowered_status or "unknown"


def _recoverable_model_failure(status: str) -> bool:
    return _safe_text(status).lower() in {"model_error", "invalid_json", "model_no_tool"}


def _required_next_step(tool_specs: list[dict[str, Any]], state: AgentLoopState) -> str:
    if _must_start_with_read_task(tool_specs, state):
        return "Return valid JSON with tool_call.name='read_task' and tool_call.input={}. Do not return tool_call=null."
    return "Return valid JSON with one tool_call using an exact tool name from tool_specs. Do not return tool_call=null unless a finish_* tool is unavailable."


def _tool_contract_recovery(
    *,
    tool_specs: list[dict[str, Any]],
    state: AgentLoopState,
    failure_status: str,
    failure_reason: str,
) -> dict[str, Any]:
    allowed_tool_names = [str(spec.get("name") or "") for spec in tool_specs if str(spec.get("name") or "")]
    if _must_start_with_read_task(tool_specs, state):
        allowed_tool_names = ["read_task"]
        required_tool = "read_task"
        required_input: dict[str, Any] = {}
    else:
        required_tool = None
        required_input = {}
    return {
        "contract_error": _safe_text(failure_status) or "model_no_tool",
        "reason": _safe_text(failure_reason)[:500],
        "allowed_tool_names": allowed_tool_names,
        "required_tool": required_tool,
        "required_input": required_input,
        "instruction": (
            "Your previous response did not contain a valid tool_call. "
            "For the next response, return JSON only with private_state and exactly one valid tool_call. "
            "Do not explain without a tool call."
        ),
    }


def _must_start_with_read_task(tool_specs: list[dict[str, Any]], state: AgentLoopState) -> bool:
    if state.model_failure_count < 1:
        return False
    tool_names = {str(spec.get("name") or "") for spec in tool_specs}
    if "read_task" not in tool_names:
        return False
    return not any(_message_mentions_tool(message, "read_task") for message in state.messages)


def _message_mentions_tool(message: dict[str, Any], tool_name: str) -> bool:
    try:
        text = json.dumps(message.get("content"), ensure_ascii=True, default=str)
    except TypeError:
        text = str(message.get("content") or "")
    return f'"name": "{tool_name}"' in text or f'"tool_name": "{tool_name}"' in text


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
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= limit:
        return value
    return {"truncated_json": text[:limit]}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
