from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable

from insightswarm.agents.agent_loop_contract import (
    CONTRACT_VIOLATION_LIMIT,
    ContractViolation,
    detect_repeated_call,
    enforce_fast_path_commitment,
    enforce_required_next_step,
    validate_tool_call,
)
from insightswarm.agents.tool_executor import ToolExecutor
from insightswarm.models.agent_turn import parse_agent_turn


AGENT_LOOP_CONTRACT = """\
Shared Agent Loop Contract:
- The runtime provides tool_specs, private_state, minimal_event_memory, and recent_tool_transcript every round.
- Return one JSON object with assistant_text, private_state, tool_call, and stop_reason.
- While the task is active, choose exactly one tool_call. The tool name must exactly match tool_specs and the input must be an object.
- Use the role's finish_* tool for complete/blocked/no-productive-tool states. Return tool_call=null only if no finish tool exists.
- The runtime validates tool names and required inputs. Invalid or missing tool calls are rejected; recovery guidance may be supplied on the next round.
- Agents collaborate only through their exposed tools and shared stores. Do not directly call another agent's execution path.
- Formal Evidence can only be created by Extractor tools. Final reports can only be published by Writer tools.
- Private state is scratchpad continuity, not shared memory. Write only concise public observations through explicit shared-store tools.
"""


@dataclass
class AgentLoopState:
    private_state: dict[str, Any] = field(default_factory=dict)
    event_memory: dict[str, Any] = field(default_factory=lambda: {"events": [], "key_decisions": [], "abandoned_paths": []})
    messages: list[dict[str, Any]] = field(default_factory=list)
    model_failure_count: int = 0
    tool_contract_recovery: dict[str, Any] | None = None
    terminal_status: str | None = None
    terminal_reason: str | None = None
    consecutive_noop_count: int = 0
    last_noop_signature: str | None = None
    # Contract enforcement state (agent_loop_contract.py).
    consecutive_contract_violations: int = 0
    last_executed_call: dict[str, Any] | None = None
    last_tool_result: dict[str, Any] | None = None
    attempt_count: int = 0
    step_count: int = 0
    stop_reason: str | None = None  # structured: done | blocked | retry_limit | step_limit | model_rate_limited | model_error | contract_violation_limit
    # Set true once any quick_read returns fast_path_ready=true. Used by the
    # contract to reject finish_research(blocked): claiming "blocked" after a
    # usable quick_read source exists is self-contradictory — the model has
    # answerable material, it just chose not to deliver it. Narrow gate: only
    # blocks the "blocked" lie, not other finish_* paths or further searching.
    seen_fast_path_ready: bool = False


# ---------------------------------------------------------------------------
# Stall policy — the single home for no-op / spin detection tuning.
#
# These caps classify and bound consecutive non-productive rounds so the loop
# forces a blocked finish instead of spinning. They are the runtime's defense
# against three stall shapes: read-only re-reads, re-compute loops, and
# identical tool failures. Keep them here as ONE structure so future tuning
# (per-role policies, config-driven limits) has a single place to grow —
# scattered module constants invited whack-a-mole edits.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StallPolicy:
    # Read-only tools that surface existing state without producing new
    # external information. Consecutive calls without a productive tool in
    # between indicate the agent is re-reading the same state.
    noop_tools: frozenset[str]
    # Re-compute tools: read-only but return a structured view of existing
    # state. The first call is informative; consecutive calls are pure
    # re-computation (e.g. re-ranking the same candidates).
    recompute_tools: frozenset[str]
    # Hard cap on consecutive read-only no-op rounds before a forced blocked
    # finish. Generous enough for legitimate re-orientation, tight enough to
    # stop extended re-reading.
    noop_limit: int
    # Tighter cap for re-compute tools: one call is informative, a few in a
    # row is a stall. The agent should fetch or publish, not re-rank endlessly.
    recompute_limit: int
    # Tightest cap: identical non-terminal tool failures retried. One retry is
    # forgivable; repeated identical failures mean the model is stuck and will
    # not self-correct.
    tool_failure_limit: int


DEFAULT_STALL_POLICY = StallPolicy(
    noop_tools=frozenset({
        "read_task",
        "read_shared_memory",
        "read_evidence",
        "read_artifact",
        "read_board",
        "read_messages",
    }),
    recompute_tools=frozenset({
        "rank_sources",
    }),
    noop_limit=6,
    recompute_limit=3,
    tool_failure_limit=2,
)


def run_agent_loop(
    *,
    model_client: object | None,
    system_prompt: str,
    tool_specs: list[dict[str, Any]],
    executor: ToolExecutor,
    initial_user_payload: dict[str, Any],
    state: AgentLoopState | None = None,
    safety_cap: int = 50,
    max_attempts: int | None = None,
    max_tokens: int = 2200,
    metadata_role: str = "agent_tool_loop",
    metadata: dict[str, Any] | None = None,
    on_tool_result: Callable[[int, dict[str, Any], dict[str, Any], AgentLoopState], None] | None = None,
) -> tuple[list[dict[str, Any]], AgentLoopState]:
    loop_state = state or AgentLoopState()
    # Attempt budget is separate from step budget. Steps count only successful
    # tool executions; attempts count every model call (including rejections and
    # model failures). This prevents malformed/invalid calls from silently
    # burning the productive step budget. Inspired by V3's max_attempts.
    effective_max_attempts = max_attempts or max(safety_cap * 3, safety_cap + 10)
    if not loop_state.messages:
        loop_state.messages.append({"role": "user", "content": initial_user_payload})

    trace: list[dict[str, Any]] = []
    for round_number in range(1, max(1, safety_cap) + 1):
        # Attempt budget: every model call counts, including rejections.
        loop_state.attempt_count += 1
        if loop_state.attempt_count > effective_max_attempts:
            loop_state.terminal_status = "blocked"
            loop_state.terminal_reason = (
                f"attempt budget exhausted: {loop_state.attempt_count} model calls "
                f"(limit {effective_max_attempts}) with only {loop_state.step_count} successful executions"
            )
            loop_state.stop_reason = "retry_limit"
            trace.append({"round": round_number, "stop_reason": "retry_limit", "reason": loop_state.terminal_reason})
            break

        turn = _call_model(
            model_client=model_client,
            system_prompt=system_prompt,
            tool_specs=tool_specs,
            state=loop_state,
            metadata_role=metadata_role,
            metadata=metadata,
            max_tokens=max_tokens,
        )
        tool_call = _parse_tool_call(turn, tool_specs)
        loop_state.private_state = _next_private_state(turn, loop_state.private_state)

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
            loop_state.stop_reason = _stop_reason_from_status(failure_status)
            break

        # --- Hard contract enforcement (agent_loop_contract.py) ---
        # The model decided what to do; the runtime enforces boundaries.
        # Invalid calls are rejected (not executed) and burn the attempt budget.
        violation = validate_tool_call(tool_call, tool_specs)
        if violation is None:
            violation = detect_repeated_call(tool_call, loop_state.last_executed_call)
        if violation is None and loop_state.last_tool_result is not None:
            violation = enforce_required_next_step(
                loop_state.last_tool_result, tool_call, tool_specs
            )
        if violation is None and loop_state.seen_fast_path_ready:
            violation = enforce_fast_path_commitment(tool_call)

        if violation is not None:
            loop_state.consecutive_contract_violations += 1
            tool_result = _build_violation_result(violation, loop_state)
            loop_state.tool_contract_recovery = _violation_recovery(violation, tool_specs)
            loop_state.messages.append({"role": "assistant", "content": _compact_json(turn, limit=3000)})
            loop_state.messages.append({"role": "tool", "content": _compact_json(tool_result, limit=2000)})
            _append_event(loop_state, round_number, tool_call, tool_result)
            if on_tool_result:
                on_tool_result(round_number, tool_call, tool_result, loop_state)
            trace.append(
                {
                    "round": round_number,
                    "assistant_text": turn.get("assistant_text"),
                    "tool_call": tool_call,
                    "executed_tool": None,
                    "tool_result": tool_result,
                    "stop_reason": None,
                    "failure_kind": violation.failure_kind,
                    "private_state": dict(loop_state.private_state),
                    "minimal_event_memory": dict(loop_state.event_memory),
                }
            )
            if loop_state.consecutive_contract_violations < CONTRACT_VIOLATION_LIMIT:
                continue
            loop_state.terminal_status = "blocked"
            loop_state.terminal_reason = (
                f"contract violation limit reached: {loop_state.consecutive_contract_violations} consecutive "
                f"rejections ({violation.failure_kind}). The model could not produce a compliant tool call."
            )
            loop_state.stop_reason = "contract_violation_limit"
            break

        # --- Execute the validated, non-repeated, compliant tool call ---
        execution = executor.execute(tool_call)
        loop_state.step_count += 1
        loop_state.model_failure_count = 0
        loop_state.consecutive_contract_violations = 0
        loop_state.tool_contract_recovery = None
        loop_state.last_executed_call = dict(tool_call)
        tool_result = execution.to_message()
        loop_state.last_tool_result = dict(tool_result)
        # Latch fast_path_ready: once any quick_read signals it, the run has
        # answerable material. Used by enforce_fast_path_commitment to reject
        # the "blocked" lie downstream. Latched (never reset) because a usable
        # source doesn't become unusable by waiting.
        if tool_result.get("fast_path_ready") is True:
            loop_state.seen_fast_path_ready = True
        _append_event(loop_state, round_number, tool_call, tool_result)
        _update_noop_counter(loop_state, tool_call, tool_result)
        loop_state.messages.append({"role": "assistant", "content": _compact_json(turn, limit=3000)})
        loop_state.messages.append({"role": "tool", "content": _compact_json(tool_result, limit=4000)})
        if on_tool_result:
            on_tool_result(round_number, tool_call, tool_result, loop_state)

        if execution.terminal:
            loop_state.terminal_status = str(tool_result.get("status") or "done")
            loop_state.terminal_reason = _safe_text(tool_result.get("reason")) or _safe_text(tool_result.get("error"))
            loop_state.stop_reason = "done" if loop_state.terminal_status == "done" else "blocked"
        else:
            # Determine which limit applies based on the last tool signature.
            sig = loop_state.last_noop_signature or ""
            is_tool_failure = ":" in sig and sig.split(":", 1)[1] != ""
            is_recompute = sig in DEFAULT_STALL_POLICY.recompute_tools
            if is_tool_failure:
                applicable_limit = DEFAULT_STALL_POLICY.tool_failure_limit
                spin_kind = "tool-failure"
            elif is_recompute:
                applicable_limit = DEFAULT_STALL_POLICY.recompute_limit
                spin_kind = "re-compute"
            else:
                applicable_limit = DEFAULT_STALL_POLICY.noop_limit
                spin_kind = "read-only"
            if loop_state.consecutive_noop_count >= applicable_limit:
                loop_state.terminal_status = "blocked"
                loop_state.terminal_reason = (
                    f"no-op spin detected: {loop_state.consecutive_noop_count} consecutive {spin_kind} tool calls "
                    f"({loop_state.last_noop_signature}) produced no new information. "
                    f"Use finish_*(blocked) when stuck instead of re-reading, re-ranking, or re-failing the same state."
                )
                loop_state.stop_reason = "blocked"
                tool_result = {
                    **tool_result,
                    "ok": False,
                    "terminal": True,
                    "status": "blocked",
                    "reason": loop_state.terminal_reason,
                    "failure_kind": "noop_spin",
                }

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
        loop_state.stop_reason = "step_limit"
        trace.append({"round": safety_cap, "stop_reason": "step_limit", "reason": loop_state.terminal_reason})

    return trace, loop_state


def _call_model(
    *,
    model_client: object | None,
    system_prompt: str,
    tool_specs: list[dict[str, Any]],
    state: AgentLoopState,
    metadata_role: str,
    metadata: dict[str, Any] | None,
    max_tokens: int = 2200,
) -> dict[str, Any]:
    if model_client is None:
        return {
            "assistant_text": "No model client was supplied.",
            "tool_call": {"name": "finish_research", "input": {"status": "blocked", "reason": "missing model client"}},
            "stop_reason": None,
        }

    composed_system_prompt = _compose_system_prompt(system_prompt)
    payload = {
        "tool_specs": _model_tool_specs(tool_specs),
        "private_state": state.private_state,
        "minimal_event_memory": state.event_memory,
        "recent_tool_transcript": _model_transcript(state.messages),
    }
    if state.tool_contract_recovery:
        payload["tool_contract_recovery"] = state.tool_contract_recovery
    call_metadata = {
        **dict(metadata or {}),
        "role": metadata_role,
        "operation": str((metadata or {}).get("operation") or metadata_role),
    }
    result = model_client.complete(
        [
            {"role": "system", "content": composed_system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=max_tokens,
        metadata=call_metadata,
    )
    if str(getattr(result, "status", "ok")) != "ok":
        error_text = str(getattr(result, "error", "") or "model call failed")
        raw_response = getattr(result, "raw_response", None)
        http_status = raw_response.get("http_status") if isinstance(raw_response, dict) else None
        lowered_error = error_text.lower()
        if (
            http_status == 429
            or "429" in error_text
            or "rate limit" in lowered_error
            or "quota" in lowered_error
            or "insufficient" in lowered_error and "balance" in lowered_error
        ):
            stop_reason = "model_rate_limited"
        else:
            stop_reason = "model_error"
        return {
            "assistant_text": error_text[:1000],
            "tool_call": None,
            "stop_reason": stop_reason,
        }
    data = getattr(result, "json_data", None)
    if not isinstance(data, dict):
        text = str(getattr(result, "text", "") or "")
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            # JSON parse failure is a model_error, not a distinct failure kind.
            # The model produced unparseable output; retrying may help (transient
            # truncation) but there is no separate "invalid_json" recovery path.
            return {"assistant_text": text[:1000], "tool_call": None, "stop_reason": "model_error"}
    # AgentTurn schema validation: shape violations are model_error, not
    # invalid_json. This is the second defense line after the API layer's
    # JSON mode. Providers with json_schema strict mode make this a no-op.
    turn, schema_error = parse_agent_turn(data)
    if turn is None:
        return {"assistant_text": (schema_error or "schema error")[:1000], "tool_call": None, "stop_reason": "model_error"}
    return turn.model_dump(exclude_none=True)


def _compose_system_prompt(role_prompt: str) -> str:
    role_prompt = str(role_prompt).strip()
    if "Shared Agent Loop Contract:" in role_prompt:
        return role_prompt
    return f"{AGENT_LOOP_CONTRACT}\n\nRole Prompt:\n{role_prompt}"


def _model_tool_specs(tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render compact, model-facing tool specs while preserving runtime specs.

    Full output schemas and side-effect notes are useful for tests/docs, but they
    are expensive and rarely help tool selection every round. Runtime validation
    still uses the original tool_specs passed to ToolExecutor and _parse_tool_call.
    """
    compact: list[dict[str, Any]] = []
    for spec in tool_specs:
        name = _safe_text(spec.get("name"))
        if not name:
            continue
        item: dict[str, Any] = {
            "name": name,
            "description": _safe_text(spec.get("description"))[:360],
            "input": _compact_input_schema(spec.get("input_schema")),
        }
        for key in ("when_to_use", "do_not_use_when"):
            value = _safe_text(spec.get(key))
            if value:
                item[key] = value[:300]
        compact.append(item)
    return compact


def _compact_input_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    properties = schema.get("properties")
    required = set(schema.get("required") or [])
    if not isinstance(properties, dict):
        return {"required": sorted(str(item) for item in required)}
    fields: dict[str, str] = {}
    for field_name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            fields[str(field_name)] = "optional"
            continue
        field_type = field_schema.get("type") or "value"
        marker = "required" if field_name in required else "optional"
        description = _safe_text(field_schema.get("description"))
        rendered = f"{field_type}, {marker}"
        if description:
            rendered = f"{rendered}; {description[:160]}"
        enum = field_schema.get("enum")
        if isinstance(enum, list) and enum:
            rendered = f"{rendered}; one of {', '.join(str(item) for item in enum[:8])}"
        fields[str(field_name)] = rendered
    return fields


def _model_transcript(messages: list[dict[str, Any]], *, recent_window: int = 4, older_window: int = 6) -> list[dict[str, Any]]:
    if not messages:
        return []
    selected = list(messages[-(recent_window + older_window):])
    recent_start = max(0, len(selected) - recent_window)
    rendered: list[dict[str, Any]] = []
    for index, message in enumerate(selected):
        recent = index >= recent_start
        rendered.append(_render_transcript_message(message, recent=recent))
    return rendered


def _render_transcript_message(message: dict[str, Any], *, recent: bool) -> dict[str, Any]:
    role = _safe_text(message.get("role")) or "unknown"
    content = message.get("content")
    if role == "assistant":
        return {
            "role": role,
            "summary": _assistant_summary(content, limit=360 if recent else 160),
        }
    if role == "tool":
        if recent:
            return {
                "role": role,
                "content": _summarize_tool_result(content, rich=True),
            }
        return {
            "role": role,
            "summary": _summarize_tool_result(content, rich=False),
        }
    return {
        "role": role,
        "content": _compact_for_model(content, limit=900 if recent else 220),
    }


def _assistant_summary(content: Any, *, limit: int) -> str:
    if isinstance(content, dict):
        text = _safe_text(content.get("assistant_text"))
        if not text and isinstance(content.get("tool_call"), dict):
            tool_call = content["tool_call"]
            text = f"called {tool_call.get('name')}"
        if not text:
            text = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    else:
        text = _safe_text(content)
    return _clip_text(text, limit)


def _summarize_tool_result(content: Any, *, rich: bool) -> Any:
    result = content
    if isinstance(content, dict) and set(content) == {"truncated_json"}:
        return {"summary": _clip_text(_safe_text(content.get("truncated_json")), 420 if rich else 180)}
    if not isinstance(result, dict):
        return _clip_text(_safe_text(result), 900 if rich else 180)

    tool_name = _safe_text(result.get("tool_name")) or _safe_text(result.get("name"))
    if result.get("error"):
        return {
            "tool_name": tool_name,
            "ok": False,
            "error": _clip_text(_safe_text(result.get("error")), 500 if rich else 180),
            "failure_kind": result.get("failure_kind"),
        }
    if tool_name == "search_web" or "candidates" in result:
        candidates = list(result.get("candidates") or [])
        visible_candidates = candidates[: (3 if rich else 2)]
        compact_candidates = [_compact_candidate(candidate, rich=rich) for candidate in visible_candidates if isinstance(candidate, dict)]
        summary: dict[str, Any] = {
            "tool_name": tool_name or "search_web",
            "query": _clip_text(_safe_text(result.get("query")), 180),
            "candidate_count": len(candidates),
            "candidates": compact_candidates,
        }
        if len(candidates) > len(compact_candidates):
            summary["omitted_candidate_count"] = len(candidates) - len(compact_candidates)
        return summary
    if tool_name in {"fetch_source", "firecrawl_source"} or "document" in result:
        document = result.get("document") if isinstance(result.get("document"), dict) else {}
        return {
            "tool_name": tool_name or "fetch_source",
            "document": _compact_document(document, rich=rich),
            "acquisition_pressure": _compact_acquisition_pressure(result.get("acquisition_pressure")),
        }
    if tool_name == "rank_sources" or "ranked_sources" in result:
        ranked = list(result.get("ranked_sources") or [])
        return {
            "tool_name": tool_name or "rank_sources",
            "ranked_sources": [_compact_source_decision(item) for item in ranked[: (5 if rich else 3)] if isinstance(item, dict)],
            "omitted_count": max(0, len(ranked) - (5 if rich else 3)),
        }
    if tool_name == "publish_raw_source" or "artifact_ids" in result:
        return {
            "tool_name": tool_name or "publish_raw_source",
            "artifact_ids": list(result.get("artifact_ids") or [])[:6],
            "extractor_task_ids": list(result.get("extractor_task_ids") or [])[:6],
            "published_count": len(list(result.get("artifact_ids") or [])),
        }
    if tool_name and tool_name.startswith("finish_"):
        return {
            "tool_name": tool_name,
            "status": result.get("status"),
            "reason": _clip_text(_safe_text(result.get("reason")), 300),
            "terminal": result.get("terminal"),
        }
    return _compact_for_model(result, limit=1200 if rich else 260)


def _compact_candidate(candidate: dict[str, Any], *, rich: bool) -> dict[str, Any]:
    return {
        "source_id": candidate.get("source_id") or candidate.get("id"),
        "url": _clip_text(_safe_text(candidate.get("url")), 240),
        "title": _clip_text(_safe_text(candidate.get("title")), 180),
        "snippet": _clip_text(_safe_text(candidate.get("snippet")), 360 if rich else 140),
        "source_category": candidate.get("source_category"),
        "estimated_fetch_risk": candidate.get("estimated_fetch_risk"),
        "content_format_hint": candidate.get("content_format_hint"),
    }


def _compact_document(document: dict[str, Any], *, rich: bool) -> dict[str, Any]:
    preview = _safe_text(document.get("text_preview") or document.get("preview"))
    return {
        "url": _clip_text(_safe_text(document.get("url")), 260),
        "title": _clip_text(_safe_text(document.get("title")), 180),
        "usable": document.get("usable"),
        "usability_reason": document.get("usability_reason"),
        "page_type": document.get("page_type"),
        "information_density": document.get("information_density"),
        "text_preview": _clip_text(preview, 500 if rich else 180),
    }


def _compact_acquisition_pressure(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    return {
        "blocked_or_rate_limited_failures": value.get("blocked_or_rate_limited_failures"),
        "firecrawl_failures": value.get("firecrawl_failures"),
        "should_request_browser": value.get("should_request_browser"),
        "issue_key": value.get("issue_key"),
        "blocked_domains": list(value.get("blocked_domains") or [])[:6],
    }


def _compact_source_decision(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": _clip_text(_safe_text(item.get("url")), 240),
        "decision": item.get("decision"),
        "reason": _clip_text(_safe_text(item.get("reason") or item.get("rationale")), 220),
        "priority": item.get("priority"),
    }


def _compact_for_model(value: Any, *, limit: int) -> Any:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= limit:
        return value
    return {"summary": _clip_text(text, limit)}


def _clip_text(text: str, limit: int) -> str:
    text = _safe_text(text)
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _failure_kind(status: str, reason: str) -> str:
    lowered_status = _safe_text(status).lower()
    lowered_reason = _safe_text(reason).lower()
    if lowered_status == "model_error" and ("timed out" in lowered_reason or "timeout" in lowered_reason):
        return "model_timeout"
    if lowered_status == "model_error":
        return lowered_status
    # model_no_tool is preserved as a failure_kind label for trace clarity,
    # but it is no longer recoverable (see _recoverable_model_failure).
    if lowered_status == "model_no_tool":
        return "model_no_tool"
    if lowered_status == "blocked" and "safety cap" in lowered_reason:
        return "safety_cap"
    return lowered_status or "unknown"


def _recoverable_model_failure(status: str) -> bool:
    # model_no_tool and invalid_json are NOT recoverable: the model chose not
    # to call a tool (or produced unparseable output), and retrying lets it
    # stall again. Only transient infra failures (model_error, rate limits)
    # get retry budget. First model_no_tool/invalid_json terminates the loop.
    return _safe_text(status).lower() in {"model_error", "model_rate_limited"}


def _stop_reason_from_status(status: str) -> str:
    """Map a terminal model-failure status to a structured stop_reason."""
    lowered = _safe_text(status).lower()
    if lowered == "model_rate_limited":
        return "model_rate_limited"
    # model_no_tool and the removed invalid_json both collapse to model_error
    # in the structured stop_reason vocabulary.
    if lowered in {"model_error", "model_no_tool"}:
        return "model_error"
    return "blocked"


def _build_violation_result(violation: ContractViolation, state: AgentLoopState) -> dict[str, Any]:
    """Build a structured tool_result for a rejected (non-executed) tool call."""
    return {
        "ok": False,
        "tool_name": violation.kind,
        "error": violation.reason,
        "terminal": False,
        "failure_kind": violation.failure_kind,
        "contract_violation": True,
        "consecutive_contract_violations": state.consecutive_contract_violations,
        "required_next_step": _violation_required_next_step(violation),
    }


def _violation_required_next_step(violation: ContractViolation) -> str:
    """Concrete, actionable guidance for the model after a contract violation."""
    if violation.failure_kind == "repeated_call":
        return (
            "You just called this tool with the same arguments. Choose a different tool, "
            "different arguments, or call a finish_* tool. Do not repeat the exact same call."
        )
    if violation.failure_kind == "ignored_browser_escalation":
        return (
            "The previous fetch reported acquisition_pressure with recommended_escalation=browser_agent. "
            "Call suggest_browser_acquisition with the target URL, or finish_*(blocked) if browser "
            "acquisition would not help. Do not call fetch_source again for the same blocked source."
        )
    if violation.failure_kind == "ignored_missing_review_basis":
        return (
            "The previous terminal tool failed because review_basis was missing. Retry the same "
            "terminal tool WITH review_basis in the input, or call finish_*(blocked)."
        )
    if violation.failure_kind == "invalid_tool_name":
        return "Use a tool name from the tool_specs list. Do not invent tool names."
    if violation.failure_kind == "missing_required_input":
        return "Provide all required input fields for the tool. Check tool_specs for required fields."
    return "Return valid JSON with one tool_call using an exact tool name and required inputs from tool_specs."


def _violation_recovery(violation: ContractViolation, tool_specs: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a tool_contract_recovery payload for a contract violation."""
    allowed_tool_names = [str(spec.get("name") or "") for spec in tool_specs if str(spec.get("name") or "")]
    return {
        "contract_error": violation.failure_kind,
        "reason": violation.reason[:500],
        "allowed_tool_names": allowed_tool_names,
        "required_tool": None,
        "required_input": {},
        "instruction": _violation_required_next_step(violation),
    }


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


def _next_private_state(turn: dict[str, Any], previous_state: dict[str, Any] | None = None) -> dict[str, Any]:
    private_state = turn.get("private_state")
    if isinstance(private_state, dict):
        return _compact_json(private_state, limit=3500)
    # Runtime errors (timeout/rate-limit) are not model reasoning — the model
    # never thought. Preserving the prior reasoning state prevents error text
    # from overwriting the model's understanding. Recovery context reaches the
    # model via the model_error tool_result (history layer), not private_state
    # (reasoning layer). Mirrors V3's separation: tool results live in history,
    # reasoning lives in prefix/memory — they don't bleed into each other.
    if _safe_text(turn.get("stop_reason")) in {"model_error", "model_rate_limited", "model_no_tool"} and previous_state:
        return dict(previous_state)
    return {
        "current_understanding": _safe_text(turn.get("current_understanding")) or _safe_text(turn.get("assistant_text")),
        "gap": _safe_text(turn.get("gap")),
        "situation_assessment": turn.get("situation_assessment") if isinstance(turn.get("situation_assessment"), dict) else None,
        "failure_reflection": _safe_text(turn.get("failure_reflection")),
        "source_priority_reasoning": turn.get("source_priority_reasoning") if isinstance(turn.get("source_priority_reasoning"), dict) else None,
        "plan": _safe_text(turn.get("plan")),
        "publish_check": turn.get("publish_check") if isinstance(turn.get("publish_check"), dict) else None,
    }


def _update_noop_counter(state: AgentLoopState, tool_call: dict[str, Any], tool_result: dict[str, Any]) -> None:
    """Track consecutive read-only tool calls that produce no new information.

    A "no-op" round is one where the agent called a read-only tool (read_task,
    read_shared_memory, etc.) AND the result carries no new artifact/evidence/
    source. Productive tools (search_web, fetch_source, publish_*, finish_*,
    create_*) reset the counter to 0.

    Re-compute tools (rank_sources) are treated as no-ops on every call after
    the first consecutive one: they re-sort existing state without producing
    new information, so 3 in a row is a stall.

    Non-terminal tool failures (ok=False, terminal=False, failure_kind set) are
    also counted as no-ops. This stops deadlocks where a terminal tool keeps
    returning a recoverable error (e.g. missing_review_basis) and the model
    retries it identically until safety_cap. Observed 2026-06-23,
    run-run_ac4eb4e41942: critic called request_repair/reject_direction 3x
    consecutively, each returning missing_review_basis=True, burning 6 rounds.
    """
    name = str(tool_call.get("name") or "")
    # Non-terminal tool failure with a failure_kind: count as no-op so repeated
    # failures hard-stop instead of spinning to safety_cap.
    if (
        tool_result.get("ok") is False
        and not tool_result.get("terminal")
        and tool_result.get("failure_kind")
    ):
        state.consecutive_noop_count += 1
        state.last_noop_signature = f"{name}:{tool_result['failure_kind']}"
        return
    if name not in DEFAULT_STALL_POLICY.noop_tools and name not in DEFAULT_STALL_POLICY.recompute_tools:
        state.consecutive_noop_count = 0
        state.last_noop_signature = None
        return
    # Re-compute tools: always count as no-op (the first call is "useful" but
    # still does not produce new external information; the recompute_limit
    # gives enough headroom for one informative rank).
    if name in DEFAULT_STALL_POLICY.recompute_tools:
        state.consecutive_noop_count += 1
        state.last_noop_signature = name
        return
    # Read-only tool. Check whether it returned anything new.
    has_new_content = bool(
        tool_result.get("candidates")
        or tool_result.get("document")
        or tool_result.get("artifact_ids")
        or tool_result.get("evidence_ids")
        or tool_result.get("messages")
        or tool_result.get("tasks")
        or tool_result.get("evidence")
        or tool_result.get("artifacts")
    )
    if has_new_content:
        # A read that surfaces previously-unseen items is productive.
        state.consecutive_noop_count = 0
        state.last_noop_signature = None
        return
    state.consecutive_noop_count += 1
    state.last_noop_signature = name


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
