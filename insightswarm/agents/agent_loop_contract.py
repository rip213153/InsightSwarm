"""Hard contract enforcement for the agent loop.

The model decides WHAT to do; this module enforces the BOUNDARIES:
- tool_call must be a valid object with a known name and required inputs
- the same tool+args must not be called twice in a row (repeated_call)
- tool_result recommendations (acquisition_pressure, failure_kind) become hard
  constraints on the next tool_call, not soft suggestions

Inspired by V3: the runtime does not negotiate with the model. Invalid calls
are rejected (not executed) and burn an attempt budget that is separate from
the step budget. The model gets structured feedback; if it keeps violating,
the loop hard-stops with a clean stop_reason.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContractViolation:
    """A hard contract violation that rejects the tool_call without execution."""

    kind: str  # invalid_tool_call | repeated_call | ignored_required_next_step
    reason: str
    failure_kind: str  # machine-readable, for spin detection and stop_reason


# Hard cap on consecutive contract violations before the loop forces a blocked
# finish. Three is enough for the model to self-correct after one rejection
# (round 1: violation, round 2: retry with fix, round 3: still wrong → stop).
CONTRACT_VIOLATION_LIMIT = 3

# Tools that may legitimately repeat: finish_*(blocked) confirms a stuck state.
_TERMINAL_TOOL_PREFIXES = ("finish_",)


def validate_tool_call(
    tool_call: dict[str, Any] | None,
    tool_specs: list[dict[str, Any]],
) -> ContractViolation | None:
    """Hard gate: structural validation of name + required inputs.

    Returns a violation if:
    - tool_call is None or not a dict
    - name is missing or not in tool_specs
    - input is not a dict
    - required inputs are missing/empty
    """
    if not isinstance(tool_call, dict):
        return ContractViolation(
            kind="invalid_tool_call",
            reason="tool_call must be a JSON object with name and input",
            failure_kind="invalid_tool_call",
        )
    name = str(tool_call.get("name") or "").strip()
    spec_by_name = {str(spec.get("name") or ""): spec for spec in tool_specs}
    if name not in spec_by_name:
        return ContractViolation(
            kind="invalid_tool_call",
            reason=f"unknown tool name: {name!r}; must be one of {sorted(spec_by_name)}",
            failure_kind="invalid_tool_name",
        )
    tool_input = tool_call.get("input")
    if not isinstance(tool_input, dict):
        return ContractViolation(
            kind="invalid_tool_call",
            reason="tool_call.input must be a JSON object",
            failure_kind="invalid_tool_input",
        )
    schema = dict(spec_by_name[name].get("input_schema") or {})
    required = list(schema.get("required") or [])
    missing = [key for key in required if key not in tool_input or tool_input.get(key) in (None, "")]
    if missing:
        return ContractViolation(
            kind="invalid_tool_call",
            reason=f"missing required input for {name}: {', '.join(missing)}",
            failure_kind="missing_required_input",
        )
    # Enum constraints declared in the input_schema are runtime-enforced here,
    # so handlers don't have to re-validate. This keeps "shape" checks in the
    # contract layer and "stateful" checks (e.g. quick_read history) in handlers.
    properties = dict(schema.get("properties") or {})
    for field_name, field_spec in properties.items():
        enum_values = list(field_spec.get("enum") or [])
        if not enum_values:
            continue
        value = tool_input.get(field_name)
        if value is None:
            continue  # missing required already checked above
        value_str = str(value).strip()
        allowed_str = [str(v) for v in enum_values]
        if value_str not in allowed_str:
            return ContractViolation(
                kind="invalid_tool_call",
                reason=(
                    f"invalid value for {name}.{field_name}: {value_str!r}; "
                    f"must be one of {allowed_str}"
                ),
                failure_kind="invalid_enum_value",
            )
    return None


def detect_repeated_call(
    tool_call: dict[str, Any],
    last_executed_call: dict[str, Any] | None,
) -> ContractViolation | None:
    """Hard gate: reject if the same tool+args was just executed.

    Checks only the immediately previous executed call. Terminal tools
    (finish_*) are exempt — finish_*(blocked) may legitimately repeat when the
    model confirms a stuck state.

    This catches the spin pattern where the model re-reads the same task,
    re-ranks the same list, or re-fetches the same URL without making progress.
    The no-op spin detector handles the broader case of different read-only
    tools called in sequence; this catches the identical-call case immediately.
    """
    if last_executed_call is None:
        return None
    name = str(tool_call.get("name") or "")
    if any(name.startswith(prefix) for prefix in _TERMINAL_TOOL_PREFIXES):
        return None
    if _call_signature(tool_call) == _call_signature(last_executed_call):
        return ContractViolation(
            kind="repeated_call",
            reason=(
                f"repeated identical call to {name} with the same arguments; "
                f"choose a different tool, different arguments, or call a finish_* tool"
            ),
            failure_kind="repeated_call",
        )
    return None


def enforce_required_next_step(
    prev_tool_result: dict[str, Any] | None,
    next_tool_call: dict[str, Any],
    tool_specs: list[dict[str, Any]],
) -> ContractViolation | None:
    """Hard gate: enforce that tool_result recommendations are followed.

    When the previous tool_result carries a hard recommendation, the next
    tool_call must comply:
    - acquisition_pressure.recommended_escalation=browser_agent → must call
      suggest_browser_acquisition or a finish_*(blocked) tool
    - failure_kind=missing_review_basis → must call a terminal tool (retry with
      fix, or finish blocked)

    This is the core V3 principle: tool returns a recommendation, runtime
    decides whether it becomes a hard constraint. InsightSwarm's tools already
    return these signals; the runtime now enforces them instead of hoping the
    model reads the suggestion text.
    """
    if not isinstance(prev_tool_result, dict):
        return None
    next_name = str(next_tool_call.get("name") or "")
    terminal_names = _terminal_tool_names(tool_specs)

    # 1. Browser escalation: acquisition_pressure says escalate to browser agent.
    pressure = prev_tool_result.get("acquisition_pressure")
    if isinstance(pressure, dict) and pressure.get("recommended_escalation") == "browser_agent":
        if next_name != "suggest_browser_acquisition" and next_name not in terminal_names:
            return ContractViolation(
                kind="ignored_required_next_step",
                reason=(
                    "previous fetch reported acquisition_pressure with "
                    "recommended_escalation=browser_agent; the next call must be "
                    "suggest_browser_acquisition or a finish_*(blocked) tool — "
                    "do not continue fetching the same blocked source"
                ),
                failure_kind="ignored_browser_escalation",
            )

    # 2. Missing review_basis: the previous terminal tool failed because
    # review_basis was missing. The next call must be a terminal tool (retry
    # the same tool WITH review_basis, or finish_*(blocked)).
    if prev_tool_result.get("failure_kind") == "missing_review_basis":
        if next_name not in terminal_names:
            return ContractViolation(
                kind="ignored_required_next_step",
                reason=(
                    "previous terminal tool failed with missing_review_basis; "
                    "retry the same tool WITH review_basis in the input, or call "
                    "finish_*(blocked) — do not switch to a non-terminal tool"
                ),
                failure_kind="ignored_missing_review_basis",
            )

    return None


def enforce_fast_path_commitment(
    next_tool_call: dict[str, Any],
) -> ContractViolation | None:
    """Reject finish_research(blocked) after a quick_read signaled fast_path_ready.

    Narrow gate: once the runtime has latched seen_fast_path_ready (a usable
    quick_read source exists), the model may NOT claim it is "blocked". That
    status is self-contradictory with having answerable material — it means
    the model chose not to deliver, not that it was stuck.

    What this DOES NOT block (deliberately, to preserve legitimate paths):
    - finish_with_answer: the intended fast-path terminal — always allowed.
    - finish_research with status != "blocked" (e.g. "done"): allowed.
    - search_web / quick_read: the model may legitimately want more sources
      before answering. Fast-path-ready is not "must answer now", it's
      "cannot claim blocked".
    - fetch_source: ladder escalation is gated separately by the reason check.

    Failure kind is distinct from ignored_required_next_step so traces can tell
    the "blocked lie after fast_path_ready" pattern apart from other violations.
    """
    name = str(next_tool_call.get("name") or "")
    if name != "finish_research":
        return None
    tool_input = next_tool_call.get("input")
    if not isinstance(tool_input, dict):
        return None
    status = str(tool_input.get("status") or "").strip().lower()
    if status != "blocked":
        return None
    return ContractViolation(
        kind="ignored_required_next_step",
        reason=(
            "a quick_read previously returned fast_path_ready=true, meaning a "
            "usable source exists; finish_research(blocked) is self-contradictory "
            "here. Call finish_with_answer to deliver from the quick_read source(s), "
            "or call search_web/quick_read to gather more — but do not claim blocked."
        ),
        failure_kind="blocked_after_fast_path_ready",
    )


def _call_signature(tool_call: dict[str, Any]) -> tuple[str, str]:
    """Stable signature for repeated-call detection: (name, sorted input json)."""
    name = str(tool_call.get("name") or "")
    tool_input = tool_call.get("input")
    if not isinstance(tool_input, dict):
        tool_input = {}
    return (name, json.dumps(tool_input, sort_keys=True, ensure_ascii=False, default=str))


def _terminal_tool_names(tool_specs: list[dict[str, Any]]) -> frozenset[str]:
    """Derive terminal tool names from specs (side_effects == 'terminal')."""
    return frozenset(
        str(spec.get("name") or "")
        for spec in tool_specs
        if str(spec.get("side_effects") or "") == "terminal"
    )
