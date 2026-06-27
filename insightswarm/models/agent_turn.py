from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ToolCallInput(BaseModel):
    """A single tool call selected by the model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class AgentTurn(BaseModel):
    """Structured schema for one agent loop round.

    Replaces ad-hoc dict validation. The runtime validates every model
    response against this schema; validation failure is treated as
    ``model_error`` (no separate ``invalid_json`` failure kind). This is
    the second defense line after the API layer's JSON mode / structured
    outputs: providers that support ``response_format=json_schema`` make
    validation a near-passthrough; providers that only support
    ``json_object`` rely on this layer to catch shape violations.
    """

    model_config = ConfigDict(extra="allow")

    assistant_text: str = ""
    private_state: dict[str, Any] | None = None
    tool_call: ToolCallInput | None = None
    stop_reason: str | None = None


def parse_agent_turn(data: Any) -> tuple[AgentTurn | None, str | None]:
    """Validate ``data`` against :class:`AgentTurn`.

    Returns ``(turn, None)`` on success or ``(None, error_message)`` on
    failure. The error message is short (for trace/tool_result consumption)
    and never leaks full validation errors into the model's reasoning layer.
    """
    if not isinstance(data, dict):
        return None, f"model returned non-object JSON: {type(data).__name__}"
    try:
        return AgentTurn.model_validate(data), None
    except ValidationError as exc:
        # Compact first-error summary; full detail lives in trace.
        try:
            first = exc.errors()[0]
            loc = ".".join(str(part) for part in first.get("loc", ()))
            msg = first.get("msg", "invalid")
            return None, f"agent_turn schema error at '{loc}': {msg}"
        except (IndexError, AttributeError):
            return None, "agent_turn schema validation failed"
