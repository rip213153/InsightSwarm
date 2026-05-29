from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    terminal: bool = False

    def to_message(self) -> dict[str, Any]:
        payload = {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "terminal": self.terminal,
            **self.data,
        }
        if self.error:
            payload["error"] = self.error
        return payload


class ToolExecutor:
    """Thin execution center: validate a model-selected tool and call its handler."""

    def __init__(self, tool_specs: list[dict[str, Any]], handlers: dict[str, ToolHandler]):
        self.tool_specs = list(tool_specs)
        self.handlers = dict(handlers)
        self._specs_by_name = {str(spec.get("name") or ""): spec for spec in self.tool_specs}

    def execute(self, tool_call: dict[str, Any]) -> ToolExecutionResult:
        name = str(tool_call.get("name") or "").strip()
        tool_input = tool_call.get("input")
        if name not in self._specs_by_name:
            return ToolExecutionResult(name or "unknown", False, error=f"unknown tool: {name}", terminal=True)
        if not isinstance(tool_input, dict):
            return ToolExecutionResult(name, False, error="tool input must be an object")
        missing = self._missing_required_inputs(name, tool_input)
        if missing:
            return ToolExecutionResult(name, False, error=f"missing required input: {', '.join(missing)}")
        handler = self.handlers.get(name)
        if handler is None:
            return ToolExecutionResult(name, False, error=f"tool handler is not registered: {name}", terminal=True)
        try:
            result = handler(dict(tool_input))
        except Exception as exc:
            return ToolExecutionResult(name, False, error=f"{type(exc).__name__}: {exc}")
        if not isinstance(result, dict):
            return ToolExecutionResult(name, False, error="tool handler returned a non-object result")
        return ToolExecutionResult(
            name,
            bool(result.pop("ok", True)),
            data=result,
            error=result.pop("error", None),
            terminal=bool(result.get("terminal")),
        )

    def _missing_required_inputs(self, name: str, tool_input: dict[str, Any]) -> list[str]:
        schema = dict(self._specs_by_name[name].get("input_schema") or {})
        required = list(schema.get("required") or [])
        return [key for key in required if key not in tool_input or tool_input.get(key) in (None, "")]
