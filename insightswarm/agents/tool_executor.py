from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Literal, Optional

from pydantic import AfterValidator, BaseModel, ValidationError, create_model


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    terminal: bool = False
    # When shape validation fails, failure_kind="invalid_input" so the runtime
    # can treat it as a contract violation (counts toward the violation limit)
    # rather than a normal tool failure.
    failure_kind: str | None = None

    def to_message(self) -> dict[str, Any]:
        payload = {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "terminal": self.terminal,
            **self.data,
        }
        if self.error:
            payload["error"] = self.error
        if self.failure_kind:
            payload["failure_kind"] = self.failure_kind
        return payload


def _json_type_to_python(spec: dict[str, Any]) -> Any:
    """Map a JSON-schema property spec to a Python/pydantic type.

    Handles: string/integer/number/boolean/array/object + enum. Nested object
    and array item schemas are not deeply validated here — the contract layer
    and handler remain responsible for semantic/inner-shape checks. This is
    the single source for top-level input shape: type, required, enum.
    """
    enum_values = spec.get("enum")
    if isinstance(enum_values, list) and enum_values:
        # Literal requires hashable, str-compatible values. Non-str enums fall
        # back to Any (handler must validate).
        if all(isinstance(v, str) for v in enum_values):
            return Literal[tuple(enum_values)]
        return Any
    json_type = spec.get("type")
    if json_type == "string":
        return str
    if json_type == "integer":
        return int
    if json_type == "number":
        return float
    if json_type == "boolean":
        return bool
    if json_type == "array":
        return list
    if json_type == "object":
        return dict
    return Any


def _build_input_model(tool_name: str, input_schema: dict[str, Any]) -> type[BaseModel] | None:
    """Build a pydantic model from a tool's input_schema (JSON-schema-like).

    Returns None if the schema has no properties (model-less validation skips
    pydantic and falls back to the simple required-key check). Each tool gets
    one model built at ToolExecutor init time — the single source of truth for
    input shape validation.
    """
    properties = input_schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return None
    required = set(input_schema.get("required") or [])
    fields: dict[str, Any] = {}
    for field_name, field_spec in properties.items():
        if not isinstance(field_spec, dict):
            field_spec = {}
        py_type = _json_type_to_python(field_spec)
        if field_name in required:
            # Required fields: pydantic ... marker. Empty-string is NOT allowed
            # for required string fields (matches prior _missing_required_inputs
            # behavior that treated "" as missing).
            if py_type is str:
                # Annotated[str, AfterValidator] rejects empty strings.
                fields[field_name] = (Annotated[str, AfterValidator(_reject_empty)], ...)
            else:
                fields[field_name] = (py_type, ...)
        else:
            fields[field_name] = (Optional[py_type], None)
    model_name = f"{tool_name}_input".replace("-", "_")
    return create_model(model_name, **fields)


def _reject_empty(value: Any) -> Any:
    """Reject empty strings for required str fields (treat '' as missing)."""
    if isinstance(value, str) and value.strip() == "":
        raise ValueError("must not be empty")
    return value


class ToolExecutor:
    """Execution center: validate input shape via pydantic, then call the handler.

    The pydantic input model is the single source of truth for top-level input
    shape (type, required, enum). Handlers no longer need to re-validate basic
    shape — they only do stateful checks (e.g. quick_read history, URL reach).
    """

    def __init__(self, tool_specs: list[dict[str, Any]], handlers: dict[str, ToolHandler]):
        self.tool_specs = list(tool_specs)
        self.handlers = dict(handlers)
        self._specs_by_name = {str(spec.get("name") or ""): spec for spec in self.tool_specs}
        # Build one pydantic model per tool at init — single source for shape.
        self._input_models: dict[str, type[BaseModel] | None] = {
            name: _build_input_model(name, dict(spec.get("input_schema") or {}))
            for name, spec in self._specs_by_name.items()
        }

    def execute(self, tool_call: dict[str, Any]) -> ToolExecutionResult:
        name = str(tool_call.get("name") or "").strip()
        tool_input = tool_call.get("input")
        if name not in self._specs_by_name:
            return ToolExecutionResult(
                name or "unknown", False, error=f"unknown tool: {name}", terminal=True
            )
        if not isinstance(tool_input, dict):
            return ToolExecutionResult(
                name, False, error="tool input must be an object", failure_kind="invalid_input"
            )
        # Pydantic shape validation (single source for type/required/enum).
        invalid = self._validate_input(name, tool_input)
        if invalid is not None:
            return ToolExecutionResult(
                name, False, error=invalid, failure_kind="invalid_input"
            )
        handler = self.handlers.get(name)
        if handler is None:
            return ToolExecutionResult(
                name, False, error=f"tool handler is not registered: {name}", terminal=True
            )
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

    def _validate_input(self, name: str, tool_input: dict[str, Any]) -> str | None:
        """Validate tool_input against the tool's pydantic model.

        Returns an error message string if invalid, None if valid. When the
        tool has no properties schema, falls back to the simple required-key
        check (preserving prior behavior for propertyless tools).
        """
        model = self._input_models.get(name)
        if model is None:
            # No properties schema: fall back to required-key check.
            schema = dict(self._specs_by_name[name].get("input_schema") or {})
            required = list(schema.get("required") or [])
            missing = [key for key in required if key not in tool_input or tool_input.get(key) in (None, "")]
            if missing:
                return f"missing required input: {', '.join(missing)}"
            return None
        try:
            model.model_validate(tool_input)
        except ValidationError as exc:
            # Compact first-error summary for the model's tool_result feedback.
            errors = exc.errors()
            if errors:
                first = errors[0]
                loc = ".".join(str(part) for part in first.get("loc", ()))
                msg = first.get("msg", "invalid")
                # Include all error locations so the model can fix everything
                # in one retry, not whack-a-mole.
                all_locs = [",".join(str(p) for p in e.get("loc", ())) for e in errors]
                return f"invalid input at '{loc}': {msg} (all errors: {'; '.join(all_locs)})"
            return "invalid input"
        return None
