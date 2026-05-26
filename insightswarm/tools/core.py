from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolContext:
    run_id: str | None = None
    task_id: str | None = None
    quality_mode: str = "production"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    status: str
    data: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    safety_policy: dict[str, Any]
    allowed_callers: list[str]
    side_effect_level: str
    network_access: str
    blocked_inputs: list[str]
    example_failures: list[dict[str, Any]]
    examples: list[dict[str, Any]]

    def run(self, tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        ...
