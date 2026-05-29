from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from insightswarm.util import loads

MAX_TASK_DEPENDENCIES = 3


@dataclass(frozen=True)
class Task:
    run_id: str
    kind: str
    status: str
    owner_role: str
    inputs: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    priority: int = 0
    lease_until: str | None = None
    created_by: str = "system"
    task_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        if len(self.depends_on) > MAX_TASK_DEPENDENCIES:
            raise ValueError(
                f"task depends_on exceeds Phase 49 limit of {MAX_TASK_DEPENDENCIES}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Task:
        return cls(
            task_id=row["task_id"],
            run_id=row["run_id"],
            kind=row["kind"],
            status=row["status"],
            owner_role=row["owner_role"],
            inputs=loads(row["inputs_json"], {}),
            depends_on=loads(row["depends_on_json"], []),
            priority=row["priority"],
            lease_until=row["lease_until"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True)
class Message:
    run_id: str
    from_role: str
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    to_role: str | None = None
    broadcast: bool = False
    related_task_id: str | None = None
    message_id: str | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        if self.broadcast and self.to_role:
            raise ValueError("broadcast message cannot target a specific role")
        if not self.broadcast and not self.to_role:
            raise ValueError("message requires to_role unless broadcast is true")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Message:
        return cls(
            message_id=row["message_id"],
            run_id=row["run_id"],
            from_role=row["from_role"],
            to_role=row["to_role"],
            broadcast=bool(row["broadcast"]),
            type=row["intent"],
            payload=loads(row["payload_json"], {}),
            related_task_id=row["related_task_id"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class Artifact:
    run_id: str
    type: str
    status: str
    payload_ref: str
    summary: str
    source_task_id: str | None = None
    artifact_id: str | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Artifact:
        return cls(
            artifact_id=row["artifact_id"],
            run_id=row["run_id"],
            type=row["type"],
            status=row["status"],
            source_task_id=row["source_task_id"],
            payload_ref=row["payload_ref"],
            summary=row["summary"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class Evidence:
    run_id: str
    artifact_id: str
    source_url: str
    quote: str
    freshness: str | None
    confidence: float
    qa_state: str
    evidence_id: str | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Evidence:
        return cls(
            evidence_id=row["evidence_id"],
            run_id=row["run_id"],
            artifact_id=row["artifact_id"],
            source_url=row["source_url"],
            quote=row["quote"],
            freshness=row["freshness"],
            confidence=row["confidence"],
            qa_state=row["qa_state"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class RunState:
    run_id: str
    objective: str
    phase: str
    budget: dict[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None
    delivery_gate: bool = False
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> RunState:
        return cls(
            run_id=row["run_id"],
            objective=row["objective"],
            phase=row["phase"],
            budget=loads(row["budget_json"], {}),
            stop_reason=row["stop_reason"],
            delivery_gate=bool(row["delivery_gate"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass(frozen=True)
class BoardItem:
    run_id: str
    kind: str
    status: str
    title: str
    payload: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    evidence_id: str | None = None
    artifact_id: str | None = None
    source_task_id: str | None = None
    dedupe_key: str | None = None
    priority: int = 0
    created_by: str = "system"
    item_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> BoardItem:
        return cls(
            item_id=row["item_id"],
            run_id=row["run_id"],
            kind=row["kind"],
            status=row["status"],
            title=row["title"],
            payload=loads(row["payload_json"], {}),
            parent_id=row["parent_id"],
            evidence_id=row["evidence_id"],
            artifact_id=row["artifact_id"],
            source_task_id=row["source_task_id"],
            dedupe_key=row["dedupe_key"],
            priority=row["priority"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
