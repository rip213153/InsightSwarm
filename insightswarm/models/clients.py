from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelResult:
    text: str
    json_data: dict[str, Any] | None
    provider: str
    model: str
    usage: dict[str, Any]
    latency_ms: int
    raw_response: dict[str, Any]
    status: str
    error: str | None = None


@dataclass(frozen=True)
class ImageInput:
    artifact_id: str | None
    mime_type: str
    data: bytes | None = None
    path: str | None = None
    source_url: str | None = None


class BaseModelClient(Protocol):
    def complete(
        self,
        messages: list[dict],
        response_format: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        metadata: dict | None = None,
    ) -> ModelResult:
        ...


class VisionModelClient(Protocol):
    def analyze_image(
        self,
        messages: list[dict],
        images: list[dict],
        response_format: dict | None = None,
        metadata: dict | None = None,
    ) -> ModelResult:
        ...
