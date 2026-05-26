from __future__ import annotations

from insightswarm.db.store import Store
from insightswarm.models.clients import BaseModelClient, ModelResult


class AuditedModelClient:
    def __init__(self, inner: BaseModelClient, store: Store):
        self.inner = inner
        self.store = store
        self.provider = getattr(inner, "provider", getattr(getattr(inner, "text", None), "provider", "unknown"))
        self.model = getattr(inner, "model", getattr(getattr(inner, "text", None), "model", "unknown"))

    def complete(
        self,
        messages: list[dict],
        response_format: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        metadata: dict | None = None,
    ) -> ModelResult:
        metadata = metadata or {}
        result = self.inner.complete(
            messages,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
            metadata=metadata,
        )
        run_id = metadata.get("run_id")
        if run_id:
            self.store.record_model_call(
                run_id,
                metadata.get("task_id"),
                result.provider,
                result.model,
                {
                    "messages": messages,
                    "response_format": response_format,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "metadata": metadata,
                    "context_artifact_id": metadata.get("context_artifact_id"),
                },
                {
                    "text": result.text,
                    "json_data": result.json_data,
                    "raw_response": result.raw_response,
                },
                result.usage,
                result.latency_ms,
                result.status,
                result.error,
            )
        return result

    def analyze_image(
        self,
        messages: list[dict],
        images: list[dict],
        response_format: dict | None = None,
        metadata: dict | None = None,
    ) -> ModelResult:
        metadata = metadata or {}
        result = self.inner.analyze_image(
            messages,
            images,
            response_format=response_format,
            metadata=metadata,
        )
        run_id = metadata.get("run_id")
        if run_id:
            self.store.record_model_call(
                run_id,
                metadata.get("task_id"),
                result.provider,
                result.model,
                {
                    "messages": messages,
                    "images": images,
                    "response_format": response_format,
                    "metadata": metadata,
                    "context_artifact_id": metadata.get("context_artifact_id"),
                },
                {
                    "text": result.text,
                    "json_data": result.json_data,
                    "raw_response": result.raw_response,
                },
                result.usage,
                result.latency_ms,
                result.status,
                result.error,
            )
        return result
