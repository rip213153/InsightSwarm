from __future__ import annotations

from insightswarm.models.clients import ModelResult


class FakeModelClient:
    provider = "fake"
    model = "fake-deterministic-v1"

    def complete(
        self,
        messages: list[dict],
        response_format: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        metadata: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ModelResult:
        metadata = metadata or {}
        role = metadata.get("role", "unknown")
        json_data = {"role": role, "ok": True}
        return ModelResult(
            text=f"fake response for {role}",
            json_data=json_data,
            provider=self.provider,
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            latency_ms=1,
            raw_response={"messages": messages, "metadata": metadata},
            status="ok",
        )

    def analyze_image(
        self,
        messages: list[dict],
        images: list[dict],
        response_format: dict | None = None,
        metadata: dict | None = None,
    ) -> ModelResult:
        metadata = metadata or {}
        return ModelResult(
            text="fake vision response",
            json_data={"bbox": [0.18, 0.12, 0.62, 0.88], "ok": True},
            provider=self.provider,
            model="fake-vision-v1",
            usage={"prompt_tokens": 12, "completion_tokens": 6},
            latency_ms=1,
            raw_response={"messages": messages, "images": images, "metadata": metadata},
            status="ok",
        )

