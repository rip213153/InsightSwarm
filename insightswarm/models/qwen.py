from __future__ import annotations

import base64
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from insightswarm.models.clients import ModelResult


class QwenConfigError(RuntimeError):
    pass


def _json_from_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json\n", "", 1).replace("JSON\n", "", 1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


class QwenOpenAICompatibleClient:
    def __init__(
        self,
        provider: str,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
    ):
        self.provider = provider
        self.model = model
        self.base_url = (base_url or os.getenv("INSIGHTSWARM_QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.api_key_env = api_key_env

    def complete(
        self,
        messages: list[dict],
        response_format: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        metadata: dict | None = None,
    ) -> ModelResult:
        return self._post_chat(
            messages,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def analyze_image(
        self,
        messages: list[dict],
        images: list[dict],
        response_format: dict | None = None,
        metadata: dict | None = None,
    ) -> ModelResult:
        multimodal = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                multimodal.append({"type": "text", "text": content})
        for image in images:
            multimodal.append({"type": "image_url", "image_url": {"url": self._image_url(image)}})
        return self._post_chat(
            [{"role": "user", "content": multimodal}],
            response_format=response_format,
            max_tokens=None,
            temperature=None,
        )

    def _image_url(self, image: dict) -> str:
        if image.get("data_url"):
            return image["data_url"]
        path = image.get("path")
        data = image.get("data")
        mime_type = image.get("mime_type") or "image/png"
        if path:
            data = Path(path).read_bytes()
        if data is None:
            raise ValueError("image input requires path, data, or data_url")
        if isinstance(data, str):
            data = data.encode("utf-8")
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _post_chat(
        self,
        messages: list[dict],
        response_format: dict | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> ModelResult:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise QwenConfigError(f"{self.api_key_env} is required for provider '{self.provider}'.")
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if response_format:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        started = time.perf_counter()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return self._error_result(
                f"Qwen API HTTP {exc.code}: {detail[:500]}",
                started,
                {"http_status": exc.code},
            )
        except urllib.error.URLError as exc:
            return self._error_result(f"Qwen API request failed: {exc.reason}", started)
        except (TimeoutError, socket.timeout) as exc:
            return self._error_result(f"Qwen API request timed out: {exc}", started)
        latency_ms = int((time.perf_counter() - started) * 1000)
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        if isinstance(text, list):
            text = json.dumps(text, ensure_ascii=False)
        return ModelResult(
            text=text,
            json_data=_json_from_text(text),
            provider=self.provider,
            model=self.model,
            usage=raw.get("usage") or {},
            latency_ms=latency_ms,
            raw_response=raw,
            status="ok",
        )

    def _error_result(
        self,
        error: str,
        started: float,
        raw_response: dict[str, Any] | None = None,
    ) -> ModelResult:
        return ModelResult(
            text="",
            json_data=None,
            provider=self.provider,
            model=self.model,
            usage={},
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw_response=raw_response or {},
            status="error",
            error=error,
        )
