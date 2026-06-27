from __future__ import annotations

import base64
import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from insightswarm.models.clients import ModelResult


class OpenAICompatibleConfigError(RuntimeError):
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


class OpenAICompatibleClient:
    def __init__(
        self,
        provider: str,
        model: str,
        *,
        base_url: str,
        api_key_env: str,
        timeout_seconds: float = 60.0,
    ):
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds

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
        return self._post_chat(
            messages,
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
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
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ModelResult:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise OpenAICompatibleConfigError(f"{self.api_key_env} is required for provider '{self.provider}'.")
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if response_format:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        started = time.perf_counter()
        request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        max_retries = 3
        raw: dict[str, Any] = {}
        for attempt in range(max_retries + 1):
            request = urllib.request.Request(
                self._chat_completions_url(),
                data=request_body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                status_code = exc.code
                if attempt < max_retries and self._should_retry_status(status_code):
                    time.sleep(self._retry_delay(exc, attempt))
                    continue
                return self._error_result(
                    f"OpenAI-compatible API HTTP {status_code}: {detail[:500]}",
                    started,
                    {"http_status": status_code},
                )
            except urllib.error.URLError as exc:
                if attempt < 1:
                    time.sleep(self._retry_delay(None, attempt))
                    continue
                return self._error_result(f"OpenAI-compatible API request failed: {exc.reason}", started)
            except (TimeoutError, socket.timeout) as exc:
                if attempt < 1:
                    time.sleep(self._retry_delay(None, attempt))
                    continue
                return self._error_result(f"OpenAI-compatible API request timed out: {exc}", started)
            except http.client.IncompleteRead as exc:
                if attempt < 1:
                    time.sleep(self._retry_delay(None, attempt))
                    continue
                return self._error_result(f"OpenAI-compatible API response incomplete: {exc}", started)
            except (ConnectionResetError, OSError) as exc:
                if attempt < 1:
                    time.sleep(self._retry_delay(None, attempt))
                    continue
                return self._error_result(f"OpenAI-compatible API connection failed: {exc}", started)
            except json.JSONDecodeError as exc:
                return self._error_result(f"OpenAI-compatible API returned invalid JSON: {exc}", started)
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

    def _should_retry_status(self, status_code: int) -> bool:
        return status_code == 429 or 500 <= status_code < 600

    def _chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _retry_delay(self, exc: urllib.error.HTTPError | None, attempt: int) -> float:
        if exc is not None:
            retry_after = exc.headers.get("Retry-After") or exc.headers.get("retry-after")
            if retry_after:
                try:
                    return min(float(retry_after), 20.0)
                except ValueError:
                    pass
        return min(2.0 ** attempt, 8.0)

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
