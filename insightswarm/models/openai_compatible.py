from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import orjson
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
)

from insightswarm.models.clients import ModelResult
from insightswarm.tools.http_utils import HttpRequestError, HttpResponseError, request_json


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


def _is_retryable(exc: BaseException) -> bool:
    """Retry network errors and HTTP 429/5xx; let other HTTP errors bubble up."""
    if isinstance(exc, HttpRequestError):
        return True
    if isinstance(exc, HttpResponseError):
        return exc.status_code == 429 or 500 <= exc.status_code < 600
    return False


def _retry_wait(retry_state: RetryCallState) -> float:
    """Honor Retry-After (cap 20s); otherwise exponential backoff capped at 8s."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, HttpResponseError) and exc.headers:
        retry_after = exc.headers.get("Retry-After") or exc.headers.get("retry-after")
        if retry_after:
            try:
                return min(float(retry_after), 20.0)
            except ValueError:
                pass
    attempt = retry_state.attempt_number - 1
    return min(2.0 ** attempt, 8.0)


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
            raise OpenAICompatibleConfigError(f"{self.api_key_env} is required for provider '{self.provider}'.")
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if response_format:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        started = time.perf_counter()
        # Hand-written retry loop (formerly ~80 lines) replaced by tenacity:
        # 3 retries with exponential backoff (1s, 2s, 4s, cap 8s) on network
        # errors and HTTP 429/5xx, honoring Retry-After (cap 20s). See git
        # history for the original inline implementation.
        retrying = Retrying(
            stop=stop_after_attempt(4),
            wait=_retry_wait,
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        )
        try:
            raw = retrying(
                request_json,
                self._chat_completions_url(),
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                body=payload,
                timeout=self.timeout_seconds,
            )
        except HttpResponseError as exc:
            return self._error_result(
                f"OpenAI-compatible API HTTP {exc.status_code}: {exc.body[:500]}",
                started,
                {"http_status": exc.status_code},
            )
        except HttpRequestError as exc:
            return self._error_result(f"OpenAI-compatible API request failed: {exc}", started)
        except (orjson.JSONDecodeError, json.JSONDecodeError) as exc:
            return self._error_result(f"OpenAI-compatible API returned invalid JSON: {exc}", started)
        if not isinstance(raw, dict):
            raw = {}
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

    def _chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

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
