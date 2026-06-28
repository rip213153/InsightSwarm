from __future__ import annotations

import http.client
import io
import json
import urllib.error

from insightswarm.models.openai_compatible import OpenAICompatibleClient


def test_incomplete_read_returns_model_error(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_API_KEY", "key")
    # tenacity retries with exponential backoff (1s, 2s, 4s); skip the sleeps
    # so the test stays fast while still exercising the retry-then-fail path.
    monkeypatch.setattr("time.sleep", lambda *args, **kwargs: None)

    def _raise_incomplete_read(*args, **kwargs):
        raise http.client.IncompleteRead(b"", 4737)

    monkeypatch.setattr("urllib.request.urlopen", _raise_incomplete_read)
    client = OpenAICompatibleClient(
        provider="test",
        model="model",
        base_url="https://example.com/v1",
        api_key_env="TEST_MODEL_API_KEY",
    )

    result = client.complete([{"role": "user", "content": "hello"}])

    assert result.status == "error"
    assert "response incomplete" in (result.error or "")


def test_full_chat_completions_base_url_is_not_double_appended(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_API_KEY", "key")
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{}"}}], "usage": {}}).encode("utf-8")

    def _capture_request(request, *args, **kwargs):
        del args, kwargs
        captured["url"] = request.full_url
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", _capture_request)
    client = OpenAICompatibleClient(
        provider="test",
        model="model",
        base_url="https://example.com/v1/chat/completions",
        api_key_env="TEST_MODEL_API_KEY",
    )

    result = client.complete([{"role": "user", "content": "hello"}])

    assert result.status == "ok"
    assert captured["url"] == "https://example.com/v1/chat/completions"


def test_retry_after_header_honored(monkeypatch):
    """Regression: tenacity migration must still honor Retry-After on 429."""
    monkeypatch.setenv("TEST_MODEL_API_KEY", "key")
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s, *a, **k: sleeps.append(s))

    def _raise_429_retry_after(*args, **kwargs):
        headers = http.client.HTTPMessage()
        headers.add_header("Retry-After", "10")
        raise urllib.error.HTTPError(
            "https://example.com/v1/chat/completions",
            429,
            "Too Many Requests",
            headers,
            io.BytesIO(b"{}"),
        )

    monkeypatch.setattr("urllib.request.urlopen", _raise_429_retry_after)
    client = OpenAICompatibleClient(
        provider="test",
        model="model",
        base_url="https://example.com/v1",
        api_key_env="TEST_MODEL_API_KEY",
    )

    result = client.complete([{"role": "user", "content": "hello"}])

    assert result.status == "error"
    assert "HTTP 429" in (result.error or "")
    # tenacity sleeps once between each of the 4 attempts (3 sleeps total);
    # every sleep must use the Retry-After value of 10 seconds, not the
    # exponential backoff (1s/2s/4s).
    assert sleeps
    assert all(s == 10.0 for s in sleeps)
