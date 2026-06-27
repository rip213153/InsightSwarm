"""Tests for the multi-backend search layer with fallback.

Covers:
  - SEARCH_BACKEND env selects the primary backend
  - Tavily advanced mode requests `include_raw_content` and surfaces `answer`
  - When the primary backend is missing an API key, the fallback chain moves
    on to the next available backend (DuckDuckGo is always available).
  - When all keyed backends fail, DuckDuckGo (keyless) is the last resort.
  - The result shape stays backward-compatible: `data.results[*]` has
    `title`/`url`/`snippet`.

Network calls are mocked — no real HTTP is made.
"""
from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from insightswarm.tools import search as search_module
from insightswarm.tools.search import SearchTool


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        self._buf.close()

    def read(self) -> bytes:
        return self._buf.read()


def _make_urlopen_mock(responses: dict[str, Any], errors: set[str] | None = None):
    """Return a urlopen mock that dispatches by URL substring.

    `responses` maps a URL-substring key to either a dict (JSON body) or a
    bytes payload. `errors` lists keys whose request should raise URLError.
    """
    errors = errors or set()

    def _mock_urlopen(request, timeout=None):  # noqa: ANN001
        url = request.full_url
        for key, payload in responses.items():
            if key in url:
                if key in errors:
                    raise URLError("simulated network error")
                if isinstance(payload, (bytes, bytearray)):
                    return _FakeResponse(bytes(payload))
                return _FakeResponse(json.dumps(payload).encode("utf-8"))
        raise URLError(f"no mock for {url}")

    return _mock_urlopen


def test_search_backend_env_selects_primary(monkeypatch) -> None:
    """SEARCH_BACKEND env var selects the primary backend."""
    monkeypatch.setenv("SEARCH_BACKEND", "brave")
    monkeypatch.setenv("BRAVE_API_KEY", "test-brave-key")
    monkeypatch.setattr(search_module.urllib.request, "urlopen",
                        _make_urlopen_mock({"api.search.brave.com": {"web": {"results": [
                            {"title": "Brave result", "url": "https://b.example/x", "description": "brave snippet"}
                        ]}}}))

    result = SearchTool().run({"query": "test", "limit": 5})

    assert result.ok
    assert result.data["provider"] == "brave"
    assert result.data["requested_provider"] == "brave"
    assert result.diagnostics["fallback_used"] is False
    assert len(result.data["results"]) == 1
    assert result.data["results"][0]["title"] == "Brave result"
    assert result.data["results"][0]["snippet"] == "brave snippet"


def test_search_falls_back_when_primary_key_missing(monkeypatch) -> None:
    """When the primary backend's API key is missing, fall through to the next
    available backend. DuckDuckGo (keyless) is always tried last.
    """
    # No TAVILY_API_KEY, no BRAVE_API_KEY, no SERPER_API_KEY — only DuckDuckGo
    # can serve the request. The fallback chain must reach it.
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("INSIGHTSWARM_TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("SEARCH_BACKEND", raising=False)

    ddg_html = (
        b'<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fddg&rut=abc">DDG Title</a>'
        b'<a class="result__snippet" href="">DDG snippet text here</a>'
    )
    monkeypatch.setattr(search_module.urllib.request, "urlopen",
                        _make_urlopen_mock({"duckduckgo.com/html": ddg_html}))

    result = SearchTool().run({"query": "test", "limit": 5})

    assert result.ok
    assert result.data["provider"] == "duckduckgo"
    assert result.diagnostics["fallback_used"] is True
    assert "tavily" in result.diagnostics["fallback_chain"]
    assert "duckduckgo" in result.diagnostics["fallback_chain"]
    assert len(result.data["results"]) == 1
    assert result.data["results"][0]["url"] == "https://example.com/ddg"
    assert result.data["results"][0]["title"] == "DDG Title"


def test_search_falls_back_on_backend_http_error(monkeypatch) -> None:
    """When the primary backend returns an HTTP error, fall through to the next."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("BRAVE_API_KEY", "test-brave-key")
    monkeypatch.delenv("SEARCH_BACKEND", raising=False)

    # Tavily endpoint returns 500 (retryable, will exhaust retries then fall back).
    # Brave endpoint returns a valid response.
    def _mock_urlopen(request, timeout=None):  # noqa: ANN001
        url = request.full_url
        if "api.tavily.com" in url:
            raise HTTPError(url, 500, "Internal Server Error", {}, io.BytesIO(b"{}"))
        if "api.search.brave.com" in url:
            return _FakeResponse(json.dumps({"web": {"results": [
                {"title": "Brave fallback", "url": "https://b.example/y", "description": "fallback snippet"}
            ]}}).encode("utf-8"))
        raise URLError(f"no mock for {url}")

    # Speed up the test: disable tenacity's wait between retries.
    monkeypatch.setattr(search_module, "_RETRY_DECORATOR",
                        search_module.retry(
                            stop=search_module.stop_after_attempt(1),
                            wait=search_module.wait_exponential(multiplier=0, min=0, max=0),
                            retry=search_module.retry_if_exception_type(
                                (search_module._RetryableHTTPError, search_module.urllib.error.URLError, TimeoutError)
                            ),
                            reraise=True,
                        ))

    monkeypatch.setattr(search_module.urllib.request, "urlopen", _mock_urlopen)

    result = SearchTool().run({"query": "test", "limit": 5})

    assert result.ok
    assert result.data["provider"] == "brave"
    assert result.diagnostics["fallback_used"] is True
    assert result.data["results"][0]["title"] == "Brave fallback"


def test_tavily_advanced_mode_requests_raw_content(monkeypatch) -> None:
    """When advanced mode is on, Tavily is asked for include_raw_content and
    the per-result raw_content is surfaced on the returned results.
    """
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.delenv("SEARCH_BACKEND", raising=False)

    captured_payload: dict[str, Any] = {}

    def _mock_urlopen(request, timeout=None):  # noqa: ANN001
        captured_payload.update(json.loads(request.data.decode("utf-8")))
        return _FakeResponse(json.dumps({
            "answer": "Synthesized answer",
            "results": [
                {"title": "T1", "url": "https://t.example/1", "content": "snippet", "raw_content": "full body text"},
            ],
        }).encode("utf-8"))

    monkeypatch.setattr(search_module.urllib.request, "urlopen", _mock_urlopen)

    result = SearchTool().run({"query": "test", "limit": 5, "advanced": True})

    assert result.ok
    assert captured_payload["search_depth"] == "advanced"
    assert captured_payload["include_raw_content"] is True
    assert result.data["answer"] == "Synthesized answer"
    assert result.data["results"][0]["raw_content"] == "full body text"


def test_search_returns_error_when_all_backends_fail(monkeypatch) -> None:
    """When every backend fails (including DuckDuckGo network error), return
    an error result with the list of attempted backends.
    """
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("SEARCH_BACKEND", raising=False)

    # Disable retries so the test is fast.
    monkeypatch.setattr(search_module, "_RETRY_DECORATOR",
                        search_module.retry(
                            stop=search_module.stop_after_attempt(1),
                            wait=search_module.wait_exponential(multiplier=0, min=0, max=0),
                            retry=search_module.retry_if_exception_type(
                                (search_module._RetryableHTTPError, search_module.urllib.error.URLError, TimeoutError)
                            ),
                            reraise=True,
                        ))

    def _mock_urlopen(request, timeout=None):  # noqa: ANN001
        raise URLError("network down")

    monkeypatch.setattr(search_module.urllib.request, "urlopen", _mock_urlopen)

    result = SearchTool().run({"query": "test", "limit": 5})

    assert not result.ok
    assert "All search backends failed" in (result.error or "")
    assert "duckduckgo" in result.diagnostics["tried_backends"]


def test_search_empty_query_returns_error(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    result = SearchTool().run({"query": "", "limit": 5})
    assert not result.ok
    assert "non-empty query" in (result.error or "")


def test_search_provider_input_overrides_env(monkeypatch) -> None:
    """Explicit `provider` in tool_input takes precedence over SEARCH_BACKEND."""
    monkeypatch.setenv("SEARCH_BACKEND", "brave")
    monkeypatch.setenv("BRAVE_API_KEY", "bk")
    monkeypatch.setenv("TAVILY_API_KEY", "tk")

    def _mock_urlopen(request, timeout=None):  # noqa: ANN001
        if "api.tavily.com" in request.full_url:
            return _FakeResponse(json.dumps({"results": [
                {"title": "T", "url": "https://t.example/x", "content": "s"}
            ]}).encode("utf-8"))
        raise URLError("unexpected brave call")

    monkeypatch.setattr(search_module.urllib.request, "urlopen", _mock_urlopen)

    # provider=tavily overrides SEARCH_BACKEND=brave
    result = SearchTool().run({"query": "q", "limit": 5, "provider": "tavily"})

    assert result.ok
    assert result.data["provider"] == "tavily"
    assert result.data["requested_provider"] == "tavily"


def test_duckduckgo_link_unwrapping() -> None:
    """DDG wraps result URLs in a redirect — the helper must extract the real URL."""
    assert search_module._unwrap_ddg_link("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fp&rut=x") == "https://example.com/p"
    assert search_module._unwrap_ddg_link("https://example.com/direct") == "https://example.com/direct"
    assert search_module._unwrap_ddg_link("//example.com/protocol-relative") == "https://example.com/protocol-relative"
    assert search_module._unwrap_ddg_link("") == ""
