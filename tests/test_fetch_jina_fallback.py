"""Tests for the fetch layer's Jina Reader fallback and ladder.

Covers:
  - FetchUrlTool tries urllib first; on urllib failure it falls back to Jina
    Reader and tags the result with `fetcher="jina"` / `ladder="L1"`.
  - When `prefer_jina=True` is set, Jina Reader is tried first (L1 quick_read
    path); urllib is the fallback.
  - Jina Reader response parsing: Title extraction, Markdown Content header
    stripping.
  - FirecrawlScrapeTool degrades to Jina Reader when Firecrawl returns too
    little text or when the API key is missing.

Network calls are mocked — no real HTTP is made.
"""
from __future__ import annotations

import io
import json
from typing import Any
from urllib.error import HTTPError, URLError

import pytest

from insightswarm.tools import fetch as fetch_module
from insightswarm.tools import firecrawl as firecrawl_module
from insightswarm.tools.core import ToolContext
from insightswarm.tools.fetch import FetchUrlTool
from insightswarm.tools.firecrawl import FirecrawlScrapeTool


class _FakeResponse:
    def __init__(self, payload: bytes, *, status_code: int = 200):
        self._buf = io.BytesIO(payload)
        self.status = status_code
        self.headers = {}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        self._buf.close()

    def read(self) -> bytes:
        return self._buf.read()


_TEST_CONTEXT = ToolContext(quality_mode="test")


def _set_urlopen_dispatch(monkeypatch, dispatch: dict[str, Any]) -> None:
    """Wire a urlopen mock that dispatches by URL substring.

    Each value is either a bytes payload (returned verbatim), a dict (JSON-encoded),
    or an Exception instance (raised).
    """
    def _mock_urlopen(request, timeout=None):  # noqa: ANN001
        url = request.full_url
        for key, payload in dispatch.items():
            if key in url:
                if isinstance(payload, Exception):
                    raise payload
                if isinstance(payload, (bytes, bytearray)):
                    return _FakeResponse(bytes(payload))
                if isinstance(payload, dict):
                    return _FakeResponse(json.dumps(payload).encode("utf-8"))
                return _FakeResponse(str(payload).encode("utf-8"))
        raise URLError(f"no mock for {url}")

    monkeypatch.setattr(fetch_module.urllib.request, "urlopen", _mock_urlopen)
    # Firecrawl imports urlopen via `import urllib.request`, so patching the
    # shared module is enough — but firecrawl.py calls urllib.request.urlopen
    # through its own imported reference. Patch both to be safe.
    monkeypatch.setattr(firecrawl_module.urllib.request, "urlopen", _mock_urlopen)


# ---------------------------------------------------------------------------
# FetchUrlTool — Jina Reader fallback
# ---------------------------------------------------------------------------


def test_fetch_url_falls_back_to_jina_when_urllib_fails(monkeypatch) -> None:
    """When urllib raises, FetchUrlTool must try Jina Reader before erroring."""
    jina_body = (
        b"Title: Jina Title\n"
        b"URL Source: https://example.com/page\n"
        b"Markdown Content:\n"
        b"This is the markdown body. " + (b"x" * 200) + b"\n"
    )
    _set_urlopen_dispatch(monkeypatch, {
        "r.jina.ai": jina_body,
        "example.com": URLError("connection refused"),  # urllib path fails
    })

    result = FetchUrlTool().run({"url": "https://example.com/page"}, _TEST_CONTEXT)

    assert result.ok
    assert result.data["fetcher"] == "jina"
    assert result.data["ladder"] == "L1"
    assert result.data["fallback_from"] == "urllib"
    assert result.data["title"] == "Jina Title"
    assert "markdown body" in result.data["text"]


def test_fetch_url_prefer_jina_uses_jina_first(monkeypatch) -> None:
    """With prefer_jina=True, Jina Reader is the primary path (L1 quick_read)."""
    jina_body = b"Title: Quick Read\nMarkdown Content:\nQuick read body. " + (b"y" * 200) + b"\n"
    # If urllib is hit at all, fail loudly so the test catches a regression.
    _set_urlopen_dispatch(monkeypatch, {
        "r.jina.ai": jina_body,
        "example.com": URLError("urllib should not be called when prefer_jina succeeds"),
    })

    result = FetchUrlTool().run({"url": "https://example.com/page", "prefer_jina": True}, _TEST_CONTEXT)

    assert result.ok
    assert result.data["fetcher"] == "jina"
    assert result.data["ladder"] == "L1"
    assert result.data["title"] == "Quick Read"
    assert "Quick read body" in result.data["text"]


def test_fetch_url_prefer_jina_falls_back_to_urllib(monkeypatch) -> None:
    """If Jina Reader fails when prefer_jina=True, urllib is the fallback."""
    html = b"<html><head><title>HTML Title</title></head><body><p>Body text. " + (b"p" * 200) + b"</p></body></html>"
    _set_urlopen_dispatch(monkeypatch, {
        "r.jina.ai": URLError("jina down"),
        "example.com": html,
    })

    result = FetchUrlTool().run({"url": "https://example.com/page", "prefer_jina": True}, _TEST_CONTEXT)

    assert result.ok
    # Jina failed → urllib was used → fetcher is urllib, ladder is L2.
    assert result.data["fetcher"] == "urllib"
    assert result.data["ladder"] == "L2"
    assert result.data["title"] == "HTML Title"


def test_fetch_url_returns_error_when_both_urllib_and_jina_fail(monkeypatch) -> None:
    _set_urlopen_dispatch(monkeypatch, {
        "r.jina.ai": URLError("jina down"),
        "example.com": URLError("urllib down"),
    })

    result = FetchUrlTool().run({"url": "https://example.com/page"}, _TEST_CONTEXT)

    assert not result.ok
    assert "Fetch failed" in (result.error or "")


def test_fetch_url_blocks_truncated_url() -> None:
    """URLs with ellipsis markers must be blocked before any network call."""
    result = FetchUrlTool().run({"url": "https://example.com/foo..."}, _TEST_CONTEXT)
    assert not result.ok
    assert result.status == "blocked"
    assert "truncated" in (result.error or "")


# ---------------------------------------------------------------------------
# Jina Reader response parsing helpers
# ---------------------------------------------------------------------------


def test_extract_jina_title() -> None:
    raw = "Title: My Page Title\nURL Source: https://x.com\nMarkdown Content:\nbody"
    assert fetch_module._extract_jina_title(raw) == "My Page Title"


def test_extract_jina_title_returns_empty_when_missing() -> None:
    assert fetch_module._extract_jina_title("no title line here") == ""


def test_strip_jina_header_drops_metadata_block() -> None:
    raw = "Title: T\nURL Source: https://x.com\nMarkdown Content:\nactual body"
    assert fetch_module._strip_jina_header(raw) == "actual body"


def test_strip_jina_header_returns_body_when_no_marker() -> None:
    assert fetch_module._strip_jina_header("just body text") == "just body text"


# ---------------------------------------------------------------------------
# FirecrawlScrapeTool — L1 fallback when L2 unavailable or low-signal
# ---------------------------------------------------------------------------


def test_firecrawl_falls_back_to_jina_when_api_key_missing(monkeypatch) -> None:
    """Without FIRECRAWL_API_KEY, FirecrawlScrapeTool must serve an L1 Jina result."""
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.delenv("INSIGHTSWARM_FIRECRAWL_API_KEY", raising=False)

    jina_body = b"Title: Fallback Title\nMarkdown Content:\nFallback body. " + (b"z" * 200) + b"\n"
    _set_urlopen_dispatch(monkeypatch, {"r.jina.ai": jina_body})

    result = FirecrawlScrapeTool().run({"url": "https://example.com/page"}, _TEST_CONTEXT)

    assert result.ok
    assert result.data["fetcher"] == "jina_fallback"
    assert result.data["ladder"] == "L1"
    assert result.data["title"] == "Fallback Title"
    assert "Fallback body" in result.data["text"]
    assert result.data["metadata"]["fallback_reason"] == "missing Firecrawl API key"


def test_firecrawl_falls_back_to_jina_when_response_low_signal(monkeypatch) -> None:
    """When Firecrawl returns < 100 chars, degrade to Jina Reader."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
    firecrawl_response = {
        "data": {
            "markdown": "short",  # < 100 chars → low signal
            "html": "<p>short</p>",
            "metadata": {"title": "Short", "sourceURL": "https://example.com/p"},
        }
    }
    jina_body = b"Title: Better Via Jina\nMarkdown Content:\nMuch longer body text via Jina Reader. " + (b"j" * 200) + b"\n"
    _set_urlopen_dispatch(monkeypatch, {
        "api.firecrawl.dev": firecrawl_response,
        "r.jina.ai": jina_body,
    })

    result = FirecrawlScrapeTool().run({"url": "https://example.com/p"}, _TEST_CONTEXT)

    assert result.ok
    assert result.data["fetcher"] == "jina_fallback"
    assert result.data["ladder"] == "L1"
    assert result.data["title"] == "Better Via Jina"
    assert "Firecrawl returned too little text" in result.data["metadata"]["fallback_reason"]


def test_firecrawl_returns_firecrawl_result_when_text_sufficient(monkeypatch) -> None:
    """When Firecrawl returns enough text, no fallback — fetcher stays firecrawl."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")
    long_markdown = "Real Firecrawl body. " + ("content " * 50)
    firecrawl_response = {
        "data": {
            "markdown": long_markdown,
            "html": "<p>content</p>",
            "metadata": {"title": "Firecrawl Title", "sourceURL": "https://example.com/p"},
        }
    }
    _set_urlopen_dispatch(monkeypatch, {
        "api.firecrawl.dev": firecrawl_response,
        "r.jina.ai": URLError("jina should not be called"),
    })

    result = FirecrawlScrapeTool().run({"url": "https://example.com/p"}, _TEST_CONTEXT)

    assert result.ok
    assert result.data["fetcher"] == "firecrawl"
    assert result.data["ladder"] == "L2"
    assert result.data["title"] == "Firecrawl Title"
    # Firecrawl strips the markdown before returning — compare against the
    # stripped form so a trailing-space difference doesn't cause a false failure.
    assert long_markdown.strip() in result.data["text"]


def test_firecrawl_falls_back_to_jina_on_http_error(monkeypatch) -> None:
    """When Firecrawl returns an HTTP error, degrade to Jina Reader."""
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-key")

    firecrawl_error = HTTPError(
        "https://api.firecrawl.dev/v1/scrape", 403, "Forbidden", {}, io.BytesIO(b'{"error":"forbidden"}')
    )
    jina_body = b"Title: Jina After 403\nMarkdown Content:\nRecovered via Jina. " + (b"r" * 200) + b"\n"
    _set_urlopen_dispatch(monkeypatch, {
        "api.firecrawl.dev": firecrawl_error,
        "r.jina.ai": jina_body,
    })

    result = FirecrawlScrapeTool().run({"url": "https://example.com/p"}, _TEST_CONTEXT)

    assert result.ok
    assert result.data["fetcher"] == "jina_fallback"
    assert "HTTP 403" in result.data["metadata"]["fallback_reason"]
    assert result.data["title"] == "Jina After 403"
