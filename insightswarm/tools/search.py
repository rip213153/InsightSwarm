from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from insightswarm.tools.core import ToolContext, ToolResult


# Ordered fallback chain. The primary backend is selected by SEARCH_BACKEND
# (env) or `provider` (tool input); on failure we walk this list, skipping the
# primary and any backend whose API key is missing. DuckDuckGo is always
# available as the last-resort fallback because it needs no key.
_BACKEND_CHAIN: tuple[str, ...] = ("tavily", "brave", "serper", "duckduckgo")


class SearchTool:
    name = "search.web"

    def run(self, tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        del context
        # `provider` (legacy input) takes precedence, then SEARCH_BACKEND env,
        # then default "tavily". This keeps the existing call sites working
        # while allowing ops to switch backends without code changes.
        provider = str(tool_input.get("provider") or os.getenv("SEARCH_BACKEND") or "tavily").lower().strip()
        query = str(tool_input.get("query") or "")
        limit = int(tool_input.get("limit") or 5)
        # Tavily advanced mode: opt in via tool input or TAVILY_SEARCH_DEPTH=advanced.
        # Advanced search returns `answer` and per-result `raw_content` fields.
        tavily_advanced = bool(tool_input.get("advanced")) or os.getenv("TAVILY_SEARCH_DEPTH", "").lower() == "advanced"

        if not query:
            return ToolResult(
                "error",
                error="search.web requires a non-empty query",
                provenance={"tool": self.name, "provider": provider},
            )

        # Build the ordered fallback list: requested provider first, then the
        # rest of the chain (deduped, preserving order).
        chain: list[str] = [provider] + [b for b in _BACKEND_CHAIN if b != provider]

        started = time.perf_counter()
        last_error: str = ""
        last_provider: str = provider
        tried: list[str] = []
        for backend in chain:
            api_key = _backend_api_key(backend)
            if backend != "duckduckgo" and not api_key:
                # Skip backends we can't authenticate against; they would just
                # fail with the same auth error every retry.
                tried.append(backend)
                last_error = f"{backend}: missing API key"
                last_provider = backend
                continue
            tried.append(backend)
            try:
                results, answer = _run_backend(backend, query=query, limit=limit, api_key=api_key, advanced=tavily_advanced)
            except _SearchBackendError as exc:
                last_error = f"{backend}: {exc}"
                last_provider = backend
                continue
            except Exception as exc:  # pragma: no cover - defensive
                last_error = f"{backend}: {exc}"
                last_provider = backend
                continue
            return ToolResult(
                "ok",
                data={
                    "provider": backend,
                    "requested_provider": provider,
                    "status": "ok",
                    "results": results,
                    **({"answer": answer} if answer else {}),
                },
                diagnostics={
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "result_count": len(results),
                    "fallback_chain": tried,
                    "fallback_used": backend != provider,
                },
                provenance={"tool": self.name, "provider": backend},
            )

        return ToolResult(
            "error",
            error=f"All search backends failed; last error: {last_error}",
            diagnostics={"tried_backends": tried, "last_provider": last_provider},
            provenance={"tool": self.name, "provider": last_provider},
        )


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------


class _SearchBackendError(Exception):
    """Raised by a backend to signal the caller should try the next one."""


def _backend_api_key(backend: str) -> str | None:
    if backend == "tavily":
        return os.getenv("TAVILY_API_KEY") or os.getenv("INSIGHTSWARM_TAVILY_API_KEY")
    if backend == "brave":
        return os.getenv("BRAVE_API_KEY") or os.getenv("INSIGHTSWARM_BRAVE_API_KEY")
    if backend == "serper":
        return os.getenv("SERPER_API_KEY") or os.getenv("INSIGHTSWARM_SERPER_API_KEY")
    return None


def _run_backend(backend: str, *, query: str, limit: int, api_key: str | None, advanced: bool) -> tuple[list[dict[str, Any]], str]:
    """Run one backend with retry; raise _SearchBackendError on failure."""
    if backend == "tavily":
        return _run_tavily(query=query, limit=limit, api_key=api_key or "", advanced=advanced)
    if backend == "brave":
        return _run_brave(query=query, limit=limit, api_key=api_key or "")
    if backend == "serper":
        return _run_serper(query=query, limit=limit, api_key=api_key or "")
    if backend == "duckduckgo":
        return _run_duckduckgo(query=query, limit=limit)
    raise _SearchBackendError(f"unknown backend: {backend}")


class _RetryableHTTPError(Exception):
    """Wrapper for HTTP 5xx / 429 that should be retried."""

    def __init__(self, code: int, body: str) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code
        self.body = body


# Retries: 3 attempts, exponential backoff 1s -> 2s -> 4s. Retry on network
# errors and HTTP 5xx. Non-retryable failures (4xx, missing key, parse errors)
# propagate immediately so the fallback chain can move on.
_RETRY_DECORATOR = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((_RetryableHTTPError, urllib.error.URLError, TimeoutError)),
    reraise=True,
)


def _http_request_json(
    *,
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: bytes | None = None,
    timeout: float = 20.0,
) -> Any:
    """Send an HTTP request and return parsed JSON. Raises _RetryableHTTPError
    on 5xx/429 (so the tenacity decorator retries), _SearchBackendError on
    other 4xx (no retry, propagate to fallback chain).
    """
    request = urllib.request.Request(url, data=payload, headers=headers or {}, method=method)

    @_RETRY_DECORATOR
    def _do() -> Any:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                return json.loads(raw.decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            if exc.code >= 500 or exc.code == 429:
                # Retryable — raise wrapper that tenacity knows about.
                raise _RetryableHTTPError(exc.code, body)
            # 4xx — not retryable, but still a backend failure for fallback.
            raise _SearchBackendError(f"HTTP {exc.code}: {body}")
        except urllib.error.URLError:
            # Network error — retryable.
            raise

    return _do()


# ---------------------------------------------------------------------------
# Tavily
# ---------------------------------------------------------------------------


def _run_tavily(*, query: str, limit: int, api_key: str, advanced: bool) -> tuple[list[dict[str, Any]], str]:
    if not api_key:
        raise _SearchBackendError("missing Tavily API key")
    payload = {
        "query": query,
        "max_results": limit,
        "search_depth": "advanced" if advanced else "basic",
        # ask for raw page content in advanced mode so the extractor can quote
        # without an extra fetch round-trip.
        "include_raw_content": advanced,
    }
    raw = _http_request_json(
        url="https://api.tavily.com/search",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        payload=json.dumps(payload).encode("utf-8"),
        timeout=20.0,
    )
    if not isinstance(raw, dict):
        raise _SearchBackendError("unexpected Tavily response shape")
    answer = str(raw.get("answer") or "")
    results: list[dict[str, Any]] = []
    for item in list(raw.get("results") or [])[:limit]:
        if not isinstance(item, dict):
            continue
        entry: dict[str, Any] = {
            "title": item.get("title"),
            "url": item.get("url"),
            "snippet": item.get("content") or item.get("snippet") or "",
        }
        # Advanced mode carries raw page text — surface it for L0 ladder use.
        if advanced and item.get("raw_content"):
            entry["raw_content"] = str(item.get("raw_content"))[:20000]
        results.append(entry)
    return results, answer


# ---------------------------------------------------------------------------
# Brave Search API
# ---------------------------------------------------------------------------


def _run_brave(*, query: str, limit: int, api_key: str) -> tuple[list[dict[str, Any]], str]:
    if not api_key:
        raise _SearchBackendError("missing Brave API key")
    params = urllib.parse.urlencode({"q": query, "count": str(limit)})
    raw = _http_request_json(
        url=f"https://api.search.brave.com/res/v1/web/search?{params}",
        method="GET",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        },
        timeout=20.0,
    )
    if not isinstance(raw, dict):
        raise _SearchBackendError("unexpected Brave response shape")
    results: list[dict[str, Any]] = []
    for item in list(raw.get("web", {}).get("results") or [])[:limit]:
        if not isinstance(item, dict):
            continue
        results.append({
            "title": item.get("title"),
            "url": item.get("url") or item.get("link"),
            "snippet": item.get("description") or item.get("snippet") or "",
        })
    return results, ""


# ---------------------------------------------------------------------------
# Serper (Google SERP)
# ---------------------------------------------------------------------------


def _run_serper(*, query: str, limit: int, api_key: str) -> tuple[list[dict[str, Any]], str]:
    if not api_key:
        raise _SearchBackendError("missing Serper API key")
    payload = json.dumps({"q": query, "num": limit}).encode("utf-8")
    raw = _http_request_json(
        url="https://google.serper.dev/search",
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-API-KEY": api_key,
        },
        payload=payload,
        timeout=20.0,
    )
    if not isinstance(raw, dict):
        raise _SearchBackendError("unexpected Serper response shape")
    results: list[dict[str, Any]] = []
    # Serper returns organic, news, images, etc. We take organic first, then
    # news, to keep result quality high for research use cases.
    for bucket in ("organic", "news", "knowledgeGraph"):
        for item in list(raw.get(bucket) or []):
            if not isinstance(item, dict):
                continue
            if bucket == "knowledgeGraph":
                # KG is a single object, not a list — skip, it's not a search hit.
                continue
            results.append({
                "title": item.get("title"),
                "url": item.get("link") or item.get("url"),
                "snippet": item.get("snippet") or item.get("description") or "",
            })
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break
    return results, ""


# ---------------------------------------------------------------------------
# DuckDuckGo HTML (keyless fallback)
# ---------------------------------------------------------------------------


_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)


def _run_duckduckgo(*, query: str, limit: int) -> tuple[list[dict[str, Any]], str]:
    """Scrape DuckDuckGo's HTML endpoint. No API key required.

    This is the fallback of last resort — quality is lower than the paid
    backends but it always works without credentials, so the search layer
    degrades gracefully when all keys are missing/expired.
    """
    params = urllib.parse.urlencode({"q": query, "kl": "us-en"})
    request = urllib.request.Request(
        f"https://duckduckgo.com/html/?{params}",
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; InsightSwarmResearch/1.0; +https://github.com/insightswarm)",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
        method="GET",
    )

    @_RETRY_DECORATOR
    def _do() -> bytes:
        try:
            with urllib.request.urlopen(request, timeout=20.0) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code >= 500 or exc.code == 429:
                raise _RetryableHTTPError(exc.code, "")
            raise _SearchBackendError(f"HTTP {exc.code}")

    try:
        html_bytes = _do()
    except RetryError as exc:  # pragma: no cover - tenacity wraps final attempt
        raise _SearchBackendError(f"network error after retries: {exc}")

    html = html_bytes.decode("utf-8", errors="replace")
    results: list[dict[str, Any]] = []
    for match in _DDG_RESULT_RE.finditer(html):
        raw_href, raw_title, raw_snippet = match.group(1), match.group(2), match.group(3)
        # DDG wraps links in a redirect like //duckduckgo.com/l/?uddg=<encoded>
        url = _unwrap_ddg_link(raw_href)
        title = _strip_tags(raw_title)
        snippet = _strip_tags(raw_snippet)
        if not url or not title:
            continue
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results, ""


def _unwrap_ddg_link(href: str) -> str:
    """Extract the real URL from a DuckDuckGo redirect wrapper."""
    if not href:
        return ""
    # DuckDuckGo's HTML results use href="//duckduckgo.com/l/?uddg=<encoded>&..."
    # or sometimes a bare redirect. Parse the uddg= parameter.
    if "uddg=" in href:
        parsed = urllib.parse.urlparse(href)
        params = urllib.parse.parse_qs(parsed.query)
        uddg = params.get("uddg", [""])[0]
        if uddg:
            return urllib.parse.unquote(uddg)
    # Some links are already direct URLs (absolute or protocol-relative).
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return ""


def _strip_tags(value: str) -> str:
    """Strip HTML tags and collapse whitespace; cheap helper for DDG scraping."""
    text = re.sub(r"<[^>]+>", " ", value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text
