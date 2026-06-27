from __future__ import annotations

import html as html_lib
import os
import re
import time
import urllib.request
from typing import Any

from insightswarm.tools.core import ToolContext, ToolResult
from insightswarm.tools.safety import validate_public_http_url


# Three-tier fetch ladder. Each tier trades cost for fidelity:
#   L0 snippet       — search-result snippet (no fetch; provided by SearchTool)
#   L1 quick_read    — Jina Reader markdown (cheap, clean, ~5s)
#   L2 fetch         — full HTML scrape (urllib) or Firecrawl deep extract
# FetchUrlTool implements L1 (preferred) and L2 (fallback). FirecrawlScrapeTool
# implements L2 with automatic L1 fallback when Firecrawl is unavailable.
LADDER_L0_SNIPPET = "L0"
LADDER_L1_QUICK_READ = "L1"
LADDER_L2_FETCH = "L2"

JINA_READER_BASE = "https://r.jina.ai/"
JINA_READER_TIMEOUT = 20.0


class FetchUrlTool:
    name = "fetch.url"

    def run(self, tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        url = str(tool_input.get("url") or "")
        blocked = validate_public_http_url(url, context)
        if blocked:
            return blocked

        # L1 quick_read mode: prefer Jina Reader markdown (cleaner, faster)
        # over raw HTML scraping. Falls back to urllib if Jina is unavailable.
        # Default (no flag) preserves the legacy behavior: urllib first, Jina
        # fallback. This keeps existing call sites unchanged while letting the
        # quick_read tool opt into the L1 path.
        prefer_jina = bool(tool_input.get("prefer_jina"))
        timeout = float(tool_input.get("timeout") or 20.0)
        started = time.perf_counter()

        if prefer_jina:
            jina_result = _try_jina_reader(url, timeout=timeout)
            if jina_result is not None:
                markdown, title = jina_result
                return ToolResult(
                    "ok",
                    data={
                        "source_url": url,
                        "fetcher": "jina",
                        "ladder": LADDER_L1_QUICK_READ,
                        "status": "ok",
                        "text": markdown,
                        "html": "",
                        "title": title,
                        "metadata": {"latency_ms": int((time.perf_counter() - started) * 1000)},
                    },
                    provenance={"tool": self.name, "fetcher": "jina"},
                )
            # Jina failed — fall through to urllib L2 fetch.

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; InsightSwarmResearch/1.0; +https://github.com/insightswarm)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                html = raw.decode("utf-8", errors="replace")
                cleaned = _clean_html(html)
        except Exception as exc:
            # L2 urllib failed — try Jina Reader as L1 fallback before giving up.
            jina_result = _try_jina_reader(url, timeout=timeout)
            if jina_result is not None:
                markdown, title = jina_result
                return ToolResult(
                    "ok",
                    data={
                        "source_url": url,
                        "fetcher": "jina",
                        "ladder": LADDER_L1_QUICK_READ,
                        "fallback_from": "urllib",
                        "status": "ok",
                        "text": markdown,
                        "html": "",
                        "title": title,
                        "metadata": {
                            "latency_ms": int((time.perf_counter() - started) * 1000),
                            "urllib_error": str(exc),
                        },
                    },
                    provenance={"tool": self.name, "fetcher": "jina"},
                )
            return ToolResult(
                "error",
                error=f"Fetch failed: {exc}",
                provenance={"tool": self.name, "fetcher": "urllib"},
            )

        return ToolResult(
            "ok",
            data={
                "source_url": url,
                "fetcher": "urllib",
                "ladder": LADDER_L2_FETCH,
                "status": "ok",
                "text": cleaned["text"],
                "html": html,
                "title": cleaned["title"],
                "metadata": {"latency_ms": int((time.perf_counter() - started) * 1000)},
            },
            provenance={"tool": self.name, "fetcher": "urllib"},
        )


def _try_jina_reader(url: str, *, timeout: float = JINA_READER_TIMEOUT) -> tuple[str, str] | None:
    """Fetch `url` via Jina Reader (`https://r.jina.ai/<url>`).

    Returns (markdown, title) on success, or None if the request fails or
    returns too little text. Jina Reader returns the page as clean markdown
    prefixed with a `Title: ...` line — we extract that for the title field.
    """
    jina_url = JINA_READER_BASE + url
    request = urllib.request.Request(
        jina_url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; InsightSwarmResearch/1.0; +https://github.com/insightswarm)",
            "Accept": "text/plain, text/markdown",
            # Jina Reader supports an auth token header for higher rate limits,
            # but works without one. We send the env-provided token if present.
            **({"X-Api-Key": token} if (token := _jina_token()) else {}),
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    if not raw or len(raw.strip()) < 50:
        return None
    title = _extract_jina_title(raw)
    # Strip the Jina metadata header (Title:/URL Source:/Markdown Content:) so
    # the caller gets pure body text. If extraction fails, return the whole
    # response — better to have the raw markdown than nothing.
    body = _strip_jina_header(raw)
    if not body.strip():
        body = raw
    return body, title


def _jina_token() -> str | None:
    """Jina Reader API token (optional — raises rate limits when set)."""
    return os.getenv("JINA_API_KEY") or os.getenv("INSIGHTSWARM_JINA_API_KEY")


def _extract_jina_title(raw: str) -> str:
    """Jina Reader prefixes output with `Title: <title>` on its own line."""
    match = re.search(r"^Title:\s*(.+?)\s*$", raw, flags=re.MULTILINE)
    return match.group(1) if match else ""


def _strip_jina_header(raw: str) -> str:
    """Drop the Jina Reader metadata block (Title:/URL Source:/Markdown Content:)."""
    # Jina's output format is:
    #   Title: ...
    #   URL Source: ...
    #   Markdown Content:
    #   <body>
    marker = "Markdown Content:"
    idx = raw.find(marker)
    if idx >= 0:
        return raw[idx + len(marker):].lstrip("\n").strip()
    return raw.strip()


def _clean_html(raw_html: str) -> dict[str, str]:
    html = raw_html or ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = _collapse_whitespace(_strip_tags(title_match.group(1))) if title_match else ""
    body = re.sub(
        r"<(script|style|nav|header|footer|svg|form|aside|noscript)\b[^>]*>.*?</\1>",
        " ",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    body = re.sub(
        r"<\w+\b[^>]*\b(class|id|aria-label)\s*=\s*[\"'][^\"']*(?:cookie|consent|newsletter|gdpr|modal|overlay|popup|subscribe)[^\"']*[\"'][^>]*>.*?</\w+>",
        " ",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.DOTALL)
    text = _collapse_whitespace(_strip_tags(body))
    return {"title": title, "text": text}


def _strip_tags(value: str) -> str:
    return html_lib.unescape(re.sub(r"<[^>]+>", " ", value))


def _collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
