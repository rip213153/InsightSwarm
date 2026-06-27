from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from insightswarm.tools.core import ToolContext, ToolResult
from insightswarm.tools.fetch import LADDER_L1_QUICK_READ, LADDER_L2_FETCH, _try_jina_reader
from insightswarm.tools.safety import validate_public_http_url


class FirecrawlScrapeTool:
    name = "firecrawl.scrape"

    def run(self, tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        url = str(tool_input.get("url") or "")
        blocked = validate_public_http_url(url, context)
        if blocked:
            return blocked

        api_key = os.getenv("FIRECRAWL_API_KEY") or os.getenv("INSIGHTSWARM_FIRECRAWL_API_KEY")
        started = time.perf_counter()
        if not api_key:
            # No Firecrawl key — degrade directly to L1 (Jina Reader). This is
            # the "auto from L1" fallback: L2 is unavailable, so we serve an
            # L1 result rather than failing. Caller can inspect `fetcher` /
            # `ladder` to know which path produced the text.
            return _jina_fallback_result(url, started, reason="missing Firecrawl API key")

        payload = {
            "url": url,
            "formats": ["markdown", "html"],
            "onlyMainContent": True,
            "timeout": int(float(tool_input.get("timeout") or 30000)),
        }
        request = urllib.request.Request(
            "https://api.firecrawl.dev/v1/scrape",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(tool_input.get("request_timeout") or 45.0)) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            # L2 failed — degrade to L1 (Jina Reader) before returning an error.
            jina = _try_jina_reader(url)
            if jina is not None:
                return _build_jina_fallback_result(
                    url, jina, started, ladder=LADDER_L2_FETCH,
                    reason=f"Firecrawl HTTP {exc.code}: {body}",
                )
            return ToolResult(
                "error",
                error=f"Firecrawl HTTP {exc.code}",
                diagnostics={"http_status": exc.code, "body_preview": body},
                provenance={"tool": self.name, "provider": "firecrawl"},
            )
        except Exception as exc:
            # Network error / timeout — degrade to L1 (Jina Reader).
            jina = _try_jina_reader(url)
            if jina is not None:
                return _build_jina_fallback_result(
                    url, jina, started, ladder=LADDER_L2_FETCH,
                    reason=f"Firecrawl request failed: {exc}",
                )
            return ToolResult(
                "error",
                error=f"Firecrawl request failed: {exc}",
                provenance={"tool": self.name, "provider": "firecrawl"},
            )

        data = raw.get("data") if isinstance(raw, dict) else {}
        if not isinstance(data, dict):
            data = {}
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        markdown = str(data.get("markdown") or "")
        html = str(data.get("html") or "")
        title = str(metadata.get("title") or data.get("title") or "")
        text = markdown.strip() or _html_to_text(html)
        if len(text) < 100:
            # Firecrawl returned too little text — degrade to L1 (Jina Reader).
            # This catches the "Firecrawl succeeded but page is JS-gated / empty"
            # case where a Jina Reader pass may still extract usable markdown.
            jina = _try_jina_reader(url)
            if jina is not None and len(jina[0]) > len(text):
                return _build_jina_fallback_result(
                    url, jina, started, ladder=LADDER_L2_FETCH,
                    reason="Firecrawl returned too little text",
                )
            return ToolResult(
                "error",
                error="Firecrawl returned too little text",
                diagnostics={"latency_ms": int((time.perf_counter() - started) * 1000), "text_length": len(text)},
                provenance={"tool": self.name, "provider": "firecrawl"},
            )

        return ToolResult(
            "ok",
            data={
                "source_url": str(metadata.get("sourceURL") or metadata.get("url") or url),
                "fetcher": "firecrawl",
                "ladder": LADDER_L2_FETCH,
                "status": "ok",
                "text": text,
                "html": html,
                "title": title,
                "metadata": {
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "firecrawl_metadata": metadata,
                },
            },
            diagnostics={"latency_ms": int((time.perf_counter() - started) * 1000), "text_length": len(text)},
            provenance={"tool": self.name, "provider": "firecrawl"},
        )


def _jina_fallback_result(url: str, started: float, *, reason: str) -> ToolResult:
    """Run Jina Reader and wrap the result as an L1 fallback for Firecrawl."""
    jina = _try_jina_reader(url)
    if jina is not None:
        return _build_jina_fallback_result(url, jina, started, ladder=LADDER_L2_FETCH, reason=reason)
    return ToolResult(
        "error",
        error=f"Firecrawl unavailable and Jina fallback failed: {reason}",
        diagnostics={"reason": reason},
        provenance={"tool": "firecrawl.scrape", "provider": "firecrawl"},
    )


def _build_jina_fallback_result(
    url: str,
    jina: tuple[str, str],
    started: float,
    *,
    ladder: str,
    reason: str,
) -> ToolResult:
    markdown, title = jina
    return ToolResult(
        "ok",
        data={
            "source_url": url,
            "fetcher": "jina_fallback",
            "ladder": LADDER_L1_QUICK_READ,
            "escalated_from": ladder,
            "status": "ok",
            "text": markdown,
            "html": "",
            "title": title,
            "metadata": {
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "fallback_reason": reason,
            },
        },
        diagnostics={
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "text_length": len(markdown),
            "fallback_reason": reason,
        },
        provenance={"tool": "firecrawl.scrape", "provider": "jina_fallback"},
    )


def _html_to_text(value: str) -> str:
    import html
    import re

    body = re.sub(r"<(script|style|nav|header|footer|svg)\b[^>]*>.*?</\1>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"<[^>]+>", " ", body)
    return re.sub(r"\s+", " ", html.unescape(body)).strip()
