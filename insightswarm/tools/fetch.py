from __future__ import annotations

import html as html_lib
import re
import time
import urllib.request
from typing import Any

from insightswarm.tools.core import ToolContext, ToolResult
from insightswarm.tools.safety import validate_public_http_url


class FetchUrlTool:
    name = "fetch.url"

    def run(self, tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        url = str(tool_input.get("url") or "")
        blocked = validate_public_http_url(url, context)
        if blocked:
            return blocked

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=float(tool_input.get("timeout") or 20.0)) as response:
                raw = response.read()
                html = raw.decode("utf-8", errors="replace")
                cleaned = _clean_html(html)
        except Exception as exc:
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
                "status": "ok",
                "text": cleaned["text"],
                "html": html,
                "title": cleaned["title"],
                "metadata": {"latency_ms": int((time.perf_counter() - started) * 1000)},
            },
            provenance={"tool": self.name, "fetcher": "urllib"},
        )

def _clean_html(raw_html: str) -> dict[str, str]:
    html = raw_html or ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = _collapse_whitespace(_strip_tags(title_match.group(1))) if title_match else ""
    body = re.sub(
        r"<(script|style|nav|header|footer|svg)\b[^>]*>.*?</\1>",
        " ",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.DOTALL)
    text = _collapse_whitespace(_strip_tags(body))
    return {"title": title, "text": text}


def _strip_tags(value: str) -> str:
    return html_lib.unescape(re.sub(r"<[^>]+>", " ", value))


def _collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
