from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from insightswarm.tools.core import ToolContext, ToolResult
from insightswarm.tools.safety import validate_public_http_url


class FirecrawlScrapeTool:
    name = "firecrawl.scrape"

    def run(self, tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        url = str(tool_input.get("url") or "")
        blocked = validate_public_http_url(url, context)
        if blocked:
            return blocked

        api_key = os.getenv("FIRECRAWL_API_KEY") or os.getenv("INSIGHTSWARM_FIRECRAWL_API_KEY")
        if not api_key:
            return ToolResult(
                "error",
                error="missing Firecrawl API key; set FIRECRAWL_API_KEY or INSIGHTSWARM_FIRECRAWL_API_KEY",
                provenance={"tool": self.name, "provider": "firecrawl"},
            )

        payload = {
            "url": url,
            "formats": ["markdown", "html"],
            "onlyMainContent": True,
            "timeout": int(float(tool_input.get("timeout") or 30000)),
        }
        started = time.perf_counter()
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
            return ToolResult(
                "error",
                error=f"Firecrawl HTTP {exc.code}",
                diagnostics={"http_status": exc.code, "body_preview": body},
                provenance={"tool": self.name, "provider": "firecrawl"},
            )
        except Exception as exc:
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


def _html_to_text(value: str) -> str:
    import html
    import re

    body = re.sub(r"<(script|style|nav|header|footer|svg)\b[^>]*>.*?</\1>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"<[^>]+>", " ", body)
    return re.sub(r"\s+", " ", html.unescape(body)).strip()
