from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from insightswarm.tools.core import ToolContext, ToolResult


class SearchTool:
    name = "search.web"

    def run(self, tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        del context
        provider = str(tool_input.get("provider") or "tavily")
        query = str(tool_input.get("query") or "")
        limit = int(tool_input.get("limit") or 5)

        if provider != "tavily":
            return ToolResult(
                "blocked",
                error=f"Unsupported search provider: {provider}",
                diagnostics={"allowed_providers": ["tavily"]},
                provenance={"tool": self.name, "provider": provider},
            )

        api_key = os.getenv("TAVILY_API_KEY") or os.getenv("INSIGHTSWARM_TAVILY_API_KEY")
        if not api_key:
            return ToolResult(
                "error",
                error="missing Tavily API key; set TAVILY_API_KEY or INSIGHTSWARM_TAVILY_API_KEY",
                provenance={"tool": self.name, "provider": provider},
            )

        payload = {
            "query": query,
            "max_results": limit,
            "search_depth": "basic",
        }
        started = time.perf_counter()
        request = urllib.request.Request(
            "https://api.tavily.com/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return ToolResult(
                "error",
                error=f"Tavily HTTP {exc.code}",
                diagnostics={"http_status": exc.code},
                provenance={"tool": self.name, "provider": provider},
            )
        except Exception as exc:
            return ToolResult(
                "error",
                error=f"Tavily request failed: {exc}",
                provenance={"tool": self.name, "provider": provider},
            )

        results = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content") or item.get("snippet") or "",
            }
            for item in list(raw.get("results") or [])[:limit]
        ]
        return ToolResult(
            "ok",
            data={"provider": provider, "status": "ok", "results": results},
            diagnostics={"latency_ms": int((time.perf_counter() - started) * 1000), "result_count": len(results)},
            provenance={"tool": self.name, "provider": provider},
        )

