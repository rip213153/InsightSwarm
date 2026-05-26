from __future__ import annotations

from typing import Any

from insightswarm.fetching import fetch_source
from insightswarm.tools.core import ToolContext, ToolResult
from insightswarm.tools.safety import validate_public_http_url


class FetchUrlTool:
    name = "fetch.url"
    description = "Fetch public URL content using httpx first and Playwright only as controlled fallback."
    input_schema = {"required": ["url"], "optional": ["timeout"]}
    output_schema = {"required": ["source_url", "fetcher", "status", "text", "html", "metadata"]}
    safety_policy = {"allowed_schemes": ["http", "https"], "blocks_local_network_by_default": True}
    allowed_callers = ["ExtractorAgent"]
    side_effect_level = "network_read"
    network_access = "public_http"
    blocked_inputs = ["file URLs", "localhost/internal URLs in production mode", "non-http schemes"]
    example_failures = [
        {"input": {"url": "file:///C:/secret.txt"}, "output": {"status": "blocked", "error": "URL scheme is not allowed"}}
    ]
    examples = [
        {
            "tool": "fetch.url",
            "input": {"url": "https://example.com/pricing"},
            "output": {"status": "ok", "data": {"fetcher": "httpx", "status": "ok", "text": "ExampleCo pricing..."}},
        },
        {
            "tool": "fetch.url",
            "input": {"url": "file:///C:/secret.txt"},
            "output": {"status": "blocked", "error": "URL scheme is not allowed", "diagnostics": {"allowed_schemes": ["http", "https"]}},
        },
    ]

    def run(self, tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        url = str(tool_input.get("url") or "")
        blocked = validate_public_http_url(url, context)
        if blocked:
            return blocked
        result = fetch_source(url, timeout=float(tool_input.get("timeout") or 20.0))
        return ToolResult(
            "ok" if result.ok else "error",
            data={
                "source_url": result.source_url,
                "fetcher": result.fetcher,
                "status": result.status,
                "text": result.text,
                "html": result.html,
                "screenshot_path": getattr(result, "screenshot_path", None),
                "latency_ms": result.latency_ms,
                "fallback_reason": result.fallback_reason,
                "metadata": result.metadata,
            },
            error=result.error,
            diagnostics=result.metadata,
            provenance={"tool": self.name, "fetcher": result.fetcher},
        )
