from __future__ import annotations

from typing import Any

from insightswarm.search import build_search_client
from insightswarm.tools.core import ToolContext, ToolResult


class SearchTool:
    name = "search.web"
    description = "Search public web sources for candidate evidence URLs."
    input_schema = {"required": ["query"], "optional": ["provider", "limit", "source_urls", "locale", "domain_filters", "recency_days"]}
    output_schema = {"required": ["provider", "status", "results"]}
    safety_policy = {"api_keys": "environment_only", "network": "provider_dependent", "allowed_providers": ["tavily", "static"]}
    allowed_callers = ["SearchAgent"]
    side_effect_level = "read_only"
    network_access = "provider_dependent"
    blocked_inputs = ["unsupported search provider"]
    example_failures = [
        {"input": {"provider": "unknown"}, "output": {"status": "blocked", "error": "Unsupported search provider: unknown"}}
    ]
    examples = [
        {
            "tool": "search.web",
            "input": {"query": "联想拯救者 2026 价格 配置", "provider": "tavily", "limit": 5},
            "output": {"status": "ok", "data": {"results": [{"title": "联想拯救者官方产品页", "url": "https://www.lenovo.com.cn/...", "snippet": "价格、配置、RTX...", "rank": 1}]}},
        },
        {
            "tool": "search.web",
            "input": {"query": "ExampleCo pricing", "provider": "static", "source_urls": ["https://example.com/pricing"]},
            "output": {"status": "ok", "data": {"provider": "static", "results": [{"url": "https://example.com/pricing", "rank": 1}]}},
        },
        {
            "tool": "search.web",
            "input": {"query": "ExampleCo pricing", "provider": "tavily"},
            "output": {"status": "error", "error": "missing Tavily API key; set TAVILY_API_KEY or INSIGHTSWARM_TAVILY_API_KEY"},
        },
    ]

    def run(self, tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        provider = str(tool_input.get("provider") or "tavily")
        if provider not in {"tavily", "static"}:
            return ToolResult(
                "blocked",
                error=f"Unsupported search provider: {provider}",
                diagnostics={"allowed_providers": ["tavily", "static"]},
                provenance={"tool": self.name, "provider": provider},
            )
        source_urls = list(tool_input.get("source_urls") or [])
        client = build_search_client(provider, source_urls)
        batch = client.search(
            str(tool_input.get("query") or ""),
            limit=int(tool_input.get("limit") or 10),
            locale=tool_input.get("locale"),
            domain_filters=tool_input.get("domain_filters"),
            recency_days=tool_input.get("recency_days"),
        )
        status = "ok" if batch.status == "ok" else "error"
        return ToolResult(
            status,
            data=batch.to_dict(),
            error=batch.error,
            diagnostics={"result_count": len(batch.results), "latency_ms": batch.latency_ms},
            provenance={"tool": self.name, "provider": batch.provider},
        )
