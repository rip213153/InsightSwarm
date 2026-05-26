from __future__ import annotations

from typing import Any

from insightswarm.browser_handoff import build_candidate_source
from insightswarm.tools.core import ToolContext, ToolResult


class BrowserPromoteSourceTool:
    name = "browser.promote_source"
    description = "Create a candidate source from sanitized BrowserAgent observations without creating formal evidence."
    input_schema = {
        "required": ["source_url"],
        "optional": ["title", "text_preview", "visible_text", "selected_text", "page_state", "source_artifact_id", "source_artifact_type"],
    }
    output_schema = {"required": ["candidate.source_url", "candidate.text_preview", "candidate.extraction_readiness"]}
    safety_policy = {
        "artifact_only": True,
        "creates_formal_evidence": False,
        "requires_raw_document_promotion": True,
        "network_access": False,
        "browser_control": False,
        "sensitive_fields_redacted": True,
    }
    allowed_callers = ["BrowserAgent"]
    side_effect_level = "artifact_only"
    network_access = "none"
    blocked_inputs = ["cookie/localStorage/header/password/token", "full HTML", "screenshots", "localhost/internal/file URL in production"]
    example_failures = [
        {
            "input": {"source_url": "file:///C:/secret.txt", "text_preview": "secret"},
            "output": {"status": "blocked", "error": "URL scheme is not allowed"},
        }
    ]
    examples = [
        {
            "tool": "browser.promote_source",
            "input": {"source_url": "https://example.com/product", "title": "Product page", "text_preview": "Price and specs..."},
            "output": {"status": "ok", "data": {"candidate": {"source_kind": "browser_handoff", "extraction_readiness": "ready"}}},
        }
    ]

    def run(self, tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        return build_candidate_source(tool_input, context)
