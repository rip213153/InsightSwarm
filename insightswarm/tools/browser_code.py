from __future__ import annotations

from typing import Any

from insightswarm.browser_code import run_browser_extract_code
from insightswarm.tools.core import ToolContext, ToolResult


class BrowserExtractCodeTool:
    name = "browser.extract_code"
    description = "Run restricted read-only Python extraction code over browser page state and sanitized text."
    input_schema = {"required": ["code"], "optional": ["session_id", "page_state", "html_text", "artifact_summaries"]}
    output_schema = {"required": ["status", "extracted_items", "candidate_targets", "namespace_state_summary"]}
    safety_policy = {
        "sandbox": "restricted_read_only_python",
        "network_access": False,
        "filesystem_access": False,
        "browser_control": False,
        "cdp_access": False,
        "javascript_eval": False,
        "approval_replay_required_for_actions": True,
    }
    allowed_callers = ["BrowserAgent"]
    side_effect_level = "artifact_only"
    network_access = "none"
    blocked_inputs = [
        "import statements",
        "requests/pandas/numpy/pypdf/subprocess/socket/os/sys",
        "open/eval/exec/compile/__import__",
        "network/file/CDP/browser control",
        "cookie/localStorage/header/password/token access",
    ]
    example_failures = [
        {
            "input": {"code": "import requests\nrequests.get('https://example.com')"},
            "output": {"status": "error", "error": "import statements are not allowed in browser code sandbox"},
        },
        {
            "input": {"code": "open('secret.txt').read()"},
            "output": {"status": "error", "error": "blocked function: open"},
        },
    ]
    examples = [
        {
            "tool": "browser.extract_code",
            "input": {
                "code": "candidate_targets = classify_page_state(page_state)",
                "page_state": {"interactable_elements": [{"stable_node_id": "n1", "role": "link", "text": "联想电脑 商品详情"}]},
            },
            "output": {"status": "ok", "data": {"candidate_targets": [{"semantic_type": "product_detail_link"}]}},
        }
    ]

    def run(self, tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        report = run_browser_extract_code(tool_input, context)
        data = {
            "status": report.status,
            "extracted_items": report.extracted_items,
            "candidate_targets": report.candidate_targets,
            "warnings": report.warnings,
            "namespace_state_summary": report.namespace_state_summary,
            "provenance": report.provenance,
        }
        if report.status == "error":
            return ToolResult("error", data=data, diagnostics={"sandbox": "browser_code.v0"}, error=report.error, provenance=report.provenance)
        return ToolResult("ok", data=data, diagnostics={"sandbox": "browser_code.v0"}, warnings=report.warnings, provenance=report.provenance)
