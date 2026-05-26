from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from insightswarm.browser_backend import BrowserBackendUnavailable, BrowserSession
from insightswarm.browser_authorization import classify_authorization_need
from insightswarm.tools.core import ToolContext, ToolResult
from insightswarm.tools.safety import validate_public_http_url


BROWSER_TOOL_NAMES = {
    "browser.extract_code",
    "browser.plan_actions",
    "browser.promote_source",
    "browser.select_target",
    "browser.snapshot",
    "browser.page_state",
    "browser.visible_text",
    "browser.screenshot",
    "browser.goto",
    "browser.scroll",
    "browser.click",
    "browser.type",
    "browser.wait",
}
SAFE_AUTO_ACTIONS = {"snapshot", "page_state", "visible_text", "screenshot", "scroll", "wait"}
REVIEW_REQUIRED_ACTIONS = {"goto", "click", "type"}
ALWAYS_BLOCKED_KEYWORDS = {
    "authorization",
        "cookie",
    "download",
    "header",
    "js",
    "localstorage",
        "password",
    "payment",
    "purchase",
    "submit",
    "token",
    "upload",
}


@dataclass(frozen=True)
class BrowserRisk:
    status: str
    reason: str


def classify_browser_action(action: str, tool_input: dict[str, Any], context: ToolContext | None = None) -> BrowserRisk:
    text = " ".join(str(value).lower() for value in tool_input.values() if value is not None)
    for keyword in ALWAYS_BLOCKED_KEYWORDS:
        if keyword in text:
            return BrowserRisk("blocked", f"blocked_keyword:{keyword}")
    if action == "goto":
        blocked = validate_public_http_url(str(tool_input.get("url") or ""), context)
        if blocked:
            return BrowserRisk("blocked", blocked.error or "browser navigation URL blocked")
    authorization_status, authorization_reason = classify_authorization_need(action, tool_input, context)
    if authorization_status:
        return BrowserRisk(authorization_status, authorization_reason or authorization_status)
    if action in SAFE_AUTO_ACTIONS:
        return BrowserRisk("safe_auto", "safe observation or inert browser action")
    if action in REVIEW_REQUIRED_ACTIONS:
        return BrowserRisk("review_required", "browser action requires human approval")
    return BrowserRisk("blocked", "unsupported browser action")


class BrowserTool:
    allowed_callers = ["BrowserAgent"]
    network_access = "browser_session_only"
    side_effect_level = "browser_sandbox"
    safety_policy = {
        "fake_execution": "default",
        "optional_real_backend": "cdp",
        "human_authorization_or_assisted_observation": True,
        "blocked_capabilities": sorted(ALWAYS_BLOCKED_KEYWORDS),
        "real_browser": "read_only_cdp_observation_only",
        "real_interaction": "authorization_gated_replay_only",
        "page_state": "bounded_dom_ax_snapshot_summary",
    }
    blocked_inputs = [
        "login",
        "submit/payment/purchase",
        "upload/download",
        "cookie/localStorage/header/password/token access",
        "arbitrary JS",
        "localhost/internal/file URL in production mode",
    ]
    example_failures = [
        {
            "input": {"target": "Submit payment"},
            "output": {"status": "blocked", "diagnostics": {"risk_status": "blocked"}},
        },
        {
            "input": {"target": "Pricing tab"},
            "output": {"status": "blocked", "diagnostics": {"risk_status": "review_required"}},
        },
    ]

    def __init__(self, name: str, action: str, description: str, input_schema: dict[str, Any]):
        self.name = name
        self.action = action
        self.description = description
        self.input_schema = input_schema
        self.output_schema = {"required": ["risk_status", "observation", "fake_execution"]}
        if action == "page_state":
            self.output_schema = {
                "required": [
                    "risk_status",
                    "observation.url",
                    "observation.title",
                    "observation.interactable_elements",
                    "fake_execution",
                ]
            }
        self.examples = [
            {
                "tool": name,
                "input": self._example_input(),
                "output": {
                    "status": "ok" if action in SAFE_AUTO_ACTIONS else "blocked",
                    "data": {"risk_status": "safe_auto" if action in SAFE_AUTO_ACTIONS else "review_required"},
                },
            }
        ]

    def run(self, tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        risk = classify_browser_action(self.action, tool_input, context)
        provenance = {"tool": self.name, "browser_action": self.action, "fake_execution": True}
        diagnostics = {
            "risk_status": risk.status,
            "risk_reason": risk.reason,
            "browser_session_id": tool_input.get("session_id") or "fake-browser-session",
            "browser_backend": str(tool_input.get("backend") or "fake"),
            "read_only": self.action in SAFE_AUTO_ACTIONS,
        }
        if risk.status == "blocked":
            return ToolResult("blocked", diagnostics=diagnostics, error=risk.reason, provenance=provenance)
        if risk.status in {"review_required", "authorization_required", "assisted_observation_required"}:
            flags = {
                "approval_replay_required": risk.status == "review_required",
                "human_approval_required": risk.status == "review_required",
                "human_authorization_required": risk.status == "authorization_required",
                "human_assisted_observation_required": risk.status == "assisted_observation_required",
            }
            return ToolResult(
                "blocked",
                diagnostics={**diagnostics, **flags},
                error=f"browser action requires {risk.status}",
                provenance=provenance,
            )
        try:
            observation = self._observe(tool_input)
        except BrowserBackendUnavailable as exc:
            return ToolResult(
                "error",
                diagnostics={**diagnostics, **exc.diagnostics, "error_kind": "browser_backend_unavailable"},
                error=str(exc),
                provenance={**provenance, "browser_backend": diagnostics["browser_backend"]},
            )
        data = {
            "risk_status": risk.status,
            "risk_reason": risk.reason,
            "browser_session_id": observation.session_id,
            "browser_backend": observation.backend,
            "fake_execution": bool(observation.diagnostics.get("fake_execution")),
            "read_only": True,
            "observation": observation.observation,
        }
        diagnostics = {**diagnostics, **observation.diagnostics, "browser_session_id": observation.session_id}
        return ToolResult("ok", data=data, diagnostics=diagnostics, provenance=provenance)

    def _example_input(self) -> dict[str, Any]:
        if self.action == "goto":
            return {"url": "https://example.com/pricing"}
        if self.action in {"click", "type"}:
            return {"target": "Pricing tab"}
        if self.action == "page_state":
            return {"session_id": "fake-browser-session", "max_elements": 20}
        return {"session_id": "fake-browser-session"}

    def _observe(self, tool_input: dict[str, Any]):
        backend = str(tool_input.get("backend") or "fake")
        session = BrowserSession(
            backend=backend,
            session_id=tool_input.get("session_id"),
            cdp_url=tool_input.get("cdp_url") or (tool_input.get("browser") or {}).get("cdp_url"),
        )
        try:
            return session.observe(self.action, tool_input)
        finally:
            session.close()


def browser_tools() -> list[BrowserTool]:
    return [
        BrowserTool("browser.snapshot", "snapshot", "Return a fake page state snapshot for BrowserAgent sandbox tests.", {"optional": ["session_id", "url"]}),
        BrowserTool("browser.page_state", "page_state", "Return a bounded DOM/AX-style page state summary for BrowserAgent.", {"optional": ["session_id", "url", "backend", "cdp_url", "max_elements", "max_text_chars", "max_name_chars"]}),
        BrowserTool("browser.visible_text", "visible_text", "Return fake visible page text without controlling a real browser.", {"optional": ["session_id"]}),
        BrowserTool("browser.screenshot", "screenshot", "Return fake screenshot metadata without creating an image.", {"optional": ["session_id"]}),
        BrowserTool("browser.goto", "goto", "Propose browser navigation; non-whitelisted navigation requires authorization in this phase.", {"required": ["url"]}),
        BrowserTool("browser.scroll", "scroll", "Return fake scroll observation.", {"optional": ["session_id", "direction"]}),
        BrowserTool("browser.click", "click", "Propose a click action that may require human authorization.", {"required": ["target"]}),
        BrowserTool("browser.type", "type", "Propose typing into a browser target that may require human authorization or assisted observation.", {"required": ["target", "text"]}),
        BrowserTool("browser.wait", "wait", "Return fake wait observation.", {"optional": ["session_id", "seconds"]}),
    ]
