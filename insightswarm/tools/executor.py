from __future__ import annotations

from typing import Any

from insightswarm.db.store import Store
from insightswarm.browser_authorization import write_assisted_observation_request, write_browser_authorization_request
from insightswarm.browser_handoff import write_candidate_source
from insightswarm.browser_interaction import write_browser_action_request
from insightswarm.browser_code import write_browser_code_result
from insightswarm.browser_planning import write_browser_action_plan
from insightswarm.browser_targeting import write_browser_target_selection
from insightswarm.tools.browser import BROWSER_TOOL_NAMES
from insightswarm.tools.core import ToolContext, ToolResult
from insightswarm.tools.registry import get_tool
from insightswarm.tool_contracts import tool_recovery_hint
from insightswarm.util import new_id


SENSITIVE_KEYS = {"api_key", "authorization", "token", "secret", "html", "text", "raw", "content"}
MAX_STRING_PREVIEW = 160


class ToolExecutor:
    def __init__(self, store: Store):
        self.store = store

    def run(self, tool_name: str, tool_input: dict[str, Any], context: ToolContext) -> tuple[ToolResult, str]:
        tool = get_tool(tool_name)
        tool_call_id = new_id("tool")
        caller = context.metadata.get("agent_name", "ToolExecutor")
        started_metadata = self._metadata(tool, tool_call_id, "started", tool_input)
        self.store.emit_event(
            context.run_id or "",
            context.task_id,
            caller,
            "tool_call_started",
            f"Tool {tool_name} started.",
            started_metadata,
        )
        if caller not in getattr(tool, "allowed_callers", []):
            error = (
                "browser tools are restricted to BrowserAgent"
                if tool_name in BROWSER_TOOL_NAMES
                else f"{tool_name} is restricted to: {', '.join(getattr(tool, 'allowed_callers', []))}"
            )
            result = ToolResult(
                "blocked",
                error=error,
                diagnostics={
                    "risk_status": "blocked",
                    "risk_reason": "caller_not_allowed",
                    "caller": caller,
                    "allowed_callers": getattr(tool, "allowed_callers", []),
                    "recovery": tool_recovery_hint(tool_name, "blocked", "caller_not_allowed"),
                },
                provenance={"tool": tool_name, "policy": "allowed_callers"},
            )
        else:
            try:
                tool_context = context
                if tool_name == "research.spawn_subagent":
                    tool_context = ToolContext(
                        context.run_id,
                        context.task_id,
                        context.quality_mode,
                        {**context.metadata, "store": self.store},
                    )
                result = tool.run(tool_input, tool_context)
            except Exception as exc:
                result = ToolResult("error", error=str(exc), provenance={"tool": tool_name})
        if tool_name in BROWSER_TOOL_NAMES and result.diagnostics.get("human_authorization_required"):
            request_artifact_id = write_browser_authorization_request(
                self.store,
                context.run_id or "",
                context.task_id,
                tool_name,
                tool_input,
                result,
                tool_call_id,
                context,
            )
            legacy_request_artifact_id = write_browser_action_request(
                self.store,
                context.run_id or "",
                context.task_id,
                tool_name,
                tool_input,
                result,
                tool_call_id,
                context,
            )
            self.store.emit_event(
                context.run_id or "",
                context.task_id,
                caller,
                "browser_authorization_required",
                "Browser action requires human authorization before the agent continues.",
                {
                    "tool_call_id": tool_call_id,
                    "request_artifact_id": request_artifact_id,
                    "tool_name": tool_name,
                    "action": tool_name.split(".", 1)[1],
                    "target_summary": self._safe_value(tool_input.get("target") or tool_input.get("url")),
                    "risk_reason": result.diagnostics.get("risk_reason"),
                    "proposed_input_summary": self._input_summary(tool_input),
                    "allowed_choices": ["approve", "reject"],
                },
            )
            self.store.emit_event(
                context.run_id or "",
                context.task_id,
                caller,
                "browser_human_approval_required",
                "Browser action compatibility approval request created; prefer browser authorization commands.",
                {
                    "tool_call_id": tool_call_id,
                    "request_artifact_id": legacy_request_artifact_id,
                    "authorization_request_artifact_id": request_artifact_id,
                    "tool_name": tool_name,
                    "action": tool_name.split(".", 1)[1],
                    "target_summary": self._safe_value(tool_input.get("target") or tool_input.get("url")),
                    "risk_reason": result.diagnostics.get("risk_reason"),
                    "proposed_input_summary": self._input_summary(tool_input),
                    "allowed_choices": ["approve_once", "reject", "manual_capture_instead"],
                },
            )
        elif tool_name in BROWSER_TOOL_NAMES and result.diagnostics.get("human_assisted_observation_required"):
            request_artifact_id = write_assisted_observation_request(
                self.store,
                context.run_id or "",
                context.task_id,
                tool_name,
                tool_input,
                result,
                tool_call_id,
            )
            self.store.emit_event(
                context.run_id or "",
                context.task_id,
                caller,
                "browser_assisted_observation_required",
                "BrowserAgent needs human-assisted observation before it can continue.",
                {
                    "tool_call_id": tool_call_id,
                    "request_artifact_id": request_artifact_id,
                    "tool_name": tool_name,
                    "action": tool_name.split(".", 1)[1],
                    "target_summary": self._safe_value(tool_input.get("target") or tool_input.get("url")),
                    "risk_reason": result.diagnostics.get("risk_reason"),
                    "proposed_input_summary": self._input_summary(tool_input),
                    "allowed_choices": ["provide_observation"],
                },
            )
        elif tool_name in BROWSER_TOOL_NAMES and result.diagnostics.get("human_approval_required"):
            request_artifact_id = write_browser_action_request(
                self.store,
                context.run_id or "",
                context.task_id,
                tool_name,
                tool_input,
                result,
                tool_call_id,
                context,
            )
            self.store.emit_event(
                context.run_id or "",
                context.task_id,
                caller,
                "browser_human_approval_required",
                "Browser action requires human approval before execution.",
                {
                    "tool_call_id": tool_call_id,
                    "request_artifact_id": request_artifact_id,
                    "tool_name": tool_name,
                    "action": tool_name.split(".", 1)[1],
                    "target_summary": self._safe_value(tool_input.get("target") or tool_input.get("url")),
                    "risk_reason": result.diagnostics.get("risk_reason"),
                    "proposed_input_summary": self._input_summary(tool_input),
                    "snapshot_artifact_id": tool_input.get("snapshot_artifact_id"),
                    "allowed_choices": ["approve_once", "reject", "manual_capture_instead"],
                },
            )
        event_type = {
            "ok": "tool_call_completed",
            "blocked": "tool_call_blocked",
            "error": "tool_call_failed",
        }.get(result.status, "tool_call_failed")
        self.store.emit_event(
            context.run_id or "",
            context.task_id,
            caller,
            event_type,
            f"Tool {tool_name} {result.status}.",
            {
                **self._metadata(tool, tool_call_id, result.status, tool_input),
                "tool_status": result.status,
                "diagnostics": self._safe_value(result.diagnostics),
                "recovery": self._safe_value(tool_recovery_hint(tool_name, result.status, result.error)),
                "warnings": self._safe_value(result.warnings),
                "error": self._safe_value(result.error),
                "provenance": self._safe_value(result.provenance),
            },
        )
        if tool_name == "browser.extract_code" and context.run_id:
            write_browser_code_result(self.store, context.run_id, context.task_id, tool_call_id, result)
        if tool_name == "browser.plan_actions" and context.run_id:
            write_browser_action_plan(self.store, context.run_id, context.task_id, tool_call_id, result)
        if tool_name == "browser.promote_source" and context.run_id:
            write_candidate_source(self.store, context.run_id, context.task_id, tool_call_id, result)
        if tool_name == "browser.select_target" and context.run_id:
            write_browser_target_selection(self.store, context.run_id, context.task_id, tool_call_id, result)
        return result, tool_call_id

    def _metadata(self, tool, tool_call_id: str, status: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        return {
            "tool_call_id": tool_call_id,
            "tool_name": tool.name,
            "tool_status": status,
            "input_summary": self._input_summary(tool_input),
            "safety_policy": self._safe_value(tool.safety_policy),
        }

    def _input_summary(self, value: dict[str, Any]) -> dict[str, Any]:
        return {key: self._safe_value(item) for key, item in value.items() if key.lower() not in SENSITIVE_KEYS}

    def _safe_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            return value if len(value) <= MAX_STRING_PREVIEW else value[:MAX_STRING_PREVIEW] + "...[truncated]"
        if isinstance(value, list):
            return [self._safe_value(item) for item in value[:12]]
        if isinstance(value, dict):
            return {
                str(key): self._safe_value(item)
                for key, item in value.items()
                if str(key).lower() not in SENSITIVE_KEYS
            }
        return str(value)
