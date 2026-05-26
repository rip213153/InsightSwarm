from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from insightswarm.db.store import Store
from insightswarm.util import loads

if TYPE_CHECKING:
    from insightswarm.tools.core import ToolContext, ToolResult


AUTHORIZATION_REQUIRED = "authorization_required"
ASSISTED_OBSERVATION_REQUIRED = "assisted_observation_required"

ASSISTED_KEYWORDS = {"captcha", "verification", "verify", "2fa", "mfa", "otp", "code", "验证码", "校验码"}
AUTHORIZATION_KEYWORDS = {"login", "signin", "sign in", "qq邮箱", "email", "mail", "邮箱"}
BLOCKED_KEYWORDS = {
    "payment",
    "purchase",
    "checkout",
    "submit payment",
    "upload",
    "download",
    "cookie",
    "cookies",
    "localstorage",
    "authorization",
    "header",
    "headers",
    "token",
    "password",
    "secret",
    "arbitrary js",
    "cdp command",
}
LOCALHOSTS = {"localhost", "127.0.0.1", "::1"}


def browser_policy_metadata(context: "ToolContext | None" = None) -> dict[str, Any]:
    metadata = (context.metadata if context else {}) or {}
    return {
        "browser_allowed_domains": _string_list(metadata.get("browser_allowed_domains")),
        "browser_authorized_domains": _string_list(metadata.get("browser_authorized_domains")),
        "browser_assisted_observation_allowed": bool(metadata.get("browser_assisted_observation_allowed", True)),
        "browser_max_authorization_requests": int(metadata.get("browser_max_authorization_requests") or 5),
    }


def classify_authorization_need(action: str, tool_input: dict[str, Any], context: "ToolContext | None" = None) -> tuple[str | None, str | None]:
    text = " ".join(str(value).lower() for value in tool_input.values() if value is not None)
    for keyword in BLOCKED_KEYWORDS:
        if keyword in text:
            return "blocked", f"blocked_keyword:{keyword}"
    policy = browser_policy_metadata(context)
    if action == "type" and any(keyword in text for keyword in ASSISTED_KEYWORDS):
        if policy["browser_assisted_observation_allowed"]:
            return ASSISTED_OBSERVATION_REQUIRED, "human_assisted_observation_required"
        return "blocked", "assisted_observation_not_allowed"
    url = str(tool_input.get("url") or "")
    domain = _domain(url)
    allowed = set(policy["browser_allowed_domains"])
    authorized = set(policy["browser_authorized_domains"])
    if action == "goto" and domain:
        if domain in LOCALHOSTS:
            return "review_required", "test-mode local navigation requires compatibility approval"
        if _domain_matches(domain, allowed) or _domain_matches(domain, authorized):
            return "safe_auto", "browser domain is allowed or pre-authorized"
        return AUTHORIZATION_REQUIRED, f"domain_not_authorized:{domain}"
    if any(keyword in text for keyword in AUTHORIZATION_KEYWORDS):
        return AUTHORIZATION_REQUIRED, "authorization_required_for_login_or_mail_context"
    if action in {"click", "type"}:
        return AUTHORIZATION_REQUIRED, "browser action requires operator authorization"
    return None, None


def write_browser_authorization_request(
    store: Store,
    run_id: str,
    task_id: str | None,
    tool_name: str,
    tool_input: dict[str, Any],
    result: "ToolResult",
    tool_call_id: str,
    context: "ToolContext",
) -> str:
    action = tool_name.split(".", 1)[1]
    payload = {
        "schema": "browser_authorization_request.v1",
        "status": "pending",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "action": action,
        "risk_status": result.diagnostics.get("risk_status"),
        "risk_reason": result.diagnostics.get("risk_reason"),
        "target_summary": _safe_text(tool_input.get("target") or tool_input.get("url") or tool_input.get("intent")),
        "input_summary": _request_input_summary(tool_input),
        "authorization_policy": browser_policy_metadata(context),
        "allowed_choices": ["approve", "reject"],
    }
    artifact_id = store.write_artifact(
        run_id,
        task_id,
        "browser_authorization_request",
        "application/json",
        json.dumps(payload, ensure_ascii=True, indent=2),
        source_url=tool_input.get("url"),
        metadata={
            "schema": "browser_authorization_request.v1",
            "status": "pending",
            "tool": tool_name,
            "tool_call_id": tool_call_id,
            "action": action,
            "risk_status": result.diagnostics.get("risk_status"),
            "risk_reason": result.diagnostics.get("risk_reason"),
        },
        suffix=".json",
    )
    store.emit_event(
        run_id,
        task_id,
        "BrowserAgent",
        "browser_authorization_required",
        "Browser action requires human authorization before the agent continues.",
        {
            "request_artifact_id": artifact_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "action": action,
            "target_summary": payload["target_summary"],
            "risk_reason": payload["risk_reason"],
            "allowed_choices": payload["allowed_choices"],
        },
    )
    return artifact_id


def write_assisted_observation_request(
    store: Store,
    run_id: str,
    task_id: str | None,
    tool_name: str,
    tool_input: dict[str, Any],
    result: "ToolResult",
    tool_call_id: str,
) -> str:
    payload = {
        "schema": "browser_assisted_observation_request.v1",
        "status": "pending",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "action": tool_name.split(".", 1)[1],
        "risk_status": result.diagnostics.get("risk_status"),
        "risk_reason": result.diagnostics.get("risk_reason"),
        "prompt": _safe_text(tool_input.get("prompt") or tool_input.get("target") or tool_input.get("url") or "Provide the requested observation."),
        "input_summary": _request_input_summary(tool_input),
    }
    artifact_id = store.write_artifact(
        run_id,
        task_id,
        "browser_assisted_observation_request",
        "application/json",
        json.dumps(payload, ensure_ascii=True, indent=2),
        source_url=tool_input.get("url"),
        metadata={
            "schema": "browser_assisted_observation_request.v1",
            "status": "pending",
            "tool": tool_name,
            "tool_call_id": tool_call_id,
            "action": payload["action"],
            "risk_status": result.diagnostics.get("risk_status"),
            "risk_reason": result.diagnostics.get("risk_reason"),
        },
        suffix=".json",
    )
    store.emit_event(
        run_id,
        task_id,
        "BrowserAgent",
        "browser_assisted_observation_required",
        "BrowserAgent needs human-assisted observation before it can continue.",
        {
            "request_artifact_id": artifact_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "action": payload["action"],
            "target_summary": payload["prompt"],
        },
    )
    return artifact_id


def list_browser_authorizations(store: Store, run_id: str) -> dict[str, Any]:
    auth_requests = [_artifact_summary(row) for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_authorization_request"]
    auth_decisions = [_artifact_summary(row) for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_authorization_decision"]
    obs_requests = [_artifact_summary(row) for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_assisted_observation_request"]
    obs_responses = [_artifact_summary(row) for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_assisted_observation_response"]
    decided = {item.get("request_id") for item in auth_decisions}
    observed = {item.get("request_id") for item in obs_responses}
    return {
        "pending_authorizations": [item for item in auth_requests if item["artifact_id"] not in decided],
        "authorization_decisions": auth_decisions,
        "authorization_requests": auth_requests,
        "pending_observations": [item for item in obs_requests if item["artifact_id"] not in observed],
        "observation_responses": obs_responses,
        "observation_requests": obs_requests,
    }


def decide_browser_authorization(store: Store, run_id: str, request_id: str, decision: str) -> dict[str, Any]:
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    request = _require_artifact(store, run_id, request_id, "browser_authorization_request")
    payload = {"schema": "browser_authorization_decision.v1", "request_id": request_id, "decision": decision}
    artifact_id = store.write_artifact(
        run_id,
        request["task_id"],
        "browser_authorization_decision",
        "application/json",
        json.dumps(payload, ensure_ascii=True, indent=2),
        source_url=request["source_url"],
        metadata={"schema": "browser_authorization_decision.v1", "request_id": request_id, "decision": decision},
        suffix=".json",
    )
    store.emit_event(
        run_id,
        request["task_id"],
        "BrowserAgent",
        "browser_authorization_decision",
        f"Browser authorization {decision}.",
        {"decision_artifact_id": artifact_id, "request_id": request_id, "decision": decision},
    )
    return {"status": decision, "decision_id": artifact_id, "request_id": request_id}


def respond_assisted_observation(store: Store, run_id: str, request_id: str, value: str) -> dict[str, Any]:
    request = _require_artifact(store, run_id, request_id, "browser_assisted_observation_request")
    payload = {
        "schema": "browser_assisted_observation_response.v1",
        "request_id": request_id,
        "status": "provided",
        "value": _safe_text(value, 500),
        "value_chars": len(value or ""),
    }
    artifact_id = store.write_artifact(
        run_id,
        request["task_id"],
        "browser_assisted_observation_response",
        "application/json",
        json.dumps(payload, ensure_ascii=True, indent=2),
        source_url=request["source_url"],
        metadata={"schema": "browser_assisted_observation_response.v1", "request_id": request_id, "status": "provided", "value_chars": len(value or "")},
        suffix=".json",
    )
    store.emit_event(
        run_id,
        request["task_id"],
        "BrowserAgent",
        "browser_assisted_observation_response",
        "Human-assisted observation response was provided.",
        {"response_artifact_id": artifact_id, "request_id": request_id, "value_chars": len(value or "")},
    )
    return {"status": "provided", "response_id": artifact_id, "request_id": request_id}


def authorization_is_approved(store: Store, run_id: str, request_id: str | None) -> bool:
    return _has_decision(store, run_id, "browser_authorization_decision", request_id, "approve")


def observation_is_provided(store: Store, run_id: str, request_id: str | None) -> bool:
    if not request_id:
        return False
    return any(
        row["artifact_type"] == "browser_assisted_observation_response"
        and loads(row["metadata_json"], {}).get("request_id") == request_id
        for row in store.list_artifacts(run_id)
    )


def _has_decision(store: Store, run_id: str, artifact_type: str, request_id: str | None, decision: str) -> bool:
    if not request_id:
        return False
    return any(
        row["artifact_type"] == artifact_type
        and loads(row["metadata_json"], {}).get("request_id") == request_id
        and loads(row["metadata_json"], {}).get("decision") == decision
        for row in store.list_artifacts(run_id)
    )


def _require_artifact(store: Store, run_id: str, artifact_id: str, artifact_type: str) -> Any:
    try:
        row = store.get_artifact(artifact_id)
    except KeyError as exc:
        raise ValueError(f"{artifact_type} not found for run") from exc
    if row["run_id"] != run_id or row["artifact_type"] != artifact_type:
        raise ValueError(f"{artifact_type} not found for run")
    return row


def _artifact_summary(row: Any) -> dict[str, Any]:
    metadata = loads(row["metadata_json"], {})
    payload = _read_json(row)
    return {
        "artifact_id": row["artifact_id"],
        "request_id": metadata.get("request_id") or payload.get("request_id"),
        "artifact_type": row["artifact_type"],
        "status": metadata.get("status") or payload.get("status") or metadata.get("decision"),
        "decision": metadata.get("decision") or payload.get("decision"),
        "action": metadata.get("action") or payload.get("action"),
        "risk_status": metadata.get("risk_status") or payload.get("risk_status"),
        "risk_reason": metadata.get("risk_reason") or payload.get("risk_reason"),
        "created_at": row["created_at"],
    }


def _request_input_summary(tool_input: dict[str, Any]) -> dict[str, Any]:
    blocked = {"password", "token", "cookie", "cookies", "authorization", "header", "headers", "localstorage", "secret"}
    summary = {}
    for key, value in tool_input.items():
        if key.lower() in blocked:
            continue
        summary[key] = _safe_text(value) if isinstance(value, str) else value
    return summary


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip().lower() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return []


def _domain(url: str) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    return parsed.hostname.lower() if parsed.hostname else None


def _domain_matches(domain: str, patterns: set[str]) -> bool:
    for pattern in patterns:
        if pattern.startswith("*.") and (domain == pattern[2:] or domain.endswith(pattern[1:])):
            return True
        if domain == pattern or domain.endswith("." + pattern):
            return True
    return False


def _safe_text(value: Any, limit: int = 160) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _read_json(row: Any) -> dict[str, Any]:
    path = Path(row["path"])
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
