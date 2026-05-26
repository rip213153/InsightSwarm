from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from insightswarm.browser_backend import BrowserBackendUnavailable, BrowserSession
from insightswarm.db.store import Store
from insightswarm.tools.core import ToolContext, ToolResult
from insightswarm.tools.safety import validate_public_http_url
from insightswarm.util import loads


INTERACTION_ACTIONS = {"goto", "click", "type", "scroll", "wait"}
APPROVAL_ACTIONS = {"goto", "click", "type"}
TYPE_ALLOWED_ROLES = {"input", "textbox", "searchbox", "combobox"}
TYPE_ALLOWED_TAGS = {"input", "textarea"}


def write_browser_action_request(
    store: Store,
    run_id: str,
    task_id: str | None,
    tool_name: str,
    tool_input: dict[str, Any],
    result: ToolResult,
    tool_call_id: str,
    context: ToolContext,
) -> str:
    action = tool_name.split(".", 1)[1]
    page_state_artifact_id = tool_input.get("page_state_artifact_id") or tool_input.get("snapshot_artifact_id")
    target_id = tool_input.get("target_id") or tool_input.get("stable_node_id")
    target_selection_artifact_id = tool_input.get("target_selection_artifact_id")
    target_selection = _read_target_selection(store, target_selection_artifact_id)
    selected_target = target_selection.get("selected_target") or {}
    if not target_id and selected_target.get("stable_node_id"):
        target_id = selected_target.get("stable_node_id")
    if not page_state_artifact_id:
        page_state_artifact_id = target_selection.get("page_state_artifact_id")
    target = _resolve_page_state_target(store, page_state_artifact_id, target_id)
    target_unresolved = bool(target_id and page_state_artifact_id and target is None)
    semantic_type = selected_target.get("semantic_type") or (target or {}).get("semantic_type")
    reject_reasons = target_selection.get("reject_reasons") or []
    risk_hint = target_selection.get("risk_hint") or (selected_target or {}).get("risk_hint")
    payload = {
        "schema": "browser_action_request.v1",
        "status": "pending",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "action": action,
        "risk_status": result.diagnostics.get("risk_status"),
        "risk_reason": result.diagnostics.get("risk_reason"),
        "target_summary": _safe_text(tool_input.get("target") or tool_input.get("url") or target_id),
        "input_summary": _request_input_summary(tool_input),
        "page_state_artifact_id": page_state_artifact_id,
        "target_id": target_id,
        "target_unresolved": target_unresolved,
        "target_selection_artifact_id": target_selection_artifact_id,
        "selected_semantic_type": semantic_type,
        "selected_target_summary": {
            "stable_node_id": selected_target.get("stable_node_id") or target_id,
            "semantic_type": semantic_type,
            "text": _safe_text(selected_target.get("text") or (target or {}).get("text") or (target or {}).get("name")),
            "href": _safe_text(selected_target.get("href") or (target or {}).get("href")),
            "container_context": _safe_text(selected_target.get("container_context") or (target or {}).get("container_context")),
            "nearby_text": _safe_text(selected_target.get("nearby_text") or (target or {}).get("nearby_text")),
        },
        "target_reject_reasons": reject_reasons,
        "target_risk_hint": risk_hint,
        "needs_human_disambiguation": bool(target_selection.get("needs_human_disambiguation")),
        "allowed_choices": ["approve_execute", "reject", "manual_capture_instead"],
    }
    artifact_id = store.write_artifact(
        run_id,
        task_id,
        "browser_action_request",
        "application/json",
        json.dumps(payload, ensure_ascii=True, indent=2),
        source_url=tool_input.get("url") or (target or {}).get("href"),
        metadata={
            "schema": "browser_action_request.v1",
            "status": "pending",
            "action": action,
            "tool": tool_name,
            "tool_call_id": tool_call_id,
            "risk_status": result.diagnostics.get("risk_status"),
            "risk_reason": result.diagnostics.get("risk_reason"),
            "page_state_artifact_id": page_state_artifact_id,
            "target_id": target_id,
            "target_unresolved": target_unresolved,
            "target_selection_artifact_id": target_selection_artifact_id,
            "selected_semantic_type": semantic_type,
            "target_reject_reasons": reject_reasons,
            "target_risk_hint": risk_hint,
            "needs_human_disambiguation": bool(target_selection.get("needs_human_disambiguation")),
            "quality_mode": context.quality_mode,
        },
        suffix=".json",
    )
    store.emit_event(
        run_id,
        task_id,
        "BrowserAgent",
        "browser_action_request_created",
        "Browser action request created for human approval.",
        {
            "request_artifact_id": artifact_id,
            "tool_call_id": tool_call_id,
            "action": action,
            "target_summary": payload["target_summary"],
            "risk_reason": payload["risk_reason"],
            "page_state_artifact_id": page_state_artifact_id,
            "target_id": target_id,
            "target_unresolved": target_unresolved,
            "target_selection_artifact_id": target_selection_artifact_id,
            "selected_semantic_type": semantic_type,
            "target_reject_reasons": reject_reasons,
            "target_risk_hint": risk_hint,
            "allowed_choices": payload["allowed_choices"],
        },
    )
    return artifact_id


def list_browser_approvals(store: Store, run_id: str) -> dict[str, Any]:
    requests = [_artifact_summary(row) for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_action_request"]
    decisions = [_artifact_summary(row) for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_approval_decision"]
    decided_ids = {item["request_id"] for item in decisions if item.get("request_id")}
    pending = [item for item in requests if item["artifact_id"] not in decided_ids]
    return {"pending": pending, "decisions": decisions, "requests": requests}


def approve_browser_action(
    store: Store,
    run_id: str,
    request_id: str,
    *,
    execute: bool = False,
    backend: str = "fake",
    cdp_url: str | None = None,
    quality_mode: str = "production",
) -> dict[str, Any]:
    try:
        request_artifact = store.get_artifact(request_id)
    except KeyError as exc:
        raise ValueError("browser action request not found for run") from exc
    if request_artifact["run_id"] != run_id or request_artifact["artifact_type"] != "browser_action_request":
        raise ValueError("browser action request not found for run")
    request = _read_json(request_artifact)
    blocked = _approval_block_reason(store, request_artifact, request, quality_mode)
    if blocked:
        raise ValueError(blocked)
    approval_id = _write_decision(store, run_id, request_artifact["task_id"], request_id, "approved", None)
    result = {"status": "approved", "approval_id": approval_id, "request_id": request_id}
    if execute:
        result["execution"] = execute_browser_action(
            store,
            run_id,
            request_id,
            approval_id,
            backend=backend,
            cdp_url=cdp_url,
            quality_mode=quality_mode,
        )
    return result


def reject_browser_action(store: Store, run_id: str, request_id: str, reason: str | None = None) -> dict[str, Any]:
    try:
        request_artifact = store.get_artifact(request_id)
    except KeyError as exc:
        raise ValueError("browser action request not found for run") from exc
    if request_artifact["run_id"] != run_id or request_artifact["artifact_type"] != "browser_action_request":
        raise ValueError("browser action request not found for run")
    decision_id = _write_decision(store, run_id, request_artifact["task_id"], request_id, "rejected", reason)
    return {"status": "rejected", "decision_id": decision_id, "request_id": request_id}


def execute_browser_action(
    store: Store,
    run_id: str,
    request_id: str,
    approval_id: str,
    *,
    backend: str = "fake",
    cdp_url: str | None = None,
    quality_mode: str = "production",
) -> dict[str, Any]:
    request_artifact = store.get_artifact(request_id)
    request = _read_json(request_artifact)
    action = request.get("action")
    tool_input = _execution_input(store, request_artifact, request, quality_mode)
    status = "ok"
    error = None
    observation: dict[str, Any] = {}
    after_page_state_artifact_id = None
    try:
        session = BrowserSession(backend=backend, cdp_url=cdp_url)
        try:
            executed = session.execute(action, tool_input)
            observation = executed.observation
            if backend == "cdp":
                after_state = session.observe("page_state", {"backend": backend, "cdp_url": cdp_url})
                after_result = ToolResult(
                    "ok",
                    data={
                        "risk_status": "safe_auto",
                        "risk_reason": "post_action_page_state",
                        "browser_session_id": after_state.session_id,
                        "browser_backend": after_state.backend,
                        "fake_execution": False,
                        "read_only": True,
                        "observation": after_state.observation,
                    },
                    diagnostics={**after_state.diagnostics, "risk_status": "safe_auto", "browser_session_id": after_state.session_id},
                    provenance={"tool": "browser.page_state", "browser_action": "page_state", "post_action": True},
                )
                from insightswarm.browser_sandbox import write_browser_observation

                after_page_state_artifact_id = write_browser_observation(
                    store,
                    run_id,
                    request_artifact["task_id"],
                    "browser.page_state",
                    after_result,
                )
        finally:
            session.close()
    except BrowserBackendUnavailable as exc:
        status = "error"
        error = str(exc)
        observation = exc.diagnostics
    payload = {
        "schema": "browser_action_execution.v1",
        "status": status,
        "action": action,
        "backend": backend,
        "request_id": request_id,
        "approval_id": approval_id,
        "target_id": request.get("target_id"),
        "observation": observation,
        "error": error,
        "after_page_state_artifact_id": after_page_state_artifact_id,
    }
    execution_id = store.write_artifact(
        run_id,
        request_artifact["task_id"],
        "browser_action_execution",
        "application/json",
        json.dumps(payload, ensure_ascii=True, indent=2),
        source_url=request_artifact["source_url"],
        metadata={
            "schema": "browser_action_execution.v1",
            "status": status,
            "action": action,
            "backend": backend,
            "request_id": request_id,
            "approval_id": approval_id,
            "target_id": request.get("target_id"),
            "after_page_state_artifact_id": after_page_state_artifact_id,
            "error": error,
        },
        suffix=".json",
    )
    store.emit_event(
        run_id,
        request_artifact["task_id"],
        "BrowserAgent",
        "browser_action_executed" if status == "ok" else "browser_action_execution_failed",
        f"Browser action {action} execution {status}.",
        {"execution_artifact_id": execution_id, "request_id": request_id, "approval_id": approval_id, "action": action, "backend": backend, "error": error, "after_page_state_artifact_id": after_page_state_artifact_id},
    )
    return {"status": status, "execution_id": execution_id, "error": error, "after_page_state_artifact_id": after_page_state_artifact_id}


def _write_decision(store: Store, run_id: str, task_id: str | None, request_id: str, decision: str, reason: str | None) -> str:
    payload = {"schema": "browser_approval_decision.v1", "request_id": request_id, "decision": decision, "reason": _safe_text(reason)}
    artifact_id = store.write_artifact(
        run_id,
        task_id,
        "browser_approval_decision",
        "application/json",
        json.dumps(payload, ensure_ascii=True, indent=2),
        metadata={"schema": "browser_approval_decision.v1", "request_id": request_id, "decision": decision},
        suffix=".json",
    )
    store.emit_event(
        run_id,
        task_id,
        "BrowserAgent",
        "browser_approval_decision",
        f"Browser action request {decision}.",
        {"decision_artifact_id": artifact_id, "request_id": request_id, "decision": decision, "reason": _safe_text(reason)},
    )
    return artifact_id


def _approval_block_reason(store: Store, request_artifact: Any, request: dict[str, Any], quality_mode: str) -> str | None:
    action = request.get("action")
    if action not in APPROVAL_ACTIONS:
        return "only review-required browser actions can be approved"
    if request.get("risk_status") == "blocked":
        return "blocked browser actions cannot be approved"
    target_semantic_type = str(request.get("selected_semantic_type") or "").lower()
    if request.get("needs_human_disambiguation") and target_semantic_type in {"customer_service", "ad", "unknown"}:
        return f"browser action target requires human disambiguation: {target_semantic_type or 'unknown'}"
    if request.get("target_unresolved"):
        return "browser action target is unresolved; capture page state again"
    try:
        _execution_input(store, request_artifact, request, quality_mode)
    except ValueError as exc:
        return str(exc)
    return None


def _execution_input(store: Store, request_artifact: Any, request: dict[str, Any], quality_mode: str) -> dict[str, Any]:
    action = request.get("action")
    summary = request.get("input_summary") or {}
    if action == "goto":
        url = summary.get("url")
        blocked = validate_public_http_url(str(url or ""), ToolContext(quality_mode=quality_mode))
        if blocked:
            raise ValueError(blocked.error or "browser navigation URL blocked")
        return {"url": url}
    if action in {"click", "type"}:
        target_id = request.get("target_id")
        target = _resolve_page_state_target(store, request.get("page_state_artifact_id"), target_id)
        if not target:
            raise ValueError("browser action target is unresolved; capture page state again")
        bbox = target.get("bbox") or {}
        if not bbox.get("width") or not bbox.get("height"):
            raise ValueError("browser action target bbox is missing or not visible")
        if action == "type":
            role = str(target.get("role") or "").lower()
            tag = str(target.get("tag") or "").lower()
            if role not in TYPE_ALLOWED_ROLES and tag not in TYPE_ALLOWED_TAGS:
                raise ValueError("browser type target must be an input or textbox-like element")
            return {"target_id": target_id, "bbox": bbox, "text": str(summary.get("text_preview") or "")[:500]}
        return {"target_id": target_id, "bbox": bbox}
    if action == "scroll":
        return {"direction": summary.get("direction") or "down", "delta": min(int(summary.get("delta") or 600), 2000)}
    if action == "wait":
        return {"seconds": min(float(summary.get("seconds") or 1), 5)}
    raise ValueError("unsupported browser action")


def _resolve_page_state_target(store: Store, page_state_artifact_id: str | None, target_id: str | None) -> dict[str, Any] | None:
    if not page_state_artifact_id or not target_id:
        return None
    try:
        artifact = store.get_artifact(page_state_artifact_id)
    except KeyError:
        return None
    if artifact["artifact_type"] != "browser_page_state":
        return None
    payload = _read_json(artifact)
    observation = ((payload.get("data") or {}).get("observation") or {}) if payload else {}
    for item in observation.get("interactable_elements") or []:
        if item.get("stable_node_id") == target_id:
            return item
    return None


def _read_target_selection(store: Store, artifact_id: str | None) -> dict[str, Any]:
    if not artifact_id:
        return {}
    try:
        artifact = store.get_artifact(artifact_id)
    except KeyError:
        return {}
    if artifact["artifact_type"] != "browser_target_selection":
        return {}
    payload = _read_json(artifact)
    data = payload.get("data") or {}
    provenance = data.get("provenance") or payload.get("provenance") or {}
    return {
        "selected_target": data.get("selected_target") or {},
        "reject_reasons": data.get("reject_reasons") or [],
        "risk_hint": data.get("risk_hint"),
        "needs_human_disambiguation": data.get("needs_human_disambiguation"),
        "page_state_artifact_id": provenance.get("page_state_artifact_id"),
    }


def _request_input_summary(tool_input: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for key, value in tool_input.items():
        lower = key.lower()
        if lower in {"password", "token", "cookie", "authorization", "header", "headers", "localstorage"}:
            continue
        if key == "text":
            text = str(value or "")
            summary["text_chars"] = len(text)
            summary["text_preview"] = _safe_text(text)
        else:
            summary[key] = _safe_text(value) if isinstance(value, str) else value
    return summary


def _artifact_summary(row: Any) -> dict[str, Any]:
    metadata = loads(row["metadata_json"], {})
    payload = _read_json(row)
    return {
        "artifact_id": row["artifact_id"],
        "request_id": metadata.get("request_id") or payload.get("request_id"),
        "artifact_type": row["artifact_type"],
        "status": metadata.get("status") or payload.get("status") or metadata.get("decision"),
        "action": metadata.get("action") or payload.get("action"),
        "decision": metadata.get("decision") or payload.get("decision"),
        "target_id": metadata.get("target_id") or payload.get("target_id"),
        "target_unresolved": metadata.get("target_unresolved"),
        "created_at": row["created_at"],
    }


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
