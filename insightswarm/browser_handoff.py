from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from insightswarm.db.store import Store
from insightswarm.tools.core import ToolContext, ToolResult
from insightswarm.tools.safety import validate_public_http_url
from insightswarm.util import loads


SENSITIVE_KEYS = {"cookie", "cookies", "authorization", "token", "headers", "header", "localstorage", "password", "secret"}
MIN_PROMOTION_TEXT_CHARS = 20


def build_candidate_source(tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
    source = _source_payload(tool_input)
    warnings = _warnings_for_source(source, context)
    if not source.get("source_url"):
        return ToolResult("blocked", data={"candidate": source}, diagnostics={"reason": "missing_url"}, error="candidate source requires a URL")
    blocked = validate_public_http_url(source["source_url"], context)
    if blocked:
        return ToolResult("blocked", data={"candidate": source}, diagnostics={"reason": blocked.error}, error=blocked.error)
    if len(source.get("text") or "") < MIN_PROMOTION_TEXT_CHARS:
        warnings.append("short_text_preview")
    candidate = {
        "schema": "candidate_source.v1",
        "status": "candidate",
        "source_url": source["source_url"],
        "title": source.get("title"),
        "text_preview": _safe_text(source.get("text"), 1200),
        "source_kind": "browser_handoff",
        "confidence": _confidence(source, warnings),
        "extraction_readiness": "ready" if len(source.get("text") or "") >= MIN_PROMOTION_TEXT_CHARS else "needs_more_text",
        "risk_warnings": warnings,
        "provenance": {
            "collector": "BrowserAgent",
            "source_artifact_id": tool_input.get("source_artifact_id"),
            "source_artifact_type": tool_input.get("source_artifact_type"),
            "browser_mode": (context.metadata or {}).get("browser_mode") if context else tool_input.get("browser_mode"),
            "tool": "browser.promote_source",
        },
    }
    return ToolResult(
        "ok",
        data={"candidate": candidate},
        diagnostics={"extraction_readiness": candidate["extraction_readiness"], "risk_warning_count": len(warnings)},
        warnings=warnings,
        provenance=candidate["provenance"],
    )


def write_candidate_source(store: Store, run_id: str, task_id: str | None, tool_call_id: str, result: ToolResult) -> str:
    candidate = (result.data or {}).get("candidate") or {}
    artifact_id = store.write_artifact(
        run_id,
        task_id,
        "candidate_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_source.v1",
                "tool_call_id": tool_call_id,
                "status": result.status,
                "candidate": candidate,
                "diagnostics": result.diagnostics,
                "warnings": result.warnings,
                "error": result.error,
                "provenance": result.provenance,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url=candidate.get("source_url"),
        metadata={
            "schema": "candidate_source.v1",
            "tool": "browser.promote_source",
            "tool_call_id": tool_call_id,
            "tool_status": result.status,
            "source_kind": "browser_handoff",
            "source_url": candidate.get("source_url"),
            "title": candidate.get("title"),
            "confidence": candidate.get("confidence", 0),
            "extraction_readiness": candidate.get("extraction_readiness"),
            "risk_warning_count": len(candidate.get("risk_warnings") or result.warnings or []),
            "citation_ready": candidate.get("extraction_readiness") == "ready" and result.status == "ok",
            "promoted": False,
            "error": result.error,
        },
        suffix=".json",
    )
    store.emit_event(
        run_id,
        task_id,
        "BrowserAgent",
        "browser_candidate_source_created" if result.status == "ok" else "browser_candidate_source_blocked",
        f"Browser candidate source {result.status}.",
        {
            "artifact_id": artifact_id,
            "tool_call_id": tool_call_id,
            "source_url": candidate.get("source_url"),
            "extraction_readiness": candidate.get("extraction_readiness"),
            "risk_warning_count": len(candidate.get("risk_warnings") or result.warnings or []),
            "error": result.error,
        },
    )
    return artifact_id


def promote_candidate_to_raw_document(store: Store, run_id: str, candidate_id: str, *, quality_mode: str = "production") -> dict[str, Any]:
    try:
        row = store.get_artifact(candidate_id)
    except KeyError as exc:
        raise ValueError("candidate source not found for run") from exc
    if row["run_id"] != run_id or row["artifact_type"] != "candidate_source":
        raise ValueError("candidate source not found for run")
    payload = _read_json(row)
    candidate = payload.get("candidate") or {}
    blocked = _promotion_block_reason(candidate, quality_mode)
    if blocked:
        artifact_id = store.write_artifact(
            run_id,
            row["task_id"],
            "browser_handoff_promotion_blocked",
            "application/json",
            json.dumps({"schema": "browser_handoff_promotion_blocked.v1", "candidate_id": candidate_id, "reason": blocked}, ensure_ascii=True, indent=2),
            source_url=candidate.get("source_url"),
            metadata={"candidate_id": candidate_id, "status": "blocked", "reason": blocked},
            suffix=".json",
        )
        store.emit_event(run_id, row["task_id"], "BrowserAgent", "browser_handoff_promotion_blocked", "Browser candidate source promotion blocked.", {"artifact_id": artifact_id, "candidate_id": candidate_id, "reason": blocked})
        return {"status": "blocked", "candidate_id": candidate_id, "blocked_artifact_id": artifact_id, "reason": blocked}
    text = _safe_text(candidate.get("text_preview"), 6000)
    raw_id = store.write_artifact(
        run_id,
        row["task_id"],
        "raw_document",
        "text/plain",
        text,
        source_url=candidate.get("source_url"),
        metadata={
            "fetcher": "browser_agent_handoff",
            "source_kind": "browser_handoff",
            "browser_candidate_source_artifact_id": candidate_id,
            "manual_or_agent_browser_capture": True,
            "requires_extractor": True,
            "title": candidate.get("title"),
            "confidence": candidate.get("confidence"),
            "risk_warnings": candidate.get("risk_warnings") or [],
        },
    )
    store.emit_event(
        run_id,
        row["task_id"],
        "BrowserAgent",
        "browser_handoff_promoted",
        "Browser candidate source promoted to raw document.",
        {"candidate_source_artifact_id": candidate_id, "raw_document_artifact_id": raw_id, "source_url": candidate.get("source_url")},
    )
    return {"status": "promoted", "candidate_id": candidate_id, "raw_document_artifact_id": raw_id}


def candidate_from_artifact(store: Store, artifact_id: str) -> dict[str, Any]:
    row = store.get_artifact(artifact_id)
    payload = _read_json(row)
    if row["artifact_type"] == "browser_page_state":
        observation = ((payload.get("data") or {}).get("observation") or {}) if payload else {}
        return {
            "source_artifact_id": artifact_id,
            "source_artifact_type": row["artifact_type"],
            "source_url": row["source_url"] or observation.get("url"),
            "title": observation.get("title"),
            "text": observation.get("text_preview"),
        }
    if row["artifact_type"] == "browser_observation":
        observation = ((payload.get("data") or {}).get("observation") or {}) if payload else {}
        return {
            "source_artifact_id": artifact_id,
            "source_artifact_type": row["artifact_type"],
            "source_url": row["source_url"] or observation.get("url"),
            "title": observation.get("title"),
            "text": observation.get("text") or observation.get("text_preview") or observation.get("status"),
        }
    if row["artifact_type"] == "browser_code_result":
        data = payload.get("data") or {}
        items = data.get("extracted_items") or []
        return {
            "source_artifact_id": artifact_id,
            "source_artifact_type": row["artifact_type"],
            "source_url": row["source_url"] or _first_value(items, "source_url") or _first_value(items, "href"),
            "title": _first_value(items, "title"),
            "text": json.dumps(items[:8], ensure_ascii=True),
        }
    return {"source_artifact_id": artifact_id, "source_artifact_type": row["artifact_type"]}


def _source_payload(tool_input: dict[str, Any]) -> dict[str, Any]:
    source = {
        "source_url": tool_input.get("source_url") or tool_input.get("url"),
        "title": tool_input.get("title"),
        "text": tool_input.get("selected_text") or tool_input.get("visible_text") or tool_input.get("text_preview") or tool_input.get("text"),
    }
    if isinstance(tool_input.get("page_state"), dict):
        page_state = tool_input["page_state"]
        source["source_url"] = source["source_url"] or page_state.get("url")
        source["title"] = source["title"] or page_state.get("title")
        source["text"] = source["text"] or page_state.get("text_preview")
    return {key: _safe_text(value, 6000 if key == "text" else 400) for key, value in source.items()}


def _warnings_for_source(source: dict[str, Any], context: ToolContext | None) -> list[str]:
    warnings = []
    text = json.dumps(source, ensure_ascii=True).lower()
    for key in SENSITIVE_KEYS:
        if key in text:
            warnings.append(f"sensitive_field_removed:{key}")
    if context and context.quality_mode != "test" and source.get("source_url"):
        blocked = validate_public_http_url(source["source_url"], context)
        if blocked:
            warnings.append(blocked.error or "url_blocked")
    return _unique(warnings)


def _promotion_block_reason(candidate: dict[str, Any], quality_mode: str) -> str | None:
    if not candidate.get("source_url"):
        return "candidate source has no URL"
    blocked = validate_public_http_url(candidate["source_url"], ToolContext(quality_mode=quality_mode))
    if blocked:
        return blocked.error or "candidate URL blocked"
    if len(candidate.get("text_preview") or "") < MIN_PROMOTION_TEXT_CHARS:
        return "candidate source has insufficient text"
    if candidate.get("extraction_readiness") != "ready":
        return "candidate source is not extraction-ready"
    return None


def _confidence(source: dict[str, Any], warnings: list[str]) -> float:
    score = 0.72
    if len(source.get("text") or "") >= 120:
        score += 0.12
    if warnings:
        score -= 0.15
    return round(max(0.1, min(score, 0.95)), 2)


def _first_value(items: list[dict[str, Any]], key: str) -> Any:
    for item in items:
        if isinstance(item, dict) and item.get(key):
            return item.get(key)
    return None


def _read_json(row: Any) -> dict[str, Any]:
    path = Path(row["path"])
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}


def _safe_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    lowered = text.lower()
    for key in SENSITIVE_KEYS:
        if key in lowered:
            text = text.replace(key, "[redacted]")
            text = text.replace(key.upper(), "[redacted]")
    text = text.replace("\x00", "")
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _unique(values: list[str]) -> list[str]:
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output
