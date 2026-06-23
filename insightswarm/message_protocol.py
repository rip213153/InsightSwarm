from __future__ import annotations

from typing import Any


MESSAGE_TYPES = {
    "request",
    "response",
    "observation",
    "suggestion",
    "hypothesis",
}

PAYLOAD_KINDS: dict[str, set[str]] = {
    "request": {
        "research_subquestion",
        "research_repair",
        "extract_evidence",
        "review_evidence",
        "review_extraction_failure",
        "delivery_request",
        "hard_acquisition",
        "source_replacement_request",
        "repair_request",
    },
    "response": {
        "completed",
        "blocked",
        "needs_repair",
        "no_action",
        "partial",
        "pass",
    },
    "observation": {
        "progress_update",
        "blocked_page",
        "extraction_failure",
        "evidence_gap",
        "conflict",
        "budget_status",
        "delivery_gap",
        "authorization_request",
        "repair_exhausted",
        "source_quality",
        "source_fetch_error",
        "browser_trace",
        "technical_failure",
        "authorization_decision",
        "direction_rejected",
    },
    "suggestion": {
        "try_browser",
        "research_more",
        "repair_query",
        "deliver_partial",
        "ask_human",
        "split_subtask",
    },
    "hypothesis": {
        "candidate_claim",
        "likely_answer",
        "conflict_interpretation",
        "timeline_guess",
        "run_direction",
    },
}


def validate_message(
    *,
    message_type: str,
    payload: dict[str, Any] | None,
    related_task_id: str | None,
) -> None:
    if message_type not in MESSAGE_TYPES:
        raise ValueError(f"unsupported message type: {message_type}")
    payload = dict(payload or {})
    payload_kind = str(payload.get("kind") or "")
    if payload_kind not in PAYLOAD_KINDS[message_type]:
        raise ValueError(f"unsupported payload.kind '{payload_kind}' for message type '{message_type}'")
    if not related_task_id and not str(payload.get("related_artifact_id") or "").strip() and not str(payload.get("issue_key") or "").strip():
        raise ValueError("message requires related_task_id, payload.related_artifact_id, or payload.issue_key")
