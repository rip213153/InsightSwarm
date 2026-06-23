from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FailureCategory = Literal["technical", "content", "safety", "budget", "unknown"]


@dataclass(frozen=True)
class AgentFailure:
    category: FailureCategory
    reason: str
    retryable: bool
    should_trigger_critic_review: bool
    should_trigger_research_repair: bool


def normalize_agent_failure(*, status: str | None, reason: str | None) -> AgentFailure:
    normalized_status = _safe_text(status).lower()
    normalized_reason = _safe_text(reason)
    lowered_reason = normalized_reason.lower()

    if normalized_status in {"model_error", "model_rate_limited", "invalid_json", "model_no_tool"}:
        return AgentFailure(
            category="technical",
            reason=normalized_reason or normalized_status,
            retryable=True,
            should_trigger_critic_review=False,
            should_trigger_research_repair=False,
        )
    if "timed out" in lowered_reason or "timeout" in lowered_reason:
        return AgentFailure(
            category="technical",
            reason=normalized_reason,
            retryable=True,
            should_trigger_critic_review=False,
            should_trigger_research_repair=False,
        )
    if normalized_status == "blocked" and "safety cap" in lowered_reason:
        return AgentFailure(
            category="technical",
            reason=normalized_reason,
            retryable=False,
            should_trigger_critic_review=False,
            should_trigger_research_repair=False,
        )
    if normalized_status in {"budget_exhausted", "exhausted"}:
        return AgentFailure(
            category="budget",
            reason=normalized_reason or normalized_status,
            retryable=False,
            should_trigger_critic_review=False,
            should_trigger_research_repair=False,
        )
    if normalized_status in {"blocked", "needs_repair"}:
        return AgentFailure(
            category="content",
            reason=normalized_reason or normalized_status,
            retryable=False,
            should_trigger_critic_review=True,
            should_trigger_research_repair=True,
        )
    return AgentFailure(
        category="unknown",
        reason=normalized_reason or normalized_status or "unknown failure",
        retryable=False,
        should_trigger_critic_review=False,
        should_trigger_research_repair=False,
    )


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()
