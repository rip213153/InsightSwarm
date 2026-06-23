from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any
from urllib.parse import urlparse

from insightswarm.schemas.swarm import Evidence, Task
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore


CRITIC_ROLE = "critic"
RESEARCHER_ROLE = "researcher"
DEFAULT_MAX_REPAIR_ATTEMPTS = 2
DEFAULT_MAX_BLOCKING_REPAIRS_PER_RUN = 2
DEFAULT_MAX_BLOCKING_REPAIRS_PER_REVIEW = 1

REVIEW_BASIS_SCHEMA = {
    "type": "object",
    "description": "The structured review conclusion formed in Critic private OODA state before taking the terminal action.",
}


CRITIC_TOOLS = [
    {
        "name": "read_review_task",
        "description": "Read the assigned evidence review task and its scoped review question.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {"type": "object", "properties": {"evidence_ids": {"type": "array"}, "question": {"type": "string"}}},
        "side_effects": "none",
    },
    {
        "name": "read_evidence_bundle",
        "description": "Read only the evidence bundle referenced by this review task, including citation payload summaries.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {"type": "object", "properties": {"evidence": {"type": "array"}}},
        "side_effects": "none",
    },
    {
        "name": "read_evidence_map",
        "description": "Read a compact source/claim map for large evidence bundles without loading every quote in full.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {
            "type": "object",
            "properties": {
                "evidence_count": {"type": "integer"},
                "source_count": {"type": "integer"},
                "sources": {"type": "array"},
                "coverage": {"type": "object"},
                "recommended_detail_reads": {"type": "array"},
            },
        },
        "side_effects": "none",
    },
    {
        "name": "validate_evidence_bundle",
        "description": "Run deterministic checks before model criticism: evidence exists, source URL exists, quotes exist, citation payloads are readable, and quote fields match.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {"type": "object", "properties": {"passed": {"type": "boolean"}, "must_fix": {"type": "array"}}},
        "side_effects": "stores validation result in private tool state",
    },
    {
        "name": "write_challenge",
        "description": "Write a concrete challenge about weak, missing, stale, or overclaimed evidence.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}, "must_fix": {"type": "array", "items": {"type": "string"}}, "confidence": {"type": "number"}},
            "required": ["summary", "must_fix"],
        },
        "output_schema": {"type": "object", "properties": {"message_id": {"type": "string"}}},
        "side_effects": "writes challenge observation",
    },
    {
        "name": "request_repair",
        "description": "Create a targeted researcher repair task when the evidence bundle cannot safely pass.",
        "input_schema": {
            "type": "object",
            "properties": {
                "targeted_query": {"type": "string"},
                "must_fix": {"type": "array", "items": {"type": "string"}},
                "preferred_source_type": {"type": "string"},
                "why_current_evidence_failed": {"type": "string"},
                "review_basis": REVIEW_BASIS_SCHEMA,
            },
            "required": ["targeted_query", "must_fix", "why_current_evidence_failed"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "repair_created": {"type": "boolean"},
                "deduped": {"type": "boolean"},
                "repair_task_id": {"type": "string"},
                "message_id": {"type": "string"},
                "issue_key": {"type": "string"},
            },
        },
        "side_effects": "writes repair request task/message unless repair budget is exhausted",
    },
    {
        "name": "record_conflict",
        "description": "Record an unresolved evidence conflict that should be surfaced rather than silently merged.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}, "conflicting_evidence_ids": {"type": "array", "items": {"type": "string"}}, "reason": {"type": "string"}, "review_basis": REVIEW_BASIS_SCHEMA},
            "required": ["summary", "reason"],
        },
        "output_schema": {"type": "object", "properties": {"conflict_id": {"type": "string"}, "message_id": {"type": "string"}}},
        "side_effects": "writes conflict board item and observation",
    },
    {
        "name": "mark_review_passed",
        "description": "Mark this evidence bundle as passing Critic review. Use caveats for non-blocking weaknesses that should be disclosed in delivery rather than repaired.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "confidence": {"type": "number"},
                "verdict": {"type": "string", "enum": ["pass", "pass_with_caveats"]},
                "caveats": {"type": "array", "items": {"type": "string"}},
                "review_basis": REVIEW_BASIS_SCHEMA,
            },
            "required": ["reason"],
        },
        "output_schema": {"type": "object", "properties": {"message_id": {"type": "string"}}},
        "side_effects": "writes critic pass response",
    },
    {
        "name": "reject_direction",
        "description": "Reject a weak research/source direction without creating a blocking repair. Bind the rejection to a direction, source path, or URL and explain why Researcher should avoid it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "direction": {"type": "string"},
                "source_url": {"type": "string"},
                "reason": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
                "review_basis": REVIEW_BASIS_SCHEMA,
            },
            "required": ["direction", "reason"],
        },
        "output_schema": {"type": "object", "properties": {"message_id": {"type": "string"}}},
        "side_effects": "writes non-blocking direction rejection observation",
    },
    {
        "name": "finish_review",
        "description": "Stop this Critic review after pass, repair, conflict, or block has been written.",
        "input_schema": {"type": "object", "properties": {"status": {"type": "string", "enum": ["complete", "blocked"]}, "reason": {"type": "string"}}, "required": ["status", "reason"]},
        "output_schema": {"type": "object", "properties": {"terminal": {"type": "boolean"}}},
        "side_effects": "marks review path terminal",
    },
]


@dataclass
class CriticToolState:
    evidence_ids: list[str] = field(default_factory=list)
    evidence_bundle: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] | None = None
    created_task_ids: list[str] = field(default_factory=list)
    created_message_ids: list[str] = field(default_factory=list)
    created_board_item_ids: list[str] = field(default_factory=list)
    repair_requested: bool = False
    deduped_issue_keys: list[str] = field(default_factory=list)
    terminal_status: str | None = None
    terminal_reason: str | None = None


class CriticToolHandlers:
    def __init__(
        self,
        *,
        task: Task,
        task_store: TaskStore,
        mailbox: Mailbox,
        artifact_store: ArtifactStore,
        board_store: BoardStore,
        state: CriticToolState,
    ):
        self.task = task
        self.task_store = task_store
        self.mailbox = mailbox
        self.artifact_store = artifact_store
        self.board_store = board_store
        self.state = state

    def handlers(self) -> dict[str, Any]:
        return {
            "read_review_task": self._guard_after_repair("read_review_task", self.read_review_task),
            "read_evidence_bundle": self._guard_after_repair("read_evidence_bundle", self.read_evidence_bundle),
            "read_evidence_map": self._guard_after_repair("read_evidence_map", self.read_evidence_map),
            "validate_evidence_bundle": self._guard_after_repair("validate_evidence_bundle", self.validate_evidence_bundle),
            "write_challenge": self._guard_after_repair("write_challenge", self.write_challenge),
            "request_repair": self._guard_after_repair("request_repair", self.request_repair),
            "record_conflict": self._guard_after_repair("record_conflict", self.record_conflict),
            "mark_review_passed": self._guard_after_repair("mark_review_passed", self.mark_review_passed),
            "reject_direction": self._guard_after_repair("reject_direction", self.reject_direction),
            "finish_review": self.finish_review,
        }

    def _guard_after_repair(self, tool_name: str, handler: Any) -> Any:
        def _wrapped(tool_input: dict[str, Any]) -> dict[str, Any]:
            if self.state.repair_requested:
                return {
                    "ok": False,
                    "error": "repair was already created for this review; call finish_review next",
                    "repair_requested": True,
                    "required_next_tool": "finish_review",
                    "attempted_tool": tool_name,
                }
            return handler(tool_input)

        return _wrapped

    def read_review_task(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        del tool_input
        evidence_ids = [str(item) for item in list(self.task.inputs.get("evidence_ids") or []) if str(item)]
        self.state.evidence_ids = evidence_ids
        return {
            "ok": True,
            "task_id": self.task.task_id,
            "review_type": "extraction_failure" if self.task.kind == "extraction_failure_review" else "evidence_review",
            "question": str(self.task.inputs.get("question") or self.task_store.store.get_swarm_run_state(self.task.run_id).objective),
            "evidence_ids": evidence_ids,
            "source_artifact_id": self.task.inputs.get("source_artifact_id"),
            "failure_reason": self.task.inputs.get("failure_reason"),
            "targeted_query": self.task.inputs.get("targeted_query"),
            "issue_key": self.task.inputs.get("issue_key"),
            "evidence_scope": self.task.inputs.get("evidence_scope") or "batch",
            "batch_id": self.task.inputs.get("batch_id"),
            "batch_ids": self.task.inputs.get("batch_ids") or [],
            "partial_bundle": bool(self.task.inputs.get("partial_bundle")),
            "batch_statuses": self.task.inputs.get("batch_statuses") or {},
            "evidence_bundle_key": self.task.inputs.get("evidence_bundle_key"),
            "repair_attempt": self.task.inputs.get("repair_attempt"),
            "max_repair_attempts": self.task.inputs.get("max_repair_attempts") or DEFAULT_MAX_REPAIR_ATTEMPTS,
            "repair_budget": {
                "blocking_repairs_used_for_run": self._blocking_repair_count_for_run(),
                "max_blocking_repairs_per_run": DEFAULT_MAX_BLOCKING_REPAIRS_PER_RUN,
                "blocking_repairs_used_for_review": self._blocking_repair_count_for_review(),
                "max_blocking_repairs_per_review": DEFAULT_MAX_BLOCKING_REPAIRS_PER_REVIEW,
            },
        }

    def read_evidence_bundle(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        del tool_input
        bundle = self._load_evidence_bundle()
        self.state.evidence_bundle = bundle
        return {"ok": True, "evidence": bundle}

    def read_evidence_map(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        del tool_input
        bundle = self._load_evidence_bundle()
        self.state.evidence_bundle = bundle
        by_source: dict[str, list[dict[str, Any]]] = {}
        for item in bundle:
            by_source.setdefault(_safe_text(item.get("source_url")) or "unknown", []).append(item)

        sources: list[dict[str, Any]] = []
        forward_looking_claims = 0
        recommended_detail_reads: list[str] = []
        source_role_hint_counts: dict[str, int] = {}
        largest_source_share = 0.0
        total_evidence = len(bundle)
        for source_url, items in sorted(by_source.items()):
            claims = [_safe_text(item.get("claim")) for item in items if _safe_text(item.get("claim"))]
            quotes = [_shorten(_safe_text(item.get("quote")), 220) for item in items if _safe_text(item.get("quote"))]
            source_role_hint = _source_role_hint(source_url)
            risk_flags = _source_risks(source_url, items, total_evidence=total_evidence)
            source_role_hint_counts[source_role_hint] = source_role_hint_counts.get(source_role_hint, 0) + 1
            if total_evidence:
                largest_source_share = max(largest_source_share, len(items) / total_evidence)
            forward_looking_claims += sum(1 for claim in claims if _looks_forward_looking(claim))
            if items:
                recommended_detail_reads.append(str(items[0].get("evidence_id") or ""))
            sources.append(
                {
                    "source_url": source_url,
                    "source_role_hint": source_role_hint,
                    "risk_flags": risk_flags,
                    "evidence_ids": [item.get("evidence_id") for item in items],
                    "evidence_count": len(items),
                    "claims": claims[:8],
                    "representative_quotes": quotes[:2],
                }
            )

        return {
            "ok": True,
            "evidence_count": len(bundle),
            "source_count": len(sources),
            "sources": sources,
            "coverage": {
                "source_role_hint_counts": source_role_hint_counts,
                "largest_source_share": round(largest_source_share, 3),
                "source_count": len(sources),
                "forward_looking_claims": forward_looking_claims,
                "ready_evidence": sum(1 for item in bundle if item.get("qa_state") == "ready"),
            },
            "recommended_detail_reads": [item for item in recommended_detail_reads if item],
            "note": "This is a compact map for subjective coverage judgment. Deterministic validation still checks the full scoped evidence set.",
        }

    def validate_evidence_bundle(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        del tool_input
        must_fix: list[str] = []
        if self.task.kind == "extraction_failure_review":
            reason = _safe_text(self.task.inputs.get("failure_reason")) or "Extractor could not create quote-backed citations."
            must_fix.append(f"Extraction failed before evidence review: {reason}")
            validation = {"passed": False, "must_fix": must_fix}
            self.state.validation = validation
            return {"ok": True, **validation}
        if not self.state.evidence_bundle:
            self.state.evidence_bundle = self._load_evidence_bundle()
        if not self.state.evidence_bundle:
            must_fix.append("No evidence is present in the review bundle.")
        for item in self.state.evidence_bundle:
            if not item.get("source_url"):
                must_fix.append(f"Evidence {item.get('evidence_id')} is missing source_url.")
            if not item.get("quote"):
                must_fix.append(f"Evidence {item.get('evidence_id')} is missing quote.")
            if item.get("citation_quote") and item.get("quote") != item.get("citation_quote"):
                must_fix.append(f"Evidence {item.get('evidence_id')} quote differs from citation payload quote.")
            if not item.get("claim"):
                must_fix.append(f"Evidence {item.get('evidence_id')} has no explicit claim in citation payload.")
        validation = {"passed": not must_fix, "must_fix": must_fix}
        self.state.validation = validation
        return {"ok": True, **validation}

    def write_challenge(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        message = self.mailbox.send(
            self.task.run_id,
            from_role=CRITIC_ROLE,
            broadcast=True,
            message_type="observation",
            payload={
                "kind": "evidence_gap",
                "summary": _safe_text(tool_input.get("summary")),
                "must_fix": [str(item) for item in list(tool_input.get("must_fix") or [])],
                "confidence": tool_input.get("confidence"),
                "evidence_ids": list(self.state.evidence_ids),
            },
            related_task_id=self.task.task_id,
        )
        self.state.created_message_ids.append(message.message_id or "")
        return {"ok": True, "message_id": message.message_id}

    def request_repair(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        targeted_query = _safe_text(tool_input.get("targeted_query"))
        must_fix = [str(item).strip() for item in list(tool_input.get("must_fix") or []) if str(item).strip()]
        review_basis = _review_basis(tool_input)
        if not review_basis:
            return _missing_review_basis("request_repair")
        issue_key = _stable_issue_key(targeted_query=targeted_query, must_fix=must_fix)
        repair_attempt = self._next_repair_attempt(issue_key)
        max_repair_attempts = int(self.task.inputs.get("max_repair_attempts") or DEFAULT_MAX_REPAIR_ATTEMPTS)
        if self._has_active_repair(issue_key):
            if issue_key not in self.state.deduped_issue_keys:
                self.state.deduped_issue_keys.append(issue_key)
            return {
                "ok": True,
                "repair_created": False,
                "deduped": True,
                "active_repair_exists": True,
                "issue_key": issue_key,
            }
        if self._blocking_repair_count_for_review() >= DEFAULT_MAX_BLOCKING_REPAIRS_PER_REVIEW:
            return self._write_repair_exhausted(
                issue_key=issue_key,
                targeted_query=targeted_query,
                must_fix=must_fix,
                error="review blocking repair budget exhausted",
                repair_attempt=repair_attempt,
                max_repair_attempts=max_repair_attempts,
            )
        if self._blocking_repair_count_for_run() >= DEFAULT_MAX_BLOCKING_REPAIRS_PER_RUN:
            return self._write_repair_exhausted(
                issue_key=issue_key,
                targeted_query=targeted_query,
                must_fix=must_fix,
                error="run blocking repair budget exhausted",
                repair_attempt=repair_attempt,
                max_repair_attempts=max_repair_attempts,
            )
        if repair_attempt > max_repair_attempts:
            return self._write_repair_exhausted(
                issue_key=issue_key,
                targeted_query=targeted_query,
                must_fix=must_fix,
                error="issue repair budget exhausted",
                repair_attempt=repair_attempt,
                max_repair_attempts=max_repair_attempts,
            )

        conflict = self.board_store.create_conflict(
            self.task.run_id,
            title=f"Evidence challenge: {targeted_query}",
            question_id=str(self.task.inputs.get("board_item_id") or "").strip() or None,
            status="open",
            priority=self.task.priority,
            created_by=CRITIC_ROLE,
            payload={"issue_key": issue_key, "evidence_ids": list(self.state.evidence_ids), "must_fix": must_fix, "why_current_evidence_failed": _safe_text(tool_input.get("why_current_evidence_failed")), "review_basis": review_basis},
            dedupe_key=f"conflict:{issue_key}:{repair_attempt}",
        )
        repair_task = self.task_store.create(
            self.task.run_id,
            kind="research_repair",
            status="pending",
            owner_role=RESEARCHER_ROLE,
            inputs={
                "targeted_query": targeted_query,
                "must_fix": must_fix,
                "preferred_source_type": _safe_text(tool_input.get("preferred_source_type")),
                "why_current_evidence_failed": _safe_text(tool_input.get("why_current_evidence_failed")),
                "issue_key": issue_key,
                "repair_attempt": repair_attempt,
                "max_repair_attempts": max_repair_attempts,
                "blocking_repair": True,
                "origin_review_task_id": self.task.task_id,
                "review_basis": review_basis,
            },
            priority=10,
            created_by=CRITIC_ROLE,
        )
        message = self.mailbox.send(
            self.task.run_id,
            from_role=CRITIC_ROLE,
            to_role=RESEARCHER_ROLE,
            message_type="request",
            payload={"kind": "research_repair", "task_id": repair_task.task_id, "targeted_query": targeted_query, "must_fix": must_fix, "issue_key": issue_key, "repair_attempt": repair_attempt, "max_repair_attempts": max_repair_attempts, "review_basis": review_basis},
            related_task_id=repair_task.task_id,
        )
        self.state.created_board_item_ids.append(conflict.item_id or "")
        self.state.created_task_ids.append(repair_task.task_id or "")
        self.state.created_message_ids.append(message.message_id or "")
        self.state.repair_requested = True
        return {
            "ok": True,
            "repair_created": True,
            "deduped": False,
            "repair_task_id": repair_task.task_id,
            "message_id": message.message_id,
            "issue_key": issue_key,
            "required_next_tool": "finish_review",
        }

    def record_conflict(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        review_basis = _review_basis(tool_input)
        if not review_basis:
            return _missing_review_basis("record_conflict")
        conflict = self.board_store.create_conflict(
            self.task.run_id,
            title=_safe_text(tool_input.get("summary")) or "Evidence conflict",
            question_id=str(self.task.inputs.get("board_item_id") or "").strip() or None,
            status="open",
            priority=self.task.priority,
            created_by=CRITIC_ROLE,
            payload={"reason": _safe_text(tool_input.get("reason")), "conflicting_evidence_ids": list(tool_input.get("conflicting_evidence_ids") or self.state.evidence_ids), "review_basis": review_basis},
            dedupe_key=f"conflict:{self.task.task_id}:{hashlib.sha1(_safe_text(tool_input.get('summary')).encode()).hexdigest()[:10]}",
        )
        message = self.mailbox.send(
            self.task.run_id,
            from_role=CRITIC_ROLE,
            broadcast=True,
            message_type="observation",
            payload={"kind": "conflict", "conflict_id": conflict.item_id, "summary": conflict.title, "reason": _safe_text(tool_input.get("reason")), "review_basis": review_basis},
            related_task_id=self.task.task_id,
        )
        self.state.created_board_item_ids.append(conflict.item_id or "")
        self.state.created_message_ids.append(message.message_id or "")
        return {"ok": True, "conflict_id": conflict.item_id, "message_id": message.message_id}

    def mark_review_passed(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        reason = _safe_text(tool_input.get("reason"))
        verdict = _safe_text(tool_input.get("verdict")) or "pass"
        if verdict not in {"pass", "pass_with_caveats"}:
            verdict = "pass"
        caveats = [str(item).strip() for item in list(tool_input.get("caveats") or []) if str(item).strip()]
        review_basis = _review_basis(tool_input)
        if not review_basis:
            return _missing_review_basis("mark_review_passed")
        issue_keys = [
            str(self.task.inputs.get("issue_key") or "").strip(),
            *self.board_store.issue_keys_for_evidence(self.task.run_id, list(self.state.evidence_ids)),
        ]
        resolved_conflicts = self.board_store.resolve_conflicts(
            self.task.run_id,
            issue_keys=[key for key in issue_keys if key],
            evidence_ids=list(self.state.evidence_ids),
            resolved_by=CRITIC_ROLE,
            reason=reason or "Critic passed the repaired evidence bundle.",
        )
        message = self.mailbox.send(
            self.task.run_id,
            from_role=CRITIC_ROLE,
            broadcast=True,
            message_type="response",
            payload={
                "kind": "pass",
                "verdict": verdict,
                "reason": reason,
                "confidence": tool_input.get("confidence"),
                "caveats": caveats,
                "evidence_ids": list(self.state.evidence_ids),
                "resolved_conflict_ids": [item.item_id for item in resolved_conflicts],
                "review_basis": review_basis,
            },
            related_task_id=self.task.task_id,
        )
        self.state.created_message_ids.append(message.message_id or "")
        self.state.created_board_item_ids.extend([item.item_id or "" for item in resolved_conflicts])
        return {"ok": True, "message_id": message.message_id, "resolved_conflict_ids": [item.item_id for item in resolved_conflicts]}

    def reject_direction(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        evidence_ids = [str(item).strip() for item in list(tool_input.get("evidence_ids") or self.state.evidence_ids) if str(item).strip()]
        review_basis = _review_basis(tool_input)
        if not review_basis:
            return _missing_review_basis("reject_direction")
        message = self.mailbox.send(
            self.task.run_id,
            from_role=CRITIC_ROLE,
            broadcast=True,
            message_type="observation",
            payload={
                "kind": "direction_rejected",
                "direction": _safe_text(tool_input.get("direction")),
                "source_url": _safe_text(tool_input.get("source_url")),
                "reason": _safe_text(tool_input.get("reason")),
                "confidence": tool_input.get("confidence"),
                "evidence_ids": evidence_ids,
                "blocking_repair": False,
                "review_basis": review_basis,
            },
            related_task_id=self.task.task_id,
        )
        self.state.created_message_ids.append(message.message_id or "")
        return {"ok": True, "message_id": message.message_id, "blocking_repair": False}

    def finish_review(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        status = _safe_text(tool_input.get("status")) or "complete"
        if status == "complete":
            status = "done"
        if status not in {"done", "blocked"}:
            status = "blocked"
        reason = _safe_text(tool_input.get("reason")) or status
        self.state.terminal_status = status
        self.state.terminal_reason = reason
        return {"ok": True, "terminal": True, "status": status, "reason": reason}

    def _summarize_evidence(self, evidence: Evidence) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        try:
            payload = self.artifact_store.read_payload(evidence.artifact_id)
        except Exception:
            payload = {}
        return {
            "evidence_id": evidence.evidence_id,
            "artifact_id": evidence.artifact_id,
            "source_url": evidence.source_url,
            "quote": evidence.quote,
            "freshness": evidence.freshness,
            "confidence": evidence.confidence,
            "qa_state": evidence.qa_state,
            "claim": payload.get("claim"),
            "citation_quote": payload.get("quote"),
            "title": payload.get("title"),
        }

    def _load_evidence_bundle(self) -> list[dict[str, Any]]:
        if not self.state.evidence_ids:
            self.state.evidence_ids = [str(item) for item in list(self.task.inputs.get("evidence_ids") or []) if str(item)]
        bundle: list[dict[str, Any]] = []
        for evidence in self.artifact_store.store.list_swarm_evidence(self.task.run_id):
            if self.state.evidence_ids and evidence.evidence_id not in self.state.evidence_ids:
                continue
            bundle.append(self._summarize_evidence(evidence))
        return bundle

    def _has_active_repair(self, issue_key: str) -> bool:
        for task in self.task_store.list_active(self.task.run_id):
            if task.kind in {"research_repair", "repair_request"} and str(task.inputs.get("issue_key") or "") == issue_key:
                return True
        return False

    def _blocking_repair_count_for_run(self) -> int:
        return sum(
            1
            for task in self.task_store.store.list_swarm_tasks(self.task.run_id)
            if task.kind in {"research_repair", "repair_request"}
            and bool(task.inputs.get("blocking_repair", True))
            and task.created_by == CRITIC_ROLE
        )

    def _blocking_repair_count_for_review(self) -> int:
        return sum(
            1
            for task in self.task_store.store.list_swarm_tasks(self.task.run_id)
            if task.kind in {"research_repair", "repair_request"}
            and bool(task.inputs.get("blocking_repair", True))
            and str(task.inputs.get("origin_review_task_id") or "") == str(self.task.task_id or "")
        )

    def _write_repair_exhausted(
        self,
        *,
        issue_key: str,
        targeted_query: str,
        must_fix: list[str],
        error: str,
        repair_attempt: int,
        max_repair_attempts: int,
    ) -> dict[str, Any]:
        exhausted = self.mailbox.send(
            self.task.run_id,
            from_role=CRITIC_ROLE,
            to_role="lead",
            message_type="observation",
            payload={
                "kind": "repair_exhausted",
                "issue_key": issue_key,
                "targeted_query": targeted_query,
                "must_fix": must_fix,
                "repair_attempt": repair_attempt,
                "max_repair_attempts": max_repair_attempts,
                "blocking_repairs_used_for_run": self._blocking_repair_count_for_run(),
                "max_blocking_repairs_per_run": DEFAULT_MAX_BLOCKING_REPAIRS_PER_RUN,
                "blocking_repairs_used_for_review": self._blocking_repair_count_for_review(),
                "max_blocking_repairs_per_review": DEFAULT_MAX_BLOCKING_REPAIRS_PER_REVIEW,
            },
            related_task_id=self.task.task_id,
        )
        self.state.created_message_ids.append(exhausted.message_id or "")
        return {"ok": False, "terminal": False, "error": error, "message_id": exhausted.message_id, "issue_key": issue_key}

    def _next_repair_attempt(self, issue_key: str) -> int:
        attempts = [
            int(task.inputs.get("repair_attempt") or 0)
            for task in self.task_store.store.list_swarm_tasks(self.task.run_id)
            if task.kind in {"research_repair", "repair_request"} and str(task.inputs.get("issue_key") or "") == issue_key
        ]
        return (max(attempts) if attempts else 0) + 1


def _stable_issue_key(*, targeted_query: str, must_fix: list[str]) -> str:
    normalized = json.dumps({"targeted_query": " ".join(targeted_query.lower().split()), "must_fix": sorted(" ".join(item.lower().split()) for item in must_fix)}, sort_keys=True)
    return f"issue.{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]}"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def _review_basis(tool_input: dict[str, Any]) -> dict[str, Any]:
    value = tool_input.get("review_basis")
    if not isinstance(value, dict):
        return {}
    allowed = {
        "quote_integrity",
        "claim_alignment",
        "coverage_for_review_scope",
        "source_concentration",
        "freshness_fit",
        "tensions",
        "review_disposition",
        "disposition_reason",
        "must_fix",
        "caveats",
        "rejected_direction",
    }
    return {key: value.get(key) for key in allowed if key in value}


def _missing_review_basis(tool_name: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": f"{tool_name} requires review_basis from the final Critic private review state",
        "required_next_step": "Form review_basis in private_state, then retry the same terminal tool with that review_basis.",
        "missing_review_basis": True,
        "terminal": False,
    }


def _source_role_hint(source_url: str) -> str:
    parsed = urlparse(_safe_text(source_url))
    tokens = " ".join(
        part.strip().lower()
        for part in [parsed.netloc, parsed.path, parsed.query]
        if part.strip()
    )
    if any(token in tokens for token in ["forum", "community", "discussion", "thread", "question", "answer"]):
        return "discussion_hint"
    if any(token in tokens for token in ["docs", "documentation", "reference", "api"]):
        return "documentation_hint"
    if any(token in tokens for token in ["press", "release", "announcement", "newsroom", "blog"]):
        return "publisher_page_hint"
    if any(token in tokens for token in ["news", "article", "story", "analysis"]):
        return "article_or_analysis_hint"
    return "unknown"


def _source_risks(source_url: str, items: list[dict[str, Any]], *, total_evidence: int) -> list[str]:
    risks: list[str] = []
    source_role_hint = _source_role_hint(source_url)
    if source_role_hint == "discussion_hint":
        risks.append("discussion_or_user_generated_possible")
    if total_evidence > 1 and len(items) / total_evidence >= 0.6:
        risks.append("source_dominates_bundle")
    if any(float(item.get("confidence") or 0.0) < 0.7 for item in items):
        risks.append("low_confidence_evidence")
    if any(not _safe_text(item.get("freshness")) for item in items):
        risks.append("missing_freshness")
    return risks


def _looks_forward_looking(value: str) -> bool:
    lower = value.lower()
    markers = [
        "next",
        "future",
        "plan",
        "aim",
        "will",
        "2025",
        "2026",
        "roadmap",
        "launch",
        "release",
        "agent",
        "agi",
        "superintelligence",
        "下一步",
        "未来",
        "计划",
        "发布",
        "智能体",
        "超级智能",
    ]
    return any(marker in lower for marker in markers)
