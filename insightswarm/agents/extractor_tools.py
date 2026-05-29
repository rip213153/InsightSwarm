from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

from insightswarm.schemas.swarm import Task
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore


EXTRACTOR_ROLE = "extractor"
RESEARCHER_ROLE = "researcher"


EXTRACTOR_TOOLS = [
    {
        "name": "read_raw_document",
        "description": "Read the raw_document artifact assigned to this extraction task.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "source_url": {"type": "string"},
                "title": {"type": "string"},
                "text_preview": {"type": "string"},
                "document_quality": {"type": "string"},
            },
        },
        "side_effects": "none",
    },
    {
        "name": "read_compressed_raw_view",
        "description": "Read a query-focused compressed view of the assigned raw document. Use this for long documents before proposing citations, but quotes must still come from the raw document text.",
        "input_schema": {
            "type": "object",
            "properties": {"focus": {"type": "string"}, "max_chars": {"type": "integer"}},
            "required": ["focus"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "source_url": {"type": "string"},
                "compressed_view": {"type": "string"},
                "selected_chunk_count": {"type": "integer"},
            },
        },
        "side_effects": "none",
    },
    {
        "name": "propose_citations",
        "description": "Submit exact quote-backed citation candidates. The tool deterministically backchecks quotes before writing citation artifacts and Evidence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "quote": {"type": "string"},
                            "rationale": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["claim", "quote"],
                    },
                },
                "why_these_quotes": {"type": "string"},
            },
            "required": ["candidates"],
        },
        "output_schema": {"type": "object", "properties": {"citation_artifact_ids": {"type": "array"}, "evidence_ids": {"type": "array"}}},
        "side_effects": "writes citation artifacts and formal Evidence if deterministic quote checks pass",
    },
    {
        "name": "request_better_source",
        "description": "Ask Researcher/Lead for a better raw source when this document cannot produce quote-backed citations.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}, "targeted_query": {"type": "string"}},
            "required": ["reason"],
        },
        "output_schema": {"type": "object", "properties": {"message_id": {"type": "string"}, "repair_task_id": {"type": "string"}}},
        "side_effects": "writes repair request message and researcher task",
    },
    {
        "name": "reject_document",
        "description": "Reject the raw document as blocked, irrelevant, boilerplate, or too low-signal.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}, "document_quality": {"type": "string"}},
            "required": ["reason"],
        },
        "output_schema": {"type": "object", "properties": {"message_id": {"type": "string"}}},
        "side_effects": "writes rejection observation",
    },
    {
        "name": "finish_extraction",
        "description": "Stop this Extractor loop after citations were written, the document was rejected, or repair was requested.",
        "input_schema": {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["complete", "blocked"]}, "reason": {"type": "string"}},
            "required": ["status", "reason"],
        },
        "output_schema": {"type": "object", "properties": {"terminal": {"type": "boolean"}}},
        "side_effects": "marks extraction path terminal",
    },
]


@dataclass
class ExtractorToolState:
    raw_document: dict[str, Any] | None = None
    artifact_id: str | None = None
    created_artifact_ids: list[str] = field(default_factory=list)
    created_evidence_ids: list[str] = field(default_factory=list)
    created_message_ids: list[str] = field(default_factory=list)
    created_task_ids: list[str] = field(default_factory=list)
    terminal_status: str | None = None
    terminal_reason: str | None = None


class ExtractorToolHandlers:
    def __init__(
        self,
        *,
        task: Task,
        task_store: TaskStore,
        mailbox: Mailbox,
        artifact_store: ArtifactStore,
        board_store: BoardStore,
        state: ExtractorToolState,
    ):
        self.task = task
        self.task_store = task_store
        self.mailbox = mailbox
        self.artifact_store = artifact_store
        self.board_store = board_store
        self.state = state

    def handlers(self) -> dict[str, Any]:
        return {
            "read_raw_document": self.read_raw_document,
            "read_compressed_raw_view": self.read_compressed_raw_view,
            "propose_citations": self.propose_citations,
            "request_better_source": self.request_better_source,
            "reject_document": self.reject_document,
            "finish_extraction": self.finish_extraction,
        }

    def read_raw_document(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        del tool_input
        artifact_id = str(self.task.inputs.get("artifact_id") or "")
        if not artifact_id:
            return {"ok": False, "error": "extractor task is missing artifact_id"}
        document = self.artifact_store.read_payload(artifact_id)
        self.state.artifact_id = artifact_id
        self.state.raw_document = document
        source_url = _source_url(document)
        text = _text(document)
        quality = _document_quality(text, source_url)
        return {
            "ok": True,
            "artifact_id": artifact_id,
            "source_url": source_url,
            "title": _safe_text(document.get("title")),
            "text_preview": text[:5000],
            "document_quality": quality,
            "freshness": document.get("freshness"),
            "question": _safe_text(document.get("question") or document.get("goal") or self.task_store.store.get_swarm_run_state(self.task.run_id).objective),
        }

    def read_compressed_raw_view(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        if self.state.raw_document is None:
            raw = self.read_raw_document({})
            if not raw.get("ok"):
                return raw
        document = self.state.raw_document or {}
        text = _text(document)
        focus = _safe_text(tool_input.get("focus")) or self.task_store.store.get_swarm_run_state(self.task.run_id).objective
        max_chars = int(tool_input.get("max_chars") or 5000)
        chunks = _rank_chunks(text, focus)
        selected: list[str] = []
        total = 0
        for chunk in chunks:
            if total >= max_chars:
                break
            selected.append(chunk["text"])
            total += len(chunk["text"])
        return {
            "ok": True,
            "source_url": _source_url(document),
            "title": _safe_text(document.get("title")),
            "focus": focus,
            "compressed_view": "\n\n---\n\n".join(selected),
            "selected_chunk_count": len(selected),
            "raw_text_length": len(text),
        }

    def propose_citations(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        if self.state.raw_document is None:
            return {"ok": False, "error": "read_raw_document first"}
        document = self.state.raw_document
        text = _text(document)
        source_url = _source_url(document)
        if not source_url:
            return {"ok": False, "error": "document is missing source_url"}
        accepted = []
        seen: set[str] = set()
        for item in list(tool_input.get("candidates") or [])[:5]:
            if not isinstance(item, dict):
                continue
            quote = _safe_text(item.get("quote"))
            normalized = _normalize_for_backcheck(quote)
            if not quote or not normalized or normalized in seen:
                continue
            if normalized not in _normalize_for_backcheck(text):
                continue
            if _looks_boilerplate(quote):
                continue
            seen.add(normalized)
            accepted.append(
                {
                    "claim": _safe_text(item.get("claim"))[:500],
                    "quote": quote,
                    "rationale": _safe_text(item.get("rationale"))[:300],
                    "confidence": _clamp_confidence(item.get("confidence")),
                }
            )
        if not accepted:
            return {"ok": False, "error": "no proposed quote passed deterministic backcheck"}

        citation_artifact_ids: list[str] = []
        evidence_ids: list[str] = []
        for candidate in accepted:
            citation_artifact = self.artifact_store.write_citation(
                self.task.run_id,
                source_task_id=self.task.task_id,
                citation={
                    "source_url": source_url,
                    "quote": candidate["quote"],
                    "title": _safe_text(document.get("title")) or source_url,
                    "freshness": document.get("freshness"),
                    "confidence": candidate["confidence"],
                    "text": text,
                    "claim": candidate["claim"],
                    "rationale": candidate["rationale"],
                },
                summary=candidate["quote"][:120],
            )
            evidence = self.artifact_store.create_evidence(
                self.task.run_id,
                artifact_id=citation_artifact.artifact_id,
                source_url=source_url,
                quote=candidate["quote"],
                freshness=document.get("freshness"),
                confidence=candidate["confidence"],
                qa_state="ready",
            )
            self.board_store.record_evidence(
                self.task.run_id,
                evidence=evidence,
                question_id=str(self.task.inputs.get("board_item_id") or "").strip() or None,
                artifact_id=citation_artifact.artifact_id,
                source_task_id=self.task.task_id,
            )
            citation_artifact_ids.append(citation_artifact.artifact_id or "")
            evidence_ids.append(evidence.evidence_id or "")

        message = self.mailbox.send(
            self.task.run_id,
            from_role=EXTRACTOR_ROLE,
            broadcast=True,
            message_type="observation",
            payload={
                "kind": "progress_update",
                "artifact_id": self.state.artifact_id,
                "citation_count": len(citation_artifact_ids),
                "evidence_ids": evidence_ids,
            },
            related_task_id=self.task.task_id,
        )
        review_task = self.task_store.create(
            self.task.run_id,
            kind="evidence_review",
            status="pending",
            owner_role="critic",
            inputs={
                "evidence_ids": evidence_ids,
                "evidence_bundle_key": "|".join(sorted(evidence_ids)),
                "question": self.task_store.store.get_swarm_run_state(self.task.run_id).objective,
            },
            depends_on=[],
            priority=self.task.priority,
            created_by=EXTRACTOR_ROLE,
        )
        review_request = self.mailbox.send(
            self.task.run_id,
            from_role=EXTRACTOR_ROLE,
            to_role="critic",
            message_type="request",
            payload={"kind": "review_evidence", "task_id": review_task.task_id, "evidence_ids": evidence_ids},
            related_task_id=review_task.task_id,
        )
        self.state.created_artifact_ids.extend(citation_artifact_ids)
        self.state.created_evidence_ids.extend(evidence_ids)
        self.state.created_message_ids.append(message.message_id or "")
        self.state.created_task_ids.append(review_task.task_id or "")
        self.state.created_message_ids.append(review_request.message_id or "")
        return {
            "ok": True,
            "citation_artifact_ids": citation_artifact_ids,
            "evidence_ids": evidence_ids,
            "message_id": message.message_id,
            "review_task_id": review_task.task_id,
        }

    def request_better_source(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        artifact_id = self.state.artifact_id or str(self.task.inputs.get("artifact_id") or "")
        reason = _safe_text(tool_input.get("reason")) or "No quote-backed citation could be extracted."
        repair_task = self.task_store.create(
            self.task.run_id,
            kind="research_repair",
            status="pending",
            owner_role=RESEARCHER_ROLE,
            inputs={
                "targeted_query": _safe_text(tool_input.get("targeted_query")) or reason,
                "source_artifact_id": artifact_id,
                "must_fix": [reason],
            },
            depends_on=[],
            priority=self.task.priority,
            created_by=EXTRACTOR_ROLE,
        )
        message = self.mailbox.send(
            self.task.run_id,
            from_role=EXTRACTOR_ROLE,
            to_role=RESEARCHER_ROLE,
            message_type="request",
            payload={"kind": "research_repair", "artifact_id": artifact_id, "must_fix": [reason]},
            related_task_id=repair_task.task_id,
        )
        self.state.created_task_ids.append(repair_task.task_id or "")
        self.state.created_message_ids.append(message.message_id or "")
        return {"ok": True, "message_id": message.message_id, "repair_task_id": repair_task.task_id}

    def reject_document(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        reason = _safe_text(tool_input.get("reason")) or "document rejected"
        message = self.mailbox.send(
            self.task.run_id,
            from_role=EXTRACTOR_ROLE,
            broadcast=True,
            message_type="observation",
            payload={"kind": "extraction_failure", "artifact_id": self.state.artifact_id or self.task.inputs.get("artifact_id"), "reason": reason, "document_quality": _safe_text(tool_input.get("document_quality"))},
            related_task_id=self.task.task_id,
        )
        self.state.created_message_ids.append(message.message_id or "")
        return {"ok": True, "message_id": message.message_id}

    def finish_extraction(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        status = _safe_text(tool_input.get("status")) or "complete"
        if status == "complete":
            status = "done"
        if status not in {"done", "blocked"}:
            status = "blocked"
        reason = _safe_text(tool_input.get("reason")) or status
        message = self.mailbox.send(
            self.task.run_id,
            from_role=EXTRACTOR_ROLE,
            to_role="lead",
            message_type="response",
            payload={"kind": "completed" if status == "done" else "blocked", "reason": reason},
            related_task_id=self.task.task_id,
        )
        self.state.created_message_ids.append(message.message_id or "")
        self.state.terminal_status = status
        self.state.terminal_reason = reason
        return {"ok": True, "terminal": True, "status": status, "reason": reason}


def _source_url(document: dict[str, Any]) -> str:
    return _safe_text(document.get("source_url") or document.get("url"))


def _text(document: dict[str, Any]) -> str:
    return _safe_text(document.get("text") or document.get("content") or document.get("body"))


def _document_quality(text: str, source_url: str) -> str:
    if not source_url or len(text) < 300:
        return "low_signal"
    lower = text.lower()
    if any(marker in lower for marker in ("captcha", "verify you are human", "access denied", "enable javascript")):
        return "blocked"
    return "usable"


def _rank_chunks(text: str, focus: str, chunk_size: int = 1200, overlap: int = 160) -> list[dict[str, Any]]:
    normalized = _safe_text(text)
    if not normalized:
        return []
    chunk_size = max(400, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2))
    focus_terms = _tokens(focus)
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_size)
        chunk_text = normalized[start:end].strip()
        if chunk_text:
            chunk_terms = _tokens(chunk_text)
            overlap_terms = sorted(focus_terms & chunk_terms)
            score = len(overlap_terms) * 2.0
            if re.search(r"\d{4}|\d+(?:\.\d+)?%|人民币|亿元|million|billion|revenue|growth", chunk_text, re.IGNORECASE):
                score += 1.5
            if any(marker in chunk_text.lower() for marker in ("annual report", "earnings", "strategy", "战略", "财报", "投入", "增长", "业务")):
                score += 1.0
            if start == 0:
                score += 0.5
            chunks.append(
                {
                    "score": round(score, 2),
                    "start": start,
                    "end": end,
                    "matched_terms": overlap_terms[:8],
                    "text": chunk_text,
                }
            )
        if end >= len(normalized):
            break
        start = end - overlap
    ranked = sorted(chunks, key=lambda item: (item["score"], -item["start"]), reverse=True)
    return ranked or chunks[:1]


def _tokens(value: str) -> set[str]:
    lowered = _safe_text(value).lower()
    ascii_tokens = set(re.findall(r"[a-z0-9][a-z0-9_\-]{2,}", lowered))
    cjk_terms = {
        term
        for term in (
            "京东",
            "外卖",
            "即时",
            "零售",
            "物流",
            "供应链",
            "机器人",
            "战略",
            "投资",
            "财报",
            "收入",
            "增长",
            "利润",
            "业务",
            "板块",
            "官方",
            "技术",
            "价格",
            "版本",
            "配置",
        )
        if term in value
    }
    return ascii_tokens | cjk_terms


def _normalize_for_backcheck(value: str) -> str:
    return re.sub(r"\s+", " ", _safe_text(value)).lower()


def _looks_boilerplate(value: str) -> bool:
    normalized = _normalize_for_backcheck(value)
    boilerplate = ("cookie", "privacy policy", "sign in", "reload to refresh", "navigation")
    min_length = 12 if _contains_cjk(normalized) else 40
    return len(normalized) < min_length or any(token in normalized for token in boilerplate)


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def _clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.5
    return max(0.0, min(1.0, number))


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
