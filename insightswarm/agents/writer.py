from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from threading import Event
from typing import Any

from insightswarm.agents.agent_loop import AgentLoopState, run_agent_loop
from insightswarm.agents.execution_cell import run_in_cell, run_supervised_once
from insightswarm.agents.tool_executor import ToolExecutor
from insightswarm.schemas.swarm import Task
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore


WRITER_TOOLS = [
    {
        "name": "read_delivery_context",
        "description": "Read the delivery gate context, critic verdict, report kind, question, and high-level board state. Does not include full evidence text.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {"type": "object", "properties": {"question": {"type": "string"}, "report_kind": {"type": "string"}}},
        "side_effects": "none",
    },
    {
        "name": "read_evidence_bundle",
        "description": "Read the citation-backed evidence available for delivery. Use this before drafting or publishing.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {"type": "object", "properties": {"evidence": {"type": "array"}}},
        "side_effects": "none",
    },
    {
        "name": "draft_report",
        "description": "Privately draft the structured intelligence report: thesis, key judgments, thematic clusters, caveats, watchlist, and source mapping. Does not publish.",
        "input_schema": {
            "type": "object",
            "properties": {
                "report": {"type": "object"},
                "readiness": {"type": "string", "enum": ["ready", "partial", "blocked"]},
                "reason": {"type": "string"},
            },
            "required": ["report", "readiness", "reason"],
        },
        "output_schema": {"type": "object", "properties": {"draft_ready": {"type": "boolean"}}},
        "side_effects": "private Writer memory only",
    },
    {
        "name": "publish_report",
        "description": "Publish the final Markdown report artifact. Use only after reading evidence and forming a structured draft.",
        "input_schema": {
            "type": "object",
            "properties": {
                "report_kind": {"type": "string", "enum": ["report", "report_partial", "report_blocked"]},
                "report": {"type": "object"},
                "why_ready": {"type": "string"},
            },
            "required": ["report_kind", "report", "why_ready"],
        },
        "output_schema": {"type": "object", "properties": {"report_artifact_id": {"type": "string"}}},
        "side_effects": "writes report artifact and completion message",
    },
    {
        "name": "finish_writing",
        "description": "Finish Writer only after a report artifact has been published, or block if publishing is impossible.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["complete", "blocked"]},
                "reason": {"type": "string"},
            },
            "required": ["status", "reason"],
        },
        "output_schema": {"type": "object", "properties": {"terminal": {"type": "boolean"}}},
        "side_effects": "marks Writer loop terminal",
    },
]


@dataclass(frozen=True)
class WriterWorkResult:
    claimed_task_id: str
    created_artifact_ids: list[str] = field(default_factory=list)
    created_message_ids: list[str] = field(default_factory=list)


@dataclass
class WriterToolState:
    context_read: bool = False
    evidence_read: bool = False
    draft_report: dict[str, Any] | None = None
    draft_readiness: str | None = None
    draft_reason: str | None = None
    created_artifact_ids: list[str] = field(default_factory=list)
    created_message_ids: list[str] = field(default_factory=list)
    terminal_status: str | None = None
    terminal_reason: str | None = None


class WriterToolHandlers:
    def __init__(
        self,
        *,
        task: Task,
        context: dict[str, object],
        artifact_store: ArtifactStore,
        mailbox: Mailbox,
        state: WriterToolState,
    ) -> None:
        self.task = task
        self.context = context
        self.artifact_store = artifact_store
        self.mailbox = mailbox
        self.state = state

    def handlers(self) -> dict[str, Any]:
        return {
            "read_delivery_context": self.read_delivery_context,
            "read_evidence_bundle": self.read_evidence_bundle,
            "draft_report": self.draft_report,
            "publish_report": self.publish_report,
            "finish_writing": self.finish_writing,
        }

    def read_delivery_context(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        del tool_input
        self.state.context_read = True
        return {
            "ok": True,
            "question": self.context["question"],
            "report_kind": self.context["report_kind"],
            "frontier_hash": self.context["frontier_hash"],
            "critic_verdict": self.context["critic_verdict"],
            "has_delivery_gap": self.context["has_delivery_gap"],
            "evidence_count": len(list(self.context["evidence_rows"])),
            "board_summary": _summarize_board(self.context.get("board")),
        }

    def read_evidence_bundle(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        del tool_input
        self.state.evidence_read = True
        return {
            "ok": True,
            "question": self.context["question"],
            "evidence": [_writer_evidence_view(item) for item in list(self.context["citations"])],
            "source_count": len({str(item.get("source_url") or "") for item in list(self.context["citations"])}),
        }

    def draft_report(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        if not self.state.evidence_read:
            return {"ok": False, "error": "read_evidence_bundle before drafting"}
        report = tool_input.get("report")
        if not isinstance(report, dict):
            return {"ok": False, "error": "draft_report requires report object"}
        self.state.draft_report = report
        self.state.draft_readiness = str(tool_input.get("readiness") or "")
        self.state.draft_reason = str(tool_input.get("reason") or "")
        return {
            "ok": True,
            "draft_ready": True,
            "readiness": self.state.draft_readiness,
            "reason": self.state.draft_reason,
            "coverage_ok": _structured_report_coverage_ok(report, list(self.context["citations"])),
        }

    def publish_report(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        if not self.state.evidence_read:
            return {"ok": False, "error": "read_evidence_bundle before publishing"}
        report_kind = _normalize_report_kind(str(tool_input.get("report_kind") or self.context["report_kind"]))
        report = tool_input.get("report") if isinstance(tool_input.get("report"), dict) else self.state.draft_report
        if not isinstance(report, dict):
            return {"ok": False, "error": "publish_report requires report object or prior draft_report"}
        citations = list(self.context["citations"])
        if citations and not _structured_report_coverage_ok(report, citations):
            return {
                "ok": False,
                "error": "report must cite all available source URLs in sources or supporting evidence",
                "required_source_urls": sorted({str(item.get("source_url") or "") for item in citations}),
            }
        if not citations:
            report_kind = "report_blocked"
        elif report_kind == "report" and (self.context["has_delivery_gap"] or self.context["critic_verdict"] != "pass"):
            report_kind = "report_partial"
        body = _structured_report_to_markdown(
            question=str(self.context["question"] or ""),
            report=report,
            citations=citations,
            report_kind=report_kind,
        )
        artifact = self.artifact_store.write_report(
            self.task.run_id,
            source_task_id=self.task.task_id,
            report_kind=report_kind,
            body=body,
            summary=f"{report_kind} for {self.context['question']}",
        )
        message = self.mailbox.send(
            self.task.run_id,
            from_role="writer",
            broadcast=True,
            message_type="observation",
            payload={
                "kind": "progress_update",
                "task_id": self.task.task_id,
                "report_artifact_id": artifact.artifact_id,
                "report_kind": report_kind,
                "frontier_hash": self.context["frontier_hash"],
                "evidence_ids": list(self.context["evidence_ids"]),
                "why_ready": str(tool_input.get("why_ready") or ""),
            },
            related_task_id=self.task.task_id,
        )
        self.state.created_artifact_ids.append(artifact.artifact_id or "")
        self.state.created_message_ids.append(message.message_id or "")
        return {
            "ok": True,
            "terminal": True,
            "status": "done",
            "report_artifact_id": artifact.artifact_id,
            "message_id": message.message_id,
            "report_kind": report_kind,
        }

    def finish_writing(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        if not self.state.created_artifact_ids:
            return {"ok": False, "error": "publish_report before finish_writing"}
        self.state.terminal_status = "done" if str(tool_input.get("status") or "") == "complete" else "blocked"
        self.state.terminal_reason = str(tool_input.get("reason") or self.state.terminal_status)
        return {"ok": True, "terminal": True, "status": self.state.terminal_status, "reason": self.state.terminal_reason}


class WriterWorker:
    def __init__(self, task_store: TaskStore, mailbox: Mailbox, artifact_store: ArtifactStore, board_store: BoardStore | None = None):
        self.task_store = task_store
        self.mailbox = mailbox
        self.artifact_store = artifact_store
        self.board_store = board_store or BoardStore(task_store.store)

    def run_once(self, run_id: str, *, model_client: object | None = None, run_root: Path | None = None) -> WriterWorkResult | None:
        run_state = self.task_store.store.get_swarm_run_state(run_id)
        if not run_state.delivery_gate or run_state.phase != "delivery":
            return None
        task = self.task_store.claim_next(run_id, owner_role="writer")
        if task is None:
            return None
        def _body(claimed: Task) -> WriterWorkResult:
            result = self._process_task(claimed, model_client=model_client)
            current = self.task_store.store.get_swarm_task(claimed.task_id)
            if current.status in {"pending", "leased"}:
                self.task_store.complete(claimed.task_id)
            return result

        return run_in_cell(
            task_store=self.task_store,
            mailbox=self.mailbox,
            task=task,
            role="writer",
            run_root=run_root,
            body=_body,
            make_failure_result=lambda failed: WriterWorkResult(claimed_task_id=failed.task_id or ""),
        )

    def _process_task(self, task: Task, *, model_client: object | None = None) -> WriterWorkResult:
        context = self._assemble_context(task)
        decision = self._decide_action(context)
        return self._run_writer_loop(task, context, decision, model_client=model_client)

    def _assemble_context(self, task: Task) -> dict[str, object]:
        evidence_ids = [str(item) for item in (task.inputs.get("evidence_ids") or []) if str(item)]
        evidence_rows = [
            self.artifact_store.store.get_swarm_evidence(evidence_id)
            for evidence_id in evidence_ids
        ]
        citations = [
            {
                "evidence_id": row.evidence_id,
                "quote": row.quote,
                "source_url": row.source_url,
                "text": row.quote,
                "confidence": row.confidence,
                "freshness": row.freshness,
                "qa_state": row.qa_state,
                "artifact_payload": _read_citation_payload(self.task_store.store, row.artifact_id),
            }
            for row in evidence_rows
        ]
        return {
            "task": task,
            "question": str(task.inputs.get("question") or ""),
            "report_kind": str(task.inputs.get("report_kind") or "report"),
            "frontier_hash": str(task.inputs.get("frontier_hash") or ""),
            "evidence_ids": evidence_ids,
            "evidence_rows": evidence_rows,
            "citations": citations,
            "critic_verdict": _latest_critic_verdict(self.task_store.store, task.run_id),
            "has_delivery_gap": _has_delivery_gap(self.task_store.store, task.run_id),
            "board": self.board_store.scoped_snapshot(
                task.run_id,
                question_text=str(task.inputs.get("question") or ""),
            ),
        }

    def _decide_action(self, context: dict[str, object]) -> dict[str, object]:
        report_kind = str(context["report_kind"] or "report")
        evidence_rows = list(context["evidence_rows"])
        if not evidence_rows:
            report_kind = "report_blocked"
        elif bool(context["has_delivery_gap"]) or str(context["critic_verdict"] or "") != "pass":
            report_kind = "report_partial"
        return {
            "action": "write_report",
            "rationale": "Writer can only synthesize citation-backed evidence once delivery is open.",
            "payload": {"report_kind": report_kind},
        }

    def _run_writer_loop(
        self,
        task: Task,
        context: dict[str, object],
        decision: dict[str, object],
        *,
        model_client: object | None = None,
    ) -> WriterWorkResult:
        del decision
        state = WriterToolState()
        if model_client is not None:
            handlers = WriterToolHandlers(
                task=task,
                context=context,
                artifact_store=self.artifact_store,
                mailbox=self.mailbox,
                state=state,
            )
            loop_state = AgentLoopState()
            run_agent_loop(
                model_client=model_client,
                system_prompt=_load_prompt(),
                tool_specs=WRITER_TOOLS,
                executor=ToolExecutor(WRITER_TOOLS, handlers.handlers()),
                initial_user_payload={
                    "delivery_task": {
                        "task_id": task.task_id,
                        "run_id": task.run_id,
                        "kind": task.kind,
                    },
                    "instruction": "Write an intelligence report. Orient on the evidence, judge confidence and caveats, then compose and publish.",
                },
                state=loop_state,
                safety_cap=8,
                max_tokens=6000,
                metadata_role="writer_tool_loop",
                metadata={
                    "run_id": task.run_id,
                    "task_id": task.task_id,
                    "operation": "writer_tool_loop",
                    "frontier_hash": context["frontier_hash"],
                },
            )
        if not state.created_artifact_ids:
            artifact_id, message_id = self._publish_fallback_report(
                task,
                context,
                reason="writer_model_did_not_publish_report",
            )
            state.created_artifact_ids.append(artifact_id)
            state.created_message_ids.append(message_id)
        return WriterWorkResult(
            claimed_task_id=task.task_id,
            created_artifact_ids=list(state.created_artifact_ids),
            created_message_ids=list(state.created_message_ids),
        )

    def _publish_fallback_report(self, task: Task, context: dict[str, object], *, reason: str) -> tuple[str, str]:
        citations = list(context["citations"])
        report_kind = _normalize_report_kind(str(context["report_kind"] or "report"))
        if not citations:
            report_kind = "report_blocked"
        elif report_kind == "report" or bool(context["has_delivery_gap"]) or str(context["critic_verdict"] or "") != "pass":
            report_kind = "report_partial"
        body = _fallback_report_body(
            question=str(context["question"] or ""),
            citations=citations,
            report_kind=report_kind,
            fallback_reason=reason,
        )
        artifact = self.artifact_store.write_report(
            task.run_id,
            source_task_id=task.task_id,
            report_kind=report_kind,
            body=body,
            summary=f"fallback {report_kind} for {context['question']}: {reason}",
        )
        message = self.mailbox.send(
            task.run_id,
            from_role="writer",
            broadcast=True,
            message_type="observation",
            payload={
                "kind": "progress_update",
                "task_id": task.task_id,
                "report_artifact_id": artifact.artifact_id,
                "report_kind": report_kind,
                "frontier_hash": context["frontier_hash"],
                "evidence_ids": list(context["evidence_ids"]),
                "fallback": True,
                "fallback_reason": reason,
            },
            related_task_id=task.task_id,
        )
        return artifact.artifact_id or "", message.message_id or ""

    def run_forever(
        self,
        run_id: str,
        stop_event: Event,
        *,
        poll_interval: float = 0.2,
        model_client: object | None = None,
        max_iterations: int | None = None,
        run_root: Path | None = None,
    ) -> list[WriterWorkResult]:
        results: list[WriterWorkResult] = []

        while not stop_event.is_set():
            result = run_supervised_once(
                stop_event=stop_event,
                poll_interval=poll_interval,
                call_once=lambda: self.run_once(run_id, model_client=model_client, run_root=run_root),
            )
            if result is None:
                stop_event.wait(poll_interval)
                continue
            results.append(result)
            if max_iterations is not None and len(results) >= max_iterations:
                break

        return results


def write_report(
    *,
    question: str,
    citations: list[dict],
    run_dir: Path,
    model_client: object | None = None,
) -> dict:
    prompt = _load_prompt()
    run_dir.mkdir(parents=True, exist_ok=True)
    body = ""
    if model_client is not None:
        try:
            result = model_client.complete(
                [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"question": question, "citations": citations},
                            ensure_ascii=False,
                        ),
                    },
                ],
                temperature=0.2,
            )
            body = (result.text or "").strip()
        except Exception:
            body = ""

    if not body:
        body = _fallback_report_body(
            question=question,
            citations=citations,
            report_kind="report_partial" if citations else "report_blocked",
            fallback_reason="legacy_writer_model_did_not_return_body",
        )

    report_path = run_dir / "report.md"
    report_path.write_text(body, encoding="utf-8")
    return {
        "body": body,
        "path": str(report_path),
    }


def _load_prompt() -> str:
    return (Path(__file__).resolve().parent.parent / "prompts" / "writer.md").read_text(encoding="utf-8")


def _read_citation_payload(store, artifact_id: str) -> dict:
    artifact = store.get_swarm_artifact(artifact_id)
    path = Path(artifact.payload_ref)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _latest_critic_verdict(store, run_id: str) -> str | None:
    messages = [
        message
        for message in store.list_swarm_messages(run_id)
        if message.from_role == "critic" and "verdict" in message.payload
    ]
    if not messages:
        return None
    return str(messages[-1].payload.get("verdict") or "")


def _has_delivery_gap(store, run_id: str) -> bool:
    return any(
        message.type == "observation" and str(message.payload.get("kind") or "") == "delivery_gap"
        for message in store.list_swarm_messages(run_id)
    )


def _normalize_report_kind(value: str) -> str:
    if value in {"report", "report_partial", "report_blocked"}:
        return value
    return "report"


def _summarize_board(value: object) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, dict):
        return {}
    summary: dict[str, list[dict[str, Any]]] = {}
    for key, items in value.items():
        rows = []
        for item in list(items)[-6:]:
            rows.append(
                {
                    "id": getattr(item, "item_id", None),
                    "kind": getattr(item, "kind", key),
                    "status": getattr(item, "status", None),
                    "title": getattr(item, "title", ""),
                    "payload": getattr(item, "payload", {}),
                }
            )
        summary[str(key)] = rows
    return summary


def _writer_evidence_view(citation: dict) -> dict[str, Any]:
    payload = dict(citation.get("artifact_payload") or {})
    return {
        "evidence_id": citation.get("evidence_id"),
        "source_url": citation.get("source_url"),
        "quote": citation.get("quote"),
        "claim": payload.get("claim"),
        "rationale": payload.get("rationale"),
        "confidence": citation.get("confidence"),
        "freshness": citation.get("freshness"),
        "qa_state": citation.get("qa_state"),
    }


def _structured_report_coverage_ok(report: dict[str, Any], citations: list[dict]) -> bool:
    if not citations:
        return True
    report_text = json.dumps(report, ensure_ascii=False, sort_keys=True, default=str)
    required_urls = {str(citation.get("source_url") or "") for citation in citations if citation.get("source_url")}
    return all(url in report_text for url in required_urls)


def _structured_report_to_markdown(
    *,
    question: str,
    report: dict[str, Any],
    citations: list[dict],
    report_kind: str,
) -> str:
    title = "# Partial Report" if report_kind == "report_partial" else f"# {question}"
    if report_kind == "report_blocked":
        title = "# Blocked Report"
    lines = [title, ""]
    summary = _safe_text(report.get("executive_summary"))
    if summary:
        lines.extend(["## Executive Summary", summary, ""])

    judgments = [item for item in list(report.get("key_judgments") or []) if isinstance(item, dict)]
    if judgments:
        lines.extend(["## Key Judgments"])
        for item in judgments:
            statement = _safe_text(item.get("statement"))
            confidence = _safe_text(item.get("confidence")) or "medium"
            refs = ", ".join(_safe_text(ref) for ref in list(item.get("supporting_evidence") or []) if _safe_text(ref))
            lines.append(f"- **{confidence} confidence**: {statement}" + (f" Evidence: {refs}" if refs else ""))
        lines.append("")

    evidence_summary = report.get("evidence_summary") if isinstance(report.get("evidence_summary"), dict) else {}
    clusters = [item for item in list(evidence_summary.get("thematic_clusters") or []) if isinstance(item, dict)]
    if clusters:
        lines.extend(["## Evidence Summary"])
        for item in clusters:
            theme = _safe_text(item.get("theme")) or "Theme"
            cluster_summary = _safe_text(item.get("summary"))
            refs = ", ".join(_safe_text(ref) for ref in list(item.get("evidence_refs") or []) if _safe_text(ref))
            lines.append(f"### {theme}")
            lines.append(cluster_summary)
            if refs:
                lines.append(f"Evidence: {refs}")
            lines.append("")

    caveats = [item for item in list(report.get("caveats") or []) if isinstance(item, dict)]
    if caveats:
        lines.extend(["## Caveats"])
        for item in caveats:
            concern = _safe_text(item.get("concern"))
            affected = ", ".join(_safe_text(ref) for ref in list(item.get("affected_judgments") or []) if _safe_text(ref))
            lines.append(f"- {concern}" + (f" Affects: {affected}" if affected else ""))
        lines.append("")

    watchlist = [item for item in list(report.get("watchlist") or []) if isinstance(item, dict)]
    if watchlist:
        lines.extend(["## What To Watch"])
        for item in watchlist:
            entry = _safe_text(item.get("item"))
            rationale = _safe_text(item.get("rationale"))
            lines.append(f"- {entry}" + (f": {rationale}" if rationale else ""))
        lines.append("")

    source_urls = _source_urls_from_report(report, citations)
    lines.extend(["## Sources"])
    for index, url in enumerate(source_urls, start=1):
        lines.append(f"- [{index}] {url}")
    return "\n".join(lines).strip()


def _source_urls_from_report(report: dict[str, Any], citations: list[dict]) -> list[str]:
    by_evidence_id = {str(citation.get("evidence_id") or ""): str(citation.get("source_url") or "") for citation in citations}
    urls: list[str] = []
    for value in list(report.get("sources") or []):
        text = _safe_text(value)
        url = by_evidence_id.get(text, text)
        if url and url not in urls:
            urls.append(url)
    for citation in citations:
        url = str(citation.get("source_url") or "")
        if url and url not in urls:
            urls.append(url)
    return urls


def _coverage_ok(body: str, citations: list[dict]) -> bool:
    if not citations:
        return False
    return all(str(citation.get("source_url") or "") in body for citation in citations)


def _fallback_report_body(*, question: str, citations: list[dict], report_kind: str, fallback_reason: str) -> str:
    notice = (
        "This is a fallback report generated because WriterAgent did not publish "
        f"a structured report. Fallback reason: {fallback_reason}."
    )
    if not citations:
        return f"# Blocked Report\n\n{notice}\n\nNo citation-backed evidence is available, so a report cannot be delivered."
    heading = "# Partial Report" if report_kind == "report_partial" else f"# {question}"
    lines = [heading, "", f"> {notice}", "", "## Findings"]
    for index, citation in enumerate(citations, start=1):
        source_url = citation.get("source_url") or ""
        quote = citation.get("quote") or ""
        lines.append(f"- {quote} [{index}] {source_url}")
    lines.append("")
    lines.append("## Sources")
    for index, citation in enumerate(citations, start=1):
        lines.append(f"- [{index}] {citation.get('source_url') or ''}")
    return "\n".join(lines).strip()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
