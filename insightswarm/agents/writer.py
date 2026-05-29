from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from threading import Event

from insightswarm.schemas.swarm import Task
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore


@dataclass(frozen=True)
class WriterWorkResult:
    claimed_task_id: str
    created_artifact_ids: list[str] = field(default_factory=list)
    created_message_ids: list[str] = field(default_factory=list)


class WriterWorker:
    def __init__(self, task_store: TaskStore, mailbox: Mailbox, artifact_store: ArtifactStore, board_store: BoardStore | None = None):
        self.task_store = task_store
        self.mailbox = mailbox
        self.artifact_store = artifact_store
        self.board_store = board_store or BoardStore(task_store.store)

    def run_once(self, run_id: str, *, model_client: object | None = None) -> WriterWorkResult | None:
        run_state = self.task_store.store.get_swarm_run_state(run_id)
        if not run_state.delivery_gate or run_state.phase != "delivery":
            return None
        task = self.task_store.claim_next(run_id, owner_role="writer")
        if task is None:
            return None
        result = self._process_task(task, model_client=model_client)
        current = self.task_store.store.get_swarm_task(task.task_id)
        if current.status in {"pending", "leased"}:
            self.task_store.complete(task.task_id)
        return result

    def _process_task(self, task: Task, *, model_client: object | None = None) -> WriterWorkResult:
        context = self._assemble_context(task)
        decision = self._decide_action(context)
        return self._write_state(task, context, decision, model_client=model_client)

    def _assemble_context(self, task: Task) -> dict[str, object]:
        evidence_ids = [str(item) for item in (task.inputs.get("evidence_ids") or []) if str(item)]
        evidence_rows = [
            self.artifact_store.store.get_swarm_evidence(evidence_id)
            for evidence_id in evidence_ids
        ]
        citations = [
            {
                "quote": row.quote,
                "source_url": row.source_url,
                "text": row.quote,
                "artifact_payload": _read_citation_payload(self.task_store.store, row.artifact_id),
            }
            for row in evidence_rows
        ]
        return {
            "task": task,
            "question": str(task.inputs.get("question") or ""),
            "report_kind": str(task.inputs.get("report_kind") or "report"),
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

    def _write_state(
        self,
        task: Task,
        context: dict[str, object],
        decision: dict[str, object],
        *,
        model_client: object | None = None,
    ) -> WriterWorkResult:
        question = str(context["question"] or "")
        citations = list(context["citations"])
        report_kind = str((decision.get("payload") or {}).get("report_kind") or "report")
        report = write_report(
            question=question,
            citations=citations,
            run_dir=self.artifact_store.store.artifact_dir / task.run_id / "writer",
            model_client=model_client,
        )
        if not _coverage_ok(str(report["body"]), citations):
            fallback_kind = report_kind if citations else "report_blocked"
            fallback_body = _fallback_report_body(question=question, citations=citations, report_kind=fallback_kind)
            if not _coverage_ok(fallback_body, citations):
                report_kind = "report_partial" if citations else "report_blocked"
                fallback_body = _fallback_report_body(question=question, citations=citations, report_kind=report_kind)
            report = {
                "body": fallback_body,
                "path": "",
            }
        artifact = self.artifact_store.write_report(
            task.run_id,
            source_task_id=task.task_id,
            report_kind=report_kind,
            body=str(report["body"]),
            summary=f"{report_kind} for {question}",
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
            },
            related_task_id=task.task_id,
        )
        return WriterWorkResult(
            claimed_task_id=task.task_id,
            created_artifact_ids=[artifact.artifact_id],
            created_message_ids=[message.message_id],
        )

    def run_forever(
        self,
        run_id: str,
        stop_event: Event,
        *,
        poll_interval: float = 0.2,
        model_client: object | None = None,
        max_iterations: int | None = None,
    ) -> list[WriterWorkResult]:
        results: list[WriterWorkResult] = []

        while not stop_event.is_set():
            result = self.run_once(run_id, model_client=model_client)
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
                            ensure_ascii=True,
                        ),
                    },
                ],
                temperature=0.2,
            )
            body = (result.text or "").strip()
        except Exception:
            body = ""

    if not body:
        body = _fallback_report_body(question=question, citations=citations, report_kind="report")

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


def _coverage_ok(body: str, citations: list[dict]) -> bool:
    if not citations:
        return False
    return all(str(citation.get("source_url") or "") in body for citation in citations)


def _fallback_report_body(*, question: str, citations: list[dict], report_kind: str) -> str:
    if not citations:
        return f"# {question}\n\nNo citation-backed evidence is available, so a report cannot be delivered."
    heading = "# Partial Report" if report_kind == "report_partial" else f"# {question}"
    lines = [heading, "", "## Findings"]
    for index, citation in enumerate(citations, start=1):
        source_url = citation.get("source_url") or ""
        quote = citation.get("quote") or ""
        lines.append(f"- {quote} [{index}] {source_url}")
    lines.append("")
    lines.append("## Sources")
    for index, citation in enumerate(citations, start=1):
        lines.append(f"- [{index}] {citation.get('source_url') or ''}")
    return "\n".join(lines).strip()
