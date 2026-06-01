from __future__ import annotations

from pathlib import Path
from typing import Any

from insightswarm.agents.writer import WriterWorker
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.swarm_store import ArtifactStore, Mailbox, TaskStore


class _ModelResult:
    status = "ok"
    text = ""

    def __init__(self, json_data: dict[str, Any]):
        self.json_data = json_data


class _StubWriterModel:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, response_format=None, max_tokens=None, temperature=None, metadata=None):
        self.calls.append({"messages": messages, "metadata": metadata or {}, "response_format": response_format})
        return _ModelResult(self.responses.pop(0))


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def _seed_evidence(store: Store, run_id: str) -> str:
    artifact_store = ArtifactStore(store)
    citation_artifact = artifact_store.write_citation(
        run_id,
        source_task_id=None,
        citation={
            "source_url": "https://example.com/openai-roadmap",
            "quote": "OpenAI wants to simplify products and unify model families into GPT-5.",
            "claim": "OpenAI is moving toward a unified intelligence product strategy.",
            "rationale": "The quote explicitly links product simplification and model unification.",
        },
        summary="OpenAI roadmap citation",
    )
    evidence = artifact_store.create_evidence(
        run_id,
        artifact_id=citation_artifact.artifact_id,
        source_url="https://example.com/openai-roadmap",
        quote="OpenAI wants to simplify products and unify model families into GPT-5.",
        freshness="2026-06-01",
        confidence=0.95,
        qa_state="ready",
    )
    return evidence.evidence_id or ""


def test_writer_ooda_model_publishes_analytic_report(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="OpenAI下一步想干什么",
        budget={"max_steps": 12},
        phase="delivery",
        delivery_gate=True,
    )
    evidence_id = _seed_evidence(store, run_state.run_id)
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        broadcast=True,
        message_type="response",
        payload={"kind": "pass", "verdict": "pass", "evidence_ids": [evidence_id], "issue_key": "issue.writer_ooda"},
    )
    task = task_store.create(
        run_state.run_id,
        kind="delivery_request",
        status="pending",
        owner_role="writer",
        inputs={"question": run_state.objective, "report_kind": "report", "evidence_ids": [evidence_id]},
        created_by="delivery_gate",
    )
    report = {
        "executive_summary": "OpenAI's next move is to reduce product complexity by turning model choice into a unified intelligence layer.",
        "key_judgments": [
            {
                "statement": "OpenAI is shifting from model selection toward a unified product experience.",
                "confidence": "high",
                "supporting_evidence": [evidence_id, "https://example.com/openai-roadmap"],
            }
        ],
        "evidence_summary": {
            "thematic_clusters": [
                {
                    "theme": "Product simplification",
                    "summary": "The evidence links simplification with model family unification.",
                    "evidence_refs": [evidence_id, "https://example.com/openai-roadmap"],
                }
            ]
        },
        "caveats": [
            {"concern": "Only one source is present, so implementation timing remains uncertain.", "affected_judgments": ["unified product experience"]}
        ],
        "watchlist": [
            {"item": "ChatGPT model picker changes", "rationale": "This would show whether simplification is reaching users."}
        ],
        "sources": [evidence_id],
    }
    model = _StubWriterModel(
        [
            {
                "assistant_text": "Orienting on delivery context first.",
                "private_state": {"thesis_draft": "", "answer_frame": "", "thematic_clusters": [], "contradictions": [], "confidence_assessment": {}, "source_limits": "", "gaps": []},
                "tool_call": {"name": "read_delivery_context", "input": {}},
                "stop_reason": None,
            },
            {
                "assistant_text": "Reading evidence before drafting.",
                "private_state": {"thesis_draft": "Unified intelligence layer.", "answer_frame": "strategy shift", "thematic_clusters": [], "contradictions": [], "confidence_assessment": {}, "source_limits": "", "gaps": []},
                "tool_call": {"name": "read_evidence_bundle", "input": {}},
                "stop_reason": None,
            },
            {
                "assistant_text": "Drafting with thesis, caveat, and watchlist.",
                "private_state": {"thesis_draft": "Unified intelligence layer.", "answer_frame": "strategy shift", "thematic_clusters": ["product"], "contradictions": [], "confidence_assessment": {"core": "high"}, "source_limits": "single source", "gaps": []},
                "tool_call": {"name": "draft_report", "input": {"report": report, "readiness": "ready", "reason": "Evidence supports a concise analytic report."}},
                "stop_reason": None,
            },
            {
                "assistant_text": "Publishing the final report.",
                "private_state": {"thesis_draft": "Unified intelligence layer.", "answer_frame": "strategy shift", "thematic_clusters": ["product"], "contradictions": [], "confidence_assessment": {"core": "high"}, "source_limits": "single source", "gaps": []},
                "tool_call": {"name": "publish_report", "input": {"report_kind": "report", "report": report, "why_ready": "The report contains thesis, judgments, caveats, watchlist, and sources."}},
                "stop_reason": None,
            },
        ]
    )

    result = WriterWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id, model_client=model)

    artifacts = [artifact for artifact in store.list_swarm_artifacts(run_state.run_id) if artifact.type == "report"]
    body = Path(artifacts[0].payload_ref).read_text(encoding="utf-8")

    assert result is not None
    assert result.claimed_task_id == task.task_id
    assert [call["metadata"]["role"] for call in model.calls] == ["writer_tool_loop"] * 4
    assert "## Executive Summary" in body
    assert "## Key Judgments" in body
    assert "## Caveats" in body
    assert "## What To Watch" in body
    assert "https://example.com/openai-roadmap" in body


def test_writer_does_not_run_before_delivery_gate(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Gate closed writer",
        budget={"max_steps": 12},
        phase="research",
        delivery_gate=False,
    )
    task_store.create(
        run_state.run_id,
        kind="delivery_request",
        status="pending",
        owner_role="writer",
        inputs={"question": run_state.objective, "report_kind": "report", "evidence_ids": []},
        created_by="test",
    )

    assert WriterWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id) is None
    assert store.list_swarm_artifacts(run_state.run_id) == []
