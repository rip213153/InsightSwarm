from __future__ import annotations

from pathlib import Path

from insightswarm.agents.critic import CriticWorker, review_evidence
from insightswarm.agents.extractor import ExtractorWorker
from insightswarm.agents.sub_researcher import SubResearcherWorker
from insightswarm.agents.writer import WriterWorker
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.delivery_gate import synchronize_delivery_gate
from insightswarm.models.clients import ModelResult
from insightswarm.swarm_store import ArtifactStore, Mailbox, TaskStore
from insightswarm.tools.core import ToolResult


REPO_ROOT = Path(__file__).resolve().parents[1]


class StubModelClient:
    def __init__(self, responses: list[dict | None]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, messages, response_format=None, max_tokens=None, temperature=None, metadata=None):
        self.calls.append(
            {
                "messages": messages,
                "response_format": response_format,
                "metadata": metadata or {},
            }
        )
        data = self.responses.pop(0) if self.responses else None
        return ModelResult(
            text="" if data is None else str(data),
            json_data=data,
            provider="stub",
            model="stub-json",
            usage={},
            latency_ms=1,
            raw_response={},
            status="ok",
        )


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def test_subresearcher_decide_action_uses_model_json(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    run_state = store.create_swarm_run_state(objective="Scoped objective", budget={})
    worker = SubResearcherWorker(TaskStore(store), Mailbox(store), ArtifactStore(store))
    task = TaskStore(store).create(
        run_state.run_id,
        kind="research_subquestion",
        status="pending",
        owner_role="sub_researcher",
        inputs={"question": "What is the release date?"},
    )
    context = worker._assemble_context(task, {
        "objective": run_state.objective,
        "search_count": 0,
        "fetch_count": 0,
        "suggest_browser_count": 0,
        "attempted_queries": [],
        "attempted_urls": [],
        "search_results": [],
        "documents": [],
        "errors": [],
    })
    client = StubModelClient(
        [
            {
                "action": "search",
                "rationale": "Try a focused query first.",
                "query": "official release date",
                "url": None,
                "source_url_key": None,
                "reason": None,
                "confidence": 0.77,
            }
        ]
    )

    decision = worker._decide_action(context, model_client=client)

    assert decision["action"] == "search"
    assert decision["query"] == "official release date"
    assert client.calls[0]["metadata"]["role"] == "sub_researcher_decision"


def test_subresearcher_invalid_action_falls_back_to_deterministic_plan(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    run_state = store.create_swarm_run_state(objective="Scoped objective", budget={})
    task_store = TaskStore(store)
    worker = SubResearcherWorker(task_store, Mailbox(store), ArtifactStore(store))
    task = task_store.create(
        run_state.run_id,
        kind="research_subquestion",
        status="pending",
        owner_role="sub_researcher",
        inputs={"question": "Need a source"},
    )
    context = worker._assemble_context(task, {
        "objective": run_state.objective,
        "search_count": 0,
        "fetch_count": 0,
        "suggest_browser_count": 0,
        "attempted_queries": [],
        "attempted_urls": [],
        "search_results": [],
        "documents": [],
        "errors": [],
    })
    client = StubModelClient([{"action": "invent_evidence"}])

    decision = worker._decide_action(context, model_client=client)
    validated = worker._validate_action(context, decision)

    assert validated["action"] == "search"
    assert validated["model_error"] is True
    assert validated["query"]


def test_subresearcher_search_action_calls_search_only(tmp_path: Path, monkeypatch) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="Search only", budget={})
    task_store.create(
        run_state.run_id,
        kind="research_subquestion",
        status="pending",
        owner_role="sub_researcher",
        inputs={"question": "Find an official source"},
        created_by="lead",
    )
    search_calls: list[str] = []
    fetch_calls: list[str] = []

    def _search(self, tool_input, context=None):
        search_calls.append(str(tool_input.get("query") or ""))
        return ToolResult("ok", data={"results": []})

    def _fetch(self, tool_input, context=None):
        fetch_calls.append(str(tool_input.get("url") or ""))
        return ToolResult("ok", data={"text": "should not happen"})

    monkeypatch.setattr("insightswarm.agents.sub_researcher.SearchTool.run", _search)
    monkeypatch.setattr("insightswarm.agents.sub_researcher.FetchUrlTool.run", _fetch)
    client = StubModelClient(
        [
            {
                "action": "search",
                "rationale": "Search first.",
                "query": "official source",
                "url": None,
                "source_url_key": None,
                "reason": None,
                "confidence": 0.8,
            },
            {
                "action": "blocked",
                "rationale": "No more candidates.",
                "query": None,
                "url": None,
                "source_url_key": None,
                "reason": "search yielded no usable candidates",
                "confidence": 0.3,
            },
        ]
    )

    result = SubResearcherWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id, model_client=client)

    assert result is not None
    assert len(search_calls) == 1
    assert fetch_calls == []


def test_subresearcher_fetch_action_writes_raw_document_and_extractor_request(tmp_path: Path, monkeypatch) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="Fetch only", budget={})
    task_store.create(
        run_state.run_id,
        kind="research_subquestion",
        status="pending",
        owner_role="sub_researcher",
        inputs={"question": "Get the source"},
        created_by="lead",
    )

    def _fetch(self, tool_input, context=None):
        return ToolResult(
            "ok",
            data={
                "text": "The official source says the release date is October 7, 2025.",
                "html": "<html><body>The official source says the release date is October 7, 2025.</body></html>",
                "title": "Official schedule",
            },
        )

    monkeypatch.setattr("insightswarm.agents.sub_researcher.FetchUrlTool.run", _fetch)
    client = StubModelClient(
        [
            {
                "action": "fetch",
                "rationale": "Fetch the only candidate URL.",
                "query": None,
                "url": "https://example.com/source",
                "source_url_key": "https://example.com/source",
                "reason": None,
                "confidence": 0.8,
            }
        ]
    )

    result = SubResearcherWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id, model_client=client)
    raw_documents = [artifact for artifact in store.list_swarm_artifacts(run_state.run_id) if artifact.type == "raw_document"]
    extractor_requests = [
        message
        for message in mailbox.inbox(run_state.run_id, role="extractor")
        if message.type == "request" and message.payload.get("kind") == "extract_evidence"
    ]

    assert result is not None
    assert len(raw_documents) == 1
    assert len(extractor_requests) == 1


def test_extractor_decide_action_uses_model_json(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="Extract", budget={})
    raw = artifact_store.write_raw_document(
        run_state.run_id,
        source_task_id=None,
        document={"url": "https://example.com", "text": "The project will ship on 2025-10-07.", "title": "Roadmap"},
        summary="Raw source",
    )
    task = task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="pending",
        owner_role="extractor",
        inputs={"artifact_id": raw.artifact_id},
    )
    worker = ExtractorWorker(task_store, Mailbox(store), artifact_store)
    client = StubModelClient(
        [
            {
                "action": "propose_citations",
                "rationale": "This quote is directly usable.",
                "document_quality": "usable",
                "candidates": [
                    {
                        "claim": "The project will ship on 2025-10-07.",
                        "quote": "The project will ship on 2025-10-07.",
                        "rationale": "Exact date quote.",
                        "confidence": 0.81,
                    }
                ],
                "repair_reason": None,
            }
        ]
    )

    decision = worker._decide_action(worker._assemble_context(task), model_client=client)

    assert decision["action"] == "propose_citations"
    assert decision["candidates"][0]["quote"] == "The project will ship on 2025-10-07."
    assert client.calls[0]["metadata"]["role"] == "extractor_decision"


def test_extractor_invalid_action_falls_back_to_deterministic_quote(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="Extract", budget={})
    raw = artifact_store.write_raw_document(
        run_state.run_id,
        source_task_id=None,
        document={
            "url": "https://example.com",
            "text": "The project will ship on 2025-10-07.",
            "title": "Roadmap",
        },
        summary="Raw source",
    )
    task = task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="pending",
        owner_role="extractor",
        inputs={"artifact_id": raw.artifact_id},
    )
    worker = ExtractorWorker(task_store, Mailbox(store), artifact_store)
    client = StubModelClient([{"action": "invent_evidence"}])

    decision = worker._decide_action(worker._assemble_context(task), model_client=client)
    validated = worker._validate_action(worker._assemble_context(task), decision)

    assert validated["action"] == "propose_citations"
    assert validated["model_error"] is True
    assert validated["candidates"][0]["quote"].startswith("The project will ship on 2025-10-07")


def test_critic_keeps_deterministic_pass_when_model_is_over_strict() -> None:
    client = StubModelClient(
        [
            {
                "verdict": "repair",
                "must_fix": ["Ask for more market detail."],
                "targeted_query": "more detail",
                "source_quality": "mixed",
                "conflicts": [],
                "confidence": 0.82,
            }
        ]
    )

    verdict = review_evidence(
        citations=[
            {
                "quote": "DeepSeek confirmed its public roadmap.",
                "source_url": "https://example.com/deepseek",
                "text": "DeepSeek confirmed its public roadmap.",
            }
        ],
        model_client=client,
        question="What is DeepSeek doing next?",
    )

    assert verdict["verdict"] == "pass"
    assert verdict["must_fix"] == []
    assert verdict["model_verdict"] == "repair"


def test_extractor_rejects_quote_not_in_document(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="Reject bad quote", budget={})
    raw = artifact_store.write_raw_document(
        run_state.run_id,
        source_task_id=None,
        document={"url": "https://example.com", "text": "Only this sentence exists.", "title": "Source"},
        summary="Raw source",
    )
    task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="pending",
        owner_role="extractor",
        inputs={"artifact_id": raw.artifact_id},
    )
    client = StubModelClient(
        [
            {
                "action": "propose_citations",
                "rationale": "Use this quote.",
                "document_quality": "usable",
                "candidates": [
                    {
                        "claim": "Invented sentence.",
                        "quote": "Invented sentence.",
                        "rationale": "bad quote",
                        "confidence": 0.9,
                    }
                ],
                "repair_reason": None,
            }
        ]
    )

    result = ExtractorWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id, model_client=client)

    assert result is not None
    assert store.list_swarm_evidence(run_state.run_id) == []
    assert any(message.type == "request" and message.payload.get("kind") == "research_repair" for message in mailbox.inbox(run_state.run_id, role="lead"))


def test_extractor_accepted_quote_writes_evidence(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="Accept quote", budget={})
    raw = artifact_store.write_raw_document(
        run_state.run_id,
        source_task_id=None,
        document={"url": "https://example.com", "text": "The official update says the launch date is 2025-10-07.", "title": "Update"},
        summary="Raw source",
    )
    task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="pending",
        owner_role="extractor",
        inputs={"artifact_id": raw.artifact_id},
    )
    client = StubModelClient(
        [
            {
                "action": "propose_citations",
                "rationale": "Use the exact sentence.",
                "document_quality": "usable",
                "candidates": [
                    {
                        "claim": "The launch date is 2025-10-07.",
                        "quote": "The official update says the launch date is 2025-10-07.",
                        "rationale": "Exact quote.",
                        "confidence": 0.82,
                    }
                ],
                "repair_reason": None,
            }
        ]
    )

    result = ExtractorWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id, model_client=client)

    assert result is not None
    assert len(store.list_swarm_evidence(run_state.run_id)) == 1


def test_extractor_parse_fail_does_not_write_evidence(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="Parse fail", budget={})
    raw = artifact_store.write_raw_document(
        run_state.run_id,
        source_task_id=None,
        document={"url": "https://example.com", "text": "A document with plain text only.", "title": "Plain"},
        summary="Raw source",
    )
    task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="pending",
        owner_role="extractor",
        inputs={"artifact_id": raw.artifact_id},
    )
    client = StubModelClient([None])

    result = ExtractorWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id, model_client=client)

    assert result is not None
    assert store.list_swarm_evidence(run_state.run_id) == []


def test_ooda_flow_runs_with_model_stubs(tmp_path: Path, monkeypatch) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="What is the release date?", budget={}, phase="research")
    task_store.create(
        run_state.run_id,
        kind="research_subquestion",
        status="pending",
        owner_role="sub_researcher",
        inputs={"question": run_state.objective},
        created_by="lead",
    )

    def _search(self, tool_input, context=None):
        return ToolResult(
            "ok",
            data={"results": [{"title": "Official update", "url": "https://example.com/release", "snippet": "Official announcement"}]},
        )

    def _fetch(self, tool_input, context=None):
        return ToolResult(
            "ok",
            data={
                "text": "The official update says the release date is 2025-10-07.",
                "html": "<html><body>The official update says the release date is 2025-10-07.</body></html>",
                "title": "Official update",
            },
        )

    monkeypatch.setattr("insightswarm.agents.sub_researcher.SearchTool.run", _search)
    monkeypatch.setattr("insightswarm.agents.sub_researcher.FetchUrlTool.run", _fetch)
    sub_client = StubModelClient(
        [
            {
                "action": "search",
                "rationale": "Search first.",
                "query": "official release date",
                "url": None,
                "source_url_key": None,
                "reason": None,
                "confidence": 0.8,
            },
            {
                "action": "fetch",
                "rationale": "Fetch the official result.",
                "query": None,
                "url": "https://example.com/release",
                "source_url_key": "https://example.com/release",
                "reason": None,
                "confidence": 0.8,
            },
        ]
    )
    extractor_client = StubModelClient(
        [
            {
                "action": "propose_citations",
                "rationale": "Use the direct statement.",
                "document_quality": "usable",
                "candidates": [
                    {
                        "claim": "The release date is 2025-10-07.",
                        "quote": "The official update says the release date is 2025-10-07.",
                        "rationale": "Exact quote.",
                        "confidence": 0.83,
                    }
                ],
                "repair_reason": None,
            }
        ]
    )

    assert SubResearcherWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id, model_client=sub_client) is not None
    assert ExtractorWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id, model_client=extractor_client) is not None
    assert CriticWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id) is not None
    synchronize_delivery_gate(store, run_state.run_id)
    assert WriterWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id) is not None

    assert store.list_swarm_evidence(run_state.run_id)
    assert any(artifact.type in {"report", "report_partial"} for artifact in store.list_swarm_artifacts(run_state.run_id))


def test_extractor_contains_no_domain_hardcoding() -> None:
    text = (REPO_ROOT / "insightswarm" / "agents" / "extractor.py").read_text(encoding="utf-8").lower()
    banned = ["python 3.14", "pep", "pricing", "release", "schedule", "final"]
    assert all(token not in text for token in banned)
