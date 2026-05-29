from __future__ import annotations

import json
from pathlib import Path

from insightswarm.agents.critic import CriticWorker
from insightswarm.agents.extractor import ExtractorWorker
from insightswarm.tools.fetch import _clean_html
from insightswarm.agents.lead import LeadWorker
from insightswarm.agents.sub_researcher import SubResearcherWorker
from insightswarm.agents.writer import WriterWorker
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.swarm_store import ArtifactStore, Mailbox, TaskStore


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def test_subresearcher_uses_search_fetch_and_only_outputs_raw_document(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INSIGHTSWARM_SCRIPTED_FIXTURE", "deliver_minimal")
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="DeepSeek roadmap", budget={})
    task_store.create(
        run_state.run_id,
        kind="research_subquestion",
        status="pending",
        owner_role="sub_researcher",
        inputs={"question": "DeepSeek roadmap"},
        created_by="lead",
    )

    result = SubResearcherWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    artifacts = store.list_swarm_artifacts(run_state.run_id)
    extractor_tasks = [
        task
        for task in store.list_swarm_tasks(run_state.run_id)
        if task.owner_role == "extractor" and task.kind == "raw_document"
    ]

    assert result is not None
    assert len([artifact for artifact in artifacts if artifact.type == "raw_document"]) == 1
    assert len([artifact for artifact in artifacts if artifact.type == "citation"]) == 0
    assert store.list_swarm_evidence(run_state.run_id) == []
    assert len(extractor_tasks) == 1


def test_extractor_rejects_document_without_source_url(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="missing source", budget={})
    raw = artifact_store.write_raw_document(
        run_state.run_id,
        source_task_id=None,
        document={"url": "", "text": "A useful quote with no usable source URL."},
        summary="No source URL",
    )
    task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="pending",
        owner_role="extractor",
        inputs={"artifact_id": raw.artifact_id},
        created_by="test",
    )

    result = ExtractorWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    lead_messages = mailbox.inbox(run_state.run_id, role="lead")

    assert result is not None
    assert result.created_evidence_ids == []
    assert store.list_swarm_evidence(run_state.run_id) == []
    assert any(message.type == "request" and message.payload.get("kind") == "research_repair" for message in lead_messages)


def test_extractor_extracts_informative_quote_from_cleaned_pep_html(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="Python 3.14 release date", budget={})
    html = """
    <html>
      <head><title>PEP 745 - Python 3.14 Release Schedule</title><style>.nav{display:none}</style></head>
      <body>
        <nav>Docs navigation download search</nav>
        <main>
          <p>Welcome to the Python release notes index.</p>
          <p>Python 3.14.0 final is scheduled for release on October 7, 2025, according to the official release schedule.</p>
          <p>This PEP tracks alpha, beta, release candidate, and final release dates for Python 3.14.</p>
        </main>
        <script>window.cookieBanner = true</script>
      </body>
    </html>
    """
    cleaned = _clean_html(html)
    raw = artifact_store.write_raw_document(
        run_state.run_id,
        source_task_id=None,
        document={
            "url": "https://peps.python.org/pep-0745/",
            "title": cleaned["title"],
            "text": cleaned["text"],
            "html": html,
        },
        summary="PEP 745 HTML",
    )
    task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="pending",
        owner_role="extractor",
        inputs={"artifact_id": raw.artifact_id},
        created_by="test",
    )

    result = ExtractorWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    evidence_rows = store.list_swarm_evidence(run_state.run_id)

    assert result is not None
    assert len(evidence_rows) == 1
    assert "3.14" in evidence_rows[0].quote.lower()
    assert "release" in evidence_rows[0].quote.lower()
    assert "october 7, 2025" in evidence_rows[0].quote.lower()


def test_extractor_rejects_verification_html(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="Reddit verification", budget={})
    html = """
    <html>
      <head><title>Reddit - Dive into anything</title></head>
      <body>Please wait for verification. Enable JavaScript to continue.</body>
    </html>
    """
    cleaned = _clean_html(html)
    raw = artifact_store.write_raw_document(
        run_state.run_id,
        source_task_id=None,
        document={
            "url": "https://www.reddit.com/r/Python/",
            "title": cleaned["title"],
            "text": cleaned["text"],
            "html": html,
        },
        summary="Verification page",
    )
    task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="pending",
        owner_role="extractor",
        inputs={"artifact_id": raw.artifact_id},
        created_by="test",
    )

    result = ExtractorWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    lead_messages = mailbox.inbox(run_state.run_id, role="lead")

    assert result is not None
    assert result.created_evidence_ids == []
    assert store.list_swarm_evidence(run_state.run_id) == []
    assert any(message.type == "request" and message.payload.get("kind") == "research_repair" for message in lead_messages)


def test_extractor_prefers_release_schedule_list_item(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="Python 3.14 release date", budget={})
    raw = artifact_store.write_raw_document(
        run_state.run_id,
        source_task_id=None,
        document={
            "url": "https://peps.python.org/pep-0745/",
            "title": "PEP 745 - Python 3.14 Release Schedule",
            "text": (
                "PEP 745 - Python 3.14 Release Schedule Release schedule 3.14.0 schedule "
                "Actual: 3.14.0 alpha 1: Tuesday, 2024-10-15 "
                "3.14.0 candidate 3: Thursday, 2025-09-18 "
                "3.14.0 final: Tuesday, 2025-10-07 Bugfix releases Actual: "
                "3.14.1: Tuesday, 2025-12-02"
            ),
        },
        summary="PEP schedule text",
    )
    task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="pending",
        owner_role="extractor",
        inputs={"artifact_id": raw.artifact_id},
        created_by="test",
    )

    result = ExtractorWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    evidence_rows = store.list_swarm_evidence(run_state.run_id)

    assert result is not None
    assert len(evidence_rows) == 1
    assert "3.14.0 final" in evidence_rows[0].quote
    assert "2025-10-07" in evidence_rows[0].quote


def test_repair_autonomy_reaches_second_round_without_operator(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INSIGHTSWARM_SCRIPTED_FIXTURE", "partial_missing_evidence")
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="DeepSeek strategy", budget={})
    task_store.create(
        run_state.run_id,
        kind="research_subquestion",
        status="pending",
        owner_role="sub_researcher",
        inputs={"question": "DeepSeek strategy"},
        created_by="lead",
    )

    assert SubResearcherWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id) is not None
    assert ExtractorWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id) is not None
    first_critic = CriticWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    assert first_critic is not None

    first_repair_tasks = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.kind == "repair_request"
    ]
    assert len(first_repair_tasks) == 1

    assert LeadWorker(task_store, mailbox).run_once(run_state.run_id) is not None
    repair_tasks = [
        task
        for task in store.list_swarm_tasks(run_state.run_id)
        if task.owner_role == "sub_researcher" and task.kind == "research_repair"
    ]
    assert len(repair_tasks) == 1

    assert SubResearcherWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id) is not None
    assert ExtractorWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id) is not None
    second_critic = CriticWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    evidence_rows = store.list_swarm_evidence(run_state.run_id)

    assert second_critic is not None
    assert len(evidence_rows) == 2
    assert all(row.source_url.startswith("https://example.com/deepseek") for row in evidence_rows)


def test_writer_without_evidence_outputs_blocked_report(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="No evidence report",
        budget={},
        phase="delivery",
        delivery_gate=True,
    )
    task_store.create(
        run_state.run_id,
        kind="delivery_request",
        status="pending",
        owner_role="writer",
        inputs={"question": run_state.objective, "evidence_ids": [], "report_kind": "report"},
        created_by="test",
    )

    result = WriterWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    blocked_reports = [
        artifact for artifact in store.list_swarm_artifacts(run_state.run_id) if artifact.type == "report_blocked"
    ]

    assert result is not None
    assert len(blocked_reports) == 1


def test_worker_static_boundaries() -> None:
    objective_runtime = (REPO_ROOT / "insightswarm" / "objective_runtime.py").read_text(encoding="utf-8")
    lead = (REPO_ROOT / "insightswarm" / "agents" / "lead.py").read_text(encoding="utf-8")
    writer = (REPO_ROOT / "insightswarm" / "agents" / "writer.py").read_text(encoding="utf-8")
    sub_researcher = (REPO_ROOT / "insightswarm" / "agents" / "sub_researcher.py").read_text(encoding="utf-8")
    critic = (REPO_ROOT / "insightswarm" / "agents" / "critic.py").read_text(encoding="utf-8")

    assert "SearchTool" not in objective_runtime
    assert "FetchUrlTool" not in objective_runtime
    assert "SearchTool" not in lead
    assert "FetchUrlTool" not in lead
    assert "write_raw_document" not in writer
    assert "read_payload" not in writer
    assert "write_citation" not in sub_researcher
    assert "create_evidence" not in sub_researcher
    assert "SearchTool" not in critic
    assert "FetchUrlTool" not in critic
