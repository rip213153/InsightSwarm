from __future__ import annotations

import threading
import time
from pathlib import Path

from insightswarm.agents.browser_agent import BrowserWorker
from insightswarm.agents.extractor import ExtractorWorker
from insightswarm.agents.lead import LeadWorker, bootstrap_lead_objective
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.swarm_store import ArtifactStore, Mailbox, TaskStore


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def test_lead_browser_extractor_pipeline_uses_tasks_messages_and_artifacts(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Collect browser-only source",
        budget={"max_steps": 12},
        phase="discovery",
    )

    bootstrap_lead_objective(
        task_store,
        mailbox,
        run_id=run_state.run_id,
        question="DeepSeek browser source",
        sub_questions=[],
        browser_goal="Open public DeepSeek news page",
    )
    LeadWorker(task_store, mailbox).run_once(run_state.run_id)

    browser_result = BrowserWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    raw_artifacts = [
        artifact for artifact in store.list_swarm_artifacts(run_state.run_id) if artifact.type == "raw_document"
    ]
    extractor_tasks = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role == "extractor"
    ]
    extractor_messages = [
        message
        for message in mailbox.inbox(run_state.run_id, role="extractor")
        if message.type == "request" and message.payload.get("kind") == "extract_evidence"
    ]

    assert browser_result is not None
    assert len(raw_artifacts) == 1
    assert len(extractor_tasks) == 1
    assert len(extractor_messages) == 1

    extractor_result = ExtractorWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    citation_artifacts = [
        artifact for artifact in store.list_swarm_artifacts(run_state.run_id) if artifact.type == "citation"
    ]
    evidence_rows = store.list_swarm_evidence(run_state.run_id)

    assert extractor_result is not None
    assert len(citation_artifacts) == 1
    assert len(evidence_rows) == 1
    assert evidence_rows[0].quote.startswith("Raw browser capture")


def test_browser_worker_emits_authorization_request_for_high_risk_goal(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="High risk browser action",
        budget={"max_steps": 12},
        phase="research",
    )
    task_store.create(
        run_state.run_id,
        kind="hard_acquisition",
        status="pending",
        owner_role="browser_agent",
        inputs={"goal": "Log in and download the private report"},
        created_by="lead",
    )

    result = BrowserWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)

    authorization_messages = [
        message
        for message in mailbox.inbox(run_state.run_id, role="lead")
        if message.type == "observation" and message.payload.get("kind") == "authorization_request"
    ]
    raw_artifacts = [
        artifact for artifact in store.list_swarm_artifacts(run_state.run_id) if artifact.type == "raw_document"
    ]
    browser_tasks = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role == "browser_agent"
    ]

    assert result is not None
    assert len(authorization_messages) == 1
    assert raw_artifacts == []
    assert browser_tasks[0].status == "blocked"


def test_browser_and_extractor_run_forever_consume_pipeline_and_exit(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Forever browser extractor loop",
        budget={"max_steps": 12},
        phase="discovery",
    )

    bootstrap_lead_objective(
        task_store,
        mailbox,
        run_id=run_state.run_id,
        question="DeepSeek browser source",
        sub_questions=[],
        browser_goal="Open public DeepSeek news page",
    )
    LeadWorker(task_store, mailbox).run_once(run_state.run_id)

    stop_event = threading.Event()
    browser_results: dict[str, object] = {}
    extractor_results: dict[str, object] = {}

    def _browser_loop() -> None:
        browser_results["results"] = BrowserWorker(task_store, mailbox, artifact_store).run_forever(
            run_state.run_id,
            stop_event,
            poll_interval=0.01,
        )

    def _extractor_loop() -> None:
        extractor_results["results"] = ExtractorWorker(task_store, mailbox, artifact_store).run_forever(
            run_state.run_id,
            stop_event,
            poll_interval=0.01,
        )

    browser_thread = threading.Thread(target=_browser_loop, daemon=True)
    extractor_thread = threading.Thread(target=_extractor_loop, daemon=True)
    browser_thread.start()
    extractor_thread.start()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if store.list_swarm_evidence(run_state.run_id):
            break
        time.sleep(0.01)

    stop_event.set()
    browser_thread.join(timeout=5.0)
    extractor_thread.join(timeout=5.0)

    assert not browser_thread.is_alive()
    assert not extractor_thread.is_alive()
    assert len(browser_results["results"]) >= 1
    assert len(extractor_results["results"]) >= 1
    assert len(store.list_swarm_evidence(run_state.run_id)) == 1
