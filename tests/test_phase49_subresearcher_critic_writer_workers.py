from __future__ import annotations

from pathlib import Path
import threading
import time

from insightswarm.agents.critic import CriticWorker
from insightswarm.agents.extractor import ExtractorWorker
from insightswarm.agents.sub_researcher import SubResearcherWorker
from insightswarm.agents.writer import WriterWorker
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.swarm_store import ArtifactStore, Mailbox, TaskStore


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def _seed_ready_evidence(store: Store, run_id: str) -> str:
    artifact_store = ArtifactStore(store)
    citation_artifact = artifact_store.write_citation(
        run_id,
        source_task_id=None,
        citation={
            "source_url": "https://example.com/deepseek",
            "quote": "DeepSeek confirmed its public roadmap.",
            "text": "DeepSeek confirmed its public roadmap.",
        },
        summary="Seeded citation",
    )
    evidence = artifact_store.create_evidence(
        run_id,
        artifact_id=citation_artifact.artifact_id,
        source_url="https://example.com/deepseek",
        quote="DeepSeek confirmed its public roadmap.",
        freshness="2026-05-26",
        confidence=0.95,
        qa_state="ready",
    )
    return evidence.evidence_id


def test_subresearcher_extractor_critic_run_forever_pipeline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("INSIGHTSWARM_SCRIPTED_FIXTURE", "deliver_minimal")
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Shared-store worker pipeline",
        budget={"max_steps": 12},
        phase="research",
    )
    task_store.create(
        run_state.run_id,
        kind="research_subquestion",
        status="pending",
        owner_role="sub_researcher",
        inputs={"question": "DeepSeek public roadmap"},
        created_by="lead",
    )

    stop_event = threading.Event()
    sub_thread = threading.Thread(
        target=lambda: SubResearcherWorker(task_store, mailbox, artifact_store).run_forever(
            run_state.run_id,
            stop_event,
            poll_interval=0.01,
        ),
        daemon=True,
    )
    extractor_thread = threading.Thread(
        target=lambda: ExtractorWorker(task_store, mailbox, artifact_store).run_forever(
            run_state.run_id,
            stop_event,
            poll_interval=0.01,
        ),
        daemon=True,
    )
    critic_thread = threading.Thread(
        target=lambda: CriticWorker(task_store, mailbox, artifact_store).run_forever(
            run_state.run_id,
            stop_event,
            poll_interval=0.01,
        ),
        daemon=True,
    )
    sub_thread.start()
    extractor_thread.start()
    critic_thread.start()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        critic_messages = [
            message
            for message in mailbox.broadcasts(run_state.run_id)
            if message.from_role == "critic" and "verdict" in message.payload
        ]
        if critic_messages:
            break
        time.sleep(0.01)

    stop_event.set()
    sub_thread.join(timeout=5.0)
    extractor_thread.join(timeout=5.0)
    critic_thread.join(timeout=5.0)

    artifacts = store.list_swarm_artifacts(run_state.run_id)
    evidence_rows = store.list_swarm_evidence(run_state.run_id)
    critic_messages = [
        message
        for message in mailbox.broadcasts(run_state.run_id)
        if message.from_role == "critic" and "verdict" in message.payload
    ]

    assert not sub_thread.is_alive()
    assert not extractor_thread.is_alive()
    assert not critic_thread.is_alive()
    assert any(artifact.type == "raw_document" for artifact in artifacts)
    assert any(artifact.type == "citation" for artifact in artifacts)
    assert len(evidence_rows) == 1
    assert critic_messages
    assert critic_messages[-1].payload["verdict"] in {"pass", "repair", "block"}


def test_writer_run_forever_consumes_delivery_request_and_writes_report(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Writer loop",
        budget={"max_steps": 12},
        phase="delivery",
        delivery_gate=True,
    )
    evidence_id = _seed_ready_evidence(store, run_state.run_id)
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        broadcast=True,
        message_type="response",
        payload={"kind": "pass", "verdict": "pass", "issue_key": "issue.writer_pass"},
    )
    task_store.create(
        run_state.run_id,
        kind="delivery_request",
        status="pending",
        owner_role="writer",
        inputs={
            "question": run_state.objective,
            "evidence_ids": [evidence_id],
            "report_kind": "report",
        },
        created_by="delivery_gate",
    )

    stop_event = threading.Event()
    writer_thread = threading.Thread(
        target=lambda: WriterWorker(task_store, mailbox, artifact_store).run_forever(
            run_state.run_id,
            stop_event,
            poll_interval=0.01,
        ),
        daemon=True,
    )
    writer_thread.start()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        reports = [artifact for artifact in store.list_swarm_artifacts(run_state.run_id) if artifact.type == "report"]
        if reports:
            break
        time.sleep(0.01)

    stop_event.set()
    writer_thread.join(timeout=5.0)

    reports = [artifact for artifact in store.list_swarm_artifacts(run_state.run_id) if artifact.type == "report"]

    assert not writer_thread.is_alive()
    assert len(reports) == 1
    assert Path(reports[0].payload_ref).read_text(encoding="utf-8")
