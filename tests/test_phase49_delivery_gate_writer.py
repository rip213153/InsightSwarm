from __future__ import annotations

from pathlib import Path

from insightswarm.agents.writer import WriterWorker
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.delivery_gate import synchronize_delivery_gate
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
        summary="Citation for delivery gate test",
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


def test_delivery_gate_closed_does_not_create_writer_task(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(
        objective="Gate closed test",
        budget={"max_steps": 12},
        phase="research",
    )
    _seed_ready_evidence(store, run_state.run_id)
    task_store.create(
        run_state.run_id,
        kind="repair_request",
        status="pending",
        owner_role="lead",
        inputs={"targeted_query": "补证"},
        priority=10,
        created_by="critic",
    )
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        broadcast=True,
        message_type="response",
        payload={"kind": "pass", "verdict": "pass", "issue_key": "issue.pass_gate_closed"},
    )

    decision = synchronize_delivery_gate(store, run_state.run_id)
    writer_tasks = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role == "writer"
    ]
    refreshed = store.get_swarm_run_state(run_state.run_id)

    assert decision.status == "closed"
    assert refreshed.delivery_gate is False
    assert writer_tasks == []


def test_delivery_gate_open_creates_writer_task_and_writer_outputs_report(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    mailbox = Mailbox(store)
    task_store = TaskStore(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Gate open test",
        budget={"max_steps": 12},
        phase="research",
    )
    _seed_ready_evidence(store, run_state.run_id)
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        broadcast=True,
        message_type="response",
        payload={"kind": "pass", "verdict": "pass", "issue_key": "issue.pass_gate_open"},
    )

    decision = synchronize_delivery_gate(store, run_state.run_id)
    writer_tasks = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role == "writer"
    ]
    refreshed = store.get_swarm_run_state(run_state.run_id)
    result = WriterWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    report_artifacts = [
        artifact for artifact in store.list_swarm_artifacts(run_state.run_id) if artifact.type == "report"
    ]

    assert decision.status == "open"
    assert refreshed.delivery_gate is True
    assert len(writer_tasks) == 1
    assert writer_tasks[0].kind == "delivery_request"
    assert result is not None
    assert len(report_artifacts) == 1
    assert Path(report_artifacts[0].payload_ref).read_text(encoding="utf-8")


def test_delivery_gate_stays_closed_while_critic_review_is_pending(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    run_state = store.create_swarm_run_state(
        objective="Gate waits for critic",
        budget={"max_steps": 12},
        phase="research",
    )
    evidence_id = _seed_ready_evidence(store, run_state.run_id)
    task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="pending",
        owner_role="critic",
        inputs={"question": run_state.objective, "evidence_ids": [evidence_id]},
        created_by="extractor",
    )

    decision = synchronize_delivery_gate(store, run_state.run_id)
    writer_tasks = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role == "writer"
    ]

    assert decision.status == "closed"
    assert "critic review is pending" in decision.reasons
    assert writer_tasks == []


def test_delivery_gate_blocked_by_authorization_request(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(
        objective="Gate blocked test",
        budget={"max_steps": 12},
        phase="research",
    )
    _seed_ready_evidence(store, run_state.run_id)
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        broadcast=True,
        message_type="response",
        payload={"kind": "pass", "verdict": "pass", "issue_key": "issue.pass_gate_blocked"},
    )
    mailbox.send(
        run_state.run_id,
        from_role="browser_agent",
        to_role="lead",
        message_type="observation",
        payload={
            "kind": "authorization_request",
            "goal": "download gated report",
            "issue_key": "issue.authorization_gate",
        },
    )

    decision = synchronize_delivery_gate(store, run_state.run_id)
    writer_tasks = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role == "writer"
    ]
    refreshed = store.get_swarm_run_state(run_state.run_id)

    assert decision.status == "blocked"
    assert refreshed.delivery_gate is False
    assert writer_tasks == []
