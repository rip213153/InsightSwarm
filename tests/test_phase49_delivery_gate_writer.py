from __future__ import annotations

from pathlib import Path

from insightswarm.agents.writer import WriterWorker
from insightswarm.authorization_flow import write_authorization_decision
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.delivery_gate import synchronize_delivery_gate
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore


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


def _seed_issue_evidence(store: Store, run_id: str, issue_key: str) -> str:
    artifact_store = ArtifactStore(store)
    task_store = TaskStore(store)
    board_store = BoardStore(store)
    extractor_task = task_store.create(
        run_id,
        kind="raw_document",
        status="done",
        owner_role="extractor",
        inputs={"issue_key": issue_key},
        created_by="researcher",
    )
    citation_artifact = artifact_store.write_citation(
        run_id,
        source_task_id=extractor_task.task_id,
        citation={
            "source_url": "https://example.com/repair",
            "quote": "Independent source confirms the repaired claim.",
            "text": "Independent source confirms the repaired claim.",
            "issue_key": issue_key,
        },
        summary="Repair citation",
    )
    evidence = artifact_store.create_evidence(
        run_id,
        artifact_id=citation_artifact.artifact_id,
        source_url="https://example.com/repair",
        quote="Independent source confirms the repaired claim.",
        freshness="2026-05-26",
        confidence=0.9,
        qa_state="ready",
    )
    board_store.record_evidence(
        run_id,
        evidence=evidence,
        question_id=None,
        artifact_id=citation_artifact.artifact_id,
        source_task_id=extractor_task.task_id,
        issue_key=issue_key,
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


def test_delivery_gate_reconciles_passed_conflict_from_evidence_lineage(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    mailbox = Mailbox(store)
    board_store = BoardStore(store)
    issue_key = "issue.repair_lineage"
    run_state = store.create_swarm_run_state(
        objective="Gate reconciles repaired conflict",
        budget={"max_steps": 12},
        phase="research",
    )
    evidence_id = _seed_issue_evidence(store, run_state.run_id, issue_key)
    review_task = TaskStore(store).create(
        run_state.run_id,
        kind="evidence_review",
        status="done",
        owner_role="critic",
        inputs={"evidence_ids": [evidence_id]},
        created_by="extractor",
    )
    conflict = board_store.create_conflict(
        run_state.run_id,
        title="Original critic conflict",
        question_id=None,
        payload={"issue_key": issue_key, "evidence_ids": ["old_evidence"], "must_fix": ["add independent source"]},
        created_by="critic",
    )
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        broadcast=True,
        message_type="response",
        payload={"kind": "pass", "verdict": "pass", "evidence_ids": [evidence_id], "reason": "Repair evidence satisfied the issue."},
        related_task_id=review_task.task_id,
    )

    decision = synchronize_delivery_gate(store, run_state.run_id)
    refreshed_conflict = store.get_swarm_board_item(conflict.item_id)

    assert refreshed_conflict.status == "resolved"
    assert decision.status == "open"


def test_delivery_gate_does_not_reconcile_conflict_created_after_pass(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    mailbox = Mailbox(store)
    board_store = BoardStore(store)
    issue_key = "issue.newer_conflict"
    run_state = store.create_swarm_run_state(
        objective="Gate keeps newer conflict open",
        budget={"max_steps": 12},
        phase="research",
    )
    evidence_id = _seed_issue_evidence(store, run_state.run_id, issue_key)
    review_task = TaskStore(store).create(
        run_state.run_id,
        kind="evidence_review",
        status="done",
        owner_role="critic",
        inputs={"evidence_ids": [evidence_id]},
        created_by="extractor",
    )
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        broadcast=True,
        message_type="response",
        payload={"kind": "pass", "verdict": "pass", "evidence_ids": [evidence_id], "reason": "Earlier pass."},
        related_task_id=review_task.task_id,
    )
    conflict = board_store.create_conflict(
        run_state.run_id,
        title="New critic conflict",
        question_id=None,
        payload={"issue_key": issue_key, "evidence_ids": [evidence_id], "must_fix": ["new issue after pass"]},
        created_by="critic",
    )

    decision = synchronize_delivery_gate(store, run_state.run_id)
    refreshed_conflict = store.get_swarm_board_item(conflict.item_id)

    assert refreshed_conflict.status == "open"
    assert decision.status == "closed"
    assert "board conflicts are still open" in decision.reasons


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
        artifact for artifact in store.list_swarm_artifacts(run_state.run_id) if artifact.type == "report_partial"
    ]

    assert decision.status == "open"
    assert refreshed.delivery_gate is True
    assert len(writer_tasks) == 1
    assert writer_tasks[0].kind == "delivery_request"
    assert result is not None
    assert len(report_artifacts) == 1
    body = Path(report_artifacts[0].payload_ref).read_text(encoding="utf-8")
    assert "fallback report generated because WriterAgent did not publish" in body


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


def test_delivery_gate_uses_latest_pass_frontier_when_newer_review_is_pending(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    mailbox = Mailbox(store)
    task_store = TaskStore(store)
    run_state = store.create_swarm_run_state(
        objective="Gate accepts passed frontier",
        budget={"max_steps": 12},
        phase="research",
    )
    passed_evidence_id = _seed_ready_evidence(store, run_state.run_id)
    newer_evidence_id = _seed_ready_evidence(store, run_state.run_id)
    passed_review = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="done",
        owner_role="critic",
        inputs={"question": run_state.objective, "evidence_ids": [passed_evidence_id]},
        created_by="evidence_review_gate",
    )
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        broadcast=True,
        message_type="response",
        payload={
            "kind": "pass",
            "verdict": "pass",
            "evidence_ids": [passed_evidence_id],
            "reason": "The accepted evidence frontier is sufficient for delivery.",
        },
        related_task_id=passed_review.task_id,
    )
    task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="pending",
        owner_role="critic",
        inputs={"question": run_state.objective, "evidence_ids": [passed_evidence_id, newer_evidence_id]},
        created_by="evidence_review_gate",
    )

    decision = synchronize_delivery_gate(store, run_state.run_id)
    writer_tasks = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role == "writer"
    ]

    assert decision.status == "open"
    assert "critic review is pending" not in decision.reasons
    assert decision.evidence_ids == [passed_evidence_id]
    assert writer_tasks[0].inputs["evidence_ids"] == [passed_evidence_id]


def test_delivery_gate_replaces_stale_writer_frontier(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    mailbox = Mailbox(store)
    task_store = TaskStore(store)
    run_state = store.create_swarm_run_state(
        objective="Gate refreshes stale writer frontier",
        budget={"max_steps": 12},
        phase="research",
    )
    first_evidence_id = _seed_ready_evidence(store, run_state.run_id)
    first_review = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="done",
        owner_role="critic",
        inputs={"question": run_state.objective, "evidence_ids": [first_evidence_id]},
        created_by="evidence_review_gate",
    )
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        broadcast=True,
        message_type="response",
        payload={"kind": "pass", "verdict": "pass", "evidence_ids": [first_evidence_id]},
        related_task_id=first_review.task_id,
    )
    first_decision = synchronize_delivery_gate(store, run_state.run_id)
    first_writer_task = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role == "writer"
    ][0]

    second_evidence_id = _seed_ready_evidence(store, run_state.run_id)
    second_review = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="done",
        owner_role="critic",
        inputs={"question": run_state.objective, "evidence_ids": [first_evidence_id, second_evidence_id]},
        created_by="evidence_review_gate",
    )
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        broadcast=True,
        message_type="response",
        payload={"kind": "pass", "verdict": "pass", "evidence_ids": [first_evidence_id, second_evidence_id]},
        related_task_id=second_review.task_id,
    )

    second_decision = synchronize_delivery_gate(store, run_state.run_id)
    writer_tasks = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role == "writer"
    ]
    active_writer_tasks = [task for task in writer_tasks if task.status in {"pending", "leased"}]
    stale_task = store.get_swarm_task(first_writer_task.task_id or "")

    assert first_decision.frontier_hash
    assert second_decision.frontier_hash != first_decision.frontier_hash
    assert stale_task.status == "blocked"
    assert len(active_writer_tasks) == 1
    assert active_writer_tasks[0].inputs["evidence_ids"] == [first_evidence_id, second_evidence_id]
    assert active_writer_tasks[0].inputs["frontier_hash"] == second_decision.frontier_hash


def test_delivery_gate_pass_frontier_still_waits_for_material_research_tasks(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    mailbox = Mailbox(store)
    task_store = TaskStore(store)
    run_state = store.create_swarm_run_state(
        objective="Gate waits for material producer",
        budget={"max_steps": 12},
        phase="research",
    )
    evidence_id = _seed_ready_evidence(store, run_state.run_id)
    review = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="done",
        owner_role="critic",
        inputs={"question": run_state.objective, "evidence_ids": [evidence_id]},
        created_by="evidence_review_gate",
    )
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        broadcast=True,
        message_type="response",
        payload={"kind": "pass", "verdict": "pass", "evidence_ids": [evidence_id]},
        related_task_id=review.task_id,
    )
    task_store.create(
        run_state.run_id,
        kind="research_repair",
        status="pending",
        owner_role="researcher",
        inputs={"question": "Find one more primary source."},
        priority=10,
        created_by="critic",
    )
    task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="pending",
        owner_role="critic",
        inputs={"question": run_state.objective, "evidence_ids": [evidence_id]},
        created_by="evidence_review_gate",
    )

    decision = synchronize_delivery_gate(store, run_state.run_id)

    assert decision.status == "closed"
    assert "research tasks are still running" in decision.reasons


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


def test_delivery_gate_treats_decided_authorization_as_recoverable_work(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    mailbox = Mailbox(store)
    task_store = TaskStore(store)
    run_state = store.create_swarm_run_state(
        objective="Gate authorization recovery test",
        budget={"max_steps": 12},
        phase="research",
    )
    _seed_ready_evidence(store, run_state.run_id)
    browser_task = task_store.create(
        run_state.run_id,
        kind="hard_acquisition",
        status="blocked",
        owner_role="browser_agent",
        inputs={"goal": "click public page expansion"},
        created_by="researcher",
    )
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        broadcast=True,
        message_type="response",
        payload={"kind": "pass", "verdict": "pass", "issue_key": "issue.pass_gate_authorized"},
    )
    mailbox.send(
        run_state.run_id,
        from_role="browser_agent",
        to_role="lead",
        message_type="observation",
        payload={
            "kind": "authorization_request",
            "task_id": browser_task.task_id,
            "goal": "click public page expansion",
            "reason": "non-link public click requires authorization",
        },
        related_task_id=browser_task.task_id,
    )

    write_authorization_decision(
        store,
        run_state.run_id,
        task_id=browser_task.task_id or "",
        decision="allow",
        reason="operator allowed this specific interaction",
    )
    decision = synchronize_delivery_gate(store, run_state.run_id)

    assert decision.status == "closed"
    assert "authorization_request is pending" not in decision.reasons
    assert "research tasks are still running" in decision.reasons

    task_store.complete(browser_task.task_id or "")
    decision = synchronize_delivery_gate(store, run_state.run_id)

    assert decision.status == "open"
