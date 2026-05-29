from __future__ import annotations

from pathlib import Path
from threading import Barrier, Thread

from insightswarm.agents.critic import CriticWorker
from insightswarm.agents.lead import LeadWorker
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


def _seed_ready_evidence(store: Store, run_id: str, quote: str = "DeepSeek confirmed its roadmap.") -> str:
    artifact_store = ArtifactStore(store)
    citation_artifact = artifact_store.write_citation(
        run_id,
        source_task_id=None,
        citation={"source_url": "https://example.com", "quote": quote, "text": quote},
        summary="Seed citation",
    )
    evidence = artifact_store.create_evidence(
        run_id,
        artifact_id=citation_artifact.artifact_id,
        source_url="https://example.com",
        quote=quote,
        freshness="2026-05-27",
        confidence=0.9,
        qa_state="ready",
    )
    return evidence.evidence_id


def test_dependency_gating_blocks_claim_until_dependencies_done(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    run_state = store.create_swarm_run_state(objective="deps", budget={})

    upstream = task_store.create(
        run_state.run_id,
        kind="research_question",
        status="pending",
        owner_role="lead",
        inputs={"question": "A"},
    )
    downstream = task_store.create(
        run_state.run_id,
        kind="research_repair",
        status="pending",
        owner_role="sub_researcher",
        inputs={"question": "B"},
        depends_on=[upstream.task_id],
    )

    assert task_store.claim_next(run_state.run_id, owner_role="sub_researcher") is None
    task_store.complete(upstream.task_id)
    claimed = task_store.claim_next(run_state.run_id, owner_role="sub_researcher")

    assert claimed is not None
    assert claimed.task_id == downstream.task_id


def test_dependency_gating_rejects_missing_dependency(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    run_state = store.create_swarm_run_state(objective="missing deps", budget={})

    task_store.create(
        run_state.run_id,
        kind="research_repair",
        status="pending",
        owner_role="sub_researcher",
        inputs={"question": "B"},
        depends_on=["task_missing"],
    )

    assert task_store.claim_next(run_state.run_id, owner_role="sub_researcher") is None


def test_explicit_idempotency_rules_reuse_active_tasks(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    run_state = store.create_swarm_run_state(objective="idem", budget={})

    extractor_1 = task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="pending",
        owner_role="extractor",
        inputs={"artifact_id": "artifact_x"},
    )
    extractor_2 = task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="pending",
        owner_role="extractor",
        inputs={"artifact_id": "artifact_x"},
    )
    repair_1 = task_store.create(
        run_state.run_id,
        kind="repair_request",
        status="pending",
        owner_role="lead",
        inputs={"issue_key": "issue.alpha"},
    )
    repair_2 = task_store.create(
        run_state.run_id,
        kind="repair_request",
        status="pending",
        owner_role="lead",
        inputs={"issue_key": "issue.alpha"},
    )
    review_1 = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="pending",
        owner_role="critic",
        inputs={"evidence_ids": ["e1", "e2"]},
    )
    review_2 = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="pending",
        owner_role="critic",
        inputs={"evidence_ids": ["e2", "e1"]},
    )
    delivery_1 = task_store.create(
        run_state.run_id,
        kind="delivery_request",
        status="pending",
        owner_role="writer",
        inputs={"question": "q"},
    )
    delivery_2 = task_store.create(
        run_state.run_id,
        kind="delivery_request",
        status="pending",
        owner_role="writer",
        inputs={"question": "q2"},
    )

    assert extractor_1.task_id == extractor_2.task_id
    assert repair_1.task_id == repair_2.task_id
    assert review_1.task_id == review_2.task_id
    assert delivery_1.task_id == delivery_2.task_id


def test_completed_tasks_do_not_block_next_idempotent_round(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    run_state = store.create_swarm_run_state(objective="next round", budget={})

    first = task_store.create(
        run_state.run_id,
        kind="repair_request",
        status="pending",
        owner_role="lead",
        inputs={"issue_key": "issue.alpha"},
    )
    task_store.complete(first.task_id)
    second = task_store.create(
        run_state.run_id,
        kind="repair_request",
        status="pending",
        owner_role="lead",
        inputs={"issue_key": "issue.alpha"},
    )

    assert first.task_id != second.task_id


def test_claim_next_is_atomic_under_concurrency(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    run_state = store.create_swarm_run_state(objective="claim", budget={})
    task = task_store.create(
        run_state.run_id,
        kind="research_subquestion",
        status="pending",
        owner_role="sub_researcher",
        inputs={"question": "Only one worker should claim this"},
    )
    barrier = Barrier(3)
    claimed_ids: list[str | None] = [None, None]

    def _claim(index: int) -> None:
        barrier.wait()
        claimed = task_store.claim_next(run_state.run_id, owner_role="sub_researcher")
        claimed_ids[index] = None if claimed is None else claimed.task_id
        barrier.wait()

    first = Thread(target=_claim, args=(0,))
    second = Thread(target=_claim, args=(1,))
    first.start()
    second.start()
    barrier.wait()
    barrier.wait()
    first.join()
    second.join()

    winners = [task_id for task_id in claimed_ids if task_id is not None]
    current = store.get_swarm_task(task.task_id)

    assert winners == [task.task_id]
    assert claimed_ids.count(None) == 1
    assert current.status == "leased"


def test_concurrent_repair_request_create_reuses_single_active_task(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    run_state = store.create_swarm_run_state(objective="repair create", budget={})
    barrier = Barrier(3)
    task_ids: list[str | None] = [None, None]

    def _create(index: int) -> None:
        barrier.wait()
        task = task_store.create(
            run_state.run_id,
            kind="repair_request",
            status="pending",
            owner_role="lead",
            inputs={"issue_key": "issue.concurrent"},
            created_by="test",
        )
        task_ids[index] = task.task_id
        barrier.wait()

    first = Thread(target=_create, args=(0,))
    second = Thread(target=_create, args=(1,))
    first.start()
    second.start()
    barrier.wait()
    barrier.wait()
    first.join()
    second.join()

    repair_tasks = [
        task
        for task in store.list_swarm_tasks(run_state.run_id)
        if task.kind == "repair_request" and task.status in {"pending", "leased"}
    ]

    assert task_ids[0] == task_ids[1]
    assert len(repair_tasks) == 1


def test_concurrent_evidence_review_create_reuses_single_task(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    run_state = store.create_swarm_run_state(objective="review create", budget={})
    barrier = Barrier(3)
    task_ids: list[str | None] = [None, None]

    def _create(index: int) -> None:
        barrier.wait()
        task = task_store.create(
            run_state.run_id,
            kind="evidence_review",
            status="pending",
            owner_role="critic",
            inputs={"evidence_ids": ["e2", "e1"]},
            created_by="test",
        )
        task_ids[index] = task.task_id
        barrier.wait()

    first = Thread(target=_create, args=(0,))
    second = Thread(target=_create, args=(1,))
    first.start()
    second.start()
    barrier.wait()
    barrier.wait()
    first.join()
    second.join()

    review_tasks = [
        task
        for task in store.list_swarm_tasks(run_state.run_id)
        if task.kind == "evidence_review"
    ]

    assert task_ids[0] == task_ids[1]
    assert len(review_tasks) == 1


def test_repair_attempt_limit_converges_to_delivery_gap(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(objective="repair convergence", budget={})
    repair_task = task_store.create(
        run_state.run_id,
        kind="repair_request",
        status="pending",
        owner_role="lead",
        inputs={
            "targeted_query": "official confirmation",
            "issue_key": "issue.confirmation",
            "repair_attempt": 3,
            "max_repair_attempts": 2,
        },
        created_by="critic",
    )

    result = LeadWorker(task_store, mailbox).run_once(run_state.run_id)
    broadcasts = mailbox.broadcasts(run_state.run_id)
    downstream = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.task_id != repair_task.task_id
    ]

    assert result is not None
    assert downstream == []
    assert any(message.type == "observation" and message.payload.get("kind") == "delivery_gap" for message in broadcasts)


def test_critic_autonomously_scans_ready_evidence_and_reviews_it(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="critic scan", budget={})
    _seed_ready_evidence(store, run_state.run_id)

    result = CriticWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    critic_messages = [
        message for message in mailbox.broadcasts(run_state.run_id) if message.from_role == "critic"
    ]

    assert result is not None
    assert critic_messages
    assert any("verdict" in message.payload for message in critic_messages)


def test_critic_does_not_recreate_review_for_same_evidence_bundle_after_done(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="critic stable review", budget={})
    _seed_ready_evidence(store, run_state.run_id, quote="Analysts speculate without a primary source.")
    worker = CriticWorker(task_store, mailbox, artifact_store)

    first = worker.run_once(run_state.run_id)
    second = worker.run_once(run_state.run_id)
    review_tasks = [
        task
        for task in store.list_swarm_tasks(run_state.run_id)
        if task.kind == "evidence_review"
    ]
    repair_tasks = [
        task
        for task in store.list_swarm_tasks(run_state.run_id)
        if task.kind == "repair_request"
    ]

    assert first is not None
    assert second is None
    assert len(review_tasks) == 1
    assert len(repair_tasks) == 1


def test_critic_repair_contains_stable_issue_metadata(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="critic repair", budget={})
    _seed_ready_evidence(store, run_state.run_id, quote="Analysts speculate without a primary source.")

    result = CriticWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    repair_tasks = [
        task
        for task in store.list_swarm_tasks(run_state.run_id)
        if task.kind == "repair_request"
    ]

    assert result is not None
    assert len(repair_tasks) == 1
    repair = repair_tasks[0]
    assert str(repair.inputs.get("issue_key") or "").startswith("issue.")
    assert repair.inputs.get("repair_attempt") == 1
    assert repair.inputs.get("max_repair_attempts") == 2


def test_critic_reuses_active_repair_for_same_issue(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="critic duplicate repair", budget={})
    _seed_ready_evidence(store, run_state.run_id, quote="Analysts speculate without a primary source.")
    worker = CriticWorker(task_store, mailbox, artifact_store)

    first = worker.run_once(run_state.run_id)
    review_again = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="pending",
        owner_role="critic",
        inputs={
            "evidence_ids": [row.evidence_id for row in store.list_swarm_evidence(run_state.run_id, qa_state="ready")],
            "question": run_state.objective,
        },
        priority=5,
        created_by="test",
    )
    second = worker.run_once(run_state.run_id)
    repair_tasks = [
        task
        for task in store.list_swarm_tasks(run_state.run_id)
        if task.kind == "repair_request" and task.status in {"pending", "leased"}
    ]

    assert first is not None
    assert review_again.task_id
    assert second is None
    assert review_again.task_id == [
        task.task_id
        for task in store.list_swarm_tasks(run_state.run_id)
        if task.kind == "evidence_review"
    ][0]
    assert len(repair_tasks) == 1


def test_critic_stops_creating_repair_after_attempt_limit(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="critic exhaustion", budget={})
    evidence_id = _seed_ready_evidence(store, run_state.run_id, quote="Analysts speculate without a primary source.")
    existing_review = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="done",
        owner_role="critic",
        inputs={"evidence_ids": [evidence_id], "question": run_state.objective},
        created_by="test",
    )
    first_repair = task_store.create(
        run_state.run_id,
        kind="repair_request",
        status="done",
        owner_role="lead",
        inputs={
            "issue_key": "issue.seed",
            "repair_attempt": 1,
            "max_repair_attempts": 2,
        },
        created_by="critic",
    )
    second_repair = task_store.create(
        run_state.run_id,
        kind="repair_request",
        status="done",
        owner_role="lead",
        inputs={
            "issue_key": "issue.seed",
            "repair_attempt": 2,
            "max_repair_attempts": 2,
        },
        created_by="critic",
    )

    worker = CriticWorker(task_store, mailbox, artifact_store)
    issue_key = "issue.seed"
    review_task = store.get_swarm_task(
        task_store.create(
            run_state.run_id,
            kind="evidence_review",
            status="done",
            owner_role="critic",
            inputs={"evidence_ids": [evidence_id], "question": run_state.objective},
            created_by="test",
        ).task_id
    )

    original_next_repair_attempt = worker._next_repair_attempt

    def _force_attempt(_: str, requested_issue_key: str) -> int:
        if requested_issue_key == issue_key:
            return 3
        return original_next_repair_attempt(_, requested_issue_key)

    worker._next_repair_attempt = _force_attempt  # type: ignore[method-assign]
    worker._find_active_repair = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    from insightswarm.agents import critic as critic_module
    original_issue_key = critic_module._stable_issue_key
    critic_module._stable_issue_key = lambda **_kwargs: issue_key
    try:
        result = worker._create_repair_request(
            task=review_task,
            question=run_state.objective,
            verdict={
                "targeted_query": "official primary source confirmation",
                "must_fix": ["Primary-source evidence is missing or explicitly speculative."],
            },
        )
    finally:
        critic_module._stable_issue_key = original_issue_key

    active_repairs = [
        task
        for task in store.list_swarm_tasks(run_state.run_id)
        if task.kind == "repair_request" and task.status in {"pending", "leased"}
    ]
    intents = [message.payload.get("kind") for message in mailbox.broadcasts(run_state.run_id)]
    direct_to_lead = [message.payload.get("kind") for message in mailbox.inbox(run_state.run_id, role="lead")]

    assert existing_review.task_id
    assert first_repair.task_id
    assert second_repair.task_id
    assert review_task.task_id
    assert result is not None
    assert active_repairs == []
    assert "delivery_gap" in intents
    assert "repair_exhausted" in direct_to_lead


def test_lead_autonomously_responds_to_repair_message_without_manual_task(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(objective="lead repair", budget={})
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        to_role="lead",
        message_type="request",
        payload={
            "kind": "research_repair",
            "targeted_query": "official primary source confirmation",
            "owner_role": "sub_researcher",
            "issue_key": "issue.primary_source",
            "repair_attempt": 1,
            "max_repair_attempts": 2,
        },
    )

    result = LeadWorker(task_store, mailbox).run_once(run_state.run_id)
    downstream = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role == "sub_researcher"
    ]
    duplicate = LeadWorker(task_store, mailbox).run_once(run_state.run_id)
    downstream_after = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role == "sub_researcher"
    ]

    assert result is not None
    assert len(downstream) == 1
    assert duplicate is None
    assert len(downstream_after) == 1


def test_lead_deduplicates_same_issue_across_duplicate_repair_messages(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(objective="lead dedupe", budget={})
    payload = {
        "kind": "research_repair",
        "targeted_query": "official primary source confirmation",
        "owner_role": "sub_researcher",
        "issue_key": "issue.primary_source",
        "repair_attempt": 1,
        "max_repair_attempts": 2,
    }
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        to_role="lead",
        message_type="request",
        payload=payload,
    )
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        to_role="lead",
        message_type="request",
        payload=payload,
    )

    result = LeadWorker(task_store, mailbox).run_until_idle(run_state.run_id)
    lead_repairs = [
        task
        for task in store.list_swarm_tasks(run_state.run_id)
        if task.owner_role == "lead" and task.kind == "repair_request"
    ]
    downstream = [
        task
        for task in store.list_swarm_tasks(run_state.run_id)
        if task.owner_role == "sub_researcher" and task.kind == "research_repair"
    ]

    assert result.iterations == 1
    assert len(lead_repairs) == 1
    assert len(downstream) == 1


def test_writer_remains_gate_only(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(objective="writer gate", budget={})
    evidence_id = _seed_ready_evidence(store, run_state.run_id)
    mailbox.send(
        run_state.run_id,
        from_role="critic",
        broadcast=True,
        message_type="response",
        payload={"kind": "pass", "verdict": "pass", "issue_key": "issue.writer_gate"},
    )

    assert WriterWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id) is None

    synchronize_delivery_gate(store, run_state.run_id)
    result = WriterWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    reports = [artifact for artifact in store.list_swarm_artifacts(run_state.run_id) if artifact.type == "report"]

    assert result is not None
    assert len(reports) == 1
    assert evidence_id
