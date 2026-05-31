from __future__ import annotations

from pathlib import Path

from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.extraction_batches import (
    create_extraction_batch,
    synchronize_extraction_batches,
    synchronize_run_evidence_review,
)
from insightswarm.swarm_store import ArtifactStore, BoardStore, TaskStore


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def _record_task_evidence(store: Store, run_id: str, task_id: str, quote: str) -> str:
    artifact_store = ArtifactStore(store)
    board_store = BoardStore(store)
    citation = artifact_store.write_citation(
        run_id,
        source_task_id=task_id,
        citation={"source_url": "https://example.com/source", "quote": quote, "text": quote},
        summary=quote,
    )
    evidence = artifact_store.create_evidence(
        run_id,
        artifact_id=citation.artifact_id,
        source_url="https://example.com/source",
        quote=quote,
        freshness=None,
        confidence=0.9,
        qa_state="ready",
    )
    board_store.record_evidence(
        run_id,
        evidence=evidence,
        question_id=None,
        artifact_id=citation.artifact_id,
        source_task_id=task_id,
    )
    return evidence.evidence_id or ""


def test_batch_sync_marks_ready_without_creating_batch_review(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="Batch review waits for the full extraction batch",
        budget={"max_steps": 12},
        phase="research",
    )
    first = task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="done",
        owner_role="extractor",
        inputs={"batch_id": "batch_test"},
        created_by="researcher",
    )
    second = task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="pending",
        owner_role="extractor",
        inputs={"batch_id": "batch_test"},
        created_by="researcher",
    )
    first_evidence_id = _record_task_evidence(store, run_state.run_id, first.task_id or "", "First quote.")
    create_extraction_batch(
        board_store=board_store,
        run_id=run_state.run_id,
        batch_id="batch_test",
        source_task_id="researcher_task",
        raw_artifact_ids=["raw_1", "raw_2"],
        extractor_task_ids=[first.task_id or "", second.task_id or ""],
        purpose="test batch",
        issue_key="",
        priority=9,
    )

    created = synchronize_extraction_batches(store, run_state.run_id)

    assert created == 0
    assert not [task for task in store.list_swarm_tasks(run_state.run_id, owner_role="critic") if task.kind == "evidence_review"]

    task_store.complete(second.task_id or "")
    second_evidence_id = _record_task_evidence(store, run_state.run_id, second.task_id or "", "Second quote.")
    created = synchronize_extraction_batches(store, run_state.run_id)
    batch = [
        item
        for item in store.list_swarm_board_items(run_state.run_id, kind="plan")
        if item.payload.get("batch_id") == "batch_test"
    ][0]

    assert created == 1
    assert batch.status == "ready_for_review"
    assert not [task for task in store.list_swarm_tasks(run_state.run_id, owner_role="critic") if task.kind == "evidence_review"]
    assert sorted(
        item.evidence_id
        for item in store.list_swarm_board_items(run_state.run_id, kind="evidence")
        if item.evidence_id
    ) == sorted([first_evidence_id, second_evidence_id])


def test_run_level_review_aggregates_ready_batches(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="Run-level review aggregates current evidence",
        budget={"max_steps": 12},
        phase="research",
    )
    first = task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="done",
        owner_role="extractor",
        inputs={"batch_id": "batch_one"},
        created_by="researcher",
    )
    second = task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="done",
        owner_role="extractor",
        inputs={"batch_id": "batch_two"},
        created_by="researcher",
    )
    first_evidence_id = _record_task_evidence(store, run_state.run_id, first.task_id or "", "First quote.")
    second_evidence_id = _record_task_evidence(store, run_state.run_id, second.task_id or "", "Second quote.")
    create_extraction_batch(
        board_store=board_store,
        run_id=run_state.run_id,
        batch_id="batch_one",
        source_task_id="researcher_task_1",
        raw_artifact_ids=["raw_1"],
        extractor_task_ids=[first.task_id or ""],
        purpose="test batch one",
        issue_key="",
        priority=9,
    )
    create_extraction_batch(
        board_store=board_store,
        run_id=run_state.run_id,
        batch_id="batch_two",
        source_task_id="researcher_task_2",
        raw_artifact_ids=["raw_2"],
        extractor_task_ids=[second.task_id or ""],
        purpose="test batch two",
        issue_key="",
        priority=8,
    )

    assert synchronize_extraction_batches(store, run_state.run_id) == 2
    created = synchronize_run_evidence_review(store, run_state.run_id)
    review_tasks = [task for task in store.list_swarm_tasks(run_state.run_id, owner_role="critic") if task.kind == "evidence_review"]

    assert created == 1
    assert len(review_tasks) == 1
    assert review_tasks[0].inputs["evidence_scope"] == "run"
    assert sorted(review_tasks[0].inputs["batch_ids"]) == ["batch_one", "batch_two"]
    assert sorted(review_tasks[0].inputs["evidence_ids"]) == sorted([first_evidence_id, second_evidence_id])
    assert "batch_id" not in review_tasks[0].inputs


def test_run_level_review_dedupes_same_evidence_set(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="Run-level review dedupes same evidence bundle",
        budget={"max_steps": 12},
        phase="research",
    )
    task = task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="done",
        owner_role="extractor",
        inputs={"batch_id": "batch_test"},
        created_by="researcher",
    )
    _record_task_evidence(store, run_state.run_id, task.task_id or "", "Quote.")
    create_extraction_batch(
        board_store=board_store,
        run_id=run_state.run_id,
        batch_id="batch_test",
        source_task_id="researcher_task",
        raw_artifact_ids=["raw_1"],
        extractor_task_ids=[task.task_id or ""],
        purpose="test batch",
        issue_key="",
        priority=9,
    )

    synchronize_extraction_batches(store, run_state.run_id)
    assert synchronize_run_evidence_review(store, run_state.run_id) == 1
    assert synchronize_run_evidence_review(store, run_state.run_id) == 0
    review_tasks = [task for task in store.list_swarm_tasks(run_state.run_id, owner_role="critic") if task.kind == "evidence_review"]

    assert len(review_tasks) == 1
