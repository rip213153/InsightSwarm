from __future__ import annotations

from pathlib import Path

import pytest

from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def test_phase49_shared_objects_round_trip(tmp_path: Path) -> None:
    store = _build_store(tmp_path)

    run_state = store.create_swarm_run_state(
        objective="验证 Phase 49 共享对象 schema",
        budget={"max_steps": 12},
        phase="discovery",
    )
    lead_task = store.create_swarm_task(
        run_state.run_id,
        kind="research_question",
        status="pending",
        owner_role="lead",
        inputs={"question": "DeepSeek 下步战略"},
        priority=10,
        created_by="run_bootstrap",
    )
    message = store.create_swarm_message(
        run_state.run_id,
        from_role="lead",
        to_role="researcher",
        message_type="request",
        payload={"kind": "research_subquestion", "task_id": lead_task.task_id},
        related_task_id=lead_task.task_id,
    )
    artifact = store.create_swarm_artifact(
        run_state.run_id,
        type="raw_document",
        status="ready",
        source_task_id=lead_task.task_id,
        payload_ref="artifacts/raw-doc-1.json",
        summary="Browser raw source",
    )
    evidence = store.create_swarm_evidence(
        run_state.run_id,
        artifact_id=artifact.artifact_id,
        source_url="https://example.com/deepseek",
        quote="DeepSeek is accelerating international expansion.",
        freshness="2026-05-01",
        confidence=0.82,
        qa_state="pending",
    )

    updated_state = store.update_swarm_run_state(
        run_state.run_id,
        phase="research",
        delivery_gate=True,
    )

    assert updated_state.phase == "research"
    assert updated_state.delivery_gate is True

    tasks = store.list_swarm_tasks(run_state.run_id, owner_role="lead")
    messages = store.list_swarm_messages(run_state.run_id, to_role="researcher")
    artifacts = store.list_swarm_artifacts(run_state.run_id, source_task_id=lead_task.task_id)
    evidence_rows = store.list_swarm_evidence(run_state.run_id, artifact_id=artifact.artifact_id)

    assert len(tasks) == 1
    assert tasks[0].inputs["question"] == "DeepSeek 下步战略"
    assert len(messages) == 1
    assert messages[0].type == "request"
    assert messages[0].payload["kind"] == "research_subquestion"
    assert len(artifacts) == 1
    assert artifacts[0].type == "raw_document"
    assert len(evidence_rows) == 1
    assert evidence_rows[0].quote.startswith("DeepSeek")
    assert evidence.evidence_id == evidence_rows[0].evidence_id


def test_phase49_task_rejects_more_than_three_dependencies(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    run_state = store.create_swarm_run_state(
        objective="依赖约束测试",
        budget={"max_steps": 3},
    )

    with pytest.raises(ValueError, match="depends_on exceeds"):
        store.create_swarm_task(
            run_state.run_id,
            kind="delivery_request",
            status="pending",
            owner_role="writer",
            depends_on=["task-a", "task-b", "task-c", "task-d"],
        )
