from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from insightswarm.agents.extractor import Extractor
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.swarm_store import ArtifactStore, Mailbox, TaskStore


@dataclass
class _ModelFailure:
    status: str = "error"
    error: str = "simulated model failure"


class _FailingModel:
    def complete(self, *args, **kwargs):
        return _ModelFailure()


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def test_extractor_technical_failure_retries_without_critic_repair(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="OpenAI strategy",
        budget={"max_steps": 12},
        phase="research",
    )
    raw_artifact = artifact_store.write_raw_document(
        run_state.run_id,
        source_task_id=None,
        document={
            "source_url": "https://example.com/openai",
            "title": "OpenAI strategy",
            "text": "OpenAI is focusing on enterprise productivity.",
            "metadata": {"issue_key": "issue.extractor_failure"},
        },
        summary="OpenAI raw source",
    )
    extractor_task = task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="pending",
        owner_role="extractor",
        inputs={"artifact_id": raw_artifact.artifact_id, "issue_key": "issue.extractor_failure"},
        created_by="researcher",
    )

    result = Extractor(task_store, mailbox, artifact_store).run_once(
        run_state.run_id,
        model_client=_FailingModel(),
    )
    failure_reviews = [
        task
        for task in store.list_swarm_tasks(run_state.run_id, owner_role="critic")
        if task.kind == "extraction_failure_review"
    ]
    retry_tasks = [
        task
        for task in store.list_swarm_tasks(run_state.run_id, owner_role="extractor")
        if task.kind == "raw_document" and task.status == "pending"
    ]

    assert result.terminal_status == "model_error"
    assert store.get_swarm_task(extractor_task.task_id).status == "blocked"
    assert failure_reviews == []
    assert len(retry_tasks) == 1
    assert retry_tasks[0].inputs["artifact_id"] == raw_artifact.artifact_id
    assert retry_tasks[0].inputs["extraction_attempt"] == 2


def test_extractor_technical_failure_blocks_after_retry_budget_without_critic_repair(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="OpenAI strategy",
        budget={"max_steps": 12},
        phase="research",
    )
    raw_artifact = artifact_store.write_raw_document(
        run_state.run_id,
        source_task_id=None,
        document={
            "source_url": "https://example.com/openai",
            "title": "OpenAI strategy",
            "text": "OpenAI is focusing on enterprise productivity.",
            "metadata": {"issue_key": "issue.extractor_failure"},
        },
        summary="OpenAI raw source",
    )
    extractor_task = task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="pending",
        owner_role="extractor",
        inputs={
            "artifact_id": raw_artifact.artifact_id,
            "issue_key": "issue.extractor_failure",
            "extraction_attempt": 2,
            "max_extraction_attempts": 2,
        },
        created_by="researcher",
    )

    result = Extractor(task_store, mailbox, artifact_store).run_once(
        run_state.run_id,
        model_client=_FailingModel(),
    )
    failure_reviews = [
        task
        for task in store.list_swarm_tasks(run_state.run_id, owner_role="critic")
        if task.kind == "extraction_failure_review"
    ]
    technical_failures = [
        message
        for message in store.list_swarm_messages(run_state.run_id)
        if message.from_role == "extractor"
        and message.type == "observation"
        and message.payload.get("failure_category") == "technical"
    ]

    assert result.terminal_status == "model_error"
    assert store.get_swarm_task(extractor_task.task_id).status == "blocked"
    assert failure_reviews == []
    assert len(technical_failures) == 1
    assert technical_failures[0].payload["should_trigger_research_repair"] is False
