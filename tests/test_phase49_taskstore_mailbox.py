from __future__ import annotations

from pathlib import Path

from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.swarm_store import Mailbox, TaskStore


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def test_taskstore_claim_heartbeat_complete(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    run_state = store.create_swarm_run_state(objective="TaskStore 测试", budget={})

    created = task_store.create(
        run_state.run_id,
        kind="research_question",
        status="pending",
        owner_role="lead",
        inputs={"question": "DeepSeek"},
    )
    claimed = task_store.claim_next(run_state.run_id, owner_role="lead")
    heartbeated = task_store.heartbeat(created.task_id)
    completed = task_store.complete(created.task_id)

    assert claimed is not None
    assert claimed.task_id == created.task_id
    assert claimed.status == "leased"
    assert claimed.lease_until is not None
    assert heartbeated.status == "leased"
    assert heartbeated.lease_until is not None
    assert completed.status == "done"


def test_mailbox_send_and_inbox(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    mailbox = Mailbox(store)
    task_store = TaskStore(store)
    run_state = store.create_swarm_run_state(objective="Mailbox 测试", budget={})
    writer_task = task_store.create(
        run_state.run_id,
        kind="delivery_request",
        status="pending",
        owner_role="writer",
        inputs={"question": "Mailbox 测试"},
    )

    direct = mailbox.send(
        run_state.run_id,
        from_role="lead",
        to_role="writer",
        message_type="request",
        payload={"kind": "delivery_request", "task_id": writer_task.task_id},
        related_task_id=writer_task.task_id,
    )
    broadcast = mailbox.send(
        run_state.run_id,
        from_role="critic",
        broadcast=True,
        message_type="observation",
        payload={"kind": "conflict", "issue_key": "issue.missing_evidence", "summary": "missing evidence"},
    )

    inbox = mailbox.inbox(run_state.run_id, role="writer")
    broadcast_only = mailbox.broadcasts(run_state.run_id)

    assert direct.to_role == "writer"
    assert broadcast.broadcast is True
    assert len(inbox) == 2
    assert any(message.type == "observation" and message.payload.get("kind") == "conflict" for message in inbox)
    assert len(broadcast_only) == 1
    assert broadcast_only[0].type == "observation"
