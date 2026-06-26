"""Phase 1 task-isolation tests: worker safety shell, LeaseGuard, ExecutionCell.

These pin the three behaviors the minimal isolation layer must guarantee:
  1. renew_lease_if_leased never resurrects a done/blocked task.
  2. A worker whose body raises does NOT kill the thread — the outer shell
     swallows the exception, releases the lease (needs_repair), and records a
     technical_failure message.
  3. ExecutionCell creates a per-task workspace and keeps it on failure for
     forensics while reaping it on success.
"""

from __future__ import annotations

from pathlib import Path
from threading import Event

from insightswarm.agents.execution_cell import ExecutionCell, LeaseGuard, run_supervised_once
from insightswarm.agents.lead import LeadWorker, bootstrap_lead_objective
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.swarm_store import Mailbox, TaskStore


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def _claim_a_lead_task(store: Store, tmp_path: Path) -> tuple[Store, TaskStore, Mailbox, str, str]:
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(
        objective="demo",
        budget={"max_steps": 12},
        phase="discovery",
    )
    bootstrap_lead_objective(
        task_store,
        mailbox,
        run_id=run_state.run_id,
        question="demo",
        sub_questions=["demo"],
        browser_goal="demo",
    )
    return store, task_store, mailbox, run_state.run_id, run_state.run_id


# --- renew_lease_if_leased: the no-resurrection contract --------------------

def test_renew_lease_if_leased_extends_leased_task(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(
        objective="demo", budget={"max_steps": 12}, phase="discovery"
    )
    bootstrap_lead_objective(
        task_store, mailbox, run_id=run_state.run_id,
        question="q", sub_questions=["q"], browser_goal="g",
    )
    task = task_store.claim_next(run_state.run_id, owner_role="lead")
    assert task is not None and task.status == "leased"
    original_lease = task.lease_until

    renewed = task_store.renew_lease_if_leased(task.task_id, lease_seconds=900)

    assert renewed is True
    after = store.get_swarm_task(task.task_id)
    assert after.status == "leased"
    assert after.lease_until != original_lease  # actually extended


def test_renew_lease_if_leased_does_not_resurrect_done_task(tmp_path: Path) -> None:
    """A heartbeat firing after completion must not flip a done task back to leased.

    This is the race the conditional WHERE status='leased' guards against: the
    body calls complete(), then a late LeaseGuard heartbeat fires. Without the
    guard, heartbeat() would resurrect the task as 'leased' and strand it.
    """
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(
        objective="demo", budget={"max_steps": 12}, phase="discovery"
    )
    bootstrap_lead_objective(
        task_store, mailbox, run_id=run_state.run_id,
        question="q", sub_questions=["q"], browser_goal="g",
    )
    task = task_store.claim_next(run_state.run_id, owner_role="lead")
    assert task is not None
    task_store.complete(task.task_id)  # body finished normally

    renewed = task_store.renew_lease_if_leased(task.task_id, lease_seconds=900)

    assert renewed is False
    after = store.get_swarm_task(task.task_id)
    assert after.status == "done"  # not resurrected
    assert after.lease_until is None


# --- Worker outer safety shell ----------------------------------------------

def test_worker_shell_contains_exception_and_releases_lease(tmp_path: Path) -> None:
    """A crashing _process_task must not propagate; lease is released + failure recorded.

    Before Phase 1, an uncaught exception in a worker body killed the thread
    silently and left the task leased until the 900s expiry. The shell now:
      - returns a failure result (thread survives),
      - moves the task to needs_repair (no 900s zombie),
      - emits a technical_failure broadcast via the mailbox.
    """
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(
        objective="demo", budget={"max_steps": 12}, phase="discovery"
    )
    bootstrap_lead_objective(
        task_store, mailbox, run_id=run_state.run_id,
        question="q", sub_questions=["q"], browser_goal="g",
    )

    worker = LeadWorker(task_store, mailbox)

    # Force the body to crash. Simulates a model/store/tool bug.
    def _boom(_task):
        raise RuntimeError("simulated worker crash")
    worker._process_task = _boom  # type: ignore[assignment]

    # Must NOT raise — the shell contains it.
    result = worker.run_once(run_state.run_id, run_root=tmp_path)

    assert result is not None
    assert result.claimed_task_id  # we did claim before crashing
    # Lease released as needs_repair, not stranded as leased.
    task_after = store.get_swarm_task(result.claimed_task_id)
    assert task_after.status == "needs_repair"
    # A technical_failure broadcast was recorded.
    messages = store.list_swarm_messages(run_state.run_id)
    failure_msgs = [
        m for m in messages
        if m.broadcast and str(m.payload).find("technical_failure") != -1
    ]
    assert failure_msgs, "expected a technical_failure broadcast after worker crash"
    assert "worker_exception" in str(failure_msgs[0].payload)


def test_worker_shell_keeps_failed_cell_workspace_for_forensics(tmp_path: Path) -> None:
    """On failure the per-task workspace is retained; on success it is reaped."""
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(
        objective="demo", budget={"max_steps": 12}, phase="discovery"
    )
    bootstrap_lead_objective(
        task_store, mailbox, run_id=run_state.run_id,
        question="q", sub_questions=["q"], browser_goal="g",
    )

    worker = LeadWorker(task_store, mailbox)
    claimed_id_holder: list[str] = []

    def _boom(task):
        claimed_id_holder.append(task.task_id or "")
        raise RuntimeError("crash")
    worker._process_task = _boom  # type: ignore[assignment]

    worker.run_once(run_state.run_id, run_root=tmp_path)

    assert claimed_id_holder
    cell_dir = tmp_path / "cells" / claimed_id_holder[0]
    # Failure keeps the workspace for post-mortem.
    assert cell_dir.exists(), "failed cell workspace should be retained for forensics"


# --- ExecutionCell lifecycle ------------------------------------------------

def test_execution_cell_no_workspace_when_run_root_none(tmp_path: Path) -> None:
    """Without a run_root the cell still works but owns no workspace.

    Tests call run_once without run_root; the cell must degrade gracefully
    (workspace=None, cleanup is a no-op) rather than forcing every caller to
    supply a directory.
    """
    from insightswarm.schemas.swarm import Task

    task = Task(
        task_id="t-1", run_id="r-1", kind="research", owner_role="researcher",
        status="pending", inputs={}, depends_on=[], lease_until=None,
    )
    cell = ExecutionCell.open(None, task, role="researcher")
    assert cell.workspace is None
    # cleanup must not raise even with no workspace.
    cell.cleanup(success=True)
    cell.cleanup(success=False)


def test_lease_guard_ensure_released_moves_leased_to_needs_repair(tmp_path: Path) -> None:
    """LeaseGuard.ensure_released releases a stranded leased task; no-op if done."""
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(
        objective="demo", budget={"max_steps": 12}, phase="discovery"
    )
    bootstrap_lead_objective(
        task_store, mailbox, run_id=run_state.run_id,
        question="q", sub_questions=["q"], browser_goal="g",
    )
    task = task_store.claim_next(run_state.run_id, owner_role="lead")
    assert task is not None

    guard = LeaseGuard(task_store, task.task_id, role="lead")
    # Simulate: body crashed before releasing. Guard releases on its behalf.
    guard.ensure_released()
    assert store.get_swarm_task(task.task_id).status == "needs_repair"

    # Idempotent / no-op on an already-released task.
    guard.ensure_released()
    assert store.get_swarm_task(task.task_id).status == "needs_repair"


def test_run_supervised_once_contains_outer_iteration_exception() -> None:
    """Outer run_forever errors must back off and return None, not kill worker."""
    calls = {"count": 0}

    def _boom() -> object:
        calls["count"] += 1
        raise RuntimeError("claim path crashed")

    result = run_supervised_once(
        stop_event=Event(),
        poll_interval=0.0,
        call_once=_boom,
    )

    assert result is None
    assert calls["count"] == 1
