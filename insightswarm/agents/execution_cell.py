"""ExecutionCell + LeaseGuard — Phase 1 minimal task isolation.

Each claim becomes a short-lived cell with:
  - a per-task workspace directory (forensic boundary for future temp files),
  - a heartbeat that keeps the lease alive while the body runs, and
  - a release safety net so an uncaught exception never leaves a task leased.

This is deliberately small: it does not change tool handlers, browser profiles,
or scheduling. It only draws the scheduler/worker boundary at the run_once call
site so a worker crash is contained to one task instead of killing the thread
and stranding the lease until expiry.
"""

from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

from insightswarm.agents.agent_failure_reporting import record_agent_technical_failure
from insightswarm.schemas.swarm import Task
from insightswarm.swarm_store import Mailbox, TaskStore


# Default lease lifetime matches claim_next's default. The heartbeat fires at
# lease/3 so a renewal always lands well before expiry even under model latency.
_DEFAULT_LEASE_SECONDS = 900
_T = TypeVar("_T")


@dataclass(frozen=True)
class ExecutionCell:
    """Per-task execution context: identity + isolated workspace.

    The workspace is created lazily under <run_root>/cells/<task_id>/. It is the
    single place Phase 2+ will route temp files / browser profiles through; for
    Phase 1 it exists to establish the lifecycle boundary and give failures a
    forensic trail (failed cells are kept, successful ones are reaped).
    """

    run_id: str
    task_id: str
    role: str
    workspace: Path | None

    @classmethod
    def open(cls, run_root: Path | None, task: Task, *, role: str) -> "ExecutionCell":
        workspace: Path | None = None
        if run_root is not None:
            workspace = Path(run_root) / "cells" / (task.task_id or "unknown")
            workspace.mkdir(parents=True, exist_ok=True)
        return cls(
            run_id=task.run_id,
            task_id=task.task_id or "",
            role=role,
            workspace=workspace,
        )

    def cleanup(self, *, success: bool) -> None:
        """Release the workspace. Success reaps it; failure keeps it for forensics."""
        if self.workspace is None:
            return
        if success:
            shutil.rmtree(self.workspace, ignore_errors=True)


class LeaseGuard:
    """Background heartbeat that keeps a leased task alive while the body runs.

    The heartbeat renews the lease ONLY while status == 'leased' (via
    renew_lease_if_leased), so it can never resurrect a task the body already
    completed/blocked. On exception, ensure_released() moves a still-leased task
    to needs_repair so it is not stranded for the full lease lifetime.
    """

    def __init__(
        self,
        task_store: TaskStore,
        task_id: str,
        *,
        role: str,
        lease_seconds: int = _DEFAULT_LEASE_SECONDS,
    ) -> None:
        self._task_store = task_store
        self._task_id = task_id
        self._role = role
        self._lease_seconds = lease_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"lease-{self._role}-{self._task_id}",
        )
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        # Wait one full interval before the first renewal: claim_next just set a
        # fresh lease_until, so an immediate beat would be pure noise.
        interval = self._lease_seconds / 3
        while not self._stop.wait(interval):
            try:
                self._task_store.renew_lease_if_leased(
                    self._task_id, lease_seconds=self._lease_seconds
                )
            except Exception:
                # A heartbeat failure must never take down the worker thread.
                # The lease will simply not be renewed this cycle; if it lapses,
                # the task becomes re-claimable rather than stuck.
                pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Bound join so a wedged heartbeat cannot hang teardown.
            self._thread.join(timeout=2.0)

    def ensure_released(self) -> None:
        """Move a still-leased task to needs_repair. No-op if already released."""
        try:
            current = self._task_store.store.get_swarm_task(self._task_id)
            if current.status in {"pending", "leased"}:
                self._task_store.needs_repair(self._task_id)
        except Exception:
            pass


def record_worker_exception(
    *,
    mailbox: Mailbox,
    role: str,
    task: Task,
    exc: BaseException,
) -> str | None:
    """Record an uncaught worker exception as a technical failure message.

    Wraps record_agent_technical_failure so every worker reports crashes through
    the same mailbox channel the runtime already monitors, instead of dying
    silently. status='worker_exception' routes through failure_policy to the
    'technical' category (retryable, no critic/research repair).
    """
    return record_agent_technical_failure(
        mailbox=mailbox,
        role=role,
        task=task,
        status="worker_exception",
        reason=f"{type(exc).__name__}: {exc}",
    )


def run_in_cell(
    *,
    task_store: TaskStore,
    mailbox: Mailbox,
    task: Task,
    role: str,
    run_root: Path | None,
    body: Callable[[Task], _T],
    make_failure_result: Callable[[Task], _T],
) -> _T:
    """Run a claimed task inside the standard cell/lease/failure shell."""
    cell = ExecutionCell.open(run_root, task, role=role)
    guard = LeaseGuard(task_store, task.task_id, role=role)
    guard.start()
    success = False
    try:
        result = body(task)
        success = True
        return result
    except Exception as exc:
        record_worker_exception(mailbox=mailbox, role=role, task=task, exc=exc)
        guard.ensure_released()
        return make_failure_result(task)
    finally:
        guard.stop()
        cell.cleanup(success=success)


def run_supervised_once(
    *,
    stop_event: threading.Event,
    poll_interval: float,
    call_once: Callable[[], _T | None],
) -> _T | None:
    """Run one worker iteration without letting an outer exception kill the thread.

    Task-body exceptions are still handled inside each worker's ExecutionCell.
    This catches failures that happen before/around claiming a task, where no
    task context exists yet; the worker backs off once and keeps serving the run.
    """
    try:
        return call_once()
    except Exception:
        stop_event.wait(poll_interval)
        return None
