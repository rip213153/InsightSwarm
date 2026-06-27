"""Process-local push-based wake-up for role workers.

Replaces the 0.5s fixed poll with a ``threading.Condition`` per role: producers
notify the target role right after the SQLite transaction that created the work
commits; workers block on ``wait(role, timeout)`` and wake near-real-time. The
worker poll interval is retained as a 5s fallback so cross-process work and any
condition-variable races still drain.

Design notes
------------
* One ``EventBus`` instance is shared per process (created in
  ``objective_runtime`` and threaded through ``TaskStore``/``Mailbox`` and each
  worker's ``run_forever``).
* No persistence: a fresh process starts with an empty bus and relies on the
  SQLite fallback poll to catch up on work another process produced.
* Lost-wakeup handling: ``threading.Condition`` does not remember notifies that
  fire while no one is waiting. The classic race (notify between
  ``claim_next`` returning ``None`` and the worker entering ``wait``) is
  therefore bounded by the 5s fallback poll — exactly the safety net the
  worker-pool-eventbus design calls for. In the common case the notify lands
  while the worker is still processing its previous task, so the next
  ``claim_next`` picks the new task up without ever waiting.
* ``threading.Condition`` (not asyncio): InsightSwarm is a thread-model
  system; the worker loop is a plain ``while``/``Event`` loop, not a coroutine.
"""

from __future__ import annotations

import threading


class EventBus:
    """Process-local role-keyed condition variables for push wake-up."""

    def __init__(self) -> None:
        # All condition variables share one lock. With ~6-14 worker threads the
        # contention is negligible, and sharing the lock keeps notify/wait
        # atomic with respect to condition creation.
        self._lock = threading.Lock()
        self._conditions: dict[str, threading.Condition] = {}

    def _get_or_create(self, role: str) -> threading.Condition:
        # Caller holds self._lock.
        cond = self._conditions.get(role)
        if cond is None:
            cond = threading.Condition(self._lock)
            self._conditions[role] = cond
        return cond

    def notify_role(self, role: str) -> None:
        """Wake every worker currently waiting on ``role``.

        Safe to call with a role no worker has ever waited on: the condition is
        created on demand and the notify becomes a no-op (no waiters).
        """
        with self._lock:
            cond = self._get_or_create(role)
            cond.notify_all()

    def notify_all_roles(self) -> None:
        """Wake every role that has ever been notified or waited on.

        Used on runtime teardown so workers blocked in ``wait`` observe
        ``stop_event`` within one wake-up instead of waiting up to the 5s
        fallback.
        """
        with self._lock:
            roles = list(self._conditions.keys())
        for role in roles:
            self.notify_role(role)

    def wait(self, role: str, timeout: float) -> bool:
        """Block up to ``timeout`` seconds for a notify on ``role``.

        Returns ``True`` if notified within the timeout, ``False`` if the
        timeout elapsed. See the module docstring for the lost-wakeup
        bound and how the 5s fallback poll covers it.
        """
        with self._lock:
            cond = self._get_or_create(role)
            return cond.wait(timeout=timeout)
