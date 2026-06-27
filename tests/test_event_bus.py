from __future__ import annotations

import threading
import time

from insightswarm.event_bus import EventBus


def test_wait_returns_true_when_notified() -> None:
    bus = EventBus()
    result_box: dict[str, object] = {}

    def _waiter() -> None:
        result_box["notified"] = bus.wait("researcher", timeout=5.0)

    thread = threading.Thread(target=_waiter, daemon=True)
    thread.start()
    # Let the waiter actually enter wait().
    time.sleep(0.05)
    bus.notify_role("researcher")
    thread.join(timeout=2.0)

    assert result_box.get("notified") is True


def test_wait_returns_false_on_timeout() -> None:
    bus = EventBus()
    start = time.monotonic()
    notified = bus.wait("lead", timeout=0.1)
    elapsed = time.monotonic() - start
    assert notified is False
    assert elapsed >= 0.1


def test_notify_role_with_no_waiters_is_noop() -> None:
    # A notify on a role nobody has waited on yet must not raise; the next wait
    # on that role still behaves correctly (timeout, since the earlier notify
    # is not retained by the condition variable — the 5s fallback covers this).
    bus = EventBus()
    bus.notify_role("extractor")
    notified = bus.wait("extractor", timeout=0.05)
    assert notified is False


def test_notify_all_roles_wakes_every_role() -> None:
    bus = EventBus()
    results: dict[str, object] = {}

    def _wait(role: str) -> None:
        results[role] = bus.wait(role, timeout=5.0)

    roles = ["researcher", "extractor", "critic"]
    threads = [threading.Thread(target=_wait, args=(role,), daemon=True) for role in roles]
    for thread in threads:
        thread.start()
    time.sleep(0.05)
    bus.notify_all_roles()
    for thread in threads:
        thread.join(timeout=2.0)

    assert all(results.get(role) is True for role in roles)


def test_notify_wakes_only_target_role() -> None:
    bus = EventBus()
    target_result: dict[str, object] = {}
    other_result: dict[str, object] = {}

    def _wait_target() -> None:
        target_result["notified"] = bus.wait("researcher", timeout=5.0)

    def _wait_other() -> None:
        other_result["notified"] = bus.wait("extractor", timeout=0.2)

    target_thread = threading.Thread(target=_wait_target, daemon=True)
    other_thread = threading.Thread(target=_wait_other, daemon=True)
    target_thread.start()
    other_thread.start()
    time.sleep(0.05)
    bus.notify_role("researcher")
    target_thread.join(timeout=2.0)
    other_thread.join(timeout=2.0)

    assert target_result.get("notified") is True
    assert other_result.get("notified") is False
