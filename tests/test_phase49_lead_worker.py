from __future__ import annotations

import threading
import time
from pathlib import Path

from insightswarm.agents.lead import LeadWorker, bootstrap_lead_objective
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.swarm_store import Mailbox, TaskStore


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def test_lead_worker_creates_only_tasks_and_messages(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(
        objective="DeepSeek 下步战略",
        budget={"max_steps": 12},
        phase="discovery",
    )

    root_task = bootstrap_lead_objective(
        task_store,
        mailbox,
        run_id=run_state.run_id,
        question="DeepSeek 下步战略",
        sub_questions=["融资进展", "海外扩张"],
        browser_goal="获取需要浏览器交互的原始页面",
    )

    result = LeadWorker(task_store, mailbox).run_once(run_state.run_id)

    downstream_tasks = store.list_swarm_tasks(run_state.run_id)
    sub_researcher_tasks = [
        task for task in downstream_tasks if task.owner_role == "sub_researcher"
    ]
    browser_tasks = [
        task for task in downstream_tasks if task.owner_role == "browser_agent"
    ]
    lead_messages = mailbox.inbox(run_state.run_id, role="sub_researcher")
    browser_messages = mailbox.inbox(run_state.run_id, role="browser_agent")
    broadcasts = mailbox.broadcasts(run_state.run_id)
    completed_root = store.get_swarm_task(root_task.task_id)

    assert result is not None
    assert result.claimed_task_id == root_task.task_id
    assert len(result.created_task_ids) == 3
    assert len(sub_researcher_tasks) == 2
    assert len(browser_tasks) == 1
    assert all(task.created_by == "lead" for task in sub_researcher_tasks + browser_tasks)
    assert completed_root.status == "done"
    assert len([message for message in lead_messages if message.type == "request" and message.payload.get("kind") == "research_subquestion"]) == 2
    assert len([message for message in browser_messages if message.type == "request" and message.payload.get("kind") == "hard_acquisition"]) == 1
    assert any(message.type == "observation" and message.payload.get("kind") == "progress_update" for message in broadcasts)


def test_lead_worker_turns_repair_request_into_followup_work_order(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(
        objective="Repair cycle",
        budget={"max_steps": 12},
        phase="research",
    )

    repair_task = task_store.create(
        run_state.run_id,
        kind="repair_request",
        status="pending",
        owner_role="lead",
        inputs={"targeted_query": "补充 DeepSeek 海外证据", "owner_role": "sub_researcher"},
        priority=8,
        created_by="critic",
    )

    result = LeadWorker(task_store, mailbox).run_once(run_state.run_id)

    downstream_tasks = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.task_id != repair_task.task_id
    ]
    repair_messages = mailbox.inbox(run_state.run_id, role="sub_researcher")

    assert result is not None
    assert result.claimed_task_id == repair_task.task_id
    assert len(downstream_tasks) == 1
    assert downstream_tasks[0].kind == "research_repair"
    assert downstream_tasks[0].inputs["question"] == "补充 DeepSeek 海外证据"
    assert len(repair_messages) == 1
    assert repair_messages[0].type == "request"
    assert repair_messages[0].payload["kind"] == "research_repair"


def test_lead_worker_runs_until_idle_across_multiple_lead_tasks(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(
        objective="Lead polling loop",
        budget={"max_steps": 12},
        phase="research",
    )

    bootstrap_lead_objective(
        task_store,
        mailbox,
        run_id=run_state.run_id,
        question="Primary objective",
        sub_questions=["question-a"],
    )
    task_store.create(
        run_state.run_id,
        kind="repair_request",
        status="pending",
        owner_role="lead",
        inputs={"targeted_query": "repair follow-up", "owner_role": "sub_researcher"},
        priority=9,
        created_by="critic",
    )

    loop_result = LeadWorker(task_store, mailbox).run_until_idle(run_state.run_id)

    lead_tasks = [task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role == "lead"]
    downstream = [task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role != "lead"]

    assert loop_result.iterations == 2
    assert len(loop_result.claimed_task_ids) == 2
    assert all(task.status == "done" for task in lead_tasks)
    assert any(task.kind == "research_subquestion" for task in downstream)
    assert any(task.kind == "research_repair" for task in downstream)


def test_lead_worker_run_forever_consumes_tasks_and_exits_on_stop_event(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    run_state = store.create_swarm_run_state(
        objective="Lead forever loop",
        budget={"max_steps": 12},
        phase="discovery",
    )

    bootstrap_lead_objective(
        task_store,
        mailbox,
        run_id=run_state.run_id,
        question="Forever objective",
        sub_questions=["loop-question"],
    )

    stop_event = threading.Event()
    result_box: dict[str, object] = {}

    def _run() -> None:
        result_box["result"] = LeadWorker(task_store, mailbox).run_forever(
            run_state.run_id,
            stop_event,
            poll_interval=0.01,
        )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if any(task.owner_role == "sub_researcher" for task in store.list_swarm_tasks(run_state.run_id)):
            break
        time.sleep(0.01)

    stop_event.set()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert result_box["result"].iterations >= 1
    assert any(task.kind == "research_subquestion" for task in store.list_swarm_tasks(run_state.run_id))
