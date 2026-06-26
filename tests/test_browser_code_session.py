from __future__ import annotations

from pathlib import Path

from insightswarm.agents.browser_agent import BrowserWorker
from insightswarm.browser_code_session import BrowserCodeSession
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.models.clients import ModelResult
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore


class _OneCellBrowserModel:
    def complete(self, *args, **kwargs):
        del args, kwargs
        code = "\n".join(
            [
                "text = 'ExampleCo pricing page. Starter plan costs 49 dollars per month. This is relevant public pricing information for the acquisition task.'",
                "publish_raw_source(text, url='https://example.com/pricing', title='Example Pricing', why_ready='substantial public pricing text')",
                "done('complete', 'published raw source')",
            ]
        )
        return ModelResult(
            text=f"```python\n{code}\n```",
            json_data=None,
            provider="stub",
            model="stub-browser-code",
            usage={},
            latency_ms=0,
            raw_response={},
            status="ok",
        )

    def analyze_image(self, *args, **kwargs):
        raise AssertionError("vision should not be used on the happy path")


class _QuotaErrorBrowserModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        return ModelResult(
            text="",
            json_data=None,
            provider="stub",
            model="stub-browser-code",
            usage={},
            latency_ms=0,
            raw_response={"http_status": 403},
            status="error",
            error="OpenAI-compatible API HTTP 403: AllocationQuota.FreeTierOnly",
        )


def test_browser_worker_uses_code_session_to_publish_raw_source(tmp_path: Path) -> None:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    store = Store(db_path, artifact_dir)
    run_state = store.create_swarm_run_state(objective="Browser code task", budget={})
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    task_store.create(
        run_state.run_id,
        kind="hard_acquisition",
        status="pending",
        owner_role="browser_agent",
        inputs={"goal": "Collect example pricing"},
        priority=1,
        created_by="test",
    )
    trace_path = tmp_path / "steps.jsonl"

    result = BrowserWorker(task_store, mailbox, artifact_store).run_once(
        run_state.run_id,
        model_client=_OneCellBrowserModel(),
        trace_path=trace_path,
    )

    artifacts = store.list_swarm_artifacts(run_state.run_id)
    extractor_tasks = store.list_swarm_tasks(run_state.run_id, owner_role="extractor")
    assert result is not None
    assert result.terminal_status == "done"
    assert [artifact.type for artifact in artifacts] == ["raw_document"]
    assert len(extractor_tasks) == 1
    assert trace_path.exists()
    assert "browser_agent_code_session" in trace_path.read_text(encoding="utf-8")


def test_browser_code_session_stops_on_nonrecoverable_model_quota_error(tmp_path: Path) -> None:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    store = Store(db_path, artifact_dir)
    run_state = store.create_swarm_run_state(objective="Browser quota task", budget={})
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    board_store = BoardStore(store)
    task = task_store.create(
        run_state.run_id,
        kind="hard_acquisition",
        status="pending",
        owner_role="browser_agent",
        inputs={"goal": "Collect page"},
        priority=1,
        created_by="test",
    )
    from insightswarm.agents.browser_agent_tools import BrowserAgentToolHandlers, BrowserAgentToolState

    state = BrowserAgentToolState()
    handlers = BrowserAgentToolHandlers(
        task=task,
        mailbox=mailbox,
        artifact_store=artifact_store,
        task_store=task_store,
        board_store=board_store,
        state=state,
        model_client=_QuotaErrorBrowserModel(),
    )
    model = _QuotaErrorBrowserModel()

    result = BrowserCodeSession(
        task=task,
        handlers=handlers,
        tool_state=state,
        model_client=model,
        trace_path=tmp_path / "steps.jsonl",
    ).run()

    assert model.calls == 1
    assert result.terminal_status == "blocked"
    assert "Browser model unavailable" in (result.terminal_reason or "")
