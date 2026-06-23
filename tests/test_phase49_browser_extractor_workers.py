from __future__ import annotations

import threading
import time
from pathlib import Path

from insightswarm.agents.browser_agent import BrowserWorker
from insightswarm.agents.browser_agent_tools import BrowserAgentToolHandlers, BrowserAgentToolState
from insightswarm.agents.extractor import Extractor as ExtractorWorker
from insightswarm.agents.extractor_tools import ExtractorToolHandlers, ExtractorToolState
from insightswarm.agents.lead import LeadWorker, bootstrap_lead_objective
from insightswarm.authorization_flow import write_authorization_decision
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


class _ExtractorCitationModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, response_format=None, max_tokens=None, temperature=None, metadata=None):
        del messages, response_format, max_tokens, temperature, metadata
        self.calls += 1
        if self.calls == 1:
            payload = {
                "private_state": {"plan": "Read the raw browser document."},
                "tool_call": {"name": "read_raw_document", "input": {}},
            }
        elif self.calls == 2:
            quote = "Raw browser capture for: Open public DeepSeek news page."
            payload = {
                "private_state": {"plan": "Create one quote-backed citation from the raw browser text."},
                "tool_call": {
                    "name": "propose_citations",
                    "input": {
                        "candidates": [
                            {
                                "claim": "BrowserAgent acquired public visible text for the requested DeepSeek browser source.",
                                "quote": quote,
                                "confidence": 0.8,
                                "rationale": "The quote is exact text from the fallback raw browser document.",
                            }
                        ],
                        "why_these_quotes": "The test only needs one quote-backed citation to verify the shared-store pipeline.",
                    },
                },
            }
        else:
            payload = {
                "private_state": {"plan": "Finish extraction after citation creation."},
                "tool_call": {
                    "name": "finish_extraction",
                    "input": {"status": "complete", "reason": "citation created"},
                },
            }
        return type("ModelResult", (), {"status": "ok", "json_data": payload, "text": ""})()


def test_lead_browser_extractor_pipeline_uses_tasks_messages_and_artifacts(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Collect browser-only source",
        budget={"max_steps": 12},
        phase="discovery",
    )

    bootstrap_lead_objective(
        task_store,
        mailbox,
        run_id=run_state.run_id,
        question="DeepSeek browser source",
        sub_questions=[],
        browser_goal="Open public DeepSeek news page",
    )
    LeadWorker(task_store, mailbox).run_once(run_state.run_id)

    browser_result = BrowserWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)
    raw_artifacts = [
        artifact for artifact in store.list_swarm_artifacts(run_state.run_id) if artifact.type == "raw_document"
    ]
    extractor_tasks = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role == "extractor"
    ]
    extractor_messages = [
        message
        for message in mailbox.inbox(run_state.run_id, role="extractor")
        if message.type == "request" and message.payload.get("kind") == "extract_evidence"
    ]

    assert browser_result is not None
    assert len(raw_artifacts) == 1
    assert len(extractor_tasks) == 1
    assert len(extractor_messages) == 1

    extractor_result = ExtractorWorker(task_store, mailbox, artifact_store).run_once(
        run_state.run_id,
        model_client=_ExtractorCitationModel(),
    )
    citation_artifacts = [
        artifact for artifact in store.list_swarm_artifacts(run_state.run_id) if artifact.type == "citation"
    ]
    evidence_rows = store.list_swarm_evidence(run_state.run_id)

    assert extractor_result is not None
    assert len(citation_artifacts) == 1
    assert len(evidence_rows) == 1
    assert evidence_rows[0].quote.startswith("Raw browser capture")


def test_browser_worker_emits_authorization_request_for_high_risk_goal(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="High risk browser action",
        budget={"max_steps": 12},
        phase="research",
    )
    task_store.create(
        run_state.run_id,
        kind="hard_acquisition",
        status="pending",
        owner_role="browser_agent",
        inputs={"goal": "Log in and download the private report"},
        created_by="lead",
    )

    result = BrowserWorker(task_store, mailbox, artifact_store).run_once(run_state.run_id)

    authorization_messages = [
        message
        for message in mailbox.inbox(run_state.run_id, role="lead")
        if message.type == "observation" and message.payload.get("kind") == "authorization_request"
    ]
    raw_artifacts = [
        artifact for artifact in store.list_swarm_artifacts(run_state.run_id) if artifact.type == "raw_document"
    ]
    browser_tasks = [
        task for task in store.list_swarm_tasks(run_state.run_id) if task.owner_role == "browser_agent"
    ]

    assert result is not None
    assert len(authorization_messages) == 1
    assert raw_artifacts == []
    assert browser_tasks[0].status == "blocked"


def test_browser_execute_code_returns_last_expression_result(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Browser expression result",
        budget={"max_steps": 12},
        phase="research",
    )
    task = task_store.create(
        run_state.run_id,
        kind="hard_acquisition",
        status="leased",
        owner_role="browser_agent",
        inputs={"goal": "Read fake browser text", "target_url": "https://example.com/pricing", "browser_backend": "fake"},
        created_by="test",
    )
    handlers = BrowserAgentToolHandlers(
        task=task,
        task_store=task_store,
        mailbox=mailbox,
        artifact_store=artifact_store,
        board_store=BoardStore(store),
        state=BrowserAgentToolState(),
    )

    handlers.read_task({})
    result = handlers.execute_browser_code(
        {
            "code": "visible_text(max_chars=200)",
            "why_this_code": "Verify the last expression is returned to the model.",
        }
    )

    assert result["ok"] is True
    assert "ExampleCo pricing page" in result["output"]
    assert "ExampleCo pricing page" in result["result"]


def test_browser_execute_code_allows_cookie_banner_helper(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Browser cookie banner helper",
        budget={"max_steps": 12},
        phase="research",
    )
    task = task_store.create(
        run_state.run_id,
        kind="hard_acquisition",
        status="leased",
        owner_role="browser_agent",
        inputs={"goal": "Dismiss public cookie overlay if present", "target_url": "https://example.com/pricing", "browser_backend": "fake"},
        created_by="test",
    )
    handlers = BrowserAgentToolHandlers(
        task=task,
        task_store=task_store,
        mailbox=mailbox,
        artifact_store=artifact_store,
        board_store=BoardStore(store),
        state=BrowserAgentToolState(),
    )

    handlers.read_task({})
    result = handlers.execute_browser_code(
        {
            "code": "dismiss_cookie_banner(prefer='reject')",
            "why_this_code": "Cookie dismissal is a low-risk browser helper, not raw cookie access.",
        }
    )

    assert result["ok"] is True
    assert result["result"]["clicked"] is False
    assert result["result"]["reason"] == "no cookie/privacy overlay control found"


def test_browser_execute_code_blocks_third_identical_snippet(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Browser repeat guard",
        budget={"max_steps": 12},
        phase="research",
    )
    task = task_store.create(
        run_state.run_id,
        kind="hard_acquisition",
        status="leased",
        owner_role="browser_agent",
        inputs={"goal": "Read fake browser text", "target_url": "https://example.com/pricing", "browser_backend": "fake"},
        created_by="test",
    )
    handlers = BrowserAgentToolHandlers(
        task=task,
        task_store=task_store,
        mailbox=mailbox,
        artifact_store=artifact_store,
        board_store=BoardStore(store),
        state=BrowserAgentToolState(),
    )

    handlers.read_task({})
    for _ in range(2):
        result = handlers.execute_browser_code(
            {
                "code": "visible_text(max_chars=200)",
                "why_this_code": "Try reading the same visible text.",
            }
        )
        assert result["ok"] is True

    blocked = handlers.execute_browser_code(
        {
            "code": "visible_text(max_chars=200)",
            "why_this_code": "This third identical call should be blocked.",
        }
    )

    assert blocked["ok"] is False
    assert "repeated_browser_code_blocked" in blocked["error"]


def test_human_authorization_allows_retrying_high_risk_navigation(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Browser login authorization retry",
        budget={"max_steps": 12},
        phase="research",
    )
    task = task_store.create(
        run_state.run_id,
        kind="hard_acquisition",
        status="leased",
        owner_role="browser_agent",
        inputs={
            "goal": "Open account-gated public page",
            "target_url": "https://example.com/login",
            "browser_backend": "fake",
            "login_allowlist": ["example.com"],
        },
        created_by="test",
    )
    first_handlers = BrowserAgentToolHandlers(
        task=task,
        task_store=task_store,
        mailbox=mailbox,
        artifact_store=artifact_store,
        board_store=BoardStore(store),
        state=BrowserAgentToolState(),
    )

    first_handlers.read_task({})
    first_result = first_handlers.execute_browser_code(
        {
            "code": "open_url(target_url)",
            "why_this_code": "Opening a login URL must request human authorization first.",
        }
    )

    assert first_result["ok"] is True
    assert first_result["authorization_requested"] is True

    write_authorization_decision(
        store,
        run_state.run_id,
        task_id=task.task_id,
        decision="allow",
        reason="operator will complete login manually in the visible browser",
    )
    retry_handlers = BrowserAgentToolHandlers(
        task=task,
        task_store=task_store,
        mailbox=mailbox,
        artifact_store=artifact_store,
        board_store=BoardStore(store),
        state=BrowserAgentToolState(),
    )

    retry_handlers.read_task({})
    retry_result = retry_handlers.execute_browser_code(
        {
            "code": "open_url(target_url)",
            "why_this_code": "After approval the BrowserAgent may navigate, but still cannot type credentials.",
        }
    )

    assert retry_result["ok"] is True
    assert retry_result["authorization_requested"] is False
    assert retry_result["result"]["status"] == "fake_action_executed"


def test_browser_login_authorization_requires_allowlisted_domain(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Browser login allowlist",
        budget={"max_steps": 12},
        phase="research",
    )
    task = task_store.create(
        run_state.run_id,
        kind="hard_acquisition",
        status="leased",
        owner_role="browser_agent",
        inputs={"goal": "Open account-gated public page", "target_url": "https://not-allowed.example/login", "browser_backend": "fake"},
        created_by="test",
    )
    handlers = BrowserAgentToolHandlers(
        task=task,
        task_store=task_store,
        mailbox=mailbox,
        artifact_store=artifact_store,
        board_store=BoardStore(store),
        state=BrowserAgentToolState(),
    )

    handlers.read_task({})
    result = handlers.execute_browser_code(
        {
            "code": "request_login_authorization(login_url=target_url, reason='Need operator login')",
            "why_this_code": "Login authorization must be constrained by an explicit allowlist.",
        }
    )

    assert result["ok"] is True
    assert result["result"]["ok"] is False
    assert "not allowlisted" in result["result"]["error"]


def test_extractor_can_request_browser_acquisition_for_dynamic_low_signal_source(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Observe a dynamic visual page",
        budget={"max_steps": 12},
        phase="research",
    )
    raw_artifact = artifact_store.write_raw_document(
        run_state.run_id,
        source_task_id=None,
        document={
            "source_url": "https://www.cartier.com/en-fr/watchesandwonders#/",
            "url": "https://www.cartier.com/en-fr/watchesandwonders#/",
            "title": "Watches & Wonders",
            "text": "Watches & Wonders\nWatchmaker of shapes, master of crafts\n360 view\nText\nClose",
            "html": "",
        },
        summary="Sparse SPA text",
    )
    task = task_store.create(
        run_state.run_id,
        kind="raw_document",
        status="leased",
        owner_role="extractor",
        inputs={"artifact_id": raw_artifact.artifact_id},
        created_by="test",
    )
    handlers = ExtractorToolHandlers(
        task=task,
        task_store=task_store,
        mailbox=mailbox,
        artifact_store=artifact_store,
        board_store=BoardStore(store),
        state=ExtractorToolState(),
    )

    handlers.read_raw_document({})
    result = handlers.request_browser_acquisition(
        {
            "goal": "Use a visible browser to observe Cartier page modules, overlays, scrolling, and interactive entry points.",
            "target_url": "https://www.cartier.com/en-fr/watchesandwonders#/",
            "why_browser_needed": "Raw text is sparse and appears to miss SPA/visual/interactive content needed for citation-backed evidence.",
        }
    )

    browser_tasks = [
        item
        for item in store.list_swarm_tasks(run_state.run_id)
        if item.kind == "hard_acquisition" and item.owner_role == "browser_agent"
    ]
    browser_messages = [
        item
        for item in mailbox.inbox(run_state.run_id, role="browser_agent")
        if item.type == "request" and item.payload.get("kind") == "hard_acquisition"
    ]

    assert result["ok"] is True
    assert result["deduped"] is False
    assert len(browser_tasks) == 1
    assert len(browser_messages) == 1
    assert browser_tasks[0].inputs["target_url"] == "https://www.cartier.com/en-fr/watchesandwonders#/"
    assert browser_tasks[0].created_by == "extractor"

    deduped = handlers.request_browser_acquisition(
        {
            "goal": "Use a visible browser to observe Cartier page modules, overlays, scrolling, and interactive entry points.",
            "target_url": "https://www.cartier.com/en-fr/watchesandwonders#/",
            "why_browser_needed": "A BrowserAgent task already exists for this dynamic source.",
        }
    )
    browser_tasks_after_dedupe = [
        item
        for item in store.list_swarm_tasks(run_state.run_id)
        if item.kind == "hard_acquisition" and item.owner_role == "browser_agent"
    ]

    assert deduped["ok"] is True
    assert deduped["deduped"] is True
    assert len(browser_tasks_after_dedupe) == 1


def test_browser_and_extractor_run_forever_consume_pipeline_and_exit(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Forever browser extractor loop",
        budget={"max_steps": 12},
        phase="discovery",
    )

    bootstrap_lead_objective(
        task_store,
        mailbox,
        run_id=run_state.run_id,
        question="DeepSeek browser source",
        sub_questions=[],
        browser_goal="Open public DeepSeek news page",
    )
    LeadWorker(task_store, mailbox).run_once(run_state.run_id)

    stop_event = threading.Event()
    browser_results: dict[str, object] = {}
    extractor_results: dict[str, object] = {}

    def _browser_loop() -> None:
        browser_results["results"] = BrowserWorker(task_store, mailbox, artifact_store).run_forever(
            run_state.run_id,
            stop_event,
            poll_interval=0.01,
        )

    def _extractor_loop() -> None:
        extractor_results["results"] = ExtractorWorker(task_store, mailbox, artifact_store).run_forever(
            run_state.run_id,
            stop_event,
            poll_interval=0.01,
            model_client=_ExtractorCitationModel(),
        )

    browser_thread = threading.Thread(target=_browser_loop, daemon=True)
    extractor_thread = threading.Thread(target=_extractor_loop, daemon=True)
    browser_thread.start()
    extractor_thread.start()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        if store.list_swarm_evidence(run_state.run_id):
            break
        time.sleep(0.01)

    stop_event.set()
    browser_thread.join(timeout=5.0)
    extractor_thread.join(timeout=5.0)

    assert not browser_thread.is_alive()
    assert not extractor_thread.is_alive()
    assert len(browser_results["results"]) >= 1
    assert len(extractor_results["results"]) >= 1
    assert len(store.list_swarm_evidence(run_state.run_id)) == 1
