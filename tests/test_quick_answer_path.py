"""Tests for the quick-answer fast path: quick_read + finish_with_answer.

This path lets Researcher deliver a final answer directly, bypassing
Extractor/Critic/Writer. It exists because most factual/news/explanatory
questions do not need quote-level evidence — the URL itself is sufficient
provenance. The full pipeline remains available for questions needing
verbatim quotes or cross-verification.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from insightswarm.agents.researcher_tools import ResearcherToolHandlers, ResearcherToolState
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.objective_runtime import _build_delivery_result, _load_report_from_store, _latest_quick_answer
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore
from insightswarm.tools.core import ToolResult


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def _build_handlers(store: Store, *, question: str = "2026年中国航空燃油费为什么屡次提高") -> tuple[ResearcherToolHandlers, ResearcherToolState]:
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective=question,
        budget={"max_steps": 12},
        phase="research",
    )
    task = task_store.create(
        run_state.run_id,
        kind="research_subquestion",
        status="leased",
        owner_role="researcher",
        inputs={"question": question},
        created_by="lead",
    )
    state = ResearcherToolState()
    handlers = ResearcherToolHandlers(
        task=task,
        task_store=task_store,
        mailbox=mailbox,
        artifact_store=artifact_store,
        board_store=board_store,
        state=state,
    )
    # read_task must be called first to populate task_context.
    handlers.read_task({})
    return handlers, state


def _stub_fetch(text: str = "中国民航局于2026年3月调整了航空燃油附加费，原因是国际原油价格持续上涨，航空公司运营成本压力增大。" * 30, title: str = "官方来源") -> Any:
    def _fetch(self, tool_input: dict[str, Any], context=None) -> ToolResult:
        return ToolResult(
            "ok",
            data={
                "source_url": tool_input["url"],
                "url": tool_input["url"],
                "title": title,
                "text": text,
                "html": f"<html><title>{title}</title></html>",
            },
        )
    return _fetch


def test_quick_read_returns_summary_and_key_points_without_extractor(tmp_path: Path, monkeypatch) -> None:
    """quick_read must fetch a URL and return summary + key_points in one call.

    It must NOT publish a raw_document or create an Extractor task — that's the
    whole point of the fast path. The URL itself is the provenance.
    """
    store = _build_store(tmp_path)
    handlers, state = _build_handlers(store)
    monkeypatch.setattr("insightswarm.agents.researcher_tools.FetchUrlTool.run", _stub_fetch())

    result = handlers.quick_read({"url": "https://caac.gov.cn/news/2026-fuel"})

    assert result["ok"] is True
    source = result["source"]
    assert source["url"] == "https://caac.gov.cn/news/2026-fuel"
    assert source["title"] == "官方来源"
    assert source["usable"] is True
    assert len(source["summary"]) > 0
    assert isinstance(source["key_points"], list)
    # The source is recorded in quick_read_sources, NOT in fetched_documents.
    assert len(state.quick_read_sources) == 1
    assert state.quick_read_sources[0]["url"] == "https://caac.gov.cn/news/2026-fuel"
    # No raw_document artifact was published.
    artifacts = store.list_swarm_artifacts(handlers.task.run_id)
    assert all(a.type != "raw_document" for a in artifacts)
    # No Extractor task was created.
    tasks = store.list_swarm_tasks(handlers.task.run_id)
    assert all(t.owner_role != "extractor" for t in tasks)


def test_quick_read_dedupes_repeated_url(tmp_path: Path, monkeypatch) -> None:
    """A second quick_read of the same normalized URL must not re-fetch."""
    store = _build_store(tmp_path)
    handlers, state = _build_handlers(store)
    fetch_calls: list[str] = []
    original_fetch = _stub_fetch()

    def _counting_fetch(self, tool_input: dict[str, Any], context=None) -> ToolResult:
        fetch_calls.append(tool_input["url"])
        return original_fetch(self, tool_input, context)

    monkeypatch.setattr("insightswarm.agents.researcher_tools.FetchUrlTool.run", _counting_fetch)

    handlers.quick_read({"url": "https://example.com/path/"})
    result = handlers.quick_read({"url": "https://example.com/path"})

    assert result["ok"] is True
    assert result.get("deduped") is True
    assert len(fetch_calls) == 1


def test_quick_read_signals_fast_path_ready_for_convergence(tmp_path: Path, monkeypatch) -> None:
    """quick_read must return fast_path_ready + required_next_step to guide the model to finish_with_answer.

    Regression for the trace where Researcher called quick_read but never
    transitioned to finish_with_answer, continuing with the full pipeline.
    The fast_path_ready signal + required_next_step must make convergence
    unambiguous.
    """
    store = _build_store(tmp_path)
    handlers, state = _build_handlers(store)
    monkeypatch.setattr("insightswarm.agents.researcher_tools.FetchUrlTool.run", _stub_fetch())

    result = handlers.quick_read({"url": "https://caac.gov.cn/news/2026-fuel"})

    # A usable source with content must signal fast_path_ready.
    assert result["ok"] is True
    assert result["fast_path_ready"] is True
    # required_next_step must point to finish_with_answer, not fetch_source.
    required = result.get("required_next_step", "")
    assert "finish_with_answer" in required
    assert "fetch_source" in required  # mentioned as "do NOT call"


def test_quick_read_no_fast_path_ready_when_source_unusable(tmp_path: Path, monkeypatch) -> None:
    """quick_read must NOT signal fast_path_ready when the source is empty/unusable."""
    store = _build_store(tmp_path)
    handlers, state = _build_handlers(store)
    monkeypatch.setattr(
        "insightswarm.agents.researcher_tools.FetchUrlTool.run",
        _stub_fetch(text="", title="empty"),
    )

    result = handlers.quick_read({"url": "https://example.com/empty"})

    assert result["ok"] is True
    assert result["fast_path_ready"] is False
    assert "required_next_step" not in result


def test_quick_read_rejects_low_density_page_as_unusable(tmp_path: Path, monkeypatch) -> None:
    """quick_read must reuse _classify_page and reject blocked/modal/boilerplate pages.

    Regression for Cartier-style shells: a page with text but no real content
    (e.g., boilerplate shell) must NOT slip through as usable. Without this,
    the fast path would surface garbage summaries to finish_with_answer.
    """
    store = _build_store(tmp_path)
    handlers, state = _build_handlers(store)
    # Captcha-like content — _classify_page returns page_type=blocked.
    monkeypatch.setattr(
        "insightswarm.agents.researcher_tools.FetchUrlTool.run",
        _stub_fetch(text="Please verify you are human. Enable javascript to continue. captcha required.", title="blocked"),
    )

    result = handlers.quick_read({"url": "https://example.com/blocked"})

    assert result["ok"] is True
    source = result["source"]
    assert source["usable"] is False
    assert source["page_type"] == "blocked"
    # No fast_path_ready signal — model must not converge on a blocked source.
    assert result["fast_path_ready"] is False
    # Acquisition pressure should hint browser escalation for blocked pages.
    pressure = source.get("acquisition_pressure") or {}
    assert pressure.get("recommended_escalation") == "browser_agent"


def test_fetch_source_snippet_insufficient_is_per_url(tmp_path: Path, monkeypatch) -> None:
    """snippet_insufficient is gated per-URL, not globally.

    Once a specific URL has been quick_read, 'snippet_insufficient' is
    self-contradictory for THAT URL (the model has seen its L1 content). But
    quick_reading source A must NOT block fetching source B with
    snippet_insufficient — B's snippet is genuinely all the model has seen of B.
    This is the per-URL L0→L1→L2 ladder, not a global latch.
    """
    store = _build_store(tmp_path)
    handlers, state = _build_handlers(store)
    monkeypatch.setattr("insightswarm.agents.researcher_tools.FetchUrlTool.run", _stub_fetch())

    # Before any quick_read: snippet_insufficient is valid (L0→L2 skip allowed
    # when the model judges L1 won't help, e.g. known PDF).
    result = handlers.fetch_source({"url": "https://example.com/doc", "reason": "snippet_insufficient"})
    assert result["ok"] is True

    # After quick_reading a DIFFERENT url: snippet_insufficient is still valid
    # for the unfetched url. Reading A at L1 does not block fetching B at L2.
    handlers.quick_read({"url": "https://example.com/other"})
    result = handlers.fetch_source({"url": "https://example.com/doc2", "reason": "snippet_insufficient"})
    assert result["ok"] is True

    # After quick_reading the SAME url: snippet_insufficient is rejected for it
    # — the model has seen its L1 content, so "snippet insufficient" is
    # self-contradictory. Must use an objective reason to escalate it.
    handlers.quick_read({"url": "https://example.com/same"})
    result = handlers.fetch_source({"url": "https://example.com/same", "reason": "snippet_insufficient"})
    assert result["ok"] is False
    assert result["failure_kind"] == "invalid_fetch_reason"
    assert "snippet_insufficient" in result["error"]

    # Objective reasons still work for a quick_read url.
    result = handlers.fetch_source({"url": "https://example.com/same", "reason": "verbatim_quote"})
    assert result["ok"] is True


def test_finish_with_answer_writes_report_and_terminates(tmp_path: Path, monkeypatch) -> None:
    """finish_with_answer must write a report artifact, broadcast quick_answer_ready, and terminate.

    This is the terminal step of the fast path: it bypasses Extractor/Critic/Writer
    and delivers the answer directly. The runtime detects the report artifact and
    stops the run.
    """
    store = _build_store(tmp_path)
    handlers, state = _build_handlers(store)
    monkeypatch.setattr("insightswarm.agents.researcher_tools.FetchUrlTool.run", _stub_fetch())

    # Quick-read two sources.
    handlers.quick_read({"url": "https://caac.gov.cn/news/2026-fuel"})
    handlers.quick_read({"url": "https://example.com/analysis", "why_this_source": "industry analysis"})

    result = handlers.finish_with_answer({
        "answer": "2026年中国航空燃油费屡次提高的主要原因是国际原油价格波动和民航局的政策调整 [1][2]。",
        "sources": [
            {"url": "https://caac.gov.cn/news/2026-fuel", "title": "民航局公告", "summary": "官方调整通知"},
            {"url": "https://example.com/analysis", "title": "行业分析", "summary": "油价波动分析"},
        ],
        "confidence": "high",
        "reason": "factual question with official source",
    })

    assert result["ok"] is True
    assert result["terminal"] is True
    assert result["status"] == "done"
    assert "report_artifact_id" in result
    assert result["source_count"] == 2
    assert state.terminal_status == "done"

    # A report artifact was written.
    artifacts = store.list_swarm_artifacts(handlers.task.run_id)
    report_artifacts = [a for a in artifacts if a.type == "report"]
    assert len(report_artifacts) == 1

    # A quick_answer_ready broadcast message was sent.
    messages = store.list_swarm_messages(handlers.task.run_id)
    quick_messages = [m for m in messages if m.payload.get("kind") == "quick_answer_ready"]
    assert len(quick_messages) == 1
    assert quick_messages[0].from_role == "researcher"
    assert quick_messages[0].payload["report_artifact_id"] == report_artifacts[0].artifact_id
    assert quick_messages[0].payload["confidence"] == "high"


def test_finish_with_answer_requires_answer_and_sources(tmp_path: Path) -> None:
    """finish_with_answer must reject calls missing answer or sources."""
    store = _build_store(tmp_path)
    handlers, _ = _build_handlers(store)

    no_answer = handlers.finish_with_answer({"sources": [{"url": "https://example.com"}]})
    assert no_answer["ok"] is False
    assert "answer" in no_answer["error"]

    no_sources = handlers.finish_with_answer({"answer": "some answer", "sources": []})
    assert no_sources["ok"] is False
    assert "source" in no_sources["error"]

    no_urls = handlers.finish_with_answer({"answer": "some answer", "sources": [{"title": "no url"}]})
    assert no_urls["ok"] is False
    assert "source" in no_urls["error"]


def test_runtime_detects_quick_answer_report_and_terminates(tmp_path: Path, monkeypatch) -> None:
    """The runtime's _load_report_from_store must detect a quick-answer report even with a non-empty frontier_hash.

    Regression risk: when no evidence exists, _frontier_hash([]) returns a non-empty
    sha256 of empty string. The old code filtered report artifacts by writer messages
    matching that hash, which would hide the quick-answer report (no writer message
    exists). The fix falls back to the unfiltered set when no writer messages match.
    """
    store = _build_store(tmp_path)
    handlers, _ = _build_handlers(store)
    monkeypatch.setattr("insightswarm.agents.researcher_tools.FetchUrlTool.run", _stub_fetch())

    handlers.quick_read({"url": "https://caac.gov.cn/news/2026-fuel"})
    handlers.finish_with_answer({
        "answer": "Test answer [1].",
        "sources": [{"url": "https://caac.gov.cn/news/2026-fuel", "title": "民航局", "summary": "official"}],
        "confidence": "medium",
    })

    run_id = handlers.task.run_id
    # Simulate a non-empty frontier_hash (as the runtime would set when no evidence exists).
    non_empty_hash = "abc123def456"
    report = _load_report_from_store(store, run_id, frontier_hash=non_empty_hash)
    assert report is not None
    assert "Test answer" in report["body"]
    assert "民航局" in report["body"]


def test_build_delivery_result_marks_quick_answer_as_completed(tmp_path: Path, monkeypatch) -> None:
    """_build_delivery_result must mark a quick-answer run as completed, not partial.

    Without the quick-answer detection, the default critic summary returns
    verdict="repair" (no citation-backed evidence), which would wrongly mark the
    run as report_partial. The quick_answer_ready message must override this.
    """
    from insightswarm.objective_runtime import RuntimeState

    store = _build_store(tmp_path)
    handlers, _ = _build_handlers(store)
    monkeypatch.setattr("insightswarm.agents.researcher_tools.FetchUrlTool.run", _stub_fetch())

    handlers.quick_read({"url": "https://caac.gov.cn/news/2026-fuel"})
    handlers.finish_with_answer({
        "answer": "Final answer [1].",
        "sources": [{"url": "https://caac.gov.cn/news/2026-fuel", "title": "民航局", "summary": "official"}],
        "confidence": "high",
    })

    run_id = handlers.task.run_id
    state = RuntimeState(
        run_id=run_id,
        run_root=tmp_path,
        question="test question",
        budget=None,  # type: ignore[arg-type]
        step_trace_path=tmp_path / "steps.jsonl",
        task_store=TaskStore(store),
        mailbox=Mailbox(store),
        artifact_store=ArtifactStore(store),
        board_store=BoardStore(store),
        last_progress_at=0.0,
    )
    result = _build_delivery_result(store, state, stop_reason="deliver_called")

    assert result.final_state == "completed"
    assert result.result_type == "report"
    assert result.report is not None
    assert "Final answer" in result.report["body"]
    assert result.critic["verdict"] == "quick_answer_unreviewed"
    assert result.critic["review_type"] == "quick_answer"


def test_latest_quick_answer_returns_none_without_quick_path(tmp_path: Path) -> None:
    """_latest_quick_answer must return None when no quick_answer_ready message exists."""
    store = _build_store(tmp_path)
    run_state = store.create_swarm_run_state(
        objective="normal question",
        budget={"max_steps": 12},
        phase="research",
    )
    assert _latest_quick_answer(store, run_state.run_id) is None
