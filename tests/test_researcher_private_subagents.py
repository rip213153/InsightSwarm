from __future__ import annotations

from pathlib import Path
from typing import Any

from insightswarm.agents.researcher_tools import ResearcherToolHandlers, ResearcherToolState
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore
from insightswarm.tools.core import ToolResult


class _ModelResult:
    status = "ok"
    text = ""

    def __init__(self, json_data: dict[str, Any]):
        self.json_data = json_data


class _StubModelClient:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, response_format=None, max_tokens=None, temperature=None, metadata=None, tools=None, tool_choice=None):
        self.calls.append({"messages": messages, "metadata": metadata or {}, "response_format": response_format})
        return _ModelResult(self.responses.pop(0))


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def test_researcher_subagents_are_private_and_do_not_write_shared_store(tmp_path: Path, monkeypatch) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Research a broad strategy question",
        budget={"max_steps": 20},
        phase="research",
    )
    task = task_store.create(
        run_state.run_id,
        kind="research_subquestion",
        status="leased",
        owner_role="researcher",
        inputs={"question": "京东下一步可能想做什么板块？"},
        created_by="lead",
    )

    def _search(self, tool_input: dict[str, Any], context=None) -> ToolResult:
        query_slug = str(tool_input["query"]).replace(" ", "-")
        return ToolResult(
            "ok",
            data={
                "results": [
                    {
                        "title": f"{tool_input['query']} official source",
                        "url": f"https://example.com/{query_slug}",
                        "snippet": "official strategic source",
                    }
                ]
            },
        )

    def _fetch(self, tool_input: dict[str, Any], context=None) -> ToolResult:
        return ToolResult(
            "ok",
            data={
                "source_url": tool_input["url"],
                "url": tool_input["url"],
                "title": "Official strategic source",
                "text": "京东 即时零售 供应链 技术 战略 " * 120,
                "html": "<html><title>Official strategic source</title></html>",
            },
        )

    monkeypatch.setattr("insightswarm.agents.researcher_tools.SearchTool.run", _search)
    monkeypatch.setattr("insightswarm.agents.researcher_tools.FetchUrlTool.run", _fetch)

    state = ResearcherToolState()
    handlers = ResearcherToolHandlers(
        task=task,
        task_store=task_store,
        mailbox=mailbox,
        artifact_store=artifact_store,
        board_store=BoardStore(store),
        state=state,
    )
    handlers.read_task({})

    result = handlers.spawn_research_subagents(
        {
            "why_parallel_needed": "The question has several plausible strategic branches.",
            "subtasks": [
                {"question": "京东即时零售战略", "search_goal": "official_source"},
                {"question": "京东供应链技术战略", "search_goal": "official_source"},
            ],
        }
    )

    assert result["ok"] is True
    assert result["shared_storage_written"] is False


def test_researcher_dedupes_repeated_normalized_url_fetch_and_publish(tmp_path: Path, monkeypatch) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="Dedupe URL test",
        budget={"max_steps": 20},
        phase="research",
    )
    task = task_store.create(
        run_state.run_id,
        kind="research_subquestion",
        status="leased",
        owner_role="researcher",
        inputs={"question": "What is new?"},
        created_by="lead",
    )

    fetch_calls: list[str] = []

    def _fetch(self, tool_input: dict[str, Any], context=None) -> ToolResult:
        fetch_calls.append(tool_input["url"])
        return ToolResult(
            "ok",
            data={
                "source_url": tool_input["url"],
                "url": tool_input["url"],
                "title": "Official source",
                "text": "Long usable text " * 100,
                "html": "<html><title>Official source</title></html>",
            },
        )

    monkeypatch.setattr("insightswarm.agents.researcher_tools.FetchUrlTool.run", _fetch)

    state = ResearcherToolState()
    handlers = ResearcherToolHandlers(
        task=task,
        task_store=task_store,
        mailbox=mailbox,
        artifact_store=artifact_store,
        board_store=board_store,
        state=state,
    )

    handlers.read_task({})
    first = handlers.fetch_source({"url": "https://example.com/path/", "reason": "snippet_insufficient"})
    second = handlers.fetch_source({"url": "https://example.com/path", "reason": "snippet_insufficient"})
    publish = handlers.publish_raw_source({"document_urls": ["https://example.com/path"], "why_ready": "usable"})
    republish = handlers.publish_raw_source({"document_urls": ["https://example.com/path/"], "why_ready": "usable again"})

    assert first["ok"] is True
    assert second["deduped"] is True
    assert publish["ok"] is True
    assert republish["deduped"] is True
    assert len(fetch_calls) == 1


def test_researcher_subagents_cap_at_three(tmp_path: Path, monkeypatch) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Research a broad strategy question",
        budget={"max_steps": 20},
        phase="research",
    )
    task = task_store.create(
        run_state.run_id,
        kind="research_subquestion",
        status="leased",
        owner_role="researcher",
        inputs={"question": "OpenAI 下一步想干什么？"},
        created_by="lead",
    )

    def _search(self, tool_input: dict[str, Any], context=None) -> ToolResult:
        return ToolResult(
            "ok",
            data={"results": [{"title": tool_input["query"], "url": f"https://example.com/{tool_input['query']}", "snippet": "source"}]},
        )

    def _fetch(self, tool_input: dict[str, Any], context=None) -> ToolResult:
        return ToolResult(
            "ok",
            data={"source_url": tool_input["url"], "title": "Source", "text": "usable source text " * 80, "html": ""},
        )

    monkeypatch.setattr("insightswarm.agents.researcher_tools.SearchTool.run", _search)
    monkeypatch.setattr("insightswarm.agents.researcher_tools.FetchUrlTool.run", _fetch)

    state = ResearcherToolState()
    handlers = ResearcherToolHandlers(
        task=task,
        task_store=task_store,
        mailbox=mailbox,
        artifact_store=artifact_store,
        board_store=BoardStore(store),
        state=state,
    )
    handlers.read_task({})

    result = handlers.spawn_research_subagents(
        {
            "why_parallel_needed": "Compare several paths privately.",
            "subtasks": [
                {"question": "产品路线"},
                {"question": "企业商业化"},
                {"question": "基础设施"},
                {"question": "安全治理"},
            ],
        }
    )

    assert result["ok"] is True
    assert len(result["findings"]) == 3
    assert [item["question"] for item in result["findings"]] == ["产品路线", "企业商业化", "基础设施"]


def test_researcher_subagent_uses_independent_model_loop(tmp_path: Path, monkeypatch) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    run_state = store.create_swarm_run_state(
        objective="Research with private child context",
        budget={"max_steps": 20},
        phase="research",
    )
    task = task_store.create(
        run_state.run_id,
        kind="research_subquestion",
        status="leased",
        owner_role="researcher",
        inputs={"question": "OpenAI 下一步想干什么？"},
        created_by="lead",
    )

    def _search(self, tool_input: dict[str, Any], context=None) -> ToolResult:
        return ToolResult(
            "ok",
            data={"results": [{"title": "OpenAI product roadmap", "url": "https://example.com/openai-roadmap", "snippet": "primary source"}]},
        )

    monkeypatch.setattr("insightswarm.agents.researcher_tools.SearchTool.run", _search)
    model_client = _StubModelClient(
        [
            {
                "assistant_text": "I need a primary roadmap source first.",
                "private_state": {"current_understanding": "Need product roadmap source.", "gap": "source URL", "plan": "search"},
                "tool_call": {"name": "search_web", "input": {"query": "OpenAI product roadmap official", "limit": 3}},
                "stop_reason": None,
            },
            {
                "assistant_text": "I found a candidate and can return it to the parent.",
                "private_state": {"current_understanding": "Candidate source found.", "gap": "parent should verify", "plan": "finish"},
                "tool_call": {
                    "name": "finish_subagent",
                    "input": {
                        "status": "complete",
                        "summary": "Found an OpenAI product roadmap candidate.",
                        "candidate_urls": ["https://example.com/openai-roadmap"],
                        "recommended_next_step": "Parent Researcher should fetch and rank this source.",
                    },
                },
                "stop_reason": None,
            },
        ]
    )
    state = ResearcherToolState()
    handlers = ResearcherToolHandlers(
        task=task,
        task_store=task_store,
        mailbox=mailbox,
        artifact_store=artifact_store,
        board_store=BoardStore(store),
        state=state,
        model_client=model_client,
    )
    handlers.read_task({})

    result = handlers.spawn_research_subagents(
        {
            "why_parallel_needed": "Use one private child context to test the model loop.",
            "subtasks": [{"question": "OpenAI 产品路线"}],
        }
    )

    assert result["ok"] is True
    assert result["findings"][0]["candidate_urls"] == ["https://example.com/openai-roadmap"]
    assert [call["metadata"]["role"] for call in model_client.calls] == ["research_subagent_loop", "research_subagent_loop"]
    assert store.list_swarm_messages(run_state.run_id) == []
    assert store.list_swarm_artifacts(run_state.run_id) == []
