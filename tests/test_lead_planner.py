from __future__ import annotations

from pathlib import Path
from typing import Any

from insightswarm.agents.lead import LeadWorker, _parse_plan_json, bootstrap_lead_objective
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.models.clients import ModelResult
from insightswarm.swarm_store import BoardStore, Mailbox, TaskStore


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


class _FakePlannerClient:
    """Returns a valid intent+strategy JSON so _plan_objective succeeds."""

    provider = "fake"
    model = "fake-planner-v1"

    def __init__(self, plan_json: str):
        self._plan_json = plan_json
        self.calls = 0

    def complete(self, messages, response_format=None, max_tokens=None, temperature=None, metadata=None):
        self.calls += 1
        return ModelResult(
            text=self._plan_json,
            json_data=None,
            provider=self.provider,
            model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            latency_ms=1,
            raw_response={"messages": messages, "metadata": metadata},
            status="ok",
        )


class _FailingPlannerClient:
    """Always returns a non-ok status to exercise the degradation path."""

    provider = "fake"
    model = "fake-planner-v1"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, response_format=None, max_tokens=None, temperature=None, metadata=None):
        self.calls += 1
        return ModelResult(
            text="",
            json_data=None,
            provider=self.provider,
            model=self.model,
            usage={},
            latency_ms=1,
            raw_response={},
            status="error",
            error="model unavailable",
        )


def test_parse_plan_json_handles_plain_json() -> None:
    text = '{"intent": "factual", "sub_questions": ["a", "b"]}'
    parsed = _parse_plan_json(text)
    assert parsed is not None
    assert parsed["intent"] == "factual"


def test_parse_plan_json_handles_markdown_fence() -> None:
    text = 'Here is the plan:\n```json\n{"intent": "comparison", "strategy": {"depth": "moderate"}}\n```\nDone.'
    parsed = _parse_plan_json(text)
    assert parsed is not None
    assert parsed["intent"] == "comparison"
    assert parsed["strategy"]["depth"] == "moderate"


def test_parse_plan_json_returns_none_on_garbage() -> None:
    assert _parse_plan_json("not json at all") is None
    assert _parse_plan_json("") is None


def test_lead_plan_objective_uses_model_sub_questions(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="compare GPT-4 vs Claude",
        budget={"max_steps": 12},
        phase="discovery",
    )
    bootstrap_lead_objective(
        task_store,
        mailbox,
        run_id=run_state.run_id,
        question="compare GPT-4 vs Claude",
    )
    plan_json = (
        '{"intent": "comparison", "intent_reason": "asks to compare two entities", '
        '"sub_questions": ["GPT-4 capabilities", "Claude capabilities", "pricing comparison"], '
        '"strategy": {"target_source_count": 4, "freshness_window_days": null, '
        '"evidence_type_priority": ["official_statement", "analysis"], "depth": "moderate"}}'
    )
    worker = LeadWorker(task_store, mailbox, board_store, model_client=_FakePlannerClient(plan_json))
    result = worker.run_once(run_state.run_id)
    assert result is not None

    # The plan payload should carry the intent and strategy.
    plans = board_store.store.list_swarm_board_items(run_state.run_id, kind="plan")
    assert plans, "expected at least one plan"
    plan_payload = plans[0].payload
    assert plan_payload.get("planned") is True
    assert plan_payload.get("intent") == "comparison"
    assert plan_payload["strategy"]["target_source_count"] == 4

    # Sub-questions should come from the model plan, not the mechanical split.
    sub_questions = board_store.store.list_swarm_board_items(run_state.run_id, kind="question")
    titles = [q.title for q in sub_questions if q.payload.get("question_type") == "subquestion"]
    assert "GPT-4 capabilities" in titles
    assert "pricing comparison" in titles


def test_lead_plan_objective_degrades_to_mechanical_split_on_model_failure(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="some question",
        budget={"max_steps": 12},
        phase="discovery",
    )
    bootstrap_lead_objective(
        task_store,
        mailbox,
        run_id=run_state.run_id,
        question="some question",
        sub_questions=["fallback sub q"],
    )
    worker = LeadWorker(task_store, mailbox, board_store, model_client=_FailingPlannerClient())
    result = worker.run_once(run_state.run_id)
    assert result is not None

    plans = board_store.store.list_swarm_board_items(run_state.run_id, kind="plan")
    plan_payload = plans[0].payload
    # Degraded path: planned=False, no intent.
    assert plan_payload.get("planned") is False
    assert plan_payload.get("intent") is None
    # Mechanical split uses the provided sub_questions.
    sub_questions = board_store.store.list_swarm_board_items(run_state.run_id, kind="question")
    titles = [q.title for q in sub_questions if q.payload.get("question_type") == "subquestion"]
    assert "fallback sub q" in titles


def test_lead_plan_objective_degrades_when_no_model_client(tmp_path: Path) -> None:
    """Without a model_client, Lead must behave exactly as before (mechanical split)."""
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="no model",
        budget={"max_steps": 12},
        phase="discovery",
    )
    bootstrap_lead_objective(
        task_store,
        mailbox,
        run_id=run_state.run_id,
        question="no model",
        sub_questions=["mechanical only"],
    )
    worker = LeadWorker(task_store, mailbox, board_store)  # no model_client
    result = worker.run_once(run_state.run_id)
    assert result is not None

    plans = board_store.store.list_swarm_board_items(run_state.run_id, kind="plan")
    plan_payload = plans[0].payload
    assert plan_payload.get("planned") is False


def test_lead_plan_objective_invalid_intent_falls_back_to_factual(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="weird intent",
        budget={"max_steps": 12},
        phase="discovery",
    )
    bootstrap_lead_objective(
        task_store,
        mailbox,
        run_id=run_state.run_id,
        question="weird intent",
    )
    plan_json = '{"intent": "nonsense_intent", "sub_questions": ["q1"], "strategy": {"depth": "shallow"}}'
    worker = LeadWorker(task_store, mailbox, board_store, model_client=_FakePlannerClient(plan_json))
    result = worker.run_once(run_state.run_id)
    assert result is not None

    plans = board_store.store.list_swarm_board_items(run_state.run_id, kind="plan")
    plan_payload = plans[0].payload
    assert plan_payload.get("planned") is True
    # Invalid intent must be coerced to "factual", not propagated.
    assert plan_payload.get("intent") == "factual"
