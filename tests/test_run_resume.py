from __future__ import annotations

from pathlib import Path

from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.objective_runtime import DeliveryResult, ObjectiveBudget, resume_objective


def _make_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    init_db(db_path)
    return Store(db_path, tmp_path / "artifacts")


def _sentinel_result(run_id: str, question: str) -> DeliveryResult:
    return DeliveryResult(
        run_id=run_id,
        status="done",
        result_type="report",
        stop_reason="deliver_called",
        final_state="completed",
        question=question,
        steps=[],
    )


def test_resume_objective_recovers_question_and_reuses_run_id(tmp_path: Path, monkeypatch) -> None:
    store = _make_store(tmp_path)
    run_state = store.create_swarm_run_state(
        run_id="run_test_resume",
        objective="What is the fuel surcharge policy?",
        budget={"max_steps": 12},
        phase="research",
    )

    captured: dict = {}

    def _fake_run_objective(s, question, budget, run_root, **kwargs):
        captured["store"] = s
        captured["question"] = question
        captured["budget"] = budget
        captured["run_root"] = run_root
        captured["run_id"] = kwargs.get("run_id")
        captured["model_registry"] = kwargs.get("model_registry")
        return _sentinel_result(kwargs.get("run_id") or run_state.run_id, question)

    monkeypatch.setattr("insightswarm.objective_runtime.run_objective", _fake_run_objective)

    result = resume_objective(
        store,
        run_state.run_id,
        model_provider="fake",
        artifact_dir=tmp_path / "artifacts",
        max_steps=5,
        max_runtime_seconds=60.0,
        max_no_progress_seconds=30.0,
        max_drain_seconds=15.0,
    )

    assert result.run_id == run_state.run_id
    assert captured["run_id"] == run_state.run_id
    assert captured["question"] == "What is the fuel surcharge policy?"
    assert isinstance(captured["budget"], ObjectiveBudget)
    assert captured["budget"].max_steps == 5
    # resume must not create a new run state
    assert store.get_swarm_run_state(run_state.run_id).objective == "What is the fuel surcharge policy?"


def test_resume_objective_raises_for_unknown_run_id(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    try:
        resume_objective(
            store,
            "run_does_not_exist",
            model_provider="fake",
            artifact_dir=tmp_path / "artifacts",
        )
    except KeyError as exc:
        assert "run_does_not_exist" in str(exc)
    else:
        raise AssertionError("expected KeyError for unknown run_id")
