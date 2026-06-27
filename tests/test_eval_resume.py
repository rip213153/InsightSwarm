from __future__ import annotations

import json

from insightswarm.eval.cases import EvalCase
from insightswarm.eval.runner import build_default_swarm_runner, resume_eval
from insightswarm.eval.store import EvalStore
from insightswarm.models.clients import ModelResult


class _Store:
    def __init__(self):
        self.conn = _Conn()

    def list_swarm_evidence(self, run_id):
        return []

    def list_swarm_artifacts(self, run_id):
        return []


class _Conn:
    def execute(self, *args, **kwargs):
        return self

    def fetchall(self):
        return []


class _Delivery:
    def __init__(self, run_id: str):
        self.run_id = run_id

    def to_dict(self):
        return {
            "run_id": self.run_id,
            "result_type": "report",
            "final_state": "completed",
            "report": {"body": "Recovered report body."},
        }


class _Judge:
    provider = "fake"
    model = "fake"

    def complete(self, *args, **kwargs):
        return ModelResult(
            text=json.dumps({
                "scores": {
                    "coverage": 1,
                    "accuracy": 1,
                    "citation_support": 1,
                    "hallucination_avoidance": 1,
                    "conflict_handling": 1,
                },
                "rationale": {},
            }),
            json_data=None,
            provider=self.provider,
            model=self.model,
            usage={},
            latency_ms=0,
            raw_response={},
            status="ok",
        )


def test_resume_eval_runs_only_missing_epochs(tmp_path):
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "a.json").write_text(
        json.dumps({"case_id": "a", "question": "A?", "suite": "golden"}),
        encoding="utf-8",
    )
    eval_store = EvalStore(tmp_path / "eval.db")
    eval_run_id = eval_store.create_eval_run(
        suite="golden",
        judge_provider="fake",
        judge_model="fake",
        target_provider="fake",
        repeat_n=3,
    )
    eval_store.record_epoch(
        eval_run_id=eval_run_id,
        case_id="a",
        epoch_idx=0,
        swarm_run_id="existing",
        result_type="report",
        score_overall=0.5,
        score_dims={"coverage": 0.5},
        citation_summary={},
        grounded_ratio=0.0,
        latency_ms=0,
        token_total=0,
        status="completed",
        error=None,
        judge_rationale="existing",
    )
    calls = []

    def _runner(case):
        calls.append(case.case_id)
        return _Delivery(f"resumed-{len(calls)}")

    resume_eval(
        store=_Store(),
        eval_store=eval_store,
        eval_run_id=eval_run_id,
        cases_dir=cases_dir,
        swarm_runner=_runner,
        judge_client=_Judge(),
        suite="golden",
    )

    epochs = eval_store.list_epochs(eval_run_id)
    agg = eval_store.list_case_aggs(eval_run_id)[0]
    run = eval_store.get_eval_run(eval_run_id)
    assert calls == ["a", "a"]
    assert len(epochs) == 3
    assert agg["n_epochs"] == 3
    assert run["status"] == "done"


def test_default_swarm_runner_forwards_browser_backend(monkeypatch, tmp_path):
    captured = {}

    def _fake_create_and_run_objective(*args, **kwargs):
        captured.update(kwargs)
        return _Delivery("run-visible")

    monkeypatch.setattr(
        "insightswarm.objective_runtime.create_and_run_objective",
        _fake_create_and_run_objective,
    )

    runner = build_default_swarm_runner(
        _Store(),
        artifact_dir=tmp_path / "artifacts",
        model_provider="default",
        model_config_path=tmp_path / "config.models.json",
        max_steps=3,
        max_runtime_seconds=45.0,
        browser_backend="visible",
        browser_cdp_url="http://127.0.0.1:9222",
    )

    runner(EvalCase(case_id="visible-browser", question="Open this page"))

    assert captured["browser_backend"] == "visible"
    assert captured["browser_cdp_url"] == "http://127.0.0.1:9222"


def test_recompute_case_agg_excludes_fallback_from_main_mean(tmp_path):
    """Fallback / no_report epochs must NOT pollute the LLM-judged mean.

    Seeds three epochs for one case:
      - epoch 0: LLM-judged, score 0.9
      - epoch 1: fallback (judge returned unparseable output), score 0.0
      - epoch 2: no_report (swarm delivered nothing), score 0.0

    The headline mean must be 0.9 (only the LLM epoch), not 0.3 (the average
    of all three). The fallback mean is reported separately, and the n_*
    counters break the total down by judge_method.
    """
    from insightswarm.eval.runner import _recompute_case_agg

    eval_store = EvalStore(tmp_path / "eval.db")
    eval_run_id = eval_store.create_eval_run(
        suite="golden",
        judge_provider="rival",
        judge_model="rival-x",
        target_provider="openai",
        repeat_n=3,
    )
    for idx, (method, score) in enumerate([
        ("llm", 0.9),
        ("fallback", 0.0),
        ("no_report", 0.0),
    ]):
        eval_store.record_epoch(
            eval_run_id=eval_run_id,
            case_id="c",
            epoch_idx=idx,
            swarm_run_id=f"run-{idx}",
            result_type="report",
            score_overall=score,
            score_dims={"coverage": score},
            citation_summary={},
            grounded_ratio=0.0,
            latency_ms=0,
            token_total=0,
            status="ok",
            error=None,
            judge_rationale="",
            judge_method=method,
        )

    _recompute_case_agg(eval_store, eval_run_id, "c")
    agg = eval_store.list_case_aggs(eval_run_id)[0]

    assert agg["n_epochs"] == 3
    assert agg["n_llm"] == 1
    assert agg["n_fallback"] == 1
    assert agg["n_no_report"] == 1
    assert abs(float(agg["mean"]) - 0.9) < 1e-9
    assert agg["fallback_mean"] == 0.0


def test_judge_method_persisted_on_epoch(tmp_path):
    """judge_method flows from judge_report through the runner into eval_epochs."""
    from insightswarm.eval.runner import run_eval

    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    (cases_dir / "a.json").write_text(
        json.dumps({"case_id": "a", "question": "A?", "suite": "golden"}),
        encoding="utf-8",
    )
    eval_store = EvalStore(tmp_path / "eval.db")

    class _FailingJudge:
        provider = "rival"
        model = "rival-x"

        def complete(self, *args, **kwargs):
            # Unparseable output -> runner records judge_method="fallback"
            return ModelResult(
                text="not json",
                json_data=None,
                provider=self.provider,
                model=self.model,
                usage={},
                latency_ms=0,
                raw_response={},
                status="ok",
            )

    def _runner(case):
        return _Delivery("run-1")

    run_eval(
        store=_Store(),
        eval_store=eval_store,
        cases_dir=cases_dir,
        swarm_runner=_runner,
        judge_client=_FailingJudge(),
        target_provider="openai",
        repeat=1,
        suite="golden",
    )

    epochs = eval_store.conn.execute(
        "SELECT judge_method, status FROM eval_epochs"
    ).fetchall()
    assert len(epochs) == 1
    # judge_method is the load-bearing field: it records that this epoch's
    # score came from the deterministic fallback, not the LLM judge. The
    # epoch-level ``status`` reflects the swarm delivery's final_state
    # ("completed"), which is independent of how the score was produced.
    assert epochs[0]["judge_method"] == "fallback"
