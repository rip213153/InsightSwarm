"""Eval orchestration: run cases over N epochs, score each, aggregate.

A single eval *run* executes every case ``repeat`` times (epochs). Each epoch
is a full, independent swarm run -- that is where the non-determinism lives, so
re-judging one transcript would only measure judge noise, not agent variance.
For every epoch we:

1. drive the swarm to a delivery result,
2. aggregate run-level telemetry (tokens, latency, citations) from the audit log,
3. deterministically verify each citation quote against fetched source text,
4. ask an independent judge model for dimension scores,

then persist the epoch and, once all epochs for a case are in, the per-case
aggregate (mean / std / stderr). The caller decides significance via
``insightswarm.eval.stats``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from insightswarm.db.store import Store
from insightswarm.eval.cases import EvalCase, load_cases
from insightswarm.eval.judge import JudgeResult, judge_report
from insightswarm.eval.quote_verify import verify_citations
from insightswarm.eval.stats import summarize_split
from insightswarm.eval.store import EvalStore
from insightswarm.eval.telemetry import (
    collect_report_citations,
    collect_run_telemetry,
    collect_source_corpus,
)


SwarmRunner = Callable[[EvalCase], Any]
"""Drives one swarm run for a case and returns its DeliveryResult-like object.

Injected so the runner can be exercised offline with a stub. The production
implementation is ``build_default_swarm_runner``.
"""


@dataclass
class EpochOutcome:
    case_id: str
    epoch_idx: int
    swarm_run_id: str
    result_type: str
    score_overall: float
    score_dims: dict[str, float]
    citation_summary: dict[str, Any]
    grounded_ratio: float
    latency_ms: int
    token_total: int
    status: str
    error: str | None
    judge_rationale: str
    citation_results: list[dict[str, Any]]
    judge_method: str = "llm"


def build_default_swarm_runner(
    store: Store,
    *,
    artifact_dir: Path,
    model_provider: str,
    model_config_path: str | Path | None = None,
    max_steps: int = 12,
    max_runtime_seconds: float = 1800.0,
    browser_backend: str | None = None,
    browser_cdp_url: str | None = None,
) -> SwarmRunner:
    """Production swarm runner: one ``create_and_run_objective`` per epoch."""
    from insightswarm.objective_runtime import create_and_run_objective

    def _run(case: EvalCase) -> Any:
        return create_and_run_objective(
            store,
            name=f"eval-{case.case_id}",
            query=case.question,
            model_provider=model_provider,
            model_config_path=model_config_path,
            artifact_dir=artifact_dir,
            max_steps=max_steps,
            max_runtime_seconds=max_runtime_seconds,
            browser_backend=browser_backend,
            browser_cdp_url=browser_cdp_url,
            input_files=list(case.input_files) or None,
        )

    return _run


def run_eval(
    *,
    store: Store,
    eval_store: EvalStore,
    cases_dir: str | Path,
    swarm_runner: SwarmRunner,
    judge_client: Any,
    target_provider: str,
    repeat: int = 1,
    suite: str | None = None,
    difficulty: str | None = None,
    case_ids: list[str] | None = None,
    git_rev: str | None = None,
    notes: str | None = None,
) -> str:
    """Execute a full eval run and return its ``eval_run_id``.

    ``swarm_runner`` and ``judge_client`` are injected so the orchestration is
    testable without live models. ``repeat`` is the epoch count per case.
    """
    cases = load_cases(cases_dir, suite=suite, difficulty=difficulty)
    if case_ids:
        wanted = set(case_ids)
        cases = [case for case in cases if case.case_id in wanted]
    if not cases:
        raise ValueError("no eval cases selected")

    repeat = max(1, int(repeat))
    suite_name = suite or "all"
    eval_run_id = eval_store.create_eval_run(
        suite=suite_name,
        judge_provider=getattr(judge_client, "provider", "none"),
        judge_model=getattr(judge_client, "model", None),
        target_provider=target_provider,
        repeat_n=repeat,
        git_rev=git_rev,
        notes=notes,
    )

    for case in cases:
        for epoch_idx in range(repeat):
            outcome = _run_one_epoch(
                store=store,
                case=case,
                epoch_idx=epoch_idx,
                swarm_runner=swarm_runner,
                judge_client=judge_client,
            )
            _record_epoch(eval_store, eval_run_id, outcome)

        _recompute_case_agg(eval_store, eval_run_id, case.case_id)

    eval_store.finish_eval_run(eval_run_id)
    return eval_run_id


def resume_eval(
    *,
    store: Store,
    eval_store: EvalStore,
    eval_run_id: str,
    cases_dir: str | Path,
    swarm_runner: SwarmRunner,
    judge_client: Any,
    suite: str | None = None,
    difficulty: str | None = None,
    case_ids: list[str] | None = None,
) -> str:
    """Resume an interrupted eval run by filling missing case/epoch rows."""
    eval_run = eval_store.get_eval_run(eval_run_id)
    if eval_run is None:
        raise ValueError(f"eval run not found: {eval_run_id}")
    repeat = max(1, int(eval_run.get("repeat_n") or 1))
    cases = load_cases(cases_dir, suite=suite or eval_run.get("suite"), difficulty=difficulty)
    if case_ids:
        wanted = set(case_ids)
        cases = [case for case in cases if case.case_id in wanted]
    if not cases:
        raise ValueError("no eval cases selected")

    existing = {
        (str(row["case_id"]), int(row["epoch_idx"]))
        for row in eval_store.list_epochs(eval_run_id)
    }

    for case in cases:
        for epoch_idx in range(repeat):
            key = (case.case_id, epoch_idx)
            if key in existing:
                continue
            outcome = _run_one_epoch(
                store=store,
                case=case,
                epoch_idx=epoch_idx,
                swarm_runner=swarm_runner,
                judge_client=judge_client,
            )
            _record_epoch(eval_store, eval_run_id, outcome)
            existing.add(key)
        _recompute_case_agg(eval_store, eval_run_id, case.case_id)

    expected = {(case.case_id, idx) for case in cases for idx in range(repeat)}
    status = "done" if expected.issubset(existing) else "running"
    if status == "done":
        eval_store.finish_eval_run(eval_run_id)
    return eval_run_id


def _record_epoch(eval_store: EvalStore, eval_run_id: str, outcome: EpochOutcome) -> str:
    epoch_id = eval_store.record_epoch(
        eval_run_id=eval_run_id,
        case_id=outcome.case_id,
        epoch_idx=outcome.epoch_idx,
        swarm_run_id=outcome.swarm_run_id or None,
        result_type=outcome.result_type,
        score_overall=outcome.score_overall,
        score_dims=outcome.score_dims,
        citation_summary=outcome.citation_summary,
        grounded_ratio=outcome.grounded_ratio,
        latency_ms=outcome.latency_ms,
        token_total=outcome.token_total,
        status=outcome.status,
        error=outcome.error,
        judge_rationale=outcome.judge_rationale,
        judge_method=outcome.judge_method,
    )
    if outcome.citation_results:
        eval_store.record_citation_checks(epoch_id, outcome.citation_results)
    return epoch_id


def _recompute_case_agg(eval_store: EvalStore, eval_run_id: str, case_id: str) -> None:
    rows = [
        row for row in eval_store.list_epochs(eval_run_id)
        if str(row["case_id"]) == case_id
    ]
    # Partition by judge_method so fallback / no_report scores never pollute
    # the headline mean. The LLM subset is the measurement; the others are
    # reported alongside for transparency.
    llm_scores: list[float] = []
    fallback_scores: list[float] = []
    n_no_report = 0
    grounded: list[float] = []
    for row in rows:
        method = str(row.get("judge_method") or "llm")
        score = float(row["score_overall"] or 0.0)
        if method == "llm":
            llm_scores.append(score)
        elif method == "no_report":
            n_no_report += 1
        else:  # "fallback" or any future non-LLM method
            fallback_scores.append(score)
        grounded.append(float(row["grounded_ratio"] or 0.0))

    split = summarize_split(llm_scores, fallback_scores, n_no_report=n_no_report)
    mean_grounded = sum(grounded) / len(grounded) if grounded else 0.0
    eval_store.upsert_case_agg(
        eval_run_id=eval_run_id,
        case_id=case_id,
        n_epochs=len(rows),
        n_llm=split.llm.n,
        n_fallback=split.fallback.n,
        n_no_report=split.n_no_report,
        mean=split.llm.mean,
        std=split.llm.std,
        stderr=split.llm.stderr,
        min_score=split.llm.min_score,
        max_score=split.llm.max_score,
        fallback_mean=split.fallback.mean if split.fallback.n else None,
        mean_grounded_ratio=round(mean_grounded, 4),
    )


def _run_one_epoch(
    *,
    store: Store,
    case: EvalCase,
    epoch_idx: int,
    swarm_runner: SwarmRunner,
    judge_client: Any,
) -> EpochOutcome:
    try:
        delivery = swarm_runner(case)
    except Exception as exc:  # one failed epoch must not abort the suite
        return EpochOutcome(
            case_id=case.case_id,
            epoch_idx=epoch_idx,
            swarm_run_id="",
            result_type="error",
            score_overall=0.0,
            score_dims={},
            citation_summary={},
            grounded_ratio=0.0,
            latency_ms=0,
            token_total=0,
            status="error",
            error=f"{type(exc).__name__}: {exc}"[:500],
            judge_rationale="swarm run raised before delivery",
            citation_results=[],
            judge_method="no_report",
        )

    payload = delivery.to_dict() if hasattr(delivery, "to_dict") else dict(delivery)
    run_id = str(payload.get("run_id") or "")
    result_type = str(payload.get("result_type") or "unknown")
    report_body = ""
    report = payload.get("report")
    if isinstance(report, dict):
        report_body = str(report.get("body") or "")

    telemetry = collect_run_telemetry(store, run_id) if run_id else None
    citations = collect_report_citations(store, run_id) if run_id else []
    source_by_url = collect_source_corpus(store, run_id) if run_id else {}
    corpus = "\n\n".join(source_by_url.values())
    citation_summary = verify_citations(citations, source_by_url, corpus=corpus)

    judge_result: JudgeResult = judge_report(
        model_client=judge_client,
        case=case,
        report_body=report_body,
        citation_summary=citation_summary,
        run_id=run_id or None,
    )

    rationale_text = "; ".join(
        f"{dimension}: {text}" for dimension, text in judge_result.rationale.items() if text
    )[:1000]

    return EpochOutcome(
        case_id=case.case_id,
        epoch_idx=epoch_idx,
        swarm_run_id=run_id,
        result_type=result_type,
        score_overall=judge_result.score_overall,
        score_dims=judge_result.scores,
        citation_summary=citation_summary.to_dict(),
        grounded_ratio=citation_summary.grounded_ratio,
        latency_ms=int(telemetry.latency_ms_total) if telemetry else 0,
        token_total=int(telemetry.token_total) if telemetry else 0,
        status=str(payload.get("final_state") or judge_result.status or "unknown"),
        error=judge_result.error,
        judge_rationale=rationale_text,
        citation_results=list(citation_summary.results),
        judge_method=judge_result.judge_method,
    )
