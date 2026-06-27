from __future__ import annotations

import json
from pathlib import Path

import pytest

from insightswarm.eval.cases import EvalCase
from insightswarm.eval.judge import (
    DIMENSIONS,
    WEIGHTS,
    JudgeResult,
    combine,
    judge_report,
)
from insightswarm.eval.quote_verify import CitationCheckSummary


REPO_RUBRIC = Path("evals") / "rubric.md"


def _case() -> EvalCase:
    return EvalCase(
        case_id="t",
        question="q",
        must_cover=["food delivery strategy", "instant retail growth"],
        must_not_claim=["precise per-venture profit figure"],
    )


def _citation_summary(total: int = 0, grounded: int = 0) -> CitationCheckSummary:
    summary = CitationCheckSummary()
    summary.total = total
    summary.exact = grounded
    return summary


def test_dimensions_match_rubric_md():
    """judge.py DIMENSIONS/WEIGHTS must stay in lock-step with evals/rubric.md.

    This guards the rubric/judge alignment bug: any drift between the rubric
    prose and the code's scoring dimensions/weights will fail here.
    """
    text = REPO_RUBRIC.read_text(encoding="utf-8")
    # Each rubric dimension heading must appear in DIMENSIONS.
    for dimension in DIMENSIONS:
        assert f"### {dimension}" in text, (
            f"rubric.md does not declare dimension '{dimension}'; "
            f"judge.py and the rubric are out of sync."
        )
    # Each weight declared in the rubric must match WEIGHTS.
    for dimension, weight in WEIGHTS.items():
        marker = f"### {dimension} (weight {weight:.2f})"
        assert marker in text, (
            f"rubric.md does not declare weight {weight:.2f} for '{dimension}'; "
            f"judge.py WEIGHTS must mirror the rubric."
        )
    # Weights must sum to 1.00 so combine() is a true weighted mean.
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_combine_uses_rubric_weights():
    # coverage 1.0, accuracy 1.0, citation_support 1.0, hallucination_avoidance
    # 1.0, conflict_handling 1.0 -> overall 1.0
    assert combine({dim: 1.0 for dim in DIMENSIONS}) == 1.0
    # All zeros -> 0.0
    assert combine({dim: 0.0 for dim in DIMENSIONS}) == 0.0
    # Only coverage and accuracy set -> weighted mean over those two weights
    # (0.30 + 0.30), so (1.0*0.30 + 0.0*0.30) / 0.60 = 0.5
    assert combine({"coverage": 1.0, "accuracy": 0.0}) == 0.5


def test_judge_report_marks_llm_success():
    class _Judge:
        provider = "openai"
        model = "gpt-x"

        def complete(self, messages, **kwargs):
            return _ModelResult(
                text=json.dumps({
                    "scores": {dim: 0.8 for dim in DIMENSIONS},
                    "rationale": {"coverage": "ok"},
                }),
            )

    result = judge_report(
        model_client=_Judge(),
        case=_case(),
        report_body="Some report body with content.",
        citation_summary=_citation_summary(),
    )
    assert isinstance(result, JudgeResult)
    assert result.judge_method == "llm"
    assert result.status == "ok"
    assert result.score_overall == pytest.approx(0.8)


def test_judge_report_marks_fallback_when_parse_fails():
    class _Judge:
        provider = "openai"
        model = "gpt-x"

        def complete(self, messages, **kwargs):
            return _ModelResult(text="not json at all")

    result = judge_report(
        model_client=_Judge(),
        case=_case(),
        report_body="Some report body with content.",
        citation_summary=_citation_summary(),
    )
    assert result.judge_method == "fallback"
    assert result.status == "unparseable"


def test_judge_report_marks_fallback_when_judge_raises():
    class _Judge:
        provider = "openai"
        model = "gpt-x"

        def complete(self, messages, **kwargs):
            raise RuntimeError("network down")

    result = judge_report(
        model_client=_Judge(),
        case=_case(),
        report_body="Some report body with content.",
        citation_summary=_citation_summary(),
    )
    assert result.judge_method == "fallback"
    assert result.status == "judge_exception"


def test_judge_report_marks_fallback_for_fake_provider():
    # The fake provider never returns a parseable judge JSON, so the runner
    # must not pretend it produced an LLM score.
    class _Fake:
        provider = "fake"
        model = "fake"

        def complete(self, messages, **kwargs):
            return _ModelResult(text="fake response for eval_judge")

    result = judge_report(
        model_client=_Fake(),
        case=_case(),
        report_body="Some report body with content.",
        citation_summary=_citation_summary(),
    )
    assert result.judge_method == "fallback"
    assert result.status == "fallback"


def test_judge_report_marks_no_report_when_body_empty():
    class _Judge:
        provider = "openai"
        model = "gpt-x"

        def complete(self, messages, **kwargs):
            raise AssertionError("should not be called when body is empty")

    result = judge_report(
        model_client=_Judge(),
        case=_case(),
        report_body="   ",
        citation_summary=_citation_summary(),
    )
    assert result.judge_method == "no_report"
    assert result.status == "no_report"
    assert result.score_overall == 0.0


def test_judge_report_to_row_exposes_judge_method():
    class _Judge:
        provider = "openai"
        model = "gpt-x"

        def complete(self, messages, **kwargs):
            return _ModelResult(text="not json")

    result = judge_report(
        model_client=_Judge(),
        case=_case(),
        report_body="Some report body with content.",
        citation_summary=_citation_summary(),
    )
    row = result.to_row()
    assert row["judge_method"] == "fallback"


class _ModelResult:
    def __init__(self, text: str):
        self.text = text
        self.json_data = None
        self.provider = "openai"
        self.model = "gpt-x"
        self.usage = {}
        self.latency_ms = 0
        self.raw_response = {}
        self.status = "ok"
        self.error = None
