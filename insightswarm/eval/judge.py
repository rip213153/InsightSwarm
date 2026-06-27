from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from insightswarm.eval.cases import EvalCase
from insightswarm.eval.quote_verify import CitationCheckSummary


# Scoring dimensions and weights. Each is scored 0.0-1.0 by the judge, then
# combined by WEIGHTS. These MUST stay in lock-step with ``evals/rubric.md``;
# any change here is a rubric change and must be reflected there (and vice
# versa). Weights sum to 1.00.
DIMENSIONS = (
    "coverage",                 # does the report address the must-cover points
    "accuracy",                 # are the factual claims correct / non-contradictory
    "citation_support",         # do citations actually support the claims they back
    "hallucination_avoidance",  # are hard assertions backed by citations (1.0 = no fabrication)
    "conflict_handling",        # are caveats / conflicts / uncertainty handled honestly
)

# Weights sourced from ``evals/rubric.md`` (coverage 0.30, accuracy 0.30,
# citation_support 0.20, hallucination_avoidance 0.10, conflict_handling 0.10).
WEIGHTS = {
    "coverage": 0.30,
    "accuracy": 0.30,
    "citation_support": 0.20,
    "hallucination_avoidance": 0.10,
    "conflict_handling": 0.10,
}


@dataclass(frozen=True)
class JudgeResult:
    score_overall: float
    scores: dict[str, float]
    rationale: dict[str, str]
    judge_provider: str
    judge_model: str
    status: str = "ok"
    error: str | None = None
    raw_text: str = ""
    # ``judge_method`` distinguishes how the score was produced so downstream
    # stats can exclude non-LLM scores from the main mean. Values:
    #   "llm"        - the judge model returned a parseable score
    #   "fallback"   - the deterministic heuristic was used because the judge
    #                  model was unavailable, fake, or returned unparseable output
    #   "no_report"  - the swarm delivered no report body; score is zero by rule
    judge_method: str = "llm"

    def to_row(self) -> dict[str, Any]:
        return {
            "score_overall": self.score_overall,
            "score_dims_json": self.scores,
            "rationale_json": self.rationale,
            "judge_provider": self.judge_provider,
            "judge_model": self.judge_model,
            "status": self.status,
            "error": self.error,
            "judge_method": self.judge_method,
        }


def _rubric_text() -> str:
    path = Path(__file__).resolve().parent.parent.parent / "evals" / "rubric.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "Score each dimension from 0.0 to 1.0. Be strict and evidence-driven."


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def combine(scores: dict[str, float]) -> float:
    total = 0.0
    weight_sum = 0.0
    for dimension, weight in WEIGHTS.items():
        if dimension in scores:
            total += _clamp(scores[dimension]) * weight
            weight_sum += weight
    if weight_sum == 0:
        return 0.0
    return round(total / weight_sum, 4)


def build_judge_payload(
    *,
    case: EvalCase,
    report_body: str,
    citation_summary: CitationCheckSummary,
) -> dict[str, Any]:
    """Assemble the structured payload handed to the judge model.

    The citation verification numbers are computed deterministically (not by the
    judge) and handed in as ground-truth signal, so the judge reasons about
    whether the *supported* claims are well-cited rather than re-checking quotes.
    """
    return {
        "question": case.question,
        "must_cover_points": case.must_cover,
        "must_not_claim": case.must_not_claim,
        "report": report_body,
        "deterministic_citation_check": {
            "total_citations": citation_summary.total,
            "exact_matches": citation_summary.exact,
            "normalized_matches": citation_summary.normalized,
            "fuzzy_matches": citation_summary.fuzzy,
            "unmatched_quotes": citation_summary.unmatched,
            "grounded_ratio": citation_summary.grounded_ratio,
        },
    }


def _system_prompt() -> str:
    return (
        "You are a strict evaluation judge for research reports. "
        "Score the report against the rubric using ONLY the provided material. "
        "A claim is 'unmatched' if its quote was not found in the source text by the "
        "deterministic checker; treat unmatched citations as unsupported. "
        "Return STRICT JSON only, no prose, in this exact shape:\n"
        '{"scores": {"coverage": 0.0, "accuracy": 0.0, "citation_support": 0.0, '
        '"hallucination_avoidance": 0.0, "conflict_handling": 0.0}, '
        '"rationale": {"coverage": "...", "accuracy": "...", "citation_support": "...", '
        '"hallucination_avoidance": "...", "conflict_handling": "..."}}\n'
        "Each score is a float in [0.0, 1.0]. For hallucination_avoidance, 1.0 means "
        "every hard claim is supported by a citation and 0.0 means many hard claims "
        "are unsupported or fabricated.\n\n"
        + _rubric_text()
    )


def _parse_judge_json(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    if not stripped:
        return None
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json\n", "", 1).replace("JSON\n", "", 1)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _deterministic_fallback(
    *,
    case: EvalCase,
    report_body: str,
    citation_summary: CitationCheckSummary,
) -> dict[str, float]:
    """Used when the judge model is unavailable or returns unparseable output.

    This is a coarse, transparent heuristic — NOT a substitute for the LLM judge.
    It keeps offline tests deterministic and gives a non-crashing baseline.
    """
    body_lower = report_body.lower()
    covered = sum(1 for point in case.must_cover if _point_hit(point, body_lower))
    coverage = covered / len(case.must_cover) if case.must_cover else 0.0
    violated = any(_point_hit(point, body_lower) for point in case.must_not_claim)
    citation_support = citation_summary.grounded_ratio if citation_summary.total else 0.5
    hallucination_avoidance = citation_summary.grounded_ratio if citation_summary.total else 0.5
    return {
        "coverage": round(coverage, 4),
        "accuracy": 0.0 if violated else round(0.5 + 0.5 * coverage, 4),
        "citation_support": round(citation_support, 4),
        "hallucination_avoidance": round(hallucination_avoidance, 4),
        "conflict_handling": 1.0 if "caveat" in body_lower or "watch" in body_lower else 0.5,
    }


def _point_hit(point: str, body_lower: str) -> bool:
    tokens = [token for token in point.lower().replace("/", " ").split() if len(token) >= 4]
    if not tokens:
        return point.lower() in body_lower
    hits = sum(1 for token in tokens if token in body_lower)
    return hits >= max(1, len(tokens) // 2)


def judge_report(
    *,
    model_client: Any | None,
    case: EvalCase,
    report_body: str,
    citation_summary: CitationCheckSummary,
    run_id: str | None = None,
) -> JudgeResult:
    provider = getattr(model_client, "provider", "none")
    model = getattr(model_client, "model", "none")
    payload = build_judge_payload(case=case, report_body=report_body, citation_summary=citation_summary)

    # No usable report -> zero score without burning a model call.
    if not report_body.strip():
        scores = {dimension: 0.0 for dimension in DIMENSIONS}
        return JudgeResult(
            score_overall=0.0,
            scores=scores,
            rationale={dimension: "no report body delivered" for dimension in DIMENSIONS},
            judge_provider=provider,
            judge_model=model,
            status="no_report",
            judge_method="no_report",
        )

    parsed: dict[str, Any] | None = None
    raw_text = ""
    status = "ok"
    error: str | None = None
    judge_method = "fallback"  # default assumption: no usable LLM score yet

    is_fake = provider == "fake"
    if model_client is not None and not is_fake:
        try:
            result = model_client.complete(
                [
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                metadata={"role": "eval_judge", "run_id": run_id},
            )
            raw_text = result.text or ""
            if result.status != "ok":
                status, error = "judge_error", result.error
            else:
                parsed = _parse_judge_json(raw_text) or (result.json_data if isinstance(result.json_data, dict) else None)
                if parsed is None:
                    status, error = "unparseable", "judge returned non-JSON output"
        except Exception as exc:  # noqa: BLE001 - judge must never crash the run
            status, error = "judge_exception", str(exc)

    if parsed and isinstance(parsed.get("scores"), dict):
        raw_scores = parsed["scores"]
        scores = {dimension: _clamp(raw_scores.get(dimension)) for dimension in DIMENSIONS}
        rationale_raw = parsed.get("rationale") if isinstance(parsed.get("rationale"), dict) else {}
        rationale = {dimension: str(rationale_raw.get(dimension, "")) for dimension in DIMENSIONS}
        judge_method = "llm"
    else:
        scores = _deterministic_fallback(case=case, report_body=report_body, citation_summary=citation_summary)
        rationale = {dimension: "deterministic fallback (judge model unavailable or unparseable)" for dimension in DIMENSIONS}
        if status == "ok":
            status = "fallback"
        # judge_method stays "fallback"

    return JudgeResult(
        score_overall=combine(scores),
        scores=scores,
        rationale=rationale,
        judge_provider=provider,
        judge_model=model,
        status=status,
        error=error,
        raw_text=raw_text,
        judge_method=judge_method,
    )
