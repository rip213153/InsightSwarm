"""Statistics for non-deterministic eval scoring.

Agent runs are stochastic, so a single score is noise. We run each case over
several epochs and summarize with mean and standard error of the mean
(``stderr = std / sqrt(n)``). When comparing two eval runs we only call a per-
case difference real if it clears a noise band built from both runs' standard
errors, so a lucky/unlucky single run never reads as a regression or a win.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreSummary:
    n: int
    mean: float
    std: float          # sample standard deviation (ddof=1)
    stderr: float
    min_score: float
    max_score: float


def summarize(scores: list[float]) -> ScoreSummary:
    valid = [float(s) for s in scores if s is not None]
    n = len(valid)
    if n == 0:
        return ScoreSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    mean = sum(valid) / n
    if n == 1:
        std = 0.0
    else:
        variance = sum((s - mean) ** 2 for s in valid) / (n - 1)
        std = math.sqrt(variance)
    stderr = std / math.sqrt(n) if n > 0 else 0.0
    return ScoreSummary(n, mean, std, stderr, min(valid), max(valid))


@dataclass(frozen=True)
class CaseComparison:
    case_id: str
    mean_a: float
    mean_b: float
    delta: float          # mean_b - mean_a
    noise_band: float     # k * sqrt(stderr_a^2 + stderr_b^2)
    verdict: str          # "improved" | "regressed" | "noise" | "new" | "dropped"


def compare_case(
    case_id: str,
    a: ScoreSummary | None,
    b: ScoreSummary | None,
    *,
    k: float = 2.0,
) -> CaseComparison:
    """Compare one case across two eval runs with a stderr-based noise band.

    ``k=2`` approximates a ~95% band for the difference of two means. A delta
    inside the band is reported as "noise" rather than a directional change.
    """
    if a is None or a.n == 0:
        mean_b = b.mean if b else 0.0
        return CaseComparison(case_id, 0.0, mean_b, mean_b, 0.0, "new")
    if b is None or b.n == 0:
        return CaseComparison(case_id, a.mean, 0.0, -a.mean, 0.0, "dropped")

    delta = b.mean - a.mean
    noise_band = k * math.sqrt(a.stderr ** 2 + b.stderr ** 2)
    if abs(delta) <= noise_band:
        verdict = "noise"
    elif delta > 0:
        verdict = "improved"
    else:
        verdict = "regressed"
    return CaseComparison(case_id, a.mean, b.mean, delta, noise_band, verdict)
