# InsightSwarm Eval Rubric

The judge scores a delivered report against the case's expectations. The judge is
a model from a **different provider** than the system under test, to reduce
self-preference bias. The judge never sees the system's own self-assessment.

Each dimension is scored in `[0.0, 1.0]`. The overall score is the weighted mean.

## Dimensions

### coverage (weight 0.30)
How many of the case's `must_cover` points are addressed by the report, with
substance (not a passing mention). Score = fraction of must-cover points that are
clearly and correctly addressed.

### accuracy (weight 0.30)
Are the factual claims correct and consistent with the cited evidence? Penalize
claims that contradict the sources or that state more certainty than the evidence
supports. A report that hedges appropriately is not penalized.

### citation_support (weight 0.20)
Do the report's load-bearing claims carry citations, and do those citations
actually support the claim? This dimension is informed by the deterministic
quote-verification pass: a report whose citations fail quote verification cannot
score above 0.5 here regardless of the judge's reading.

### hallucination_avoidance (weight 0.10)
Penalize any hard assertion that has no supporting citation, or any
`forbidden` point from the case appearing as an asserted fact. A single
confident fabrication should pull this dimension toward 0.

### conflict_handling (weight 0.10)
When sources disagree or evidence is thin, does the report surface the conflict /
caveat rather than papering over it? Reports that correctly flag their own
coverage gaps score well here.

## Output contract

The judge must return strict JSON:

```json
{
  "scores": {
    "coverage": 0.0,
    "accuracy": 0.0,
    "citation_support": 0.0,
    "hallucination_avoidance": 0.0,
    "conflict_handling": 0.0
  },
  "covered_points": ["..."],
  "missed_points": ["..."],
  "unsupported_assertions": ["..."],
  "rationale": "2-4 sentences explaining the scores"
}
```
