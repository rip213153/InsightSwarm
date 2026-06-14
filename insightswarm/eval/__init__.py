"""Evaluation subsystem for InsightSwarm.

This package is a side-channel quality harness. It schedules existing swarm
runs over a golden case suite, aggregates per-run telemetry, scores reports
with an LLM judge, and deterministically verifies that report citations are
actually grounded in fetched source text.

Evaluation state lives in a separate SQLite database (default
``.insightswarm/eval.db``) so it never mixes with production run state.
"""

from __future__ import annotations
