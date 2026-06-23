"""Aggregate a single swarm run into run-level evaluation telemetry.

This module is read-only against the main InsightSwarm store. It never writes
business data; it only summarizes what a completed run produced so the eval
layer can score it and persist metrics to the separate eval database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from insightswarm.db.store import Store


@dataclass
class RunTelemetry:
    run_id: str
    token_total: int = 0
    token_prompt: int = 0
    token_completion: int = 0
    model_call_count: int = 0
    model_error_count: int = 0
    latency_ms_total: int = 0
    provider_breakdown: dict[str, int] = field(default_factory=dict)
    citation_count: int = 0
    evidence_count: int = 0
    raw_document_count: int = 0

    def as_metrics(self) -> dict[str, Any]:
        return {
            "token_total": self.token_total,
            "token_prompt": self.token_prompt,
            "token_completion": self.token_completion,
            "model_call_count": self.model_call_count,
            "model_error_count": self.model_error_count,
            "latency_ms_total": self.latency_ms_total,
            "provider_breakdown": self.provider_breakdown,
            "citation_count": self.citation_count,
            "evidence_count": self.evidence_count,
            "raw_document_count": self.raw_document_count,
        }


def _usage_int(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def collect_run_telemetry(store: Store, run_id: str) -> RunTelemetry:
    """Aggregate model calls and swarm evidence for one run."""
    telemetry = RunTelemetry(run_id=run_id)

    rows = store.conn.execute(
        "SELECT provider, usage_json, latency_ms, status FROM model_calls WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    for row in rows:
        telemetry.model_call_count += 1
        telemetry.latency_ms_total += int(row["latency_ms"] or 0)
        provider = str(row["provider"] or "unknown")
        telemetry.provider_breakdown[provider] = telemetry.provider_breakdown.get(provider, 0) + 1
        if str(row["status"] or "") != "ok":
            telemetry.model_error_count += 1
        try:
            usage = json.loads(row["usage_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            usage = {}
        if isinstance(usage, dict):
            telemetry.token_prompt += _usage_int(usage, "prompt_tokens", "input_tokens")
            telemetry.token_completion += _usage_int(usage, "completion_tokens", "output_tokens")
            telemetry.token_total += _usage_int(usage, "total_tokens")

    if telemetry.token_total == 0:
        telemetry.token_total = telemetry.token_prompt + telemetry.token_completion

    telemetry.evidence_count = len(store.list_swarm_evidence(run_id))
    telemetry.citation_count = telemetry.evidence_count
    telemetry.raw_document_count = sum(
        1 for artifact in store.list_swarm_artifacts(run_id) if artifact.type == "raw_document"
    )
    return telemetry


def collect_source_corpus(store: Store, run_id: str) -> dict[str, str]:
    """Map source_url -> raw document text for every raw_document artifact.

    Used by quote verification to check whether report citations are actually
    grounded in fetched source text. Multiple documents from the same URL are
    concatenated so a quote found in any fetch of that URL counts as grounded.
    """
    corpus: dict[str, list[str]] = {}
    for artifact in store.list_swarm_artifacts(run_id):
        if artifact.type != "raw_document":
            continue
        payload_ref = getattr(artifact, "payload_ref", "")
        if not payload_ref or not Path(payload_ref).exists():
            continue
        try:
            payload = json.loads(Path(payload_ref).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        url = str(payload.get("source_url") or payload.get("url") or "").strip()
        text = str(payload.get("text") or "")
        if not url or not text:
            continue
        corpus.setdefault(url, []).append(text)
    return {url: "\n\n".join(texts) for url, texts in corpus.items()}


def collect_report_citations(store: Store, run_id: str) -> list[dict[str, Any]]:
    """Return citation rows as plain dicts for quote verification and judging.

    Current runs write quote-backed citations as ``swarm_evidence``. The old
    top-level ``citations`` table was removed with the legacy runtime.
    """
    citations: list[dict[str, Any]] = []
    for evidence in store.list_swarm_evidence(run_id):
        citations.append(
            {
                "source_url": evidence.source_url,
                "quote": evidence.quote,
                "claim": "",
                "confidence": evidence.confidence,
                "evidence_id": evidence.evidence_id,
            }
        )
    return citations
