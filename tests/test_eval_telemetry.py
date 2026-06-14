from __future__ import annotations

from dataclasses import dataclass

from insightswarm.eval.telemetry import collect_report_citations


@dataclass(frozen=True)
class _Evidence:
    source_url: str
    quote: str
    confidence: float
    evidence_id: str


class _StoreWithSwarmEvidence:
    def list_swarm_evidence(self, run_id: str):
        assert run_id == "run_1"
        return [
            _Evidence(
                source_url="https://example.com/a",
                quote="exact quote",
                confidence=0.9,
                evidence_id="ev_1",
            )
        ]

    def list_citations(self, run_id: str):
        raise AssertionError("legacy citations should not be read when swarm evidence exists")


class _StoreWithLegacyCitations:
    def list_swarm_evidence(self, run_id: str):
        return []

    def list_citations(self, run_id: str):
        assert run_id == "run_1"
        return [
            {
                "source_url": "https://example.com/legacy",
                "quote": "legacy quote",
                "claim": "legacy claim",
                "confidence": 0.8,
            }
        ]


def test_collect_report_citations_prefers_swarm_evidence():
    citations = collect_report_citations(_StoreWithSwarmEvidence(), "run_1")

    assert citations == [
        {
            "source_url": "https://example.com/a",
            "quote": "exact quote",
            "claim": "",
            "confidence": 0.9,
            "evidence_id": "ev_1",
        }
    ]


def test_collect_report_citations_falls_back_to_legacy_rows():
    citations = collect_report_citations(_StoreWithLegacyCitations(), "run_1")

    assert citations == [
        {
            "source_url": "https://example.com/legacy",
            "quote": "legacy quote",
            "claim": "legacy claim",
            "confidence": 0.8,
        }
    ]
