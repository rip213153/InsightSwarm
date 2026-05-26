from __future__ import annotations

import json
from pathlib import Path

from insightswarm.db.store import Store
from insightswarm.quality import evaluate_run_quality
from insightswarm.util import loads


def _short(citation_id: str) -> str:
    return citation_id.split("_", 1)[1]


def render_report(store: Store, run_id: str) -> str:
    metadata = store.get_run_metadata(run_id)
    competitor = metadata.get("competitor") or "Unknown Competitor"
    citations = store.list_citations(run_id)
    quality = evaluate_run_quality(store, run_id)
    docs = [row for row in citations if row["source_type"] == "document"]
    imgs = [row for row in citations if row["source_type"] == "image"]
    infs = [row for row in citations if row["source_type"] == "inference"]
    lines = [
        "# Competitive Analysis Report",
        "",
        "## Competitor",
        "",
        competitor,
        "",
        "## Source Health",
        "",
    ]
    artifacts = store.list_artifacts(run_id)
    raw_docs = [row for row in artifacts if row["artifact_type"] == "raw_document"]
    failures = [row for row in artifacts if row["artifact_type"] == "fetch_failure"]
    snippets = [row for row in raw_docs if loads(row["metadata_json"], {}).get("fetcher") == "tavily_snippet"]
    synthetic = [row for row in raw_docs if loads(row["metadata_json"], {}).get("fetcher") == "synthetic_fallback"]
    pdf_text = [row for row in raw_docs if loads(row["metadata_json"], {}).get("fetcher") == "pdf_text_source"]
    cleaned = [row for row in artifacts if row["artifact_type"] == "cleaned_document"]
    stale = [row for row in cleaned if loads(row["metadata_json"], {}).get("freshness_status") == "stale"]
    noisy = [row for row in cleaned if int(loads(row["metadata_json"], {}).get("noise_removed_count", 0)) > 0]
    lines.extend(
        [
            f"- Quality status: {quality.quality_status}",
            f"- Final status reason: {quality.run_final_status_reason}",
            f"- Raw document artifacts: {len(raw_docs)}",
            f"- Cleaned document artifacts: {len(cleaned)}",
            f"- Tavily snippet artifacts: {len(snippets)}",
            f"- PDF text source artifacts: {len(pdf_text)}",
            f"- Synthetic fallback artifacts: {len(synthetic)}",
            f"- Synthetic fallback policy: {'diagnostic-only for production query' if quality.source_trust['production_query'] else 'eligible in non-production/test runs'}",
            f"- Formal real evidence available: {quality.source_trust['formal_evidence_available']}",
            f"- Fetch failure artifacts: {len(failures)}",
            f"- Stale source warnings: {len(stale)}",
            f"- Noise-cleaned sources: {len(noisy)}",
            "",
        ]
    )
    if quality.degraded_reasons:
        lines.extend(["## Degraded Output Warnings", ""])
        for reason in quality.degraded_reasons:
            lines.append(f"- [{reason['severity']}] {reason['message']}")
        lines.append("")
    skeptic_review = latest_skeptic_review(store, run_id)
    if skeptic_review:
        lines.extend(["## Skeptic Review", ""])
        for gap in skeptic_review.get("evidence_gaps", [])[:3]:
            lines.append(f"- Evidence gap: {gap}")
        for risk in skeptic_review.get("source_risks", [])[:3]:
            lines.append(f"- Source risk: {risk}")
        lines.append("")
    lines.extend(
        [
            "## Evidence-Backed Findings",
            "",
        ]
    )
    for doc in docs:
        lines.append(f"- {doc['quote']} [[doc:{_short(doc['citation_id'])}]].")
    for img in imgs:
        lines.append(f"- Visual evidence was captured from {img['source_url']} [[img:{_short(img['citation_id'])}]].")
    lines.extend(["", "## Strategic Read", ""])
    for inf in infs:
        lines.append(f"- {inf['claim']} [[inf:{_short(inf['citation_id'])}]].")
    lines.extend(["", "## Citation Index", ""])
    for doc in docs:
        lines.append(f"- [[doc:{_short(doc['citation_id'])}]] {doc['source_url']} quote: {doc['quote']}")
    for img in imgs:
        lines.append(f"- [[img:{_short(img['citation_id'])}]] {img['source_url']} bbox: {img['image_bbox_json']}")
    for inf in infs:
        evidence_ids = loads(inf["evidence_ids_json"], [])
        lines.append(f"- [[inf:{_short(inf['citation_id'])}]] evidence: {', '.join(evidence_ids)}")
    lines.append("")
    return "\n".join(lines)


def render_quality_section(store: Store, run_id: str) -> str:
    quality = evaluate_run_quality(store, run_id)
    health = quality.source_health
    lines = [
        "## Source Health",
        "",
        f"- Quality status: {quality.quality_status}",
        f"- Final status reason: {quality.run_final_status_reason}",
        f"- Raw document artifacts: {health['raw_document_count']}",
        f"- Real raw document artifacts: {health['real_raw_document_count']}",
        f"- Synthetic fallback artifacts: {health['synthetic_fallback_count']}",
        f"- PDF text source artifacts: {health['pdf_text_source_count']}",
        f"- Synthetic fallback policy: {'diagnostic-only for production query' if quality.source_trust['production_query'] else 'eligible in non-production/test runs'}",
        f"- Formal real evidence available: {quality.source_trust['formal_evidence_available']}",
        f"- Fetch failure artifacts: {health['fetch_failure_count']}",
        f"- Stale source warnings: {health['stale_document_count']}",
        f"- Noise-cleaned sources: {health['noisy_document_count']}",
        "",
    ]
    if quality.degraded_reasons:
        lines.extend(["## Degraded Output Warnings", ""])
        for reason in quality.degraded_reasons:
            lines.append(f"- [{reason['severity']}] {reason['message']}")
        lines.append("")
    return "\n".join(lines)


def latest_skeptic_review(store: Store, run_id: str) -> dict | None:
    reviews = [row for row in store.list_artifacts(run_id) if row["artifact_type"] == "skeptic_review"]
    if not reviews:
        return None
    path = Path(reviews[-1]["path"])
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def render_citations_json(store: Store, run_id: str) -> str:
    return json.dumps([dict(row) for row in store.list_citations(run_id)], ensure_ascii=True, indent=2)


def latest_qa_report_json(store: Store, run_id: str) -> str:
    qa_reports = [row for row in store.list_artifacts(run_id) if row["artifact_type"] == "qa_report"]
    if not qa_reports:
        return json.dumps({"passed": False, "reason": "no qa report"}, indent=2)
    return Path(qa_reports[-1]["path"]).read_text(encoding="utf-8", errors="replace")
