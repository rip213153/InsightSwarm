from __future__ import annotations

import os

import pytest

from insightswarm.cli import main as cli_main
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.observability.trace import latest_collaboration_trace
from insightswarm.research_graph import build_research_graph_validation


def make_store(tmp_path):
    db_path = tmp_path / "real_governance.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def test_real_model_governance_acceptance(tmp_path, capsys):
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.fail("DASHSCOPE_API_KEY is required for Phase 44 real-model governance acceptance.")
    store = make_store(tmp_path)
    source_file = tmp_path / "salesforce_sales_cloud_pricing.txt"
    source_file.write_text(
        (
            "Salesforce Sales Cloud pricing public source. Starter Suite is priced at $25 USD per user per month "
            "when billed annually. Professional Suite is priced at $100 USD per user per month when billed annually. "
            "Enterprise is priced at $175 USD per user per month when billed annually. The page positions Sales Cloud "
            "as CRM software for sales teams with automation, pipeline management, and AI-assisted selling capabilities."
        ),
        encoding="utf-8",
    )
    args = ["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "--model-provider", "qwen_text"]
    assert (
        cli_main(
            [
                *args,
                "run",
                "create",
                "--name",
                "phase44-real-governance",
                "--query",
                "Research Salesforce Sales Cloud B2B pricing and positioning using public sources.",
                "--competitor",
                "Salesforce Sales Cloud",
                "--source-url",
                "https://www.salesforce.com/sales/pricing/",
                "--source-text-file",
                str(source_file),
                "--search-provider",
                "static",
            ]
        )
        == 0
    )
    run_id = capsys.readouterr().out.strip()
    raw_id = store.write_artifact(
        run_id,
        None,
        "raw_document",
        "text/plain",
        source_file.read_text(encoding="utf-8"),
        source_url="https://www.salesforce.com/sales/pricing/",
        metadata={"fetcher": "real_model_acceptance_fixture", "status": "ok", "competitor": "Salesforce Sales Cloud"},
    )
    assert cli_main([*args, "run", "extract", "--run-id", run_id, "--raw-document-id", raw_id]) == 0
    capsys.readouterr()
    assert cli_main([*args, "run", "start", "--run-id", run_id]) == 0
    capsys.readouterr()
    assert cli_main([*args, "run", "govern", "--run-id", run_id, "--max-steps", "8", "--allow-delivery", "--json"]) == 0
    capsys.readouterr()

    artifacts = [dict(row) for row in store.list_artifacts(run_id)]
    citations = [dict(row) for row in store.list_citations(run_id)]
    model_calls = list(store.conn.execute("SELECT * FROM model_calls WHERE run_id = ?", (run_id,)))
    validation = build_research_graph_validation(store, run_id)
    trace = latest_collaboration_trace(store, run_id)

    assert model_calls
    assert citations
    assert validation["summary"]["error_count"] == 0
    assert any(row["artifact_type"] == "qa_report" for row in artifacts)
    assert any(row["artifact_type"] == "evidence_convergence_decision" for row in artifacts)
    assert any(row["artifact_type"] in {"report", "report_blocked"} for row in artifacts)
    assert any(row["artifact_type"] == "governance_decision" for row in artifacts)
    assert any(row["artifact_type"] == "isolated_context_envelope" for row in artifacts)
    assert not any(
        row["artifact_type"] in {"candidate_source", "candidate_research_source", "research_finding", "subagent_handoff"}
        and row["artifact_id"] in {citation["artifact_id"] for citation in citations}
        for row in artifacts
    )
    assert trace
    assert (trace.get("lead_agent_governance") or {}).get("summary", {}).get("decision_count", 0) >= 1
