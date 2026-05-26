from __future__ import annotations

import json
from pathlib import Path

from insightswarm.agents.qa import _targeted_evidence_request
from insightswarm.cli import main as cli_main
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.objective_runtime import ObjectiveDrivenSwarmRuntime
from insightswarm.tools import ToolContext
from insightswarm.tools.executor import ToolExecutor
from insightswarm.util import loads


def make_store(tmp_path):
    db_path = tmp_path / "phase48.db"
    init_db(db_path)
    return Store(db_path, tmp_path / "artifacts")


def test_phase48_run_ask_delivers_by_default_and_marks_legacy_commands(tmp_path, capsys):
    store = make_store(tmp_path)
    args = ["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "--model-provider", "fake"]

    rc = cli_main(
        [
            *args,
            "run",
            "ask",
            "--query",
            "ExampleCo public evidence pricing intelligence",
            "--max-steps",
            "8",
            "--quality-mode",
            "test",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    artifacts = [dict(row) for row in store.list_artifacts(result["run_id"])]

    assert rc == 0
    assert result["final_state"] == "delivered"
    assert any(row["artifact_type"] == "delivery_request" for row in artifacts)
    assert any(row["artifact_type"] in {"report", "report_blocked"} for row in artifacts)

    assert cli_main([*args, "run", "continue-writer", "--run-id", result["run_id"], "--qa-report-id", _latest_qa_report_id(store, result["run_id"])]) in {0, 2}
    captured = capsys.readouterr()
    assert "legacy/internal after Phase 48" in captured.err


def test_phase48_critic_targeted_evidence_request_keeps_validator_first():
    request = _targeted_evidence_request(
        [
            {"category": "evidence", "gate": "quote_span_backcheck"},
            {"category": "permission", "gate": "real_document_coverage"},
        ],
        {"evidence_gaps": ["Need official pricing page"]},
        "ExampleCo pricing",
    )

    assert request is not None
    assert request["schema"] == "targeted_evidence_request.v1"
    assert request["needs_browser_escalation"] is True
    assert "quote_span_backcheck" in request["deterministic_failure_gates"]
    assert "Need official pricing page" in request["model_evidence_gaps"]


def test_phase48_objective_browser_escalation_promotes_raw_source_without_formal_browser_evidence(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "phase48-browser-raw-source",
        {"quality_mode": "test", "query": "ExampleCo pricing", "browser_backend": "fake", "objective_runtime": True},
    )
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    result, _ = ToolExecutor(store).run(
        "browser.promote_source",
        {
            "source_url": "http://localhost:9876/pricing",
            "title": "ExampleCo pricing",
            "text_preview": "ExampleCo pricing page. Starter plan costs $49 per month.",
        },
        ToolContext(run_id, task_id, "test", {"agent_name": "BrowserAgent", "browser_mode": "free_browser"}),
    )
    assert result.status == "ok"

    runtime = ObjectiveDrivenSwarmRuntime(store, run_id, model_provider="fake", allow_delivery=True)
    promotions = runtime._promote_browser_candidates_to_raw_documents()
    artifacts = [dict(row) for row in store.list_artifacts(run_id)]
    raw_doc = next(row for row in artifacts if row["artifact_type"] == "raw_document")
    raw_text = Path(raw_doc["path"]).read_text(encoding="utf-8")

    assert promotions[0]["status"] == "promoted"
    assert "Starter plan costs $49 per month" in raw_text
    assert loads(raw_doc["metadata_json"], {})["fetcher"] == "browser_agent_handoff"
    assert not store.list_citations(run_id)


def _latest_qa_report_id(store: Store, run_id: str) -> str:
    return [row["artifact_id"] for row in store.list_artifacts(run_id) if row["artifact_type"] == "qa_report"][-1]
