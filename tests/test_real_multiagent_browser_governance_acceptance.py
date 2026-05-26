from __future__ import annotations

import json
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from insightswarm.cli import main as cli_main
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.observability.trace import latest_collaboration_trace
from insightswarm.research_graph import build_research_graph_validation
from insightswarm.util import loads


def make_store(tmp_path):
    db_path = tmp_path / "real_multiagent_browser.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003, ANN001
        return None


def serve_directory(path: Path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}"


def test_real_multiagent_browser_governance_acceptance(tmp_path, capsys):
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.fail("DASHSCOPE_API_KEY is required for Phase 45 real multi-agent browser governance acceptance.")
    store = make_store(tmp_path)
    site = tmp_path / "site"
    site.mkdir()
    (site / "pricing.html").write_text(
        """
        <!doctype html>
        <html>
          <head><title>ExampleCo governed pricing</title></head>
          <body>
            <main>
              <h1>ExampleCo B2B Intelligence Suite pricing</h1>
              <p>Starter plan costs $49 per user per month.</p>
              <p>Growth plan costs $129 per user per month and includes analyst workflows.</p>
              <a href="/pricing.html">Pricing source</a>
            </main>
          </body>
        </html>
        """,
        encoding="utf-8",
    )
    server, base_url = serve_directory(site)
    target_url = f"{base_url}/pricing.html"
    args = ["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "--model-provider", "qwen_text"]
    try:
        assert (
            cli_main(
                [
                    *args,
                    "run",
                    "create",
                    "--name",
                    "phase45-real-multiagent-browser",
                    "--quality-mode",
                    "test",
                    "--query",
                    "Research ExampleCo B2B Intelligence Suite pricing using governed multi-agent source acquisition.",
                    "--competitor",
                    "ExampleCo",
                    "--browser-source-target-url",
                    target_url,
                    "--browser-backend",
                    "fake",
                    "--search-provider",
                    "static",
                ]
            )
            == 0
        )
        run_id = capsys.readouterr().out.strip()
        parent = store.create_task(run_id, "Discovery", "ScraperAgent")
        store.set_task_status(parent, "completed")
        store.write_artifact(
            run_id,
            parent,
            "fetch_failure",
            "application/json",
            json.dumps({"source_url": "https://example.invalid/pricing", "status": "error", "error": "seed evidence gap"}),
            source_url="https://example.invalid/pricing",
            metadata={"source_url": "https://example.invalid/pricing", "status": "error"},
            suffix=".json",
        )

        assert cli_main([*args, "run", "govern", "--run-id", run_id, "--max-steps", "6", "--json"]) == 0
        capsys.readouterr()
        assert cli_main([*args, "run", "govern", "--run-id", run_id, "--max-steps", "3", "--allow-delivery", "--json"]) == 0
        capsys.readouterr()

        artifacts = [dict(row) for row in store.list_artifacts(run_id)]
        citations = [dict(row) for row in store.list_citations(run_id)]
        model_calls = [dict(row) for row in store.conn.execute("SELECT * FROM model_calls WHERE run_id = ?", (run_id,))]
        decisions = [row for row in artifacts if row["artifact_type"] == "governance_decision"]
        first_decision = json.loads(Path(decisions[0]["path"]).read_text(encoding="utf-8"))
        validation = build_research_graph_validation(store, run_id)
        trace = latest_collaboration_trace(store, run_id)

        assert any(row["model"] == "qwen3.6-35b-a3b" for row in model_calls)
        assert first_decision["model_governed"] is True
        assert first_decision["fallback_reason"] is None
        assert any(row["artifact_type"] == "browser_page_state" for row in artifacts)
        assert any(row["artifact_type"] == "browser_code_result" for row in artifacts)
        assert any(row["artifact_type"] == "candidate_source" for row in artifacts)
        assert any(row["artifact_type"] == "source_acquisition_gateway" for row in artifacts)
        assert any(row["artifact_type"] == "candidate_research_source" for row in artifacts)
        assert citations
        assert any(row["artifact_type"] == "qa_report" for row in artifacts)
        assert any(row["artifact_type"] == "evidence_convergence_decision" for row in artifacts)
        assert any(row["artifact_type"] == "delivery_request" for row in artifacts)
        assert any(row["artifact_type"] in {"report", "report_blocked"} for row in artifacts)
        assert any(loads(row["metadata_json"], {}).get("agent_name") == "LeadAgent" for row in artifacts if row["artifact_type"] == "isolated_context_envelope")
        assert any(loads(row["metadata_json"], {}).get("agent_name") == "BrowserAgent" for row in artifacts if row["artifact_type"] == "isolated_context_envelope")
        assert not any(
            row["artifact_type"] in {"candidate_source", "browser_swarm_operation", "subagent_handoff", "research_finding"}
            and row["artifact_id"] in {citation["artifact_id"] for citation in citations}
            for row in artifacts
        )
        assert validation["summary"]["error_count"] == 0
        assert (trace.get("lead_agent_governance") or {}).get("summary", {}).get("decision_count", 0) >= 1
        assert trace.get("browser_code_executions")
    finally:
        server.shutdown()
