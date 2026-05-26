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
from insightswarm.util import loads, now_iso


def make_store(tmp_path: Path, name: str) -> Store:
    db_path = tmp_path / f"{name}.db"
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


def _require_real_env() -> None:
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.fail("DASHSCOPE_API_KEY is required for Phase 48 acceptance.")
    if not (os.getenv("TAVILY_API_KEY") or os.getenv("INSIGHTSWARM_TAVILY_API_KEY")):
        pytest.fail("TAVILY_API_KEY or INSIGHTSWARM_TAVILY_API_KEY is required for Phase 48 acceptance.")


def _events(store: Store, run_id: str) -> list[dict]:
    return [dict(row) for row in store.conn.execute("SELECT * FROM agent_events WHERE run_id = ? ORDER BY created_at", (run_id,))]


def _artifacts(store: Store, run_id: str) -> list[dict]:
    return [dict(row) for row in store.list_artifacts(run_id)]


def _payload(row: dict) -> dict:
    return json.loads(Path(row["path"]).read_text(encoding="utf-8", errors="replace"))


def _assert_no_legacy_continuation(events: list[dict]) -> None:
    forbidden = {"continue-followup", "continue-candidate", "continue-repair", "continue-writer"}
    for row in events:
        payload = loads(row.get("metadata_json"), {})
        text = json.dumps(payload, ensure_ascii=False)
        assert not any(item in text for item in forbidden), payload
        assert not any(item in (row.get("event_type") or "") for item in forbidden), row
        assert not any(item in (row.get("message") or "") for item in forbidden), row


def test_phase48_acceptance_critic_gap_loop_real(tmp_path, capsys):
    _require_real_env()
    store = make_store(tmp_path, "phase48_critic_gap")
    site = tmp_path / "official_site"
    site.mkdir()
    (site / "index.html").write_text(
        """
        <!doctype html>
        <html>
          <head>
            <meta charset="utf-8" />
            <title>Official Pricing</title>
          </head>
          <body>
            <main>
              <h1>Official Pricing</h1>
              <p>Cursor Pro is priced at $20 per month.</p>
              <p>GitHub Copilot Individual is priced at $10 per month.</p>
            </main>
          </body>
        </html>
        """,
        encoding="utf-8",
    )
    server, base_url = serve_directory(site)
    args = ["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "--model-provider", "qwen_text"]
    try:
        assert (
            cli_main(
                [
                    *args,
                    "run",
                    "create",
                    "--name",
                    "phase48-critic-gap-loop",
                    "--quality-mode",
                    "production",
                    "--query",
                    "Compare the current official public pricing for Cursor Pro and GitHub Copilot Individual in USD, and cite each official pricing page.",
                    "--competitor",
                    "Cursor vs GitHub Copilot",
                    "--search-provider",
                    "tavily",
                    "--search-limit",
                    "1",
                    "--link-gate-max-selected",
                    "1",
                ]
            )
            == 0
        )
        run_id = capsys.readouterr().out.strip()
        metadata = store.get_run_metadata(run_id)
        metadata["max_search_calls"] = 1
        metadata["browser_backend"] = "fake"
        metadata["critic_browser_source_target_url"] = f"{base_url}/index.html"
        with store.transaction() as conn:
            conn.execute(
                "UPDATE runs SET metadata_json = ?, updated_at = ? WHERE run_id = ?",
                (json.dumps(metadata, ensure_ascii=True), now_iso(), run_id),
            )
        rc = cli_main([*args, "run", "start", "--run-id", run_id, "--max-steps", "9", "--json"])
        result = json.loads(capsys.readouterr().out)
        artifacts = _artifacts(store, run_id)
        events = _events(store, run_id)
        qa_reports = [row for row in artifacts if row["artifact_type"] == "qa_report"]
        targeted = [
            _payload(row).get("targeted_evidence_request")
            for row in qa_reports
            if _payload(row).get("targeted_evidence_request") is not None
        ]
        search_events = [row for row in events if loads(row["metadata_json"], {}).get("tool_name") == "search.web"]

        assert rc == 0
        _assert_no_legacy_continuation(events)
        assert targeted, "expected at least one Critic targeted_evidence_request"
        assert len(search_events) >= 1, "expected first-round real search before Critic gap"
        assert len(qa_reports) >= 2, "expected second QA after补证"
    finally:
        server.shutdown()


def test_phase48_acceptance_browser_escalation_real(tmp_path, capsys):
    _require_real_env()
    store = make_store(tmp_path, "phase48_browser_escalation")
    site = tmp_path / "spa_site"
    site.mkdir()
    (site / "seed.html").write_text(
        """
        <!doctype html>
        <html>
          <head>
            <meta charset="utf-8" />
            <title>Example SPA Shell</title>
          </head>
          <body>
            <main>
              <h1>Example SPA</h1>
              <p>Pricing content requires browser rendering.</p>
            </main>
          </body>
        </html>
        """,
        encoding="utf-8",
    )
    (site / "index.html").write_text(
        """
        <!doctype html>
        <html>
          <head>
            <meta charset="utf-8" />
            <title>Example SPA Pricing</title>
          </head>
          <body>
            <div id="app">Loading pricing...</div>
            <script>
              const app = document.getElementById('app');
              app.innerHTML = `
                <main>
                  <h1>Example SPA Pricing</h1>
                  <p>Professional plan costs $79 per seat per month.</p>
                  <p>Enterprise plan requires sales contact.</p>
                </main>
              `;
            </script>
          </body>
        </html>
        """,
        encoding="utf-8",
    )
    server, base_url = serve_directory(site)
    args = ["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "--model-provider", "qwen_text"]
    try:
        assert (
            cli_main(
                [
                    *args,
                    "run",
                    "create",
                    "--name",
                    "phase48-browser-escalation",
                    "--quality-mode",
                    "test",
                    "--query",
                    "Collect pricing evidence from the target SPA page and cite the price.",
                    "--competitor",
                    "Example SPA",
                    "--browser-source-target-url",
                    f"{base_url}/index.html",
                    "--browser-backend",
                    "fake",
                ]
            )
            == 0
        )
        run_id = capsys.readouterr().out.strip()
        metadata = store.get_run_metadata(run_id)
        metadata["max_search_calls"] = 0
        with store.transaction() as conn:
            conn.execute(
                "UPDATE runs SET metadata_json = ?, updated_at = ? WHERE run_id = ?",
                (json.dumps(metadata, ensure_ascii=True), now_iso(), run_id),
            )
        rc = cli_main([*args, "run", "start", "--run-id", run_id, "--max-steps", "12", "--json"])
        result = json.loads(capsys.readouterr().out)
        artifacts = _artifacts(store, run_id)
        events = _events(store, run_id)
        decisions = [_payload(row) for row in artifacts if row["artifact_type"] == "governance_decision"]

        assert rc == 0
        _assert_no_legacy_continuation(events)
        assert any("browse_sources" in json.dumps(item, ensure_ascii=False) for item in decisions), "expected browser path to be present in governance decisions"
        assert any(row["artifact_type"] == "browser_page_state" for row in artifacts), "expected browser page state"
        assert any(row["artifact_type"] == "raw_document" and "browser_agent_handoff" in json.dumps(loads(row["metadata_json"], {})) for row in artifacts), "expected browser raw source promotion"
        assert store.list_citations(run_id), "expected Extractor citation from browser raw source"
        assert result["final_state"] in {"delivered", "exhausted", "blocked", "needs_human"}
    finally:
        server.shutdown()
