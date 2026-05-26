from __future__ import annotations

import json
import os

import pytest

from insightswarm.cli import main as cli_main
from insightswarm.db.store import Store
from insightswarm.observability.diagnosis import build_run_diagnosis
from insightswarm.research_graph import build_research_graph_validation
from insightswarm.util import loads


def test_real_phase47_deepseek_objective_swarm_acceptance(tmp_path, capsys):
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.fail("DASHSCOPE_API_KEY is required for Phase 47 real DeepSeek objective-swarm acceptance.")
    if not (os.getenv("TAVILY_API_KEY") or os.getenv("INSIGHTSWARM_TAVILY_API_KEY")):
        pytest.fail("TAVILY_API_KEY or INSIGHTSWARM_TAVILY_API_KEY is required for Phase 47 real search acceptance.")

    store = Store(tmp_path / "phase47_real.db", tmp_path / "artifacts")
    args = ["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "--model-provider", "qwen_text"]
    rc = cli_main(
        [
            *args,
            "run",
            "ask",
            "--query",
            "DeepSeek 下步的战略规划是什么？请基于公开证据做竞争情报分析",
            "--max-steps",
            "12",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    run_id = result["run_id"]
    artifacts = [dict(row) for row in store.list_artifacts(run_id)]
    events = [dict(row) for row in store.conn.execute("SELECT * FROM agent_events WHERE run_id = ? ORDER BY created_at", (run_id,))]
    model_calls = [dict(row) for row in store.conn.execute("SELECT * FROM model_calls WHERE run_id = ? ORDER BY created_at", (run_id,))]
    diagnosis = build_run_diagnosis(store, run_id)
    validation = build_research_graph_validation(store, run_id)

    assert rc == 0
    assert result["status"] == "objective_governed"
    assert result["final_state"] in {"delivered", "exhausted", "needs_human"}
    assert any(row["artifact_type"] == "intelligence_objective" for row in artifacts)
    assert any(row["artifact_type"] == "capability_arbitration" for row in artifacts)
    assert any(row["artifact_type"] == "objective_state_transition" for row in artifacts)
    assert any(row["artifact_type"] == "isolated_context_envelope" and loads(row["metadata_json"], {}).get("agent_name") == "LeadAgent" for row in artifacts)
    assert any(loads(row["metadata_json"], {}).get("tool_name") == "search.web" for row in events)
    assert any("qwen3.6-35b-a3b" in (row.get("model") or "") for row in model_calls)
    assert diagnosis["objective_runtime_summary"]["model_governed_decision_count"] >= 1
    assert diagnosis["objective_runtime_summary"]["loop_counters"]["search_calls"] >= 1
    if result["final_state"] != "delivered":
        assert not any(row["artifact_type"] == "report" for row in artifacts)
    assert validation["summary"]["error_count"] == 0
    if result["final_state"] == "delivered":
        assert len(store.list_citations(run_id)) >= 1
        assert diagnosis["qa_gate_summary"]["passed"] is True
        assert diagnosis["evidence_convergence_summary"]["decision_count"] >= 1
        assert any(row["artifact_type"] == "delivery_request" for row in artifacts)
        assert any(row["artifact_type"] in {"report", "report_blocked"} for row in artifacts)
    else:
        assert result["stop_reason"] in {
            "blocked_no_verifiable_source",
            "blocked_no_citation",
            "exhausted_search_budget",
            "exhausted_browser_budget",
            "exhausted_extractor_repair_budget",
            "exhausted_no_citation_after_source_acquisition",
            "needs_human_authorization",
            "max_steps_reached",
        }
