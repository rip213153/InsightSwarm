from __future__ import annotations

import json
import os
import socket
import threading
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from insightswarm.config import load_settings
from insightswarm.collaboration_contract import default_research_contract, research_contract_summary
from insightswarm.collaboration_protocol import ROLE_PROTOCOLS, collaboration_protocol
from insightswarm.browser_authorization import classify_authorization_need
from insightswarm.browser_backend import BrowserSession
from insightswarm.browser_sandbox import write_browser_observation
from insightswarm.browser_interaction import approve_browser_action, list_browser_approvals, reject_browser_action
from insightswarm.browser_planning import page_fingerprint, run_browser_operation
from insightswarm.context_policy import ContextPolicy
from insightswarm.context.budget import ContextBudgeter, TokenEstimator
from insightswarm.cleaning import DocumentCleaner, trim_quote, freshness_status
from insightswarm.db.connection import get_db_connection
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.fetching import (
    FetchResult,
    PlaywrightFetcher,
    classify_fetch_error,
    fetch_source,
    normalize_html_text,
    parse_price_snapshot,
    prices_are_equivalent,
    should_fallback_to_browser,
)
from insightswarm.link_gate import gate_links, rule_gate
from insightswarm.search import StaticSearchClient, SearchResult
from insightswarm.cli import main as cli_main
from insightswarm.collector.ingest import ingest_collector_payload
from insightswarm.harness.runner import Runner
from insightswarm.harness.gates import find_text_span, normalize_currency_label, normalize_billing_cycle
from insightswarm.models.clients import ModelResult
from insightswarm.models.qwen import QwenConfigError, QwenOpenAICompatibleClient
from insightswarm.models.router import build_model_client
from insightswarm.observability.diagnosis import build_run_diagnosis
from insightswarm.observability.inspect import inspect_run
from insightswarm.reporting.validation import extract_citation_markers
from insightswarm.schemas.citation import ImageBBox, TextSpan
from insightswarm.tools import ToolContext, get_tool, list_tools
from insightswarm.tools.executor import ToolExecutor
from insightswarm.util import loads


def make_store(tmp_path):
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def test_sqlite_thread_local_connection_per_thread(tmp_path):
    db_path = tmp_path / "threaded.db"
    init_db(db_path)
    main_conn = get_db_connection(db_path)
    assert get_db_connection(db_path) is main_conn
    ids = []

    def worker():
        ids.append(id(get_db_connection(db_path)))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert ids[0] != id(main_conn)


def test_message_lease_ack_and_timeout(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("lease")
    message_id = store.create_message(
        run_id,
        None,
        "A",
        "B",
        {"hello": "world"},
        "lease-test",
    )
    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    leased = store.lease_messages(run_id, "B", "worker", expired)
    assert [row["message_id"] for row in leased] == [message_id]
    assert store.recover_expired_leases() == 1
    leased_again = store.lease_messages(
        run_id,
        "B",
        "worker",
        (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(),
    )
    store.ack_messages([leased_again[0]["message_id"]])
    row = store.conn.execute(
        "SELECT status FROM messages WHERE message_id = ?", (message_id,)
    ).fetchone()
    assert row["status"] == "acked"


def test_runtime_kernel_step_claims_context_and_acks_mailbox(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("runtime-step", {"quality_mode": "test"})
    task_id = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    message_id = store.create_message(
        run_id,
        task_id,
        "Tester",
        "ResearchLeadAgent",
        {"intent": "handoff", "note": "start"},
        "runtime-step-message",
    )

    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "runtime-step",
            "--run-id",
            run_id,
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "executed"
    assert result["task_id"] == task_id
    assert result["task_lease"]["lease_state"] == "leased"
    assert result["mailbox_lease"]["message_ids"] == [message_id]
    assert result["context_artifact_id"]
    task = store.get_task(task_id)
    assert task["status"] == "completed"
    metadata = loads(task["metadata_json"], {})
    assert metadata["runtime_claimed_by"] == "ResearchLeadAgent"
    assert metadata["runtime_task_lease"]["task_id"] == task_id
    row = store.conn.execute("SELECT status FROM messages WHERE message_id = ?", (message_id,)).fetchone()
    assert row["status"] == "acked"
    events = [dict(row) for row in store.list_events(run_id, limit=50)]
    assert any(row["event_type"] == "runtime_task_claimed" for row in events)
    assert any(row["event_type"] == "runtime_task_completed" for row in events)


def test_runtime_cli_projection_and_policy_gate(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("runtime-policy", {"quality_mode": "test"})
    task_id = store.create_task(run_id, "Synthesize", "StrategicAnalystAgent")
    store.set_task_status(task_id, "pending", {"human_intervention_required": True})

    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "runtime",
            "--run-id",
            run_id,
            "--json",
        ]
    ) == 0
    runtime = json.loads(capsys.readouterr().out)
    assert runtime["summary"]["task_board_item_count"] == 1
    assert runtime["summary"]["active_policy_gate_count"] >= 1
    assert any(gate["name"] == "human_intervention" for gate in runtime["policy_gates"])

    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "runtime-step",
            "--run-id",
            run_id,
        ]
    ) == 2
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "blocked"
    assert blocked["policy_gate"]["gate"] == "human_gate"
    assert store.get_task(task_id)["status"] == "pending"


def test_runner_uses_runtime_kernel_events(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("runtime-runner", {"quality_mode": "test"})
    Runner(store).start(run_id)
    assert store.get_run(run_id)["status"] == "completed"
    events = [dict(row) for row in store.list_events(run_id, limit=200)]
    assert any(row["event_type"] == "runtime_task_claimed" for row in events)
    assert any(row["event_type"] == "runtime_heartbeat" for row in events)
    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["multiagent_runtime_summary"]["contracts"]["runtime_step_is_bounded"] is True
    assert diagnosis["multiagent_runtime_summary"]["runtime_claimed_task_count"] >= 1


def test_normalized_bbox_validation():
    ImageBBox((0.1, 0.2, 0.8, 0.9), 100, 100).validate()
    with pytest.raises(ValueError):
        ImageBBox((0.8, 0.2, 0.1, 0.9), 100, 100).validate()
    with pytest.raises(ValueError):
        ImageBBox((0.1, -0.1, 0.8, 0.9), 100, 100).validate()


def test_fake_model_e2e(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("fake-e2e")
    Runner(store).start(run_id)
    run = store.get_run(run_id)
    assert run["status"] == "completed"

    tasks = store.list_tasks(run_id)
    assert {task["status"] for task in tasks} == {"completed"}
    analyst = next(task for task in tasks if task["agent_name"] == "StrategicAnalystAgent")
    assert analyst["retry_count"] >= 0
    analyst_meta = loads(analyst["metadata_json"], {})
    assert analyst_meta["analysis"]["inferences"][0]["inference_citation_id"].startswith("inf_")

    citations = store.list_citations(run_id)
    citation_types = {row["source_type"] for row in citations}
    assert {"document", "inference"}.issubset(citation_types)
    inf = next(row for row in citations if row["source_type"] == "inference")
    assert loads(inf["evidence_ids_json"], [])

    reports = [
        row for row in store.list_artifacts(run_id) if row["artifact_type"] == "report"
    ]
    assert len(reports) == 1
    report_meta = loads(reports[0]["metadata_json"], {})
    assert report_meta["skeptic_review_present"] is True
    report_text = (tmp_path / "artifacts" / run_id / f"{reports[0]['artifact_id']}.md").read_text()
    assert "[[doc:" in report_text
    assert "[[inf:" in report_text
    assert "## Skeptic Review" in report_text

    qa_reports = [
        row for row in store.list_artifacts(run_id) if row["artifact_type"] == "qa_report"
    ]
    assert len(qa_reports) >= 1
    qa_payloads = [json.loads(open(row["path"], encoding="utf-8").read()) for row in qa_reports]
    assert any(payload["passed"] for payload in qa_payloads)

    context_artifacts = [
        row for row in store.list_artifacts(run_id) if row["artifact_type"] == "context_envelope"
    ]
    executed_agents = [task for task in tasks if task["status"] == "completed"]
    assert len(context_artifacts) >= len(executed_agents)
    context_payloads = [
        json.loads(open(row["path"], encoding="utf-8").read()) for row in context_artifacts
    ]
    assert all(payload["schema_contract"] for payload in context_payloads)
    assert all(payload["minimal_valid_example"] for payload in context_payloads)
    assert all("collaboration_protocol" in payload for payload in context_payloads)
    assert any(payload["task"]["agent_name"] == "ResearchLeadAgent" for payload in context_payloads)
    assert any(payload["task"]["agent_name"] == "SkepticReviewAgent" for payload in context_payloads)
    assert any(row["artifact_type"] == "research_contract" for row in store.list_artifacts(run_id))
    traces = [row for row in store.list_artifacts(run_id) if row["artifact_type"] == "collaboration_trace"]
    assert traces
    trace_payload = json.loads(Path(traces[-1]["path"]).read_text(encoding="utf-8"))
    assert trace_payload["schema"] == "collaboration_trace.v1"
    assert trace_payload["role_steps"]
    assert trace_payload["handoffs"]
    assert trace_payload["qa_gate"]["present"] is True
    assert trace_payload["delivery"]["artifact_type"] == "report"
    assert "delivered" in trace_payload["convergence_status"]
    assert all("api_key" not in json.dumps(call, ensure_ascii=True).lower() for call in trace_payload["tool_calls"])
    skeptic_reviews = [row for row in store.list_artifacts(run_id) if row["artifact_type"] == "skeptic_review"]
    assert len(skeptic_reviews) == 1
    assert loads(skeptic_reviews[0]["metadata_json"], {})["non_blocking"] is True
    analyst_retry_contexts = [
        payload
        for payload in context_payloads
        if payload["task"]["agent_name"] == "StrategicAnalystAgent"
        and payload["retry_context"]
    ]
    for retry_payload in analyst_retry_contexts:
        retry_context = retry_payload["retry_context"]
        assert retry_context["current_rejection"]
        assert retry_context["evidence_anchors"]
        assert "older_rejections" not in retry_context

    model_calls = list(store.conn.execute("SELECT * FROM model_calls"))
    assert model_calls
    request_payloads = [loads(row["request_json"], {}) for row in model_calls]
    assert all(payload.get("context_artifact_id") for payload in request_payloads)


def test_model_router_fake_provider():
    client = build_model_client("fake")
    result = client.complete([], metadata={"role": "test"})
    assert isinstance(result, ModelResult)
    assert result.provider == "fake"


def test_qwen_text_routes_vision_to_omni(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": "{\"ok\": true}"}}], "usage": {}}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        calls.append(payload["model"])
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = build_model_client("qwen_text")
    client.complete([{"role": "user", "content": "text"}])
    client.analyze_image([{"role": "user", "content": "image"}], images=[{"data": b"x"}])
    assert calls == ["qwen3.6-35b-a3b", "qwen3.5-omni-plus-2026-03-15"]


def test_model_router_unimplemented_provider_error():
    client = build_model_client("deepseek")
    with pytest.raises(NotImplementedError):
        client.complete([])
    with pytest.raises(ValueError):
        build_model_client("unknown")


def test_config_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "db_path: yaml.db\nartifact_dir: yaml_artifacts\nmodel_provider: qwen_text\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("INSIGHTSWARM_MODEL_PROVIDER", "deepseek")
    (tmp_path / ".env").write_text(
        "INSIGHTSWARM_MODEL_PROVIDER=openai_compatible\n",
        encoding="utf-8",
    )
    settings = load_settings(model_provider="fake")
    assert settings.model_provider == "fake"
    settings = load_settings()
    assert settings.model_provider == "openai_compatible"
    (tmp_path / ".env").unlink()
    settings = load_settings()
    assert settings.model_provider == "deepseek"
    monkeypatch.delenv("INSIGHTSWARM_MODEL_PROVIDER")
    settings = load_settings()
    assert settings.model_provider == "qwen_text"


def test_token_budget_estimator():
    estimator = TokenEstimator()
    assert estimator.estimate({"a": "1234"}) == estimator.estimate({"a": "1234"})
    assert estimator.estimate({"a": "1234"}) > 0


def test_context_budget_trimming_order():
    envelope = {
        "run_id": "run",
        "task": {"agent_name": "StrategicAnalystAgent"},
        "role_instructions": "role",
        "schema_contract": {"required": ["claim"]},
        "minimal_valid_example": {"claim": "x"},
        "artifacts": [{"snippet": "a" * 1000}],
        "citations": [{"confidence": 0.5}, {"confidence": 0.9}],
        "messages": [{"payload": {"x": "y"}, "is_current_rejection": False}],
        "team_snapshot": [{"x": "y"} for _ in range(10)],
        "retry_context": {"current_rejection": {"reason": "bad"}},
        "budget_metadata": {},
    }
    trimmed = ContextBudgeter(max_tokens=80).apply(envelope)
    assert trimmed["schema_contract"] == {"required": ["claim"]}
    assert trimmed["minimal_valid_example"] == {"claim": "x"}
    assert trimmed["budget_metadata"]["trimmed_items"]
    assert "team_snapshot" in trimmed["budget_metadata"]["trimmed_items"]


def test_default_research_contract_and_protocol_are_stable():
    contract = default_research_contract(
        {
            "query": "Lenovo Legion pricing",
            "competitor": "Lenovo",
            "source_urls": ["https://example.com/legion"],
        }
    )
    assert contract["schema"] == "research_contract.v1"
    assert contract["mode"] == "query"
    assert contract["competitor"] == "Lenovo"
    assert contract["evidence_requirements"]["minimum_document_citations"] == 1
    expected_agents = {
        "ResearchLeadAgent",
        "SearchAgent",
        "LinkGateAgent",
        "ScraperAgent",
        "ExtractorAgent",
        "VisualAgent",
        "StrategicAnalystAgent",
        "SkepticReviewAgent",
        "QAAgent",
        "WriterAgent",
        "BrowserAgent",
    }
    assert expected_agents.issubset(set(ROLE_PROTOCOLS))
    writer_protocol = collaboration_protocol("WriterAgent", contract)
    assert writer_protocol["role"] == "ReportWriter"
    assert writer_protocol["contract_schema"] == "research_contract.v1"
    skeptic_protocol = collaboration_protocol("SkepticReviewAgent", contract)
    assert skeptic_protocol["role"] == "SkepticReviewer"
    assert "QAAgent" in skeptic_protocol["handoff_to"]
    browser_protocol = collaboration_protocol("BrowserAgent", contract)
    assert browser_protocol["role"] == "BrowserOperator"
    assert "use browser.* tools only" in browser_protocol["allowed_actions"]


def test_source_acquisition_tools_registry_examples_and_safety_are_stable():
    tools = {tool.name: tool for tool in list_tools()}
    assert {"search.web", "fetch.url", "source.quality"}.issubset(tools)
    for tool in tools.values():
        assert tool.examples
        json.dumps(tool.examples, ensure_ascii=True)
    static_result = get_tool("search.web").run(
        {
            "query": "ExampleCo pricing",
            "provider": "static",
            "source_urls": ["https://example.com/pricing"],
            "limit": 1,
        }
    )
    assert static_result.status == "ok"
    assert static_result.data["results"][0]["url"] == "https://example.com/pricing"
    tavily_result = get_tool("search.web").run({"query": "ExampleCo pricing", "provider": "tavily"})
    if tavily_result.status == "error":
        assert "API key" in (tavily_result.error or "")
        assert "sk-" not in (tavily_result.error or "")
    blocked_file = get_tool("fetch.url").run({"url": "file:///C:/secret.txt"})
    assert blocked_file.status == "blocked"
    blocked_localhost = get_tool("fetch.url").run({"url": "http://localhost:9999"})
    assert blocked_localhost.status == "blocked"
    allowed_test_localhost = get_tool("fetch.url").run(
        {"url": "http://localhost:9999"},
        ToolContext(quality_mode="test"),
    )
    assert allowed_test_localhost.status in {"error", "blocked"}
    assert allowed_test_localhost.error != "Localhost URLs are blocked in production mode"
    blocked_provider = get_tool("search.web").run({"query": "x", "provider": "unknown"})
    assert blocked_provider.status == "blocked"


def test_source_quality_tool_explains_boosts_and_penalties():
    official = get_tool("source.quality").run(
        {
            "url": "https://www.lenovo.com.cn/legion/pricing",
            "title": "联想拯救者官方产品页",
            "snippet": "价格 配置 RTX Y9000",
            "query": "联想拯救者 价格 配置",
        }
    )
    assert official.status == "ok"
    assert official.data["source_quality_score"] >= 0.75
    assert "official_domain" in official.data["boosts"]
    stale = get_tool("source.quality").run(
        {
            "url": "https://example.com/2019-used-laptop",
            "title": "2019款 二手 拯救者",
            "snippet": "95新 二手价格",
            "query": "联想拯救者 价格 配置",
        }
    )
    assert stale.data["trust_tier"] == "low"
    assert stale.data["reject_reason"] == "stale_price_evidence"
    assert "used_or_refurbished" in stale.data["penalties"]


def test_tool_executor_emits_audited_events_and_sanitizes_inputs(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("tool-audit", {"quality_mode": "production"})
    task_id = store.create_task(run_id, "Search", "SearchAgent")
    result, tool_call_id = ToolExecutor(store).run(
        "search.web",
        {
            "query": "ExampleCo pricing",
            "provider": "static",
            "source_urls": ["https://example.com/pricing"],
            "api_key": "sk-secret",
            "text": "x" * 1000,
        },
        ToolContext(run_id, task_id, "production", {"agent_name": "SearchAgent"}),
    )
    assert result.status == "ok"
    events = store.list_events(run_id, limit=10)
    event_types = [row["event_type"] for row in events]
    assert "tool_call_started" in event_types
    assert "tool_call_completed" in event_types
    completed = next(row for row in events if row["event_type"] == "tool_call_completed")
    metadata = loads(completed["metadata_json"], {})
    assert metadata["tool_call_id"] == tool_call_id
    assert metadata["tool_name"] == "search.web"
    assert "api_key" not in metadata["input_summary"]
    assert "text" not in metadata["input_summary"]
    json.dumps(metadata, ensure_ascii=True)


def test_tool_executor_blocks_policy_failures_without_throwing(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("tool-block", {"quality_mode": "production"})
    task_id = store.create_task(run_id, "Extract", "ExtractorAgent")
    result, tool_call_id = ToolExecutor(store).run(
        "fetch.url",
        {"url": "file:///C:/secret.txt"},
        ToolContext(run_id, task_id, "production", {"agent_name": "ExtractorAgent"}),
    )
    assert result.status == "blocked"
    events = store.list_events(run_id, limit=10)
    blocked = next(row for row in events if row["event_type"] == "tool_call_blocked")
    metadata = loads(blocked["metadata_json"], {})
    assert metadata["tool_call_id"] == tool_call_id
    assert metadata["tool_status"] == "blocked"


def test_browser_tools_registry_examples_and_policy_are_stable():
    tools = {tool.name: tool for tool in list_tools()}
    expected = {
        "browser.extract_code",
        "browser.plan_actions",
        "browser.promote_source",
        "browser.select_target",
        "browser.snapshot",
        "browser.page_state",
        "browser.visible_text",
        "browser.screenshot",
        "browser.goto",
        "browser.scroll",
        "browser.click",
        "browser.type",
        "browser.wait",
    }
    assert expected.issubset(tools)
    for name in expected:
        tool = tools[name]
        assert tool.allowed_callers == ["BrowserAgent"]
        assert tool.examples
        assert tool.example_failures
        json.dumps(tool.examples, ensure_ascii=True)
        json.dumps(tool.example_failures, ensure_ascii=True)


def test_browser_authorization_policy_classifies_allowed_auth_observation_and_blocked():
    allowed_context = ToolContext(quality_mode="test", metadata={"browser_allowed_domains": ["example.com"]})
    assert classify_authorization_need("goto", {"url": "https://example.com/pricing"}, allowed_context)[0] == "safe_auto"
    assert classify_authorization_need("goto", {"url": "https://mail.qq.com/"}, ToolContext(quality_mode="test"))[0] == "authorization_required"
    assert classify_authorization_need("type", {"target": "验证码", "text": "123456"}, ToolContext(quality_mode="test"))[0] == "assisted_observation_required"
    assert classify_authorization_need("click", {"target": "Submit payment"}, ToolContext(quality_mode="test"))[0] == "blocked"


def test_browser_extract_code_sandbox_outputs_candidates_and_trace(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-code", {"quality_mode": "production"})
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    page_state = {
        "url": "https://search.jd.com/Search?keyword=lenovo",
        "title": "联想电脑 - 商品搜索 - 京东",
        "interactable_elements": [
            {
                "stable_node_id": "product-1",
                "role": "link",
                "tag": "a",
                "text": "联想拯救者 笔记本电脑 商品详情",
                "href": "https://item.jd.com/10001.html",
                "bbox": {"x": 100, "y": 200, "width": 300, "height": 32},
            },
            {
                "stable_node_id": "service-1",
                "role": "button",
                "tag": "button",
                "text": "在线客服",
                "bbox": {"x": 88, "y": 210, "width": 24, "height": 24},
            },
        ],
    }
    result, tool_call_id = ToolExecutor(store).run(
        "browser.extract_code",
        {
            "session_id": "jd",
            "page_state": page_state,
            "code": "candidate_targets = classify_page_state(page_state)\nextracted_items = [{'title': candidate_targets[0]['text']}]\nwarnings = []",
        },
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert result.status == "ok"
    candidates = result.data["candidate_targets"]
    assert candidates[0]["stable_node_id"] == "product-1"
    assert candidates[0]["semantic_type"] == "product_detail_link"
    assert any(item["semantic_type"] == "customer_service" for item in candidates)
    artifacts = store.list_artifacts(run_id)
    code_artifact = next(row for row in artifacts if row["artifact_type"] == "browser_code_result")
    metadata = loads(code_artifact["metadata_json"], {})
    assert metadata["tool_call_id"] == tool_call_id
    assert metadata["candidate_target_count"] >= 2

    from insightswarm.observability.diagnosis import build_run_diagnosis, render_diagnosis_text
    from insightswarm.observability.trace import ensure_collaboration_trace

    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["browser_code_summary"]["code_execution_count"] == 1
    assert diagnosis["browser_code_summary"]["candidate_target_count"] >= 2
    assert "Browser Code Sandbox" in render_diagnosis_text(diagnosis)
    trace = ensure_collaboration_trace(store, run_id)["trace"]
    assert trace["browser_code_executions"][0]["tool_call_id"] == tool_call_id


def test_browser_select_target_prefers_product_detail_over_customer_service(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-target-select", {"quality_mode": "production"})
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    page_state = {
        "url": "https://search.jd.com/Search?keyword=lenovo",
        "title": "联想电脑 - 商品搜索 - 京东",
        "interactable_elements": [
            {
                "stable_node_id": "service-1",
                "dom_index": 1,
                "role": "link",
                "tag": "a",
                "text": "在线客服",
                "href": "https://chat.jd.com/",
                "bbox": {"x": 88, "y": 210, "width": 24, "height": 24},
                "container_context": "联想拯救者 在线客服 店铺",
            },
            {
                "stable_node_id": "product-1",
                "dom_index": 2,
                "role": "link",
                "tag": "a",
                "text": "联想拯救者 R9000P 2026 笔记本电脑",
                "href": "https://item.jd.com/10001.html",
                "bbox": {"x": 120, "y": 200, "width": 360, "height": 40},
                "container_context": "联想拯救者 R9000P 价格 ￥8999 自营 商品卡片",
            },
        ],
    }
    result, tool_call_id = ToolExecutor(store).run(
        "browser.select_target",
        {
            "intent": "第一个商品详情链接，不要客服",
            "page_state": page_state,
            "candidate_targets": [{"stable_node_id": "product-1", "semantic_type": "product_detail_link", "confidence": 0.9}],
        },
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert result.status == "ok"
    assert result.data["selected_target"]["stable_node_id"] == "product-1"
    assert result.data["selected_target"]["semantic_type"] == "product_detail_link"
    assert result.data["ranked_candidates"][0]["stable_node_id"] != "service-1"
    selection = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_target_selection")
    metadata = loads(selection["metadata_json"], {})
    assert metadata["tool_call_id"] == tool_call_id
    assert metadata["selected_semantic_type"] == "product_detail_link"

    from insightswarm.observability.diagnosis import build_run_diagnosis, render_diagnosis_text
    from insightswarm.observability.trace import ensure_collaboration_trace

    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["browser_target_selection_summary"]["selection_count"] == 1
    assert "Browser Target Selection" in render_diagnosis_text(diagnosis)
    trace = ensure_collaboration_trace(store, run_id)["trace"]
    assert trace["browser_target_selections"][0]["tool_call_id"] == tool_call_id


def test_browser_select_target_marks_service_only_as_disambiguation(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-target-service-only", {"quality_mode": "production"})
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    page_state = {
        "interactable_elements": [
            {
                "stable_node_id": "service-1",
                "role": "link",
                "tag": "a",
                "text": "在线客服",
                "href": "https://chat.jd.com/",
                "bbox": {"x": 10, "y": 10, "width": 20, "height": 20},
            }
        ]
    }
    result, _ = ToolExecutor(store).run(
        "browser.select_target",
        {"intent": "第一个商品详情链接，不要客服", "page_state": page_state},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert result.status == "ok"
    assert result.data["needs_human_disambiguation"] is True
    assert "top_candidate_is_customer_service" in result.data["reject_reasons"]
    blocked, _ = ToolExecutor(store).run(
        "browser.select_target",
        {"intent": "first product", "page_state": page_state},
        ToolContext(run_id, task_id, "production", {"agent_name": "SearchAgent"}),
    )
    assert blocked.status == "blocked"
    assert blocked.error == "browser tools are restricted to BrowserAgent"


def test_browser_action_request_references_target_selection(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-target-action", {"quality_mode": "production"})
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    page_state_result, _ = ToolExecutor(store).run(
        "browser.page_state",
        {"url": "https://example.com/pricing"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    page_state_id = write_browser_observation(store, run_id, task_id, "browser.page_state", page_state_result)
    page_state_payload = json.loads(Path(store.get_artifact(page_state_id)["path"]).read_text(encoding="utf-8"))
    page_state = page_state_payload["data"]["observation"]
    select_result, _ = ToolExecutor(store).run(
        "browser.select_target",
        {"intent": "pricing link", "page_state": page_state, "page_state_artifact_id": page_state_id},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert select_result.status == "ok"
    selection_id = next(row["artifact_id"] for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_target_selection")
    ToolExecutor(store).run(
        "browser.click",
        {"target": "selected target", "target_selection_artifact_id": selection_id, "page_state_artifact_id": page_state_id},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    request = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_action_request")
    payload = json.loads(Path(request["path"]).read_text(encoding="utf-8"))
    metadata = loads(request["metadata_json"], {})
    assert payload["target_selection_artifact_id"] == selection_id
    assert payload["selected_semantic_type"] == metadata["selected_semantic_type"]
    assert payload["selected_target_summary"]["stable_node_id"] == payload["target_id"]


def test_browser_plan_actions_structured_contract_and_policy(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-plan", {"quality_mode": "production"})
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    strict, _ = ToolExecutor(store).run(
        "browser.plan_actions",
        {"goal": "open first product", "mode": "strict"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert strict.status == "blocked"
    result, tool_call_id = ToolExecutor(store).run(
        "browser.plan_actions",
        {"goal": "open first product", "mode": "assisted", "page_state": {"interactable_elements": []}},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent", "browser_mode": "assisted"}),
    )
    assert result.status == "ok"
    plan = result.data["plan"]
    assert plan["schema"] == "browser_action_plan.v1"
    assert plan["mode"] == "assisted"
    assert all("tool_name" in step and "risk_status" in step and "requires_approval" in step for step in plan["steps"])
    artifact = [row for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_action_plan"][-1]
    metadata = loads(artifact["metadata_json"], {})
    assert metadata["tool_call_id"] == tool_call_id
    assert metadata["browser_mode"] == "assisted"
    blocked, _ = ToolExecutor(store).run(
        "browser.plan_actions",
        {"goal": "open first product", "mode": "assisted"},
        ToolContext(run_id, task_id, "production", {"agent_name": "SearchAgent"}),
    )
    assert blocked.status == "blocked"


def test_browser_free_mode_pauses_and_resumes_from_checkpoint(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-free-resume", {"quality_mode": "production"})
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    page_state_result, _ = ToolExecutor(store).run(
        "browser.page_state",
        {"url": "https://example.com/pricing"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    page_state_id = write_browser_observation(store, run_id, task_id, "browser.page_state", page_state_result)
    page_state = page_state_result.data["observation"]
    select_result, _ = ToolExecutor(store).run(
        "browser.select_target",
        {"intent": "open pricing", "page_state": page_state, "page_state_artifact_id": page_state_id},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert select_result.status == "ok"
    first = run_browser_operation(store, run_id, mode="free_browser", goal="open pricing", backend="fake")
    assert first["status"] == "pending_authorization"
    checkpoint = store.get_artifact(first["checkpoint_artifact_id"])
    checkpoint_payload = json.loads(Path(checkpoint["path"]).read_text(encoding="utf-8"))
    assert checkpoint_payload["step_index"] == 3
    request_id = first["request_artifact_id"]
    assert request_id
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "browser", "authorize", "--run-id", run_id, "--request-id", request_id, "--decision", "approve"]) == 0
    resumed = run_browser_operation(store, run_id, mode="free_browser", goal="open pricing", backend="fake")
    assert resumed["status"] == "completed"

    from insightswarm.observability.diagnosis import build_run_diagnosis, render_diagnosis_text
    from insightswarm.observability.trace import ensure_collaboration_trace

    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["browser_operation_summary"]["latest_mode"] == "free_browser"
    assert diagnosis["browser_operation_summary"]["authorization_request_count"] >= 1
    assert "Browser Operation Mode" in render_diagnosis_text(diagnosis)
    trace = ensure_collaboration_trace(store, run_id)["trace"]
    assert trace["browser_action_plans"]
    assert any(row["status"] == "pending_authorization" for row in trace["browser_operation_checkpoints"])


def test_browser_page_fingerprint_breakout_is_stable(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-fingerprint", {"quality_mode": "production"})
    result = run_browser_operation(store, run_id, mode="free_browser", goal="observe stable page", backend="fake", max_iterations=5)
    assert result["status"] == "fingerprint_breakout"
    checkpoints = [row for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_operation_checkpoint"]
    assert any(json.loads(Path(row["path"]).read_text(encoding="utf-8"))["status"] == "fingerprint_breakout" for row in checkpoints)
    assert page_fingerprint({"url": "u", "title": "t", "text_preview": "same"}) == page_fingerprint({"url": "u", "title": "t", "text_preview": "same"})


def test_browser_promote_source_creates_candidate_and_blocks_non_browser_callers(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-handoff", {"quality_mode": "production"})
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    result, tool_call_id = ToolExecutor(store).run(
        "browser.promote_source",
        {
            "source_url": "https://example.com/product",
            "title": "Example product",
            "text_preview": "ExampleCo product page. Starter plan costs $49 per month. token cookie header password",
        },
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent", "browser_mode": "free_browser"}),
    )
    assert result.status == "ok"
    candidate = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "candidate_source")
    metadata = loads(candidate["metadata_json"], {})
    payload = json.loads(Path(candidate["path"]).read_text(encoding="utf-8"))
    assert metadata["tool_call_id"] == tool_call_id
    assert metadata["source_kind"] == "browser_handoff"
    assert metadata["citation_ready"] is True
    serialized = json.dumps(payload, ensure_ascii=True).lower()
    assert "cookie" not in serialized
    assert "token" not in serialized
    assert "password" not in serialized
    blocked, _ = ToolExecutor(store).run(
        "browser.promote_source",
        {"source_url": "https://example.com/product", "text_preview": "text"},
        ToolContext(run_id, task_id, "production", {"agent_name": "SearchAgent"}),
    )
    assert blocked.status == "blocked"


def test_browser_promote_cli_converts_candidate_to_raw_document_and_extractor(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-handoff-raw", {"competitor": "ExampleCo", "quality_mode": "test", "model_provider": "fake"})
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    result, _ = ToolExecutor(store).run(
        "browser.promote_source",
        {
            "source_url": "http://localhost:9876/product",
            "title": "Example product",
            "text_preview": "ExampleCo browser handoff source. Starter plan costs $49 per month.",
        },
        ToolContext(run_id, task_id, "test", {"agent_name": "BrowserAgent", "browser_mode": "free_browser"}),
    )
    assert result.status == "ok"
    candidate_id = next(row["artifact_id"] for row in store.list_artifacts(run_id) if row["artifact_type"] == "candidate_source")
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "browser", "promote", "--run-id", run_id, "--candidate-id", candidate_id, "--quality-mode", "test"]) == 0
    raw_doc = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "raw_document")
    raw_meta = loads(raw_doc["metadata_json"], {})
    assert raw_meta["fetcher"] == "browser_agent_handoff"
    assert raw_meta["source_kind"] == "browser_handoff"
    assert raw_meta["browser_candidate_source_artifact_id"] == candidate_id
    assert raw_meta["requires_extractor"] is True

    from insightswarm.agents.extractor import ExtractorAgent
    from insightswarm.models.fake import FakeModelClient

    extractor_task = store.create_task(run_id, "Extract", "ExtractorAgent")
    ExtractorAgent(store, FakeModelClient()).execute(run_id, extractor_task)
    assert any(row["source_type"] == "document" for row in store.list_citations(run_id))
    inspected = json.loads(inspect_run(store, run_id))
    assert inspected["source_health"]["browser_handoff_count"] == 1
    summary = inspected["diagnosis"]["browser_evidence_handoff_summary"]
    assert summary["candidate_source_count"] == 1
    assert summary["promoted_raw_document_count"] == 1
    trace = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "collaboration_trace")
    trace_payload = json.loads(Path(trace["path"]).read_text(encoding="utf-8"))
    assert trace_payload["browser_candidate_sources"]
    assert trace_payload["browser_evidence_handoffs"]


def test_browser_handoff_blocks_bad_candidate_and_counts_formal_evidence(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "browser-handoff-production",
        {"query": "ExampleCo pricing", "competitor": "ExampleCo", "quality_mode": "production", "model_provider": "qwen_text"},
    )
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    bad, _ = ToolExecutor(store).run(
        "browser.promote_source",
        {"source_url": "http://localhost:9999/product", "text_preview": "ExampleCo pricing text."},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert bad.status == "blocked"
    result, _ = ToolExecutor(store).run(
        "browser.promote_source",
        {
            "source_url": "https://example.com/product",
            "title": "ExampleCo pricing",
            "text_preview": "ExampleCo browser handoff source. Enterprise plan costs $199 per month.",
        },
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert result.status == "ok"
    candidate_id = [row["artifact_id"] for row in store.list_artifacts(run_id) if row["artifact_type"] == "candidate_source"][-1]
    from insightswarm.browser_handoff import promote_candidate_to_raw_document

    promoted = promote_candidate_to_raw_document(store, run_id, candidate_id)
    assert promoted["status"] == "promoted"
    from insightswarm.agents.extractor import ExtractorAgent
    from insightswarm.models.fake import FakeModelClient

    extractor_task = store.create_task(run_id, "Extract", "ExtractorAgent")
    ExtractorAgent(store, FakeModelClient()).execute(run_id, extractor_task)
    inspected = json.loads(inspect_run(store, run_id))
    assert inspected["source_trust"]["formal_evidence_available"] is True
    assert inspected["source_trust"]["blocked_reason"] is None


def test_browser_extract_code_blocks_imports_io_and_non_browser_callers(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-code-block", {"quality_mode": "production"})
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    import_result, _ = ToolExecutor(store).run(
        "browser.extract_code",
        {"code": "import requests\ncandidate_targets = []"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert import_result.status == "error"
    assert "import statements" in (import_result.error or "")
    open_result, _ = ToolExecutor(store).run(
        "browser.extract_code",
        {"code": "data = open('secret.txt').read()"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert open_result.status == "error"
    assert "blocked function: open" in (open_result.error or "")
    search_task = store.create_task(run_id, "Discovery", "SearchAgent")
    blocked, _ = ToolExecutor(store).run(
        "browser.extract_code",
        {"code": "candidate_targets = []"},
        ToolContext(run_id, search_task, "production", {"agent_name": "SearchAgent"}),
    )
    assert blocked.status == "blocked"
    assert blocked.error == "browser tools are restricted to BrowserAgent"


def test_browser_tools_are_restricted_to_browser_agent(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-policy")
    task_id = store.create_task(run_id, "Discovery", "SearchAgent")
    result, _ = ToolExecutor(store).run(
        "browser.snapshot",
        {"session_id": "s1"},
        ToolContext(run_id, task_id, "production", {"agent_name": "SearchAgent"}),
    )
    assert result.status == "blocked"
    assert result.error == "browser tools are restricted to BrowserAgent"
    blocked = next(row for row in store.list_events(run_id, limit=10) if row["event_type"] == "tool_call_blocked")
    metadata = loads(blocked["metadata_json"], {})
    assert metadata["tool_name"] == "browser.snapshot"
    assert metadata["diagnostics"]["risk_reason"] == "caller_not_allowed"


def test_browser_safe_fake_tool_writes_observation_and_diagnosis(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-safe")
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    result, tool_call_id = ToolExecutor(store).run(
        "browser.snapshot",
        {"url": "https://example.com/pricing"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert result.status == "ok"
    artifact_id = write_browser_observation(store, run_id, task_id, "browser.snapshot", result, source_url="https://example.com/pricing")
    artifact = store.get_artifact(artifact_id)
    assert artifact["artifact_type"] == "browser_observation"
    assert loads(artifact["metadata_json"], {})["risk_status"] == "safe_auto"
    events = store.list_events(run_id, limit=20)
    assert any(row["event_type"] == "tool_call_started" for row in events)
    completed = next(row for row in events if row["event_type"] == "tool_call_completed")
    metadata = loads(completed["metadata_json"], {})
    assert metadata["tool_call_id"] == tool_call_id
    assert metadata["diagnostics"]["risk_status"] == "safe_auto"
    from insightswarm.observability.diagnosis import build_run_diagnosis, render_diagnosis_text

    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["browser_sandbox_summary"]["browser_tool_call_count"] == 1
    assert diagnosis["browser_sandbox_summary"]["risk_status_counts"]["safe_auto"] == 1
    assert diagnosis["browser_sandbox_summary"]["backend_counts"]["fake"] == 1
    assert "Browser Sandbox" in render_diagnosis_text(diagnosis)


def test_browser_page_state_fake_tool_writes_artifact_diagnosis_and_trace(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-page-state")
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    result, tool_call_id = ToolExecutor(store).run(
        "browser.page_state",
        {"url": "https://example.com/pricing", "max_elements": 1, "max_text_chars": 80},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert result.status == "ok"
    observation = result.data["observation"]
    assert observation["url"] == "https://example.com/pricing"
    assert observation["interactable_elements"][0]["stable_node_id"]
    assert observation["truncated"] is True
    serialized = json.dumps(result.to_dict(), ensure_ascii=True).lower()
    assert "cookie" not in serialized
    assert "localstorage" not in serialized
    assert "password" not in serialized
    artifact_id = write_browser_observation(store, run_id, task_id, "browser.page_state", result)
    artifact = store.get_artifact(artifact_id)
    assert artifact["artifact_type"] == "browser_page_state"
    metadata = loads(artifact["metadata_json"], {})
    assert metadata["browser_backend"] == "fake"
    assert metadata["read_only"] is True
    assert metadata["interactable_count"] == 2
    assert metadata["truncated"] is True

    from insightswarm.observability.diagnosis import build_run_diagnosis, render_diagnosis_text
    from insightswarm.observability.trace import ensure_collaboration_trace

    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["browser_page_state_summary"]["page_state_count"] == 1
    assert diagnosis["browser_page_state_summary"]["latest_page_state_url"] == "https://example.com/pricing"
    assert diagnosis["browser_page_state_summary"]["page_state_truncated"] is True
    assert "Browser Page State" in render_diagnosis_text(diagnosis)
    trace = ensure_collaboration_trace(store, run_id)["trace"]
    assert trace["browser_page_states"][0]["artifact_id"] == artifact_id
    assert any(call["tool_call_id"] == tool_call_id for call in trace["tool_calls"])


def test_browser_cdp_backend_unavailable_is_structured(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-cdp-missing")
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    result, _ = ToolExecutor(store).run(
        "browser.snapshot",
        {"backend": "cdp"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert result.status == "error"
    assert result.diagnostics["error_kind"] == "browser_backend_unavailable"
    assert result.diagnostics["browser_backend"] == "cdp"
    assert result.diagnostics.get("browser_backend_unavailable") is True
    from insightswarm.observability.diagnosis import build_run_diagnosis

    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["browser_sandbox_summary"]["backend_counts"]["cdp"] == 1
    assert diagnosis["browser_sandbox_summary"]["cdp_unavailable_or_error_count"] == 1
    assert diagnosis["browser_sandbox_summary"]["latest_browser_backend_error"]["tool_name"] == "browser.snapshot"


def test_browser_page_state_cdp_unavailable_is_structured(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-page-state-cdp-missing")
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    result, _ = ToolExecutor(store).run(
        "browser.page_state",
        {"backend": "cdp"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert result.status == "error"
    assert result.diagnostics["error_kind"] == "browser_backend_unavailable"
    assert result.diagnostics["browser_backend"] == "cdp"
    from insightswarm.observability.diagnosis import build_run_diagnosis

    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["browser_page_state_summary"]["cdp_page_state_error_count"] == 1


def test_browser_observation_metadata_records_backend(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-observation-backend")
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    result, _ = ToolExecutor(store).run(
        "browser.visible_text",
        {"backend": "fake", "session_id": "session-a"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    artifact_id = write_browser_observation(store, run_id, task_id, "browser.visible_text", result)
    metadata = loads(store.get_artifact(artifact_id)["metadata_json"], {})
    assert metadata["browser_backend"] == "fake"
    assert metadata["read_only"] is True
    assert metadata["fake_execution"] is True
    assert metadata["risk_status"] == "safe_auto"


def test_browser_session_cdp_requires_url_without_crashing():
    session = BrowserSession(backend="cdp")
    with pytest.raises(Exception) as excinfo:
        session.connect()
    assert "cdp_url is required" in str(excinfo.value) or "optional browser extra" in str(excinfo.value)


def test_browser_review_required_actions_emit_human_approval_and_trace(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-hitl")
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    result, _ = ToolExecutor(store).run(
        "browser.click",
        {"target": "Pricing tab", "snapshot_artifact_id": "artifact_snapshot"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert result.status == "blocked"
    events = store.list_events(run_id, limit=20)
    approval = next(row for row in events if row["event_type"] == "browser_human_approval_required")
    approval_meta = loads(approval["metadata_json"], {})
    assert approval_meta["action"] == "click"
    assert approval_meta["target_summary"] == "Pricing tab"
    assert approval_meta["allowed_choices"] == ["approve_once", "reject", "manual_capture_instead"]
    requests = [row for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_action_request"]
    assert len(requests) == 1
    request_payload = json.loads(Path(requests[0]["path"]).read_text(encoding="utf-8"))
    assert request_payload["action"] == "click"
    assert request_payload["status"] == "pending"
    assert request_payload["allowed_choices"] == ["approve_execute", "reject", "manual_capture_instead"]
    from insightswarm.observability.diagnosis import build_run_diagnosis
    from insightswarm.observability.trace import ensure_collaboration_trace

    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["browser_sandbox_summary"]["human_approval_request_count"] == 1
    assert diagnosis["browser_sandbox_summary"]["blocked_browser_action_count"] == 1
    trace = ensure_collaboration_trace(store, run_id)["trace"]
    event_types = [row["event_type"] for row in trace["timeline"]]
    assert "browser_human_approval_required" in event_types
    assert "browser_action_request_created" in event_types


def test_browser_approval_fake_execute_and_diagnosis_trace(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-approve", {"quality_mode": "production"})
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    page_state_result, _ = ToolExecutor(store).run(
        "browser.page_state",
        {"url": "https://example.com/pricing"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    page_state_id = write_browser_observation(store, run_id, task_id, "browser.page_state", page_state_result)
    result, _ = ToolExecutor(store).run(
        "browser.click",
        {"target": "Pricing", "target_id": "fake-link-pricing", "page_state_artifact_id": page_state_id},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert result.status == "blocked"
    request_id = next(row["artifact_id"] for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_action_request")
    approvals = list_browser_approvals(store, run_id)
    assert len(approvals["pending"]) == 1
    approved = approve_browser_action(store, run_id, request_id, execute=True, backend="fake")
    assert approved["status"] == "approved"
    assert approved["execution"]["status"] == "ok"
    artifacts = store.list_artifacts(run_id)
    assert any(row["artifact_type"] == "browser_approval_decision" for row in artifacts)
    assert any(row["artifact_type"] == "browser_action_execution" for row in artifacts)

    from insightswarm.observability.diagnosis import build_run_diagnosis, render_diagnosis_text
    from insightswarm.observability.trace import ensure_collaboration_trace

    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["browser_interaction_summary"]["approved_count"] == 1
    assert diagnosis["browser_interaction_summary"]["executed_action_count"] == 1
    assert "Browser Interaction" in render_diagnosis_text(diagnosis)
    trace = ensure_collaboration_trace(store, run_id)["trace"]
    assert trace["browser_interactions"]
    assert any(row["event_type"] == "browser_action_executed" for row in trace["timeline"])


def test_browser_reject_and_blocked_approval_paths(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-reject", {"quality_mode": "production"})
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    result, _ = ToolExecutor(store).run(
        "browser.goto",
        {"url": "https://example.com/pricing"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert result.status == "blocked"
    request_id = next(row["artifact_id"] for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_action_request")
    rejected = reject_browser_action(store, run_id, request_id, "not needed")
    assert rejected["status"] == "rejected"
    blocked_result, _ = ToolExecutor(store).run(
        "browser.goto",
        {"url": "file:///C:/secret.txt"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    assert blocked_result.status == "blocked"
    with pytest.raises(ValueError):
        approve_browser_action(store, run_id, request_id + "-missing", execute=True)


def test_browser_approval_requires_resolved_and_typeable_target(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-target", {"quality_mode": "production"})
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    page_state_result, _ = ToolExecutor(store).run(
        "browser.page_state",
        {"url": "https://example.com/pricing"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    page_state_id = write_browser_observation(store, run_id, task_id, "browser.page_state", page_state_result)
    ToolExecutor(store).run(
        "browser.click",
        {"target": "Missing", "target_id": "missing-node", "page_state_artifact_id": page_state_id},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    unresolved_request = next(row["artifact_id"] for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_action_request")
    with pytest.raises(ValueError, match="target is unresolved"):
        approve_browser_action(store, run_id, unresolved_request, execute=True)

    ToolExecutor(store).run(
        "browser.type",
        {"target": "Pricing", "target_id": "fake-link-pricing", "page_state_artifact_id": page_state_id, "text": "hello"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    type_request = [row["artifact_id"] for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_action_request"][-1]
    with pytest.raises(ValueError, match="input or textbox-like"):
        approve_browser_action(store, run_id, type_request, execute=True)


def test_browser_approval_cli_lists_and_executes(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "cli.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    store = Store(db_path, artifact_dir)
    run_id = store.create_run("browser-cli", {"quality_mode": "production"})
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    ToolExecutor(store).run(
        "browser.goto",
        {"url": "https://example.com/pricing"},
        ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"}),
    )
    request_id = next(row["artifact_id"] for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_action_request")
    monkeypatch.setenv("INSIGHTSWARM_DB_PATH", str(db_path))
    monkeypatch.setenv("INSIGHTSWARM_ARTIFACT_DIR", str(artifact_dir))
    assert cli_main(["--db-path", str(db_path), "--artifact-dir", str(artifact_dir), "browser", "approvals", "--run-id", run_id]) == 0
    approvals_output = capsys.readouterr().out
    assert request_id in approvals_output
    assert cli_main(["--db-path", str(db_path), "--artifact-dir", str(artifact_dir), "browser", "approve", "--run-id", run_id, "--request-id", request_id, "--execute", "--backend", "fake"]) == 0
    approve_output = capsys.readouterr().out
    assert "execution_id" in approve_output
    assert cli_main(["--db-path", str(db_path), "--artifact-dir", str(artifact_dir), "browser", "run", "--run-id", run_id, "--mode", "assisted", "--goal", "inspect pricing"]) == 0
    run_output = capsys.readouterr().out
    assert "plan_artifact_id" in run_output


def test_browser_tools_block_high_risk_and_production_local_navigation(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-block")
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    context = ToolContext(run_id, task_id, "production", {"agent_name": "BrowserAgent"})
    risky, _ = ToolExecutor(store).run("browser.type", {"target": "password field", "text": "secret"}, context)
    local, _ = ToolExecutor(store).run("browser.goto", {"url": "http://localhost:9999"}, context)
    test_local, _ = ToolExecutor(store).run(
        "browser.goto",
        {"url": "http://localhost:9999"},
        ToolContext(run_id, task_id, "test", {"agent_name": "BrowserAgent"}),
    )
    assert risky.status == "blocked"
    assert risky.diagnostics["risk_reason"] == "blocked_keyword:password"
    assert local.status == "blocked"
    assert local.error == "Localhost URLs are blocked in production mode"
    assert test_local.status == "blocked"
    assert test_local.diagnostics["human_approval_required"] is True


def test_browser_agent_context_only_exposes_browser_tools(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-context")
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    from insightswarm.context.builder import ContextBuilder

    payload, _ = ContextBuilder(store).build_and_persist(run_id, task_id, [])
    tool_names = {tool["name"] for tool in payload["available_tools"]}
    assert tool_names == {
        "browser.extract_code",
        "browser.plan_actions",
        "browser.promote_source",
        "browser.select_target",
        "browser.snapshot",
        "browser.page_state",
        "browser.visible_text",
        "browser.screenshot",
        "browser.goto",
        "browser.scroll",
        "browser.click",
        "browser.type",
        "browser.wait",
    }
    assert "search.web" not in tool_names
    assert "fetch.url" not in tool_names


def test_browser_collector_ingest_writes_artifacts_and_sanitizes_payload(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("collector", {"quality_mode": "test"})
    result = ingest_collector_payload(
        store,
        run_id,
        {
            "source_url": "http://localhost:9876/product",
            "title": "ExampleCo pricing",
            "captured_at": "2026-05-22T10:00:00Z",
            "visible_text": "ExampleCo pricing is $49 per month. " + "v" * 13000,
            "selected_text": "Selected pricing fact. " + "s" * 5000,
            "html_excerpt": "<html>" + "h" * 9000,
            "collector": {"name": "test", "token": "secret-token"},
            "page_metadata": {
                "lang": "en",
                "cookie": "session=secret",
                "authorization": "Bearer secret",
                "localStorage": {"token": "secret"},
                "password": "secret",
            },
        },
    )
    assert result["status"] == "ok"
    artifacts = store.list_artifacts(run_id)
    collected = next(row for row in artifacts if row["artifact_type"] == "browser_collected_page")
    raw = next(row for row in artifacts if row["artifact_type"] == "raw_document")
    collected_payload = json.loads(Path(collected["path"]).read_text(encoding="utf-8"))
    serialized = json.dumps(collected_payload, ensure_ascii=True).lower()
    assert "secret-token" not in serialized
    assert "session=secret" not in serialized
    assert "authorization" not in serialized
    assert "localstorage" not in serialized
    assert "password" not in serialized
    assert "visible_text_truncated" in collected_payload["sanitizer_warnings"]
    raw_meta = loads(raw["metadata_json"], {})
    assert raw_meta["fetcher"] == "browser_extension_collector"
    assert raw_meta["collector_payload_artifact_id"] == collected["artifact_id"]
    assert raw_meta["manual_browser_capture"] is True
    assert raw_meta["sanitizer_warning_count"] >= 3


def test_browser_collector_rejects_invalid_and_production_local_urls(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("collector-reject", {"quality_mode": "production"})
    file_result = ingest_collector_payload(store, run_id, {"source_url": "file:///C:/secret.txt"})
    local_result = ingest_collector_payload(store, run_id, {"source_url": "http://localhost:9876/page"})
    assert file_result["status"] == "blocked"
    assert local_result["status"] == "blocked"
    assert not store.list_artifacts(run_id)
    events = store.list_events(run_id, limit=10)
    assert sum(1 for row in events if row["event_type"] == "collector_payload_rejected") == 2


def test_browser_collector_raw_document_can_feed_extractor(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "collector-extract",
        {"competitor": "ExampleCo", "quality_mode": "test", "model_provider": "fake"},
    )
    ingest_collector_payload(
        store,
        run_id,
        {
            "source_url": "http://localhost:9876/pricing",
            "title": "ExampleCo pricing",
            "visible_text": "ExampleCo pricing page. Starter plan costs $49 per month.",
            "collector": {"name": "test"},
        },
    )
    task_id = store.create_task(run_id, "Extract", "ExtractorAgent")
    from insightswarm.agents.extractor import ExtractorAgent
    from insightswarm.models.fake import FakeModelClient

    ExtractorAgent(store, FakeModelClient()).execute(run_id, task_id)
    citations = store.list_citations(run_id)
    assert any(row["source_type"] == "document" for row in citations)
    structured = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "structured_knowledge")
    assert loads(store.get_task(task_id)["metadata_json"], {})["accepted_fact_count"] >= 1
    assert Path(structured["path"]).exists()


def test_browser_collector_visible_in_diagnosis_cli_and_trace(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("collector-diagnosis", {"quality_mode": "test"})
    ingest_collector_payload(
        store,
        run_id,
        {
            "source_url": "http://localhost:9876/page",
            "title": "Collected",
            "visible_text": "Collected source text with $49 pricing.",
        },
    )
    blocked = ingest_collector_payload(store, run_id, {"source_url": "file:///C:/secret.txt"})
    assert blocked["status"] == "blocked"
    inspected = json.loads(inspect_run(store, run_id))
    summary = inspected["diagnosis"]["browser_collector_summary"]
    assert summary["collector_artifact_count"] == 1
    assert summary["raw_document_count"] == 1
    assert summary["rejected_payload_count"] == 1
    assert summary["latest_collected_url"] == "http://localhost:9876/page"
    assert inspected["source_health"]["browser_collector_count"] == 1
    text = cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "diagnose", "--run-id", run_id])
    assert text == 0
    trace = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "collaboration_trace")
    trace_payload = json.loads(Path(trace["path"]).read_text(encoding="utf-8"))
    event_types = [item["event_type"] for item in trace_payload["timeline"]]
    assert "collector_payload_received" in event_types
    assert "collector_payload_rejected" in event_types


def test_research_contract_summary_handles_missing_file(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("missing-contract")
    artifact_id = store.write_artifact(
        run_id,
        None,
        "research_contract",
        "application/json",
        json.dumps(default_research_contract({"competitor": "ExampleCo"})),
        suffix=".json",
    )
    Path(store.get_artifact(artifact_id)["path"]).unlink()
    assert research_contract_summary(store, run_id) is None


def test_context_policy_decisions_are_role_scoped():
    policy = ContextPolicy()
    own = {"artifact_type": "raw_document", "task_id": "task_a", "source_url": "https://a.example"}
    other = {"artifact_type": "raw_document", "task_id": "task_b", "source_url": "https://b.example"}
    contract = {"artifact_type": "research_contract", "task_id": "lead", "source_url": None}
    assert policy.include_artifact(contract, "lead", {}, "ResearchLeadAgent").include is False
    assert policy.include_artifact(contract, "search", {}, "SearchAgent").include is True
    assert policy.include_artifact(own, "task_a", {"source_url": "https://a.example"}, "ExtractorAgent").include is True
    assert policy.include_artifact(other, "task_a", {"source_url": "https://a.example"}, "ExtractorAgent").include is False
    assert policy.include_artifact(other, "writer", {}, "WriterAgent").include is False
    structured = {"artifact_type": "structured_knowledge", "task_id": "extract", "source_url": None}
    assert policy.include_artifact(structured, "writer", {}, "WriterAgent").include is True
    analysis = {"artifact_type": "strategic_analysis", "task_id": "analyst", "source_url": None}
    skeptic_review = {"artifact_type": "skeptic_review", "task_id": "review", "source_url": None}
    assert policy.include_artifact(analysis, "review", {}, "SkepticReviewAgent").include is True
    assert policy.include_artifact(skeptic_review, "extract", {}, "ExtractorAgent").include is False
    assert policy.include_artifact(skeptic_review, "qa", {}, "QAAgent").include is True
    assert policy.include_artifact(skeptic_review, "writer", {}, "WriterAgent").include is True


def test_run_metadata_inputs(tmp_path):
    store = make_store(tmp_path)
    text_file = tmp_path / "source.txt"
    screenshot_file = tmp_path / "screen.png"
    text_file.write_text("ExampleCo pricing is $29 per month.", encoding="utf-8")
    screenshot_file.write_bytes(b"png")
    pdf_text_file = tmp_path / "pricing_pages.json"
    pdf_text_file.write_text('{"pages":[{"page_number":1,"text":"PDF price is $99."}]}', encoding="utf-8")
    run_id = store.create_run(
        "inputs",
        {
            "competitor": "ExampleCo",
            "source_urls": ["https://example.com/pricing"],
            "source_text_file": str(text_file),
            "source_pdf_text_file": str(pdf_text_file),
            "screenshot_file": str(screenshot_file),
            "model_provider": "qwen_text",
        },
    )
    metadata = store.get_run_metadata(run_id)
    assert metadata["competitor"] == "ExampleCo"
    assert metadata["source_urls"] == ["https://example.com/pricing"]
    assert metadata["source_text_file"].endswith("source.txt")
    assert metadata["source_pdf_text_file"].endswith("pricing_pages.json")
    assert metadata["screenshot_file"].endswith("screen.png")


def test_qwen_missing_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    client = QwenOpenAICompatibleClient("qwen_text", "qwen3.6-35b-a3b")
    with pytest.raises(QwenConfigError):
        client.complete([{"role": "user", "content": "hi"}])


def test_qwen_mocked_response(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "{\"competitor\":\"ExampleCo\",\"facts\":[]}"
                            }
                        }
                    ],
                    "usage": {"total_tokens": 3},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert request.full_url.endswith("/chat/completions")
        assert request.headers["Authorization"] == "Bearer test-key"
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = QwenOpenAICompatibleClient("qwen_text", "qwen3.6-35b-a3b")
    result = client.complete([{"role": "user", "content": "hi"}])
    assert result.provider == "qwen_text"
    assert result.model == "qwen3.6-35b-a3b"
    assert result.json_data["competitor"] == "ExampleCo"


def test_qwen_timeout_returns_error_result(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    def fake_urlopen(request, timeout):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = QwenOpenAICompatibleClient("qwen_text", "qwen3.6-35b-a3b")
    result = client.complete([{"role": "user", "content": "hi"}])
    assert result.status == "error"
    assert result.error and "timed out" in result.error
    assert result.provider == "qwen_text"


def test_m3_text_only_e2e_outputs_delivery_bundle(tmp_path):
    store = make_store(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text(
        "ExampleCo pricing page. Starter plan is $29 per user per month. "
        "The product emphasizes analytics collaboration.",
        encoding="utf-8",
    )
    run_id = store.create_run(
        "m3-text",
        {
            "competitor": "ExampleCo",
            "source_urls": ["https://example.com/pricing"],
            "source_text_file": str(source),
            "screenshot_file": None,
            "model_provider": "fake",
        },
    )
    Runner(store).start(run_id)
    assert store.get_run(run_id)["status"] == "completed"
    artifacts = store.list_artifacts(run_id)
    assert any(row["artifact_type"] == "visual_analysis" for row in artifacts)
    visual = next(row for row in artifacts if row["artifact_type"] == "visual_analysis")
    assert json.loads(Path(visual["path"]).read_text(encoding="utf-8"))["status"] == "skipped"
    assert any(row["artifact_type"] == "citations_export" for row in artifacts)
    assert any(row["artifact_type"] == "qa_report_export" for row in artifacts)
    report = next(row for row in artifacts if row["artifact_type"] == "report")
    report_text = Path(report["path"]).read_text(encoding="utf-8")
    assert "ExampleCo" in report_text
    assert "[[doc:" in report_text
    assert "[[inf:" in report_text


def test_pdf_text_json_source_generates_artifacts_diagnosis_and_trace(tmp_path):
    store = make_store(tmp_path)
    pdf_text = tmp_path / "pricing_pages.json"
    pdf_text.write_text(
        json.dumps(
            {
                "source_url": "https://example.com/pricing.pdf",
                "title": "ExampleCo Pricing PDF",
                "pages": [
                    {"page_number": 1, "text": "ExampleCo pricing overview."},
                    {"text": "Starter plan costs $49 per month for analytics teams."},
                    {"page_number": 3, "text": ""},
                ],
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    run_id = store.create_run(
        "pdf-text",
        {
            "competitor": "ExampleCo",
            "source_urls": ["https://example.com/pricing.pdf"],
            "source_pdf_text_file": str(pdf_text),
            "model_provider": "fake",
        },
    )
    Runner(store).start(run_id)
    assert store.get_run(run_id)["status"] == "completed"
    artifacts = store.list_artifacts(run_id)
    pdf_source = next(row for row in artifacts if row["artifact_type"] == "pdf_text_source")
    raw_doc = next(
        row
        for row in artifacts
        if row["artifact_type"] == "raw_document"
        and loads(row["metadata_json"], {}).get("fetcher") == "pdf_text_source"
    )
    raw_meta = loads(raw_doc["metadata_json"], {})
    assert raw_meta["source_kind"] == "pdf_text"
    assert raw_meta["pdf_text_source_artifact_id"] == pdf_source["artifact_id"]
    assert raw_meta["page_count"] == 3
    assert raw_meta["manual_file_source"] is True
    assert "missing_page_number" in raw_meta["warnings"]
    assert "empty_page_text" in raw_meta["warnings"]
    assert any(row["source_type"] == "document" for row in store.list_citations(run_id))
    inspected = json.loads(inspect_run(store, run_id))
    summary = inspected["diagnosis"]["pdf_text_source_summary"]
    assert summary["pdf_text_source_count"] == 1
    assert summary["pdf_page_count"] == 3
    assert summary["latest_pdf_text_source"] == "https://example.com/pricing.pdf"
    assert "missing_page_number" in summary["pdf_text_source_warnings"]
    assert inspected["source_health"]["pdf_text_source_count"] == 1
    trace = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "collaboration_trace")
    trace_payload = json.loads(Path(trace["path"]).read_text(encoding="utf-8"))
    pdf_evidence = [item for item in trace_payload["evidence_chain"] if item.get("source_kind") == "pdf_text"]
    assert pdf_evidence
    assert pdf_evidence[0]["page_count"] == 3


def test_pdf_text_txt_source_can_feed_extractor(tmp_path):
    store = make_store(tmp_path)
    pdf_text = tmp_path / "pricing.txt"
    pdf_text.write_text(
        "ExampleCo PDF pricing sheet. Starter plan costs $39 per month.",
        encoding="utf-8",
    )
    run_id = store.create_run(
        "pdf-text-txt",
        {
            "competitor": "ExampleCo",
            "source_urls": ["https://example.com/pricing.pdf"],
            "source_pdf_text_file": str(pdf_text),
            "model_provider": "fake",
        },
    )
    Runner(store).start(run_id)
    raw_doc = next(
        row
        for row in store.list_artifacts(run_id)
        if row["artifact_type"] == "raw_document"
        and loads(row["metadata_json"], {}).get("fetcher") == "pdf_text_source"
    )
    assert loads(raw_doc["metadata_json"], {})["page_count"] == 1
    assert any(row["source_type"] == "document" for row in store.list_citations(run_id))


def test_pdf_text_source_counts_as_formal_production_evidence(tmp_path):
    store = make_store(tmp_path)
    pdf_text = tmp_path / "pricing_pages.json"
    pdf_text.write_text(
        json.dumps(
            {
                "source_url": "https://example.com/pricing.pdf",
                "pages": [{"page_number": 1, "text": "ExampleCo enterprise plan costs $199 per month."}],
            }
        ),
        encoding="utf-8",
    )
    run_id = store.create_run(
        "pdf-production",
        {
            "query": "ExampleCo pricing",
            "competitor": "ExampleCo",
            "source_urls": ["https://example.com/pricing.pdf"],
            "source_pdf_text_file": str(pdf_text),
            "model_provider": "qwen_text",
            "quality_mode": "production",
        },
    )
    scraper_task = store.create_task(run_id, "Discovery", "ScraperAgent")
    from insightswarm.agents.scraper import ScraperAgent
    from insightswarm.models.fake import FakeModelClient

    ScraperAgent(store, FakeModelClient()).execute(run_id, scraper_task)
    extractor_task = store.create_task(run_id, "Extract", "ExtractorAgent")
    from insightswarm.agents.extractor import ExtractorAgent

    ExtractorAgent(store, FakeModelClient()).execute(run_id, extractor_task)
    inspected = json.loads(inspect_run(store, run_id))
    assert inspected["source_trust"]["formal_evidence_available"] is True
    assert inspected["source_trust"]["real_evidence_count"] >= 1
    assert inspected["source_trust"]["blocked_reason"] is None


def test_pdf_text_empty_source_records_warning_without_raw_document(tmp_path):
    store = make_store(tmp_path)
    pdf_text = tmp_path / "empty_pages.json"
    pdf_text.write_text(
        json.dumps({"source_url": "https://example.com/empty.pdf", "pages": [{"page_number": 1, "text": ""}]}),
        encoding="utf-8",
    )
    run_id = store.create_run(
        "pdf-empty",
        {
            "competitor": "ExampleCo",
            "source_urls": ["https://example.com/empty.pdf"],
            "source_pdf_text_file": str(pdf_text),
            "model_provider": "fake",
        },
    )
    scraper_task = store.create_task(run_id, "Discovery", "ScraperAgent")
    from insightswarm.agents.scraper import ScraperAgent
    from insightswarm.models.fake import FakeModelClient

    ScraperAgent(store, FakeModelClient()).execute(run_id, scraper_task)
    artifacts = store.list_artifacts(run_id)
    assert any(row["artifact_type"] == "pdf_text_source_warning" for row in artifacts)
    assert not any(
        row["artifact_type"] == "raw_document"
        and loads(row["metadata_json"], {}).get("fetcher") == "pdf_text_source"
        for row in artifacts
    )


def test_spa_fallback_heuristics_respect_static_structure():
    fallback, meta = should_fallback_to_browser("<html><body><p>Price</p></body></html>", "Price")
    assert fallback is False
    assert meta["reason"] == "short_text_but_static_structure"
    fallback, meta = should_fallback_to_browser("<html><body><div id='root'></div></body></html>", "")
    assert fallback is True
    assert meta["reason"] == "short_text_and_spa_shell"


def test_html_text_normalization_and_price_equivalence():
    assert normalize_html_text("<html><body><p> Hello <strong>world</strong> </p></body></html>") == "Hello world"
    assert normalize_currency_label("¥") == "CNY"
    assert normalize_billing_cycle("年付") == "yearly"
    assert parse_price_snapshot("$12 per month")["currency"] == "USD"
    assert prices_are_equivalent("$12 per month", "$144 per year")
    assert prices_are_equivalent("¥12/月", "RMB 144/年")


def test_find_text_span_matches_equivalent_yuan_symbols():
    text = "纯电动BMW i3 车型售价 ￥278,000起 预约试驾"
    span = find_text_span(text, "纯电动BMW i3 车型售价 ¥278,000起")
    assert text[span["start"] : span["end"]] == "纯电动BMW i3 车型售价 ￥278,000起"


def test_scraper_manifest_records_fetch_failures_and_successes(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "fetch",
        {
            "competitor": "ExampleCo",
            "source_urls": ["https://a.example", "https://b.example"],
            "model_provider": "fake",
        },
    )
    scraper_task = store.create_task(run_id, "Discovery", "ScraperAgent")
    from insightswarm.agents.scraper import ScraperAgent
    from insightswarm.models.fake import FakeModelClient

    def fake_fetch(url, timeout=20.0):
        if url.endswith("a.example"):
            return FetchResult(
                source_url=url,
                fetcher="httpx",
                status="ok",
                html="<html><body><p>ExampleCo pricing is $12 per month.</p></body></html>",
                text="ExampleCo pricing is $12 per month.",
                latency_ms=7,
                metadata={"attempts": [{"fetcher": "httpx"}]},
            )
        return FetchResult(
            source_url=url,
            fetcher="playwright",
            status="error",
            error="boom",
            latency_ms=11,
            metadata={"attempts": [{"fetcher": "httpx", "status": "error"}]},
        )

    monkeypatch.setattr("insightswarm.agents.scraper.fetch_source", fake_fetch)
    agent = ScraperAgent(store, FakeModelClient())
    agent.execute(run_id, scraper_task)
    artifacts = store.list_artifacts(run_id)
    assert any(row["artifact_type"] == "fetch_manifest" for row in artifacts)
    assert any(row["artifact_type"] == "fetch_failure" for row in artifacts)
    assert any(row["artifact_type"] == "raw_document" for row in artifacts)
    manifest = next(row for row in artifacts if row["artifact_type"] == "fetch_manifest")
    payload = json.loads(Path(manifest["path"]).read_text(encoding="utf-8"))
    assert payload["success_count"] == 1
    assert payload["failure_count"] == 1


def test_static_search_client_and_link_gate_rule_fallback():
    client = StaticSearchClient(["https://example.com/a", "https://example.com/b"])
    batch = client.search("example query", limit=2)
    assert batch.status == "ok"
    assert [result.url for result in batch.results] == ["https://example.com/a", "https://example.com/b"]
    model = build_model_client("fake")
    decisions, metadata = gate_links(model, "example query", batch.results, max_selected=2)
    assert metadata["strategy"] == "rule_fallback"
    assert [decision.url for decision in decisions] == ["https://example.com/a", "https://example.com/b"]


def test_link_gate_regex_recovery():
    class RegexModel:
        provider = "regex"
        model = "regex"

        def complete(self, *args, **kwargs):
            return ModelResult(
                text="I choose https://example.com/selected",
                json_data=None,
                provider="regex",
                model="regex",
                usage={},
                latency_ms=1,
                raw_response={},
                status="ok",
            )

    results = [SearchResult("Selected", "https://example.com/selected", "good snippet", 1)]
    decisions, metadata = gate_links(RegexModel(), "query", results, max_selected=1)
    assert metadata["strategy"] == "regex_recovery"
    assert decisions[0].url == "https://example.com/selected"


def test_search_query_dynamic_extract_and_snippet_first(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "query",
        {
            "query": "Lenovo Legion pricing",
            "competitor": "Lenovo",
            "source_urls": [
                "https://example.com/legion",
                "https://example.com/yoga",
            ],
            "model_provider": "fake",
            "search_provider": "static",
            "search_limit": 2,
            "link_gate_max_selected": 2,
        },
    )
    Runner(store).start(run_id)
    tasks = store.list_tasks(run_id)
    assert any(task["agent_name"] == "ResearchLeadAgent" for task in tasks)
    dynamic_extracts = [
        task
        for task in tasks
        if task["agent_name"] == "ExtractorAgent"
        and loads(task["metadata_json"], {}).get("dynamic_group_id")
    ]
    assert len(dynamic_extracts) == 2
    assert all(task["status"] == "completed" for task in dynamic_extracts)
    review_task = next(task for task in tasks if task["agent_name"] == "SkepticReviewAgent")
    qa_task = next(task for task in tasks if task["agent_name"] == "QAAgent")
    assert review_task["status"] == "completed"
    assert review_task["task_id"] in loads(qa_task["depends_on_json"], [])
    raw_docs = [row for row in store.list_artifacts(run_id) if row["artifact_type"] == "raw_document"]
    assert len(raw_docs) >= 2
    assert {loads(row["metadata_json"], {}).get("fetcher") for row in raw_docs}
    assert all(loads(row["metadata_json"], {}).get("tool_call_id") for row in raw_docs if loads(row["metadata_json"], {}).get("tool") == "fetch.url")
    link_gate = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "link_gate")
    link_payload = json.loads(Path(link_gate["path"]).read_text(encoding="utf-8"))
    assert link_payload["source_reviews"]
    assert link_payload["source_reviews"][0]["source_quality"]["trust_tier"] in {"high", "medium", "low"}
    trace = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "collaboration_trace")
    trace_payload = json.loads(Path(trace["path"]).read_text(encoding="utf-8"))
    trace_agents = [step["agent_name"] for step in trace_payload["role_steps"]]
    for agent_name in ["SearchAgent", "LinkGateAgent", "ExtractorAgent", "StrategicAnalystAgent", "SkepticReviewAgent", "QAAgent", "WriterAgent"]:
        assert agent_name in trace_agents
    assert trace_payload["tool_calls"]
    assert store.get_run(run_id)["status"] in {"completed", "blocked"}


def test_context_policy_scopes_dynamic_extractor_to_own_source(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "context-scope",
        {
            "query": "Lenovo Legion pricing",
            "competitor": "Lenovo",
            "source_urls": [
                "https://example.com/legion",
                "https://example.com/yoga",
            ],
            "model_provider": "fake",
            "search_provider": "static",
            "search_limit": 2,
            "link_gate_max_selected": 2,
            "quality_mode": "test",
        },
    )
    Runner(store).start(run_id)
    context_payloads = [
        json.loads(Path(row["path"]).read_text(encoding="utf-8"))
        for row in store.list_artifacts(run_id)
        if row["artifact_type"] == "context_envelope"
    ]
    extractor_payloads = [
        payload
        for payload in context_payloads
        if payload["task"]["agent_name"] == "ExtractorAgent"
        and payload["task"]["metadata"].get("source_url") == "https://example.com/legion"
    ]
    assert extractor_payloads
    search_payload = next(payload for payload in context_payloads if payload["task"]["agent_name"] == "SearchAgent")
    assert any(tool["name"] == "search.web" and tool["examples"] for tool in search_payload["available_tools"])
    assert search_payload["available_tools"][0]["allowed_callers"]
    assert search_payload["available_tools"][0]["example_failures"]
    link_gate_payload = next(payload for payload in context_payloads if payload["task"]["agent_name"] == "LinkGateAgent")
    assert any(tool["name"] == "source.quality" and tool["examples"] for tool in link_gate_payload["available_tools"])
    assert any(tool["name"] == "fetch.url" and tool["examples"] for tool in extractor_payloads[-1]["available_tools"])
    source_urls = {
        artifact["source_url"]
        for artifact in extractor_payloads[-1]["artifacts"]
        if artifact["source_url"]
    }
    assert "https://example.com/yoga" not in source_urls
    writer_payload = next(payload for payload in context_payloads if payload["task"]["agent_name"] == "WriterAgent")
    assert writer_payload["research_contract"]
    assert writer_payload["collaboration_protocol"]["role"] == "ReportWriter"


def test_phase_gate_blocks_synthesis_without_evidence(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("gate")
    analyst = store.create_task(run_id, "Synthesize", "StrategicAnalystAgent")
    Runner(store).start(run_id)
    assert store.get_run(run_id)["status"] == "blocked"
    assert store.get_task(analyst)["status"] == "blocked"
    artifacts = store.list_artifacts(run_id)
    assert any(row["artifact_type"] == "phase_gate_report" for row in artifacts)
    events = store.list_events(run_id, limit=20)
    assert any(row["event_type"] == "phase_gate_blocked" for row in events)


def test_run_inspect_reports_collaboration_summary(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("inspect-collab")
    Runner(store).start(run_id)
    payload = json.loads(inspect_run(store, run_id))
    assert "collaboration" in payload
    assert payload["collaboration"]["research_contract"]["schema"] == "research_contract.v1"
    assert payload["collaboration"]["roles"]
    assert payload["collaboration"]["role_progress"]
    assert payload["collaboration"]["handoff_chain"]
    assert payload["collaboration"]["contract_goal"]
    assert payload["collaboration"]["key_questions"]
    assert "contract_created" in payload["collaboration"]["convergence_status"]
    assert "evidence_extracted" in payload["collaboration"]["convergence_status"]
    assert "qa_passed" in payload["collaboration"]["convergence_status"]
    assert "delivered" in payload["collaboration"]["convergence_status"]
    assert payload["collaboration"]["skeptic_review"]["passed"] is True
    assert payload["collaboration"]["skeptic_review_artifact_count"] == 1
    assert payload["collaboration"]["latest_skeptic_review"]["non_blocking"] is True
    assert payload["collaboration"]["review_handoff"]["to"] == "SkepticQA"
    assert payload["collaboration"]["review_handoff"]["non_blocking"] is True
    assert "phase_gate_report_count" in payload["collaboration"]
    assert "repair_task_count" in payload["collaboration"]


def test_dynamic_extractor_uses_sufficient_snippet(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("snippet", {"competitor": "Lenovo"})
    task_id = store.create_task(
        run_id,
        "Extract",
        "ExtractorAgent",
        metadata={
            "source_url": "https://example.com/snippet",
            "snippet": (
                "Lenovo Legion laptop pricing starts at $1299 with gaming features, "
                "dedicated GPU options, high refresh displays, and cooling improvements."
            ),
        },
    )
    from insightswarm.agents.extractor import ExtractorAgent
    from insightswarm.models.fake import FakeModelClient

    ExtractorAgent(store, FakeModelClient()).execute(run_id, task_id)
    raw_doc = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "raw_document")
    assert loads(raw_doc["metadata_json"], {}).get("fetcher") == "tavily_snippet"


def test_extractor_discards_unverifiable_fact_without_failing_batch(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("discard-bad-fact", {"competitor": "BMW"})
    task_id = store.create_task(
        run_id,
        "Extract",
        "ExtractorAgent",
        metadata={
            "source_url": "https://example.com/bmw",
            "snippet": (
                "纯电动BMW i3 车型售价 ￥278,000起 预约试驾。"
                "纯电动BMW i4 车型售价 ￥348,000起 预约试驾。"
            ),
        },
    )

    class MixedFactModel:
        provider = "test"
        model = "mixed-facts"

        def complete(self, *args, **kwargs):
            return ModelResult(
                text="",
                json_data={
                    "competitor": "BMW",
                    "facts": [
                        {
                            "field": "pricing",
                            "value": "￥278,000起",
                            "quote": "纯电动BMW i3 车型售价 ¥278,000起",
                            "source_url": "https://example.com/bmw",
                            "confidence": 0.9,
                        },
                        {
                            "field": "pricing",
                            "value": "not present",
                            "quote": "纯电动BMW iX9 车型售价 ￥999,000起",
                            "source_url": "https://example.com/bmw",
                            "confidence": 0.9,
                        },
                    ],
                },
                provider="test",
                model="mixed-facts",
                usage={},
                latency_ms=1,
                raw_response={},
                status="ok",
            )

    from insightswarm.agents.extractor import ExtractorAgent

    ExtractorAgent(store, MixedFactModel()).execute(run_id, task_id)
    task = store.get_task(task_id)
    metadata = loads(task["metadata_json"], {})
    assert task["status"] == "completed"
    assert metadata["accepted_fact_count"] == 1
    assert metadata["discarded_fact_count"] == 1
    assert len(store.list_citations(run_id)) == 1
    events = store.list_events(run_id, limit=20)
    assert any(row["event_type"] == "extract_fact_discarded" for row in events)


def test_fetch_error_classification():
    assert classify_fetch_error("403 Forbidden") == "http_403"
    assert classify_fetch_error("429 Too Many Requests") == "http_429"
    assert classify_fetch_error("Page.goto Timeout exceeded") == "timeout"


def test_playwright_unavailable_metadata_is_diagnostic(monkeypatch):
    monkeypatch.setattr("insightswarm.fetching.sync_playwright", None)
    result = PlaywrightFetcher(timeout=0.1).fetch("http://localhost/unavailable")
    assert result.status == "error"
    assert result.metadata["backend"] == "unavailable"
    assert result.metadata["error_kind"] == "browser_unavailable"
    assert result.metadata["partial"] is False
    assert result.metadata["screenshot_captured"] is False
    assert result.metadata["html_chars"] == 0
    assert result.metadata["text_chars"] == 0


def test_diagnosis_browser_failure_metadata_and_recommendation(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "browser-diagnosis",
        {
            "query": "ExampleCo pricing",
            "competitor": "ExampleCo",
            "source_urls": ["https://example.invalid"],
            "model_provider": "test_real",
        },
    )
    store.write_artifact(
        run_id,
        None,
        "fetch_failure",
        "application/json",
        json.dumps(
            {
                "source_url": "https://example.invalid",
                "fetcher": "playwright",
                "status": "error",
                "error": "playwright unavailable",
                "fallback_reason": "httpx_failed",
                "metadata": {
                    "attempts": [
                        {"fetcher": "httpx", "status": "error", "metadata": {"error_kind": "timeout"}},
                        {
                            "fetcher": "playwright",
                            "status": "error",
                            "metadata": {
                                "backend": "unavailable",
                                "error_kind": "browser_unavailable",
                                "partial": False,
                                "screenshot_captured": False,
                            },
                        },
                    ]
                },
            },
            ensure_ascii=True,
        ),
        source_url="https://example.invalid",
        metadata={
            "fetcher": "playwright",
            "error": "playwright unavailable",
            "fallback_reason": "httpx_failed",
            "backend": "unavailable",
            "error_kind": "browser_unavailable",
            "attempts": [
                {"fetcher": "httpx", "status": "error", "metadata": {"error_kind": "timeout"}},
                {
                    "fetcher": "playwright",
                    "status": "error",
                    "metadata": {
                        "backend": "unavailable",
                        "error_kind": "browser_unavailable",
                        "partial": False,
                        "screenshot_captured": False,
                    },
                },
            ],
        },
        suffix=".json",
    )
    from insightswarm.observability.diagnosis import build_run_diagnosis, render_diagnosis_text

    diagnosis = build_run_diagnosis(store, run_id)
    coverage = diagnosis["fetch_coverage"]
    assert coverage["browser_attempt_count"] == 1
    assert coverage["browser_failure_count"] == 1
    assert coverage["browser_unavailable_count"] == 1
    assert coverage["browser_smoke_recommended"] is True
    assert diagnosis["source_failures"][0]["attempted_browser"] is True
    assert diagnosis["source_failures"][0]["browser_error_kind"] == "browser_unavailable"
    assert any("browser smoke" in action for action in diagnosis["recommended_next_actions"])
    assert "Browser Fetch" in render_diagnosis_text(diagnosis)


def test_document_cleaner_removes_common_web_noise_and_trims_quote():
    text = (
        "京东 你好，请登录 免费注册 我的订单 我的购物车 全部分类 "
        "联想拯救者 Y7000P 配置 RTX4060，售价 ¥6999，适合游戏和创作。"
        "返回搜狐，查看更多 平台声明 阅读 ()"
    )
    cleaned = DocumentCleaner().clean(text, query="联想拯救者 价格 配置", competitor="Lenovo")
    assert "你好，请登录" not in cleaned.cleaned_text
    assert "我的购物车" not in cleaned.cleaned_text
    assert "¥6999" in cleaned.cleaned_text
    assert cleaned.noise_removed_count > 0
    assert len(trim_quote("联想拯救者 " * 100, ["联想"], max_chars=80)) <= 80


def test_link_gate_rule_fallback_penalizes_stale_sources():
    results = [
        SearchResult(
            "2019款拯救者价格",
            "https://www.sohu.com/a/old",
            "2019款拯救者电脑价格汇总",
            1,
            metadata={"freshness_status": "stale"},
        ),
        SearchResult(
            "联想拯救者新品价格",
            "https://www.lenovo.com.cn/legion",
            "联想拯救者 新品 配置 价格 ¥6999",
            2,
            metadata={"freshness_status": "fresh"},
        ),
    ]
    decisions = rule_gate(results, max_selected=1)
    assert decisions[0].url == "https://www.lenovo.com.cn/legion"
    assert freshness_status("2019款拯救者价格") == "stale"


def test_query_fake_model_report_marks_real_model_required(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "quality",
        {
            "query": "Lenovo Legion pricing",
            "competitor": "Lenovo",
            "source_urls": ["https://example.com/legion"],
            "model_provider": "fake",
            "search_provider": "static",
            "search_limit": 1,
            "link_gate_max_selected": 1,
        },
    )
    Runner(store).start(run_id)
    assert store.get_run(run_id)["status"] == "blocked"
    qa_reports = [row for row in store.list_artifacts(run_id) if row["artifact_type"] == "qa_report"]
    assert qa_reports
    payload = json.loads(Path(qa_reports[-1]["path"]).read_text(encoding="utf-8"))
    assert any(failure["gate"] == "real_model_required" for failure in payload["rejection"]["failures"])
    artifacts = store.list_artifacts(run_id)
    assert not any(row["artifact_type"] == "report" for row in artifacts)
    assert any(row["artifact_type"] == "report_blocked" for row in artifacts)
    analysis_artifact = next(row for row in artifacts if row["artifact_type"] == "strategic_analysis")
    analysis_metadata = loads(analysis_artifact["metadata_json"], {})
    assert analysis_metadata["formal_result"] is False
    inspect_payload = json.loads(inspect_run(store, run_id))
    diagnosis = inspect_payload["diagnosis"]
    assert diagnosis["validation_categories"]["permission"]["blocking"] is True
    assert "permission" in diagnosis["blocking_validation_categories"]
    assert any(row["category"] == "permission" for row in diagnosis["qa_failure_summary"])
    assert any("real production model" in action for action in diagnosis["recommended_next_actions"])
    collaboration = inspect_payload["collaboration"]
    assert "blocked" in collaboration["convergence_status"]
    assert "delivered" in collaboration["convergence_status"]
    assert collaboration["skeptic_review"]["passed"] is False
    assert any(failure["gate"] == "real_model_required" for failure in collaboration["skeptic_review"]["failures"])


def test_synthetic_only_production_query_blocks_without_formal_evidence(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "synthetic-degraded",
        {
            "query": "ExampleCo pricing",
            "competitor": "ExampleCo",
            "source_urls": ["https://example.invalid/pricing"],
            "model_provider": "test_real",
            "search_provider": "static",
            "search_limit": 1,
            "link_gate_max_selected": 1,
        },
    )

    class RealishModel:
        provider = "test_real"
        model = "test-real"

        def complete(self, messages, response_format=None, metadata=None):
            role = (metadata or {}).get("role")
            if role == "StrategicAnalystAgent":
                return ModelResult(
                    text="",
                    json_data={
                        "inferences": [
                            {
                                "claim": "ExampleCo has visible pricing evidence.",
                                "evidence_ids": [row["citation_id"] for row in store.list_citations(run_id) if row["source_type"] == "document"],
                                "confidence": 0.8,
                            }
                        ]
                    },
                    provider=self.provider,
                    model=self.model,
                    usage={},
                    latency_ms=1,
                    raw_response={},
                    status="ok",
                )
            return build_model_client("fake").complete(messages, response_format=response_format, metadata=metadata)

        def analyze_image(self, *args, **kwargs):
            return build_model_client("fake").analyze_image(*args, **kwargs)

    def fake_fetch(url, timeout=20.0):
        return FetchResult(
            source_url=url,
            fetcher="httpx",
            status="error",
            error="offline",
            latency_ms=1,
            metadata={"attempts": [{"fetcher": "httpx", "status": "error"}]},
        )

    monkeypatch.setattr("insightswarm.agents.extractor.fetch_source", fake_fetch)
    runner = Runner(store)
    runner.model = RealishModel()
    runner.start(run_id)
    assert store.get_run(run_id)["status"] == "blocked"
    artifacts = store.list_artifacts(run_id)
    assert not any(row["artifact_type"] == "report" for row in artifacts)
    report_blocked = next(row for row in artifacts if row["artifact_type"] == "report_blocked")
    blocked_payload = json.loads(Path(report_blocked["path"]).read_text(encoding="utf-8"))
    assert blocked_payload["blocked_reason"] == "no_real_evidence"
    assert blocked_payload["source_trust"]["diagnostic_fallback_count"] >= 1
    assert not any(row["source_type"] == "document" for row in store.list_citations(run_id))
    inspect_payload = json.loads(inspect_run(store, run_id))
    assert inspect_payload["quality_status"] == "blocked_no_real_evidence"
    assert "diagnosis" in inspect_payload
    assert inspect_payload["diagnosis"]["source_trust_status"] == "blocked_no_real_evidence"
    assert inspect_payload["diagnosis"]["fetch_coverage"]["synthetic_diagnostic_fallback_count"] >= 1
    assert inspect_payload["diagnosis"]["source_failures"]
    assert inspect_payload["diagnosis"]["validation_categories"]["evidence"]["blocking"] is True
    assert "evidence" in inspect_payload["diagnosis"]["blocking_validation_categories"]
    assert any("manual primary source" in action for action in inspect_payload["diagnosis"]["recommended_next_actions"])
    assert any(reason["code"] == "no_real_evidence" for reason in inspect_payload["degraded_reasons"])
    assert inspect_payload["source_health"]["synthetic_fallback_count"] >= 1
    assert inspect_payload["real_evidence_count"] == 0
    assert inspect_payload["formal_evidence_available"] is False
    assert inspect_payload["source_trust_status"] == "blocked_no_real_evidence"
    assert inspect_payload["blocked_reason"] == "no_real_evidence"
    assert "blocked" in inspect_payload["collaboration"]["convergence_status"]
    assert "completed_degraded" not in inspect_payload["collaboration"]["convergence_status"]
    assert inspect_payload["collaboration"]["diagnosis_handoff"]["final_delivery"]["artifact_type"] == "report_blocked"
    trace = next(row for row in artifacts if row["artifact_type"] == "collaboration_trace")
    trace_payload = json.loads(Path(trace["path"]).read_text(encoding="utf-8"))
    assert trace_payload["delivery"]["artifact_type"] == "report_blocked"
    assert any(reason["code"] == "no_real_evidence" for reason in trace_payload["diagnosis"]["degraded_or_blocked_reasons"])


def test_production_query_with_real_evidence_and_fetch_warning_completes_degraded(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "real-evidence-degraded",
        {
            "query": "ExampleCo pricing",
            "competitor": "ExampleCo",
            "source_urls": ["https://example.com/pricing", "https://example.invalid/pricing"],
            "model_provider": "test_real",
            "search_provider": "static",
            "search_limit": 2,
            "link_gate_max_selected": 2,
        },
    )

    class RealishModel:
        provider = "test_real"
        model = "test-real"

        def complete(self, messages, response_format=None, metadata=None):
            role = (metadata or {}).get("role")
            if role == "StrategicAnalystAgent":
                return ModelResult(
                    text="",
                    json_data={
                        "inferences": [
                            {
                                "claim": "ExampleCo has public pricing evidence.",
                                "evidence_ids": [
                                    row["citation_id"]
                                    for row in store.list_citations(run_id)
                                    if row["source_type"] == "document"
                                ],
                                "confidence": 0.8,
                            }
                        ]
                    },
                    provider=self.provider,
                    model=self.model,
                    usage={},
                    latency_ms=1,
                    raw_response={},
                    status="ok",
                )
            return build_model_client("fake").complete(messages, response_format=response_format, metadata=metadata)

        def analyze_image(self, *args, **kwargs):
            return build_model_client("fake").analyze_image(*args, **kwargs)

    def fake_fetch(url, timeout=20.0):
        if "example.com/pricing" in url:
            return FetchResult(
                source_url=url,
                fetcher="httpx",
                status="ok",
                text=(
                    "ExampleCo pricing page. Starter plan is $29 per user per month. "
                    "The product emphasizes analytics collaboration."
                ),
                html="<html><body>ExampleCo pricing page</body></html>",
                latency_ms=1,
                metadata={"attempts": [{"fetcher": "httpx", "status": "ok"}]},
            )
        return FetchResult(
            source_url=url,
            fetcher="httpx",
            status="error",
            error="offline",
            latency_ms=1,
            metadata={"attempts": [{"fetcher": "httpx", "status": "error"}]},
        )

    monkeypatch.setattr("insightswarm.agents.extractor.fetch_source", fake_fetch)
    runner = Runner(store)
    runner.model = RealishModel()
    runner.start(run_id)
    assert store.get_run(run_id)["status"] == "completed_degraded"
    report = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "report")
    report_text = Path(report["path"]).read_text(encoding="utf-8")
    assert "## Degraded Output Warnings" in report_text
    assert "diagnostic-only for production query" in report_text
    inspect_payload = json.loads(inspect_run(store, run_id))
    assert inspect_payload["quality_status"] == "degraded"
    assert inspect_payload["collaboration_trace_artifact_id"]
    assert inspect_payload["collaboration_trace_summary"]["present"] is True
    assert inspect_payload["collaboration_trace_summary"]["tool_call_count"] >= 1
    assert inspect_payload["diagnosis"]["fetch_coverage"]["fetch_failure_count"] >= 1
    assert "tool_usage_summary" in inspect_payload["diagnosis"]
    assert "source_quality_summary" in inspect_payload["diagnosis"]
    assert inspect_payload["diagnosis"]["tool_usage_summary"]["tools"]["search.web"] >= 1
    assert inspect_payload["diagnosis"]["tool_audit_summary"]["tool_call_count"] >= 1
    assert inspect_payload["diagnosis"]["tool_audit_summary"]["tool_call_status_counts"]["ok"] >= 1
    assert inspect_payload["diagnosis"]["source_quality_summary"]["reviewed_count"] >= 1
    assert inspect_payload["diagnosis"]["validation_categories"]["freshness"]["status"] == "warning"
    assert inspect_payload["diagnosis"]["validation_categories"]["freshness"]["blocking"] is False
    assert "freshness" in inspect_payload["diagnosis"]["warning_validation_categories"]
    assert any("failed source URLs" in action for action in inspect_payload["diagnosis"]["recommended_next_actions"])
    assert any("freshness/source-quality warnings" in action for action in inspect_payload["diagnosis"]["recommended_next_actions"])
    assert inspect_payload["real_evidence_count"] >= 1
    assert inspect_payload["diagnostic_fallback_count"] >= 1
    assert inspect_payload["formal_evidence_available"] is True
    assert "completed_degraded" in inspect_payload["collaboration"]["convergence_status"]


def test_writer_model_report_preserves_citations_and_passes_validation(tmp_path):
    store = make_store(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text(
        "ExampleCo pricing page. Starter plan is $29 per user per month. "
        "The product emphasizes analytics collaboration.",
        encoding="utf-8",
    )
    run_id = store.create_run(
        "writer-model-valid",
        {
            "competitor": "ExampleCo",
            "source_urls": ["https://example.com/pricing"],
            "source_text_file": str(source),
            "model_provider": "test_real",
            "quality_mode": "test",
        },
    )

    class CitationKeepingModel:
        provider = "test_real"
        model = "citation-keeping"

        def complete(self, messages, response_format=None, metadata=None):
            role = (metadata or {}).get("role")
            if role == "StrategicAnalystAgent":
                return ModelResult(
                    text="",
                    json_data={
                        "inferences": [
                            {
                                "claim": "ExampleCo exposes pricing for public evaluation.",
                                "evidence_ids": [
                                    row["citation_id"]
                                    for row in store.list_citations(run_id)
                                    if row["source_type"] == "document"
                                ],
                                "confidence": 0.8,
                            }
                        ]
                    },
                    provider=self.provider,
                    model=self.model,
                    usage={},
                    latency_ms=1,
                    raw_response={},
                    status="ok",
                )
            if role == "WriterAgent":
                draft = json.loads(messages[-1]["content"])["draft_report"]
                markers = " ".join(extract_citation_markers(draft))
                return ModelResult(
                    text=(
                        "# Model Report\n\n"
                        "## Source Health\n\n"
                        "- Quality status: good\n\n"
                        "## Evidence-Backed Findings\n\n"
                        f"- Model-polished evidence line {markers}.\n\n"
                        "## Strategic Read\n\n"
                        f"- Model-polished strategic line {markers}."
                    ),
                    json_data=None,
                    provider=self.provider,
                    model=self.model,
                    usage={},
                    latency_ms=1,
                    raw_response={},
                    status="ok",
                )
            return build_model_client("fake").complete(messages, response_format=response_format, metadata=metadata)

        def analyze_image(self, *args, **kwargs):
            return build_model_client("fake").analyze_image(*args, **kwargs)

    runner = Runner(store)
    runner.model = CitationKeepingModel()
    runner.start(run_id)
    report = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "report")
    report_metadata = loads(report["metadata_json"], {})
    assert report_metadata["writer_status"] == "model_written"
    assert report_metadata["writer_validation_passed"] is True
    assert report_metadata["writer_fallback_used"] is False
    assert Path(report["path"]).read_text(encoding="utf-8").startswith("# Model Report")


def test_writer_model_missing_marker_falls_back_to_template(tmp_path):
    store = make_store(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text(
        "ExampleCo pricing page. Starter plan is $29 per user per month. "
        "The product emphasizes analytics collaboration.",
        encoding="utf-8",
    )
    run_id = store.create_run(
        "writer-model-fallback",
        {
            "competitor": "ExampleCo",
            "source_urls": ["https://example.com/pricing"],
            "source_text_file": str(source),
            "model_provider": "test_real",
            "quality_mode": "test",
        },
    )

    class CitationDroppingModel:
        provider = "test_real"
        model = "citation-dropping"

        def complete(self, messages, response_format=None, metadata=None):
            role = (metadata or {}).get("role")
            if role == "StrategicAnalystAgent":
                return ModelResult(
                    text="",
                    json_data={
                        "inferences": [
                            {
                                "claim": "ExampleCo exposes pricing for public evaluation.",
                                "evidence_ids": [
                                    row["citation_id"]
                                    for row in store.list_citations(run_id)
                                    if row["source_type"] == "document"
                                ],
                                "confidence": 0.8,
                            }
                        ]
                    },
                    provider=self.provider,
                    model=self.model,
                    usage={},
                    latency_ms=1,
                    raw_response={},
                    status="ok",
                )
            if role == "WriterAgent":
                return ModelResult(
                    text="# Model Report\n\n## Source Health\n\n- Quality status: good\n\n## Strategic Read\n\n- Unsupported new strategy.",
                    json_data=None,
                    provider=self.provider,
                    model=self.model,
                    usage={},
                    latency_ms=1,
                    raw_response={},
                    status="ok",
                )
            return build_model_client("fake").complete(messages, response_format=response_format, metadata=metadata)

        def analyze_image(self, *args, **kwargs):
            return build_model_client("fake").analyze_image(*args, **kwargs)

    runner = Runner(store)
    runner.model = CitationDroppingModel()
    runner.start(run_id)
    report = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "report")
    report_metadata = loads(report["metadata_json"], {})
    report_text = Path(report["path"]).read_text(encoding="utf-8")
    assert report_metadata["writer_status"] == "template_fallback_after_validation"
    assert report_metadata["writer_validation_passed"] is False
    assert report_metadata["writer_fallback_used"] is True
    assert report_metadata["missing_citation_markers"]
    assert report_text.startswith("# Competitive Analysis Report")
    assert "[[doc:" in report_text
    assert any(row["event_type"] == "writer_citation_repair" for row in store.list_events(run_id, limit=50))
    inspect_payload = json.loads(inspect_run(store, run_id))
    assert inspect_payload["writer_quality"]["writer_fallback_used"] is True
    assert inspect_payload["writer_quality"]["missing_citation_markers"]
    assert inspect_payload["diagnosis"]["writer_delivery_summary"]["writer_fallback_used"] is True
    assert any("Writer model output quality" in action for action in inspect_payload["diagnosis"]["recommended_next_actions"])


def test_run_diagnose_cli_outputs_text_and_json(tmp_path, capsys):
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    store = Store(db_path, artifact_dir)
    run_id = store.create_run("diagnose-cli")
    Runner(store).start(run_id)
    cli_main(["--db-path", str(db_path), "--artifact-dir", str(artifact_dir), "run", "diagnose", "--run-id", run_id])
    text_output = capsys.readouterr().out
    assert "Source Coverage" in text_output
    assert "Validation Categories" in text_output
    assert "Tool Audit" in text_output
    assert "Collaboration Trace" in text_output
    assert "- Evidence:" in text_output
    assert "Recommended Next Actions" in text_output
    cli_main(["--db-path", str(db_path), "--artifact-dir", str(artifact_dir), "run", "diagnose", "--run-id", run_id, "--json"])
    json_output = capsys.readouterr().out
    payload = json.loads(json_output)
    assert payload["run_status"] == "completed"
    assert "fetch_coverage" in payload
    assert "validation_categories" in payload
    assert payload["collaboration_trace_summary"]["present"] is True


def test_research_graph_cli_is_read_only_and_idempotent(tmp_path, capsys):
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    store = Store(db_path, artifact_dir)
    run_id = store.create_run("research-graph-cli", {"quality_mode": "test"})
    Runner(store).start(run_id)
    before_artifacts = len(store.list_artifacts(run_id))
    before_tasks = len(store.list_tasks(run_id))
    before_citations = len(store.list_citations(run_id))

    cli_main(["--db-path", str(db_path), "--artifact-dir", str(artifact_dir), "run", "graph", "--run-id", run_id, "--json"])
    first = json.loads(capsys.readouterr().out)
    cli_main(["--db-path", str(db_path), "--artifact-dir", str(artifact_dir), "run", "graph", "--run-id", run_id, "--json"])
    second = json.loads(capsys.readouterr().out)

    assert first == second
    assert len(store.list_artifacts(run_id)) == before_artifacts
    assert len(store.list_tasks(run_id)) == before_tasks
    assert len(store.list_citations(run_id)) == before_citations
    assert len({node["node_id"] for node in first["nodes"]}) == len(first["nodes"])
    assert len({edge["edge_id"] for edge in first["edges"]}) == len(first["edges"])
    assert first["contracts"]["edge_direction"] == "upstream/cause/source -> downstream/effect/consumer/result"


def test_phase_39_44_runtime_protocol_projection_is_read_only(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("runtime-protocol", {"quality_mode": "test"})
    Runner(store).start(run_id)
    before_artifacts = len(store.list_artifacts(run_id))
    before_tasks = len(store.list_tasks(run_id))
    before_citations = len(store.list_citations(run_id))

    cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "graph",
            "--run-id",
            run_id,
            "--json",
            "--protocol",
            "--runtime",
            "--governance",
        ]
    )
    first = json.loads(capsys.readouterr().out)
    cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "graph",
            "--run-id",
            run_id,
            "--json",
            "--protocol",
            "--runtime",
            "--governance",
        ]
    )
    second = json.loads(capsys.readouterr().out)

    assert first == second
    assert len(store.list_artifacts(run_id)) == before_artifacts
    assert len(store.list_tasks(run_id)) == before_tasks
    assert len(store.list_citations(run_id)) == before_citations
    assert first["contracts"]["protocol_registry"] == "research_runtime_protocol.v1"
    assert first["protocol"]["contracts"]["protocol_registry_is_source_of_truth"] is True
    assert "candidate_research_source" in first["protocol"]["command_templates"]
    assert first["multiagent_runtime"]["contracts"]["not_a_fixed_workflow_scheduler"] is True
    assert first["multiagent_runtime"]["summary"]["agent_identity_count"] >= 1
    assert first["graph_governance"]["contracts"]["rollback_preserves_history"] is True
    assert first["graph_governance"]["summary"]["by_phase"]["phase_44"] == 1


def test_research_graph_edges_evidence_and_snapshot_boundary(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("research-graph-boundary", {"quality_mode": "test"})
    Runner(store).start(run_id)

    from insightswarm.research_graph import build_research_graph, write_research_graph_artifact

    graph = build_research_graph(store, run_id)
    payload = graph.to_dict()
    nodes = {node["node_id"]: node for node in payload["nodes"]}
    edges = payload["edges"]
    citations = store.list_citations(run_id)
    citation_id = citations[0]["citation_id"]
    citation_artifact_id = citations[0]["artifact_id"]

    assert nodes[f"evidence:{citation_id}"]["data"]["is_formal_evidence"] is True
    assert any(
        edge["kind"] == "references"
        and edge["from_node_id"] == f"artifact:{citation_artifact_id}"
        and edge["to_node_id"] == f"evidence:{citation_id}"
        for edge in edges
    )
    assert any(
        edge["kind"] == "depends_on"
        and edge["from_node_id"].startswith("task:")
        and edge["to_node_id"].startswith("task:")
        for edge in edges
    )
    assert any(node["kind"] == "agent_actor" for node in payload["nodes"])

    snapshot_id = write_research_graph_artifact(store, run_id, graph)
    assert store.get_artifact(snapshot_id)["artifact_type"] == "research_graph"
    after_snapshot = build_research_graph(store, run_id).to_dict()
    assert payload == after_snapshot

    cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "diagnose", "--run-id", run_id])
    text_output = capsys.readouterr().out
    assert "Research Graph" in text_output
    cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "diagnose", "--run-id", run_id, "--json"])
    diagnosis = json.loads(capsys.readouterr().out)
    assert diagnosis["research_graph_summary"]["latest_graph_artifact_id"] == snapshot_id
    assert diagnosis["research_graph_summary"]["node_count"] == payload["summary"]["node_count"]
    assert diagnosis["research_runtime_protocol_summary"]["schema"] == "research_runtime_protocol.v1"
    assert diagnosis["multiagent_runtime_summary"]["contracts"]["not_a_fixed_workflow_scheduler"] is True
    assert diagnosis["graph_governance_summary"]["contracts"]["rollback_preserves_history"] is True

    from insightswarm.observability.trace import build_collaboration_trace

    trace = build_collaboration_trace(store, run_id)
    assert trace["research_graph"]["summary"]["edge_count"] == payload["summary"]["edge_count"]
    assert trace["research_runtime_protocol"]["schema"] == "research_runtime_protocol.v1"
    assert trace["multiagent_runtime"]["contracts"]["task_board_is_external_state"] is True
    assert trace["graph_governance"]["contracts"]["convergence_requires_formal_evidence"] is True


def test_research_graph_validation_and_frontiers_are_read_only(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("research-graph-validation", {"quality_mode": "test"})
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    before_artifacts = len(store.list_artifacts(run_id))
    before_tasks = len(store.list_tasks(run_id))
    before_citations = len(store.list_citations(run_id))

    cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "graph",
            "--run-id",
            run_id,
            "--validate",
            "--frontiers",
            "--json",
        ]
    )
    first = json.loads(capsys.readouterr().out)
    cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "graph",
            "--run-id",
            run_id,
            "--validate",
            "--frontiers",
            "--json",
        ]
    )
    second = json.loads(capsys.readouterr().out)

    assert first == second
    assert len(store.list_artifacts(run_id)) == before_artifacts
    assert len(store.list_tasks(run_id)) == before_tasks
    assert len(store.list_citations(run_id)) == before_citations
    assert first["validation"]["summary"]["error_count"] == 0
    assert any(
        frontier["status"] == "resumable"
        and frontier["kind"] == "candidate_research_source"
        and candidate_id in frontier["recommended_command"]
        for frontier in first["frontiers"]["frontiers"]
    )
    candidate_node = next(node for node in first["nodes"] if node["ref_id"] == candidate_id)
    assert candidate_node["data"]["is_formal_evidence"] is False


def test_research_graph_frontiers_detect_repair_writer_and_browser_boundaries(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "research-graph-frontiers",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )

    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-candidate",
            "--run-id",
            run_id,
            "--candidate-id",
            candidate_id,
        ]
    ) == 0
    passed = json.loads(capsys.readouterr().out)

    failed_analyst = store.create_task(run_id, "Synthesize", "StrategicAnalystAgent", [parent_task])
    store.set_task_status(failed_analyst, "needs_repair")
    failed_qa_task = store.create_task(run_id, "QA", "QAAgent", [failed_analyst])
    failed_qa_report_id = store.write_artifact(
        run_id,
        failed_qa_task,
        "qa_report",
        "application/json",
        json.dumps(
            {
                "score": 40,
                "passed": False,
                "rejection": {
                    "failures": [
                        {"category": "evidence", "gate": "counter_evidence", "message": "missing counter-evidence"}
                    ]
                },
                "analyst_task_id": failed_analyst,
            }
        ),
        metadata={"passed": False, "analyst_task_id": failed_analyst},
        suffix=".json",
    )
    failed_review_id = store.write_artifact(
        run_id,
        failed_qa_task,
        "review_qa_continuation",
        "application/json",
        json.dumps(
            {
                "schema": "review_qa_continuation.v1",
                "status": "qa_failed_needs_repair",
                "source_analysis_continuation_artifact_id": passed["analysis_continuation_artifact_id"],
                "analyst_task_id": failed_analyst,
                "qa_task_id": failed_qa_task,
                "qa_report_artifact_id": failed_qa_report_id,
                "stop_reason": "qa_failed_needs_repair",
            },
            ensure_ascii=True,
            indent=2,
        ),
        metadata={"schema": "review_qa_continuation.v1", "status": "qa_failed_needs_repair", "qa_report_artifact_id": failed_qa_report_id, "analyst_task_id": failed_analyst},
        suffix=".json",
    )

    from insightswarm.browser_authorization import write_assisted_observation_request, write_browser_authorization_request
    from insightswarm.research_graph import build_research_graph_frontiers
    from insightswarm.tools.core import ToolResult

    auth_id = write_browser_authorization_request(
        store,
        run_id,
        parent_task,
        "browser.goto",
        {"url": "https://mail.qq.com/"},
        ToolResult("blocked", diagnostics={"risk_status": "authorization_required", "risk_reason": "domain_not_authorized:mail.qq.com"}),
        "tool-call-auth",
        ToolContext(run_id, parent_task, "test", {"agent_name": "BrowserAgent"}),
    )
    obs_id = write_assisted_observation_request(
        store,
        run_id,
        parent_task,
        "browser.type",
        {"target": "captcha", "text": ""},
        ToolResult("blocked", diagnostics={"risk_status": "assisted_observation_required", "risk_reason": "human_assisted_observation_required"}),
        "tool-call-observe",
    )
    frontiers = build_research_graph_frontiers(store, run_id)["frontiers"]
    assert any(frontier["kind"] == "writer_delivery" and passed["review_qa_continuation_artifact_id"] in frontier["recommended_command"] for frontier in frontiers)
    assert any(frontier["kind"] == "qa_repair" and failed_review_id in (frontier["recommended_command"] or "") for frontier in frontiers)
    assert any(frontier["kind"] == "browser_authorization" and auth_id in frontier["recommended_command"] for frontier in frontiers)
    assert any(frontier["kind"] == "browser_assisted_observation" and obs_id in frontier["recommended_command"] for frontier in frontiers)

    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["research_graph_summary"]["validation"]["error_count"] == 0
    assert diagnosis["research_graph_summary"]["frontiers"]["resumable_count"] >= 2
    assert diagnosis["research_graph_summary"]["frontiers"]["human_intervention_count"] >= 2

    from insightswarm.observability.trace import build_collaboration_trace

    trace = build_collaboration_trace(store, run_id)
    assert "research_graph_validation" in trace
    assert "research_graph_frontiers" in trace
    assert trace["research_graph_frontiers"]["summary"]["frontier_count"] >= 4


def test_research_graph_plan_is_read_only_idempotent_and_orders_runtime_actions(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "research-graph-plan",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-candidate",
            "--run-id",
            run_id,
            "--candidate-id",
            candidate_id,
        ]
    ) == 0
    passed = json.loads(capsys.readouterr().out)

    failed_analyst = store.create_task(run_id, "Synthesize", "StrategicAnalystAgent", [parent_task])
    store.set_task_status(failed_analyst, "needs_repair")
    failed_qa_task = store.create_task(run_id, "QA", "QAAgent", [failed_analyst])
    failed_qa_report_id = store.write_artifact(
        run_id,
        failed_qa_task,
        "qa_report",
        "application/json",
        json.dumps(
            {
                "score": 40,
                "passed": False,
                "rejection": {
                    "failures": [
                        {"category": "evidence", "gate": "counter_evidence", "message": "missing counter-evidence"}
                    ]
                },
                "analyst_task_id": failed_analyst,
            }
        ),
        metadata={"passed": False, "analyst_task_id": failed_analyst},
        suffix=".json",
    )
    failed_review_id = store.write_artifact(
        run_id,
        failed_qa_task,
        "review_qa_continuation",
        "application/json",
        json.dumps(
            {
                "schema": "review_qa_continuation.v1",
                "status": "qa_failed_needs_repair",
                "source_analysis_continuation_artifact_id": passed["analysis_continuation_artifact_id"],
                "analyst_task_id": failed_analyst,
                "qa_task_id": failed_qa_task,
                "qa_report_artifact_id": failed_qa_report_id,
                "stop_reason": "qa_failed_needs_repair",
            },
            ensure_ascii=True,
            indent=2,
        ),
        metadata={"schema": "review_qa_continuation.v1", "status": "qa_failed_needs_repair", "qa_report_artifact_id": failed_qa_report_id, "analyst_task_id": failed_analyst},
        suffix=".json",
    )
    skeptic_id = store.write_artifact(
        run_id,
        failed_qa_task,
        "skeptic_review",
        "application/json",
        json.dumps({"evidence_gaps": [{"question": "Need counter-evidence"}], "source_risks": []}),
        suffix=".json",
    )
    from insightswarm.browser_authorization import write_browser_authorization_request
    from insightswarm.tools.core import ToolResult

    auth_id = write_browser_authorization_request(
        store,
        run_id,
        parent_task,
        "browser.goto",
        {"url": "https://mail.qq.com/"},
        ToolResult("blocked", diagnostics={"risk_status": "authorization_required", "risk_reason": "domain_not_authorized:mail.qq.com"}),
        "tool-call-plan-auth",
        ToolContext(run_id, parent_task, "test", {"agent_name": "BrowserAgent"}),
    )
    before_artifacts = len(store.list_artifacts(run_id))
    before_tasks = len(store.list_tasks(run_id))
    before_events = len(store.list_events(run_id, limit=1000))

    cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "graph",
            "--run-id",
            run_id,
            "--plan",
            "--json",
        ]
    )
    first = json.loads(capsys.readouterr().out)["plan"]
    cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "graph",
            "--run-id",
            run_id,
            "--plan",
            "--json",
        ]
    )
    second = json.loads(capsys.readouterr().out)["plan"]

    assert first == second
    assert len(store.list_artifacts(run_id)) == before_artifacts
    assert len(store.list_tasks(run_id)) == before_tasks
    assert len(store.list_events(run_id, limit=1000)) == before_events
    assert first["status"] == "plan_ready"
    assert {"resume_plan", "rollback_plan", "branch_plan", "human_gate_plan"}.issubset(set(first["plan_kinds"]))
    assert any(step["plan_kind"] == "human_gate_plan" and auth_id in (step["recommended_command"] or "") for step in first["steps"])
    assert any(step["plan_kind"] == "resume_plan" and failed_review_id in (step["recommended_command"] or "") for step in first["steps"])
    assert any(step["plan_kind"] == "resume_plan" and passed["review_qa_continuation_artifact_id"] in (step["recommended_command"] or "") for step in first["steps"])
    assert any(step["plan_kind"] == "rollback_plan" and step["action"] == "rollback_to_analysis_repair_boundary" for step in first["steps"])
    assert any(step["plan_kind"] == "branch_plan" and step["data"].get("source_artifact_id") == skeptic_id for step in first["steps"])
    assert first["steps"][0]["plan_kind"] == "human_gate_plan"
    assert any("browser authorize" in command for command in first["recommended_commands"])

    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["research_graph_summary"]["plan"]["status"] == "plan_ready"
    assert diagnosis["research_graph_summary"]["plan"]["step_count"] == first["summary"]["step_count"]

    from insightswarm.observability.trace import build_collaboration_trace

    trace = build_collaboration_trace(store, run_id)
    assert trace["research_graph_plan"]["summary"]["step_count"] == first["summary"]["step_count"]


def test_runtime_workspace_cli_is_deterministic_and_graph_read_is_passive(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("runtime-workspace", {"quality_mode": "test"})
    workspace_path = store.artifact_dir.parent / "runtime" / run_id

    cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "graph", "--run-id", run_id, "--json"])
    graph = json.loads(capsys.readouterr().out)
    assert graph["summary"]["workspace_node_count"] == 0
    assert not workspace_path.exists()

    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "workspace", "--run-id", run_id, "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "workspace", "--run-id", run_id, "--json"]) == 0
    second = json.loads(capsys.readouterr().out)

    assert workspace_path.exists()
    assert (workspace_path / "manifest.json").exists()
    assert first["summary"]["work_order_count"] == 0
    assert second["summary"]["work_order_count"] == 0
    assert first["records"] == second["records"]


def test_graph_governed_executor_creates_branch_work_order_and_graph_nodes(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("graph-executor-branch", {"quality_mode": "test"})
    parent_task = store.create_task(run_id, "Skeptic", "SkepticReviewAgent")
    store.set_task_status(parent_task, "completed")
    skeptic_id = store.write_artifact(
        run_id,
        parent_task,
        "skeptic_review",
        "application/json",
        json.dumps({"evidence_gaps": [{"question": "Need independent pricing source"}], "source_risks": []}),
        suffix=".json",
    )

    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "execute-plan",
            "--run-id",
            run_id,
            "--kind",
            "branch_plan",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "branch_ready"
    assert result["plan_kind"] == "branch_plan"
    assert len(result["workspace_record_ids"]) == 2

    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "workspace", "--run-id", run_id, "--json"]) == 0
    workspace = json.loads(capsys.readouterr().out)
    assert workspace["summary"]["work_order_count"] == 1
    assert workspace["summary"]["branch_count"] == 1
    branch_payload = workspace["records"]["branch"][0]["payload"]
    assert branch_payload["status"] == "branch_ready"
    assert branch_payload["data"]["source_artifact_id"] == skeptic_id

    from insightswarm.research_graph import build_research_graph

    graph = build_research_graph(store, run_id).to_dict()
    assert any(node["kind"] == "work_order" for node in graph["nodes"])
    assert any(node["kind"] == "branch" for node in graph["nodes"])
    assert any(edge["from_node_id"].startswith("work_order:") and edge["to_node_id"].startswith("branch:") for edge in graph["edges"])


def test_graph_governed_executor_human_gate_records_authorization(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("graph-executor-human-gate", {"quality_mode": "test"})
    parent_task = store.create_task(run_id, "Browser", "BrowserAgent")
    store.set_task_status(parent_task, "completed")
    from insightswarm.browser_authorization import write_browser_authorization_request
    from insightswarm.tools.core import ToolResult

    auth_id = write_browser_authorization_request(
        store,
        run_id,
        parent_task,
        "browser.goto",
        {"url": "https://mail.qq.com/"},
        ToolResult("blocked", diagnostics={"risk_status": "authorization_required", "risk_reason": "domain_not_authorized:mail.qq.com"}),
        "tool-call-executor-auth",
        ToolContext(run_id, parent_task, "test", {"agent_name": "BrowserAgent"}),
    )

    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "execute-plan",
            "--run-id",
            run_id,
            "--kind",
            "human_gate_plan",
        ]
    ) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked_human_gate"
    assert "browser authorize" in result["recommended_command"]

    workspace_path = store.artifact_dir.parent / "runtime" / run_id
    assert workspace_path.exists()
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "workspace", "--run-id", run_id, "--json"]) == 0
    workspace = json.loads(capsys.readouterr().out)
    assert workspace["summary"]["work_order_count"] == 1
    assert workspace["summary"]["authorization_pending_count"] == 1
    assert workspace["records"]["authorization"][0]["payload"]["data"]["request_artifact_id"] == auth_id

    from insightswarm.observability.trace import build_collaboration_trace

    trace = build_collaboration_trace(store, run_id)
    assert trace["runtime_workspace"]["summary"]["authorization_pending_count"] == 1


def test_graph_governed_executor_candidate_resume_uses_existing_continuation(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "graph-executor-candidate",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )

    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "execute-plan",
            "--run-id",
            run_id,
            "--kind",
            "resume_plan",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "executed_candidate_continuation"
    assert result["continuation_result"]["candidate_research_source_ids"] == [candidate_id]
    assert result["continuation_result"]["status"] == "citation_ready"

    workspace = json.loads(
        (store.artifact_dir.parent / "runtime" / run_id / "work_orders" / f"{result['work_order_id'].replace(':', '_')}.json").read_text(encoding="utf-8")
    )
    assert workspace["status"] == "completed"
    assert workspace["execution_result"]["status"] == "citation_ready"
    assert len(list((store.artifact_dir.parent / "runtime" / run_id / "work_orders").glob("*.json"))) == 1


def test_graph_governed_executor_rollback_can_create_arbitration_record(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("graph-executor-rollback", {"quality_mode": "test"})
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    first_analyst = store.create_task(run_id, "Synthesize", "StrategicAnalystAgent", [parent_task])
    second_analyst = store.create_task(run_id, "Synthesize", "StrategicAnalystAgent", [parent_task])
    store.set_task_status(first_analyst, "needs_repair")
    store.set_task_status(second_analyst, "blocked", {"human_intervention_required": True}, retry_delta=3)
    for index, analyst_task in enumerate([first_analyst, second_analyst], start=1):
        qa_task = store.create_task(run_id, "QA", "QAAgent", [analyst_task])
        qa_report_id = store.write_artifact(
            run_id,
            qa_task,
            "qa_report",
            "application/json",
            json.dumps({"score": 40, "passed": False, "rejection": {"failures": [{"category": "evidence", "message": "gap"}]}, "analyst_task_id": analyst_task}),
            metadata={"passed": False, "analyst_task_id": analyst_task},
            suffix=".json",
        )
        store.write_artifact(
            run_id,
            qa_task,
            "review_qa_continuation",
            "application/json",
            json.dumps(
                {
                    "schema": "review_qa_continuation.v1",
                    "status": "qa_failed_needs_repair",
                    "analyst_task_id": analyst_task,
                    "qa_task_id": qa_task,
                    "qa_report_artifact_id": qa_report_id,
                    "stop_reason": f"qa_failed_needs_repair_{index}",
                }
            ),
            metadata={"schema": "review_qa_continuation.v1", "status": "qa_failed_needs_repair", "qa_report_artifact_id": qa_report_id, "analyst_task_id": analyst_task},
            suffix=".json",
        )

    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "execute-plan",
            "--run-id",
            run_id,
            "--kind",
            "rollback_plan",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "rollback_ready"
    assert any(record_id.startswith("arbitration:") for record_id in result["workspace_record_ids"])

    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "workspace", "--run-id", run_id, "--json"]) == 0
    workspace = json.loads(capsys.readouterr().out)
    assert workspace["summary"]["rollback_count"] == 1
    assert workspace["summary"]["arbitration_count"] == 1
    assert workspace["records"]["arbitration"][0]["payload"]["status"] == "arbitration_required"


def test_swarm_projection_is_read_only_for_open_branch_work_order(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("swarm-projection", {"quality_mode": "test"})
    parent_task = store.create_task(run_id, "Skeptic", "SkepticReviewAgent")
    store.set_task_status(parent_task, "completed")
    store.write_artifact(
        run_id,
        parent_task,
        "skeptic_review",
        "application/json",
        json.dumps({"evidence_gaps": [{"question": "Need source"}], "source_risks": []}),
        suffix=".json",
    )
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "execute-plan",
            "--run-id",
            run_id,
            "--kind",
            "branch_plan",
        ]
    ) == 0
    capsys.readouterr()
    before_tasks = len(store.list_tasks(run_id))
    before_artifacts = len(store.list_artifacts(run_id))
    before_events = len(store.list_events(run_id, limit=1000))

    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "swarm", "--run-id", run_id, "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "swarm", "--run-id", run_id, "--json"]) == 0
    second = json.loads(capsys.readouterr().out)

    assert first == second
    assert first["summary"]["open_work_order_count"] == 1
    assert first["open_work_orders"][0]["payload"]["plan_kind"] == "branch_plan"
    assert len(store.list_tasks(run_id)) == before_tasks
    assert len(store.list_artifacts(run_id)) == before_artifacts
    assert len(store.list_events(run_id, limit=1000)) == before_events


def test_swarm_step_runs_browseragent_from_branch_work_order_without_formal_evidence(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("swarm-browser", {"quality_mode": "test", "query": "ExampleCo pricing"})
    parent_task = store.create_task(run_id, "Skeptic", "SkepticReviewAgent")
    store.set_task_status(parent_task, "completed")
    store.write_artifact(
        run_id,
        parent_task,
        "skeptic_review",
        "application/json",
        json.dumps({"evidence_gaps": [{"question": "Need BrowserAgent source acquisition"}], "source_risks": []}),
        suffix=".json",
    )
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "execute-plan",
            "--run-id",
            run_id,
            "--kind",
            "branch_plan",
        ]
    ) == 0
    capsys.readouterr()

    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "swarm-step", "--run-id", run_id, "--agent", "BrowserAgent"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] in {"browser_candidate_ready", "browser_operation_completed"}
    assert result["agent_name"] == "BrowserAgent"
    assert result["assignment_id"].startswith("swarm_assignment:")
    assert result["workspace_record_ids"]

    artifacts = store.list_artifacts(run_id)
    assert any(row["artifact_type"] == "candidate_source" for row in artifacts)
    assert not store.list_citations(run_id)
    assert not any(row["artifact_type"] in {"strategic_analysis", "qa_report", "report"} for row in artifacts)

    workspace = json.loads(
        cli_json(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "workspace",
                "--run-id",
                run_id,
                "--json",
            ],
            capsys,
        )
    )
    assert workspace["summary"]["swarm_assignment_count"] == 1
    assert workspace["summary"]["browser_swarm_operation_count"] == 1
    assert workspace["summary"]["swarm_handoff_count"] == 1

    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["swarm_runtime_summary"]["browser_swarm_operation_count"] == 1
    assert diagnosis["runtime_workspace_summary"]["swarm_assignment_count"] == 1
    from insightswarm.observability.trace import build_collaboration_trace

    trace = build_collaboration_trace(store, run_id)
    assert trace["swarm_runtime"]["summary"]["browser_swarm_operation_count"] == 1
    assert trace["runtime_workspace"]["browser_swarm_operations"]
    from insightswarm.research_graph import build_research_graph

    graph = build_research_graph(store, run_id).to_dict()
    assert any(node["kind"] == "swarm_assignment" for node in graph["nodes"])
    assert any(node["kind"] == "browser_swarm_operation" for node in graph["nodes"])
    assert any(edge["from_node_id"].startswith("work_order:") and edge["to_node_id"].startswith("swarm_assignment:") for edge in graph["edges"])


def test_swarm_step_can_assign_bounded_subagent_without_evidence(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "swarm-subagent",
        {
            "quality_mode": "test",
            "allowed_subagent_roles": ["SearchAgent"],
            "max_subagents_per_run": 1,
            "max_parallel_subagents": 1,
        },
    )
    parent_task = store.create_task(run_id, "Skeptic", "SkepticReviewAgent")
    store.set_task_status(parent_task, "completed")
    store.write_artifact(
        run_id,
        parent_task,
        "skeptic_review",
        "application/json",
        json.dumps({"evidence_gaps": [{"question": "Need SearchAgent branch"}], "source_risks": []}),
        suffix=".json",
    )
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "execute-plan",
            "--run-id",
            run_id,
            "--kind",
            "branch_plan",
        ]
    ) == 0
    plan_result = json.loads(capsys.readouterr().out)

    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "swarm-step",
            "--run-id",
            run_id,
            "--work-order-id",
            plan_result["work_order_id"],
            "--agent",
            "SearchAgent",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "subagent_task_ready"
    assert result["agent_name"] == "SearchAgent"
    task = store.get_task(result["task_id"])
    assert loads(task["metadata_json"], {})["subagent"] is True
    assert not store.list_citations(run_id)
    assert not any(row["artifact_type"] in {"qa_report", "report"} for row in store.list_artifacts(run_id))


def test_source_acquisition_frontier_from_fetch_failure_runs_browser_swarm(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("swarm-fetch-failure", {"quality_mode": "test", "query": "ExampleCo pricing"})
    parent_task = store.create_task(run_id, "Discovery", "ScraperAgent")
    store.set_task_status(parent_task, "completed")
    failure_id = store.write_artifact(
        run_id,
        parent_task,
        "fetch_failure",
        "application/json",
        json.dumps(
            {
                "source_url": "https://example.invalid/pricing",
                "fetcher": "httpx",
                "status": "error",
                "error": "timeout",
                "fallback_reason": "httpx_failed",
            }
        ),
        source_url="https://example.invalid/pricing",
        metadata={"source_url": "https://example.invalid/pricing", "fetcher": "httpx", "status": "error"},
        suffix=".json",
    )

    from insightswarm.research_graph import build_research_graph_frontiers, build_research_graph_plan

    frontiers = build_research_graph_frontiers(store, run_id)["frontiers"]
    source_frontier = next(frontier for frontier in frontiers if frontier["kind"] == "source_acquisition")
    assert source_frontier["data"]["trigger"] == "fetch_failure"
    assert source_frontier["data"]["preferred_actor"] == "BrowserAgent"
    assert source_frontier["data"]["source_artifact_id"] == failure_id
    plan = build_research_graph_plan(store, run_id)
    branch_step = next(step for step in plan["steps"] if step["action"] == "branch_from_source_acquisition")
    assert branch_step["plan_kind"] == "branch_plan"
    assert branch_step["data"]["source_artifact_id"] == failure_id

    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "execute-plan", "--run-id", run_id, "--step-id", branch_step["step_id"]]) == 0
    plan_result = json.loads(capsys.readouterr().out)
    assert plan_result["status"] == "branch_ready"

    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "swarm-step", "--run-id", run_id, "--work-order-id", plan_result["work_order_id"], "--agent", "BrowserAgent"]) == 0
    swarm_result = json.loads(capsys.readouterr().out)
    assert swarm_result["status"] in {"browser_candidate_ready", "browser_operation_completed"}
    assert swarm_result["agent_name"] == "BrowserAgent"
    assert any(row["artifact_type"] == "candidate_source" for row in store.list_artifacts(run_id))
    assert not store.list_citations(run_id)
    assert not any(row["artifact_type"] in {"strategic_analysis", "qa_report", "report"} for row in store.list_artifacts(run_id))


def test_collaboration_kernel_translates_zero_citation_extractor_failure(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "collaboration-zero-citation",
        {
            "quality_mode": "test",
            "query": "DeepSeek strategy",
            "browser_source_target_url": "https://www.deepseek.com/",
            "browser_backend": "fake",
        },
    )
    lead_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(lead_task, "completed")
    extract_task = store.create_task(run_id, "Extract", "ExtractorAgent")
    raw_id = store.write_artifact(
        run_id,
        extract_task,
        "raw_document",
        "text/plain",
        "DeepSeek-V4 预览版本发布，Agent 能力大幅提高。",
        source_url="https://www.deepseek.com/",
        metadata={"fetcher": "httpx", "status": "ok"},
    )
    structured_id = store.write_artifact(
        run_id,
        extract_task,
        "structured_knowledge",
        "application/json",
        json.dumps({"competitor": "DeepSeek", "facts": []}, ensure_ascii=True),
        metadata={
            "schema": "competitor_knowledge.v1",
            "accepted_fact_count": 0,
            "discarded_fact_count": 1,
            "discarded_facts": [{"field": "model_release", "quote": "深 度 求 索 V4", "reason": "quote not found in source text"}],
        },
        suffix=".json",
    )
    analyst_task = store.create_task(run_id, "Synthesize", "StrategicAnalystAgent")
    store.set_task_status(analyst_task, "blocked", {"error": "inference citation requires at least one evidence id"})

    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "collaborate", "--run-id", run_id, "--apply", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    artifacts = [dict(row) for row in store.list_artifacts(run_id)]
    frontiers = payload["frontiers"]["frontiers"]

    assert payload["summary"]["translation_count"] == 2
    assert any(row["artifact_type"] == "failure_frontier_translation" for row in artifacts)
    assert any(row["artifact_type"] == "collaboration_intent" for row in artifacts)
    assert any(row["artifact_type"] == "tool_contract_snapshot" for row in artifacts)
    assert any(frontier["kind"] == "source_acquisition" and frontier["data"]["trigger"] == "extractor_zero_citation" for frontier in frontiers)
    assert any(frontier["kind"] == "source_acquisition" and frontier["data"]["trigger"] == "analyst_no_evidence_blocked" for frontier in frontiers)
    source_frontiers = [frontier for frontier in frontiers if frontier["kind"] == "source_acquisition"]
    assert not any(f"artifact:{structured_id}" in frontier["source_node_ids"] for frontier in source_frontiers)
    assert not any(f"task:{analyst_task}" in frontier["source_node_ids"] for frontier in source_frontiers)
    assert any(node.startswith("collaboration_intent:") for frontier in source_frontiers for node in frontier["source_node_ids"])
    messages = [
        loads(row["payload_json"], {})
        for row in store.conn.execute("SELECT payload_json FROM messages WHERE run_id = ?", (run_id,))
    ]
    assert any(message.get("intent") == "evidence_gap" for message in messages)
    assert any(message.get("intent") == "source_acquisition_request" for message in messages)

    before = len([row for row in store.list_artifacts(run_id) if row["artifact_type"] == "failure_frontier_translation"])
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "collaborate", "--run-id", run_id, "--apply", "--json"]) == 0
    capsys.readouterr()
    after = len([row for row in store.list_artifacts(run_id) if row["artifact_type"] == "failure_frontier_translation"])
    assert before == after
    assert len([row for row in store.list_artifacts(run_id) if row["artifact_type"] == "tool_contract_snapshot"]) == 1

    from insightswarm.observability.trace import build_collaboration_trace

    trace = build_collaboration_trace(store, run_id)
    collaboration = trace["agent_collaboration_kernel"]
    assert collaboration["summary"]["tool_contract_count"] >= 1
    assert any(row["intent"] in {"evidence_gap", "source_acquisition_request"} for row in collaboration["mailbox"])
    assert raw_id


def test_source_acquisition_frontier_from_qa_evidence_failure_is_swarm_eligible(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("swarm-qa-evidence-failure", {"quality_mode": "test", "query": "ExampleCo pricing"})
    qa_task = store.create_task(run_id, "QA", "QAAgent")
    store.set_task_status(qa_task, "completed")
    qa_report_id = store.write_artifact(
        run_id,
        qa_task,
        "qa_report",
        "application/json",
        json.dumps(
            {
                "score": 55,
                "passed": False,
                "rejection": {"failures": [{"category": "evidence", "gate": "citation_coverage", "message": "Need stronger source"}]},
            }
        ),
        metadata={"passed": False},
        suffix=".json",
    )

    from insightswarm.research_graph import build_research_graph_frontiers

    frontiers = build_research_graph_frontiers(store, run_id)["frontiers"]
    source_frontier = next(frontier for frontier in frontiers if frontier["kind"] == "source_acquisition")
    assert source_frontier["data"]["trigger"] == "qa_evidence_failure"
    assert source_frontier["data"]["qa_report_artifact_id"] == qa_report_id

    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "execute-plan", "--run-id", run_id, "--kind", "branch_plan"]) == 0
    plan_result = json.loads(capsys.readouterr().out)
    assert plan_result["status"] == "branch_ready"
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "swarm", "--run-id", run_id, "--json"]) == 0
    swarm = json.loads(capsys.readouterr().out)
    assert swarm["summary"]["open_work_order_count"] == 1
    assert swarm["open_work_orders"][0]["payload"]["data"]["trigger"] == "qa_evidence_failure"


def test_swarm_browseragent_writes_policy_block_when_browser_action_needs_human_gate(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("swarm-browser-human-gate", {"quality_mode": "test", "query": "ExampleCo pricing"})
    parent_task = store.create_task(run_id, "Skeptic", "SkepticReviewAgent")
    store.set_task_status(parent_task, "completed")
    page_state_result, _ = ToolExecutor(store).run(
        "browser.page_state",
        {"url": "https://example.com/pricing"},
        ToolContext(run_id, parent_task, "test", {"agent_name": "BrowserAgent"}),
    )
    page_state_id = write_browser_observation(store, run_id, parent_task, "browser.page_state", page_state_result)
    ToolExecutor(store).run(
        "browser.select_target",
        {"intent": "open pricing", "page_state": page_state_result.data["observation"], "page_state_artifact_id": page_state_id},
        ToolContext(run_id, parent_task, "test", {"agent_name": "BrowserAgent"}),
    )
    store.write_artifact(
        run_id,
        parent_task,
        "skeptic_review",
        "application/json",
        json.dumps({"evidence_gaps": [{"question": "Need BrowserAgent source acquisition"}], "source_risks": []}),
        suffix=".json",
    )

    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "execute-plan", "--run-id", run_id, "--kind", "branch_plan"]) == 0
    capsys.readouterr()
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "swarm-step", "--run-id", run_id, "--agent", "BrowserAgent"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked_browser_policy"
    assert result["browser_result"]["status"] in {"pending_authorization", "pending_approval"}

    workspace = json.loads(cli_json(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "workspace", "--run-id", run_id, "--json"], capsys))
    assert workspace["summary"]["swarm_policy_block_count"] == 1
    assert workspace["records"]["swarm_policy_block"][0]["payload"]["requires_human"] is True
    assert any(row["artifact_type"] in {"browser_authorization_request", "browser_action_request"} for row in store.list_artifacts(run_id))


def cli_json(args, capsys):
    assert cli_main(args) == 0
    return capsys.readouterr().out


def test_qa_blocks_after_three_gate_rejections(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("blocked")
    analyst = store.create_task(run_id, "Synthesize", "StrategicAnalystAgent")
    qa = store.create_task(run_id, "QA", "QAAgent", [analyst])
    store.set_task_status(analyst, "completed", {"analysis": {"status": "bad"}})
    from insightswarm.agents.qa import QAAgent
    from insightswarm.models.fake import FakeModelClient

    agent = QAAgent(store, FakeModelClient())
    for _ in range(3):
        store.set_task_status(qa, "pending")
        agent.execute(run_id, qa)
        if store.get_task(analyst)["status"] == "blocked":
            break
        store.set_task_status(analyst, "completed")
    analyst_row = store.get_task(analyst)
    assert analyst_row["status"] == "blocked"
    assert loads(analyst_row["metadata_json"], {})["human_intervention_required"] is True
    inspect_payload = json.loads(inspect_run(store, run_id))
    collaboration = inspect_payload["collaboration"]
    assert "blocked" in collaboration["convergence_status"]
    assert collaboration["repair_task_count"] >= 1
    assert collaboration["repair_rounds"][-1]["human_intervention_required"] is True
    assert collaboration["repair_rounds"][-1]["retry_count"] >= 1
    analyst_meta = loads(store.get_task(analyst)["metadata_json"], {})
    repair_contract = analyst_meta["repair_contract"]["repair_contract"]
    assert repair_contract["validation_categories"]["evidence"]["blocking"] is True
    assert any(row["category"] == "evidence" for row in repair_contract["qa_failure_summary"])


def test_runner_finishes_blocked_instead_of_raising_on_agent_exception(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    run_id = store.create_run("runner-blocked")
    task_id = store.create_task(run_id, "Extract", "ExtractorAgent")

    from insightswarm.harness import runner as runner_module

    class ExplodingAgent:
        def __init__(self, store, model):
            self.store = store

        def execute(self, run_id, task_id):
            self.store.set_task_status(task_id, "blocked", {"error": "boom"})
            raise RuntimeError("boom")

    monkeypatch.setitem(runner_module.AGENT_CLASSES, "ExtractorAgent", ExplodingAgent)
    Runner(store).start(run_id)
    assert store.get_run(run_id)["status"] == "blocked"
    assert store.get_task(task_id)["status"] == "blocked"
    events = store.list_events(run_id, limit=20)
    assert any(row["event_type"] == "task_blocked_after_exception" for row in events)


def test_run_extract_cli_continues_browser_handoff_raw_document_to_formal_evidence(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "browser-handoff-extract",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    result, _ = ToolExecutor(store).run(
        "browser.promote_source",
        {
            "source_url": "https://example.com/pricing",
            "title": "ExampleCo pricing",
            "text": "ExampleCo pricing page. Starter plan is $29 per user per month. Analytics collaboration is included.",
        },
        ToolContext(run_id, task_id, "test", {"agent_name": "BrowserAgent", "browser_mode": "free_browser"}),
    )
    assert result.status == "ok"
    candidate_id = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "candidate_source")["artifact_id"]
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "browser",
                "promote",
                "--run-id",
                run_id,
                "--candidate-id",
                candidate_id,
                "--quality-mode",
                "test",
            ]
        )
        == 0
    )
    capsys.readouterr()
    raw_document = next(
        row
        for row in store.list_artifacts(run_id)
        if row["artifact_type"] == "raw_document"
        and loads(row["metadata_json"], {}).get("fetcher") == "browser_agent_handoff"
    )
    diagnosis_before = build_run_diagnosis(store, run_id)
    assert diagnosis_before["browser_evidence_handoff_summary"]["pending_extract_raw_document_ids"] == [raw_document["artifact_id"]]
    assert any("run extract" in action for action in diagnosis_before["recommended_next_actions"])

    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "extract",
                "--run-id",
                run_id,
                "--raw-document-id",
                raw_document["artifact_id"],
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "completed"
    assert output["structured_knowledge_artifact_id"]
    assert output["citation_ids"]
    assert any(row["artifact_type"] == "structured_knowledge" for row in store.list_artifacts(run_id))
    assert any(row["source_type"] == "document" for row in store.list_citations(run_id))
    diagnosis_after = build_run_diagnosis(store, run_id)
    assert diagnosis_after["formal_evidence_available"] is True
    assert diagnosis_after["browser_evidence_handoff_summary"]["pending_extract_raw_document_count"] == 0


def test_browser_authorization_and_assisted_observation_cli_diagnosis_trace(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("browser-authorization", {"quality_mode": "test"})
    task_id = store.create_task(run_id, "BrowserOperation", "BrowserAgent")
    executor = ToolExecutor(store)
    auth_result, _ = executor.run(
        "browser.goto",
        {"url": "https://mail.qq.com/"},
        ToolContext(run_id, task_id, "test", {"agent_name": "BrowserAgent"}),
    )
    assert auth_result.diagnostics["human_authorization_required"] is True
    obs_result, _ = executor.run(
        "browser.type",
        {"target": "验证码", "text": "123456"},
        ToolContext(run_id, task_id, "test", {"agent_name": "BrowserAgent"}),
    )
    assert obs_result.diagnostics["human_assisted_observation_required"] is True
    auth_request = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_authorization_request")
    obs_request = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_assisted_observation_request")

    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "browser", "authorizations", "--run-id", run_id]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["pending_authorizations"][0]["artifact_id"] == auth_request["artifact_id"]
    assert listed["pending_observations"][0]["artifact_id"] == obs_request["artifact_id"]
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "browser", "authorize", "--run-id", run_id, "--request-id", auth_request["artifact_id"], "--decision", "approve"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "approve"
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "browser", "observe", "--run-id", run_id, "--request-id", obs_request["artifact_id"], "--value", "654321"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "provided"

    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["browser_authorization_summary"]["authorization_request_count"] == 1
    assert diagnosis["browser_authorization_summary"]["authorization_decision_count"] == 1
    assert diagnosis["browser_authorization_summary"]["observation_request_count"] == 1
    assert diagnosis["browser_authorization_summary"]["observation_response_count"] == 1
    from insightswarm.observability.trace import ensure_collaboration_trace

    trace = ensure_collaboration_trace(store, run_id)["trace"]
    assert len(trace["browser_authorizations"]) == 4


def test_subagent_policy_cli_context_runner_diagnosis_trace(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "subagent-runtime",
        {
            "quality_mode": "test",
            "query": "ExampleCo pricing",
            "source_urls": ["https://example.com/pricing"],
            "max_subagents_per_run": 1,
            "max_spawn_depth": 1,
            "max_parallel_subagents": 1,
            "max_context_tokens_per_subagent": 1200,
            "allowed_subagent_roles": ["SearchAgent"],
        },
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    store.write_artifact(
        run_id,
        parent_task,
        "research_contract",
        "application/json",
        json.dumps({"schema": "research_contract.v1", "goal": "Find primary pricing evidence."}),
        metadata={"schema": "research_contract.v1"},
        suffix=".json",
    )
    store.write_artifact(
        run_id,
        None,
        "raw_document",
        "text/plain",
        "Unrelated large source should not enter subagent context.",
        source_url="https://irrelevant.example/source",
        metadata={"fetcher": "manual"},
    )

    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "spawn-subagent",
                "--run-id",
                run_id,
                "--parent-task-id",
                parent_task,
                "--role",
                "SearchAgent",
                "--scope",
                "Find primary pricing URLs.",
            ]
        )
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    subagent_task_id = created["task_id"]
    subagent_task = store.get_task(subagent_task_id)
    subagent_metadata = loads(subagent_task["metadata_json"], {})
    assert subagent_metadata["subagent"] is True
    assert subagent_metadata["parent_task_id"] == parent_task
    assert subagent_metadata["scope"] == "Find primary pricing URLs."
    assert subagent_metadata["budget"]["max_context_tokens"] == 1200
    assert subagent_metadata["allowed_tools"] == ["search.web"]

    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "spawn-subagent",
                "--run-id",
                run_id,
                "--parent-task-id",
                parent_task,
                "--role",
                "SearchAgent",
                "--scope",
                "Second subagent should exceed budget.",
            ]
        )
        == 2
    )
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "max_subagents_per_run_exceeded"

    Runner(store).start(run_id)
    assert store.get_task(subagent_task_id)["status"] == "completed"
    handoff = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "subagent_handoff")
    finding = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "research_finding")
    handoff_payload = json.loads(Path(handoff["path"]).read_text(encoding="utf-8"))
    assert handoff_payload["finding_artifact_id"] == finding["artifact_id"]
    assert "not formal evidence" in handoff_payload["formal_evidence_boundary"]

    context_artifact = next(
        row
        for row in store.list_artifacts(run_id)
        if row["artifact_type"] == "context_envelope" and row["task_id"] == subagent_task_id
    )
    context_payload = json.loads(Path(context_artifact["path"]).read_text(encoding="utf-8"))
    assert context_payload["parent_task"]["task_id"] == parent_task
    assert context_payload["subagent_scope"] == "Find primary pricing URLs."
    assert context_payload["context_budget"]["max_tokens"] == 1200
    assert context_payload["handoff_requirements"]["artifact_type"] == "subagent_handoff"
    assert all(
        artifact["source_url"] != "https://irrelevant.example/source"
        for artifact in context_payload["artifacts"]
    )

    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["subagent_summary"]["total_count"] == 1
    assert diagnosis["subagent_summary"]["completed_count"] == 1
    assert diagnosis["subagent_summary"]["blocked_spawn_count"] == 1
    from insightswarm.observability.trace import ensure_collaboration_trace

    trace = ensure_collaboration_trace(store, run_id)["trace"]
    assert trace["subagents"][0]["task_id"] == subagent_task_id
    assert trace["subagents"][0]["handoff_artifact_id"] == handoff["artifact_id"]


def test_subagent_finding_promotion_merges_into_link_gate_and_extract(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "subagent-promotion",
        {
            "quality_mode": "test",
            "query": "ExampleCo pricing",
            "competitor": "ExampleCo",
            "search_provider": "static",
            "source_urls": [],
            "link_gate_max_selected": 3,
        },
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    subagent_task = store.create_task(
        run_id,
        "Search",
        "SearchAgent",
        [parent_task],
        {
            "subagent": True,
            "parent_task_id": parent_task,
            "spawn_depth": 1,
            "scope": "Find pricing evidence",
            "role": "SearchAgent",
        },
    )
    store.set_task_status(subagent_task, "completed")
    finding_payload = {
        "claim": "Subagent found a pricing source.",
        "confidence": 0.82,
        "evidence_candidates": [
            {
                "title": "ExampleCo Pricing",
                "url": "https://example.com/pricing",
                "snippet": "ExampleCo pricing page. Starter plan is $29 per user per month. Analytics collaboration is included.",
                "rank": 1,
            }
        ],
        "source_urls": ["https://example.com/pricing"],
        "open_questions": [],
        "risk_flags": [],
        "recommended_next_tasks": [],
    }
    finding_id = store.write_artifact(
        run_id,
        subagent_task,
        "research_finding",
        "application/json",
        json.dumps(finding_payload, ensure_ascii=True, indent=2),
        metadata={"schema": "research_finding.v1", "parent_task_id": parent_task, "subagent": True},
        suffix=".json",
    )
    handoff_id = store.write_artifact(
        run_id,
        subagent_task,
        "subagent_handoff",
        "application/json",
        json.dumps(
            {
                "schema": "subagent_handoff.v1",
                "subagent_task_id": subagent_task,
                "parent_task_id": parent_task,
                "finding_artifact_id": finding_id,
                "finding": finding_payload,
            },
            ensure_ascii=True,
            indent=2,
        ),
        metadata={"schema": "subagent_handoff.v1", "parent_task_id": parent_task, "finding_artifact_id": finding_id},
        suffix=".json",
    )
    assert len(store.list_citations(run_id)) == 0
    diagnosis_before = build_run_diagnosis(store, run_id)
    assert diagnosis_before["subagent_summary"]["pending_promotion_finding_ids"] == [finding_id]
    assert any("promote-finding" in action for action in diagnosis_before["recommended_next_actions"])

    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "promote-finding",
                "--run-id",
                run_id,
                "--handoff-id",
                handoff_id,
            ]
        )
        == 0
    )
    promoted = json.loads(capsys.readouterr().out)
    assert promoted["status"] == "promoted"
    assert promoted["candidate_count"] == 1
    candidate = store.get_artifact(promoted["candidate_research_source_ids"][0])
    candidate_payload = json.loads(Path(candidate["path"]).read_text(encoding="utf-8"))
    assert candidate_payload["source_url"] == "https://example.com/pricing"
    assert candidate_payload["requires_link_gate"] is True
    assert len(store.list_citations(run_id)) == 0

    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "promote-finding",
                "--run-id",
                run_id,
                "--finding-id",
                finding_id,
            ]
        )
        == 2
    )
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "no_new_source_urls"

    link_gate_task = store.create_task(run_id, "LinkGate", "LinkGateAgent", [parent_task])
    extract_placeholder = store.create_task(
        run_id,
        "Extract",
        "ExtractorAgent",
        [link_gate_task],
        {"placeholder": True, "dynamic_expand_after": link_gate_task},
    )
    visual_task = store.create_task(run_id, "Extract", "VisualAgent", [link_gate_task])
    store.set_task_status(visual_task, "completed")
    analyst_task = store.create_task(
        run_id,
        "Synthesize",
        "StrategicAnalystAgent",
        [extract_placeholder, visual_task],
        {"wait_for_dynamic_extract_group": link_gate_task},
    )
    skeptic_task = store.create_task(run_id, "SkepticReview", "SkepticReviewAgent", [analyst_task])
    qa_task = store.create_task(run_id, "QA", "QAAgent", [skeptic_task])
    store.create_task(run_id, "Deliver", "WriterAgent", [qa_task])
    Runner(store).start(run_id)
    link_gate = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "link_gate")
    link_gate_payload = json.loads(Path(link_gate["path"]).read_text(encoding="utf-8"))
    assert any(item["url"] == "https://example.com/pricing" for item in link_gate_payload["source_reviews"])
    assert any(item["source"] == "subagent_candidate" for item in link_gate_payload["source_reviews"])
    dynamic_extracts = [
        row
        for row in store.list_tasks(run_id)
        if row["agent_name"] == "ExtractorAgent"
        and loads(row["metadata_json"], {}).get("source_url") == "https://example.com/pricing"
    ]
    assert dynamic_extracts
    assert any(row["artifact_type"] == "structured_knowledge" for row in store.list_artifacts(run_id))
    assert any(row["source_type"] == "document" for row in store.list_citations(run_id))
    diagnosis_after = build_run_diagnosis(store, run_id)
    assert diagnosis_after["formal_evidence_available"] is True
    assert diagnosis_after["subagent_summary"]["candidate_research_source_count"] == 1
    assert diagnosis_after["subagent_summary"]["pending_promotion_count"] == 0
    from insightswarm.observability.trace import ensure_collaboration_trace

    trace = ensure_collaboration_trace(store, run_id)["trace"]
    assert any(item["artifact_type"] == "candidate_research_source" for item in trace["subagent_source_promotions"])


def test_subagent_finding_promotion_blocks_without_urls(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("subagent-promotion-blocked", {"quality_mode": "test"})
    task_id = store.create_task(run_id, "Search", "SearchAgent")
    finding_id = store.write_artifact(
        run_id,
        task_id,
        "research_finding",
        "application/json",
        json.dumps(
            {
                "claim": "No usable sources found.",
                "confidence": 0.4,
                "evidence_candidates": [],
                "source_urls": [],
                "open_questions": ["Need source discovery"],
                "risk_flags": [],
                "recommended_next_tasks": [],
            },
            ensure_ascii=True,
            indent=2,
        ),
        metadata={"schema": "research_finding.v1"},
        suffix=".json",
    )
    from insightswarm.subagent_promotion import promote_finding_sources

    result = promote_finding_sources(store, run_id, finding_id=finding_id)
    assert result["status"] == "blocked"
    assert result["reason"] == "no_new_source_urls"
    assert any(row["artifact_type"] == "subagent_source_promotion" for row in store.list_artifacts(run_id))
    assert not any(row["artifact_type"] == "candidate_research_source" for row in store.list_artifacts(run_id))


def test_followup_plan_and_spawn_followup_cli_diagnosis_trace(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "followup-plan",
        {
            "quality_mode": "production",
            "query": "ExampleCo pricing",
            "competitor": "ExampleCo",
            "max_subagents_per_run": 2,
            "max_spawn_depth": 1,
            "max_parallel_subagents": 2,
            "allowed_subagent_roles": ["SearchAgent", "SkepticReviewAgent"],
        },
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    skeptic_task = store.create_task(run_id, "SkepticReview", "SkepticReviewAgent", [parent_task])
    skeptic_payload = {
        "status": "reviewed",
        "reviewed_analysis_artifact_id": None,
        "challenged_claims": [],
        "evidence_gaps": ["Need fresher official pricing evidence."],
        "source_risks": [],
        "recommended_checks": [],
        "non_blocking": True,
    }
    store.write_artifact(
        run_id,
        skeptic_task,
        "skeptic_review",
        "application/json",
        json.dumps(skeptic_payload, ensure_ascii=True, indent=2),
        metadata={"non_blocking": True, "evidence_gap_count": 1},
        suffix=".json",
    )
    finding_id = store.write_artifact(
        run_id,
        skeptic_task,
        "research_finding",
        "application/json",
        json.dumps(
            {
                "claim": "Subagent needs promotion.",
                "confidence": 0.5,
                "evidence_candidates": [{"url": "https://example.com/pricing", "snippet": "ExampleCo pricing"}],
                "source_urls": ["https://example.com/pricing"],
                "open_questions": [],
                "risk_flags": [],
                "recommended_next_tasks": [],
            },
            ensure_ascii=True,
            indent=2,
        ),
        metadata={"schema": "research_finding.v1"},
        suffix=".json",
    )
    diagnosis_before = build_run_diagnosis(store, run_id)
    assert any("plan-followups" in action for action in diagnosis_before["recommended_next_actions"])

    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "plan-followups",
                "--run-id",
                run_id,
            ]
        )
        == 0
    )
    plan_result = json.loads(capsys.readouterr().out)
    assert plan_result["status"] == "planned"
    assert plan_result["item_count"] >= 3
    assert any(item["priority"] == "high" for item in plan_result["items"])
    assert any(item.get("source_hint") == finding_id for item in plan_result["items"])
    plan_id = plan_result["plan_artifact_id"]
    assert not store.list_citations(run_id)

    item_id = next(item["item_id"] for item in plan_result["items"] if item["recommended_role"] == "SearchAgent")
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "spawn-followup",
                "--run-id",
                run_id,
                "--plan-id",
                plan_id,
                "--item-id",
                item_id,
            ]
        )
        == 0
    )
    decision = json.loads(capsys.readouterr().out)
    assert decision["status"] == "spawned"
    assert decision["subagent_task_id"]
    subagent_meta = loads(store.get_task(decision["subagent_task_id"])["metadata_json"], {})
    assert subagent_meta["subagent"] is True
    assert subagent_meta["spawn_reason"] == f"followup:{plan_id}:{item_id}"
    assert any(row["artifact_type"] == "research_followup_decision" for row in store.list_artifacts(run_id))
    diagnosis_after = build_run_diagnosis(store, run_id)
    assert diagnosis_after["followup_summary"]["plan_count"] == 1
    assert diagnosis_after["followup_summary"]["spawned_decision_count"] == 1
    from insightswarm.observability.trace import ensure_collaboration_trace

    trace = ensure_collaboration_trace(store, run_id)["trace"]
    assert any(item["artifact_type"] == "research_followup_plan" for item in trace["research_followups"])
    assert any(item.get("subagent_task_id") == decision["subagent_task_id"] for item in trace["research_followups"])


def test_spawn_followup_rejects_when_role_not_allowed(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "followup-rejected",
        {
            "quality_mode": "test",
            "query": "ExampleCo pricing",
            "allowed_subagent_roles": ["SkepticReviewAgent"],
            "max_subagents_per_run": 1,
        },
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "plan-followups", "--run-id", run_id]) == 0
    plan = json.loads(capsys.readouterr().out)
    item_id = next(item["item_id"] for item in plan["items"] if item["recommended_role"] == "SearchAgent")
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "spawn-followup",
                "--run-id",
                run_id,
                "--plan-id",
                plan["plan_artifact_id"],
                "--item-id",
                item_id,
            ]
        )
        == 2
    )
    decision = json.loads(capsys.readouterr().out)
    assert decision["status"] == "rejected"
    assert decision["reason"] == "role_not_allowed"
    assert any(
        row["artifact_type"] == "research_followup_decision"
        and loads(row["metadata_json"], {}).get("status") == "rejected"
        for row in store.list_artifacts(run_id)
    )


def test_continue_followup_plan_item_reaches_candidate_source(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "followup-round",
        {
            "quality_mode": "test",
            "query": "ExampleCo pricing",
            "competitor": "ExampleCo",
            "source_urls": ["https://example.com/pricing"],
            "allowed_subagent_roles": ["SearchAgent"],
            "max_subagents_per_run": 2,
            "max_parallel_subagents": 2,
        },
    )
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "plan-followups", "--run-id", run_id]) == 0
    plan = json.loads(capsys.readouterr().out)
    item_id = plan["items"][0]["item_id"]
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "continue-followup",
                "--run-id",
                run_id,
                "--plan-id",
                plan["plan_artifact_id"],
                "--item-id",
                item_id,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "candidate_ready"
    assert result["candidate_research_source_ids"]
    assert result["research_finding_artifact_id"]
    assert result["promotion_artifact_id"]
    assert not store.list_citations(run_id)
    artifacts = store.list_artifacts(run_id)
    assert any(row["artifact_type"] == "research_followup_round" for row in artifacts)
    assert any(row["artifact_type"] == "research_followup_round_step" for row in artifacts)
    assert any(row["artifact_type"] == "candidate_research_source" for row in artifacts)
    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["followup_summary"]["round_count"] == 1
    assert diagnosis["followup_summary"]["candidate_ready_round_count"] == 1
    assert diagnosis["followup_summary"]["latest_ready_candidate_source_ids"] == result["candidate_research_source_ids"]
    assert diagnosis["continuation_runtime_summary"]["by_kind"]["research_followup_round"]["continuation_count"] == 1
    assert any("LinkGate/Extract" in action for action in diagnosis["recommended_next_actions"])
    from insightswarm.observability.trace import build_collaboration_trace

    trace = build_collaboration_trace(store, run_id)
    assert any(item["artifact_type"] == "research_followup_round" for item in trace["research_followups"])
    assert any(item["artifact_type"] == "research_followup_round_step" for item in trace["research_followups"])


def test_continue_followup_existing_decision_resume(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "followup-round-resume",
        {
            "quality_mode": "test",
            "query": "ExampleCo pricing",
            "competitor": "ExampleCo",
            "source_urls": ["https://example.com/pricing"],
            "allowed_subagent_roles": ["SearchAgent"],
            "max_subagents_per_run": 2,
            "max_parallel_subagents": 2,
        },
    )
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "plan-followups", "--run-id", run_id]) == 0
    plan = json.loads(capsys.readouterr().out)
    item_id = plan["items"][0]["item_id"]
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "spawn-followup",
                "--run-id",
                run_id,
                "--plan-id",
                plan["plan_artifact_id"],
                "--item-id",
                item_id,
            ]
        )
        == 0
    )
    decision = json.loads(capsys.readouterr().out)
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "continue-followup",
                "--run-id",
                run_id,
                "--decision-id",
                decision["decision_artifact_id"],
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "candidate_ready"
    assert result["decision_artifact_id"] == decision["decision_artifact_id"]


def test_continue_followup_rejected_decision_blocks_without_new_task(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "followup-round-rejected",
        {
            "quality_mode": "test",
            "query": "ExampleCo pricing",
            "competitor": "ExampleCo",
            "allowed_subagent_roles": ["SkepticReviewAgent"],
            "max_subagents_per_run": 1,
        },
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "plan-followups", "--run-id", run_id]) == 0
    plan = json.loads(capsys.readouterr().out)
    search_item = next(item for item in plan["items"] if item["recommended_role"] == "SearchAgent")
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "spawn-followup",
                "--run-id",
                run_id,
                "--plan-id",
                plan["plan_artifact_id"],
                "--item-id",
                search_item["item_id"],
            ]
        )
        == 2
    )
    decision = json.loads(capsys.readouterr().out)
    before_tasks = len(store.list_tasks(run_id))
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "continue-followup",
                "--run-id",
                run_id,
                "--decision-id",
                decision["decision_artifact_id"],
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked"
    assert result["stop_reason"] == "followup_decision_not_spawned"
    assert len(store.list_tasks(run_id)) == before_tasks


def test_continue_followup_blocks_when_finding_has_no_urls(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "followup-round-no-url",
        {
            "quality_mode": "test",
            "query": "ExampleCo pricing",
            "competitor": "ExampleCo",
            "allowed_subagent_roles": ["SearchAgent"],
            "max_subagents_per_run": 2,
            "max_parallel_subagents": 2,
        },
    )
    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "plan-followups", "--run-id", run_id]) == 0
    plan = json.loads(capsys.readouterr().out)
    item_id = plan["items"][0]["item_id"]
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "spawn-followup",
                "--run-id",
                run_id,
                "--plan-id",
                plan["plan_artifact_id"],
                "--item-id",
                item_id,
            ]
        )
        == 0
    )
    decision = json.loads(capsys.readouterr().out)
    subagent_task_id = decision["subagent_task_id"]
    store.set_task_status(subagent_task_id, "completed")
    finding_id = store.write_artifact(
        run_id,
        subagent_task_id,
        "research_finding",
        "application/json",
        json.dumps(
            {
                "claim": "No usable source URL found.",
                "confidence": 0.2,
                "evidence_candidates": [],
                "source_urls": [],
                "open_questions": [],
                "risk_flags": ["no_sources"],
                "recommended_next_tasks": [],
            },
            ensure_ascii=True,
            indent=2,
        ),
        metadata={"schema": "research_finding.v1", "subagent": True},
        suffix=".json",
    )
    store.write_artifact(
        run_id,
        subagent_task_id,
        "subagent_handoff",
        "application/json",
        json.dumps(
            {
                "schema": "subagent_handoff.v1",
                "subagent_task_id": subagent_task_id,
                "finding_artifact_id": finding_id,
            },
            ensure_ascii=True,
            indent=2,
        ),
        metadata={"schema": "subagent_handoff.v1", "finding_artifact_id": finding_id, "subagent": True},
        suffix=".json",
    )
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "continue-followup",
                "--run-id",
                run_id,
                "--decision-id",
                decision["decision_artifact_id"],
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked_no_candidate_source"
    assert result["stop_reason"] == "no_new_source_urls"
    assert any(
        row["artifact_type"] == "subagent_source_promotion"
        and loads(row["metadata_json"], {}).get("status") == "blocked"
        for row in store.list_artifacts(run_id)
    )


def test_continue_candidate_single_candidate_reaches_citations(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "candidate-continuation-single",
        {
            "quality_mode": "test",
            "query": "ExampleCo pricing",
            "competitor": "ExampleCo",
        },
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "ExampleCo Pricing",
                "snippet": "ExampleCo pricing starts at $29 per seat monthly with analytics included.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "continue-candidate",
                "--run-id",
                run_id,
                "--candidate-id",
                candidate_id,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "citation_ready"
    assert result["candidate_research_source_ids"] == [candidate_id]
    assert result["link_gate_artifact_id"]
    assert result["dynamic_extract_task_ids"]
    assert result["raw_document_artifact_ids"]
    assert result["structured_knowledge_artifact_ids"]
    assert result["citation_ids"]
    assert result["analysis_continuation_artifact_id"]
    assert result["analysis_continuation_status"] == "analysis_ready"
    assert result["strategic_analysis_artifact_id"]
    assert result["review_qa_continuation_artifact_id"]
    assert result["review_qa_continuation_status"] == "qa_passed"
    assert result["qa_report_artifact_id"]
    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["candidate_continuation_summary"]["continuation_count"] == 1
    assert diagnosis["candidate_continuation_summary"]["citation_ready_count"] == 1
    assert diagnosis["analysis_continuation_summary"]["continuation_count"] == 1
    assert diagnosis["analysis_continuation_summary"]["analysis_ready_count"] == 1
    assert diagnosis["review_qa_continuation_summary"]["continuation_count"] == 1
    assert diagnosis["review_qa_continuation_summary"]["qa_passed_count"] == 1
    assert diagnosis["continuation_runtime_summary"]["continuation_count"] == 3
    assert diagnosis["continuation_runtime_summary"]["by_kind"]["candidate_continuation"]["continuation_count"] == 1
    assert diagnosis["continuation_runtime_summary"]["by_kind"]["analysis_continuation"]["continuation_count"] == 1
    assert diagnosis["continuation_runtime_summary"]["by_kind"]["review_qa_continuation"]["continuation_count"] == 1
    assert diagnosis["continuation_runtime_summary"]["latest_continuation"]["kind"] == "review_qa_continuation"
    assert any(
        edge["from_artifact_id"] == result["continuation_artifact_id"]
        and edge["to_artifact_type"] == "analysis_continuation"
        for edge in diagnosis["continuation_runtime_summary"]["lineage_edges"]
    )
    assert diagnosis["analysis_continuation_summary"]["latest_continuation"]["strategic_analysis_artifact_id"] == result["strategic_analysis_artifact_id"]
    assert any("continue-candidate" in action for action in diagnosis["recommended_next_actions"]) is False
    assert any("artifact inspect" in action for action in diagnosis["recommended_next_actions"])
    assert not any(row["artifact_type"] in {"report", "report_blocked"} for row in store.list_artifacts(run_id))
    from insightswarm.observability.trace import build_collaboration_trace

    trace = build_collaboration_trace(store, run_id)
    assert len(trace["continuation_runtime"]["continuations"]) == 3
    assert any(
        edge["from_artifact_id"] == result["analysis_continuation_artifact_id"]
        and edge["to_artifact_type"] == "review_qa_continuation"
        for edge in trace["continuation_runtime"]["lineage"]
    )
    assert any(item["artifact_type"] == "candidate_continuation" for item in trace["candidate_continuations"])
    assert any(item["artifact_type"] == "candidate_continuation_step" for item in trace["candidate_continuations"])
    assert any(item["artifact_type"] == "analysis_continuation" for item in trace["analysis_continuations"])
    assert any(item["artifact_type"] == "analysis_continuation_step" for item in trace["analysis_continuations"])
    assert any(item["artifact_type"] == "review_qa_continuation" for item in trace["review_qa_continuations"])
    assert any(item["artifact_type"] == "review_qa_continuation_step" for item in trace["review_qa_continuations"])
    candidate_continuation = json.loads(
        Path(store.get_artifact(result["continuation_artifact_id"])["path"]).read_text(encoding="utf-8")
    )
    assert candidate_continuation["continuation_kind"] == "candidate_continuation"
    assert candidate_continuation["scope"]["source_artifact_id"] == candidate_id


def test_continue_candidate_round_scopes_only_round_candidates(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "candidate-continuation-round",
        {
            "quality_mode": "test",
            "query": "ExampleCo pricing",
            "competitor": "ExampleCo",
        },
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_one = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing is $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    candidate_two = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/features",
                "title": "Features",
                "snippet": "Features include AI dashboards and alerts.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/features",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/features", "requires_link_gate": True},
        suffix=".json",
    )
    other_candidate = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://other.example/outside-scope",
                "title": "Outside Scope",
                "snippet": "Should not be consumed by this round.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://other.example/outside-scope",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://other.example/outside-scope", "requires_link_gate": True},
        suffix=".json",
    )
    round_id = store.write_artifact(
        run_id,
        parent_task,
        "research_followup_round",
        "application/json",
        json.dumps(
            {
                "schema": "research_followup_round.v1",
                "status": "candidate_ready",
                "candidate_research_source_ids": [candidate_one, candidate_two],
                "stop_reason": "candidate_research_source_ready",
            },
            ensure_ascii=True,
            indent=2,
        ),
        metadata={
            "schema": "research_followup_round.v1",
            "status": "candidate_ready",
            "stop_reason": "candidate_research_source_ready",
            "candidate_research_source_count": 2,
        },
        suffix=".json",
    )
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "continue-candidate",
                "--run-id",
                run_id,
                "--round-id",
                round_id,
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "citation_ready"
    assert result["candidate_research_source_ids"] == [candidate_one, candidate_two]
    link_gate = store.get_artifact(result["link_gate_artifact_id"])
    payload = json.loads(Path(link_gate["path"]).read_text(encoding="utf-8"))
    selected_urls = [item["url"] for item in payload["selected_urls"]]
    assert "https://example.com/pricing" in selected_urls
    assert "https://example.com/features" in selected_urls
    assert "https://other.example/outside-scope" not in selected_urls
    assert other_candidate not in result["candidate_research_source_ids"]
    assert result["analysis_continuation_status"] == "analysis_ready"
    assert result["review_qa_continuation_status"] == "qa_passed"
    assert result["strategic_analysis_artifact_id"]


def test_continue_candidate_blocks_when_candidate_missing(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("candidate-continuation-missing", {"quality_mode": "test"})
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "continue-candidate",
                "--run-id",
                run_id,
                "--candidate-id",
                "artifact_missing",
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked_no_candidate"
    assert result["stop_reason"] == "no_candidate_research_source"


def test_continue_candidate_blocks_without_linkgate_selection(tmp_path, monkeypatch, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "candidate-continuation-no-selection",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing is $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    from insightswarm import link_gate as link_gate_module

    def fake_gate_links(model, query, search_results, max_selected=5):
        return ([], {"strategy": "test_block"})

    monkeypatch.setattr(link_gate_module, "gate_links", fake_gate_links)
    monkeypatch.setattr("insightswarm.agents.link_gate.gate_links", fake_gate_links)
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "continue-candidate",
                "--run-id",
                run_id,
                "--candidate-id",
                candidate_id,
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked_no_linkgate_selection"
    assert result["stop_reason"] == "no_linkgate_selection"


def test_continue_candidate_blocks_without_citations(tmp_path, monkeypatch, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "candidate-continuation-no-citation",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing is $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    from insightswarm.agents import extractor as extractor_module

    original_run = extractor_module.ExtractorAgent.run

    def run_without_citations(self, context):
        original_run(self, context)
        task = context.store.get_task(context.task_id)
        task_metadata = loads(task["metadata_json"], {})
        task_metadata["citation_ids"] = []
        context.store.set_task_status(context.task_id, "completed", task_metadata)

    monkeypatch.setattr(extractor_module.ExtractorAgent, "run", run_without_citations)
    assert (
        cli_main(
            [
                "--db-path",
                str(store.db_path),
                "--artifact-dir",
                str(store.artifact_dir),
                "run",
                "continue-candidate",
                "--run-id",
                run_id,
                "--candidate-id",
                candidate_id,
            ]
        )
        == 2
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked_no_citation"
    assert result["stop_reason"] == "no_document_citation"
    assert not any(row["artifact_type"] == "analysis_continuation" for row in store.list_artifacts(run_id))


def test_continue_candidate_analyst_uses_full_run_evidence(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "candidate-continuation-full-run-analysis",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    historical_source = tmp_path / "historical.txt"
    historical_source.write_text("ExampleCo annual plan costs $99 and includes email support.", encoding="utf-8")
    raw_id = store.write_artifact(
        run_id,
        parent_task,
        "raw_document",
        "text/plain",
        historical_source.read_text(encoding="utf-8"),
        source_url="https://example.com/historical",
        metadata={"fetcher": "manual_source", "status": "ok", "competitor": "ExampleCo"},
    )
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "extract",
            "--run-id",
            run_id,
            "--raw-document-id",
            raw_id,
        ]
    ) == 0
    _ = capsys.readouterr().out
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly for teams.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-candidate",
            "--run-id",
            run_id,
            "--candidate-id",
            candidate_id,
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    analysis = json.loads(Path(store.get_artifact(result["strategic_analysis_artifact_id"])["path"]).read_text(encoding="utf-8"))
    assert analysis["inferences"]
    assert len(analysis["inferences"][0]["evidence_ids"]) >= 2
    continuation = json.loads(
        Path(store.get_artifact(result["analysis_continuation_artifact_id"])["path"]).read_text(encoding="utf-8")
    )
    assert continuation["citation_ids"] == result["citation_ids"]
    assert continuation["source_candidate_continuation_artifact_id"] == result["continuation_artifact_id"]


def test_continue_candidate_creates_new_analysis_each_time(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "candidate-continuation-repeat-analysis",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    first_candidate = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    second_candidate = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/features",
                "title": "Features",
                "snippet": "Features include AI dashboards and alerts.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/features",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/features", "requires_link_gate": True},
        suffix=".json",
    )
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-candidate",
            "--run-id",
            run_id,
            "--candidate-id",
            first_candidate,
        ]
    ) == 0
    first = json.loads(capsys.readouterr().out)
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-candidate",
            "--run-id",
            run_id,
            "--candidate-id",
            second_candidate,
        ]
    ) == 0
    second = json.loads(capsys.readouterr().out)
    analysis_continuations = [row for row in store.list_artifacts(run_id) if row["artifact_type"] == "analysis_continuation"]
    review_qa_continuations = [row for row in store.list_artifacts(run_id) if row["artifact_type"] == "review_qa_continuation"]
    strategic_analyses = [row for row in store.list_artifacts(run_id) if row["artifact_type"] == "strategic_analysis"]
    qa_reports = [row for row in store.list_artifacts(run_id) if row["artifact_type"] == "qa_report"]
    assert len(analysis_continuations) == 2
    assert len(review_qa_continuations) == 2
    assert len(strategic_analyses) == 2
    assert len(qa_reports) == 2
    assert first["analysis_continuation_artifact_id"] != second["analysis_continuation_artifact_id"]
    assert first["strategic_analysis_artifact_id"] != second["strategic_analysis_artifact_id"]
    assert first["review_qa_continuation_artifact_id"] != second["review_qa_continuation_artifact_id"]
    first_review = json.loads(Path(store.get_artifact(first["review_qa_continuation_artifact_id"])["path"]).read_text(encoding="utf-8"))
    second_review = json.loads(Path(store.get_artifact(second["review_qa_continuation_artifact_id"])["path"]).read_text(encoding="utf-8"))
    assert first_review["analyst_task_id"] != second_review["analyst_task_id"]
    first_qa = json.loads(Path(store.get_artifact(first["qa_report_artifact_id"])["path"]).read_text(encoding="utf-8"))
    second_qa = json.loads(Path(store.get_artifact(second["qa_report_artifact_id"])["path"]).read_text(encoding="utf-8"))
    assert first_qa["analyst_task_id"] == first_review["analyst_task_id"]
    assert second_qa["analyst_task_id"] == second_review["analyst_task_id"]


def test_analysis_continuation_blocks_on_phase_gate(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("analysis-continuation-phase-gate", {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"})
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    raw_id = store.write_artifact(
        run_id,
        parent_task,
        "raw_document",
        "text/plain",
        "ExampleCo pricing starts at $29.",
        source_url="https://example.com/pricing",
        metadata={"fetcher": "manual_source", "status": "ok", "competitor": "ExampleCo"},
    )
    citation_id = store.create_document_citation(run_id, parent_task, raw_id, "https://example.com/pricing", "$29", TextSpan(start=0, end=3), 0.9)
    continuation_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_continuation",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_continuation.v1",
                "status": "citation_ready",
                "candidate_research_source_ids": ["artifact_candidate"],
                "citation_ids": [citation_id],
                "structured_knowledge_artifact_ids": [],
                "stop_reason": "document_citations_ready",
            },
            ensure_ascii=True,
            indent=2,
        ),
        metadata={
            "schema": "candidate_continuation.v1",
            "status": "citation_ready",
            "stop_reason": "document_citations_ready",
            "citation_count": 1,
            "structured_knowledge_count": 0,
        },
        suffix=".json",
    )
    from insightswarm.analysis_continuation import continue_analysis_from_candidate_continuation

    result = continue_analysis_from_candidate_continuation(store, run_id, candidate_continuation_id=continuation_id)
    assert result["status"] == "blocked_phase_gate"
    assert result["stop_reason"] == "analyst_phase_gate_blocked"
    assert any(row["artifact_type"] == "analysis_continuation" for row in store.list_artifacts(run_id))
    assert not any(
        row["artifact_type"] == "strategic_analysis" and row["task_id"] == result["analyst_task_id"]
        for row in store.list_artifacts(run_id)
    )


def test_continue_candidate_qa_failure_marks_targeted_analyst_for_repair(tmp_path, monkeypatch, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "candidate-continuation-qa-failure",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    monkeypatch.setattr(
        "insightswarm.agents.qa.validate_qa_gates",
        lambda store, run_id: (
            70,
            [{"category": "evidence", "gate": "citation_coverage", "error": "forced qa failure"}],
        ),
    )
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-candidate",
            "--run-id",
            run_id,
            "--candidate-id",
            candidate_id,
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["review_qa_continuation_status"] == "qa_failed_needs_repair"
    review = json.loads(Path(store.get_artifact(result["review_qa_continuation_artifact_id"])["path"]).read_text(encoding="utf-8"))
    qa_report = json.loads(Path(store.get_artifact(result["qa_report_artifact_id"])["path"]).read_text(encoding="utf-8"))
    analyst_task = dict(store.get_task(review["analyst_task_id"]))
    analyst_metadata = loads(analyst_task["metadata_json"], {})
    assert analyst_task["status"] == "needs_repair"
    assert qa_report["passed"] is False
    assert qa_report["analyst_task_id"] == review["analyst_task_id"]
    assert analyst_metadata["qa_artifact_id"] == result["qa_report_artifact_id"]
    assert not any(row["artifact_type"] in {"report", "report_blocked"} for row in store.list_artifacts(run_id))


def test_continue_repair_review_qa_id_repairs_analysis_and_reruns_qa(tmp_path, monkeypatch, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "repair-continuation-review-qa",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    qa_calls = {"count": 0}

    def fail_once_then_pass(store, run_id):
        qa_calls["count"] += 1
        if qa_calls["count"] == 1:
            return (70, [{"category": "evidence", "gate": "citation_coverage", "error": "forced qa failure"}])
        return (100, [])

    monkeypatch.setattr("insightswarm.agents.qa.validate_qa_gates", fail_once_then_pass)
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-candidate",
            "--run-id",
            run_id,
            "--candidate-id",
            candidate_id,
        ]
    ) == 0
    failed = json.loads(capsys.readouterr().out)
    assert failed["review_qa_continuation_status"] == "qa_failed_needs_repair"
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-repair",
            "--run-id",
            run_id,
            "--review-qa-id",
            failed["review_qa_continuation_artifact_id"],
        ]
    ) == 0
    repaired = json.loads(capsys.readouterr().out)
    assert repaired["status"] == "repair_ready"
    assert repaired["source_review_qa_continuation_artifact_id"] == failed["review_qa_continuation_artifact_id"]
    assert repaired["repaired_analyst_task_id"]
    assert repaired["repaired_analyst_task_id"] != repaired["original_analyst_task_id"]
    assert repaired["repaired_strategic_analysis_artifact_id"]
    assert repaired["downstream_review_qa_continuation_artifact_id"]
    assert repaired["downstream_review_qa_status"] == "qa_passed"
    assert repaired["downstream_qa_report_artifact_id"]
    assert not any(row["artifact_type"] in {"report", "report_blocked"} for row in store.list_artifacts(run_id))
    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["repair_continuation_summary"]["continuation_count"] == 1
    assert diagnosis["repair_continuation_summary"]["repair_ready_count"] == 1
    assert diagnosis["continuation_runtime_summary"]["by_kind"]["repair_continuation"]["continuation_count"] == 1
    assert any(repaired["repaired_strategic_analysis_artifact_id"] in action for action in diagnosis["recommended_next_actions"])
    assert any(repaired["downstream_qa_report_artifact_id"] in action for action in diagnosis["recommended_next_actions"])
    from insightswarm.observability.trace import build_collaboration_trace

    trace = build_collaboration_trace(store, run_id)
    assert any(item["artifact_type"] == "repair_continuation" for item in trace["repair_continuations"])
    assert any(
        edge["from_artifact_id"] == failed["review_qa_continuation_artifact_id"]
        and edge["to_artifact_type"] == "repair_continuation"
        for edge in trace["continuation_runtime"]["lineage"]
    )


def test_continue_repair_qa_report_id_works(tmp_path, monkeypatch, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "repair-continuation-qa-report",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    qa_calls = {"count": 0}

    def fail_once_then_pass(store, run_id):
        qa_calls["count"] += 1
        if qa_calls["count"] == 1:
            return (70, [{"category": "evidence", "gate": "citation_coverage", "error": "forced qa failure"}])
        return (100, [])

    monkeypatch.setattr("insightswarm.agents.qa.validate_qa_gates", fail_once_then_pass)
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-candidate",
            "--run-id",
            run_id,
            "--candidate-id",
            candidate_id,
        ]
    ) == 0
    failed = json.loads(capsys.readouterr().out)
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-repair",
            "--run-id",
            run_id,
            "--qa-report-id",
            failed["qa_report_artifact_id"],
        ]
    ) == 0
    repaired = json.loads(capsys.readouterr().out)
    assert repaired["status"] == "repair_ready"
    assert repaired["source_qa_report_artifact_id"] == failed["qa_report_artifact_id"]
    assert repaired["downstream_review_qa_status"] == "qa_passed"


def test_continue_repair_blocks_when_qa_already_passed(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "repair-continuation-passed",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-candidate",
            "--run-id",
            run_id,
            "--candidate-id",
            candidate_id,
        ]
    ) == 0
    passed = json.loads(capsys.readouterr().out)
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-repair",
            "--run-id",
            run_id,
            "--review-qa-id",
            passed["review_qa_continuation_artifact_id"],
        ]
    ) == 2
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "blocked_not_repairable"
    assert blocked["stop_reason"] == "qa_already_passed"
    assert not blocked["repaired_analyst_task_id"]


def test_continue_repair_blocks_when_retry_limit_reached(tmp_path, monkeypatch, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "repair-continuation-retry-limit",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    monkeypatch.setattr(
        "insightswarm.agents.qa.validate_qa_gates",
        lambda store, run_id: (
            70,
            [{"category": "evidence", "gate": "citation_coverage", "error": "forced qa failure"}],
        ),
    )
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-candidate",
            "--run-id",
            run_id,
            "--candidate-id",
            candidate_id,
        ]
    ) == 0
    failed = json.loads(capsys.readouterr().out)
    review = json.loads(Path(store.get_artifact(failed["review_qa_continuation_artifact_id"])["path"]).read_text(encoding="utf-8"))
    store.set_task_status(review["analyst_task_id"], "blocked", {"human_intervention_required": True}, retry_delta=2)
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-repair",
            "--run-id",
            run_id,
            "--review-qa-id",
            failed["review_qa_continuation_artifact_id"],
        ]
    ) == 2
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "blocked_human_intervention_required"
    assert blocked["stop_reason"] == "analyst_human_intervention_required"


def test_continue_writer_review_qa_id_creates_delivery_bundle(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "writer-continuation-review-qa",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-candidate",
            "--run-id",
            run_id,
            "--candidate-id",
            candidate_id,
        ]
    ) == 0
    candidate = json.loads(capsys.readouterr().out)
    assert candidate["review_qa_continuation_status"] == "qa_passed"
    diagnosis_before = build_run_diagnosis(store, run_id)
    assert any("continue-writer" in action for action in diagnosis_before["recommended_next_actions"])
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-writer",
            "--run-id",
            run_id,
            "--review-qa-id",
            candidate["review_qa_continuation_artifact_id"],
        ]
    ) == 0
    writer = json.loads(capsys.readouterr().out)
    assert writer["status"] == "report_ready"
    assert writer["source_review_qa_continuation_artifact_id"] == candidate["review_qa_continuation_artifact_id"]
    assert writer["qa_report_artifact_id"] == candidate["qa_report_artifact_id"]
    assert writer["writer_task_id"]
    assert writer["report_artifact_id"]
    assert writer["citations_export_artifact_id"]
    assert writer["qa_report_export_artifact_id"]
    assert not any(
        row["artifact_type"] == "report"
        and row["task_id"] != writer["writer_task_id"]
        for row in store.list_artifacts(run_id)
    )
    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["writer_continuation_summary"]["continuation_count"] == 1
    assert diagnosis["writer_continuation_summary"]["report_ready_count"] == 1
    assert diagnosis["continuation_runtime_summary"]["by_kind"]["writer_continuation"]["continuation_count"] == 1
    assert any(writer["report_artifact_id"] in action for action in diagnosis["recommended_next_actions"])
    from insightswarm.observability.trace import build_collaboration_trace

    trace = build_collaboration_trace(store, run_id)
    assert any(item["artifact_type"] == "writer_continuation" for item in trace["writer_continuations"])
    assert any(
        edge["from_artifact_id"] == candidate["review_qa_continuation_artifact_id"]
        and edge["to_artifact_type"] == "writer_continuation"
        for edge in trace["continuation_runtime"]["lineage"]
    )


def test_continue_writer_qa_report_id_creates_delivery_bundle(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "writer-continuation-qa-report",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-candidate",
            "--run-id",
            run_id,
            "--candidate-id",
            candidate_id,
        ]
    ) == 0
    candidate = json.loads(capsys.readouterr().out)
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-writer",
            "--run-id",
            run_id,
            "--qa-report-id",
            candidate["qa_report_artifact_id"],
        ]
    ) == 0
    writer = json.loads(capsys.readouterr().out)
    assert writer["status"] == "report_ready"
    assert writer["source_qa_report_artifact_id"] == candidate["qa_report_artifact_id"]
    assert writer["report_artifact_id"]


def test_continue_writer_repair_id_uses_downstream_qa(tmp_path, monkeypatch, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "writer-continuation-repair",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    qa_calls = {"count": 0}

    def fail_once_then_pass(store, run_id):
        qa_calls["count"] += 1
        if qa_calls["count"] == 1:
            return (70, [{"category": "evidence", "gate": "citation_coverage", "error": "forced qa failure"}])
        return (100, [])

    monkeypatch.setattr("insightswarm.agents.qa.validate_qa_gates", fail_once_then_pass)
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-candidate",
            "--run-id",
            run_id,
            "--candidate-id",
            candidate_id,
        ]
    ) == 0
    failed = json.loads(capsys.readouterr().out)
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-repair",
            "--run-id",
            run_id,
            "--review-qa-id",
            failed["review_qa_continuation_artifact_id"],
        ]
    ) == 0
    repaired = json.loads(capsys.readouterr().out)
    assert repaired["downstream_review_qa_status"] == "qa_passed"
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-writer",
            "--run-id",
            run_id,
            "--repair-id",
            repaired["repair_continuation_artifact_id"],
        ]
    ) == 0
    writer = json.loads(capsys.readouterr().out)
    assert writer["status"] == "report_ready"
    assert writer["source_repair_continuation_artifact_id"] == repaired["repair_continuation_artifact_id"]
    assert writer["source_review_qa_continuation_artifact_id"] == repaired["downstream_review_qa_continuation_artifact_id"]
    assert writer["report_artifact_id"]


def test_continue_writer_blocks_when_qa_not_passed(tmp_path, monkeypatch, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "writer-continuation-qa-failed",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    candidate_id = store.write_artifact(
        run_id,
        parent_task,
        "candidate_research_source",
        "application/json",
        json.dumps(
            {
                "schema": "candidate_research_source.v1",
                "source_url": "https://example.com/pricing",
                "title": "Pricing",
                "snippet": "Pricing starts at $29 monthly.",
                "requires_link_gate": True,
            },
            ensure_ascii=True,
            indent=2,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    monkeypatch.setattr(
        "insightswarm.agents.qa.validate_qa_gates",
        lambda store, run_id: (
            70,
            [{"category": "evidence", "gate": "citation_coverage", "error": "forced qa failure"}],
        ),
    )
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-candidate",
            "--run-id",
            run_id,
            "--candidate-id",
            candidate_id,
        ]
    ) == 0
    failed = json.loads(capsys.readouterr().out)
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-writer",
            "--run-id",
            run_id,
            "--review-qa-id",
            failed["review_qa_continuation_artifact_id"],
        ]
    ) == 2
    blocked = json.loads(capsys.readouterr().out)
    assert blocked["status"] == "blocked_qa_not_passed"
    assert not blocked["writer_task_id"]
    assert not any(loads(row["metadata_json"], {}).get("writer_continuation") for row in store.list_tasks(run_id))


def test_continue_writer_reports_blocked_delivery_ready(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "writer-continuation-blocked-delivery",
        {"query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    raw_id = store.write_artifact(
        run_id,
        parent_task,
        "raw_document",
        "text/plain",
        "ExampleCo pricing starts at $29 monthly.",
        source_url="https://example.com/pricing",
        metadata={"fetcher": "manual_source", "status": "ok", "competitor": "ExampleCo"},
    )
    citation_id = store.create_document_citation(run_id, parent_task, raw_id, "https://example.com/pricing", "$29", TextSpan(start=28, end=31), 0.9)
    structured_id = store.write_artifact(
        run_id,
        parent_task,
        "structured_knowledge",
        "application/json",
        json.dumps({"facts": [{"field": "price", "value": "$29", "citation_id": citation_id}]}),
        metadata={"citation_ids": [citation_id]},
        suffix=".json",
    )
    analyst_task = store.create_task(run_id, "Synthesize", "StrategicAnalystAgent", [parent_task])
    store.set_task_status(analyst_task, "completed", {"analysis": {"status": "ok"}})
    analysis_id = store.write_artifact(
        run_id,
        analyst_task,
        "strategic_analysis",
        "application/json",
        json.dumps({"status": "ok", "inferences": [{"claim": "Pricing starts at $29.", "evidence_ids": [citation_id]}]}),
        metadata={"inference_ids": []},
        suffix=".json",
    )
    qa_task = store.create_task(run_id, "QA", "QAAgent", [analyst_task])
    qa_report_id = store.write_artifact(
        run_id,
        qa_task,
        "qa_report",
        "application/json",
        json.dumps({"score": 100, "passed": True, "rejection": None, "analyst_task_id": analyst_task}),
        metadata={"passed": True, "analyst_task_id": analyst_task},
        suffix=".json",
    )
    review_id = store.write_artifact(
        run_id,
        qa_task,
        "review_qa_continuation",
        "application/json",
        json.dumps(
            {
                "schema": "review_qa_continuation.v1",
                "status": "qa_passed",
                "source_analysis_continuation_artifact_id": "artifact_analysis_continuation",
                "analyst_task_id": analyst_task,
                "strategic_analysis_artifact_id": analysis_id,
                "qa_task_id": qa_task,
                "qa_report_artifact_id": qa_report_id,
                "structured_knowledge_artifact_ids": [structured_id],
                "stop_reason": "qa_passed",
            }
        ),
        metadata={"schema": "review_qa_continuation.v1", "status": "qa_passed", "qa_report_artifact_id": qa_report_id},
        suffix=".json",
    )
    assert cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "continue-writer",
            "--run-id",
            run_id,
            "--review-qa-id",
            review_id,
        ]
    ) == 0
    writer = json.loads(capsys.readouterr().out)
    assert writer["status"] == "blocked_delivery_ready"
    assert writer["report_blocked_artifact_id"]
    assert not writer["report_artifact_id"]
    diagnosis = build_run_diagnosis(store, run_id)
    assert diagnosis["writer_continuation_summary"]["blocked_delivery_ready_count"] == 1
    assert any(writer["report_blocked_artifact_id"] in action for action in diagnosis["recommended_next_actions"])


def test_review_qa_continuation_blocks_without_analysis(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "review-qa-continuation-no-analysis",
        {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"},
    )
    parent_task = store.create_task(run_id, "ResearchLead", "ResearchLeadAgent")
    store.set_task_status(parent_task, "completed")
    analysis_continuation_id = store.write_artifact(
        run_id,
        parent_task,
        "analysis_continuation",
        "application/json",
        json.dumps(
            {
                "schema": "analysis_continuation.v1",
                "status": "analysis_ready",
                "source_candidate_continuation_artifact_id": "artifact_candidate_continuation",
                "candidate_research_source_ids": ["artifact_candidate"],
                "citation_ids": ["citation_doc"],
                "structured_knowledge_artifact_ids": ["artifact_sk"],
                "analyst_task_id": "task_missing_analysis",
                "strategic_analysis_artifact_id": None,
                "stop_reason": "strategic_analysis_ready",
            },
            ensure_ascii=True,
            indent=2,
        ),
        metadata={
            "schema": "analysis_continuation.v1",
            "status": "analysis_ready",
            "stop_reason": "strategic_analysis_ready",
            "source_candidate_continuation_artifact_id": "artifact_candidate_continuation",
            "analyst_task_id": "task_missing_analysis",
            "strategic_analysis_artifact_id": None,
        },
        suffix=".json",
    )
    from insightswarm.review_qa_continuation import continue_review_qa_from_analysis_continuation

    result = continue_review_qa_from_analysis_continuation(
        store,
        run_id,
        analysis_continuation_id=analysis_continuation_id,
    )
    assert result["status"] == "blocked_no_analysis"
    assert result["stop_reason"] == "missing_strategic_analysis"
    assert any(row["artifact_type"] == "review_qa_continuation" for row in store.list_artifacts(run_id))
def test_source_acquisition_gateway_normalizes_candidate_source_without_evidence(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("phase43-gateway", {"quality_mode": "test"})
    source_id = store.write_artifact(
        run_id,
        None,
        "candidate_source",
        "application/json",
        json.dumps(
            {
                "candidate": {
                    "source_url": "https://example.com/pricing",
                    "title": "Example pricing",
                    "text_preview": "Pricing source candidate from BrowserAgent.",
                    "confidence": 0.8,
                    "risk_warnings": [],
                }
            },
            ensure_ascii=True,
        ),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_source.v1"},
        suffix=".json",
    )

    rc = cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "normalize-source",
            "--run-id",
            run_id,
            "--artifact-id",
            source_id,
        ]
    )
    result = json.loads(capsys.readouterr().out)
    artifacts = [dict(row) for row in store.list_artifacts(run_id)]

    assert rc == 0
    assert result["status"] == "candidate_ready"
    assert result["candidate_research_source_ids"]
    assert any(row["artifact_type"] == "source_acquisition_gateway" for row in artifacts)
    assert any(row["artifact_type"] == "candidate_research_source" for row in artifacts)
    assert store.list_citations(run_id) == []
    assert build_run_diagnosis(store, run_id)["source_acquisition_gateway_summary"]["candidate_ready_count"] == 1


def test_source_acquisition_gateway_duplicate_blocks(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("phase43-gateway-duplicate", {"quality_mode": "test"})
    finding_id = store.write_artifact(
        run_id,
        None,
        "research_finding",
        "application/json",
        json.dumps(
            {
                "evidence_candidates": [{"source_url": "https://example.com/source", "title": "Example source"}],
                "confidence": 0.7,
                "risk_flags": [],
            },
            ensure_ascii=True,
        ),
        metadata={"schema": "research_finding.v1"},
        suffix=".json",
    )

    args = [
        "--db-path",
        str(store.db_path),
        "--artifact-dir",
        str(store.artifact_dir),
        "run",
        "normalize-source",
        "--run-id",
        run_id,
        "--artifact-id",
        finding_id,
    ]
    assert cli_main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "candidate_ready"
    assert cli_main(args) == 2
    second = json.loads(capsys.readouterr().out)

    candidates = [row for row in store.list_artifacts(run_id) if row["artifact_type"] == "candidate_research_source"]
    assert second["status"] == "blocked_duplicate_source"
    assert len(candidates) == 1


def test_evidence_convergence_records_retain_without_writer(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("phase43-convergence", {"quality_mode": "test"})
    candidate_id = store.write_artifact(
        run_id,
        None,
        "candidate_research_source",
        "application/json",
        json.dumps({"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing"}, ensure_ascii=True),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_research_source.v1", "source_url": "https://example.com/pricing", "requires_link_gate": True},
        suffix=".json",
    )
    candidate_continuation_id = store.write_artifact(
        run_id,
        None,
        "candidate_continuation",
        "application/json",
        json.dumps(
            {
                "status": "citation_ready",
                "candidate_research_source_ids": [candidate_id],
                "citation_ids": ["doc_test"],
                "stop_reason": "citation_ready",
            },
            ensure_ascii=True,
        ),
        metadata={"schema": "candidate_continuation.v1", "status": "citation_ready"},
        suffix=".json",
    )
    analysis_continuation_id = store.write_artifact(
        run_id,
        None,
        "analysis_continuation",
        "application/json",
        json.dumps(
            {
                "status": "analysis_ready",
                "source_candidate_continuation_artifact_id": candidate_continuation_id,
                "candidate_research_source_ids": [candidate_id],
                "citation_ids": ["doc_test"],
                "strategic_analysis_artifact_id": "artifact_analysis",
            },
            ensure_ascii=True,
        ),
        metadata={"schema": "analysis_continuation.v1", "status": "analysis_ready"},
        suffix=".json",
    )
    qa_report_id = store.write_artifact(
        run_id,
        None,
        "qa_report",
        "application/json",
        json.dumps({"passed": True, "status": "passed", "score": 1.0, "rejection": None, "warnings": []}, ensure_ascii=True),
        metadata={"passed": True},
        suffix=".json",
    )
    review_id = store.write_artifact(
        run_id,
        None,
        "review_qa_continuation",
        "application/json",
        json.dumps(
            {
                "status": "qa_passed",
                "source_analysis_continuation_artifact_id": analysis_continuation_id,
                "qa_report_artifact_id": qa_report_id,
                "qa_task_id": "task_qa",
            },
            ensure_ascii=True,
        ),
        metadata={"schema": "review_qa_continuation.v1", "status": "qa_passed", "qa_report_artifact_id": qa_report_id},
        suffix=".json",
    )

    rc = cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "converge-evidence",
            "--run-id",
            run_id,
            "--review-qa-id",
            review_id,
        ]
    )
    result = json.loads(capsys.readouterr().out)
    artifacts = [dict(row) for row in store.list_artifacts(run_id)]

    assert rc == 0
    assert result["status"] == "retain"
    assert result["candidate_research_source_ids"] == [candidate_id]
    assert result["citation_ids"] == ["doc_test"]
    assert any(row["artifact_type"] == "evidence_convergence_decision" for row in artifacts)
    assert not any(row["artifact_type"] in {"report", "report_blocked"} for row in artifacts)


def test_evidence_convergence_qa_failed_needs_more_evidence(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("phase43-convergence-failed", {"quality_mode": "test"})
    qa_report_id = store.write_artifact(
        run_id,
        None,
        "qa_report",
        "application/json",
        json.dumps(
            {
                "passed": False,
                "status": "needs_repair",
                "rejection": {"failures": [{"category": "evidence", "message": "citation coverage missing"}]},
                "warnings": [],
            },
            ensure_ascii=True,
        ),
        metadata={"passed": False},
        suffix=".json",
    )
    review_id = store.write_artifact(
        run_id,
        None,
        "review_qa_continuation",
        "application/json",
        json.dumps({"status": "qa_failed_needs_repair", "qa_report_artifact_id": qa_report_id}, ensure_ascii=True),
        metadata={"schema": "review_qa_continuation.v1", "status": "qa_failed_needs_repair", "qa_report_artifact_id": qa_report_id},
        suffix=".json",
    )

    rc = cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "converge-evidence",
            "--run-id",
            run_id,
            "--review-qa-id",
            review_id,
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert rc == 2
    assert result["status"] == "needs_more_evidence"


def test_govern_creates_isolated_decision_without_unbounded_loop(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("phase44-govern", {"quality_mode": "test"})
    source_id = store.write_artifact(
        run_id,
        None,
        "candidate_source",
        "application/json",
        json.dumps({"candidate": {"source_url": "https://example.com/pricing", "title": "Pricing"}}, ensure_ascii=True),
        source_url="https://example.com/pricing",
        metadata={"schema": "candidate_source.v1"},
        suffix=".json",
    )

    rc = cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "govern",
            "--run-id",
            run_id,
            "--max-steps",
            "1",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    artifacts = [dict(row) for row in store.list_artifacts(run_id)]
    context = next(row for row in artifacts if row["artifact_type"] == "isolated_context_envelope")
    context_payload = json.loads(Path(context["path"]).read_text(encoding="utf-8"))
    decision = next(row for row in artifacts if row["artifact_type"] == "governance_decision")
    decision_payload = json.loads(Path(decision["path"]).read_text(encoding="utf-8"))

    assert rc == 0
    assert result["status"] == "governed"
    assert len(result["decision_artifact_ids"]) == 1
    assert decision_payload["decision_type"] == "normalize_source"
    assert decision_payload["data"]["artifact_id"] == source_id
    assert context_payload["isolation"]["full_run_payload_history_included"] is False
    assert context_payload["agent_identity"]["agent_name"] == "LeadAgent"
    assert any(row["artifact_type"] == "agent_scratchpad" for row in artifacts)
    assert any(row["artifact_type"] == "candidate_research_source" for row in artifacts)
    assert not any(row["artifact_type"] in {"report", "report_blocked"} for row in artifacts)


def test_govern_assigns_code_mediated_browser_swarm_and_gateway(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run(
        "phase45-browser-governance",
        {
            "quality_mode": "test",
            "query": "Recover ExampleCo pricing source",
            "browser_source_target_url": "https://example.com/pricing",
            "browser_backend": "fake",
            "model_provider": "fake",
        },
    )
    task_id = store.create_task(run_id, "Discovery", "ScraperAgent")
    store.set_task_status(task_id, "completed")
    store.write_artifact(
        run_id,
        task_id,
        "fetch_failure",
        "application/json",
        json.dumps({"source_url": "https://example.invalid/pricing", "status": "error"}, ensure_ascii=True),
        source_url="https://example.invalid/pricing",
        metadata={"source_url": "https://example.invalid/pricing", "status": "error"},
        suffix=".json",
    )

    rc = cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "govern",
            "--run-id",
            run_id,
            "--max-steps",
            "1",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    artifacts = [dict(row) for row in store.list_artifacts(run_id)]
    browser_context = next(
        row
        for row in artifacts
        if row["artifact_type"] == "isolated_context_envelope"
        and loads(row["metadata_json"], {}).get("agent_name") == "BrowserAgent"
    )
    browser_context_payload = json.loads(Path(browser_context["path"]).read_text(encoding="utf-8"))
    decision = next(row for row in artifacts if row["artifact_type"] == "governance_decision")
    decision_payload = json.loads(Path(decision["path"]).read_text(encoding="utf-8"))

    assert rc == 0
    assert result["executed_results"][0]["swarm_execution"]["status"] == "browser_candidate_ready"
    assert decision_payload["decision_type"] == "assign_source_acquisition"
    assert decision_payload["model_governed"] is False
    assert browser_context_payload["agent_identity"]["agent_name"] == "BrowserAgent"
    assert browser_context_payload["isolation"]["full_run_payload_history_included"] is False
    assert any(row["artifact_type"] == "browser_page_state" for row in artifacts)
    assert any(row["artifact_type"] == "browser_code_result" for row in artifacts)
    assert any(row["artifact_type"] == "candidate_source" for row in artifacts)
    assert any(row["artifact_type"] == "source_acquisition_gateway" for row in artifacts)
    assert any(row["artifact_type"] == "candidate_research_source" for row in artifacts)
    assert not any(row["artifact_type"] in {"citation", "strategic_analysis", "qa_report", "report"} for row in artifacts)


def test_browser_vision_escalation_is_explicit_fallback(tmp_path):
    store = make_store(tmp_path)
    run_id = store.create_run("phase45-vision-fallback", {"quality_mode": "test", "model_provider": "fake"})
    result = run_browser_operation(
        store,
        run_id,
        mode="free_browser",
        goal="Inspect visually sparse page",
        backend="fake",
        target_url="https://example.com/pricing",
        vision_required=True,
        model_provider="fake",
        max_iterations=1,
    )
    model_calls = [dict(row) for row in store.conn.execute("SELECT * FROM model_calls WHERE run_id = ?", (run_id,))]

    assert result["vision_escalated"] is True
    assert result["vision_model"] == "fake-vision-v1"
    assert any(row["model"] == "fake-vision-v1" for row in model_calls)
    assert any(row["artifact_type"] == "browser_code_result" for row in store.list_artifacts(run_id))
    assert not store.list_citations(run_id)


def test_govern_delivery_requires_allow_delivery(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("phase44-delivery-request", {"quality_mode": "test"})
    qa_report_id = store.write_artifact(
        run_id,
        None,
        "qa_report",
        "application/json",
        json.dumps({"passed": True, "status": "passed", "score": 1.0}, ensure_ascii=True),
        metadata={"passed": True},
        suffix=".json",
    )
    review_id = store.write_artifact(
        run_id,
        None,
        "review_qa_continuation",
        "application/json",
        json.dumps({"status": "qa_passed", "qa_report_artifact_id": qa_report_id}, ensure_ascii=True),
        metadata={"schema": "review_qa_continuation.v1", "status": "qa_passed", "qa_report_artifact_id": qa_report_id},
        suffix=".json",
    )
    convergence_id = store.write_artifact(
        run_id,
        None,
        "evidence_convergence_decision",
        "application/json",
        json.dumps(
            {
                "status": "retain",
                "source_review_qa_continuation_artifact_id": review_id,
                "qa_report_artifact_id": qa_report_id,
                "citation_ids": [],
            },
            ensure_ascii=True,
        ),
        metadata={"schema": "evidence_convergence_decision.v1", "status": "retain", "source_review_qa_continuation_artifact_id": review_id},
        suffix=".json",
    )

    assert cli_main(["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "run", "govern", "--run-id", run_id, "--max-steps", "1", "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    artifacts = [dict(row) for row in store.list_artifacts(run_id)]
    request = next(row for row in artifacts if row["artifact_type"] == "delivery_request")
    request_payload = json.loads(Path(request["path"]).read_text(encoding="utf-8"))

    assert first["executed_results"][0]["status"] == "delivery_requested"
    assert request_payload["status"] == "pending_allow_delivery"
    assert request_payload["source_evidence_convergence_decision_artifact_id"] == convergence_id
    assert not any(row["artifact_type"] in {"report", "report_blocked", "writer_continuation"} for row in artifacts)


def test_govern_delivery_with_allow_delivery_executes_writer_boundary(tmp_path, capsys):
    store = make_store(tmp_path)
    run_id = store.create_run("phase44-delivery-execute", {"quality_mode": "test", "query": "ExampleCo pricing", "competitor": "ExampleCo"})
    qa_report_id = store.write_artifact(
        run_id,
        None,
        "qa_report",
        "application/json",
        json.dumps({"passed": True, "status": "passed", "score": 1.0}, ensure_ascii=True),
        metadata={"passed": True},
        suffix=".json",
    )
    review_id = store.write_artifact(
        run_id,
        None,
        "review_qa_continuation",
        "application/json",
        json.dumps({"status": "qa_passed", "qa_report_artifact_id": qa_report_id}, ensure_ascii=True),
        metadata={"schema": "review_qa_continuation.v1", "status": "qa_passed", "qa_report_artifact_id": qa_report_id},
        suffix=".json",
    )
    store.write_artifact(
        run_id,
        None,
        "evidence_convergence_decision",
        "application/json",
        json.dumps({"status": "retain", "source_review_qa_continuation_artifact_id": review_id, "qa_report_artifact_id": qa_report_id}, ensure_ascii=True),
        metadata={"schema": "evidence_convergence_decision.v1", "status": "retain", "source_review_qa_continuation_artifact_id": review_id},
        suffix=".json",
    )

    rc = cli_main(
        [
            "--db-path",
            str(store.db_path),
            "--artifact-dir",
            str(store.artifact_dir),
            "run",
            "govern",
            "--run-id",
            run_id,
            "--max-steps",
            "1",
            "--allow-delivery",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    artifacts = [dict(row) for row in store.list_artifacts(run_id)]

    assert rc == 0
    assert result["executed_results"][0]["status"] in {"report_ready", "blocked_delivery_ready"}
    assert any(row["artifact_type"] == "delivery_request" for row in artifacts)
    assert any(row["artifact_type"] == "writer_continuation" for row in artifacts)
    assert any(row["artifact_type"] in {"report", "report_blocked"} for row in artifacts)


def test_phase47_run_ask_objective_state_swarm_reaches_delivery_request(tmp_path, capsys):
    store = make_store(tmp_path)
    args = ["--db-path", str(store.db_path), "--artifact-dir", str(store.artifact_dir), "--model-provider", "fake"]

    rc = cli_main(
        [
            *args,
            "run",
            "ask",
            "--query",
            "DeepSeek next strategy public evidence intelligence",
            "--max-steps",
            "8",
            "--quality-mode",
            "test",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    run_id = result["run_id"]
    artifacts = [dict(row) for row in store.list_artifacts(run_id)]
    events = [dict(row) for row in store.conn.execute("SELECT * FROM agent_events WHERE run_id = ? ORDER BY created_at", (run_id,))]
    objective_summary = build_run_diagnosis(store, run_id)["objective_runtime_summary"]

    assert rc == 0
    assert result["status"] == "objective_governed"
    assert result["final_state"] == "delivered"
    assert result["stop_reason"] in {"delivered", "writer_report_ready", "writer_blocked_delivery_ready"}
    assert any(row["artifact_type"] == "intelligence_objective" for row in artifacts)
    assert any(row["artifact_type"] == "capability_arbitration" for row in artifacts)
    assert any(row["artifact_type"] == "objective_state_transition" for row in artifacts)
    assert any(row["artifact_type"] == "delivery_request" for row in artifacts)
    assert any(row["artifact_type"] in {"report", "report_blocked"} for row in artifacts)
    assert any(loads(row["metadata_json"], {}).get("tool_name") == "search.web" for row in events)
    assert len(store.list_citations(run_id)) >= 1
    assert objective_summary["objective_state"] == "delivered"
    assert objective_summary["evidence_workbench"]["formal_evidence_count"] >= 1
