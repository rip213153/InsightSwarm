from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from insightswarm.browser_interaction import approve_browser_action, list_browser_approvals, reject_browser_action
from insightswarm.agent_collaboration import build_agent_collaboration_view
from insightswarm.browser_authorization import decide_browser_authorization, list_browser_authorizations, respond_assisted_observation
from insightswarm.browser_handoff import candidate_from_artifact, promote_candidate_to_raw_document
from insightswarm.browser_planning import run_browser_operation
from insightswarm.tools.core import ToolContext
from insightswarm.tools.executor import ToolExecutor
from insightswarm.collector.gateway import serve_collector_gateway
from insightswarm.collector.ingest import ingest_collector_payload
from insightswarm.config import load_settings
from insightswarm.candidate_continuation import continue_candidate_sources
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.evidence_convergence import converge_evidence
from insightswarm.followup_planning import plan_followups, spawn_followup
from insightswarm.followup_round import continue_followup_round
from insightswarm.agents.extractor import ExtractorAgent
from insightswarm.harness.runner import Runner
from insightswarm.models.router import build_audited_model_client
from insightswarm.observability.trace import ensure_collaboration_trace
from insightswarm.observability.diagnosis import build_run_diagnosis, render_diagnosis_text
from insightswarm.graph_governance import build_graph_governance_projection
from insightswarm.graph_executor import GraphGovernedExecutor
from insightswarm.governance_runtime import LeadAgentGovernanceRuntime
from insightswarm.objective_runtime import ObjectiveDrivenSwarmRuntime, create_and_run_objective
from insightswarm.multiagent_runtime import build_multiagent_runtime_projection
from insightswarm.repair_continuation import continue_repair
from insightswarm.research_graph import (
    build_research_graph,
    build_research_graph_frontiers,
    build_research_graph_plan,
    build_research_graph_validation,
    write_research_graph_artifact,
)
from insightswarm.research_runtime_protocol import build_research_runtime_protocol
from insightswarm.runtime_kernel import MultiAgentRuntimeKernel
from insightswarm.runtime_workspace import RuntimeWorkspace
from insightswarm.source_acquisition_gateway import normalize_source
from insightswarm.swarm_runtime import SwarmRuntime, build_swarm_runtime_projection
from insightswarm.subagent_promotion import promote_finding_sources
from insightswarm.subagent_runtime import spawn_subagent
from insightswarm.writer_continuation import continue_writer
from insightswarm.observability.inspect import (
    inspect_artifact,
    inspect_citation,
    inspect_run,
    inspect_task,
    tail_events,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="insightswarm")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--model-provider", default=None)
    parser.add_argument("--config-path", default=None)
    sub = parser.add_subparsers(dest="resource", required=True)

    run = sub.add_parser("run")
    run_sub = run.add_subparsers(dest="action", required=True)
    run_create = run_sub.add_parser("create")
    run_create.add_argument("--name", default="demo")
    run_create.add_argument("--query", default=None)
    run_create.add_argument("--quality-mode", default="production", choices=["production", "test"])
    run_create.add_argument("--competitor", default="Acme Analytics")
    run_create.add_argument("--source-url", action="append", default=[])
    run_create.add_argument("--source-text-file", default=None)
    run_create.add_argument("--source-pdf-text-file", default=None)
    run_create.add_argument("--screenshot-file", default=None)
    run_create.add_argument("--search-provider", default=None)
    run_create.add_argument("--search-limit", type=int, default=10)
    run_create.add_argument("--link-gate-max-selected", type=int, default=5)
    run_create.add_argument("--browser-allowed-domain", action="append", default=[])
    run_create.add_argument("--browser-authorized-domain", action="append", default=[])
    run_create.add_argument("--browser-assisted-observation-allowed", action="store_true")
    run_create.add_argument("--browser-max-authorization-requests", type=int, default=5)
    run_create.add_argument("--browser-source-target-url", default=None)
    run_create.add_argument("--browser-backend", default=None, choices=["fake", "cdp"])
    run_create.add_argument("--browser-cdp-url", default=None)
    run_create.add_argument("--max-subagents-per-run", type=int, default=3)
    run_create.add_argument("--max-spawn-depth", type=int, default=1)
    run_create.add_argument("--max-parallel-subagents", type=int, default=1)
    run_create.add_argument("--max-context-tokens-per-subagent", type=int, default=2000)
    run_create.add_argument("--allowed-subagent-role", action="append", default=[])
    run_ask = run_sub.add_parser("ask")
    run_ask.add_argument("--query", required=True)
    run_ask.add_argument("--name", default="objective-intelligence")
    run_ask.add_argument("--max-steps", type=int, default=12)
    run_ask.add_argument("--allow-delivery", action="store_true", help="Deprecated for run ask; delivery is enabled by default in Phase 48.")
    run_ask.add_argument("--quality-mode", default="production", choices=["production", "test"])
    run_ask.add_argument("--search-provider", default="tavily", choices=["tavily", "static"])
    run_ask.add_argument("--browser-backend", default=None, choices=["fake", "cdp"])
    run_ask.add_argument("--browser-cdp-url", default=None)
    run_ask.add_argument("--json", action="store_true")
    run_start = run_sub.add_parser("start")
    run_start.add_argument("--run-id", required=True)
    run_start.add_argument("--max-steps", type=int, default=12)
    run_start.add_argument("--allow-delivery", action="store_true")
    run_start.add_argument("--legacy-dag", action="store_true")
    run_start.add_argument("--json", action="store_true")
    run_start_legacy = run_sub.add_parser("start-legacy")
    run_start_legacy.add_argument("--run-id", required=True)
    run_inspect = run_sub.add_parser("inspect")
    run_inspect.add_argument("--run-id", required=True)
    run_diagnose = run_sub.add_parser("diagnose")
    run_diagnose.add_argument("--run-id", required=True)
    run_diagnose.add_argument("--json", action="store_true")
    run_graph = run_sub.add_parser("graph")
    run_graph.add_argument("--run-id", required=True)
    run_graph.add_argument("--json", action="store_true")
    run_graph.add_argument("--write-artifact", action="store_true")
    run_graph.add_argument("--validate", action="store_true")
    run_graph.add_argument("--frontiers", action="store_true")
    run_graph.add_argument("--plan", action="store_true")
    run_graph.add_argument("--protocol", action="store_true")
    run_graph.add_argument("--runtime", action="store_true")
    run_graph.add_argument("--governance", action="store_true")
    run_runtime = run_sub.add_parser("runtime")
    run_runtime.add_argument("--run-id", required=True)
    run_runtime.add_argument("--json", action="store_true")
    run_runtime_step = run_sub.add_parser("runtime-step")
    run_runtime_step.add_argument("--run-id", required=True)
    run_runtime_step.add_argument("--agent", default=None)
    run_collaborate = run_sub.add_parser("collaborate")
    run_collaborate.add_argument("--run-id", required=True)
    run_collaborate.add_argument("--apply", action="store_true")
    run_collaborate.add_argument("--json", action="store_true")
    run_govern = run_sub.add_parser("govern")
    run_govern.add_argument("--run-id", required=True)
    run_govern.add_argument("--max-steps", type=int, default=5)
    run_govern.add_argument("--allow-delivery", action="store_true")
    run_govern.add_argument("--json", action="store_true")
    run_workspace = run_sub.add_parser("workspace")
    run_workspace.add_argument("--run-id", required=True)
    run_workspace.add_argument("--json", action="store_true")
    run_execute_plan = run_sub.add_parser("execute-plan")
    run_execute_plan.add_argument("--run-id", required=True)
    run_execute_plan.add_argument("--step-id", default=None)
    run_execute_plan.add_argument("--kind", default=None, choices=["resume_plan", "rollback_plan", "branch_plan", "human_gate_plan"])
    run_swarm = run_sub.add_parser("swarm")
    run_swarm.add_argument("--run-id", required=True)
    run_swarm.add_argument("--json", action="store_true")
    run_swarm_step = run_sub.add_parser("swarm-step")
    run_swarm_step.add_argument("--run-id", required=True)
    run_swarm_step.add_argument("--work-order-id", default=None)
    run_swarm_step.add_argument("--agent", default=None)
    run_extract = run_sub.add_parser("extract")
    run_extract.add_argument("--run-id", required=True)
    run_extract.add_argument("--raw-document-id", required=True)
    run_spawn = run_sub.add_parser("spawn-subagent")
    run_spawn.add_argument("--run-id", required=True)
    run_spawn.add_argument("--parent-task-id", required=True)
    run_spawn.add_argument("--role", required=True)
    run_spawn.add_argument("--scope", required=True)
    run_spawn.add_argument("--spawn-reason", default=None)
    run_promote_finding = run_sub.add_parser("promote-finding")
    run_promote_finding.add_argument("--run-id", required=True)
    run_promote_finding.add_argument("--finding-id", default=None)
    run_promote_finding.add_argument("--handoff-id", default=None)
    run_plan_followups = run_sub.add_parser("plan-followups")
    run_plan_followups.add_argument("--run-id", required=True)
    run_spawn_followup = run_sub.add_parser("spawn-followup")
    run_spawn_followup.add_argument("--run-id", required=True)
    run_spawn_followup.add_argument("--plan-id", required=True)
    run_spawn_followup.add_argument("--item-id", required=True)
    run_continue_followup = run_sub.add_parser("continue-followup")
    run_continue_followup.add_argument("--run-id", required=True)
    run_continue_followup.add_argument("--decision-id", default=None)
    run_continue_followup.add_argument("--plan-id", default=None)
    run_continue_followup.add_argument("--item-id", default=None)
    run_continue_candidate = run_sub.add_parser("continue-candidate")
    run_continue_candidate.add_argument("--run-id", required=True)
    run_continue_candidate.add_argument("--candidate-id", default=None)
    run_continue_candidate.add_argument("--round-id", default=None)
    run_normalize_source = run_sub.add_parser("normalize-source")
    run_normalize_source.add_argument("--run-id", required=True)
    run_normalize_source.add_argument("--artifact-id", default=None)
    run_normalize_source.add_argument("--work-order-id", default=None)
    run_converge_evidence = run_sub.add_parser("converge-evidence")
    run_converge_evidence.add_argument("--run-id", required=True)
    run_converge_evidence.add_argument("--review-qa-id", required=True)
    run_continue_repair = run_sub.add_parser("continue-repair")
    run_continue_repair.add_argument("--run-id", required=True)
    run_continue_repair.add_argument("--review-qa-id", default=None)
    run_continue_repair.add_argument("--qa-report-id", default=None)
    run_continue_writer = run_sub.add_parser("continue-writer")
    run_continue_writer.add_argument("--run-id", required=True)
    run_continue_writer.add_argument("--review-qa-id", default=None)
    run_continue_writer.add_argument("--qa-report-id", default=None)
    run_continue_writer.add_argument("--repair-id", default=None)

    task = sub.add_parser("task")
    task_sub = task.add_subparsers(dest="action", required=True)
    task_inspect = task_sub.add_parser("inspect")
    task_inspect.add_argument("--task-id", required=True)

    artifact = sub.add_parser("artifact")
    artifact_sub = artifact.add_subparsers(dest="action", required=True)
    artifact_inspect = artifact_sub.add_parser("inspect")
    artifact_inspect.add_argument("--artifact-id", required=True)

    citation = sub.add_parser("citation")
    citation_sub = citation.add_subparsers(dest="action", required=True)
    citation_inspect = citation_sub.add_parser("inspect")
    citation_inspect.add_argument("--citation-id", required=True)

    events = sub.add_parser("events")
    events_sub = events.add_subparsers(dest="action", required=True)
    events_tail = events_sub.add_parser("tail")
    events_tail.add_argument("--run-id", required=True)
    events_tail.add_argument("--limit", type=int, default=20)

    collector = sub.add_parser("collector")
    collector_sub = collector.add_subparsers(dest="action", required=True)
    collector_serve = collector_sub.add_parser("serve")
    collector_serve.add_argument("--run-id", required=True)
    collector_serve.add_argument("--port", type=int, required=True)
    collector_ingest = collector_sub.add_parser("ingest")
    collector_ingest.add_argument("--run-id", required=True)
    collector_ingest.add_argument("--payload-file", required=True)

    browser = sub.add_parser("browser")
    browser_sub = browser.add_subparsers(dest="action", required=True)
    browser_approvals = browser_sub.add_parser("approvals")
    browser_approvals.add_argument("--run-id", required=True)
    browser_authorizations = browser_sub.add_parser("authorizations")
    browser_authorizations.add_argument("--run-id", required=True)
    browser_authorize = browser_sub.add_parser("authorize")
    browser_authorize.add_argument("--run-id", required=True)
    browser_authorize.add_argument("--request-id", required=True)
    browser_authorize.add_argument("--decision", required=True, choices=["approve", "reject"])
    browser_observe = browser_sub.add_parser("observe")
    browser_observe.add_argument("--run-id", required=True)
    browser_observe.add_argument("--request-id", required=True)
    browser_observe.add_argument("--value", required=True)
    browser_approve = browser_sub.add_parser("approve")
    browser_approve.add_argument("--run-id", required=True)
    browser_approve.add_argument("--request-id", required=True)
    browser_approve.add_argument("--execute", action="store_true")
    browser_approve.add_argument("--backend", default="fake", choices=["fake", "cdp"])
    browser_approve.add_argument("--cdp-url", default=None)
    browser_approve.add_argument("--quality-mode", default="production")
    browser_reject = browser_sub.add_parser("reject")
    browser_reject.add_argument("--run-id", required=True)
    browser_reject.add_argument("--request-id", required=True)
    browser_reject.add_argument("--reason", default=None)
    browser_run = browser_sub.add_parser("run")
    browser_run.add_argument("--run-id", required=True)
    browser_run.add_argument("--mode", required=True, choices=["assisted", "free_browser"])
    browser_run.add_argument("--goal", required=True)
    browser_run.add_argument("--backend", default="fake", choices=["fake", "cdp"])
    browser_run.add_argument("--cdp-url", default=None)
    browser_run.add_argument("--target-url", default=None)
    browser_run.add_argument("--vision-required", action="store_true")
    browser_run.add_argument("--max-iterations", type=int, default=5)
    browser_promote = browser_sub.add_parser("promote")
    browser_promote.add_argument("--run-id", required=True)
    browser_promote.add_argument("--candidate-id", default=None)
    browser_promote.add_argument("--source-artifact-id", default=None)
    browser_promote.add_argument("--quality-mode", default="production")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(
        args.db_path,
        args.artifact_dir,
        args.model_provider,
        args.config_path,
    )
    init_db(settings.db_path)
    store = Store(settings.db_path, settings.artifact_dir)

    if args.resource == "run" and args.action == "create":
        metadata = {
            "query": args.query,
            "quality_mode": args.quality_mode,
            "competitor": args.competitor,
            "source_urls": args.source_url,
            "source_text_file": args.source_text_file,
            "source_pdf_text_file": args.source_pdf_text_file,
            "screenshot_file": args.screenshot_file,
            "model_provider": settings.model_provider,
            "search_provider": args.search_provider,
            "search_limit": args.search_limit,
            "link_gate_max_selected": args.link_gate_max_selected,
            "browser_allowed_domains": args.browser_allowed_domain,
            "browser_authorized_domains": args.browser_authorized_domain,
            "browser_assisted_observation_allowed": args.browser_assisted_observation_allowed,
            "browser_max_authorization_requests": args.browser_max_authorization_requests,
            "browser_source_target_url": args.browser_source_target_url,
            "browser_backend": args.browser_backend,
            "browser_cdp_url": args.browser_cdp_url,
            "max_subagents_per_run": args.max_subagents_per_run,
            "max_spawn_depth": args.max_spawn_depth,
            "max_parallel_subagents": args.max_parallel_subagents,
            "max_context_tokens_per_subagent": args.max_context_tokens_per_subagent,
            "allowed_subagent_roles": args.allowed_subagent_role or ["SearchAgent", "SkepticReviewAgent"],
        }
        run_id = store.create_run(args.name, metadata)
        print(run_id)
        return 0
    if args.resource == "run" and args.action == "ask":
        result = create_and_run_objective(
            store,
            name=args.name,
            query=args.query,
            model_provider=settings.model_provider,
            artifact_dir=settings.artifact_dir,
            max_steps=args.max_steps,
            allow_delivery=True,
            quality_mode=args.quality_mode,
            search_provider=args.search_provider,
            browser_backend=args.browser_backend,
            browser_cdp_url=args.browser_cdp_url or os.getenv("INSIGHTSWARM_CDP_URL"),
        )
        try:
            ensure_collaboration_trace(store, result.run_id)
        except Exception as exc:
            store.emit_event(
                result.run_id,
                None,
                "Harness",
                "collaboration_trace_failed",
                "Failed to refresh collaboration trace after objective run.",
                {"error": str(exc)},
            )
        payload = result.to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        else:
            print(
                "\n".join(
                    [
                        "Objective Runtime",
                        f"- Run ID: {payload['run_id']}",
                        f"- Status: {payload['status']}",
                        f"- Final state: {payload['final_state']}",
                        f"- Stop reason: {payload['stop_reason']}",
                    ]
                )
            )
        return 0 if result.final_state not in {"blocked"} else 2
    if args.resource == "run" and args.action == "start":
        if args.legacy_dag:
            Runner(store, settings.model_provider).start(args.run_id)
            print(inspect_run(store, args.run_id))
            return 0
        result = ObjectiveDrivenSwarmRuntime(
            store,
            args.run_id,
            workspace_root=settings.artifact_dir.parent / "runtime",
            model_provider=settings.model_provider,
            allow_delivery=args.allow_delivery,
        ).run(max_steps=args.max_steps)
        try:
            ensure_collaboration_trace(store, args.run_id)
        except Exception as exc:
            store.emit_event(args.run_id, None, "Harness", "collaboration_trace_failed", "Failed to refresh collaboration trace after objective start.", {"error": str(exc)})
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=True, indent=2, sort_keys=True))
        else:
            print(inspect_run(store, args.run_id))
        return 0 if result.final_state not in {"blocked"} else 2
    if args.resource == "run" and args.action == "start-legacy":
        Runner(store, settings.model_provider).start(args.run_id)
        print(inspect_run(store, args.run_id))
        return 0
    if args.resource == "run" and args.action == "inspect":
        print(inspect_run(store, args.run_id))
        return 0
    if args.resource == "run" and args.action == "diagnose":
        diagnosis = build_run_diagnosis(store, args.run_id)
        if args.json:
            print(json.dumps(diagnosis, ensure_ascii=True, indent=2))
        else:
            print(render_diagnosis_text(diagnosis))
        return 0
    if args.resource == "run" and args.action == "runtime":
        payload = build_multiagent_runtime_projection(store, args.run_id)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        else:
            summary = payload["summary"]
            lines = [
                "Multi-Agent Runtime",
                f"- Run: {args.run_id}",
                f"- Agent identities: {summary['agent_identity_count']}",
                f"- Task board items: {summary['task_board_item_count']}",
                f"- Claimable/leased/repairable: {summary['claimable_task_count']}/{summary['leased_task_count']}/{summary['repairable_task_count']}",
                f"- Mailbox messages: {summary['mailbox_message_count']}",
                f"- Open mailbox messages: {summary['open_mailbox_message_count']}",
                f"- Policy gates: {summary['policy_gate_count']}",
                f"- Active policy gates: {summary['active_policy_gate_count']}",
            ]
            print("\n".join(lines))
        return 0
    if args.resource == "run" and args.action == "runtime-step":
        runner = Runner(store, settings.model_provider)
        if not store.list_tasks(args.run_id):
            from insightswarm.harness.runner import build_fake_dag

            build_fake_dag(store, args.run_id)
        runner._expand_dynamic_extract_tasks(args.run_id)
        result = MultiAgentRuntimeKernel(store, runner.model, runner._build_agent).execute_next(args.run_id, args.agent)
        try:
            ensure_collaboration_trace(store, args.run_id)
        except Exception as exc:
            store.emit_event(
                args.run_id,
                result.task_id,
                "Harness",
                "collaboration_trace_failed",
                "Failed to refresh collaboration trace after runtime step.",
                {"error": str(exc)},
            )
        print(json.dumps(result.to_dict(), ensure_ascii=True, indent=2, sort_keys=True))
        return 0 if result.status in {"executed", "idle"} else 2
    if args.resource == "run" and args.action == "collaborate":
        payload = build_agent_collaboration_view(
            store,
            args.run_id,
            apply=args.apply,
            workspace_root=str(settings.artifact_dir.parent / "runtime"),
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        else:
            summary = payload["summary"]
            lines = [
                "Agent Collaboration Kernel",
                f"- Run: {args.run_id}",
                f"- Translations: {summary['translation_count']}",
                f"- Pending intents: {summary['pending_intent_count']}",
                f"- Null-intent legacy messages: {summary['null_intent_count']}",
                f"- Tool contracts: {summary['tool_contract_count']}",
                f"- Source acquisition frontiers: {summary['source_acquisition_frontier_count']}",
                f"- Applied: {summary['applied']}",
            ]
            print("\n".join(lines))
        return 0
    if args.resource == "run" and args.action == "govern":
        _warn_legacy_command("run govern is legacy/internal after Phase 48; prefer run ask for product execution.")
        result = LeadAgentGovernanceRuntime(
            store,
            args.run_id,
            workspace_root=settings.artifact_dir.parent / "runtime",
            model_provider=settings.model_provider,
            allow_delivery=args.allow_delivery,
        ).govern(max_steps=args.max_steps)
        try:
            ensure_collaboration_trace(store, args.run_id)
        except Exception as exc:
            store.emit_event(
                args.run_id,
                None,
                "Harness",
                "collaboration_trace_failed",
                "Failed to refresh collaboration trace after governance run.",
                {"error": str(exc)},
            )
        payload = result.to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        else:
            lines = [
                "LeadAgent Governance",
                f"- Run: {args.run_id}",
                f"- Status: {payload['status']}",
                f"- Steps: {len(payload['decision_artifact_ids'])}",
                f"- Allow delivery: {payload['allow_delivery']}",
                f"- Stop reason: {payload['stop_reason']}",
                f"- Latest decision: {payload['decision_artifact_ids'][-1] if payload['decision_artifact_ids'] else None}",
            ]
            print("\n".join(lines))
        return 0 if result.status in {"governed", "idle"} else 2
    if args.resource == "run" and args.action == "workspace":
        payload = RuntimeWorkspace(settings.artifact_dir.parent / "runtime", args.run_id).summary()
        if args.json:
            print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        else:
            summary = payload["summary"]
            lines = [
                "Runtime Workspace",
                f"- Run: {args.run_id}",
                f"- Path: {payload['workspace_path']}",
                f"- Work orders: {summary['work_order_count']}",
                f"- Branches: {summary['branch_count']}",
                f"- Rollbacks: {summary['rollback_count']}",
                f"- Pending authorizations: {summary['authorization_pending_count']}",
                f"- Arbitrations: {summary['arbitration_count']}",
                f"- Swarm assignments: {summary['swarm_assignment_count']}",
                f"- Swarm handoffs: {summary['swarm_handoff_count']}",
                f"- Browser swarm operations: {summary['browser_swarm_operation_count']}",
                f"- Latest snapshot: {summary['latest_workspace_snapshot_id']}",
            ]
            print("\n".join(lines))
        return 0
    if args.resource == "run" and args.action == "execute-plan":
        _warn_legacy_command("run execute-plan is legacy/internal after Phase 48; prefer run ask for product execution.")
        executor = GraphGovernedExecutor(
            store,
            args.run_id,
            workspace_root=settings.artifact_dir.parent / "runtime",
            model_provider=settings.model_provider,
        )
        result = executor.execute(step_id=args.step_id, kind=args.kind)
        try:
            ensure_collaboration_trace(store, args.run_id)
        except Exception as exc:
            store.emit_event(
                args.run_id,
                None,
                "Harness",
                "collaboration_trace_failed",
                "Failed to refresh collaboration trace after graph-governed execution.",
                {"error": str(exc)},
            )
        print(json.dumps(result.to_dict(), ensure_ascii=True, indent=2, sort_keys=True))
        return 0 if result.status not in {"blocked_no_plan_step", "blocked_unsupported_plan_kind", "blocked_unsupported_resume", "blocked_human_gate"} else 2
    if args.resource == "run" and args.action == "swarm":
        payload = build_swarm_runtime_projection(store, args.run_id, workspace_root=settings.artifact_dir.parent / "runtime")
        if args.json:
            print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        else:
            summary = payload["summary"]
            lines = [
                "Swarm Runtime",
                f"- Run: {args.run_id}",
                f"- Open work orders: {summary['open_work_order_count']}",
                f"- Assignments: {summary['assignment_count']}",
                f"- Claimable assignments: {summary['claimable_assignment_count']}",
                f"- Handoffs: {summary['handoff_count']}",
                f"- Browser operations: {summary['browser_swarm_operation_count']}",
                f"- Policy blocks: {summary['policy_block_count']}",
                f"- Subagent capacity remaining: {summary['subagent_capacity_remaining']}",
            ]
            print("\n".join(lines))
        return 0
    if args.resource == "run" and args.action == "swarm-step":
        _warn_legacy_command("run swarm-step is legacy/internal after Phase 48; Browser escalation should be Lead-selected inside run ask.")
        result = SwarmRuntime(store, args.run_id, workspace_root=settings.artifact_dir.parent / "runtime").step(
            work_order_id=args.work_order_id,
            agent_name=args.agent,
        )
        try:
            ensure_collaboration_trace(store, args.run_id)
        except Exception as exc:
            store.emit_event(
                args.run_id,
                result.task_id,
                "Harness",
                "collaboration_trace_failed",
                "Failed to refresh collaboration trace after swarm step.",
                {"error": str(exc)},
            )
        print(json.dumps(result.to_dict(), ensure_ascii=True, indent=2, sort_keys=True))
        return 0 if result.status not in {"blocked_browser_policy", "blocked_subagent_policy"} else 2
    if args.resource == "run" and args.action == "graph":
        graph = build_research_graph(store, args.run_id)
        payload = graph.to_dict()
        if args.validate:
            payload["validation"] = build_research_graph_validation(store, args.run_id)
        if args.frontiers:
            payload["frontiers"] = build_research_graph_frontiers(store, args.run_id)
        if args.plan:
            payload["plan"] = build_research_graph_plan(store, args.run_id)
        if args.protocol:
            payload["protocol"] = build_research_runtime_protocol()
        if args.runtime:
            payload["multiagent_runtime"] = build_multiagent_runtime_projection(store, args.run_id)
        if args.governance:
            payload["graph_governance"] = build_graph_governance_projection(store, args.run_id)
        if args.write_artifact:
            payload["artifact_id"] = write_research_graph_artifact(store, args.run_id, graph)
        if args.json or args.write_artifact:
            print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        else:
            summary = payload["summary"]
            lines = [
                "Research Graph",
                f"- Run: {args.run_id}",
                f"- Nodes: {summary['node_count']}",
                f"- Edges: {summary['edge_count']}",
                f"- Continuations: {summary['continuation_chain_count']}",
                f"- Formal evidence references: {summary['evidence_count']}",
                f"- Delivery nodes: {summary['delivery_count']}",
                f"- Collaboration nodes: {summary['collaboration_node_count']}",
                "- Edge direction: upstream/cause/source -> downstream/effect/consumer/result",
            ]
            if args.validate:
                validation = payload["validation"]["summary"]
                lines.extend(
                    [
                        "",
                        "Research Graph Validation",
                        f"- Findings: {validation['finding_count']}",
                        f"- Errors: {validation['error_count']}",
                        f"- Warnings: {validation['warning_count']}",
                        f"- Info: {validation['info_count']}",
                    ]
                )
            if args.frontiers:
                frontiers = payload["frontiers"]["summary"]
                lines.extend(
                    [
                        "",
                        "Research Graph Frontiers",
                        f"- Frontiers: {frontiers['frontier_count']}",
                        f"- Resumable: {frontiers['resumable_count']}",
                        f"- Needs human: {frontiers['human_intervention_count']}",
                        f"- Blocked: {frontiers['blocked_count']}",
                    ]
                )
            if args.plan:
                plan = payload["plan"]["summary"]
                lines.extend(
                    [
                        "",
                        "Research Graph Plan",
                        f"- Status: {plan['status']}",
                        f"- Steps: {plan['step_count']}",
                        f"- Resume/Rollback/Branch/Human: {plan['resume_step_count']}/{plan['rollback_step_count']}/{plan['branch_step_count']}/{plan['human_gate_step_count']}",
                        f"- Top action: {plan['top_action']}",
                        f"- Top command: {plan['top_recommended_command']}",
                    ]
                )
            if args.protocol:
                protocol = payload["protocol"]
                lines.extend(
                    [
                        "",
                        "Research Runtime Protocol",
                        f"- Schema: {protocol['schema']}",
                        f"- Continuation kinds: {len(protocol['continuation_kinds'])}",
                        f"- Node/edge kinds: {len(protocol['node_kinds'])}/{len(protocol['edge_kinds'])}",
                        f"- Boundary count: {len(protocol['boundaries'])}",
                    ]
                )
            if args.runtime:
                runtime = payload["multiagent_runtime"]["summary"]
                lines.extend(
                    [
                        "",
                        "Multi-Agent Runtime",
                        f"- Agent identities: {runtime['agent_identity_count']}",
                        f"- Task board items: {runtime['task_board_item_count']}",
                        f"- Mailbox messages: {runtime['mailbox_message_count']}",
                        f"- Active policy gates: {runtime['active_policy_gate_count']}",
                    ]
                )
            if args.governance:
                governance = payload["graph_governance"]["summary"]
                lines.extend(
                    [
                        "",
                        "Graph Governance",
                        f"- Capabilities: {governance['capability_count']}",
                        f"- Ready/blocked/human: {governance['ready_count']}/{governance['blocked_count']}/{governance['needs_human_count']}",
                        f"- By phase: {governance['by_phase']}",
                    ]
                )
            print("\n".join(lines))
        return 0
    if args.resource == "run" and args.action == "extract":
        raw_document = store.get_artifact(args.raw_document_id)
        if raw_document["run_id"] != args.run_id or raw_document["artifact_type"] != "raw_document":
            raise SystemExit("--raw-document-id must reference a raw_document artifact in the run")
        raw_metadata = json.loads(raw_document["metadata_json"] or "{}")
        task_id = store.create_task(
            args.run_id,
            "Extract",
            "ExtractorAgent",
            metadata={
                "raw_document_id": args.raw_document_id,
                "manual_evidence_handoff_continuation": True,
                "source_url": raw_document["source_url"],
                "source_fetcher": raw_metadata.get("fetcher"),
            },
        )
        ExtractorAgent(store, build_audited_model_client(settings.model_provider, store)).execute(args.run_id, task_id)
        try:
            ensure_collaboration_trace(store, args.run_id)
        except Exception as exc:
            store.emit_event(
                args.run_id,
                task_id,
                "Harness",
                "collaboration_trace_failed",
                "Failed to refresh collaboration trace after manual extract.",
                {"error": str(exc)},
            )
        task = store.get_task(task_id)
        task_metadata = json.loads(task["metadata_json"] or "{}")
        print(
            json.dumps(
                {
                    "status": task["status"],
                    "task_id": task_id,
                    "raw_document_artifact_id": args.raw_document_id,
                    "structured_knowledge_artifact_id": task_metadata.get("artifact_id"),
                    "citation_ids": task_metadata.get("citation_ids", []),
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return 0
    if args.resource == "run" and args.action == "spawn-subagent":
        result = spawn_subagent(
            store,
            args.run_id,
            args.parent_task_id,
            args.role,
            args.scope,
            requester="CLI",
            spawn_reason=args.spawn_reason,
        )
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["status"] == "created" else 2
    if args.resource == "run" and args.action == "promote-finding":
        result = promote_finding_sources(
            store,
            args.run_id,
            finding_id=args.finding_id,
            handoff_id=args.handoff_id,
        )
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["status"] == "promoted" else 2
    if args.resource == "run" and args.action == "plan-followups":
        _warn_legacy_command("run plan-followups is legacy/internal after Phase 48; Critic gaps should be consumed by run ask.")
        print(json.dumps(plan_followups(store, args.run_id), ensure_ascii=True, indent=2))
        return 0
    if args.resource == "run" and args.action == "spawn-followup":
        _warn_legacy_command("run spawn-followup is legacy/internal after Phase 48; Critic gaps should be consumed by run ask.")
        result = spawn_followup(store, args.run_id, args.plan_id, args.item_id)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["status"] == "spawned" else 2
    if args.resource == "run" and args.action == "continue-followup":
        _warn_legacy_command("run continue-followup is legacy/internal after Phase 48; use run ask for autonomous continuation.")
        result = continue_followup_round(
            store,
            args.run_id,
            decision_id=args.decision_id,
            plan_id=args.plan_id,
            item_id=args.item_id,
            model_provider=settings.model_provider,
        )
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["status"] == "candidate_ready" else 2
    if args.resource == "run" and args.action == "continue-candidate":
        _warn_legacy_command("run continue-candidate is legacy/internal after Phase 48; use run ask for autonomous continuation.")
        result = continue_candidate_sources(
            store,
            args.run_id,
            candidate_id=args.candidate_id,
            round_id=args.round_id,
            model_provider=settings.model_provider,
        )
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["status"] == "citation_ready" else 2
    if args.resource == "run" and args.action == "normalize-source":
        if not args.artifact_id and not args.work_order_id:
            raise SystemExit("--artifact-id or --work-order-id is required")
        result = normalize_source(store, args.run_id, artifact_id=args.artifact_id, work_order_id=args.work_order_id)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["status"] == "candidate_ready" else 2
    if args.resource == "run" and args.action == "converge-evidence":
        result = converge_evidence(store, args.run_id, review_qa_id=args.review_qa_id)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["status"] in {"retain", "merge", "supersede", "discard"} else 2
    if args.resource == "run" and args.action == "continue-repair":
        _warn_legacy_command("run continue-repair is legacy/internal after Phase 48; use run ask for autonomous Critic repair.")
        result = continue_repair(
            store,
            args.run_id,
            review_qa_id=args.review_qa_id,
            qa_report_id=args.qa_report_id,
            model_provider=settings.model_provider,
        )
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["status"] == "repair_ready" else 2
    if args.resource == "run" and args.action == "continue-writer":
        _warn_legacy_command("run continue-writer is legacy/internal after Phase 48; run ask now delivers by default.")
        try:
            result = continue_writer(
                store,
                args.run_id,
                review_qa_id=args.review_qa_id,
                qa_report_id=args.qa_report_id,
                repair_id=args.repair_id,
                model_provider=settings.model_provider,
            )
        except ValueError as exc:
            result = {"status": "blocked_legacy_command_error", "error": str(exc), "legacy_internal": True}
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["status"] in {"report_ready", "blocked_delivery_ready"} else 2
    if args.resource == "task" and args.action == "inspect":
        print(inspect_task(store, args.task_id))
        return 0
    if args.resource == "artifact" and args.action == "inspect":
        print(inspect_artifact(store, args.artifact_id))
        return 0
    if args.resource == "citation" and args.action == "inspect":
        print(inspect_citation(store, args.citation_id))
        return 0
    if args.resource == "events" and args.action == "tail":
        print(tail_events(store, args.run_id, args.limit))
        return 0
    if args.resource == "collector" and args.action == "serve":
        serve_collector_gateway(store, args.run_id, args.port)
        return 0
    if args.resource == "collector" and args.action == "ingest":
        payload = json.loads(Path(args.payload_file).read_text(encoding="utf-8"))
        result = ingest_collector_payload(store, args.run_id, payload)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    if args.resource == "browser" and args.action == "approvals":
        print(json.dumps(list_browser_approvals(store, args.run_id), ensure_ascii=True, indent=2))
        return 0
    if args.resource == "browser" and args.action == "authorizations":
        print(json.dumps(list_browser_authorizations(store, args.run_id), ensure_ascii=True, indent=2))
        return 0
    if args.resource == "browser" and args.action == "authorize":
        print(json.dumps(decide_browser_authorization(store, args.run_id, args.request_id, args.decision), ensure_ascii=True, indent=2))
        return 0
    if args.resource == "browser" and args.action == "observe":
        print(json.dumps(respond_assisted_observation(store, args.run_id, args.request_id, args.value), ensure_ascii=True, indent=2))
        return 0
    if args.resource == "browser" and args.action == "approve":
        result = approve_browser_action(
            store,
            args.run_id,
            args.request_id,
            execute=args.execute,
            backend=args.backend,
            cdp_url=args.cdp_url or os.getenv("INSIGHTSWARM_CDP_URL"),
            quality_mode=args.quality_mode,
        )
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    if args.resource == "browser" and args.action == "reject":
        print(json.dumps(reject_browser_action(store, args.run_id, args.request_id, args.reason), ensure_ascii=True, indent=2))
        return 0
    if args.resource == "browser" and args.action == "run":
        result = run_browser_operation(
            store,
            args.run_id,
            mode=args.mode,
            goal=args.goal,
            backend=args.backend,
            cdp_url=args.cdp_url or os.getenv("INSIGHTSWARM_CDP_URL"),
            target_url=args.target_url,
            vision_required=args.vision_required,
            model_provider=settings.model_provider,
            max_iterations=args.max_iterations,
        )
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    if args.resource == "browser" and args.action == "promote":
        if args.source_artifact_id:
            task_id = None
            try:
                source_row = store.get_artifact(args.source_artifact_id)
                task_id = source_row["task_id"]
            except KeyError:
                task_id = None
            source_input = candidate_from_artifact(store, args.source_artifact_id)
            result, _ = ToolExecutor(store).run(
                "browser.promote_source",
                source_input,
                ToolContext(args.run_id, task_id, args.quality_mode, {"agent_name": "BrowserAgent", "browser_mode": "free_browser"}),
            )
            print(json.dumps(result.to_dict(), ensure_ascii=True, indent=2))
            return 0
        if not args.candidate_id:
            raise SystemExit("--candidate-id or --source-artifact-id is required")
        print(json.dumps(promote_candidate_to_raw_document(store, args.run_id, args.candidate_id, quality_mode=args.quality_mode), ensure_ascii=True, indent=2))
        return 0
    raise AssertionError("unreachable")


def _warn_legacy_command(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
