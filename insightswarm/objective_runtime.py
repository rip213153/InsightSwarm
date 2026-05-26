from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from insightswarm.agents.search import SearchAgent
from insightswarm.browser_handoff import promote_candidate_to_raw_document
from insightswarm.db.store import Store
from insightswarm.governance_runtime import AgentTemplateRegistry
from insightswarm.models.router import build_audited_model_client
from insightswarm.runtime_workspace import RuntimeWorkspace
from insightswarm.source_acquisition_gateway import normalize_source, pending_gateway_sources
from insightswarm.tool_contracts import build_tool_contract_registry
from insightswarm.util import loads


OBJECTIVE_STATES = {
    "initialized",
    "planning",
    "acquiring_sources",
    "extracting_evidence",
    "analyzing",
    "reviewing",
    "converging",
    "delivery_ready",
    "delivered",
    "blocked",
    "needs_human",
    "exhausted",
}
WORK_ORDER_STATES = {"pending", "claimable", "claimed", "in_progress", "completed", "blocked", "superseded", "abandoned"}
EVIDENCE_STATES = {
    "advisory_source",
    "candidate_research_source",
    "raw_document",
    "quote_candidate",
    "citation",
    "qa_accepted_evidence",
    "delivery_evidence",
}
STOP_REASONS = {
    "delivered",
    "delivery_requested",
    "blocked_no_verifiable_source",
    "blocked_no_citation",
    "exhausted_search_budget",
    "exhausted_browser_budget",
    "exhausted_extractor_repair_budget",
    "exhausted_no_citation_after_source_acquisition",
    "needs_human_authorization",
    "qa_failed_after_repair_limit",
    "max_steps_reached",
    "no_progress_exhausted",
}
ARBITER_ACTIONS = {
    "search_sources",
    "browse_sources",
    "normalize_source",
    "continue_candidate",
    "repair_extract",
    "advance_pipeline",
    "continue_repair",
    "analyze_evidence",
    "converge_evidence",
    "request_delivery",
    "execute_delivery",
    "deliver_partial",
    "deliver_blocked",
    "ask_human",
    "arbitrate",
    "stop",
}


@dataclass(frozen=True)
class ObjectiveBudget:
    max_search_calls: int = 3
    max_browser_operations: int = 2
    max_extractor_repairs: int = 2
    max_subagents: int = 2
    max_no_progress_steps: int = 3
    max_model_tool_budget: int = 24

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntelligenceObjective:
    run_id: str
    query: str
    objective_id: str
    status: str
    key_questions: list[str]
    success_criteria: list[str]
    evidence_requirements: list[str]
    freshness_requirement: str
    forbidden_actions: list[str]
    budget: ObjectiveBudget
    artifact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["budget"] = self.budget.to_dict()
        payload["schema"] = "intelligence_objective.v1"
        return payload


@dataclass(frozen=True)
class EvidenceWorkbench:
    advisory_source_ids: list[str] = field(default_factory=list)
    candidate_research_source_ids: list[str] = field(default_factory=list)
    raw_document_ids: list[str] = field(default_factory=list)
    structured_knowledge_ids: list[str] = field(default_factory=list)
    citation_ids: list[str] = field(default_factory=list)
    quote_candidate_count: int = 0
    discarded_fact_count: int = 0
    qa_passed: bool = False
    latest_review_qa_id: str | None = None
    latest_qa_report_id: str | None = None
    latest_convergence_decision_id: str | None = None
    delivery_request_id: str | None = None
    report_id: str | None = None
    report_blocked_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = "evidence_workbench.v1"
        payload["formal_evidence_count"] = len(self.citation_ids)
        payload["evidence_boundary"] = {
            "advisory_sources_are_formal_evidence": False,
            "candidate_sources_are_formal_evidence": False,
            "formal_evidence_starts_at": "citation",
        }
        return payload


@dataclass(frozen=True)
class CapabilityAction:
    action: str
    actor: str
    priority: int
    reason: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveRuntimeResult:
    status: str
    run_id: str
    objective_artifact_id: str | None
    max_steps: int
    allow_delivery: bool
    step_artifact_ids: list[str]
    decision_artifact_ids: list[str]
    executed_results: list[dict[str, Any]]
    stop_reason: str
    final_state: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CapabilityArbiter:
    def __init__(self, store: Store, run_id: str, objective: IntelligenceObjective, workbench: EvidenceWorkbench):
        self.store = store
        self.run_id = run_id
        self.objective = objective
        self.workbench = workbench
        self.metadata = store.get_run_metadata(run_id)

    def allowed_actions(self) -> list[CapabilityAction]:
        actions: list[CapabilityAction] = []
        counters = _runtime_counters(self.store, self.run_id)
        pending_sources = pending_gateway_sources(self.store, self.run_id)
        candidate_continuation_ready = self._latest_candidate_continuation_ready()
        pending_pipeline_task = self._pending_linkgate_or_extract_task()
        has_browser_candidate = self._has_advisory_browser_source()
        unprocessed_candidate_id = self._unprocessed_candidate_id()
        targeted_gap = self._targeted_evidence_request()
        needs_review_refresh = self._has_new_citation_since_latest_qa()
        if pending_sources and (not self.workbench.candidate_research_source_ids or has_browser_candidate):
            actions.append(
                CapabilityAction(
                    "normalize_source",
                    "SourceAcquisitionGateway",
                    20,
                    "advisory_source_waits_for_gateway",
                    {"artifact_id": pending_sources[0]["artifact_id"]},
                )
            )
        if (
            unprocessed_candidate_id
            and not candidate_continuation_ready
            and not pending_pipeline_task
        ):
            actions.append(
                CapabilityAction(
                    "continue_candidate",
                    "LinkGateAgent",
                    6,
                    "candidate_source_ready_for_linkgate_extract",
                    {"candidate_id": unprocessed_candidate_id},
                )
            )
        if pending_pipeline_task:
            actions.append(CapabilityAction("advance_pipeline", "RuntimeKernel", 28, "linkgate_or_extract_task_pending"))
        if self.workbench.citation_ids and (not self.workbench.latest_qa_report_id or needs_review_refresh):
            actions.append(CapabilityAction("analyze_evidence", "StrategicAnalystAgent", 40, "formal_citation_ready_for_analysis"))
        if self.workbench.latest_qa_report_id and not self.workbench.qa_passed:
            review_id = self.workbench.latest_review_qa_id
            actions.append(CapabilityAction("continue_repair", "StrategicAnalystAgent", 50, "qa_failed_repairable", {"review_qa_id": review_id}))
        if targeted_gap and counters["search_calls"] < self.objective.budget.max_search_calls:
            actions.append(
                CapabilityAction(
                    "search_sources",
                    "Researcher",
                    8,
                    "critic_targeted_evidence_request_needs_search",
                    {"query": targeted_gap.get("query"), "targeted_evidence_request": targeted_gap},
                )
            )
        if targeted_gap and counters["browser_operations"] < self.objective.budget.max_browser_operations:
            actions.append(
                CapabilityAction(
                    "browse_sources",
                    "BrowserAgent",
                    18,
                    "critic_targeted_evidence_request_allows_browser_escalation",
                    {"target_url": targeted_gap.get("target_url"), "targeted_evidence_request": targeted_gap},
                )
            )
        if self.workbench.qa_passed and not self.workbench.latest_convergence_decision_id and not targeted_gap and not needs_review_refresh:
            actions.append(
                CapabilityAction(
                    "converge_evidence",
                    "EvidenceConvergenceRuntime",
                    60,
                    "qa_passed_evidence_needs_convergence",
                    {"review_qa_id": self.workbench.latest_review_qa_id},
                )
            )
        if self.workbench.latest_convergence_decision_id and not targeted_gap:
            actions.append(CapabilityAction("execute_delivery", "WriterAgent", 80, "qa_passed_convergence_ready_delivery_default"))
        if unprocessed_candidate_id and counters["browser_operations"] > 0:
            actions.append(
                CapabilityAction(
                    "continue_candidate",
                    "LinkGateAgent",
                    6,
                    "new_source_acquisition_candidates_need_linkgate_extract",
                    {"candidate_id": unprocessed_candidate_id},
                )
            )
        if (
            len(self.workbench.citation_ids) == 0
            and not self.workbench.candidate_research_source_ids
            and counters["search_calls"] < self.objective.budget.max_search_calls
        ):
            actions.append(CapabilityAction("search_sources", "Researcher", 10, "lead_selected_search_first_source_acquisition"))
        if (
            len(self.workbench.citation_ids) == 0
            and counters["search_calls"] > 0
            and candidate_continuation_ready
            and not unprocessed_candidate_id
            and counters["browser_operations"] < self.objective.budget.max_browser_operations
        ):
            actions.append(CapabilityAction("browse_sources", "BrowserAgent", 15, "search_or_fetch_insufficient_dynamic_observation_allowed"))
        if (
            len(self.workbench.citation_ids) == 0
            and counters["browser_operations"] < self.objective.budget.max_browser_operations
            and self.metadata.get("browser_source_target_url")
            and self.metadata.get("browser_backend")
        ):
            actions.append(
                CapabilityAction(
                    "browse_sources",
                    "BrowserAgent",
                    11,
                    "explicit_browser_source_target_available",
                    {"target_url": self.metadata.get("browser_source_target_url")},
                )
            )
        if len(self.workbench.citation_ids) == 0 and counters["extractor_repairs"] < self.objective.budget.max_extractor_repairs and self.workbench.raw_document_ids:
            actions.append(CapabilityAction("repair_extract", "ExtractorAgent", 25, "raw_document_exists_without_citation"))
        if not actions and not self.workbench.citation_ids and counters["browser_operations"] >= self.objective.budget.max_browser_operations:
            actions.append(
                CapabilityAction(
                    "deliver_blocked",
                    "WriterAgent",
                    900,
                    "exhausted_no_citation_after_source_acquisition",
                    {"stop_reason": "exhausted_no_citation_after_source_acquisition"},
                )
            )
        if not actions and not self.workbench.citation_ids and counters["search_calls"] >= self.objective.budget.max_search_calls:
            actions.append(
                CapabilityAction(
                    "deliver_blocked",
                    "WriterAgent",
                    910,
                    "exhausted_search_budget_without_citation",
                    {"stop_reason": "exhausted_search_budget"},
                )
            )
        if not actions:
            reason = "blocked_no_citation" if not self.workbench.citation_ids else "stop"
            actions.append(CapabilityAction("stop", "LeadAgent", 999, reason, {"stop_reason": reason}))
        return sorted(actions, key=lambda item: item.priority)

    def _latest_candidate_continuation_ready(self) -> bool:
        continuations = [row for row in self.store.list_artifacts(self.run_id) if row["artifact_type"] == "candidate_continuation"]
        if not continuations:
            return False
        latest = _read_json(continuations[-1])
        return latest.get("status") in {"citation_ready", "blocked_no_citation", "blocked_no_linkgate_selection"}

    def _unprocessed_candidate_id(self) -> str | None:
        processed: set[str] = set()
        for row in self.store.list_artifacts(self.run_id):
            if row["artifact_type"] != "candidate_continuation":
                continue
            payload = _read_json(row)
            processed.update(payload.get("candidate_research_source_ids") or [])
            scope = payload.get("scope") or {}
            processed.update(scope.get("scope_artifact_ids") or [])
            source_id = scope.get("source_artifact_id")
            if source_id:
                processed.add(source_id)
        for candidate_id in reversed(self.workbench.candidate_research_source_ids):
            if candidate_id not in processed:
                return candidate_id
        return None

    def _pending_linkgate_or_extract_task(self) -> bool:
        for row in self.store.list_tasks(self.run_id):
            if row["agent_name"] in {"LinkGateAgent", "ExtractorAgent"} and row["status"] in {"pending", "in_progress"}:
                return True
        return False

    def _has_advisory_browser_source(self) -> bool:
        for row in self.store.list_artifacts(self.run_id):
            if row["artifact_type"] in {"candidate_source", "browser_swarm_operation", "swarm_handoff"}:
                return True
        return False

    def _has_new_citation_since_latest_qa(self) -> bool:
        latest_qa = None
        for row in reversed(self.store.list_artifacts(self.run_id)):
            if row["artifact_type"] == "qa_report":
                latest_qa = row
                break
        if latest_qa is None:
            return False
        latest_qa_created_at = str(latest_qa["created_at"] or "")
        for row in reversed(self.store.list_citations(self.run_id)):
            if str(row["created_at"] or "") > latest_qa_created_at:
                return True
            break
        return False

    def _targeted_evidence_request(self) -> dict[str, Any] | None:
        consumed = set(self.metadata.get("consumed_targeted_qa_report_ids") or [])
        for row in reversed(self.store.list_artifacts(self.run_id)):
            if row["artifact_type"] != "qa_report":
                continue
            payload = _read_json(row)
            request = _targeted_evidence_request_from_qa(payload, self.objective.query)
            if request:
                if row["artifact_id"] in consumed:
                    return None
                request["qa_report_artifact_id"] = row["artifact_id"]
                return request
            if payload.get("passed"):
                return None
        return None


class ObjectiveStateMachine:
    def next_state(self, current: str, workbench: EvidenceWorkbench, action: str, result: dict[str, Any]) -> tuple[str, str]:
        if action == "ask_human" or result.get("status") in {"blocked_human_gate", "pending_authorization"}:
            return "needs_human", "needs_human_authorization"
        if action == "execute_delivery" and result.get("status") in {"report_ready", "blocked_delivery_ready"}:
            return "delivered", "delivered"
        if action in {"deliver_partial", "deliver_blocked"} and result.get("status") in {"report_ready", "blocked_delivery_ready"}:
            return "delivered", result.get("stop_reason") or action
        if action == "request_delivery":
            return "delivery_ready", "delivery_requested"
        if action == "converge_evidence":
            return "delivery_ready" if result.get("status") in {"retain", "merge", "supersede"} else "converging", result.get("stop_reason") or "convergence_recorded"
        if action in {"search_sources", "browse_sources", "normalize_source"}:
            return "acquiring_sources", result.get("stop_reason") or "source_acquisition_progress"
        if action in {"continue_candidate", "repair_extract"}:
            return "extracting_evidence", result.get("stop_reason") or "evidence_extraction_progress"
        if action in {"analyze_evidence", "continue_repair"}:
            return "reviewing", result.get("stop_reason") or "qa_review_progress"
        if action == "stop":
            stop_reason = result.get("stop_reason") or "max_steps_reached"
            return ("exhausted" if stop_reason.startswith("exhausted") or stop_reason in {"blocked_no_citation", "max_steps_reached"} else "blocked"), stop_reason
        if workbench.qa_passed:
            return "converging", "qa_passed"
        if workbench.citation_ids:
            return "analyzing", "citation_ready"
        return current if current in OBJECTIVE_STATES else "planning", "state_unchanged"


class ObjectiveDrivenSwarmRuntime:
    def __init__(
        self,
        store: Store,
        run_id: str,
        *,
        workspace_root: str | Path = ".insightswarm/runtime",
        model_provider: str = "fake",
        allow_delivery: bool = False,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self.workspace_root = Path(workspace_root)
        self.workspace = RuntimeWorkspace(self.workspace_root, run_id)
        self.model_provider = model_provider
        self.allow_delivery = allow_delivery
        self.state_machine = ObjectiveStateMachine()
        self.templates = AgentTemplateRegistry()

    def run(self, max_steps: int = 12) -> ObjectiveRuntimeResult:
        objective = self.ensure_objective()
        self.store.update_run_status(self.run_id, "running")
        self._write_state_transition(None, "initialized", "objective_runtime_started", 0, None)
        step_ids: list[str] = []
        decision_ids: list[str] = []
        executed: list[dict[str, Any]] = []
        state = self._latest_state() or "initialized"
        stop_reason = "max_steps_reached"
        no_progress = 0
        last_signature = None
        for step_index in range(max(1, int(max_steps))):
            workbench = build_evidence_workbench(self.store, self.run_id)
            arbiter = CapabilityArbiter(self.store, self.run_id, objective, workbench)
            allowed = [action.to_dict() for action in arbiter.allowed_actions()]
            arbitration_id = self._write_arbitration(step_index, objective, workbench, allowed, state)
            envelope_id, envelope = self._write_lead_context(step_index, objective, workbench, allowed, arbitration_id, state)
            decision = self._lead_decision(envelope, allowed)
            decision_id = self._write_decision(step_index, decision, envelope_id, arbitration_id)
            result = self._execute(decision)
            progress_signal = self._progress_signal(workbench, build_evidence_workbench(self.store, self.run_id), result)
            signature = (decision.get("action"), decision.get("actor"), json.dumps(decision.get("data") or {}, sort_keys=True))
            no_progress = no_progress + 1 if not progress_signal and signature == last_signature else 0
            last_signature = signature
            next_state, transition_reason = self.state_machine.next_state(state, build_evidence_workbench(self.store, self.run_id), decision["action"], result)
            if no_progress >= objective.budget.max_no_progress_steps:
                next_state, transition_reason = "exhausted", "no_progress_exhausted"
                result = {**result, "status": "exhausted", "stop_reason": "no_progress_exhausted", "terminal": True}
            state_id = self._write_state_transition(state, next_state, transition_reason, step_index, decision_id)
            step_id = self._write_step(step_index, decision, result, envelope_id, arbitration_id, state_id, progress_signal)
            step_ids.append(step_id)
            decision_ids.append(decision_id)
            executed.append(result)
            state = next_state
            stop_reason = result.get("stop_reason") or transition_reason
            if result.get("terminal") or state in {"delivered", "needs_human", "exhausted", "blocked"}:
                break
        else:
            state_id = self._write_state_transition(state, "exhausted", "max_steps_reached", max_steps, None)
            state = "exhausted"
            stop_reason = "max_steps_reached"
            self.store.emit_event(self.run_id, None, "ObjectiveRuntime", "objective_runtime_max_steps", "Objective runtime reached max steps.", {"state_artifact_id": state_id})
        self.store.update_run_status(self.run_id, "completed" if state in {"delivered", "delivery_ready"} else state)
        return ObjectiveRuntimeResult(
            "objective_governed",
            self.run_id,
            objective.artifact_id,
            max_steps,
            self.allow_delivery,
            step_ids,
            decision_ids,
            executed,
            stop_reason,
            state,
        )

    def ensure_objective(self) -> IntelligenceObjective:
        existing = [row for row in self.store.list_artifacts(self.run_id) if row["artifact_type"] == "intelligence_objective"]
        if existing:
            payload = _read_json(existing[-1])
            budget_data = payload.get("budget") or {}
            return IntelligenceObjective(
                self.run_id,
                payload.get("query") or self.store.get_run_metadata(self.run_id).get("query") or "",
                payload.get("objective_id") or f"objective:{self.run_id}",
                payload.get("status") or "active",
                payload.get("key_questions") or [],
                payload.get("success_criteria") or [],
                payload.get("evidence_requirements") or [],
                payload.get("freshness_requirement") or "prefer_recent_public_sources",
                payload.get("forbidden_actions") or [],
                ObjectiveBudget(**{key: budget_data.get(key, getattr(ObjectiveBudget(), key)) for key in ObjectiveBudget().__dict__}),
                existing[-1]["artifact_id"],
            )
        metadata = self.store.get_run_metadata(self.run_id)
        query = metadata.get("query") or metadata.get("competitor") or self.store.get_run(self.run_id)["name"]
        objective = IntelligenceObjective(
            self.run_id,
            query,
            f"objective:{self.run_id}",
            "active",
            [
                query,
                "What public sources support the answer?",
                "What evidence gaps or confidence limits remain?",
            ],
            [
                "At least one formal citation or an explicit evidence-safe blocked state.",
                "QA and convergence must be visible before delivery.",
                "Writer should deliver report, report_partial, or report_blocked when QA/convergence permits.",
            ],
            [
                "Use public verifiable sources.",
                "Search/Browser/subagent outputs are advisory until Extractor creates citations.",
                "Prefer primary or high-reliability sources for strategic intelligence.",
            ],
            "freshness_sensitive" if any(term in query.lower() for term in ["next", "下步", "战略", "latest", "recent", "2026"]) else "prefer_recent_public_sources",
            [
                "fabricate_evidence",
                "browser_creates_citation_directly",
                "infinite_loop",
            ],
            ObjectiveBudget(
                max_search_calls=int(metadata.get("max_search_calls") or 3),
                max_browser_operations=int(metadata.get("max_browser_operations") or 2),
                max_extractor_repairs=int(metadata.get("max_extractor_repairs") or 2),
                max_subagents=int(metadata.get("max_subagents_per_run") or 2),
                max_no_progress_steps=int(metadata.get("max_no_progress_steps") or 3),
                max_model_tool_budget=int(metadata.get("max_model_tool_budget") or 24),
            ),
        )
        artifact_id = self.store.write_artifact(
            self.run_id,
            self._lead_task_id(),
            "intelligence_objective",
            "application/json",
            json.dumps(objective.to_dict(), ensure_ascii=True, indent=2),
            metadata={"schema": "intelligence_objective.v1", "objective_id": objective.objective_id, "query": query, "status": "active"},
            suffix=".json",
        )
        self.store.emit_event(self.run_id, self._lead_task_id(), "ObjectiveRuntime", "intelligence_objective_created", "Intelligence objective created.", {"artifact_id": artifact_id})
        return IntelligenceObjective(
            objective.run_id,
            objective.query,
            objective.objective_id,
            objective.status,
            objective.key_questions,
            objective.success_criteria,
            objective.evidence_requirements,
            objective.freshness_requirement,
            objective.forbidden_actions,
            objective.budget,
            artifact_id,
        )

    def _lead_decision(self, envelope: dict[str, Any], allowed: list[dict[str, Any]]) -> dict[str, Any]:
        fallback = allowed[0] if allowed else {"action": "stop", "actor": "LeadAgent", "reason": "no_allowed_action", "data": {"stop_reason": "blocked_no_verifiable_source"}}
        decision = {
            "schema": "objective_governance_decision.v1",
            "action": fallback["action"],
            "actor": fallback["actor"],
            "reason": fallback["reason"],
            "data": fallback.get("data") or {},
            "allowed_actions": allowed,
            "model_governed": False,
            "model_decision": None,
            "fallback_reason": None,
            "policy_result": self._policy_result(fallback["action"]),
        }
        if self.model_provider == "fake":
            return decision
        model = build_audited_model_client(self.model_provider, self.store)
        result = model.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are LeadAgent selecting one action for an objective-state intelligence swarm. "
                        "Choose only from allowed_actions. Respect evidence and delivery boundaries. "
                        "Return a valid JSON object only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "objective": envelope.get("objective"),
                            "current_state": envelope.get("objective_state"),
                            "allowed_actions": allowed,
                            "evidence_workbench": envelope.get("evidence_workbench"),
                            "loop_control": envelope.get("loop_control"),
                            "output_schema": {"action": "one allowed action", "actor": "actor from selected action", "reason": "short reason", "data": "optional object"},
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=600,
            temperature=0,
            metadata={"run_id": self.run_id, "task_id": self._lead_task_id(), "role": "ObjectiveLeadAgentDecision"},
        )
        parsed = result.json_data if result.status == "ok" and isinstance(result.json_data, dict) else {}
        action = parsed.get("action") or parsed.get("decision_type")
        by_action = {item["action"]: item for item in allowed}
        if action in by_action and self._policy_result(action).get("passed", True):
            selected = by_action[action]
            return {
                **decision,
                "action": action,
                "actor": parsed.get("actor") or selected["actor"],
                "reason": parsed.get("reason") or selected["reason"],
                "data": {**(selected.get("data") or {}), **(parsed.get("data") if isinstance(parsed.get("data"), dict) else {})},
                "model_governed": True,
                "model_decision": {"status": result.status, "provider": result.provider, "model": result.model, "parsed": parsed},
            }
        return {**decision, "model_decision": {"status": result.status, "provider": result.provider, "model": result.model, "parsed": parsed, "error": result.error}, "fallback_reason": "model_action_invalid_or_unavailable"}

    def _execute(self, decision: dict[str, Any]) -> dict[str, Any]:
        action = decision["action"]
        data = decision.get("data") or {}
        if action == "search_sources":
            return self._execute_search(data)
        if action == "browse_sources":
            return self._execute_browser(data)
        if action == "normalize_source":
            return normalize_source(self.store, self.run_id, artifact_id=data.get("artifact_id"))
        if action == "continue_candidate":
            from insightswarm.candidate_continuation import continue_candidate_sources

            return continue_candidate_sources(self.store, self.run_id, candidate_id=data.get("candidate_id"), model_provider=self.model_provider)
        if action == "advance_pipeline":
            return self._advance_pipeline()
        if action == "repair_extract":
            return self._execute_extract_repair()
        if action == "analyze_evidence":
            from insightswarm.governance_runtime import LeadAgentGovernanceRuntime

            return LeadAgentGovernanceRuntime(self.store, self.run_id, workspace_root=self.workspace_root, model_provider=self.model_provider, allow_delivery=self.allow_delivery)._execute_analysis_reentry()
        if action == "continue_repair":
            from insightswarm.repair_continuation import continue_repair

            return continue_repair(self.store, self.run_id, review_qa_id=data.get("review_qa_id"), model_provider=self.model_provider)
        if action == "converge_evidence":
            from insightswarm.evidence_convergence import converge_evidence

            return converge_evidence(self.store, self.run_id, review_qa_id=data.get("review_qa_id"))
        if action == "request_delivery":
            return self._execute_delivery(data)
        if action == "execute_delivery":
            return self._execute_delivery(data)
        if action == "deliver_blocked":
            return self._execute_delivery(data)
        if action == "deliver_partial":
            return self._execute_delivery(data)
        if action == "ask_human":
            record = self.workspace.write_record("authorization", status="pending", stop_reason="needs_human_authorization", payload_fields={"decision": decision})
            return {"status": "blocked_human_gate", "authorization_record_id": record.record_id, "stop_reason": "needs_human_authorization", "terminal": True}
        if action == "arbitrate":
            record = self.workspace.write_record("arbitration", status="arbitration_required", stop_reason="arbitration_required", payload_fields={"decision": decision})
            return {"status": "arbitration_required", "arbitration_record_id": record.record_id, "stop_reason": "arbitration_required", "terminal": True}
        return {"status": "stopped", "stop_reason": (data.get("stop_reason") or decision.get("reason") or "max_steps_reached"), "terminal": True}

    def _execute_delivery(self, data: dict[str, Any]) -> dict[str, Any]:
        from insightswarm.writer_continuation import continue_writer

        workbench = build_evidence_workbench(self.store, self.run_id)
        review_id = data.get("review_qa_id") or workbench.latest_review_qa_id
        qa_report_id = workbench.latest_qa_report_id
        status = "approved" if self.allow_delivery else "phase48_default_delivery"
        request_id = self._write_delivery_request(status)
        if review_id:
            result = continue_writer(self.store, self.run_id, review_qa_id=review_id, model_provider=self.model_provider)
        else:
            result = continue_writer(self.store, self.run_id, qa_report_id=qa_report_id, model_provider=self.model_provider)
        return {**result, "delivery_request_artifact_id": request_id, "terminal": True}

    def _execute_search(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        data = data or {}
        metadata = self.store.get_run_metadata(self.run_id)
        query_override = data.get("query")
        targeted = data.get("targeted_evidence_request") if isinstance(data.get("targeted_evidence_request"), dict) else {}
        qa_report_id = targeted.get("qa_report_artifact_id")
        if qa_report_id:
            consumed = list(metadata.get("consumed_targeted_qa_report_ids") or [])
            if qa_report_id not in consumed:
                consumed.append(qa_report_id)
                _update_run_metadata(self.store, self.run_id, {"consumed_targeted_qa_report_ids": consumed})
        task_id = self.store.create_task(
            self.run_id,
            "Search",
            "SearchAgent",
            [self._lead_task_id()],
            {
                "objective_runtime": True,
                "source_acquisition_actor": True,
                "researcher_runner": True,
                "lead_selected_tool": "search.web",
                "lead_selected_query": query_override,
            },
        )
        if query_override:
            _update_run_metadata(self.store, self.run_id, {"query": query_override, "original_query": metadata.get("original_query") or metadata.get("query")})
        model = build_audited_model_client(self.model_provider, self.store)
        SearchAgent(self.store, model).execute(self.run_id, task_id)
        if query_override:
            _update_run_metadata(self.store, self.run_id, {"query": metadata.get("query")})
        search_artifacts = [row for row in self.store.list_artifacts(self.run_id) if row["task_id"] == task_id and row["artifact_type"] == "search_results"]
        candidate_ids = self._search_results_to_candidates(search_artifacts[-1] if search_artifacts else None)
        return {
            "status": "search_completed" if search_artifacts else "search_failed",
            "search_task_id": task_id,
            "search_results_artifact_id": search_artifacts[-1]["artifact_id"] if search_artifacts else None,
            "candidate_research_source_ids": candidate_ids,
            "stop_reason": "search_source_candidates_ready" if candidate_ids else "blocked_no_verifiable_source",
        }

    def _advance_pipeline(self) -> dict[str, Any]:
        from insightswarm.harness.runner import Runner

        runner = Runner(self.store, self.model_provider)
        runner._expand_dynamic_extract_tasks(self.run_id)
        result = runner.runtime.execute_next(self.run_id)
        return {
            "status": result.status,
            "task_id": result.task_id,
            "agent_name": result.agent_name,
            "stop_reason": result.stop_reason or "pipeline_advanced",
        }

    def _execute_browser(self, data: dict[str, Any]) -> dict[str, Any]:
        from insightswarm.swarm_runtime import SwarmRuntime

        metadata = self.store.get_run_metadata(self.run_id)
        record = self.workspace.write_record(
            "work_order",
            status="claimable",
            stop_reason="objective_browser_source_acquisition",
            payload_fields={
                "plan_kind": "branch_plan",
                "action": "branch_from_source_acquisition",
                "data": {
                    "preferred_actor": "BrowserAgent",
                    "goal": metadata.get("query") or data.get("goal") or "Acquire source candidates.",
                    "browser_backend": metadata.get("browser_backend") or ("cdp" if os.getenv("INSIGHTSWARM_CDP_URL") else "fake"),
                    "cdp_url": metadata.get("browser_cdp_url") or os.getenv("INSIGHTSWARM_CDP_URL"),
                    "target_url": metadata.get("browser_source_target_url") or data.get("target_url"),
                },
            },
        )
        swarm = SwarmRuntime(self.store, self.run_id, workspace_root=self.workspace_root).step(work_order_id=record.record_id, agent_name="BrowserAgent")
        gateway = normalize_source(self.store, self.run_id, work_order_id=record.record_id) if swarm.work_order_id else {}
        raw_promotions = self._promote_browser_candidates_to_raw_documents()
        return {
            "status": swarm.status,
            "work_order_id": record.record_id,
            "swarm_execution": swarm.to_dict(),
            "gateway_result": gateway,
            "browser_raw_source_promotions": raw_promotions,
            "stop_reason": swarm.stop_reason or "browser_source_acquisition_completed",
        }

    def _promote_browser_candidates_to_raw_documents(self) -> list[dict[str, Any]]:
        promoted: list[dict[str, Any]] = []
        already_promoted = {
            loads(row["metadata_json"], {}).get("browser_candidate_source_artifact_id")
            for row in self.store.list_artifacts(self.run_id)
            if row["artifact_type"] == "raw_document"
        }
        for row in self.store.list_artifacts(self.run_id):
            if row["artifact_type"] != "candidate_source":
                continue
            metadata = loads(row["metadata_json"], {})
            if row["artifact_id"] in already_promoted or metadata.get("source_kind") != "browser_handoff":
                continue
            result = promote_candidate_to_raw_document(
                self.store,
                self.run_id,
                row["artifact_id"],
                quality_mode=self.store.get_run_metadata(self.run_id).get("quality_mode", "production"),
            )
            promoted.append(result)
        return promoted

    def _execute_extract_repair(self) -> dict[str, Any]:
        from insightswarm.agents.extractor import ExtractorAgent

        raw_documents = [row for row in self.store.list_artifacts(self.run_id) if row["artifact_type"] == "raw_document"]
        if not raw_documents:
            return {"status": "blocked_no_raw_document", "stop_reason": "blocked_no_verifiable_source"}
        task_id = self.store.create_task(
            self.run_id,
            "Extract",
            "ExtractorAgent",
            [self._lead_task_id()],
            {
                "objective_runtime": True,
                "extractor_repair": True,
                "raw_document_id": raw_documents[-1]["artifact_id"],
                "extract_strategy": "repair_quote_normalization",
            },
        )
        ExtractorAgent(self.store, build_audited_model_client(self.model_provider, self.store)).execute(self.run_id, task_id)
        task = self.store.get_task(task_id)
        task_meta = loads(task["metadata_json"], {})
        return {"status": "extract_repair_completed", "extractor_task_id": task_id, "citation_ids": task_meta.get("citation_ids", []), "stop_reason": "extractor_repair_completed" if task_meta.get("citation_ids") else "blocked_no_citation"}

    def _search_results_to_candidates(self, artifact: Any | None) -> list[str]:
        if not artifact:
            return []
        payload = _read_json(dict(artifact))
        results = payload.get("results") or []
        task_id = self._lead_task_id()
        candidate_ids = []
        existing = {
            (_normalize_url(row["source_url"]), loads(row["metadata_json"], {}).get("origin_artifact_id"))
            for row in self.store.list_artifacts(self.run_id)
            if row["artifact_type"] == "candidate_research_source"
        }
        for item in results[:5]:
            url = item.get("url")
            if not url:
                continue
            key = (_normalize_url(url), artifact["artifact_id"])
            if key in existing:
                continue
            source_kind = "search_result"
            candidate_payload = {
                "schema": "candidate_research_source.v1",
                "source_url": url,
                "normalized_url": key[0],
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "source_kind": source_kind,
                "origin_artifact_id": artifact["artifact_id"],
                "origin_artifact_type": "search_results",
                "origin_task_id": artifact["task_id"],
                "confidence": item.get("score"),
                "risk_flags": [],
                "source_requirements": {"requires_public_url": True, "must_pass_link_gate": True, "must_not_create_formal_evidence": True},
                "requires_link_gate": True,
                "gateway_normalized": True,
                "formal_evidence": False,
            }
            candidate_id = self.store.write_artifact(
                self.run_id,
                task_id,
                "candidate_research_source",
                "application/json",
                json.dumps(candidate_payload, ensure_ascii=True, indent=2),
                source_url=url,
                metadata={
                    "schema": "candidate_research_source.v1",
                    "source_url": url,
                    "normalized_url": key[0],
                    "source_kind": source_kind,
                    "origin_artifact_id": artifact["artifact_id"],
                    "origin_artifact_type": "search_results",
                    "requires_link_gate": True,
                    "gateway_normalized": True,
                },
                suffix=".json",
            )
            candidate_ids.append(candidate_id)
            existing.add(key)
        return candidate_ids

    def _write_lead_context(self, step_index: int, objective: IntelligenceObjective, workbench: EvidenceWorkbench, allowed: list[dict[str, Any]], arbitration_id: str, state: str) -> tuple[str, dict[str, Any]]:
        template = self.templates.get("LeadAgent").to_dict()
        payload = {
            "schema": "isolated_context_envelope.v1",
            "agent_identity": {"agent_name": "LeadAgent", "role": template["role"]},
            "role_template": template,
            "task_scope": {"run_id": self.run_id, "objective_id": objective.objective_id, "step_index": step_index},
            "objective": objective.to_dict(),
            "objective_state": state,
            "allowed_tools": [],
            "forbidden_actions": template.get("forbidden_actions", []),
            "mailbox_scope": {"messages": self._mailbox_scope()},
            "evidence_workbench": workbench.to_dict(),
            "allowed_actions": allowed,
            "tool_contracts": build_tool_contract_registry().get("contracts", [])[:24],
            "loop_control": {"counters": _runtime_counters(self.store, self.run_id), "stop_taxonomy": sorted(STOP_REASONS)},
            "negative_constraints": objective.forbidden_actions,
            "budget": objective.budget.to_dict(),
            "output_contract": {"actions": sorted(ARBITER_ACTIONS)},
            "capability_arbitration_artifact_id": arbitration_id,
            "isolation": {"full_run_payload_history_included": False, "scoped_mailbox_only": True},
        }
        artifact_id = self.store.write_artifact(
            self.run_id,
            self._lead_task_id(),
            "isolated_context_envelope",
            "application/json",
            json.dumps(payload, ensure_ascii=True, indent=2),
            metadata={"schema": "isolated_context_envelope.v1", "agent_name": "LeadAgent", "objective_runtime": True, "isolated": True},
            suffix=".json",
        )
        return artifact_id, payload

    def _write_arbitration(self, step_index: int, objective: IntelligenceObjective, workbench: EvidenceWorkbench, allowed: list[dict[str, Any]], state: str) -> str:
        payload = {
            "schema": "capability_arbitration.v1",
            "objective_id": objective.objective_id,
            "step_index": step_index,
            "objective_state": state,
            "allowed_actions": allowed,
            "actor_candidates": sorted({item["actor"] for item in allowed}),
            "evidence_workbench": workbench.to_dict(),
        }
        return self.store.write_artifact(
            self.run_id,
            self._lead_task_id(),
            "capability_arbitration",
            "application/json",
            json.dumps(payload, ensure_ascii=True, indent=2),
            metadata={"schema": "capability_arbitration.v1", "objective_state": state, "action_count": len(allowed)},
            suffix=".json",
        )

    def _write_decision(self, step_index: int, decision: dict[str, Any], envelope_id: str, arbitration_id: str) -> str:
        payload = {**decision, "step_index": step_index, "isolated_context_envelope_artifact_id": envelope_id, "capability_arbitration_artifact_id": arbitration_id}
        return self.store.write_artifact(
            self.run_id,
            self._lead_task_id(),
            "governance_decision",
            "application/json",
            json.dumps(payload, ensure_ascii=True, indent=2),
            metadata={
                "schema": "objective_governance_decision.v1",
                "objective_runtime": True,
                "decision_type": decision["action"],
                "actor": decision["actor"],
                "model_governed": decision.get("model_governed"),
                "fallback_reason": decision.get("fallback_reason"),
            },
            suffix=".json",
        )

    def _write_step(self, step_index: int, decision: dict[str, Any], result: dict[str, Any], envelope_id: str, arbitration_id: str, state_id: str, progress_signal: dict[str, Any]) -> str:
        payload = {
            "schema": "governance_step.v1",
            "objective_runtime": True,
            "step_index": step_index,
            "decision": decision,
            "execution_result": result,
            "progress_signal": progress_signal,
            "isolated_context_envelope_artifact_id": envelope_id,
            "capability_arbitration_artifact_id": arbitration_id,
            "objective_state_transition_artifact_id": state_id,
        }
        return self.store.write_artifact(
            self.run_id,
            self._lead_task_id(),
            "governance_step",
            "application/json",
            json.dumps(payload, ensure_ascii=True, indent=2),
            metadata={"schema": "governance_step.v1", "objective_runtime": True, "decision_type": decision["action"], "status": result.get("status")},
            suffix=".json",
        )

    def _write_state_transition(self, previous: str | None, next_state: str, reason: str, step_index: int, decision_id: str | None) -> str:
        if next_state not in OBJECTIVE_STATES:
            next_state = "blocked"
        payload = {
            "schema": "objective_state_transition.v1",
            "previous_state": previous,
            "next_state": next_state,
            "reason": reason,
            "step_index": step_index,
            "governance_decision_artifact_id": decision_id,
        }
        artifact_id = self.store.write_artifact(
            self.run_id,
            self._lead_task_id(),
            "objective_state_transition",
            "application/json",
            json.dumps(payload, ensure_ascii=True, indent=2),
            metadata={"schema": "objective_state_transition.v1", "previous_state": previous, "next_state": next_state, "reason": reason},
            suffix=".json",
        )
        self.workspace.write_record("objective_state", status=next_state, stop_reason=reason, payload_fields={"artifact_id": artifact_id, "step_index": step_index})
        return artifact_id

    def _write_delivery_request(self, status: str) -> str:
        workbench = build_evidence_workbench(self.store, self.run_id)
        payload = {
            "schema": "delivery_request.v1",
            "status": status,
            "allow_delivery": self.allow_delivery,
            "source_review_qa_continuation_artifact_id": workbench.latest_review_qa_id,
            "source_evidence_convergence_decision_artifact_id": workbench.latest_convergence_decision_id,
            "formal_delivery_boundary": True,
        }
        return self.store.write_artifact(
            self.run_id,
            self._lead_task_id(),
            "delivery_request",
            "application/json",
            json.dumps(payload, ensure_ascii=True, indent=2),
            metadata={"schema": "delivery_request.v1", "status": status, "allow_delivery": self.allow_delivery, "objective_runtime": True},
            suffix=".json",
        )

    def _policy_result(self, action: str) -> dict[str, Any]:
        if action == "execute_delivery" and not self.allow_delivery:
            return {"passed": True, "gate": "delivery_boundary", "reason": "phase48_default_delivery"}
        if action == "browse_sources" and not (self.store.get_run_metadata(self.run_id).get("browser_backend") or os.getenv("INSIGHTSWARM_CDP_URL")):
            return {"passed": True, "gate": "browser_authority", "reason": "fake_backend_allowed_for_non_acceptance"}
        return {"passed": True, "gate": "objective_policy", "reason": "passed"}

    def _mailbox_scope(self) -> list[dict[str, Any]]:
        rows = []
        for row in self.store.conn.execute("SELECT * FROM messages WHERE run_id = ? ORDER BY created_at", (self.run_id,)):
            payload = loads(row["payload_json"], {})
            rows.append({"message_id": row["message_id"], "sender": row["sender"], "recipient": row["recipient"], "status": row["status"], "intent": payload.get("intent") or "handoff"})
        return rows[-24:]

    def _lead_task_id(self) -> str:
        for row in self.store.list_tasks(self.run_id):
            if row["agent_name"] == "ResearchLeadAgent":
                if row["status"] == "pending":
                    self.store.set_task_status(row["task_id"], "completed", {"objective_runtime_parent": True})
                return row["task_id"]
        task_id = self.store.create_task(self.run_id, "ResearchLead", "ResearchLeadAgent", metadata={"objective_runtime_parent": True})
        self.store.set_task_status(task_id, "completed", {"objective_runtime_parent": True})
        return task_id

    def _latest_state(self) -> str | None:
        transitions = [row for row in self.store.list_artifacts(self.run_id) if row["artifact_type"] == "objective_state_transition"]
        if not transitions:
            return None
        return _read_json(transitions[-1]).get("next_state")

    def _progress_signal(self, before: EvidenceWorkbench, after: EvidenceWorkbench, result: dict[str, Any]) -> dict[str, Any]:
        signal = {
            "new_source": len(after.advisory_source_ids) > len(before.advisory_source_ids) or len(after.candidate_research_source_ids) > len(before.candidate_research_source_ids),
            "new_raw_document": len(after.raw_document_ids) > len(before.raw_document_ids),
            "new_quote_candidate": after.quote_candidate_count > before.quote_candidate_count,
            "new_citation": len(after.citation_ids) > len(before.citation_ids),
            "qa_or_convergence_changed": after.latest_qa_report_id != before.latest_qa_report_id or after.latest_convergence_decision_id != before.latest_convergence_decision_id,
            "human_gate_reached": result.get("status") in {"blocked_human_gate", "pending_authorization"},
        }
        signal["progressed"] = any(signal.values())
        return signal


def build_evidence_workbench(store: Store, run_id: str) -> EvidenceWorkbench:
    artifacts = [dict(row) for row in store.list_artifacts(run_id)]
    citations = [dict(row) for row in store.list_citations(run_id)]
    advisory = [row["artifact_id"] for row in artifacts if row["artifact_type"] in {"candidate_source", "research_finding", "subagent_handoff", "browser_swarm_operation", "swarm_handoff"}]
    structured = [row for row in artifacts if row["artifact_type"] == "structured_knowledge"]
    discarded = 0
    facts = 0
    for row in structured:
        payload = _read_json(row)
        facts += len(payload.get("facts") or [])
        discarded += len(payload.get("discarded_facts") or [])
    qa_reports = [row for row in artifacts if row["artifact_type"] == "qa_report"]
    latest_qa = qa_reports[-1] if qa_reports else None
    latest_qa_payload = _read_json(latest_qa) if latest_qa else {}
    review_continuations = [row for row in artifacts if row["artifact_type"] == "review_qa_continuation"]
    convergence = [row for row in artifacts if row["artifact_type"] == "evidence_convergence_decision"]
    delivery_requests = [row for row in artifacts if row["artifact_type"] == "delivery_request"]
    reports = [row for row in artifacts if row["artifact_type"] == "report"]
    blocked_reports = [row for row in artifacts if row["artifact_type"] == "report_blocked"]
    return EvidenceWorkbench(
        advisory_source_ids=advisory,
        candidate_research_source_ids=[row["artifact_id"] for row in artifacts if row["artifact_type"] == "candidate_research_source"],
        raw_document_ids=[row["artifact_id"] for row in artifacts if row["artifact_type"] == "raw_document"],
        structured_knowledge_ids=[row["artifact_id"] for row in structured],
        citation_ids=[row["citation_id"] for row in citations],
        quote_candidate_count=facts + discarded,
        discarded_fact_count=discarded,
        qa_passed=bool(latest_qa_payload.get("passed")),
        latest_review_qa_id=review_continuations[-1]["artifact_id"] if review_continuations else None,
        latest_qa_report_id=latest_qa["artifact_id"] if latest_qa else None,
        latest_convergence_decision_id=convergence[-1]["artifact_id"] if convergence else None,
        delivery_request_id=delivery_requests[-1]["artifact_id"] if delivery_requests else None,
        report_id=reports[-1]["artifact_id"] if reports else None,
        report_blocked_id=blocked_reports[-1]["artifact_id"] if blocked_reports else None,
    )


def objective_runtime_summary(store: Store, run_id: str) -> dict[str, Any]:
    artifacts = [dict(row) for row in store.list_artifacts(run_id)]
    objectives = [row for row in artifacts if row["artifact_type"] == "intelligence_objective"]
    transitions = [row for row in artifacts if row["artifact_type"] == "objective_state_transition"]
    arbitrations = [row for row in artifacts if row["artifact_type"] == "capability_arbitration"]
    decisions = [row for row in artifacts if row["artifact_type"] == "governance_decision" and loads(row["metadata_json"], {}).get("objective_runtime")]
    latest_transition = _read_json(transitions[-1]) if transitions else {}
    counters = _runtime_counters(store, run_id)
    workbench = build_evidence_workbench(store, run_id).to_dict()
    return {
        "schema": "objective_runtime_summary.v1",
        "objective_count": len(objectives),
        "objective_state": latest_transition.get("next_state"),
        "state_transition_count": len(transitions),
        "capability_arbitration_count": len(arbitrations),
        "objective_decision_count": len(decisions),
        "model_governed_decision_count": sum(1 for row in decisions if _read_json(row).get("model_governed")),
        "loop_counters": counters,
        "evidence_workbench": workbench,
        "latest_stop_reason": latest_transition.get("reason"),
        "latest_objective_artifact_id": objectives[-1]["artifact_id"] if objectives else None,
    }


def create_and_run_objective(
    store: Store,
    *,
    name: str,
    query: str,
    model_provider: str,
    artifact_dir: Path,
    max_steps: int = 12,
    allow_delivery: bool = False,
    quality_mode: str = "production",
    search_provider: str = "tavily",
    browser_backend: str | None = None,
    browser_cdp_url: str | None = None,
) -> ObjectiveRuntimeResult:
    metadata = {
        "query": query,
        "quality_mode": quality_mode,
        "model_provider": model_provider,
        "search_provider": search_provider,
        "search_limit": 8,
        "link_gate_max_selected": 3,
        "allow_delivery_runtime": allow_delivery,
        "objective_runtime": True,
        "browser_backend": browser_backend,
        "browser_cdp_url": browser_cdp_url,
    }
    run_id = store.create_run(name, metadata)
    return ObjectiveDrivenSwarmRuntime(
        store,
        run_id,
        workspace_root=artifact_dir.parent / "runtime",
        model_provider=model_provider,
        allow_delivery=allow_delivery,
    ).run(max_steps=max_steps)


def _runtime_counters(store: Store, run_id: str) -> dict[str, int]:
    events = [dict(row) for row in store.conn.execute("SELECT * FROM agent_events WHERE run_id = ? ORDER BY created_at", (run_id,))]
    artifacts = [dict(row) for row in store.list_artifacts(run_id)]
    workspace_records = RuntimeWorkspace(store.artifact_dir.parent / "runtime", run_id, create=False).records()
    return {
        "search_calls": sum(1 for row in events if loads(row["metadata_json"], {}).get("tool_name") == "search.web" and row["event_type"] == "tool_call_completed"),
        "browser_operations": sum(1 for row in artifacts if row["artifact_type"] == "browser_swarm_operation") + len(workspace_records.get("browser_swarm_operation", [])),
        "extractor_repairs": sum(1 for row in store.list_tasks(run_id) if loads(row["metadata_json"], {}).get("extractor_repair")),
        "subagents": sum(1 for row in store.list_tasks(run_id) if loads(row["metadata_json"], {}).get("subagent")),
        "tool_calls": sum(1 for row in events if row["event_type"].startswith("tool_call_")),
    }


def _read_json(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {}
    path = Path(row["path"])
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_url(url: str | None) -> str:
    if not url:
        return ""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    return urlunsplit(((parts.scheme or "https").lower(), parts.netloc.lower(), parts.path.rstrip("/") or "/", "", ""))


def _update_run_metadata(store: Store, run_id: str, updates: dict[str, Any]) -> None:
    metadata = store.get_run_metadata(run_id)
    metadata.update({key: value for key, value in updates.items() if value is not None})
    with store.transaction() as conn:
        from insightswarm.util import dumps, now_iso

        conn.execute(
            "UPDATE runs SET metadata_json = ?, updated_at = ? WHERE run_id = ?",
            (dumps(metadata), now_iso(), run_id),
        )


def _targeted_evidence_request_from_qa(payload: dict[str, Any], query: str) -> dict[str, Any] | None:
    direct_request = payload.get("targeted_evidence_request")
    if isinstance(direct_request, dict) and direct_request:
        request = dict(direct_request)
        request.setdefault("query", query)
        request.setdefault("schema", "targeted_evidence_request.v1")
        request.setdefault("source_requirements", ["public verifiable source", "extractable quote", "formal citation required"])
        return request
    rejection = payload.get("rejection") or {}
    failures = list(rejection.get("failures") or [])
    skeptic = payload.get("skeptic_review") or {}
    evidence_gaps = list(skeptic.get("evidence_gaps") or [])
    if not failures and not evidence_gaps:
        return None
    categories = sorted({str(item.get("category") or "evidence") for item in failures if isinstance(item, dict)})
    gates = sorted({str(item.get("gate") or "unknown") for item in failures if isinstance(item, dict)})
    needs_browser = any(gate in {"quote_span_backcheck", "source_url_mismatch"} for gate in gates)
    gap_text = "; ".join(str(item) for item in evidence_gaps[:3])
    suffix = " ".join(part for part in [gap_text, "official source", "latest evidence"] if part)
    return {
        "schema": "targeted_evidence_request.v1",
        "query": f"{query} {suffix}".strip(),
        "target_url": None,
        "failure_categories": categories,
        "failure_gates": gates,
        "evidence_gaps": evidence_gaps,
        "needs_browser_escalation": needs_browser,
        "source_requirements": ["public verifiable source", "extractable quote", "formal citation required"],
    }
