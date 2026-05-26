from __future__ import annotations

import json

from insightswarm.agents.base import Agent, AgentContext
from insightswarm.observability.diagnosis import build_run_diagnosis
from insightswarm.quality import evaluate_source_trust
from insightswarm.reporting.markdown import latest_qa_report_json, latest_skeptic_review, render_citations_json, render_quality_section, render_report
from insightswarm.reporting.validation import ensure_quality_section, validate_writer_report


class WriterAgent(Agent):
    name = "WriterAgent"
    phase = "Deliver"

    def run(self, context: AgentContext) -> None:
        metadata = context.store.get_run_metadata(context.run_id)
        production_query = bool(metadata.get("query")) and metadata.get("quality_mode", "production") != "test"
        model_provider = getattr(context.model, "provider", "unknown")
        source_trust = evaluate_source_trust(context.store, context.run_id)
        blocked_reason = source_trust.get("blocked_reason")
        base_report = render_report(context.store, context.run_id)
        report = base_report
        writer_status = "template"
        writer_validation = {
            "passed": True,
            "missing_citation_markers": [],
            "missing_sections": [],
            "unsupported_bullets": [],
            "fallback_used": False,
        }
        if production_query and blocked_reason:
            report = ""
            writer_status = "blocked_no_real_evidence"
            writer_validation["passed"] = False
        elif production_query and model_provider == "fake":
            report = ""
            writer_status = "blocked_for_real_model"
            writer_validation["passed"] = False
        elif model_provider != "fake":
            result = context.model.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "Write a concise Chinese competitive research report. Use only provided evidence. "
                            "Call out stale sources, noisy sources, and uncertainty. Preserve citation markers."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "query": metadata.get("query"),
                                "competitor": metadata.get("competitor"),
                                "draft_report": base_report,
                                "citations": [dict(row) for row in context.store.list_citations(context.run_id)],
                                "qa_report": latest_qa_report_json(context.store, context.run_id),
                                "skeptic_review": latest_skeptic_review(context.store, context.run_id),
                            },
                            ensure_ascii=True,
                        ),
                    },
                ],
                metadata={
                    "run_id": context.run_id,
                    "task_id": context.task_id,
                    "role": self.name,
                    "context_artifact_id": context.context_artifact_id,
                    "provider": model_provider,
                },
            )
            candidate_report = result.text or base_report
            validation = validate_writer_report(candidate_report, base_report)
            writer_validation = validation.to_dict()
            writer_validation["fallback_used"] = False
            if validation.passed:
                report = candidate_report
                writer_status = "model_written"
            else:
                report = ensure_quality_section(base_report, render_quality_section(context.store, context.run_id))
                writer_status = "template_fallback_after_validation"
                writer_validation["fallback_used"] = True
                context.store.emit_event(
                    context.run_id,
                    context.task_id,
                    self.name,
                    "writer_citation_repair",
                    "Writer model output failed deterministic citation/section validation; template fallback used.",
                    metadata=writer_validation,
                )
        else:
            context.model.complete(
                [{"role": "system", "content": context.context["role_instructions"]}],
                metadata={
                    "run_id": context.run_id,
                    "task_id": context.task_id,
                    "role": self.name,
                    "context_artifact_id": context.context_artifact_id,
                },
            )
        context.store.emit_event(
            context.run_id,
            context.task_id,
            self.name,
            "writer_quality_status",
            writer_status,
            metadata={
                "model_provider": model_provider,
                "production_query": production_query,
                "writer_validation": writer_validation,
                "source_trust": source_trust,
            },
        )
        report_metadata = {
            "format": "markdown",
            "writer_status": writer_status,
            "formal_result": True,
            "writer_validation_passed": writer_validation["passed"],
            "writer_fallback_used": writer_validation["fallback_used"],
            "missing_citation_markers": writer_validation["missing_citation_markers"],
            "writer_validation": writer_validation,
            "skeptic_review_present": latest_skeptic_review(context.store, context.run_id) is not None,
        }
        if production_query and (model_provider == "fake" or blocked_reason):
            reason = "production query requires a real writer model"
            status = "blocked_for_real_model"
            if blocked_reason:
                reason = "production query has no real document evidence; synthetic fallback is diagnostic-only"
                status = "blocked_no_real_evidence"
            diagnosis = build_run_diagnosis(context.store, context.run_id)
            artifact_id = context.store.write_artifact(
                context.run_id,
                context.task_id,
                "report_blocked",
                "application/json",
                json.dumps(
                    {
                        "status": status,
                        "reason": reason,
                        "blocked_reason": blocked_reason or "real_model_required",
                        "source_trust": source_trust,
                        "source_failures": diagnosis["source_failures"],
                        "formal_evidence_available": diagnosis["formal_evidence_available"],
                        "recommended_next_actions": diagnosis["recommended_next_actions"],
                        "provider": model_provider,
                        "formal_result": False,
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                metadata={**report_metadata, "format": "json", "formal_result": False},
                suffix=".json",
            )
        else:
            artifact_id = context.store.write_artifact(
                context.run_id,
                context.task_id,
                "report",
                "text/markdown",
                report,
                metadata=report_metadata,
                suffix=".md",
            )
        citations_id = context.store.write_artifact(
            context.run_id,
            context.task_id,
            "citations_export",
            "application/json",
            render_citations_json(context.store, context.run_id),
            metadata={"format": "json"},
            suffix=".json",
        )
        qa_id = context.store.write_artifact(
            context.run_id,
            context.task_id,
            "qa_report_export",
            "application/json",
            latest_qa_report_json(context.store, context.run_id),
            metadata={"format": "json"},
            suffix=".json",
        )
        if production_query and (model_provider == "fake" or blocked_reason):
            context.store.set_task_status(
                context.task_id,
                "blocked",
                {
                    "writer_status": writer_status,
                    "production_gate": blocked_reason or "real_model_required",
                    "source_trust": source_trust,
                    "writer_validation": writer_validation,
                },
            )
        else:
            context.store.set_task_status(
                context.task_id,
                "completed",
                {
                    "artifact_id": artifact_id,
                    "citations_artifact_id": citations_id,
                    "qa_report_artifact_id": qa_id,
                    "writer_status": writer_status,
                    "writer_validation_passed": writer_validation["passed"],
                    "writer_fallback_used": writer_validation["fallback_used"],
                    "missing_citation_markers": writer_validation["missing_citation_markers"],
                    "writer_validation": writer_validation,
                },
            )
