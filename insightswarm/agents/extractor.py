from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from insightswarm.agents.base import Agent, AgentContext
from insightswarm.cleaning import DocumentCleaner, trim_quote, query_terms
from insightswarm.schemas.citation import TextSpan
from insightswarm.schemas.knowledge import validate_competitor_knowledge
from insightswarm.harness.gates import find_text_span
from insightswarm.fetching import fetch_source
from insightswarm.tools import ToolContext
from insightswarm.tools.executor import ToolExecutor
import insightswarm.tools.fetch as fetch_tool_module
from insightswarm.util import loads


class ExtractorAgent(Agent):
    name = "ExtractorAgent"
    phase = "Extract"

    def run(self, context: AgentContext) -> None:
        task = context.store.get_task(context.task_id)
        task_metadata = loads(task["metadata_json"], {})
        run_metadata = context.store.get_run_metadata(context.run_id)
        production_query = bool(run_metadata.get("query")) and run_metadata.get("quality_mode", "production") != "test"
        competitor = run_metadata.get("competitor") or "Acme Analytics"
        query = run_metadata.get("query") or competitor
        raw_artifact = self._resolve_input_artifact(context, task_metadata, competitor)
        raw_text = Path(raw_artifact["path"]).read_text(encoding="utf-8", errors="replace")
        cleaned = DocumentCleaner().clean(raw_text, query=query, competitor=competitor)
        cleaned_artifact_id = context.store.write_artifact(
            context.run_id,
            context.task_id,
            "cleaned_document",
            "text/plain",
            cleaned.cleaned_text,
            source_url=raw_artifact["source_url"],
            metadata={
                **cleaned.metadata,
                "raw_artifact_id": raw_artifact["artifact_id"],
                "noise_removed_count": cleaned.noise_removed_count,
                "chunk_count": len(cleaned.chunks),
            },
        )
        if cleaned.chunks:
            context.store.write_artifact(
                context.run_id,
                context.task_id,
                "evidence_chunks",
                "application/json",
                json.dumps(cleaned.to_dict(), ensure_ascii=True, indent=2),
                source_url=raw_artifact["source_url"],
                metadata={"cleaned_artifact_id": cleaned_artifact_id},
                suffix=".json",
            )
        extract_text = cleaned.cleaned_text or raw_text
        result = context.model.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Extract competitive facts as JSON with keys competitor and facts. "
                        "Each fact must include field, value, quote, source_url."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "competitor": competitor,
                            "source_url": raw_artifact["source_url"],
                            "text": extract_text[:6000],
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            metadata={
                "run_id": context.run_id,
                "task_id": context.task_id,
                "role": self.name,
                "context_artifact_id": context.context_artifact_id,
            },
        )
        if result.status != "ok":
            context.store.emit_event(
                context.run_id,
                context.task_id,
                self.name,
                "model_result_degraded",
                "Extractor model call failed; using deterministic fallback.",
                {"provider": result.provider, "model": result.model, "error": result.error},
            )
        structured = _normalize_structured_result(result.json_data) or self._fallback_extract(competitor, raw_artifact["source_url"], extract_text)
        errors = validate_competitor_knowledge(structured)
        if errors:
            context.store.emit_event(
                context.run_id,
                context.task_id,
                self.name,
                "format_repair",
                "Extractor output failed schema validation; using deterministic repair.",
                {"errors": errors},
            )
            structured = self._fallback_extract(competitor, raw_artifact["source_url"], extract_text)
        citation_ids = []
        accepted_facts = []
        discarded_facts = []
        terms = query_terms(query, competitor)
        raw_metadata = loads(raw_artifact["metadata_json"], {})
        diagnostic_only = production_query and raw_metadata.get("fetcher") == "synthetic_fallback"
        for fact in structured["facts"]:
            fact["quote"] = trim_quote(fact["quote"], terms)
            span_text = extract_text
            try:
                span = find_text_span(span_text, fact["quote"])
            except ValueError as exc:
                discarded_facts.append(
                    {
                        "field": fact.get("field"),
                        "value": fact.get("value"),
                        "quote": fact.get("quote"),
                        "reason": str(exc),
                    }
                )
                context.store.emit_event(
                    context.run_id,
                    context.task_id,
                    self.name,
                    "extract_fact_discarded",
                    "Discarded fact because its quote could not be verified in source text.",
                    {"quote": fact.get("quote"), "field": fact.get("field"), "error": str(exc)},
                )
                continue
            if diagnostic_only:
                fact["diagnostic_only"] = True
                fact["citation_skipped_reason"] = "synthetic_fallback_diagnostic_only"
                discarded_facts.append(
                    {
                        "field": fact.get("field"),
                        "value": fact.get("value"),
                        "quote": fact.get("quote"),
                        "reason": "synthetic_fallback_diagnostic_only",
                    }
                )
                context.store.emit_event(
                    context.run_id,
                    context.task_id,
                    self.name,
                    "extract_fact_diagnostic_only",
                    "Synthetic fallback fact was kept diagnostic-only and not cited for a production query.",
                    {"quote": fact.get("quote"), "field": fact.get("field"), "source_url": raw_artifact["source_url"]},
                )
                continue
            citation_id = context.store.create_document_citation(
                context.run_id,
                context.task_id,
                cleaned_artifact_id,
                fact.get("source_url") or raw_artifact["source_url"],
                fact["quote"],
                TextSpan(start=span["start"], end=span["end"]),
                float(fact.get("confidence", 0.86)),
            )
            fact["citation_id"] = citation_id
            accepted_facts.append(fact)
            citation_ids.append(citation_id)
        structured["facts"] = accepted_facts
        artifact_id = context.store.write_artifact(
            context.run_id,
            context.task_id,
            "structured_knowledge",
            "application/json",
            json.dumps(structured, ensure_ascii=True, indent=2),
            metadata={
                "schema": "competitor_knowledge.v1",
                "accepted_fact_count": len(accepted_facts),
                "discarded_fact_count": len(discarded_facts),
                "discarded_facts": discarded_facts,
            },
            suffix=".json",
        )
        context.store.create_message(
            context.run_id,
            context.task_id,
            self.name,
            "StrategicAnalystAgent",
            {
                "intent": "handoff" if citation_ids else "evidence_gap",
                "artifact_id": artifact_id,
                "citation_ids": citation_ids,
                "evidence_status": "citation_ready" if citation_ids else "no_citation_created",
                "goal": "Use extracted formal citations for analysis." if citation_ids else "Recover verifiable source evidence before analysis.",
            },
            f"{context.task_id}:facts-ready",
        )
        context.store.set_task_status(
            context.task_id,
            "completed",
            {
                "artifact_id": artifact_id,
                "citation_ids": citation_ids,
                "source_url": raw_artifact["source_url"],
                "snippet_used": loads(raw_artifact["metadata_json"], {}).get("fetcher") == "tavily_snippet",
                "diagnostic_only": diagnostic_only,
                "accepted_fact_count": len(accepted_facts),
                "discarded_fact_count": len(discarded_facts),
            },
        )

    def _resolve_input_artifact(self, context: AgentContext, task_metadata: dict, competitor: str):
        raw_document_id = task_metadata.get("raw_document_id")
        if raw_document_id:
            raw_artifact = context.store.get_artifact(raw_document_id)
            if raw_artifact["run_id"] != context.run_id or raw_artifact["artifact_type"] != "raw_document":
                raise ValueError(f"raw document not found in run: {raw_document_id}")
            context.store.emit_event(
                context.run_id,
                context.task_id,
                self.name,
                "manual_raw_document_extract_started",
                f"Extracting existing raw document {raw_document_id}",
                {"raw_document_artifact_id": raw_document_id, "source_url": raw_artifact["source_url"]},
            )
            return raw_artifact
        source_url = task_metadata.get("source_url")
        snippet = (task_metadata.get("snippet") or "").strip()
        if source_url and self._snippet_is_sufficient(snippet):
            artifact_id = context.store.write_artifact(
                context.run_id,
                context.task_id,
                "raw_document",
                "text/plain",
                snippet,
                source_url=source_url,
                metadata={
                    "fetcher": "tavily_snippet",
                    "status": "ok",
                    "competitor": competitor,
                    "title": task_metadata.get("title"),
                    "search_result_rank": task_metadata.get("search_result_rank"),
                    "link_gate_score": task_metadata.get("link_gate_score"),
                },
            )
            context.store.emit_event(
                context.run_id,
                context.task_id,
                self.name,
                "snippet_first_used",
                f"Used Tavily snippet for {source_url}",
                {"artifact_id": artifact_id, "source_url": source_url},
            )
            return context.store.get_artifact(artifact_id)
        if source_url:
            run_metadata = context.store.get_run_metadata(context.run_id)
            fetch_tool_module.fetch_source = fetch_source
            result, tool_call_id = ToolExecutor(context.store).run(
                "fetch.url",
                {"url": source_url},
                ToolContext(context.run_id, context.task_id, run_metadata.get("quality_mode", "production"), {**run_metadata, "agent_name": self.name}),
            )
            data = result.data
            if result.ok:
                artifact_id = context.store.write_artifact(
                    context.run_id,
                    context.task_id,
                    "raw_document",
                    "text/plain",
                    data.get("text") or data.get("html") or "",
                    source_url=source_url,
                    metadata={
                        "fetcher": data.get("fetcher"),
                        "status": data.get("status"),
                        "competitor": competitor,
                        "fallback_reason": data.get("fallback_reason"),
                        "fetch_attempts": (data.get("metadata") or {}).get("attempts", []),
                        "tool": "fetch.url",
                        "tool_status": result.status,
                        "tool_call_id": tool_call_id,
                    },
                )
                return context.store.get_artifact(artifact_id)
            failure_id = context.store.write_artifact(
                context.run_id,
                context.task_id,
                "fetch_failure",
                "application/json",
                json.dumps(
                    {
                        "source_url": source_url,
                        "fetcher": data.get("fetcher"),
                        "status": data.get("status") or result.status,
                        "error": result.error,
                        "fallback_reason": data.get("fallback_reason"),
                        "metadata": data.get("metadata") or result.diagnostics,
                        "tool": "fetch.url",
                        "tool_status": result.status,
                        "tool_call_id": tool_call_id,
                    },
                    ensure_ascii=True,
                    indent=2,
                ),
                source_url=source_url,
                metadata={
                    "fetcher": data.get("fetcher") or "fetch.url",
                    "error": result.error,
                    "fallback_reason": data.get("fallback_reason"),
                    "tool": "fetch.url",
                    "tool_status": result.status,
                    "tool_call_id": tool_call_id,
                },
                suffix=".json",
            )
            context.store.emit_event(
                context.run_id,
                context.task_id,
                self.name,
                "fetch_failure",
                f"Failed dynamic fetch for {source_url}",
                {"artifact_id": failure_id, "error": result.error},
            )
            artifact_id = context.store.write_artifact(
                context.run_id,
                context.task_id,
                "raw_document",
                "text/plain",
                snippet or f"{competitor} source unavailable for {source_url}.",
                source_url=source_url,
                metadata={
                    "fetcher": "synthetic_fallback",
                    "status": "ok",
                    "competitor": competitor,
                    "fallback_reason": "dynamic_fetch_failed",
                    "failure_artifact_id": failure_id,
                },
            )
            return context.store.get_artifact(artifact_id)
        raw_artifacts = [
            row
            for row in context.store.list_artifacts(context.run_id)
            if row["artifact_type"] == "raw_document"
        ]
        return raw_artifacts[-1]

    def _snippet_is_sufficient(self, snippet: str) -> bool:
        if len(snippet) >= 120:
            return True
        signals = ("$", "¥", "￥", "元", "pricing", "price", "plan", "feature", "laptop", "thinkpad", "legion", "yoga")
        return len(snippet) >= 60 and any(signal.lower() in snippet.lower() for signal in signals)

    def _fallback_extract(self, competitor: str, source_url: str, raw_text: str) -> dict:
        sentences = [part.strip() for part in raw_text.replace("\n", " ").split(".") if part.strip()]
        quote = next((sentence for sentence in sentences if "$" in sentence), sentences[0] if sentences else raw_text[:200])
        value = quote
        if "$" in quote:
            value = quote[quote.find("$") :].strip()
        return {
            "competitor": competitor,
            "facts": [
                {
                    "field": "pricing_or_positioning",
                    "value": value,
                    "quote": quote,
                    "source_url": source_url,
                    "confidence": 0.82,
                }
            ],
        }


def _normalize_structured_result(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and ("facts" in item or "competitor" in item):
                return item
    return None
