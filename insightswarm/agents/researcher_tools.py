from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from insightswarm.agents.agent_loop import AgentLoopState, run_agent_loop
from insightswarm.agents.tool_executor import ToolExecutor
from insightswarm.extraction_batches import create_extraction_batch
from insightswarm.schemas.swarm import Task
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore
from insightswarm.tools.core import ToolContext
from insightswarm.tools.fetch import FetchUrlTool
from insightswarm.tools.firecrawl import FirecrawlScrapeTool
from insightswarm.tools.search import SearchTool
from insightswarm.util import new_id


RESEARCHER_ROLE = "researcher"
EXTRACTOR_ROLE = "extractor"
BROWSER_ROLE = "browser_agent"


RESEARCH_SUBAGENT_TOOLS = [
    {
        "name": "search_web",
        "description": "Search privately for sources that may answer this scoped subtask. Does not write shared storage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "source_goal": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        "output_schema": {"type": "object", "properties": {"candidates": {"type": "array"}}},
        "side_effects": "private subagent memory only",
    },
    {
        "name": "fetch_source",
        "description": "Fetch one candidate URL privately and classify whether its text is useful for this subtask.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "why_this_source": {"type": "string"},
            },
            "required": ["url"],
        },
        "output_schema": {"type": "object", "properties": {"document": {"type": "object"}}},
        "side_effects": "private subagent memory only",
    },
    {
        "name": "finish_subagent",
        "description": "Return the private subagent finding to the parent Researcher. Use blocked if no useful path remains.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["complete", "blocked"]},
                "summary": {"type": "string"},
                "candidate_urls": {"type": "array", "items": {"type": "string"}},
                "recommended_next_step": {"type": "string"},
            },
            "required": ["status", "summary"],
        },
        "output_schema": {"type": "object", "properties": {"finding": {"type": "object"}}},
        "side_effects": "returns to parent Researcher only",
    },
]


RESEARCHER_TOOLS = [
    {
        "name": "read_task",
        "description": "Read the assigned research sub-question, objective, scope, and known constraints.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "objective": {"type": "string"},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "task_id": {"type": "string"},
            },
        },
        "side_effects": "none",
    },
    {
        "name": "read_shared_memory",
        "description": "Read scoped shared work memory written by other agents, including observations, hypotheses, suggestions, conflicts, plans, and recent messages.",
        "input_schema": {
            "type": "object",
            "properties": {"focus": {"type": "string", "description": "Optional focus such as source_quality, open_gaps, browser_suggestions, or prior_findings."}},
            "required": [],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "observations": {"type": "array"},
                "hypotheses": {"type": "array"},
                "suggestions": {"type": "array"},
                "conflicts": {"type": "array"},
                "plans": {"type": "array"},
            },
        },
        "side_effects": "none",
    },
    {
        "name": "search_web",
        "description": "Search the web for candidate source URLs relevant to the research sub-question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "source_goal": {"type": "string", "description": "Desired source type, e.g. official_docs, primary_source, pricing_page, changelog, technical_analysis."},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
        "output_schema": {"type": "object", "properties": {"candidates": {"type": "array"}}},
        "side_effects": "records search attempt locally",
    },
    {
        "name": "fetch_source",
        "description": "Fetch one candidate URL and classify whether it produced usable raw text.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}, "why_this_source": {"type": "string"}},
            "required": ["url"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "document": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "title": {"type": "string"},
                        "usable": {"type": "boolean"},
                        "usability_reason": {"type": "string"},
                        "page_type": {"type": "string"},
                        "information_density": {"type": "string"},
                        "text_preview": {"type": "string"},
                    },
                }
            },
        },
        "side_effects": "stores fetched raw document in private working state only",
    },
    {
        "name": "firecrawl_source",
        "description": "Use Firecrawl to acquire cleaner main text for a URL when static fetch fails, returns low-signal text, or the page likely needs stronger extraction. This is more expensive than fetch_source and still only stores the document privately.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "why_firecrawl_needed": {"type": "string"},
                "extract_goal": {"type": "string"},
            },
            "required": ["url", "why_firecrawl_needed"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "document": {"type": "object"},
                "unresolved_publishable_count": {"type": "integer"},
            },
        },
        "side_effects": "stores Firecrawl-acquired raw document in private working state only",
    },
    {
        "name": "rank_sources",
        "description": "Privately rank candidate and fetched sources by relevance, authority, freshness hints, fetch risk, uniqueness, and current source decisions. Use before deciding what to fetch, publish, defer, or reject.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "include_fetched": {"type": "boolean"},
            },
            "required": ["goal"],
        },
        "output_schema": {"type": "object", "properties": {"ranked_sources": {"type": "array"}}},
        "side_effects": "none",
    },
    {
        "name": "spawn_research_subagents",
        "description": (
            "Privately run 1-3 temporary scoped research subagents in parallel when the task is broad, branchy, or a repair needs several source paths checked. "
            "Subagents have independent context and limited search/fetch tools, cannot write shared storage, cannot create tasks/messages/artifacts, and cannot spawn subagents. "
            "Use their findings to decide your next Researcher tool call; publish/write/suggest actions remain your responsibility."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "why_parallel_needed": {"type": "string"},
                "subtasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "search_goal": {"type": "string"},
                            "constraints": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["question"],
                    },
                },
            },
            "required": ["why_parallel_needed", "subtasks"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "findings": {"type": "array"},
                "candidate_sources": {"type": "array"},
                "recommended_next_steps": {"type": "array"},
            },
        },
        "side_effects": "private Researcher memory only",
    },
    {
        "name": "publish_raw_source",
        "description": "Publish one or more fetched usable raw documents to shared storage so Extractor can create citations.",
        "input_schema": {
            "type": "object",
            "properties": {"document_urls": {"type": "array", "items": {"type": "string"}}, "why_ready": {"type": "string"}},
            "required": ["document_urls", "why_ready"],
        },
        "output_schema": {"type": "object", "properties": {"artifact_ids": {"type": "array"}, "extractor_task_ids": {"type": "array"}}},
        "side_effects": "writes raw_document artifacts and extractor tasks",
    },
    {
        "name": "defer_source",
        "description": "Privately keep a fetched usable source for later comparison instead of publishing it immediately. Deferred sources must be published or rejected before finishing research as complete.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["url", "reason"],
        },
        "output_schema": {"type": "object", "properties": {"deferred_url": {"type": "string"}, "unresolved_publishable_count": {"type": "integer"}}},
        "side_effects": "updates private source decision state only",
    },
    {
        "name": "reject_source",
        "description": "Privately reject a fetched source that is usable but not worth publishing, such as a duplicate, stale, low-authority, or off-target source.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["url", "reason"],
        },
        "output_schema": {"type": "object", "properties": {"rejected_url": {"type": "string"}, "unresolved_publishable_count": {"type": "integer"}}},
        "side_effects": "updates private source decision state only",
    },
    {
        "name": "suggest_browser_acquisition",
        "description": (
            "Create a BrowserAgent hard_acquisition task for a source or source class that static fetch/Firecrawl cannot acquire reliably. "
            "Use this when acquisition_pressure recommends browser_agent, or explicitly explain why not in failure_reflection."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_url": {"type": "string"},
                "goal": {"type": "string"},
                "why_browser_needed": {"type": "string"},
                "failed_attempts": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["goal", "why_browser_needed"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "suggestion_message_id": {"type": "string"},
                "browser_task_id": {"type": "string"},
                "deduped": {"type": "boolean"},
            },
        },
        "side_effects": "writes a suggestion message to shared memory",
    },
    {
        "name": "write_observation",
        "description": "Write a concise reusable observation to shared memory. Use for facts about acquisition attempts, source behavior, or research state that other agents may benefit from.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}, "basis": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["summary"],
        },
        "output_schema": {"type": "object", "properties": {"message_id": {"type": "string"}}},
        "side_effects": "writes observation message",
    },
    {
        "name": "write_hypothesis",
        "description": "Write a tentative, testable hypothesis to shared memory. Use for claims that may guide further research but are not formal evidence.",
        "input_schema": {
            "type": "object",
            "properties": {"claim": {"type": "string"}, "basis": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["claim"],
        },
        "output_schema": {"type": "object", "properties": {"board_item_id": {"type": "string"}}},
        "side_effects": "writes proposed hypothesis claim",
    },
    {
        "name": "write_suggestion",
        "description": "Write an actionable suggestion for another agent or future step.",
        "input_schema": {
            "type": "object",
            "properties": {"target_role": {"type": "string"}, "suggestion": {"type": "string"}, "reason": {"type": "string"}, "confidence": {"type": "number"}},
            "required": ["suggestion"],
        },
        "output_schema": {"type": "object", "properties": {"message_id": {"type": "string"}}},
        "side_effects": "writes suggestion message",
    },
    {
        "name": "finish_research",
        "description": "Stop this Researcher loop when enough raw source material has been published, the path is blocked, or no productive tool call remains.",
        "input_schema": {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["complete", "blocked"]}, "reason": {"type": "string"}},
            "required": ["status", "reason"],
        },
        "output_schema": {"type": "object", "properties": {"terminal": {"type": "boolean"}}},
        "side_effects": "marks worker path terminal",
    },
]


@dataclass
class ResearcherToolState:
    task_context: dict[str, Any] | None = None
    candidate_sources: list[dict[str, Any]] = field(default_factory=list)
    fetched_documents: list[dict[str, Any]] = field(default_factory=list)
    acquisition_failures: list[dict[str, Any]] = field(default_factory=list)
    subagent_findings: list[dict[str, Any]] = field(default_factory=list)
    created_task_ids: list[str] = field(default_factory=list)
    created_message_ids: list[str] = field(default_factory=list)
    created_artifact_ids: list[str] = field(default_factory=list)
    terminal_status: str | None = None
    terminal_reason: str | None = None


class ResearcherToolHandlers:
    def __init__(
        self,
        *,
        task: Task,
        task_store: TaskStore,
        mailbox: Mailbox,
        artifact_store: ArtifactStore,
        board_store: BoardStore,
        state: ResearcherToolState,
        model_client: object | None = None,
    ):
        self.task = task
        self.task_store = task_store
        self.mailbox = mailbox
        self.artifact_store = artifact_store
        self.board_store = board_store
        self.state = state
        self.model_client = model_client

    def handlers(self) -> dict[str, Any]:
        return {
            "read_task": self.read_task,
            "read_shared_memory": self.read_shared_memory,
            "search_web": self.search_web,
            "fetch_source": self.fetch_source,
            "firecrawl_source": self.firecrawl_source,
            "rank_sources": self.rank_sources,
            "spawn_research_subagents": self.spawn_research_subagents,
            "publish_raw_source": self.publish_raw_source,
            "defer_source": self.defer_source,
            "reject_source": self.reject_source,
            "suggest_browser_acquisition": self.suggest_browser_acquisition,
            "write_observation": self.write_observation,
            "write_hypothesis": self.write_hypothesis,
            "write_suggestion": self.write_suggestion,
            "finish_research": self.finish_research,
        }

    def read_task(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        del tool_input
        question = _task_question(self.task)
        snapshot = self.board_store.scoped_snapshot(self.task.run_id, question_text=question)
        context = {
            "question": question,
            "objective": question,
            "constraints": list(self.task.inputs.get("constraints") or []),
            "task_id": self.task.task_id,
            "task_kind": self.task.kind,
            "board_summary": _summarize_board_snapshot(snapshot),
            "user_inputs": list(self.task.inputs.get("user_inputs") or []),
        }
        self.state.task_context = context
        return {"ok": True, **context}

    def read_shared_memory(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        del tool_input
        question = _task_question(self.task)
        snapshot = self.board_store.scoped_snapshot(self.task.run_id, question_text=question)
        messages = [
            {
                "message_id": message.message_id,
                "from_role": message.from_role,
                "type": message.type,
                "payload": message.payload,
                "related_task_id": message.related_task_id,
            }
            for message in self.mailbox.inbox(self.task.run_id, role=RESEARCHER_ROLE)
            if message.related_task_id in {None, self.task.task_id}
        ][-10:]
        board = _summarize_board_snapshot(snapshot)
        return {
            "ok": True,
            "observations": [message for message in messages if message["type"] == "observation"],
            "hypotheses": board.get("claim", []),
            "suggestions": [message for message in messages if message["type"] == "suggestion"],
            "conflicts": board.get("conflict", []),
            "plans": board.get("plan", []),
            "local_candidates": list(self.state.candidate_sources[-8:]),
            "subagent_findings": list(self.state.subagent_findings[-6:]),
            "acquisition_pressure": _acquisition_pressure(self.state, self.task),
        }

    def search_web(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        if self.state.task_context is None:
            return {"ok": False, "error": "read_task first"}
        query = _safe_text(tool_input.get("query")) or _task_question(self.task)
        limit = int(tool_input.get("limit") or 5)
        result = SearchTool().run({"query": query, "limit": limit}, ToolContext(run_id=self.task.run_id, task_id=self.task.task_id))
        raw_results = list(result.data.get("results") or []) if result.ok else []
        candidates = [_candidate_summary(item) for item in raw_results]
        self.state.candidate_sources.extend(candidates)
        return {"ok": result.ok, "query": query, "source_goal": tool_input.get("source_goal"), "candidates": candidates, "diagnostics": result.diagnostics, "error": result.error}

    def fetch_source(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        if self.state.task_context is None:
            return {"ok": False, "error": "read_task first"}
        url = _safe_text(tool_input.get("url"))
        if not url:
            return {"ok": False, "error": "fetch_source requires url"}
        result = FetchUrlTool().run({"url": url}, ToolContext(run_id=self.task.run_id, task_id=self.task.task_id))
        if not result.ok:
            document = {"url": url, "usable": False, "usability_reason": result.error or "fetch_failed"}
            self._record_acquisition_failure(url=url, tool="fetch_source", reason=document["usability_reason"], document=document)
            return {"ok": False, "error": result.error, "document": document, "acquisition_pressure": _acquisition_pressure(self.state, self.task)}
        document = dict(result.data)
        page_profile = _classify_page(document, url)
        visible_document = {
            "url": url,
            "title": _safe_text(document.get("title")),
            "usable": bool(page_profile["likely_extractable"]),
            "usability_reason": page_profile["reason"],
            "page_type": page_profile["page_type"],
            "information_density": page_profile["estimated_information_density"],
            "text_preview": _safe_text(document.get("text"))[:900],
        }
        document_record = {
            **document,
            **visible_document,
            "page_profile": page_profile,
            "researcher_status": "pending" if visible_document["usable"] else "rejected",
            "decision_reason": "" if visible_document["usable"] else visible_document["usability_reason"],
        }
        self.state.fetched_documents.append(document_record)
        if not visible_document["usable"]:
            self._record_acquisition_failure(url=url, tool="fetch_source", reason=visible_document["usability_reason"], document=visible_document)
        return {
            "ok": True,
            "document": visible_document,
            "unresolved_publishable_count": len(_unresolved_publishable_documents(self.state.fetched_documents)),
            "acquisition_pressure": _acquisition_pressure(self.state, self.task),
        }

    def firecrawl_source(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        if self.state.task_context is None:
            return {"ok": False, "error": "read_task first"}
        url = _safe_text(tool_input.get("url"))
        if not url:
            return {"ok": False, "error": "firecrawl_source requires url"}
        result = FirecrawlScrapeTool().run({"url": url}, ToolContext(run_id=self.task.run_id, task_id=self.task.task_id))
        if not result.ok:
            document = {"url": url, "usable": False, "usability_reason": result.error or "firecrawl_failed"}
            self._record_acquisition_failure(url=url, tool="firecrawl_source", reason=document["usability_reason"], document=document)
            return {"ok": False, "error": result.error, "document": document, "acquisition_pressure": _acquisition_pressure(self.state, self.task)}
        document = dict(result.data)
        page_profile = _classify_page(document, url)
        visible_document = {
            "url": url,
            "title": _safe_text(document.get("title")),
            "usable": bool(page_profile["likely_extractable"]),
            "usability_reason": page_profile["reason"],
            "page_type": page_profile["page_type"],
            "information_density": page_profile["estimated_information_density"],
            "text_preview": _safe_text(document.get("text"))[:900],
            "fetcher": "firecrawl",
        }
        document_record = {
            **document,
            **visible_document,
            "page_profile": page_profile,
            "researcher_status": "pending" if visible_document["usable"] else "rejected",
            "decision_reason": _safe_text(tool_input.get("why_firecrawl_needed")),
            "extract_goal": _safe_text(tool_input.get("extract_goal")),
        }
        self.state.fetched_documents.append(document_record)
        if not visible_document["usable"]:
            self._record_acquisition_failure(url=url, tool="firecrawl_source", reason=visible_document["usability_reason"], document=visible_document)
        return {
            "ok": True,
            "document": visible_document,
            "unresolved_publishable_count": len(_unresolved_publishable_documents(self.state.fetched_documents)),
            "acquisition_pressure": _acquisition_pressure(self.state, self.task),
        }

    def rank_sources(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        goal = _safe_text(tool_input.get("goal")) or _task_question(self.task)
        include_fetched = bool(tool_input.get("include_fetched", True))
        sources: list[dict[str, Any]] = []
        for candidate in self.state.candidate_sources:
            sources.append({**candidate, "origin": "candidate"})
        if include_fetched:
            for document in self.state.fetched_documents:
                sources.append(
                    {
                        "url": _document_url(document),
                        "title": _safe_text(document.get("title")),
                        "snippet": _safe_text(document.get("text"))[:500],
                        "source_category": _source_category(_document_url(document), _safe_text(document.get("title")), _safe_text(document.get("text"))[:500]),
                        "estimated_fetch_risk": "low" if document.get("usable") else "high",
                        "content_format_hint": _format_hint(_document_url(document), _safe_text(document.get("title")), _safe_text(document.get("text"))[:500]),
                        "origin": "fetched",
                        "researcher_status": document.get("researcher_status"),
                        "fetcher": document.get("fetcher"),
                    }
                )
        ranked = sorted((_rank_source(source, goal) for source in sources), key=lambda item: item["score"], reverse=True)
        return {"ok": True, "ranked_sources": ranked[:12], "acquisition_pressure": _acquisition_pressure(self.state, self.task)}

    def spawn_research_subagents(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        if self.state.task_context is None:
            return {"ok": False, "error": "read_task first"}
        raw_subtasks = list(tool_input.get("subtasks") or [])
        subtasks = [_normalize_subagent_task(item, index=index) for index, item in enumerate(raw_subtasks) if isinstance(item, dict)]
        subtasks = [item for item in subtasks if item["question"]][:3]
        if not subtasks:
            return {"ok": False, "error": "spawn_research_subagents requires 1-3 subtasks with question"}

        findings: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=min(3, len(subtasks))) as executor:
            futures = [
                executor.submit(
                    _run_research_subagent,
                    parent_task=self.task,
                    subtask=subtask,
                    model_client=self.model_client,
                )
                for subtask in subtasks
            ]
            for future in as_completed(futures):
                findings.append(future.result())

        findings.sort(key=lambda item: int(item.get("index") or 0))
        self.state.subagent_findings.extend(findings)
        candidate_sources = [
            _candidate_summary({"url": url, "title": "", "snippet": finding.get("summary", "")})
            for finding in findings
            for url in list(finding.get("candidate_urls") or [])[:5]
        ]
        self.state.candidate_sources.extend(candidate_sources)
        return {
            "ok": True,
            "why_parallel_needed": _safe_text(tool_input.get("why_parallel_needed")),
            "findings": findings,
            "candidate_sources": candidate_sources[:12],
            "recommended_next_steps": [_safe_text(item.get("recommended_next_step")) for item in findings if _safe_text(item.get("recommended_next_step"))],
            "shared_storage_written": False,
        }

    def publish_raw_source(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        if self.state.task_context is None:
            return {"ok": False, "error": "read_task first"}
        requested_urls = {_safe_text(url) for url in list(tool_input.get("document_urls") or [])}
        documents = _select_publishable_documents(self.state.fetched_documents, requested_urls)
        if not documents:
            return {"ok": False, "error": "no usable fetched document is available"}
        artifact_ids: list[str] = []
        task_ids: list[str] = []
        batch_id = new_id("batch")
        issue_key = _safe_text(self.task.inputs.get("issue_key"))
        for document in documents:
            artifact = self.artifact_store.write_raw_document(
                self.task.run_id,
                source_task_id=self.task.task_id,
                document={
                    "source_url": document.get("source_url") or document.get("url"),
                    "title": document.get("title"),
                    "text": document.get("text"),
                    "html": document.get("html"),
                    "metadata": {
                        "produced_by": RESEARCHER_ROLE,
                        "publish_reason": _safe_text(tool_input.get("why_ready")),
                        "page_profile": document.get("page_profile"),
                        "issue_key": issue_key,
                        "repair_attempt": self.task.inputs.get("repair_attempt"),
                        "batch_id": batch_id,
                    },
                    "batch_id": batch_id,
                },
                summary=_safe_text(document.get("title")) or _safe_text(document.get("source_url")) or "raw document",
            )
            extractor_task = self.task_store.create(
                self.task.run_id,
                kind="raw_document",
                status="pending",
                owner_role=EXTRACTOR_ROLE,
                inputs={
                    "artifact_id": artifact.artifact_id,
                    "source_task_id": self.task.task_id,
                    "issue_key": issue_key,
                    "repair_attempt": self.task.inputs.get("repair_attempt"),
                    "batch_id": batch_id,
                },
                depends_on=[],
                priority=self.task.priority,
                created_by=RESEARCHER_ROLE,
            )
            message = self.mailbox.send(
                self.task.run_id,
                from_role=RESEARCHER_ROLE,
                to_role=EXTRACTOR_ROLE,
                message_type="request",
                payload={"kind": "extract_evidence", "artifact_id": artifact.artifact_id, "batch_id": batch_id},
                related_task_id=extractor_task.task_id,
            )
            artifact_ids.append(artifact.artifact_id or "")
            task_ids.append(extractor_task.task_id or "")
            self.state.created_message_ids.append(message.message_id or "")
            document["researcher_status"] = "published"
            document["decision_reason"] = _safe_text(tool_input.get("why_ready"))
            _close_same_url_documents(
                self.state.fetched_documents,
                document,
                status="published_duplicate",
                reason="duplicate of published source",
            )
        create_extraction_batch(
            board_store=self.board_store,
            run_id=self.task.run_id,
            batch_id=batch_id,
            source_task_id=self.task.task_id or "",
            raw_artifact_ids=artifact_ids,
            extractor_task_ids=task_ids,
            purpose=_safe_text(tool_input.get("why_ready")),
            issue_key=issue_key,
            priority=self.task.priority,
        )
        self.state.created_artifact_ids.extend(artifact_ids)
        self.state.created_task_ids.extend(task_ids)
        return {
            "ok": True,
            "batch_id": batch_id,
            "artifact_ids": artifact_ids,
            "extractor_task_ids": task_ids,
            "unresolved_publishable_count": len(_unresolved_publishable_documents(self.state.fetched_documents)),
            "acquisition_pressure": _acquisition_pressure(self.state, self.task),
        }

    def defer_source(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        documents = _matching_decidable_documents(self.state.fetched_documents, _safe_text(tool_input.get("url")))
        if not documents:
            return {"ok": False, "error": "source has not been fetched"}
        if not any(bool(document.get("usable")) for document in documents):
            return {"ok": False, "error": "only usable fetched sources can be deferred"}
        reason = _safe_text(tool_input.get("reason"))
        for document in documents:
            document["researcher_status"] = "deferred"
            document["decision_reason"] = reason
        return {
            "ok": True,
            "deferred_url": _document_url(documents[0]),
            "affected_documents": len(documents),
            "unresolved_publishable_count": len(_unresolved_publishable_documents(self.state.fetched_documents)),
        }

    def reject_source(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        documents = _matching_decidable_documents(self.state.fetched_documents, _safe_text(tool_input.get("url")))
        if not documents:
            return {"ok": False, "error": "source has not been fetched"}
        reason = _safe_text(tool_input.get("reason"))
        for document in documents:
            document["researcher_status"] = "rejected"
            document["decision_reason"] = reason
        return {
            "ok": True,
            "rejected_url": _document_url(documents[0]),
            "affected_documents": len(documents),
            "unresolved_publishable_count": len(_unresolved_publishable_documents(self.state.fetched_documents)),
        }

    def suggest_browser_acquisition(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        goal = _safe_text(tool_input.get("goal"))
        pressure = _acquisition_pressure(self.state, self.task)
        target_url = _safe_text(tool_input.get("target_url")) or _safe_text(pressure.get("latest_failed_url"))
        issue_key = _safe_text(self.task.inputs.get("issue_key")) or _stable_browser_issue_key(goal=goal, target_url=target_url, reason=_safe_text(tool_input.get("why_browser_needed")))
        failed_attempts = list(tool_input.get("failed_attempts") or pressure.get("failed_attempts") or [])
        existing_browser_task = self._active_browser_task(issue_key=issue_key, target_url=target_url)
        if existing_browser_task is not None:
            message = self.mailbox.send(
                self.task.run_id,
                from_role=RESEARCHER_ROLE,
                to_role=BROWSER_ROLE,
                message_type="observation",
                payload={"kind": "progress_update", "status": "browser_already_requested", "task_id": existing_browser_task.task_id, "issue_key": issue_key, "goal": goal, "target_url": target_url},
                related_task_id=existing_browser_task.task_id,
            )
            self.state.created_message_ids.append(message.message_id or "")
            return {"ok": True, "suggestion_message_id": message.message_id, "browser_task_id": existing_browser_task.task_id, "deduped": True}

        browser_task = self.task_store.create(
            self.task.run_id,
            kind="hard_acquisition",
            status="pending",
            owner_role=BROWSER_ROLE,
            inputs={
                "goal": goal,
                "target_url": target_url,
                "reason": _safe_text(tool_input.get("why_browser_needed")),
                "issue_key": issue_key,
                "failed_attempts": failed_attempts,
            },
            depends_on=[],
            priority=self.task.priority,
            created_by=RESEARCHER_ROLE,
        )
        message = self.mailbox.send(
            self.task.run_id,
            from_role=RESEARCHER_ROLE,
            to_role=BROWSER_ROLE,
            message_type="request",
            payload={"kind": "hard_acquisition", "goal": goal, "target_url": target_url, "issue_key": issue_key, "failed_attempts": failed_attempts},
            related_task_id=browser_task.task_id,
        )
        self.state.created_task_ids.append(browser_task.task_id or "")
        self.state.created_message_ids.append(message.message_id or "")
        return {"ok": True, "suggestion_message_id": message.message_id, "browser_task_id": browser_task.task_id, "deduped": False}

    def write_observation(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        message = self.mailbox.send(
            self.task.run_id,
            from_role=RESEARCHER_ROLE,
            to_role="lead",
            message_type="observation",
            payload={"kind": "progress_update", "summary": _safe_text(tool_input.get("summary")), "basis": _safe_text(tool_input.get("basis")), "confidence": tool_input.get("confidence")},
            related_task_id=self.task.task_id,
        )
        self.state.created_message_ids.append(message.message_id or "")
        return {"ok": True, "message_id": message.message_id}

    def write_hypothesis(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        claim = _safe_text(tool_input.get("claim"))
        item = self.board_store.create_claim(
            self.task.run_id,
            title=claim,
            question_id=None,
            claim_type="candidate",
            status="proposed",
            created_by=RESEARCHER_ROLE,
            payload={"basis": _safe_text(tool_input.get("basis")), "confidence": tool_input.get("confidence")},
            source_task_id=self.task.task_id,
            dedupe_key=f"hypothesis:{self.task.task_id}:{claim.lower()}",
        )
        return {"ok": True, "board_item_id": item.item_id}

    def write_suggestion(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        message = self.mailbox.send(
            self.task.run_id,
            from_role=RESEARCHER_ROLE,
            to_role=_safe_text(tool_input.get("target_role")) or "lead",
            message_type="suggestion",
            payload={"kind": "research_more", "suggestion": _safe_text(tool_input.get("suggestion")), "reason": _safe_text(tool_input.get("reason")), "confidence": tool_input.get("confidence")},
            related_task_id=self.task.task_id,
        )
        self.state.created_message_ids.append(message.message_id or "")
        return {"ok": True, "message_id": message.message_id}

    def finish_research(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        status = _safe_text(tool_input.get("status")) or "blocked"
        if status == "complete":
            status = "done"
        if status not in {"done", "blocked"}:
            status = "blocked"
        reason = _safe_text(tool_input.get("reason")) or status
        unresolved = _unresolved_publishable_documents(self.state.fetched_documents)
        if status == "done" and unresolved:
            return {
                "ok": False,
                "error": "publish or reject unresolved usable sources before finishing complete",
                "unresolved_publishable_sources": [_document_summary(document) for document in unresolved],
                "allowed_next_tools": ["publish_raw_source", "reject_source"],
            }
        message = self.mailbox.send(
            self.task.run_id,
            from_role=RESEARCHER_ROLE,
            to_role="lead",
            message_type="response",
            payload={"kind": "completed" if status == "done" else "blocked", "reason": reason},
            related_task_id=self.task.task_id,
        )
        self.state.created_message_ids.append(message.message_id or "")
        self.state.terminal_status = status
        self.state.terminal_reason = reason
        return {"ok": True, "terminal": True, "status": status, "reason": reason}

    def _record_acquisition_failure(self, *, url: str, tool: str, reason: str, document: dict[str, Any] | None = None) -> None:
        failure = {
            "url": url,
            "tool": tool,
            "reason": reason,
            "domain": urlparse(url).netloc.lower(),
            "failure_kind": _failure_kind(reason, document=document or {}),
            "page_type": (document or {}).get("page_type"),
            "information_density": (document or {}).get("information_density"),
        }
        self.state.acquisition_failures.append(failure)

    def _active_browser_task(self, *, issue_key: str, target_url: str) -> Task | None:
        for task in self.task_store.store.list_swarm_tasks(self.task.run_id):
            if task.owner_role != BROWSER_ROLE or task.kind != "hard_acquisition" or task.status not in {"pending", "leased"}:
                continue
            task_issue_key = _safe_text(task.inputs.get("issue_key"))
            task_target_url = _safe_text(task.inputs.get("target_url"))
            if issue_key and task_issue_key == issue_key:
                return task
            if target_url and task_target_url == target_url:
                return task
        return None


@dataclass
class _ResearchSubagentState:
    subtask: dict[str, Any]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    fetched_documents: list[dict[str, Any]] = field(default_factory=list)
    finding: dict[str, Any] | None = None


class _ResearchSubagentHandlers:
    def __init__(self, *, parent_task: Task, state: _ResearchSubagentState):
        self.parent_task = parent_task
        self.state = state

    def handlers(self) -> dict[str, Any]:
        return {
            "search_web": self.search_web,
            "fetch_source": self.fetch_source,
            "finish_subagent": self.finish_subagent,
        }

    def search_web(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        query = _safe_text(tool_input.get("query")) or _safe_text(self.state.subtask.get("question"))
        limit = min(max(int(tool_input.get("limit") or 5), 1), 6)
        result = SearchTool().run({"query": query, "limit": limit}, ToolContext(run_id=self.parent_task.run_id, task_id=self.parent_task.task_id))
        raw_results = list(result.data.get("results") or []) if result.ok else []
        candidates = [_candidate_summary(item) for item in raw_results]
        self.state.candidates.extend(candidates)
        return {"ok": result.ok, "query": query, "source_goal": tool_input.get("source_goal"), "candidates": candidates, "diagnostics": result.diagnostics, "error": result.error}

    def fetch_source(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        url = _safe_text(tool_input.get("url"))
        if not url:
            return {"ok": False, "error": "fetch_source requires url"}
        result = FetchUrlTool().run({"url": url}, ToolContext(run_id=self.parent_task.run_id, task_id=self.parent_task.task_id))
        if not result.ok:
            document = {"url": url, "usable": False, "usability_reason": result.error or "fetch_failed"}
            self.state.fetched_documents.append(document)
            return {"ok": False, "error": result.error, "document": document}
        document = dict(result.data)
        page_profile = _classify_page(document, url)
        visible_document = {
            "url": url,
            "title": _safe_text(document.get("title")),
            "usable": bool(page_profile["likely_extractable"]),
            "usability_reason": page_profile["reason"],
            "page_type": page_profile["page_type"],
            "information_density": page_profile["estimated_information_density"],
            "text_preview": _safe_text(document.get("text"))[:700],
        }
        self.state.fetched_documents.append({**document, **visible_document, "page_profile": page_profile})
        return {"ok": True, "document": visible_document}

    def finish_subagent(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        status = _safe_text(tool_input.get("status")) or "blocked"
        if status not in {"complete", "blocked"}:
            status = "blocked"
        candidate_urls = [_safe_text(url) for url in list(tool_input.get("candidate_urls") or []) if _safe_text(url)]
        if not candidate_urls:
            candidate_urls = [_safe_text(item.get("url")) for item in self.state.candidates if _safe_text(item.get("url"))][:5]
        finding = {
            "status": status,
            "question": _safe_text(self.state.subtask.get("question")),
            "summary": _safe_text(tool_input.get("summary")) or _fallback_subagent_summary(self.state),
            "candidate_urls": candidate_urls[:8],
            "recommended_next_step": _safe_text(tool_input.get("recommended_next_step")),
            "searched_candidates": list(self.state.candidates[-8:]),
            "fetched_documents": [_subagent_document_summary(document) for document in self.state.fetched_documents[-4:]],
        }
        self.state.finding = finding
        return {"ok": True, "terminal": True, "status": status, "finding": finding}


def _run_research_subagent(*, parent_task: Task, subtask: dict[str, Any], model_client: object | None) -> dict[str, Any]:
    state = _ResearchSubagentState(subtask=subtask)
    if model_client is None:
        return _run_fallback_subagent(parent_task=parent_task, state=state)

    loop_state = AgentLoopState()
    executor = ToolExecutor(RESEARCH_SUBAGENT_TOOLS, _ResearchSubagentHandlers(parent_task=parent_task, state=state).handlers())
    trace, final_state = run_agent_loop(
        model_client=model_client,
        system_prompt=_research_subagent_prompt(),
        tool_specs=RESEARCH_SUBAGENT_TOOLS,
        executor=executor,
        initial_user_payload={
            "subtask": subtask,
            "parent_task": {
                "task_id": parent_task.task_id,
                "run_id": parent_task.run_id,
                "kind": parent_task.kind,
            },
            "instruction": "Explore this subtask privately. Return useful source leads and a concise finding. Do not write shared storage.",
        },
        state=loop_state,
        safety_cap=8,
        metadata_role="research_subagent_loop",
    )
    finding = state.finding or _fallback_subagent_finding(state=state, terminal_reason=final_state.terminal_reason)
    finding["rounds"] = len(trace)
    finding["index"] = int(subtask.get("index") or 0)
    return finding


def _run_fallback_subagent(*, parent_task: Task, state: _ResearchSubagentState) -> dict[str, Any]:
    query = _safe_text(state.subtask.get("question"))
    result = SearchTool().run({"query": query, "limit": 5}, ToolContext(run_id=parent_task.run_id, task_id=parent_task.task_id))
    raw_results = list(result.data.get("results") or []) if result.ok else []
    state.candidates.extend(_candidate_summary(item) for item in raw_results)
    best = state.candidates[0] if state.candidates else {}
    url = _safe_text(best.get("url"))
    if url:
        fetch_result = FetchUrlTool().run({"url": url}, ToolContext(run_id=parent_task.run_id, task_id=parent_task.task_id))
        if fetch_result.ok:
            document = dict(fetch_result.data)
            profile = _classify_page(document, url)
            state.fetched_documents.append({**document, "usable": bool(profile["likely_extractable"]), "page_profile": profile})
    return _fallback_subagent_finding(state=state, terminal_reason=result.error)


def _fallback_subagent_finding(*, state: _ResearchSubagentState, terminal_reason: str | None = None) -> dict[str, Any]:
    candidate_urls = [_safe_text(item.get("url")) for item in state.candidates if _safe_text(item.get("url"))][:8]
    usable_docs = [document for document in state.fetched_documents if bool(document.get("usable"))]
    status = "complete" if candidate_urls or usable_docs else "blocked"
    return {
        "status": status,
        "question": _safe_text(state.subtask.get("question")),
        "summary": _fallback_subagent_summary(state) if status == "complete" else (_safe_text(terminal_reason) or "No useful private source path found."),
        "candidate_urls": candidate_urls,
        "recommended_next_step": "Researcher should rank these candidates, fetch the best source, then publish only usable raw documents." if candidate_urls else "",
        "searched_candidates": list(state.candidates[-8:]),
        "fetched_documents": [_subagent_document_summary(document) for document in state.fetched_documents[-4:]],
        "index": int(state.subtask.get("index") or 0),
    }


def _fallback_subagent_summary(state: _ResearchSubagentState) -> str:
    if state.fetched_documents:
        doc = state.fetched_documents[-1]
        return f"Private subagent fetched {_document_url(doc)}; usable={bool(doc.get('usable'))}; title={_safe_text(doc.get('title'))}."
    if state.candidates:
        return f"Private subagent found {len(state.candidates)} candidate source(s); top source: {_safe_text(state.candidates[0].get('url'))}."
    return "Private subagent did not find useful source candidates."


def _subagent_document_summary(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": _document_url(document),
        "title": _safe_text(document.get("title")),
        "usable": bool(document.get("usable")),
        "page_type": _safe_text(document.get("page_type") or dict(document.get("page_profile") or {}).get("page_type")),
        "information_density": _safe_text(document.get("information_density") or dict(document.get("page_profile") or {}).get("estimated_information_density")),
    }


def _normalize_subagent_task(item: dict[str, Any], *, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "question": _safe_text(item.get("question")),
        "search_goal": _safe_text(item.get("search_goal")),
        "constraints": [_safe_text(value) for value in list(item.get("constraints") or []) if _safe_text(value)][:5],
    }


def _research_subagent_prompt() -> str:
    return (Path(__file__).resolve().parent.parent / "prompts" / "research_subagent.md").read_text(encoding="utf-8")


def _task_question(task: Task) -> str:
    for key in ("question", "sub_question", "targeted_query", "objective", "query"):
        value = _safe_text(task.inputs.get(key))
        if value:
            return value
    return f"{task.kind} {task.task_id or ''}".strip()


def _summarize_board_snapshot(snapshot: dict[str, list[Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        key: [
            {"id": item.item_id, "status": item.status, "title": item.title, "payload": item.payload}
            for item in values[-8:]
        ]
        for key, values in snapshot.items()
    }


def _candidate_summary(item: dict[str, Any]) -> dict[str, Any]:
    url = _safe_text(item.get("url"))
    title = _safe_text(item.get("title"))
    snippet = _safe_text(item.get("snippet") or item.get("content"))
    return {
        "url": url,
        "title": title,
        "snippet": snippet[:500],
        "source_category": _source_category(url, title, snippet),
        "estimated_fetch_risk": _fetch_risk(url),
        "content_format_hint": _format_hint(url, title, snippet),
    }


def _source_category(url: str, title: str, snippet: str) -> str:
    domain = urlparse(url).netloc.lower()
    text = f"{title} {snippet}".lower()
    if any(part in domain for part in ("github.com", "docs.", "readthedocs", "wikipedia.org")) or "/docs" in url:
        return "official_doc"
    if any(part in domain for part in ("reddit.com", "stackoverflow.com", "news.ycombinator.com")):
        return "forum"
    if any(part in domain for part in ("youtube.com", "youtu.be", "bilibili.com")):
        return "video_platform"
    if any(part in domain for part in ("twitter.com", "x.com", "linkedin.com")):
        return "social_media"
    if any(word in text for word in ("release", "changelog", "pricing", "documentation")):
        return "documentation"
    if any(part in domain for part in ("techcrunch.com", "theverge.com", "reuters.com", "bloomberg.com")):
        return "news"
    return "unknown"


def _fetch_risk(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if any(part in domain for part in ("reddit.com", "linkedin.com", "twitter.com", "x.com", "youtube.com", "youtu.be")):
        return "high"
    if any(part in domain for part in ("substack.com", "medium.com", "stackoverflow.com")):
        return "medium"
    return "low"


def _format_hint(url: str, title: str, snippet: str) -> str:
    text = f"{url} {title} {snippet}".lower()
    if "pricing" in text:
        return "product_page"
    if "changelog" in text or "release" in text:
        return "changelog"
    if "tutorial" in text or "how to" in text:
        return "tutorial"
    if "youtube" in text or "youtu.be" in text:
        return "video_description"
    if "forum" in text or "reddit" in text or "stackoverflow" in text:
        return "discussion"
    return "article"


def _rank_source(source: dict[str, Any], goal: str) -> dict[str, Any]:
    url = _safe_text(source.get("url"))
    title = _safe_text(source.get("title"))
    snippet = _safe_text(source.get("snippet"))
    domain = urlparse(url).netloc.lower()
    score = 0.0
    reasons: list[str] = []

    category = _safe_text(source.get("source_category"))
    if category in {"official_doc", "documentation"}:
        score += 3.0
        reasons.append("official/documentation source")
    if category == "news":
        score += 1.0
        reasons.append("news source")
    if source.get("origin") == "fetched" and source.get("researcher_status") in {"pending", "deferred"}:
        score += 1.5
        reasons.append("already fetched and still publishable")
    if source.get("fetcher") == "firecrawl":
        score += 0.5
        reasons.append("cleaned by Firecrawl")

    risk = _safe_text(source.get("estimated_fetch_risk"))
    if risk == "low":
        score += 1.0
        reasons.append("low fetch risk")
    elif risk == "high":
        score -= 1.0
        reasons.append("high fetch risk")

    goal_terms = _tokens(goal)
    text_terms = _tokens(f"{title} {snippet} {url}")
    overlap = sorted(goal_terms & text_terms)
    if overlap:
        score += min(3.0, len(overlap) * 0.4)
        reasons.append(f"matches goal terms: {', '.join(overlap[:6])}")

    if any(part in domain for part in ("jd.com", "ir.jd.com", "sec.gov", "hkexnews.hk", "reuters.com", "bloomberg.com")):
        score += 2.0
        reasons.append("high-authority domain for company strategy")
    if any(part in domain for part in ("reddit.com", "linkedin.com", "twitter.com", "x.com", "youtube.com")):
        score -= 1.5
        reasons.append("platform source may be hard to extract or low authority")

    return {
        "url": url,
        "title": title,
        "score": round(score, 2),
        "origin": source.get("origin"),
        "source_category": category,
        "estimated_fetch_risk": risk,
        "content_format_hint": source.get("content_format_hint"),
        "researcher_status": source.get("researcher_status"),
        "reasons": reasons,
    }


def _tokens(value: str) -> set[str]:
    lowered = value.lower()
    ascii_tokens = set(token for token in lowered.replace("/", " ").replace("-", " ").split() if len(token) >= 3)
    cjk_terms = {
        term
        for term in ("京东", "外卖", "即时", "零售", "物流", "供应链", "机器人", "战略", "投资", "财报", "官方", "生鲜", "七鲜")
        if term in value
    }
    return ascii_tokens | cjk_terms


def _classify_page(document: dict[str, Any], url: str) -> dict[str, Any]:
    text = _safe_text(document.get("text"))
    html = _safe_text(document.get("html"))
    domain = urlparse(url).netloc.lower()
    lower = f"{text} {html[:1000]}".lower()
    if any(marker in lower for marker in ("captcha", "verify you are human", "access denied", "enable javascript", "rate limit")):
        return {"page_type": "blocked", "estimated_information_density": "low", "likely_extractable": False, "reason": "blocked_or_verification_page"}
    if any(part in domain for part in ("youtube.com", "youtu.be", "reddit.com", "linkedin.com", "twitter.com", "x.com")) and len(text) < 1200:
        return {"page_type": "social_feed", "estimated_information_density": "low", "likely_extractable": False, "reason": "platform_shell_low_signal"}
    if len(text) < 500:
        return {"page_type": "article", "estimated_information_density": "low", "likely_extractable": False, "reason": "too_little_visible_text"}
    if len(text) < 1500:
        return {"page_type": "article", "estimated_information_density": "medium", "likely_extractable": True, "reason": "medium_text_density"}
    return {"page_type": "article", "estimated_information_density": "high", "likely_extractable": True, "reason": "sufficient_text_density"}


def _acquisition_pressure(state: ResearcherToolState, task: Task) -> dict[str, Any]:
    failures = list(state.acquisition_failures[-12:])
    static_failures = [item for item in failures if item.get("tool") == "fetch_source"]
    stronger_failures = [item for item in failures if item.get("tool") == "firecrawl_source"]
    blocked_failures = [
        item
        for item in failures
        if item.get("failure_kind") in {"http_403", "http_429", "verification_or_blocked", "blocked_page", "platform_shell_low_signal"}
    ]
    domains = sorted({str(item.get("domain") or "") for item in blocked_failures if item.get("domain")})
    target_url = _latest_failed_url(blocked_failures) or _latest_failed_url(failures)
    issue_key = _safe_text(task.inputs.get("issue_key"))
    recommended = None
    reason = ""
    if len(static_failures) >= 2 and blocked_failures:
        recommended = "browser_agent"
        reason = "Multiple static fetch attempts are blocked or rate-limited."
    if static_failures and stronger_failures and blocked_failures:
        recommended = "browser_agent"
        reason = "Static fetch and Firecrawl both failed or returned blocked/verification content."
    return {
        "static_fetch_failures": len(static_failures),
        "firecrawl_failures": len(stronger_failures),
        "blocked_or_rate_limited_failures": len(blocked_failures),
        "blocked_domains": domains[:8],
        "latest_failed_url": target_url,
        "issue_key": issue_key,
        "recommended_escalation": recommended,
        "reason": reason,
        "failed_attempts": failures,
    }


def _failure_kind(reason: str, *, document: dict[str, Any]) -> str:
    lowered = f"{reason} {document.get('usability_reason') or ''} {document.get('page_type') or ''}".lower()
    if "403" in lowered or "forbidden" in lowered:
        return "http_403"
    if "429" in lowered or "too many requests" in lowered or "rate limit" in lowered:
        return "http_429"
    if "captcha" in lowered or "verify" in lowered or "verification" in lowered or "access denied" in lowered:
        return "verification_or_blocked"
    if "platform_shell" in lowered:
        return "platform_shell_low_signal"
    if "blocked" in lowered:
        return "blocked_page"
    if "too_little_visible_text" in lowered or "low" in lowered:
        return "low_signal"
    return "fetch_failed"


def _latest_failed_url(failures: list[dict[str, Any]]) -> str:
    for item in reversed(failures):
        url = _safe_text(item.get("url"))
        if url:
            return url
    return ""


def _stable_browser_issue_key(*, goal: str, target_url: str, reason: str) -> str:
    normalized = json.dumps(
        {
            "goal": " ".join(goal.lower().split()),
            "target_url": target_url.strip().lower(),
            "reason": " ".join(reason.lower().split())[:300],
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return f"browser.{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:16]}"


def _select_publishable_documents(documents: list[dict[str, Any]], requested_urls: set[str]) -> list[dict[str, Any]]:
    selected_by_url: dict[str, dict[str, Any]] = {}
    normalized_requested_urls = {_normalize_document_url(item) for item in requested_urls}
    for document in documents:
        url = _document_url(document)
        normalized_url = _normalize_document_url(url)
        if normalized_requested_urls and normalized_url not in normalized_requested_urls:
            continue
        if not bool(document.get("usable")) or document.get("researcher_status") not in {"pending", "deferred"}:
            continue
        current = selected_by_url.get(normalized_url)
        if current is None or _publishable_document_score(document) > _publishable_document_score(current):
            selected_by_url[normalized_url] = document
    return list(selected_by_url.values())


def _find_fetched_document(documents: list[dict[str, Any]], url: str) -> dict[str, Any] | None:
    normalized_url = _normalize_document_url(url)
    for document in reversed(documents):
        if _normalize_document_url(_document_url(document)) == normalized_url:
            return document
    return None


def _matching_decidable_documents(documents: list[dict[str, Any]], url: str) -> list[dict[str, Any]]:
    normalized_url = _normalize_document_url(url)
    return [
        document
        for document in documents
        if _normalize_document_url(_document_url(document)) == normalized_url
        and bool(document.get("usable"))
        and document.get("researcher_status") in {"pending", "deferred"}
    ]


def _close_same_url_documents(documents: list[dict[str, Any]], published_document: dict[str, Any], *, status: str, reason: str) -> None:
    normalized_url = _normalize_document_url(_document_url(published_document))
    for document in documents:
        if document is published_document:
            continue
        if _normalize_document_url(_document_url(document)) != normalized_url:
            continue
        if bool(document.get("usable")) and document.get("researcher_status") in {"pending", "deferred"}:
            document["researcher_status"] = status
            document["decision_reason"] = reason


def _unresolved_publishable_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        document
        for document in documents
        if bool(document.get("usable")) and document.get("researcher_status") in {"pending", "deferred"}
    ]


def _document_summary(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": _document_url(document),
        "title": _safe_text(document.get("title")),
        "status": _safe_text(document.get("researcher_status")),
        "reason": _safe_text(document.get("decision_reason")),
    }


def _document_url(document: dict[str, Any]) -> str:
    return _safe_text(document.get("url") or document.get("source_url"))


def _normalize_document_url(url: str) -> str:
    parsed = urlparse(_safe_text(url))
    if not parsed.scheme or not parsed.netloc:
        return _safe_text(url).rstrip("/")
    path = parsed.path.rstrip("/") or "/"
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}{query}"


def _publishable_document_score(document: dict[str, Any]) -> int:
    text_len = len(_safe_text(document.get("text")))
    score = min(text_len, 5000)
    if document.get("fetcher") == "firecrawl":
        score += 500
    if _safe_text(document.get("information_density")) == "high":
        score += 250
    if _safe_text(document.get("page_type")) == "article":
        score += 100
    return score


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
