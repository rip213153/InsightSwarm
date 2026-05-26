# InsightSwarm Project Status

## Current State

Milestone 1, Milestone 2, Milestone 3, the core of Milestone 4, the first search-driven discovery slice, report-quality hardening, the structured state-machine safety slice, the collaboration-kernel slices, the production trust/degraded delivery slice, the BrowserAgent safety/source-acquisition slices, the bounded continuation stack, the Agent-Governed Research Graph skeleton, the Runtime Workspace + Graph-Governed Executor, the Workspace-Governed Swarm Runtime, Source Acquisition Gateway + Evidence Convergence, Isolated Agent Swarm Governance Runtime, and Phase 45 Real Multi-Agent Governance Acceptance / Code-Mediated Browser Swarm v1 are implemented.

Milestone 1 established the local fake E2E pipeline:

- SQLite WAL state core with thread-local connections.
- Fixed DAG: `Discovery -> Extract -> Synthesize -> QA -> Deliver`.
- Message lease, acknowledgement, and lease recovery.
- Artifact and citation provenance.
- QA rejection and retry.
- Markdown report output with citation markers.
- CLI inspection commands.

Milestone 2 added:

- Model provider registry and audited model calls.
- Explicit provider placeholders for `deepseek`, `openai_compatible`, `qwen_text`, `aliyun_text`, and `aliyun_vision`.
- ContextEnvelope generation before every Agent execution.
- ContextEnvelope persistence as `context_envelope` JSON artifacts.
- Deterministic token estimation and budget trimming metadata.
- QA rejection condensation for retry context.
- Root project status tracking in this file.

Milestone 3 added:

- CLI run inputs for competitor, source URLs, source text files, and screenshot files.
- Real Qwen OpenAI-compatible provider support for text and omni/vision calls.
- Source artifacts generated from manual inputs instead of hard-coded fake-only content.
- Lightweight schema validation and QA gates.
- Document, image, and inference citation generation from Agent outputs.
- Writer delivery of `report.md`, `citations.json`, and `qa_report.json` artifacts.

Milestone 4 added the real scraping and provenance layer:

- `httpx`-first fetcher chain with browser fallback.
- SPA fallback heuristics that combine text length and HTML structure signals.
- Playwright concurrency limiting with a global semaphore defaulting to 2.
- Successful fetches now persist raw document, HTML, screenshot, and manifest artifacts where available.
- Failed fetches now persist `fetch_failure` artifacts and `agent_events` with fetcher, URL, latency, error summary, and fallback reason.
- QA gates now backcheck document quotes against raw text artifacts and validate image bbox payloads.
- `run inspect` now reports fetch totals, HTTP success count, Playwright fallback count, failure artifact count, and latest failure reason.
- Conflict normalization for price and billing-cycle comparisons now canonicalizes currency and monthly/yearly equivalence before declaring unresolved conflicts.
- Browser-dependent behavior is still optional in test environments; missing Playwright/browser support should skip browser smoke coverage rather than fail the suite.

The search-driven discovery slice added:

- Unified `SearchClient` abstraction with Tavily as the first live provider and static search for deterministic tests/manual URL fallback.
- CLI `--query`, `--search-provider`, `--search-limit`, and `--link-gate-max-selected` inputs.
- Search and LinkGate phases with persisted `search_results` and `link_gate` artifacts.
- Dynamic Extract task expansion after LinkGate selection, so search result count no longer needs to be known before the run starts.
- LinkGate schema robustness with structured JSON parsing, regex URL recovery, and top-result rule fallback.
- Snippet-first extraction that can convert high-signal Tavily snippets into `raw_document` artifacts without waking Playwright.
- Multi-source synthesis support by passing all structured knowledge artifacts into the analyst instead of only the latest one.
- Fetch hardening with browser-like headers, `domcontentloaded` Playwright waits, content stability checks, lazy-load scrolling, partial capture, and error classification.
- QA/reporting now surface source health, snippet evidence, synthetic fallback evidence, and degraded source warnings.

The report-quality hardening slice added:

- `DocumentCleaner` now removes common navigation/login/sidebar boilerplate, creates `cleaned_document` artifacts, and persists `evidence_chunks`.
- Extractor now reads cleaned text, creates citations against cleaned documents, and trims report quotes to concise evidence excerpts.
- LinkGate now applies source quality and freshness scoring, penalizing stale pricing/configuration evidence and low-quality aggregator/second-hand pages.
- Query-driven production runs now require a real model for strategic analysis; `fake` is blocked by QA instead of producing misleading strategic claims.
- WriterAgent can call a real model for final Chinese report writing when the provider is not fake; fake remains available for deterministic non-query tests.
- QA now separates hard failures from warnings, with gates for real-model-required, stale cleaned documents, synthetic fallback, and noisy-source cleanup.

The structured state-machine safety slice added:

- Task lifecycle now explicitly recognizes `pending`, `in_progress`, `needs_repair`, `degraded`, `blocked`, and `completed` as the supported execution states.
- Runner can schedule `needs_repair` tasks, so QA rejection becomes a state transition instead of a plain exception or blind retry.
- QA rejection condensation now emits structured repair contracts with `negative_constraints`, `banned_quotes`, `banned_patterns`, `evidence_hints`, and `allowed_sources`.
- Retry contexts carry repair contracts through task metadata and messages, so rejected agents receive new constrained inputs instead of silently replaying the original prompt.
- QA validation is split into explicit `validate_format`, `validate_evidence`, `validate_permissions`, and `validate_freshness` layers while preserving the legacy `validate_qa_gates` facade.
- Quote backchecking now attempts exact match, whitespace-normalized match, punctuation/full-width normalization, and sentence-level repair before rejecting evidence.
- Production query runs using `fake` for StrategicAnalystAgent or WriterAgent are blocked from producing formal strategic analysis or formal report artifacts.
- WriterAgent now emits `report_blocked` JSON for production fake-provider blocks instead of a misleading formal markdown `report`.

The multi-agent collaboration kernel slice added:

- `ResearchLeadAgent` now creates a `research_contract` artifact at the start of every run.
- Query runs now begin with `ResearchLead -> Search -> LinkGate -> dynamic Extract -> Synthesize -> QA -> Deliver`; manual source runs also receive a simplified contract before the original Scraper path.
- `ContextEnvelope` now carries a `collaboration_protocol` and `research_contract` summary so each Agent receives explicit role, responsibility, forbidden-action, and handoff guidance.
- LinkGate outputs `source_reviews` with selection rationale and risk flags in addition to selected URLs.
- QA repair contracts include the active research contract summary, so repair is tied to the original research questions and completion criteria.
- Lightweight DB-backed phase gates now emit `phase_gate_report` artifacts and events before Synthesize and Deliver.
- `run inspect` now includes a collaboration summary with contract, role/task state, phase-gate, QA, and repair information.

The collaboration-kernel hardening slice added:

- Collaboration concerns are now split into focused modules for contract creation/loading, role protocol, context policy, and phase gates.
- `ResearchContract`, `CollaborationProtocol`, `ContextPolicyDecision`, and `PhaseGateResult` provide lightweight typed boundaries without adding dependencies or changing artifact wire formats.
- `ContextBuilder` delegates artifact visibility to `ContextPolicy`, keeping role-scoped context rules out of envelope assembly.
- `Runner` keeps the existing single-process behavior while isolating per-task execution and phase-gate application behind a cleaner scheduling boundary.
- Tests now cover contract/protocol stability, missing contract files, role-scoped context policy, phase-gate blocking, collaboration inspect output, and the existing fake/query E2E paths.

The collaboration-experience slice added:

- `run inspect` now presents the collaboration flow as a team narrative without changing the DAG or adding new Agents.
- Collaboration inspect output now includes `contract_goal`, `key_questions`, `role_progress`, `handoff_chain`, `convergence_status`, `skeptic_review`, and `repair_rounds`.
- The existing QA and repair loop is surfaced as a SkepticQA-to-HypothesisAnalyst exchange, including failures, warnings, do-not-repeat constraints, repair state, retry count, and human-intervention status.
- Collaboration summarization lives in an observability helper, keeping inspect readable and avoiding new runtime side effects.
- Tests now cover completed, blocked, and repair-heavy collaboration summaries.

The production trust and degraded delivery slice added:

- Run status now supports `completed_degraded` for formally delivered reports with hard degraded evidence quality.
- Synthetic-only production query evidence is treated as degraded delivery rather than clean completion or full blockage, while fake production model gates remain blocked.
- Source health and degradation evaluation are centralized in a quality helper shared by Runner, reporting, inspect, and collaboration summaries.
- Formal reports now include Source Health and Degraded Output Warnings sections, including synthetic fallback, fetch failure, stale/noisy source, QA warning, and phase-gate warning signals.
- `run inspect` now exposes `quality_status`, `degraded_reasons`, `source_health`, and `run_final_status_reason`.
- Collaboration convergence now surfaces degraded completion when applicable.

The writer citation reliability slice added:

- Model-written reports are now validated against the deterministic template report before delivery.
- Writer validation requires preserving all `[[doc:...]]`, `[[img:...]]`, and `[[inf:...]]` citation markers from the template report.
- Writer validation also requires the Source Health section, and degraded runs must preserve Degraded Output Warnings.
- If a model-written report fails citation or quality-section validation, WriterAgent falls back to the template report, emits a writer repair event, and records the fallback without blocking the run.
- `report` and `report_blocked` artifact metadata now include writer validation status, fallback usage, and missing marker details.
- `run inspect` now exposes writer quality information so users can see whether final delivery used model prose or a safe template fallback.

The production source trust hardening slice added:

- Production query runs now treat `synthetic_fallback` as diagnostic-only rather than formal evidence.
- Extractor skips formal document citations for production synthetic fallback facts while preserving diagnostic artifacts and events.
- Strategic analysis now produces a formal blocked analysis artifact when no real document evidence is available, allowing QA and Writer to deliver an explicit `report_blocked` bundle.
- QA gates now hard-fail production runs without real document citations, while mixed real evidence plus fetch/source warnings can still complete as `completed_degraded`.
- Source trust evaluation is centralized in `quality.py` and shared by QA gates, Runner status decisions, Writer blocked delivery, reporting, inspect, and collaboration summaries.
- `run inspect` now exposes real evidence count, diagnostic fallback count, formal evidence availability, source trust status, and blocked reason.
- Source Health report sections now state that synthetic fallback is diagnostic-only for production queries.

The non-blocking SkepticReview collaboration slice added:

- `SkepticReviewAgent` now runs after StrategicAnalyst and before QA in both query and manual-source DAGs.
- SkepticReview creates a `skeptic_review` artifact with challenged claims, evidence gaps, source risks, recommended checks, and `non_blocking: true`.
- QA remains the only hard gate; SkepticReview never creates citations, blocks delivery, or changes run status directly.
- Collaboration protocol and context policy now include the SkepticReviewer role, with QA and Writer allowed to read skeptic review artifacts.
- QA reports and Writer inputs now carry the latest skeptic review as review context.
- Final reports can include a Skeptic Review section summarizing evidence gaps and source risks without adding unsupported facts.
- `run inspect` collaboration output now exposes skeptic review artifact count, latest review, challenges, evidence gaps, and review handoff.

The run diagnosis and fetch coverage summary slice added:

- A read-only diagnosis helper now summarizes run status, source trust, fetch coverage, source failures, QA gates, Writer delivery, SkepticReview context, and deterministic next actions.
- `run inspect` now includes a top-level `diagnosis` block while preserving existing inspect fields.
- `run diagnose --run-id <run_id>` provides a human-readable diagnosis summary, with `--json` for structured output.
- Fetch coverage now reports requested/selected sources, attempted fetches, real source successes, Tavily snippets, Playwright successes, fetch failures, synthetic diagnostic fallbacks, and formal document citations.
- `report_blocked` payloads now include source failures, formal evidence availability, and recommended next actions.
- Collaboration summaries now include a lightweight `diagnosis_handoff` from source trust through SkepticReview, QA, and Writer delivery.

The real browser smoke and fetcher hardening slice added:

- Browser smoke tests are now registered behind the `browser_smoke` pytest marker and `INSIGHTSWARM_BROWSER_SMOKE=1` opt-in flag.
- Browser smoke coverage uses local HTTP fixtures for static HTML and SPA shell fallback paths, avoiding external network dependencies.
- Playwright unavailable environments skip browser smoke tests with an explicit reason instead of failing default test runs.
- Playwright fetch metadata now consistently records backend, error kind, wait strategy, partial status, screenshot capture, HTML/text length, fallback reason, and attempts.
- Browser partial capture remains diagnostic and is not promoted into formal source trust.
- Run diagnosis now reports browser attempts, successes, partial captures, failures, unavailable counts, and whether browser smoke is recommended.
- Diagnosis source failures now expose browser fallback details such as attempted browser, browser error kind, partial status, and screenshot capture.

The QA validation category transparency slice added:

- QA failures and warnings are now summarized into stable validation categories: format, evidence, permission, and freshness.
- `run inspect` and the top-level diagnosis payload now expose validation category status, blocking categories, warning categories, and a compact QA failure summary.
- `run diagnose` now includes a Validation Categories section so users can distinguish schema/format issues, evidence defects, production permission blocks, and freshness/source-quality warnings.
- QA repair contracts now carry validation category summaries, giving retrying agents clearer repair guidance without changing QA pass/fail semantics.
- Deterministic diagnosis recommendations now include category-specific next actions while preserving source, browser, and Writer fallback guidance.

The source acquisition tools boundary slice added:

- A lightweight in-process tools layer now defines shared tool protocol fields, `ToolContext`, `ToolResult`, and a static registry.
- Search, URL fetch, and source quality/freshness scoring are exposed as `search.web`, `fetch.url`, and `source.quality` tools with deterministic examples for model context.
- SearchAgent, ExtractorAgent, and LinkGate now call these tools while preserving the existing DAG, artifact types, and delivery semantics.
- Fetch tooling now applies a minimal URL safety boundary: only HTTP(S), local/internal URLs blocked by default, and localhost permitted only for test-mode fixtures.
- ContextEnvelope now surfaces relevant available tools, schemas, safety policy, and examples to Search, LinkGate, and Extractor roles.
- Diagnosis and `run diagnose` now include tool usage/failure summaries and source quality explanations from LinkGate source reviews.

The tool invocation audit and policy gate slice added:

- `ToolExecutor` now wraps tool calls and emits `tool_call_started`, `tool_call_completed`, `tool_call_blocked`, and `tool_call_failed` events.
- Tool event metadata includes sanitized input summaries, policy metadata, diagnostics, warnings, provenance, and a correlation `tool_call_id`.
- SearchAgent, ExtractorAgent, and LinkGateAgent now call tools through the executor, preserving existing DAG and artifact semantics.
- Search provider policy now blocks unsupported providers; URL fetch policy continues to block non-HTTP(S), localhost/internal, and file inputs outside test fixtures.
- ContextEnvelope tool descriptions now include allowed callers, side-effect level, network access, blocked input summaries, and example failures.
- Diagnosis and `run diagnose` now include Tool Audit summaries with call counts, status counts, blocked calls, failed calls, and latest tool failure.

The collaboration trace artifact slice added:

- Runs now produce a `collaboration_trace` artifact summarizing the collaboration process without adding tables or changing the DAG.
- Trace payloads include contract summary, timeline, role steps, handoffs, tool calls, evidence chain, SkepticReview, QA gate, repair rounds, delivery, and convergence status.
- Trace payloads reference artifact/citation/event ids and short previews rather than copying full raw documents, HTML, prompts, or secrets.
- `run inspect` now exposes the latest collaboration trace artifact id and a compact trace summary.
- `run diagnose` now includes a Collaboration Trace section for trace presence, role step count, tool call count, repair count, and final delivery status.

The browser extension collector input slice added:

- A local collector module now accepts user-triggered browser page summary payloads through a restricted ingest path.
- CLI commands `collector serve --run-id <run_id> --port <port>` and `collector ingest --run-id <run_id> --payload-file <file>` provide the local gateway and file-based ingestion entry points.
- Accepted captures create a `browser_collected_page` artifact and a `raw_document` artifact marked with `fetcher = browser_extension_collector` and `manual_browser_capture = true`.
- Collector sanitization enforces HTTP(S) source URLs, blocks localhost/internal/file URLs outside test mode, truncates large text/html fields, and strips sensitive cookie/header/token/localStorage/password-like inputs.
- Collector accepted/rejected events are emitted into `agent_events`, and diagnosis, inspect, source health, and collaboration trace now surface browser collector counts, latest collected URL, rejection count, and sanitizer warning counts.
- A minimal MV3 browser extension skeleton is included for local development, with user-click capture only and no agent-driven browsing, login, form, navigation, cookie, localStorage, or header collection.

The PDF text evidence provenance slice added:

- CLI `run create` now accepts `--source-pdf-text-file` for text already extracted from PDF-like sources.
- PDF text inputs support plain text files and structured JSON with source URL, title, and page text entries.
- ScraperAgent converts valid PDF text input into a `pdf_text_source` artifact plus a `raw_document` artifact marked with `fetcher = pdf_text_source`, `source_kind = pdf_text`, page count, and manual file provenance.
- Empty or malformed PDF text input emits a diagnostic warning artifact instead of pretending to create formal evidence.
- PDF text raw documents reuse the existing Extractor, cleaned document, document citation, QA, Writer, diagnosis, and collaboration trace paths.
- Source Health, `run inspect`, `run diagnose`, and collaboration trace now surface PDF text source counts, page counts, latest source, warnings, and evidence provenance.

The BrowserAgent sandbox and HITL policy slice added:

- `BrowserAgent` is now represented in collaboration protocol as a restricted browser operator, but it is not part of the main research DAG.
- The static tool registry now includes deterministic fake `browser.*` tools for snapshot, visible text, screenshot, goto, scroll, click, type, and wait.
- Browser tools are restricted to `BrowserAgent`; other callers are policy-blocked through `ToolExecutor`.
- Browser action risk classification distinguishes safe automatic observations, review-required actions, and blocked high-risk actions.
- Review-required browser actions emit `browser_human_approval_required` events instead of executing or replaying actions.
- Browser observation persistence is supported through `browser_observation` artifacts for fake/tool-driven observations.
- ContextEnvelope exposes only `browser.*` tools to BrowserAgent, while existing Search/LinkGate/Extractor tool scopes remain unchanged.
- Diagnosis, `run diagnose`, and collaboration trace now surface browser sandbox tool calls, risk counts, blocked actions, and human approval requests.

The CDP read-only browser observation slice added:

- Browser observation now has a backend/session layer with deterministic fake backend by default and optional CDP backend for real read-only observation.
- `browser.snapshot`, `browser.visible_text`, and `browser.screenshot` can opt into `backend = cdp` while keeping BrowserAgent-only policy and ToolExecutor audit.
- CDP support is an optional `browser` extra using a small WebSocket client dependency; default installs and tests do not require it.
- Missing CDP dependency, missing CDP URL, invalid URL, or backend errors return structured `ToolResult` diagnostics instead of crashing agents.
- CDP observation is read-only: fixed internal Runtime evaluation and Page screenshot capture only; arbitrary CDP commands, arbitrary JS, click/type/navigation, cookies, storage, headers, downloads, uploads, and extension bridge control remain unavailable.
- `browser_observation` metadata, diagnosis, `run diagnose`, and collaboration trace now surface browser backend, read-only status, CDP success/error counts, and latest backend error.
- A `cdp_smoke` pytest marker and `INSIGHTSWARM_CDP_SMOKE` / `INSIGHTSWARM_CDP_URL` opt-in flow cover real CDP endpoint smoke tests without affecting default runs.

The Browser Page State / DOM+AX Snapshot slice added:

- `browser.page_state` is now a BrowserAgent-only tool that returns a bounded, read-only page state summary while leaving `browser.snapshot` backward-compatible.
- Fake backend page state is deterministic for offline protocol and diagnosis tests; optional CDP page state returns structured unavailable/error diagnostics when the browser extra or URL is missing.
- Page state output includes URL, title, text preview, visible text length, node/interactable counts, truncation status, and capped interactable element summaries with stable node ids, roles, names, tags, href/action hints, bounding boxes, frame ids, and visibility hints.
- `browser_page_state` artifacts preserve backend, read-only, node/interactable count, truncation, partial, and CDP method metadata without copying full HTML, secrets, storage, headers, or large page text.
- Diagnosis, `run diagnose`, and collaboration trace now surface Browser Page State counts, latest URL, interactable totals, truncation, and CDP page-state errors.

The CDP real browser interaction v1 slice added:

- `browser.goto`, `browser.click`, and `browser.type` now create auditable `browser_action_request` artifacts for human approval instead of executing directly.
- CLI `browser approvals`, `browser approve --execute`, and `browser reject` provide the human-in-the-loop decision path without adding database tables or changing the main research DAG.
- Approved requests can execute through deterministic fake replay or the optional CDP backend; CDP execution is limited to public HTTP(S) navigation, page-state target-id clicks, textbox-like typing, scroll, and wait flows.
- Real interaction writes `browser_approval_decision` and `browser_action_execution` artifacts, emits approval/execution events, and captures post-action page state when CDP execution succeeds.
- Diagnosis, `run diagnose`, and collaboration trace now surface pending/approved/rejected approvals, execution counts, failures, latest execution errors, and after-page-state links.
- A `cdp_interaction_smoke` pytest marker and `INSIGHTSWARM_CDP_INTERACTION_SMOKE` opt-in flow cover real browser interaction smoke without affecting default offline tests.

The Browser Code Interpreter Sandbox v0 slice added:

- `browser.extract_code` now provides a BrowserAgent-only, read-only Python extraction sandbox for `browser_page_state`, sanitized text, and artifact summaries.
- The sandbox keeps a small persistent namespace while blocking imports, network/file/browser/CDP access, arbitrary JS, `open`, `eval`, `exec`, `compile`, `__import__`, and unsafe modules.
- Built-in semantic helpers can turn page-state elements into candidate targets such as `product_detail_link`, `price_text`, `store_link`, `search_box`, `filter`, `customer_service`, and `ad`.
- Code execution writes `browser_code_result` artifacts and emits browser code events, with candidate counts, extracted item counts, namespace summaries, warnings, and errors.
- Diagnosis, `run diagnose`, and collaboration trace now surface browser code execution counts, namespace variable counts, candidate target counts, success/failure counts, and latest code errors.
- Packaging metadata now explicitly limits setuptools package discovery to `insightswarm*`, so editable installs do not treat `browser_extension` as a Python package.

The Browser Target Selection / DOM Index v2 slice added:

- `browser.select_target` is now a BrowserAgent-only, artifact-only decision tool that fuses `browser_page_state`, `browser.extract_code` candidates, and target intent.
- Page-state elements and selection output now carry stronger target identity fields such as DOM index, semantic type, container/nearby context, negative signals, preferred action, confidence, and reject reasons.
- Product-detail links are ranked above customer-service, ad, store, filter, and unknown controls for ecommerce/search-result pages; ambiguous or risky selections require human disambiguation.
- `browser_action_request` can reference a `browser_target_selection` artifact and includes selected semantic type, target summary, risk hints, and reject reasons for approval review.
- Diagnosis, `run diagnose`, and collaboration trace now surface target selection counts, semantic type counts, low-confidence selections, human-disambiguation requirements, and selection artifacts.

The BrowserAgent mode switch / code-driven operation v0 slice added:

- BrowserAgent now has explicit `strict`, `assisted`, and `free_browser` mode semantics, with `strict` remaining the default safety mode.
- `browser.plan_actions` creates structured `browser_action_plan` artifacts with step ids, tools, risk status, approval requirements, and expected observations instead of free-form plan text.
- CLI `browser run --mode assisted|free_browser --goal ...` provides a bounded deterministic source-acquisition loop without adding BrowserAgent to the main research DAG.
- Free mode can auto-run safe browser observations but pauses on review-required `goto`/`click`/`type` by writing normal approval requests.
- `browser_operation_checkpoint` artifacts preserve plan id, step index, request id, status, and page fingerprint, so approval replay can resume without restarting step 1.
- Repeated page fingerprints trigger a breakout checkpoint, preventing wait/scroll loops from burning the full step budget on non-responsive pages.
- Diagnosis, `run diagnose`, and collaboration trace now surface browser mode, action plans, free-loop iterations, safe actions, approval pauses, blocked planned actions, candidate-source counts, and fingerprint breakouts.

The Browser Evidence Handoff / Candidate Source slice added:

- `browser.promote_source` creates BrowserAgent-only `candidate_source` artifacts from sanitized browser observations without creating formal evidence.
- CLI `browser promote --source-artifact-id` can derive a candidate source from browser page state/observation/code artifacts, and `browser promote --candidate-id` explicitly converts a ready candidate into a `raw_document`.
- Browser handoff raw documents are marked with `fetcher = browser_agent_handoff`, `source_kind = browser_handoff`, candidate provenance, and `requires_extractor = true`.
- Free browser mode can emit candidate sources from successful observations, but it does not automatically promote every observation into evidence.
- Source Health, diagnosis, `run diagnose`, and collaboration trace now surface browser candidate sources, promoted raw documents, blocked promotions, citation-ready counts, and handoff provenance.

The Manual Evidence Handoff Continuation slice added:

- CLI `run extract --run-id <run_id> --raw-document-id <artifact_id>` creates a one-off `ExtractorAgent` task for an existing raw document without resuming or rewriting the full DAG.
- Extractor can now resolve an explicit `raw_document_id`, preserving raw-document provenance while producing normal `cleaned_document`, `evidence_chunks`, `structured_knowledge`, and document citations.
- BrowserAgent handoff raw documents can now continue through `raw_document -> Extractor -> document citation -> formal evidence` using an operator-facing command.
- Diagnosis and `run diagnose` now detect promoted `browser_agent_handoff` raw documents that have not yet been extracted and recommend the exact `run extract` continuation command.
- The default test suite now covers BrowserAgent candidate source promotion, raw-document handoff, manual extract continuation, citation creation, and formal evidence availability without requiring a real browser.

The Browser Authorization Gate / Assisted Observation slice added:

- BrowserAgent interaction policy now distinguishes `safe_auto`, `authorization_required`, `assisted_observation_required`, and `blocked` outcomes instead of treating every sensitive action as manual browser takeover.
- New artifacts `browser_authorization_request`, `browser_authorization_decision`, `browser_assisted_observation_request`, and `browser_assisted_observation_response` preserve authorization/observation state without adding database tables.
- CLI `browser authorizations`, `browser authorize`, and `browser observe` provide operator authorization and observation input while leaving BrowserAgent as the browser operator.
- Run metadata can carry `browser_allowed_domains`, `browser_authorized_domains`, `browser_assisted_observation_allowed`, and `browser_max_authorization_requests` for policy checks.
- `free_browser` checkpoints now pause on pending authorization or assisted observation and resume after the corresponding decision/response.
- Diagnosis, `run diagnose`, and collaboration trace now surface browser authorization and assisted observation counts, pending requests, latest request details, and recommended continuation commands.
- Legacy `browser_action_request` / `browser approve` compatibility remains available, but Browser Authorization is the preferred Phase 25 path.

The Bounded Subagent Runtime / Research Task Spawn v0 slice added:

- A guarded subagent runtime creates ordinary tasks with `subagent`, `parent_task_id`, `spawn_depth`, `scope`, `budget`, `output_contract`, and `allowed_tools` metadata instead of adding new database tables.
- CLI `run spawn-subagent --run-id <run_id> --parent-task-id <task_id> --role <role> --scope "..."` provides a narrow operator-facing spawn entry point.
- `research.spawn_subagent` is available as a controlled tool for ResearchLead and SkepticReview contexts, with the same runtime policy checks as the CLI.
- Run metadata now controls `max_subagents_per_run`, `max_spawn_depth`, `max_parallel_subagents`, `max_context_tokens_per_subagent`, and `allowed_subagent_roles`.
- Completed subagent tasks write `research_finding` and `subagent_handoff` artifacts and send handoff messages to the parent role and QA.
- Subagent ContextEnvelope payloads include parent task, subagent scope, context budget, and handoff requirements while using trimmed artifact visibility.
- Diagnosis, `run diagnose`, and collaboration trace now surface subagent counts, spawn tree, blocked spawns, latest handoff, and research findings.
- Subagent findings are advisory only; formal report evidence still must pass raw document extraction, document citations, QA, and Writer citation rendering.

The Subagent Finding Promotion / Evidence Candidate Merge v0 slice added:

- CLI `run promote-finding --run-id <run_id> --finding-id <artifact_id>` and `--handoff-id <artifact_id>` explicitly promote subagent findings into source candidates.
- New artifacts `candidate_research_source` and `subagent_source_promotion` preserve finding, handoff, subagent task, confidence, risk flags, and LinkGate requirement metadata.
- LinkGate now reads promoted subagent candidates alongside normal `search_results`, applies existing source quality scoring, and passes selected candidates through the existing dynamic Extract path.
- Candidate research sources remain non-evidence; only Extractor-created document citations can satisfy formal evidence and source trust.
- Diagnosis and `run diagnose` now surface candidate research source counts, pending finding promotions, pending LinkGate candidates, and a concrete `run promote-finding` recommendation.
- Collaboration trace now includes subagent source promotions and candidate research sources so the handoff-to-evidence path is auditable.

The Evidence Gap Follow-up Planner / Subagent Orchestration v0 slice added:

- CLI `run plan-followups --run-id <run_id>` creates a `research_followup_plan` artifact from source-trust gaps, missing formal evidence, QA/SkepticReview gaps, unpromoted findings, and pending LinkGate candidates without creating tasks.
- CLI `run spawn-followup --run-id <run_id> --plan-id <artifact_id> --item-id <item_id>` explicitly turns one plan item into a bounded subagent task through the Phase 26 spawn runtime.
- New artifacts `research_followup_plan` and `research_followup_decision` preserve the gap reason, recommended role, scope, priority, blocked-until hint, spawn result, and rejection reason without adding database tables.
- Follow-up spawning continues to enforce `max_subagents_per_run`, `max_spawn_depth`, `max_parallel_subagents`, and `allowed_subagent_roles`; rejected items write a decision artifact instead of silently failing.
- ResearchLead and SkepticReview contexts can see follow-up plans and decisions, while the planner does not directly create citations, reports, or deepsearch work.
- Diagnosis and `run diagnose` now surface follow-up counts, pending plan items, spawned/rejected decisions, and recommend `run plan-followups` when evidence gaps remain.
- Collaboration trace now includes `gap -> followup_plan -> spawn_followup -> subagent -> finding -> promotion -> evidence` observability hooks.

The Follow-up Research Round / Candidate Evidence Continuation v0 slice added:

- CLI `run continue-followup --run-id <run_id> --plan-id <artifact_id> --item-id <item_id>` resolves or creates a follow-up decision and advances it through one bounded research round.
- CLI `run continue-followup --run-id <run_id> --decision-id <artifact_id>` resumes an existing spawned follow-up decision.
- New artifacts `research_followup_round` and `research_followup_round_step` record decision resolution, subagent execution, handoff discovery, finding promotion, blocked reasons, and candidate source ids.
- The continuation uses the existing Runner as the task executor, then reuses Phase 27 promotion to create `candidate_research_source` artifacts from subagent findings.
- The round stops at candidate readiness; it does not create citations, run QA/Writer directly, call deepsearch, or make BrowserAgent a main DAG role.
- Diagnosis and collaboration trace now surface follow-up round counts, candidate-ready rounds, blocked rounds, latest status, continuation recommendations, and candidate-source handoff links.

The Candidate Research Source LinkGate / Extract Continuation v0 slice added:

- CLI `run continue-candidate --run-id <run_id> --candidate-id <artifact_id>` explicitly continues one `candidate_research_source` into the existing LinkGate and dynamic Extract path.
- CLI `run continue-candidate --run-id <run_id> --round-id <artifact_id>` continues only the candidate ids emitted by one `research_followup_round`.
- New artifacts `candidate_continuation` and `candidate_continuation_step` record scoped candidate resolution, LinkGate selection, dynamic Extract expansion, citation readiness, and blocked reasons.
- LinkGate can now operate in a scoped candidate-only mode so continuation does not accidentally consume all historical `candidate_research_source` artifacts in the run.
- The continuation stops at document citations; it does not continue into StrategicAnalyst, QA, Writer, deepsearch, BrowserAgent DAG roles, or automatic research loops.
- Diagnosis and collaboration trace now surface candidate continuation counts, citation-ready continuations, blocked continuations, latest continuation details, and concrete `continue-candidate` recommendations.

The Candidate Continuation Auto Analyst Reentry v0 slice added:

- Successful `continue-candidate` runs now automatically create and execute an `analysis_continuation` that launches a new `StrategicAnalystAgent` task.
- The auto reentry uses run-wide structured knowledge and citations, while recording the triggering candidate continuation and citation scope for auditability.
- New artifacts `analysis_continuation` and `analysis_continuation_step` record source linkage, analyst task creation, analysis completion, and blocked reasons.
- The automatic reentry stops at `strategic_analysis`; it does not auto-start SkepticReview, QA, Writer, deepsearch, or broader research loops.
- Diagnosis and collaboration trace now surface analysis continuation counts, analysis-ready rounds, latest analysis continuation, and direct inspection recommendations for the newest strategic analysis.

The Analysis Auto Skeptic/QA Continuation v0 slice added:

- Successful analysis continuations now automatically create and execute a bounded `review_qa_continuation` that runs `SkepticReviewAgent -> QAAgent`.
- New artifacts `review_qa_continuation` and `review_qa_continuation_step` record the source analysis continuation, scoped analyst/review/QA task ids, QA result, and blocked reasons.
- `SkepticReviewAgent` and `QAAgent` now support scoped targeting through task metadata so repeated candidate continuations in one run do not fall back to the wrong `strategic_analysis`, `skeptic_review`, or analyst task.
- QA pass in this continuation path suppresses Writer handoff, while QA failure still reuses the existing analyst repair semantics by marking the targeted analyst task `needs_repair` or `blocked`.
- The automatic continuation now stops at `qa_report` / repair decision; it does not auto-start Writer, auto-run repair rounds, or widen into broader automatic research loops.
- Diagnosis and collaboration trace now surface review/QA continuation counts, QA-passed vs repair-needed outcomes, latest QA artifacts, and continuation-to-QA lineage.

The Unified Continuation / Research Runtime Contract v0 slice added:

- Follow-up rounds, candidate continuations, analysis continuations, and review/QA continuations now share a common internal runtime contract for scope resolution, lineage metadata, step artifacts, blocked handling, result artifacts, and trace refresh.
- New internal continuation runtime types standardize `ContinuationContext`, `ContinuationScope`, and result emission without adding new tables or changing user-facing CLI entry points.
- Continuation-created tasks now receive shared runtime metadata for source artifact, scope artifact ids, scope task ids, resolution reason, and lineage, reducing ad hoc `latest`/`first` fallback semantics.
- Diagnosis now includes a unified continuation runtime summary with per-kind counts, latest continuation, and lineage edges while preserving the existing phase-specific sections.
- Collaboration trace now includes a unified continuation runtime view alongside the existing per-kind continuation artifact lists, so runtime evolution stays auditable during the transition toward a fuller Research Runtime.

The QA Repair Continuation / Reversible Analysis Repair v0 slice added:

- CLI `run continue-repair --run-id <run_id> --review-qa-id <artifact_id>` explicitly repairs a failed bounded Review/QA continuation.
- CLI `run continue-repair --run-id <run_id> --qa-report-id <artifact_id>` resolves the associated Review/QA continuation from a failed `qa_report` and repairs the same scoped analyst lineage.
- New artifacts `repair_continuation` and `repair_continuation_step` record source QA lineage, targeted original analyst task, original analysis, new repaired analyst task, repaired strategic analysis, and downstream Review/QA result.
- Repair uses a new `StrategicAnalystAgent` task carrying the existing `qa_rejection` and `repair_contract`, preserving old analysis/QA artifacts instead of overwriting them.
- After repair analysis succeeds, the system creates a repaired `analysis_continuation` and runs one scoped `review_qa_continuation`, then stops again at `qa_report`.
- QA-passed sources are blocked as not repairable, while blocked or retry-exhausted analyst tasks produce `blocked_human_intervention_required`.
- Diagnosis and collaboration trace now surface repair continuation counts, latest repaired analysis/QA artifacts, concrete `continue-repair` recommendations after QA failure, and lineage from QA failure through repair to second QA.

The QA-Passed Writer Continuation / Formal Delivery Boundary v0 slice added:

- CLI `run continue-writer --run-id <run_id> --review-qa-id <artifact_id>` explicitly continues a QA-passed bounded Review/QA continuation into Writer delivery.
- CLI `run continue-writer --run-id <run_id> --qa-report-id <artifact_id>` resolves the same delivery boundary from a passed `qa_report`.
- CLI `run continue-writer --run-id <run_id> --repair-id <artifact_id>` follows a repair continuation to its downstream passed QA result before delivery.
- New artifacts `writer_continuation` and `writer_continuation_step` record source QA lineage, optional repair lineage, Writer task id, report/report_blocked id, export ids, and delivery stop reason.
- Writer continuation creates and executes exactly one scoped `WriterAgent` task, preserving existing WriterAgent template/model/fallback/production-blocking behavior.
- A normal `report` returns `report_ready`; a production-blocked `report_blocked` returns `blocked_delivery_ready`, distinguishing valid blocked delivery from incomplete Writer execution.
- Diagnosis and collaboration trace now surface writer continuation counts, report-ready vs blocked-delivery-ready outcomes, concrete `continue-writer` recommendations after QA pass, and lineage from QA/repair to final delivery artifacts.

The Agent-Governed Research Graph v1 Architecture Skeleton slice added:

- Phase 39 now has a `research_runtime_protocol` registry that centralizes graph node/edge kinds, continuation kinds, evidence/delivery/authority boundaries, validation/frontier/plan states, command templates, and planner priorities.
- Phase 40 now has a read-only multi-agent runtime projection for agent identity, task board, mailbox, context/policy contracts, lease visibility, and recovery boundaries without becoming a fixed workflow scheduler.
- Phase 41-44 now have a graph governance projection that makes branch, rollback, arbitration, controlled BrowserAgent, evidence convergence, and v1 closure capability status visible without executing them.
- `run graph` now supports `--protocol`, `--runtime`, and `--governance` views, all read-only by default and compatible with `--json`.
- Diagnosis and collaboration trace now include protocol, multi-agent runtime, and graph governance summaries so the system can see its architectural skeleton before adding executors.

The Multi-Agent Runtime Kernel v1 slice added:

- `run start` now routes task execution through `MultiAgentRuntimeKernel` instead of directly invoking the next runnable task from the Runner loop.
- The runtime kernel claims task leases, leases typed mailbox messages, builds scoped `ContextEnvelope` artifacts, applies runtime and phase policy gates, emits heartbeat/recovery events, runs the target agent, and acks mailbox messages after successful execution.
- New CLI commands `run runtime` and `run runtime-step` expose the runtime task board, mailbox, policy gates, leases, recovery state, and one bounded execution step without turning the runtime into a fixed workflow scheduler.
- Existing main DAG, dynamic Extract, continuation, repair, and writer behavior remain compatible, while runtime events make claim, completion, block, heartbeat, and recovery visible to diagnosis and collaboration trace.
- Runtime remains evidence-boundary preserving: mailbox messages are collaboration intent, not facts; BrowserAgent, subagent findings, and candidate sources still must pass Extract/citation/QA before formal use.

The Runtime Workspace + Graph-Governed Executor v1 slice added:

- A filesystem-backed runtime workspace under `.insightswarm/runtime/<run_id>/` with deterministic manifest, work order, branch, rollback, authorization, arbitration, and snapshot JSON records.
- CLI commands `run workspace` and `run execute-plan` expose the workspace and execute exactly one bounded graph-governed plan step.
- `execute-plan` consumes Research Graph planner steps and delegates resume actions to existing explicit continuations while preserving Writer as an explicit delivery boundary.
- Rollback, branch, human gate, and arbitration execution now create auditable workspace records without deleting artifacts, overwriting history, or starting an infinite scheduler.
- Research Graph projection now includes workspace nodes and lineage edges, while diagnosis and collaboration trace summarize workspace/executor state.
- Graph projection remains passive with respect to workspace creation; the workspace command or executor creates the filesystem workroom, not read-only graph inspection.

The Workspace-Governed Swarm Runtime + Controlled BrowserAgent v1 slice added:

- CLI commands `run swarm` and `run swarm-step` expose open graph-governed work orders, swarm assignments, handoffs, policy blocks, BrowserAgent operations, and subagent capacity.
- Workspace records now include `swarm_assignment`, `swarm_handoff`, `swarm_recovery`, `swarm_policy_block`, and `browser_swarm_operation`.
- Research Graph planning now surfaces source-acquisition frontiers from fetch failures, LinkGate candidate rejection, QA evidence failures, and Skeptic evidence gaps.
- BrowserAgent is now a first-class swarm actor for source acquisition when graph branch/source-acquisition work orders need browser/code-driven collection.
- `swarm-step --agent BrowserAgent` reuses the existing bounded BrowserAgent operation loop and records browser operation/candidate source lineage in workspace, Research Graph, diagnosis, and trace.
- BrowserAgent swarm execution writes `swarm_policy_block` when sensitive browser actions require human authorization or assisted observation.
- Branch work orders can also be assigned to bounded subagents such as `SearchAgent`, while preserving existing subagent depth/count/parallelism policy.
- BrowserAgent observations, subagent findings, and candidate sources remain advisory only; formal evidence still starts at Extractor-created citations accepted by QA.

The Real Multi-Agent Governance Acceptance / Code-Mediated Browser Swarm v1 slice added:

- `run govern` can now use the real Qwen text model (`qwen-flash-character-2026-02-26`) for LeadAgent decisions when a source-acquisition, human-gate, or arbitration collaboration choice is required.
- Protocol-deterministic boundaries such as candidate continuation, convergence, repair, and delivery remain policy-driven to avoid unnecessary model calls and prevent governance drift.
- BrowserAgent governed swarm assignments now create isolated context envelopes, run page-state observation, execute `browser.extract_code`, write browser operation/handoff records, and route advisory sources through Source Acquisition Gateway.
- BrowserAgent can carry a target URL/backend/CDP URL from Research Graph frontier/workspace metadata; multimodal Qwen (`qwen3.5-omni-plus-2026-03-15`) is reserved for explicit vision escalation when DOM/CDP/code extraction is insufficient.
- `run create` now accepts `--quality-mode`, `--browser-source-target-url`, `--browser-backend`, and `--browser-cdp-url` so real governance acceptance can use stable local source-acquisition targets.
- Real acceptance now proves LeadAgent model governance, BrowserAgent code-mediated acquisition, Gateway normalization, citation creation, QA/convergence, and explicit Writer delivery boundary with `DASHSCOPE_API_KEY`.

## Runtime Paths

Default local paths are intentionally inside the project:

- DB: `.insightswarm/insightswarm.db`
- Artifacts and reports: `.insightswarm/artifacts/`

Example:

```powershell
python -m insightswarm.cli run create --name demo
python -m insightswarm.cli run start --run-id <run_id>
python -m insightswarm.cli run inspect --run-id <run_id>
python -m insightswarm.cli events tail --run-id <run_id>
```

Search-driven example:

```powershell
$env:TAVILY_API_KEY="..."
python -m insightswarm.cli run create --name lenovo-search --query "联想拯救者 笔记本 价格 配置" --competitor "Lenovo" --search-provider tavily --search-limit 10 --link-gate-max-selected 5
python -m insightswarm.cli run start --run-id <run_id>
```

Qwen live run example:

```powershell
$env:DASHSCOPE_API_KEY="..."
python -m insightswarm.cli --model-provider qwen_text run create --name qwen-demo --competitor "ExampleCo" --source-url "https://example.com/pricing" --source-text-file .\sample.txt --screenshot-file .\sample.png
python -m insightswarm.cli --model-provider qwen_text run start --run-id <run_id>
```

## Next Goals

The next work should harden the now-real multi-agent governance path:

- Replace more heuristic LeadAgent decisions with model-selected choices only where policy has a bounded action set and a hard safety gate.
- Add an optional CDP live BrowserAgent acceptance that uses a real browser session, while keeping the local fake-backend acceptance stable and mandatory.
- Make subagent swarm acceptance as strong as BrowserAgent acceptance, including isolated context, mailbox handoff, Gateway normalization, and evidence-boundary assertions.
- Continue tightening agent context isolation so older agents cannot fall back to unscoped full-run payload history.
- Keep Writer delivery explicit and governed; future work should improve delivery package quality without letting Writer trigger hidden research loops.

## Design Decisions

- ContextEnvelope is persisted as an artifact, not a dedicated table. This keeps Milestone 2 flexible and reuses the existing artifact hash/path/inspect flow.
- External model providers are registered but not implemented in Milestone 2. Selecting one raises a clear `NotImplementedError` instead of silently falling back to fake.
- Fake E2E remains deterministic and API-key free.
- Project progress is rewritten here after major milestones rather than appended after every small edit.
- Completing a major milestone requires updating this `PROJECT_STATUS.md` file in the same work session.
- Keep code changes clean and cohesive. Avoid leaving fragmented patch-style edits when a small, coherent rewrite would make ownership and behavior easier to review.
- Qwen API keys are read only from `DASHSCOPE_API_KEY`; real keys must never be committed.
- Milestone 3 treated `--source-url` as provenance and read actual content from `--source-text-file`. Milestone 4 now prefers real URL fetching and only falls back to synthetic content when all fetch attempts fail.
- Playwright is treated as an optional capability. The codebase should continue to run useful offline tests and skip browser smoke coverage when the browser stack is absent.
- Price conflict handling currently normalizes currency and billing period only. It is intentionally narrow so unresolved semantic conflicts stay visible.
- Query runs use dynamic task expansion after LinkGate. Manual URL runs still use the original fixed Scraper -> Extract path for compatibility.
- Tavily API keys are read from `TAVILY_API_KEY` or `INSIGHTSWARM_TAVILY_API_KEY`; keys must never be committed or echoed into artifacts beyond provider status metadata.
- Collaboration is intentionally in-process and SQLite/artifact-backed for now. ClawTeam-style external CLI workers, tmux sessions, file inboxes, and worktree isolation are out of scope until a later runtime-concurrency milestone.
- Collaboration contracts and protocols are persisted as artifacts/context payloads rather than new tables, preserving the existing provenance and inspection model.
- Collaboration experience is inspect-first for now. The system surfaces existing QA/repair/message state as a readable team narrative instead of adding a new DAG node or trace table.
- `completed_degraded` is a run-level state only; task states remain unchanged. Synthetic-only production query evidence degrades a delivered report, while fake production model use still blocks formal output.
- Model writing may improve final prose, but it cannot overwrite evidence boundaries. If deterministic writer validation finds missing citation markers or missing quality warning sections, the system falls back to the template report instead of blocking the run.
- For production query runs, synthetic fallback is diagnostic material only. `completed_degraded` requires at least one real document citation; synthetic-only production runs must produce blocked delivery instead of a formal report.
- SkepticReview is a collaboration challenge step, not a gate. It can surface critique and risk context, but QA remains the only component that requests repair or blocks delivery.
- Run diagnosis is a read-only observability layer. It does not write artifacts, change DAG execution, or call models; full collaboration trace remains a later milestone.
- Browser smoke is explicit opt-in. Default offline tests must not require Playwright browser binaries, and partial browser captures are diagnostic signals rather than formal trusted evidence.
- Validation category summaries are observability and repair guidance, not a second QA gate. QAAgent remains the hard gate owner, while diagnosis and repair contracts make its reasons easier to inspect and act on.
- Search, fetch, and source quality scoring are tools because they cross collection, safety, and provenance boundaries. General browser interaction is not yet a tool; Playwright remains a controlled `fetch.url` fallback until a separate browser-operation safety design exists.
- Tool invocation audit uses `agent_events` and artifact metadata correlation instead of a new `tool_calls` table. This keeps the tool layer inspectable without changing the SQLite schema.
- Collaboration trace is a read-only observability artifact, not a scheduler, state machine, or replacement for agent events, artifacts, citations, and diagnosis.
- The browser extension collector is a manual source acquisition input channel, not an Agent browser operation tool. Captured pages can become formal evidence only after the existing Extractor, citation, QA, source-trust, diagnosis, and trace pipeline accepts them.
- PDF evidence v1 accepts already extracted text and preserves page-level provenance in artifacts/metadata. It deliberately does not parse binary PDFs or add parser dependencies yet.
- BrowserAgent is a source acquisition submodule, not a core research role. Real browser control should evolve CDP-first and extension-bridge-aware, but only behind the BrowserAgent-only tool policy and authorization/assisted-observation gates.
- CDP browser observation is optional and read-only. It is a BrowserAgent backend, not a core pipeline dependency, and must not expose arbitrary CDP commands or user-provided JavaScript.
- Browser page state is a read-only perception layer for future gated browser actions. It is diagnostic by default and must not become formal evidence unless converted through the existing raw document, Extractor citation, and QA path.
- Real browser interaction is allowed only through BrowserAgent gated replay. It remains a source acquisition submodule, does not enter the main DAG, and cannot bypass blocked categories such as payment, purchase, submit, upload/download, cookies, storage, headers, passwords, tokens, arbitrary JavaScript, or arbitrary CDP commands.
- Browser code-use is restricted to read-only extraction and target candidate generation. It can suggest targets, but cannot execute browser actions or bypass the authorization/assisted-observation policy path.
- Browser target selection improves action quality, not action authority. It can recommend a stable target and explain rejected alternatives, but real click/type/goto still requires BrowserAgent policy gating.
- Browser free mode is source acquisition, not autonomous research delivery. It can plan and run bounded browser collection steps, but cannot bypass tool policy, authorization/observation checkpoints, or the raw-document -> citation -> QA -> Writer evidence path.
- Browser candidate sources are not evidence. They become usable for reports only after explicit raw-document promotion, Extractor citation generation, QA checks, and Writer citation rendering.
- Manual handoff continuation is intentionally narrow. `run extract` targets one existing raw document and does not imply full DAG resume, automatic BrowserAgent promotion, or direct BrowserAgent citation authority.
- Subagent runtime v0 is a bounded research delegation primitive, not a full Research Graph scheduler. It reuses task metadata, artifacts, messages, and events so rollback, diagnosis, and tests remain understandable.
- Subagent finding promotion is explicit and evidence-boundary preserving. It creates LinkGate candidates, not citations, so subagent discovery must still pass source selection, extraction, citation, QA, and Writer validation.
- Follow-up orchestration is deliberately operator-confirmed. Planning evidence gaps is safe and read-only, but creating more research work still goes through explicit `spawn-followup` and the bounded subagent policy rather than an automatic loop.
- Follow-up research rounds are continuations, not evidence gates. They may advance a spawned subagent into candidate sources, but formal evidence still starts only after LinkGate/Extract produce document citations.
- Candidate continuation is explicit and citation-bounded. It turns selected `candidate_research_source` artifacts into LinkGate/Extract work and stops at document citations instead of resuming the full report DAG.
- Analysis continuation and review/QA continuation are bounded automatic reentry steps. They reconnect candidate evidence to `StrategicAnalystAgent -> SkepticReviewAgent -> QAAgent`, but still stop before Writer unless a later phase explicitly extends the path.
- Repair continuation is explicit and reversible. It repairs only a QA-failed analyst lineage, writes new tasks/artifacts, reruns one scoped Review/QA pass, and never starts Writer, search, browser work, or an automatic repair loop.
- Writer continuation is explicit and delivery-boundary preserving. It consumes only QA-passed continuation results, writes a normal Writer delivery bundle, treats `report_blocked` as blocked delivery rather than formal report, and never starts new research or repair work.
- Continuation runtime is now an explicit internal architecture layer. It standardizes scope, lineage, blocked/resume semantics, and observability across continuation kinds without yet becoming a full scheduler or Research Graph engine.
- Research runtime protocol is now the intended source of truth for graph/runtime hardcoded constants. New runtime behavior should depend on protocol registry entries instead of adding scattered local constants.
- Multi-agent runtime is currently a read-only projection, not a workflow engine. It should evolve through task board, mailbox, context envelope, policy gate, lease, heartbeat, and recovery semantics rather than fixed agent ordering.
- Graph governance is currently a read-only architecture projection. Branch, rollback, arbitration, BrowserAgent governance, evidence convergence, and v1 closure are visible contracts before they become executors.
- Runtime workspace is a filesystem-backed collaboration workroom, not a fact source. SQLite plus artifacts/citations/events remain the authoritative state; workspace records express bounded work orders and graph-governed execution audit.
- Graph-governed execution is explicitly one-step and operator-triggered. It can create branch/rollback/authorization/arbitration records or delegate to existing continuations, but it must not become a daemon scheduler or bypass evidence and delivery boundaries.
- Swarm runtime is workspace-governed, not a fixed workflow scheduler. Agents claim one bounded work order at a time, and BrowserAgent/subagent outputs remain candidate material until the formal evidence pipeline accepts them.

## Lessons And Pitfalls

- SQLite insert statements should use explicit column lists. Early Milestone 1 tests caught placeholder count drift in `messages` and `model_calls`.
- QA retry must reset both the rejected task and the QA task flow correctly. If QA stays completed after rejecting, the revised output is never reviewed.
- Context should be built before task status changes to `in_progress`, so the artifact captures the task state and retry metadata that caused the execution.
- Keep DB transactions short. Artifact file writes and model calls should stay outside DB transactions.
- Do not treat mailbox messages as the sole source of truth. Rejections are also stored in task metadata and QA artifacts.
- Provider-specific model selection must not be hidden inside prompts. Keep model names in config/env so live smoke tests can prove which model ran.
- Visual evidence should be optional until scraping/screenshot capture is automated; missing screenshots should produce a skipped artifact, not block text-only analysis.
- Real web fetches need a graceful degradation path. In this repo, that means fetch failure artifacts, manifest artifacts, and a synthetic fallback that keeps the pipeline observable rather than silently stopping.
- Avoid circular metadata payloads when serializing fetch attempts. Only persist summary fields, not recursive attempt collections.
- Dynamic DAGs should be represented through task metadata and database state, not by precomputing unknown Extract task counts.
- Small-model JSON output must be treated as unreliable. LinkGate therefore has structured, regex, and rule-based fallback paths.
- `networkidle` is too brittle for modern commercial sites; prefer DOM readiness, body availability, content stability, and partial capture.
- A completed run is not the same as a trustworthy report. Query-driven production reports must block or degrade when only fake-model analysis is available.
- Raw webpage text is too noisy for citation display. Keep raw artifacts for provenance, but extract and cite against cleaned documents and short evidence chunks.
- Do not retry a rejected task with the same unconstrained input. QA-driven repair must carry structured negative constraints and evidence/source hints.
- Formal artifacts and diagnostic artifacts must stay separate. A blocked production report should be represented as `report_blocked`, not as a normal `report`.
- Collaboration rules should live in the collaboration kernel modules, not be scattered across individual Agents. Context visibility in particular belongs in `ContextPolicy`.
- Milestone-scale collaboration changes must update this file in the same work session, not only tests or code.
- Model-written reports must preserve or receive an appended Source Health section so degraded warnings are not lost during final writing.
- Source Health appending alone is not enough for real WriterAgent output. Citation markers and degraded warning sections must survive deterministic validation, or the final artifact should be the stable template fallback.
- Phase gates should not prevent blocked diagnostic delivery when a later QA/Writer path can produce a clearer `report_blocked` artifact for the user.
- New collaboration roles should first be non-blocking unless they need to own a clear state transition; this keeps role richness from destabilizing the trusted delivery path.
- Deterministic diagnosis recommendations should be specific enough to guide reruns, but must not mask formal QA/source-trust outcomes.
- Browser fallback metadata should preserve enough detail for diagnosis without expanding the runtime scheduler or changing fetch ordering.
- Browser collector payloads must never store cookies, authorization headers, localStorage, tokens, passwords, screenshots, or full request headers; sanitized summaries are enough for provenance and downstream evidence extraction.
- File evidence should enter the same artifact/citation/QA path as web evidence. Avoid special report-only shortcuts that bypass quote backchecking or source trust.
- Browser actions should be auditable before they are powerful. Snapshot-style observation can be automatic, but clicks, typing, navigation, and any sensitive or state-changing action must stay policy-gated through authorization or assisted observation before broader code-driven browser control is added.
- CDP smoke tests must stay opt-in because real browser endpoints are environment-specific and should never be required for the default offline test suite.
- Browser handoff can stall at a valid raw document if no extraction task is scheduled. A narrow manual continuation CLI keeps this recoverable without making BrowserAgent part of the main research DAG.
