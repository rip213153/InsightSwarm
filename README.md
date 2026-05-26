# InsightSwarm

InsightSwarm is a local-first competitive analysis agent collaboration system.
The first milestone ships a deterministic fake end-to-end pipeline with:

- SQLite state and WAL connections
- A fixed `Discovery -> Extract -> Synthesize -> QA -> Deliver` DAG
- Message lease, acknowledgement, and expiry recovery
- Artifact and citation provenance
- QA rejection and retry
- Markdown report output
- CLI inspection commands

Run the fake pipeline:

```powershell
python -m insightswarm.cli run create --name demo
python -m insightswarm.cli run start --run-id <run_id>
python -m insightswarm.cli run inspect --run-id <run_id>
```

By default the database is written to `.insightswarm/insightswarm.db` and files
are written under `.insightswarm/artifacts/`.

Manual browser collector:

```powershell
python -m insightswarm.cli collector serve --run-id <run_id> --port 8765
python -m insightswarm.cli collector ingest --run-id <run_id> --payload-file .\collector_payload.json
```

The collector is a local input channel for user-triggered browser captures. It
listens only on `127.0.0.1`, accepts page-summary JSON, writes a
`browser_collected_page` artifact plus a `raw_document` artifact, and leaves
Extractor, citations, QA, diagnosis, and collaboration trace to decide whether
the capture becomes formal evidence.

A minimal MV3 development extension lives in `browser_extension/`. Load it as an
unpacked extension, start `collector serve`, then click the extension action on
the page you want to send. The extension does not automate browsing, click
pages, submit forms, capture cookies, read localStorage, or send request headers.

PDF text evidence input:

```powershell
python -m insightswarm.cli run create --name pdf-demo --competitor "ExampleCo" --source-url "https://example.com/pricing.pdf" --source-pdf-text-file .\pricing_pages.json
python -m insightswarm.cli run start --run-id <run_id>
```

`--source-pdf-text-file` accepts either plain `.txt` text or a JSON file shaped
like:

```json
{
  "source_url": "https://example.com/pricing.pdf",
  "title": "ExampleCo Pricing PDF",
  "pages": [
    {"page_number": 1, "text": "Extracted page text..."}
  ]
}
```

InsightSwarm does not parse binary PDFs yet. The PDF text input is for text
already extracted by another tool, then stored as `pdf_text_source` plus
`raw_document` artifacts for normal extraction, citation, QA, diagnosis, and
trace handling.

BrowserAgent sandbox:

InsightSwarm has a browser-operation safety foundation. The `browser.*` tool
family is restricted to `BrowserAgent`, uses deterministic fake execution by
default, and routes policy-sensitive actions to authorization or assisted
observation requests instead of asking humans to operate the browser manually.

Allowed fake observation tools include `browser.snapshot`,
`browser.visible_text`, `browser.screenshot`, `browser.scroll`, and
`browser.wait`. Actions such as `browser.click`, `browser.type`, and
`browser.goto` are policy-gated; high-risk inputs such as login, payment,
upload/download, cookies, storage, headers, passwords, tokens, arbitrary JS, and
production localhost/internal/file navigation are blocked or require human
authorization or assisted observation.

This is a source acquisition submodule, not the core product identity. Future
real browser work should reuse this tool policy, audit, diagnosis, and trace
boundary before adding CDP sessions or extension bridge control.

CDP read-only browser observation:

```powershell
pip install -e ".[browser]"
$env:INSIGHTSWARM_CDP_URL="ws://127.0.0.1:9222/devtools/page/<target_id>"
```

Start Chrome or Chromium with a remote debugging port, then use BrowserAgent
tools with `backend = "cdp"` for read-only observation. The CDP backend is
optional: if the extra dependency or `INSIGHTSWARM_CDP_URL` is missing, browser
tools return a structured backend-unavailable result instead of failing the main
pipeline.

Only `browser.snapshot`, `browser.visible_text`, and `browser.screenshot` use
the CDP backend in this slice. Click, type, navigation, form submission, cookies,
storage, headers, arbitrary JavaScript, downloads, uploads, and extension bridge
control remain outside the real browser path.

Browser page state:

`browser.page_state` adds a bounded DOM/Accessibility-style summary for
BrowserAgent. It reports URL, title, text preview, visible text length, node and
interactable counts, and a capped list of interactable elements with stable node
ids, roles, names, tags, href/action hints, bounding boxes, frame ids, and
visibility hints.

The tool is read-only, BrowserAgent-only, and uses the same optional CDP backend
as the other observation tools. It does not expose arbitrary CDP commands,
user-provided JavaScript, full HTML, cookies, localStorage, headers, passwords,
tokens, screenshots, or large page text. Page-state artifacts are diagnostic
observations; they do not become formal evidence unless a later source path
converts them into raw documents that pass Extractor citation and QA gates.

Real CDP smoke tests are explicit opt-in:

```powershell
$env:INSIGHTSWARM_CDP_SMOKE="1"
$env:INSIGHTSWARM_CDP_URL="ws://127.0.0.1:9222/devtools/page/<target_id>"
python -m pytest -q -m cdp_smoke
```

CDP real browser interaction:

Real interaction is available only through BrowserAgent gated replay. First use
BrowserAgent tools to collect `browser.page_state`, then call a sensitive action
such as `browser.click`, `browser.type`, or `browser.goto`; the first call writes
authorization/observation artifacts, with legacy `browser_action_request`
compatibility preserved. A human can authorize, reject, or provide observation
data explicitly:

```powershell
python -m insightswarm.cli browser authorizations --run-id <run_id>
python -m insightswarm.cli browser authorize --run-id <run_id> --request-id <artifact_id> --decision approve
python -m insightswarm.cli browser observe --run-id <run_id> --request-id <artifact_id> --value "123456"
python -m insightswarm.cli browser approvals --run-id <run_id>
python -m insightswarm.cli browser approve --run-id <run_id> --request-id <artifact_id> --execute --backend cdp
python -m insightswarm.cli browser reject --run-id <run_id> --request-id <artifact_id> --reason "not needed"
```

`browser.scroll` and `browser.wait` are safe automatic actions. Real `goto`,
`click`, and `type` are gated CDP flows only: public HTTP(S) navigation,
page-state target ids with bounding boxes, and textbox-like typing targets.
Non-whitelisted domains, login-like pages, mail/verification contexts, and
CAPTCHA/2FA-style prompts pause for authorization or assisted observation.
Payment, purchase, submit, upload/download, cookie/storage/header access,
password/token handling, arbitrary JavaScript, and arbitrary CDP commands remain
blocked.

Real interaction smoke tests are explicit opt-in:

```powershell
$env:INSIGHTSWARM_CDP_INTERACTION_SMOKE="1"
$env:INSIGHTSWARM_CDP_URL="ws://127.0.0.1:9222/devtools/page/<target_id>"
python -m pytest -q -m cdp_interaction_smoke
```

Browser code extraction sandbox:

`browser.extract_code` is a restricted, read-only Python extraction sandbox for
BrowserAgent. It is meant for turning `browser_page_state` and sanitized page
text into structured candidates such as product links, prices, stores, filters,
search boxes, ads, and customer-service controls.

The sandbox does not control the browser. It cannot click, type, navigate, call
CDP, evaluate JavaScript, access cookies/storage/headers, read files, spawn
processes, or make network requests. If code suggests a target to click, that
target must still pass BrowserAgent authorization policy before any browser
action executes.

Example code:

```python
candidate_targets = classify_page_state(page_state)
extracted_items = [
    {"title": item["text"], "target_id": item["stable_node_id"]}
    for item in candidate_targets
    if item["semantic_type"] == "product_detail_link"
]
warnings = []
```

Browser authorization and assisted observation:

BrowserAgent remains the browser operator. Humans authorize policy-sensitive
continuation or provide observations such as one-time codes; they do not take
over browser control by default. Run metadata can define
`browser_allowed_domains`, `browser_authorized_domains`,
`browser_assisted_observation_allowed`, and
`browser_max_authorization_requests` without adding database tables.

Browser target selection:

`browser.select_target` sits between page/code understanding and gated action
requests. It combines `browser.page_state` elements, `browser.extract_code`
candidate targets, and an intent such as "第一个商品详情链接，不要客服" to pick a
semantic target like `product_detail_link` while down-ranking `customer_service`,
ads, login/payment/submit controls, filters, and unknown targets.

The tool only writes a `browser_target_selection` artifact. It never clicks,
types, navigates, calls CDP, or bypasses authorization replay. A click still
follows the normal path:

```text
browser.page_state -> browser.extract_code -> browser.select_target
  -> browser.click -> browser_authorization_request -> browser authorize
```

`browser_action_request` includes the selected semantic type, target text/href,
context, risk hints, and reject reasons for compatibility so older approval
flows can still catch ambiguous targets before real CDP execution.

BrowserAgent modes and code-driven operation:

BrowserAgent supports three modes. `strict` is the default single-tool safety
mode. `assisted` creates a structured `browser_action_plan` but does not run
browser actions. `free_browser` runs a bounded deterministic loop over browser
tools for source acquisition:

```powershell
python -m insightswarm.cli browser run --run-id <run_id> --mode assisted --goal "open first product"
python -m insightswarm.cli browser run --run-id <run_id> --mode free_browser --goal "open first product"
```

The loop can auto-run safe observation steps such as page state, visible text,
scroll, and wait. Policy-sensitive steps such as click, type, and goto pause as
authorization or assisted observation artifacts. Replay resumes from the latest
`browser_operation_checkpoint` instead of restarting at step 1. Repeated page
fingerprints trigger a breakout checkpoint so loading loops do not consume the
full step budget.

Free mode is still only a source-acquisition submodule. It cannot write
citations, analysis, or reports directly; any collected page text must become
`raw_document` and pass Extractor, QA, and Writer boundaries before it can support
formal output.

Browser evidence handoff:

BrowserAgent observations can be turned into candidate sources, then explicitly
promoted into the normal evidence pipeline:

```powershell
python -m insightswarm.cli browser promote --run-id <run_id> --source-artifact-id <browser_page_state_id>
python -m insightswarm.cli browser promote --run-id <run_id> --candidate-id <candidate_source_id>
```

The first command creates a `candidate_source` from sanitized browser output. The
second creates a `raw_document` marked with `fetcher = browser_agent_handoff` and
`requires_extractor = true`. Candidate sources are not formal evidence; reports
can only rely on document citations produced later by Extractor and accepted by
QA.

Manual evidence handoff continuation:

After a BrowserAgent handoff has been promoted to `raw_document`, continue the
formal evidence chain with:

```powershell
python -m insightswarm.cli run extract --run-id <run_id> --raw-document-id <raw_document_artifact_id>
```

This creates a one-off `ExtractorAgent` task for that raw document and produces
normal `cleaned_document`, `structured_knowledge`, and document citation
artifacts. `run diagnose` recommends this command when browser handoff raw
documents exist but have not yet been extracted.

Bounded subagent runtime:

InsightSwarm now has a small, controlled subagent spawn primitive for research
delegation without replacing the existing evidence pipeline or adding database
tables. Subagents are ordinary tasks marked with parent, scope, depth, budget,
allowed tools, and an output contract in task metadata.

```powershell
python -m insightswarm.cli run create --name subagent-demo --query "ExampleCo pricing" --max-subagents-per-run 3 --max-spawn-depth 1 --allowed-subagent-role SearchAgent
python -m insightswarm.cli run spawn-subagent --run-id <run_id> --parent-task-id <task_id> --role SearchAgent --scope "Find primary pricing sources"
python -m insightswarm.cli run start --run-id <run_id>
```

The default policy allows up to 3 subagents per run, depth 1, one active
subagent at a time, and 2000 context tokens per subagent. Subagents receive a
trimmed `ContextEnvelope` focused on the research contract, parent task, own
artifacts, and relevant handoffs. Completed subagents write `research_finding`
and `subagent_handoff` artifacts. These findings are advisory only; formal
evidence still requires `raw_document -> Extractor -> citation -> QA -> Writer`.

Subagent finding promotion:

Subagent findings can be explicitly promoted into LinkGate-compatible source
candidates:

```powershell
python -m insightswarm.cli run promote-finding --run-id <run_id> --finding-id <research_finding_artifact_id>
python -m insightswarm.cli run promote-finding --run-id <run_id> --handoff-id <subagent_handoff_artifact_id>
```

Promotion writes `candidate_research_source` artifacts plus a
`subagent_source_promotion` audit artifact. LinkGate reads these candidates
alongside normal `search_results`, applies the same source quality and selection
logic, and then the existing dynamic Extract path can produce raw documents,
structured knowledge, and citations. Candidate research sources are not evidence
until Extractor creates document citations accepted by QA.

Evidence gap follow-up planning:

InsightSwarm can now turn weak source trust, missing formal evidence, QA/Skeptic
gaps, unpromoted subagent findings, and pending research candidates into an
auditable follow-up plan:

```powershell
python -m insightswarm.cli run plan-followups --run-id <run_id>
python -m insightswarm.cli run spawn-followup --run-id <run_id> --plan-id <research_followup_plan_id> --item-id followup-1
```

`plan-followups` is read-only planning: it writes a `research_followup_plan`
artifact and creates no tasks. `spawn-followup` is the explicit operator action
that converts one plan item into a bounded subagent task using the Phase 26
policy limits for allowed roles, depth, count, and parallelism. The resulting
subagent can then use the existing Phase 26/27 path:

```text
followup_plan -> spawn_followup -> subagent -> research_finding
  -> promote-finding -> candidate_research_source -> LinkGate -> Extract
```

Follow-up plans and decisions are collaboration artifacts, not evidence. They do
not create citations, bypass QA, or implement deepsearch; formal evidence still
requires Extractor-created document citations accepted by the existing gates.

Follow-up research rounds:

After a follow-up item has been planned, a bounded research round can continue it
to candidate source readiness:

```powershell
python -m insightswarm.cli run continue-followup --run-id <run_id> --plan-id <research_followup_plan_id> --item-id followup-1
python -m insightswarm.cli run continue-followup --run-id <run_id> --decision-id <research_followup_decision_id>
```

The command resolves or creates the follow-up decision, runs the bounded
subagent through the existing Runner, finds the resulting handoff/finding, and
promotes source URLs into `candidate_research_source` artifacts. It stops there:
candidate research sources still need the existing LinkGate and Extract path
before they can become citations or formal evidence.

Candidate research source continuation:

```powershell
python -m insightswarm.cli run continue-candidate --run-id <run_id> --candidate-id <candidate_research_source_id>
python -m insightswarm.cli run continue-candidate --run-id <run_id> --round-id <research_followup_round_id>
```

This continuation creates a scoped LinkGate task for the selected
`candidate_research_source` artifacts, reuses the existing dynamic Extract path,
and stops at document citations for evidence handoff. After a successful
`citation_ready` result, InsightSwarm now auto-creates bounded Analyst and
SkepticReview/QA continuations. The automatic chain now stops at
`qa_report` / analyst repair decision. It still does not continue into Writer,
deepsearch, BrowserAgent DAG integration, or automatic research loops.

QA repair continuation:

```powershell
python -m insightswarm.cli run continue-repair --run-id <run_id> --review-qa-id <review_qa_continuation_id>
python -m insightswarm.cli run continue-repair --run-id <run_id> --qa-report-id <qa_report_artifact_id>
```

If a bounded Review/QA continuation failed and marked its target analyst task as
`needs_repair`, this explicit command creates a new repair analyst task, produces
a new `strategic_analysis`, and runs one scoped Review/QA pass over that repaired
analysis. It stops again at `qa_report`; it does not auto-start Writer, auto-open
new research, or loop repairs.

Writer continuation:

```powershell
python -m insightswarm.cli run continue-writer --run-id <run_id> --review-qa-id <review_qa_continuation_id>
python -m insightswarm.cli run continue-writer --run-id <run_id> --qa-report-id <qa_report_artifact_id>
python -m insightswarm.cli run continue-writer --run-id <run_id> --repair-id <repair_continuation_id>
```

When a bounded Review/QA continuation has passed, this explicit command creates
one scoped `WriterAgent` task and emits the normal delivery bundle: `report` or
`report_blocked`, plus `citations_export` and `qa_report_export`. A
`report_blocked` artifact is treated as a valid delivery-boundary result, but not
as a formal report. The command never triggers new research, repair, BrowserAgent
work, or automatic Writer retries.

Phase 33 unified the continuation runtime contract behind these flows. Follow-up
rounds, candidate continuations, analysis continuations, review/QA continuations,
repair continuations, and writer continuations now share the same internal scope, lineage,
blocked/ready, step trace, and observability semantics while preserving the
existing explicit operator boundaries.

Research Graph architecture skeleton:

```powershell
python -m insightswarm.cli run graph --run-id <run_id> --protocol
python -m insightswarm.cli run graph --run-id <run_id> --runtime
python -m insightswarm.cli run graph --run-id <run_id> --governance
python -m insightswarm.cli run graph --run-id <run_id> --json --protocol --runtime --governance
```

The protocol view centralizes runtime constants such as graph node/edge kinds,
continuation kinds, evidence/delivery/authority boundaries, frontier statuses,
plan kinds, command templates, and planner priorities. The multi-agent runtime
view projects agent identities, the external task board, mailbox state, and
policy gates without becoming a fixed workflow scheduler. The governance view
projects Phase 41-44 capability readiness for branch, rollback, arbitration,
controlled BrowserAgent, evidence convergence, and v1 closure. These projections
are read-only by default and do not create tasks, artifacts, citations, browser
work, repair loops, or Writer delivery.

Multi-Agent Runtime Kernel:

```powershell
python -m insightswarm.cli run runtime --run-id <run_id>
python -m insightswarm.cli run runtime --run-id <run_id> --json
python -m insightswarm.cli run runtime-step --run-id <run_id>
python -m insightswarm.cli run runtime-step --run-id <run_id> --agent <agent_name>
```

`run start` now executes tasks through the runtime kernel rather than directly
scanning and invoking agents. The kernel claims task leases, leases mailbox
messages, builds the scoped `ContextEnvelope`, applies runtime/phase policy
gates, emits heartbeat and recovery events, executes one agent, and acks leased
messages after success. `runtime-step` performs exactly one bounded execution
step for inspection and recovery workflows; it is not a fixed workflow scheduler
and does not add automatic Writer, repair, BrowserAgent, branch, or deepsearch
behavior.

Runtime workspace and graph-governed execution:

```powershell
python -m insightswarm.cli run workspace --run-id <run_id>
python -m insightswarm.cli run workspace --run-id <run_id> --json
python -m insightswarm.cli run execute-plan --run-id <run_id> --step-id <plan_step_id>
python -m insightswarm.cli run execute-plan --run-id <run_id> --kind resume_plan
python -m insightswarm.cli run execute-plan --run-id <run_id> --kind rollback_plan
python -m insightswarm.cli run execute-plan --run-id <run_id> --kind branch_plan
python -m insightswarm.cli run execute-plan --run-id <run_id> --kind human_gate_plan
```

The runtime workspace is stored under `.insightswarm/runtime/<run_id>/` with
manifest, work order, branch, rollback, authorization, arbitration, and snapshot
JSON files. SQLite remains the source of truth for runs, tasks, artifacts,
citations, messages, and events; workspace files are the collaborative workroom
for graph plan execution intent and audit records.

`execute-plan` consumes one Research Graph planner step at a time. Resume steps
delegate to existing explicit continuations such as `continue-candidate`,
`continue-repair`, and `continue-writer`; rollback, branch, human gate, and
arbitration steps write bounded workspace records instead of deleting history or
starting an unbounded scheduler. Workspace records are projected back into
Research Graph, diagnosis, and collaboration trace as first-class collaboration
nodes.

Workspace-governed swarm runtime:

```powershell
python -m insightswarm.cli run swarm --run-id <run_id>
python -m insightswarm.cli run swarm --run-id <run_id> --json
python -m insightswarm.cli run swarm-step --run-id <run_id>
python -m insightswarm.cli run swarm-step --run-id <run_id> --work-order-id <work_order_id>
python -m insightswarm.cli run swarm-step --run-id <run_id> --agent BrowserAgent
python -m insightswarm.cli run swarm-step --run-id <run_id> --agent SearchAgent
```

Swarm runtime lets workspace work orders be claimed by bounded collaborators.
Research Graph planning can now surface source-acquisition frontiers from fetch
failures, LinkGate candidate rejection, QA evidence failures, and Skeptic
evidence gaps. BrowserAgent is a first-class source-acquisition actor for those
frontiers: it can run the existing safe/code-driven browser operation loop,
write browser operation records, pause on human gates, and produce browser
candidate sources. Bounded subagents can also be assigned from branch work orders
through the existing subagent policy.

Swarm execution is still one bounded step at a time. BrowserAgent and subagent
outputs are not formal evidence; they must still enter the candidate/source
promotion path and then pass `LinkGate -> Extract -> citation -> QA` before
supporting analysis or delivery.

Real multi-agent governance acceptance:

```powershell
$env:DASHSCOPE_API_KEY="..."
python -m insightswarm.cli --model-provider qwen_text run govern --run-id <run_id> --max-steps 6
python -m insightswarm.cli --model-provider qwen_text run govern --run-id <run_id> --max-steps 3 --allow-delivery
```

`run govern` is the autonomous-collaboration entrypoint. LeadAgent can use the
configured Qwen text model (`qwen-flash-character-2026-02-26` by default) to
choose bounded collaboration actions such as source acquisition, while protocol
and policy gates keep evidence, delivery, browser authority, and human
authorization boundaries hard. BrowserAgent is included as a code-mediated swarm
actor: it observes page state, runs `browser.extract_code`, writes candidate
source handoffs, and must pass the Source Acquisition Gateway plus
`LinkGate -> Extract -> citation -> QA` before anything is formal evidence.
Multimodal Qwen (`qwen3.5-omni-plus-2026-03-15`) is available only as a vision
escalation path when DOM/CDP/code extraction is insufficient.
