# Coding Guidelines

These guidelines define how InsightSwarm code should be organized and changed.
They are intentionally practical. If a rule cannot guide a normal code review,
it does not belong here.

For merge-blocking rules, see [Engineering Standards](engineering-standards.md).
This file explains how to write code that naturally satisfies those standards.

## File Responsibilities

Keep files boring and easy to name.

- `objective_runtime.py` is bootstrap and governance only: start workers,
  synchronize gates, track budgets, leases, stop reasons, and output paths.
- `agents/*.py` files own worker loops and role-level decisions.
- `agents/*_tools.py` files own tool schemas, tool handlers, tool result shapes,
  side effects, and idempotency.
- `swarm_store.py` and `db/` own persistence, transactions, scoped queries, and
  store-level invariants.
- `delivery_gate.py` owns delivery readiness checks, not research planning.
- `prompts/*.md` owns model role instructions. Do not bury large prompts inside
  Python strings.
- `tools/*.py` owns external acquisition or utility integrations. Tool modules
  should not know about agent loops.

If a file needs a second unrelated reason to exist, split it.

## Runtime Code

Runtime code should stay thin.

Allowed in runtime:

- create a run,
- initialize stores,
- start workers,
- synchronize extraction batches and delivery gate,
- update run state,
- enforce budgets and leases,
- write trace events and final delivery result.

Not allowed in runtime:

- call a specific agent's research function directly,
- decide whether a topic needs search, browser, extraction, or repair,
- assemble fake steps from test fixtures,
- contain model prompts or role-specific business logic,
- become a second runtime family.

Runtime is allowed to govern. Runtime is not allowed to think for the agents.

## Agent Code

Agents are autonomous participants, not advanced helper functions.

- Each agent should claim tasks or read scoped messages.
- Each agent should decide its next tool call from its role prompt, private
  state, event memory, task context, and tool results.
- Agents may create tasks, messages, artifacts, or board items only through
  shared-store tools.
- Agents must not import another agent and call its execution path.
- Agents must not default to reading all tasks, all messages, or all artifacts
  for a run.
- Agent private state is private. Shared storage receives concise handoff facts,
  not full reasoning.

When changing an agent, ask: would this still work if another worker loop
claimed the downstream task later? If no, the change is probably too
function-call-shaped.

## Tool Code

Every agent-facing tool needs a small contract.

Each tool should define:

- `name`,
- `description`,
- `input_schema`,
- `output_schema` or stable output shape,
- `side_effects`,
- what counts as success,
- what counts as recoverable failure,
- what is idempotent or deduped.

Tool handlers should return useful operational facts:

- `ok`,
- `error` when failed,
- `deduped` when applicable,
- created ids when shared state was written,
- counts or pressure signals when they help the next decision,
- `terminal` only when the worker path should stop.

Do not hide important control flow inside a tool. If a tool refused work because
it was duplicated, unsafe, blocked, or already complete, say that explicitly in
the result.

## Store Code

Stores are not planners.

Store code may:

- validate schema invariants,
- enforce dependency limits,
- claim and lease tasks,
- dedupe creation,
- provide scoped views,
- run transactions.

Store code must not:

- rank sources,
- decide research direction,
- interpret evidence quality,
- choose delivery content,
- inspect prompt/private-state semantics.

If store code starts reading like product logic, move that logic to an agent
tool or a small policy module.

## Prompt Code

Prompts are runtime assets.

- Keep prompts in `insightswarm/prompts/`.
- Prompt changes should be reviewed like code changes.
- A prompt should state role, boundaries, available tools, private state shape,
  finish criteria, and safety constraints.
- A prompt should not promise tools that do not exist.
- A prompt should not use old names like `action` when the runtime expects
  `tool_call`.
- If a prompt depends on a code behavior, the code behavior should have a test or
  explicit guard.

Large prompt changes should be paired with at least one real `steps.jsonl`
inspection, not only unit tests.

## Configuration

Configuration should be predictable from a clean clone.

Use this priority order:

```text
CLI flag > environment variable > local config file > code default
```

Repository rules:

- Put examples in committed example files.
- Put local provider names, base URLs, and key env names in local config.
- Route runtime output to `.insightswarm/` or another explicit project-local
  directory.
- Route browser profiles to project-local configured paths.
- Keep secrets and local machine paths out of defaults.

## Dependencies

Dependencies are part of the product surface.

- Core runtime dependencies belong in `pyproject.toml`.
- Browser-only dependencies belong in a browser extra.
- Evaluation-only dependencies belong in an eval or dev extra.
- Test-only dependencies belong in a dev group or dev extra.
- Do not rely on packages that are only installed in one developer's global
  environment.

If README says `pip install -e ".[browser]"`, that command should install what
the browser path actually needs.

## Test and Fixture Boundaries

Tests should exercise runtime behavior without becoming runtime behavior.

Allowed:

- fake model clients,
- fake search/fetch/browser providers,
- fixture payloads under `tests/` or explicitly named fixture modules,
- dependency injection from tests,
- opt-in acceptance tests for real providers.

Not allowed:

- fake success paths that bypass worker/store/tool contracts,
- adding tests that require local secrets by default.

If a fixture exists to preserve a behavior, the production behavior should still
be explainable without mentioning the fixture.

## Model Calls

Every model call must be auditable.

Required metadata:

- `run_id`,
- `task_id` when available,
- `role`,
- operation or tool context.

Agent loops should attach metadata automatically. Individual agents should not
have to remember the audit contract by hand.

Provider errors should become explicit technical failure records when they
affect a task. Do not silently downgrade a model failure into a weak report.

## Browser Code

Browser automation should be capability-first.

Preferred shape:

- named browser capabilities,
- clear input/output schemas,
- page state snapshots,
- visible text or DOM summaries,
- explicit authorization for high-risk actions,
- raw source publication only after acquisition succeeds.

Use arbitrary generated Python only where the task truly needs code-level
composition. Keep common page operations behind named capabilities.

Browser code must not:

- access credentials, cookies, storage, headers, downloads, uploads, payment,
  or account mutation without authorization,
- write outside project-local run/profile directories,
- publish formal evidence directly.

## Fallbacks

Fallbacks must be visible.

- Fallback-producing functions should return structured reason metadata.
- The caller should decide delivery downgrade from that metadata.
- Fallback content should explain the failure class in user-visible language.
- Fallback helpers should be small enough to test directly.

## Naming

Prefer names that reveal system boundaries.

Good:

- `Researcher`
- `BrowserAgent`
- `Extractor`
- `Critic`
- `Writer`
- `TaskStore`
- `Mailbox`
- `ArtifactStore`
- `Evidence`
- `RunState`
- `delivery_gate`
- `repair_budget`
- `source_url_ledger`

Avoid:

- `manager` when it is really an orchestrator,
- `controller` when it plans research,
- `graph` when it hides scheduling,
- `claim` unless the object is truly a formal claim contract,
- `action` when the protocol is `tool_call`,
- `fallback` without visible degradation semantics.

## Code Review Checklist

Before calling a change clean, check:

- Is the file responsibility still narrow?
- Did runtime stay bootstrap/governance only?
- Is shared storage still the collaboration medium?
- Did a worker gain a direct call path to another worker?
- Does every new tool have schema, side effect, error, and dedupe behavior?
- Does every model call carry audit metadata?
- Did any fixture or fake provider leak into production code?
- Are local configs and outputs kept out of git?
- Did the change add a hidden fallback?
- Can a clean clone follow README and run the documented smoke path?
