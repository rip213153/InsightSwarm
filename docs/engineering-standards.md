# Engineering Standards

InsightSwarm is an active prototype, but prototype code must not pretend to be
stable platform code. This document defines merge-blocking engineering
standards. Day-to-day implementation guidance lives in
[Coding Guidelines](coding-guidelines.md).

## Runtime Contract

- Runtime governs; agents decide.
- Workers collaborate through shared stores and tools, not direct function calls.
- Test fixtures and scripted golden paths must not affect production runtime.
- Formal evidence starts at Extractor-created quote-backed citations.
- BrowserAgent is hard acquisition escalation, not a default fetch replacement.
- User-visible reports must distinguish complete, partial, blocked, fallback,
  and technical-failure states.

## Blocking Violations

These issues block a stability claim and should block a merge unless the change
is explicitly marked as temporary debt with an owner and removal plan.

- Production code reads `INSIGHTSWARM_SCRIPTED_FIXTURE` or any test fixture
  payload.
- Runtime assembles user-visible steps from fixture data instead of actual run
  state.
- Runtime directly calls a specific agent execution path such as
  `invoke_writer`, `call_agent`, or `run_step_*`.
- An agent imports another agent and calls its execution path.
- A model call can happen without `run_id`, `task_id` when available, `role`,
  and operation metadata.
- Writer fallback can produce a normal-looking report without explicit fallback
  metadata and delivery downgrade.
- Browser code claims AST filtering is a security sandbox or bypasses
  authorization for high-risk actions.
- Local real configuration such as `config.models.json`, `.env`, browser
  profiles, run databases, or artifacts is committed.
- Core install instructions require undeclared dependencies or a maintainer's
  global Python environment.

## Current Known Risks

These are known risks in the current repository. They should be paid down before
presenting the project as stable.

- Writer fallback exists and must stay visibly degraded.
- Browser generated-code execution is guarded, but it is not a sandbox.
- Critic quality standards still need a fuller rubric/config surface.
- The worktree may contain rollback residue such as untracked local tests.

## Required Guardrail Tests

The project should have tests or static checks for these regressions:

- fixture leakage into production modules,
- direct central agent invocation,
- missing model audit metadata,
- fallback masquerading as normal report,
- committed local model configuration,
- duplicate repair creation,
- duplicate URL publish/extract,
- delivery gate deadlock after Critic pass or pass-with-caveats.

## Merge Checklist

Before calling a change stable, answer:

- Did runtime remain bootstrap/governance only?
- Is shared storage still the only collaboration medium?
- Did any worker gain a direct call path to another worker?
- Did any fixture or fake provider leak into production code?
- Does every model call carry audit metadata?
- Are fallback states visible and degraded?
- Is new shared state scoped, small, and idempotent?
- Did we add a hard-coded rule that should be rubric/config?
- Can a clean clone install and run the documented smoke path?
