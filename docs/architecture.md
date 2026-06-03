# Architecture

InsightSwarm is a shared-store autonomous multi-agent runtime.

The runtime is not meant to be a central planner. It bootstraps a run, starts
workers, keeps budgets and leases moving, evaluates delivery conditions, and
records stop reasons. Research decisions are made by workers through scoped
tasks, mailbox messages, artifacts, evidence, and board items.

## Core Stores

- `TaskStore`: claimable work with leases, dependencies, status, owner role,
  priority, and task inputs.
- `Mailbox`: scoped requests, responses, observations, suggestions, repair
  requests, and authorization messages.
- `ArtifactStore`: raw documents, citations, reports, and other durable outputs.
- `Evidence`: formal quote-backed evidence created by Extractor.
- `BoardStore`: shared work memory for plans, conflicts, evidence batches, and
  run-level coordination.
- `RunState`: objective, phase, budget, stop reason, and delivery gate state.

## Workers

- `LeadWorker`: creates the initial objective and downstream work. It should not
  directly call other agents.
- `Researcher`: runs an OODA-style tool loop. It can search, fetch, publish raw
  sources, write observations/hypotheses/suggestions, request browser
  acquisition, and spawn private scoped subagents.
- `BrowserWorker`: uses BrowserAgent tools and a code session to operate visible
  browser/CDP acquisition inside safety boundaries.
- `Extractor`: reads raw documents and proposes quote-backed citations.
- `Critic`: reviews evidence bundles, validates citation quality, resolves or
  opens conflicts, and requests targeted repair when needed.
- `WriterWorker`: activates only when delivery is open and writes the final
  report artifact.

## Runtime Loop

`objective_runtime.py` starts workers in the current process and periodically:

- synchronizes extraction batches,
- creates run-level evidence reviews when batches are ready,
- evaluates the delivery gate,
- watches no-progress and runtime budgets,
- resumes after human authorization when allowed,
- stops on delivery, blocking, or budget exhaustion.

Workers collaborate through stores. Direct central calls such as
`runtime.invoke_writer(...)` or `controller.call_agent(...)` are considered an
architecture regression.

## Evidence Path

Raw source material can come from static fetch, Firecrawl, BrowserAgent, or
user-provided context. It is not formal evidence until Extractor creates a
citation with a source URL, quote, confidence, and claim payload.

The normal path is:

```text
Researcher/BrowserAgent -> raw_document artifact
Extractor -> citation artifact + Evidence row
Critic -> pass / challenge / repair_request
Writer -> report / report_partial / report_blocked
```

## Known Gap

Run recovery after provider failures is intentionally not over-built yet. Model
quota exhaustion, rate limits, or provider outages are recorded as technical
failures, but a full resumable policy is still future work.
