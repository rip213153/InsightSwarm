# Browser Safety

BrowserAgent is a hard acquisition subsystem, not a general browser automation
bot.

## Allowed Direction

BrowserAgent can:

- open public URLs,
- inspect page state,
- collect visible text,
- scroll and wait,
- dismiss obvious public cookie/modals when safe,
- run bounded read-only page inspection,
- publish relevant raw source text for Extractor,
- request human authorization for high-risk actions.

## Blocked Or Gated Actions

The system blocks or requires explicit authorization for:

- login or credential entry,
- account-gated flows,
- payment, purchase, or submit actions,
- uploads or downloads,
- cookie, localStorage, sessionStorage, header, token, or password access,
- arbitrary JavaScript,
- private file/internal/localhost navigation in production-like flows.

## Visible Browser Profiles

Visible browser runs use project-local browser profiles by default:

```powershell
$env:INSIGHTSWARM_BROWSER_PROFILE_ROOT="E:\code\InsightSwarm\.tmp\browser-profiles"
```

This avoids writing run-specific browser profiles into unrelated system temp
locations.

## Vision Escalation

BrowserAgent may use a multimodal model when DOM/CDP text is insufficient, such
as canvas-heavy pages, image text, blocked visual overlays, or visual-first
interfaces. Vision output is an observation only. It does not become formal
evidence until converted into raw source material and processed by Extractor.

## Human Authorization

When BrowserAgent hits a high-risk path, the CLI can pause and print
`HumanAuthorizationRequired`. If the operator approves in the current CLI
session, the run can resume with an explicit authorization decision recorded in
shared storage. If denied or interrupted, the run should report blocked rather
than silently bypassing the boundary.
