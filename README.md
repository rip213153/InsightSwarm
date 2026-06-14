# InsightSwarm

InsightSwarm is a local-first, shared-store multi-agent research runtime for
turning an intelligence question into a cited report.

The current prototype focuses on autonomous workers collaborating through
SQLite-backed shared stores instead of a central controller calling agents as
functions. A thin runtime bootstraps a run, starts workers, watches budgets and
delivery conditions, and writes outputs under the local project directory.

## Current Shape

- `Researcher` searches, fetches, evaluates source quality, publishes raw
  sources, and can use private scoped subagents for parallel source discovery.
- `BrowserAgent` handles hard web acquisition with a visible browser/CDP path,
  code-driven page inspection, and human authorization for high-risk actions.
- `Extractor` converts raw documents into quote-backed citations and formal
  evidence.
- `Critic` reviews scoped evidence bundles, challenges weak coverage, and can
  request targeted repair.
- `Writer` creates the final `report`, `report_partial`, or `report_blocked`
  after the delivery gate opens.
- `Lead` bootstraps initial work and maintains constraints, but workers advance
  through shared task/message/artifact/evidence stores.

## Install

```powershell
git clone <repo-url>
cd InsightSwarm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[browser]"
```

Copy `.env.example` to `.env` or set environment variables directly.
Copy `config.models.example.json` to `config.models.json`, then edit only the
provider `base_url`, `api_key_env`, and model names for your OpenAI-compatible
endpoint.

```powershell
$env:MODEL_API_KEY="..."
$env:TAVILY_API_KEY="..."
$env:INSIGHTSWARM_MODEL_CONFIG="config.models.json"
$env:INSIGHTSWARM_MODEL_PROVIDER="default"
```

Optional:

```powershell
$env:FIRECRAWL_API_KEY="..."
```

Never commit real keys. `.env` files, run databases, artifacts, browser
profiles, and temporary outputs are ignored by `.gitignore`.

## Run

Use UTF-8 in Windows terminals when asking Chinese questions:

```powershell
chcp 65001
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
```

Run a real research question:

```powershell
python -m insightswarm.cli --model-provider default run ask "为什么2026年中国航空公司燃油费屡次上调" --search-provider tavily --max-runtime-seconds 1800 --max-no-progress-seconds 180 --max-drain-seconds 900
```

Use the visible browser path for hard acquisition:

```powershell
$env:INSIGHTSWARM_BROWSER_BACKEND="visible"
$env:INSIGHTSWARM_BROWSER_PROFILE_ROOT="E:\code\InsightSwarm\.tmp\browser-profiles"

python -m insightswarm.cli --model-provider default run ask "了解这个复杂网站" --browser-backend visible --search-provider tavily
```

Attach local multimodal input, such as an image, as user-provided context:

```powershell
python -m insightswarm.cli --model-provider default run ask "我想了解这张图片里的网站" --input-file "C:\path\to\image.png" --browser-backend visible
```

Outputs are written locally under:

- `.insightswarm/insightswarm.db`
- `.insightswarm/artifacts/`
- `.insightswarm/.tmp/run-<run_id>/steps.jsonl`

## Smoke Check

```powershell
python -m insightswarm.cli run smoke "smoke test"
```

Run unit tests:

```powershell
python -m pytest -q
```

Acceptance tests require real model credentials and scripted fixtures.

Run them explicitly when credentials are configured:

```powershell
python -m pytest -q tests/acceptance
```

## Documentation

- [Architecture](docs/architecture.md)
- [Running Locally](docs/running.md)
- [Browser Safety](docs/browser-safety.md)

## Safety Notes

BrowserAgent is intentionally conservative. Login, credential entry, payment,
uploads/downloads, cookie/storage/header access, and other high-risk actions
require explicit authorization or are blocked. Browser observations are source
acquisition material, not formal evidence. Formal evidence begins when
Extractor creates quote-backed citations from raw documents.

## Project Status

This repository is an active prototype. The most important known gap is run
recovery after provider failures such as model quota exhaustion or rate limits.
The shared-store runtime can preserve progress, but robust resume/retry policy
is still being designed.
