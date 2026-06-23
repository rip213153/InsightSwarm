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

Copy `.env.example` to `.env` or set environment variables directly. The
interactive launcher can run from environment variables without a local model
config file.

```powershell
$env:TAVILY_API_KEY="..."
$env:DASHSCOPE_API_KEY="..."
$env:INSIGHTSWARM_QWEN_TEXT_MODEL="qwen3.7-plus"
```

Optional file-based routing:

```powershell
Copy-Item config.models.example.json config.models.json
$env:INSIGHTSWARM_MODEL_CONFIG="config.models.json"
```

Optional acquisition tools:

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

Start the interactive runtime:

```powershell
python main.py
```

Inside the prompt:

```text
InsightSwarm> /model qwen3.7-plus
InsightSwarm> /ask 为什么2026年中国航空公司燃油费屡次上调？
InsightSwarm> /ask --image C:\path\to\image.png 我想了解这张图片里的网站
InsightSwarm> /exit
```

You can still run one question directly:

```powershell
python main.py "为什么2026年中国航空公司燃油费屡次上调？"
```

Each completed run prints a compact metrics line with elapsed minutes, token
usage, model call count, evidence count, and raw document count.

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
