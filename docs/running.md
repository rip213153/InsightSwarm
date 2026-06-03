# Running Locally

## Environment

Recommended Windows terminal setup:

```powershell
chcp 65001
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
```

Model and search setup:

```powershell
$env:DASHSCOPE_API_KEY="..."
$env:TAVILY_API_KEY="..."
$env:INSIGHTSWARM_MODEL_PROVIDER="qwen"
$env:INSIGHTSWARM_QWEN_TEXT_MODEL="qwen3.7-plus"
```

Optional browser and vision setup:

```powershell
$env:INSIGHTSWARM_BROWSER_BACKEND="visible"
$env:INSIGHTSWARM_BROWSER_PROFILE_ROOT="E:\code\InsightSwarm\.tmp\browser-profiles"
$env:INSIGHTSWARM_QWEN_OMNI_MODEL="qwen3.5-omni-plus-2026-03-15"
```

## Commands

Smoke:

```powershell
python -m insightswarm.cli run smoke "smoke test"
```

Ask:

```powershell
python -m insightswarm.cli --model-provider qwen run ask "OpenAI 下一步想做什么？" --search-provider tavily --max-runtime-seconds 1800 --max-no-progress-seconds 180 --max-drain-seconds 900
```

Ask with visible browser:

```powershell
python -m insightswarm.cli --model-provider qwen run ask "了解这个网站" --browser-backend visible --search-provider tavily
```

Ask with a local image:

```powershell
python -m insightswarm.cli --model-provider qwen run ask "我想了解这张图片里的网站" --input-file "C:\path\to\image.png" --browser-backend visible
```

JSON output:

```powershell
python -m insightswarm.cli --model-provider qwen run ask "DeepSeek 下一步战略" --json
```

## Output Locations

- Database: `.insightswarm/insightswarm.db`
- Artifacts: `.insightswarm/artifacts/`
- Run traces: `.insightswarm/.tmp/run-<run_id>/steps.jsonl`
- Browser profiles: `.tmp/browser-profiles/` by default, or
  `INSIGHTSWARM_BROWSER_PROFILE_ROOT` when set.

These paths are ignored by git.

## Debugging Runs

`steps.jsonl` is the most useful file for understanding behavior. It records
worker rounds, tool calls, private state summaries, and runtime events.

Good signs:

- Researcher calls `read_task`, searches, fetches, and publishes raw documents.
- Extractor creates citation artifacts and evidence ids.
- Critic reads evidence map, validates, and passes or requests repair.
- BrowserAgent appears when static fetch cannot acquire a page.
- Runtime stops with `deliver_called`.

Known bad signs:

- repeated `model_error` from quota or rate limits,
- `no_progress_budget_exhausted`,
- BrowserAgent repeatedly inspecting without changing strategy,
- Extractor blocked by quote backcheck or provider timeout.

## Acceptance Tests

Default `pytest` ignores `tests/acceptance` because those tests can require real
model credentials and longer runtime budgets.

Run acceptance explicitly:

```powershell
python -m pytest -q tests/acceptance
```
