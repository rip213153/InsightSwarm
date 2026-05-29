from __future__ import annotations

import argparse
import json
from pathlib import Path

from insightswarm.agents.browser_agent import HumanAuthorizationRequired
from insightswarm.config import load_settings
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.objective_runtime import create_and_run_objective
from insightswarm.util import new_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="insightswarm")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--model-provider", default=None)
    parser.add_argument("--config-path", default=None)
    sub = parser.add_subparsers(dest="resource", required=True)

    run = sub.add_parser("run")
    run_sub = run.add_subparsers(dest="action", required=True)

    run_ask = run_sub.add_parser("ask")
    run_ask.add_argument("question", nargs="?")
    run_ask.add_argument("--query", default=None)
    run_ask.add_argument("--name", default="objective-intelligence")
    run_ask.add_argument("--max-steps", type=int, default=12)
    run_ask.add_argument("--max-runtime-seconds", type=float, default=300.0)
    run_ask.add_argument("--max-no-progress-seconds", type=float, default=30.0)
    run_ask.add_argument("--quality-mode", default="production", choices=["production", "test"])
    run_ask.add_argument("--search-provider", default="tavily")
    run_ask.add_argument("--browser-backend", default=None)
    run_ask.add_argument("--browser-cdp-url", default=None)
    run_ask.add_argument("--json", action="store_true")

    run_smoke = run_sub.add_parser("smoke")
    run_smoke.add_argument("question", nargs="?", default="smoke test")
    run_smoke.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(
        args.db_path,
        args.artifact_dir,
        args.model_provider,
        args.config_path,
    )
    init_db(settings.db_path)
    store = Store(settings.db_path, settings.artifact_dir)

    if args.resource == "run" and args.action == "ask":
        query = args.query or args.question
        if not query:
            raise SystemExit("run ask requires a question")
        try:
            result = create_and_run_objective(
                store,
                name=args.name,
                query=query,
                model_provider=settings.model_provider,
                artifact_dir=settings.artifact_dir,
                max_steps=args.max_steps,
                max_runtime_seconds=args.max_runtime_seconds,
                max_no_progress_seconds=args.max_no_progress_seconds,
                allow_delivery=True,
                quality_mode=args.quality_mode,
                search_provider=args.search_provider,
                browser_backend=args.browser_backend,
                browser_cdp_url=args.browser_cdp_url,
            )
        except HumanAuthorizationRequired as exc:
            print(f"HumanAuthorizationRequired: {exc}", file=__import__("sys").stderr)
            raise
        payload = result.to_dict()
        if payload["stop_reason"] == "human_required":
            print("HumanAuthorizationRequired", file=__import__("sys").stderr)
        if args.json:
            print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
        else:
            print(payload["report"]["body"] if payload.get("report") else payload["result_type"])
        return 0 if payload["result_type"] != "report_blocked" else 2

    if args.resource == "run" and args.action == "smoke":
        smoke_dir = Path(settings.artifact_dir).parent / ".tmp" / f"run-smoke-{new_id('smoke')}"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "ok",
            "question": args.question,
            "artifact_dir": str(Path(settings.artifact_dir)),
            "smoke_dir": str(smoke_dir),
        }
        (smoke_dir / "smoke.json").write_text(
            json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
        else:
            print(f"smoke ok: {smoke_dir}")
        return 0

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
