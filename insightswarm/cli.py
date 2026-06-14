from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from insightswarm.agents.browser_agent import HumanAuthorizationRequired
from insightswarm.authorization_flow import pending_authorization_requests, write_authorization_decision
from insightswarm.config import load_settings
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.models.router import build_model_client
from insightswarm.objective_runtime import ObjectiveBudget, create_and_run_objective, run_objective
from insightswarm.util import new_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="insightswarm")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--model-provider", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--model-config-path", default=None)
    sub = parser.add_subparsers(dest="resource", required=True)

    run = sub.add_parser("run")
    run_sub = run.add_subparsers(dest="action", required=True)

    run_ask = run_sub.add_parser("ask")
    run_ask.add_argument("question", nargs="?")
    run_ask.add_argument("--query", default=None)
    run_ask.add_argument("--name", default="objective-intelligence")
    run_ask.add_argument("--max-steps", type=int, default=12)
    run_ask.add_argument("--max-runtime-seconds", type=float, default=1800.0)
    run_ask.add_argument("--max-no-progress-seconds", type=float, default=120.0)
    run_ask.add_argument("--max-drain-seconds", type=float, default=900.0)
    run_ask.add_argument("--quality-mode", default="production", choices=["production", "test"])
    run_ask.add_argument("--search-provider", default="tavily")
    run_ask.add_argument("--browser-backend", default=None)
    run_ask.add_argument("--browser-cdp-url", default=None)
    run_ask.add_argument("--input-file", action="append", default=[], help="Attach an image, audio, or other local file as user-provided run context.")
    run_ask.add_argument("--json", action="store_true")

    run_smoke = run_sub.add_parser("smoke")
    run_smoke.add_argument("question", nargs="?", default="smoke test")
    run_smoke.add_argument("--json", action="store_true")

    eval_parser = sub.add_parser("eval")
    eval_sub = eval_parser.add_subparsers(dest="action", required=True)

    eval_run = eval_sub.add_parser("run")
    eval_run.add_argument("--cases-dir", default=str(Path("evals") / "cases"))
    eval_run.add_argument("--eval-db-path", default=str(Path(".insightswarm") / "eval.db"))
    eval_run.add_argument("--suite", default="golden")
    eval_run.add_argument("--difficulty", choices=["light", "heavy"], default=None)
    eval_run.add_argument("--case-id", action="append", default=[])
    eval_run.add_argument("--repeat", type=int, default=1)
    eval_run.add_argument("--judge-provider", default=None)
    eval_run.add_argument("--max-steps", type=int, default=12)
    eval_run.add_argument("--max-runtime-seconds", type=float, default=1800.0)
    eval_run.add_argument("--notes", default=None)
    eval_run.add_argument("--json", action="store_true")

    eval_summary = eval_sub.add_parser("summary")
    eval_summary.add_argument("eval_run_id")
    eval_summary.add_argument("--eval-db-path", default=str(Path(".insightswarm") / "eval.db"))
    eval_summary.add_argument("--json", action="store_true")

    eval_compare = eval_sub.add_parser("compare")
    eval_compare.add_argument("baseline_eval_run_id")
    eval_compare.add_argument("candidate_eval_run_id")
    eval_compare.add_argument("--eval-db-path", default=str(Path(".insightswarm") / "eval.db"))
    eval_compare.add_argument("--json", action="store_true")

    eval_review = eval_sub.add_parser("review")
    eval_review.add_argument("epoch_id")
    eval_review.add_argument("--eval-db-path", default=str(Path(".insightswarm") / "eval.db"))
    eval_review.add_argument("--human-score", type=float, required=True)
    eval_review.add_argument("--human-label", default=None)
    eval_review.add_argument("--human-comment", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(
        args.db_path,
        args.artifact_dir,
        args.model_provider,
        args.config_path,
        args.model_config_path,
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
                model_config_path=settings.model_config_path,
                artifact_dir=settings.artifact_dir,
                max_steps=args.max_steps,
                max_runtime_seconds=args.max_runtime_seconds,
                max_no_progress_seconds=args.max_no_progress_seconds,
                max_drain_seconds=args.max_drain_seconds,
                allow_delivery=True,
                quality_mode=args.quality_mode,
                search_provider=args.search_provider,
                browser_backend=args.browser_backend,
                browser_cdp_url=args.browser_cdp_url,
                input_files=args.input_file,
            )
        except HumanAuthorizationRequired as exc:
            print(f"HumanAuthorizationRequired: {exc}", file=__import__("sys").stderr)
            raise
        payload = result.to_dict()
        if payload["stop_reason"] == "human_required":
            print("HumanAuthorizationRequired", file=__import__("sys").stderr)
            if not args.json:
                result = _authorize_and_resume_if_allowed(
                    store,
                    result=result,
                    model_provider=settings.model_provider,
                    model_config_path=settings.model_config_path,
                    artifact_dir=settings.artifact_dir,
                    max_steps=args.max_steps,
                    max_runtime_seconds=args.max_runtime_seconds,
                    max_no_progress_seconds=args.max_no_progress_seconds,
                    max_drain_seconds=args.max_drain_seconds,
                )
                payload = result.to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
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
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"smoke ok: {smoke_dir}")
        return 0

    if args.resource == "eval":
        return _handle_eval(args, settings, store)

    raise AssertionError("unreachable")


def _handle_eval(args: argparse.Namespace, settings: object, store: Store) -> int:
    from insightswarm.eval.runner import build_default_swarm_runner, run_eval
    from insightswarm.eval.stats import compare_case
    from insightswarm.eval.store import EvalStore

    eval_store = EvalStore(args.eval_db_path)

    if args.action == "run":
        judge_provider = args.judge_provider or settings.model_provider
        judge_client = build_model_client(
            judge_provider,
            model_config_path=str(settings.model_config_path) if settings.model_config_path else None,
        )
        swarm_runner = build_default_swarm_runner(
            store,
            artifact_dir=settings.artifact_dir,
            model_provider=settings.model_provider,
            model_config_path=settings.model_config_path,
            max_steps=args.max_steps,
            max_runtime_seconds=args.max_runtime_seconds,
        )
        eval_run_id = run_eval(
            store=store,
            eval_store=eval_store,
            cases_dir=args.cases_dir,
            swarm_runner=swarm_runner,
            judge_client=judge_client,
            target_provider=settings.model_provider,
            repeat=args.repeat,
            suite=args.suite,
            difficulty=args.difficulty,
            case_ids=args.case_id or None,
            notes=args.notes,
        )
        payload = _eval_summary_payload(eval_store, eval_run_id)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"eval run complete: {eval_run_id}")
            _print_eval_summary(payload)
        return 0

    if args.action == "summary":
        payload = _eval_summary_payload(eval_store, args.eval_run_id)
        if payload["run"] is None:
            raise SystemExit(f"eval run not found: {args.eval_run_id}")
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            _print_eval_summary(payload)
        return 0

    if args.action == "compare":
        baseline = _agg_by_case(eval_store, args.baseline_eval_run_id)
        candidate = _agg_by_case(eval_store, args.candidate_eval_run_id)
        case_ids = sorted(set(baseline) | set(candidate))
        comparisons = []
        for case_id in case_ids:
            cmp = compare_case(case_id, baseline.get(case_id), candidate.get(case_id))
            comparisons.append({
                "case_id": cmp.case_id,
                "mean_a": cmp.mean_a,
                "mean_b": cmp.mean_b,
                "delta": cmp.delta,
                "noise_band": cmp.noise_band,
                "verdict": cmp.verdict,
            })
        payload = {
            "baseline_eval_run_id": args.baseline_eval_run_id,
            "candidate_eval_run_id": args.candidate_eval_run_id,
            "comparisons": comparisons,
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            for item in comparisons:
                print(
                    f"{item['case_id']}: {item['verdict']} "
                    f"delta={item['delta']:.4f} band=+/-{item['noise_band']:.4f}"
                )
        return 0

    if args.action == "review":
        eval_store.set_human_score(
            args.epoch_id,
            human_score=args.human_score,
            human_label=args.human_label,
            human_comment=args.human_comment,
        )
        print(f"recorded human review for {args.epoch_id}")
        return 0

    raise AssertionError("unreachable")


def _eval_summary_payload(eval_store: object, eval_run_id: str) -> dict:
    run = eval_store.get_eval_run(eval_run_id)
    aggs = eval_store.list_case_aggs(eval_run_id)
    epochs = eval_store.list_epochs(eval_run_id)
    suite_mean = sum(float(row["mean"] or 0.0) for row in aggs) / len(aggs) if aggs else 0.0
    return {
        "run": run,
        "suite_mean": round(suite_mean, 4),
        "cases": aggs,
        "epoch_count": len(epochs),
    }


def _print_eval_summary(payload: dict) -> None:
    run = payload.get("run") or {}
    print(
        f"suite={run.get('suite')} status={run.get('status')} "
        f"repeat={run.get('repeat_n')} suite_mean={payload.get('suite_mean'):.4f}"
    )
    for row in payload.get("cases") or []:
        print(
            f"{row['case_id']}: mean={float(row['mean'] or 0.0):.4f} "
            f"stderr={float(row['stderr'] or 0.0):.4f} "
            f"grounded={float(row['mean_grounded_ratio'] or 0.0):.4f}"
        )


def _agg_by_case(eval_store: object, eval_run_id: str) -> dict[str, object]:
    from insightswarm.eval.stats import ScoreSummary

    result: dict[str, ScoreSummary] = {}
    for row in eval_store.list_case_aggs(eval_run_id):
        result[str(row["case_id"])] = ScoreSummary(
            n=int(row["n_epochs"] or 0),
            mean=float(row["mean"] or 0.0),
            std=float(row["std"] or 0.0),
            stderr=float(row["stderr"] or 0.0),
            min_score=float(row["min_score"] or 0.0),
            max_score=float(row["max_score"] or 0.0),
        )
    return result


def _authorize_and_resume_if_allowed(
    store: Store,
    *,
    result: object,
    model_provider: str,
    model_config_path: Path | None,
    artifact_dir: Path,
    max_steps: int,
    max_runtime_seconds: float,
    max_no_progress_seconds: float,
    max_drain_seconds: float,
) -> object:
    current = result
    while True:
        payload = current.to_dict()
        if payload.get("stop_reason") != "human_required":
            return current
        pending = pending_authorization_requests(store, payload["run_id"])
        if not pending:
            return current
        request = pending[-1]
        print("", file=sys.stderr)
        print("Browser authorization requested:", file=sys.stderr)
        print(f"  task_id: {request.task_id}", file=sys.stderr)
        print(f"  goal: {request.goal}", file=sys.stderr)
        print(f"  reason: {request.reason}", file=sys.stderr)
        answer = input("Allow this specific browser action and resume this run? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            write_authorization_decision(
                store,
                payload["run_id"],
                task_id=request.task_id,
                decision="deny",
                reason="operator denied browser authorization",
            )
            return current

        write_authorization_decision(
            store,
            payload["run_id"],
            task_id=request.task_id,
            decision="allow",
            reason="operator allowed this specific browser action in the active CLI session",
        )
        current = run_objective(
            store,
            payload["question"],
            ObjectiveBudget(
                max_steps=max_steps,
                max_runtime_seconds=max_runtime_seconds,
                max_no_progress_seconds=max_no_progress_seconds,
                max_drain_seconds=max_drain_seconds,
            ),
            artifact_dir.parent,
            model_client=build_model_client(
                model_provider,
                model_config_path=str(model_config_path) if model_config_path else None,
            ),
            run_id=payload["run_id"],
        )


if __name__ == "__main__":
    raise SystemExit(main())
