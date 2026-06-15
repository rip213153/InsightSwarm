from __future__ import annotations

import json

from insightswarm.cli import build_parser, main
from insightswarm.eval.store import EvalStore


def _seed_eval_run(path, *, case_id: str, scores: list[float]) -> str:
    store = EvalStore(path)
    eval_run_id = store.create_eval_run(
        suite="golden",
        judge_provider="fake",
        judge_model="fake",
        target_provider="fake",
        repeat_n=len(scores),
    )
    epoch_ids = []
    for idx, score in enumerate(scores):
        epoch_ids.append(
            store.record_epoch(
                eval_run_id=eval_run_id,
                case_id=case_id,
                epoch_idx=idx,
                swarm_run_id=f"run-{idx}",
                result_type="report",
                score_overall=score,
                score_dims={"coverage": score},
                citation_summary={"grounded_ratio": 1.0},
                grounded_ratio=1.0,
                latency_ms=10,
                token_total=20,
                status="ok",
                error=None,
                judge_rationale="ok",
            )
        )
    mean = sum(scores) / len(scores)
    store.upsert_case_agg(
        eval_run_id=eval_run_id,
        case_id=case_id,
        n_epochs=len(scores),
        mean=mean,
        std=0.0,
        stderr=0.0,
        min_score=min(scores),
        max_score=max(scores),
        mean_grounded_ratio=1.0,
    )
    store.finish_eval_run(eval_run_id)
    return eval_run_id, epoch_ids[-1]


def test_eval_summary_json(tmp_path, capsys):
    eval_db = tmp_path / "eval.db"
    eval_run_id, _ = _seed_eval_run(eval_db, case_id="c1", scores=[0.8])

    rc = main(["eval", "summary", eval_run_id, "--eval-db-path", str(eval_db), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run"]["eval_run_id"] == eval_run_id
    assert payload["suite_mean"] == 0.8
    assert payload["epoch_count"] == 1


def test_browser_backend_defaults_to_visible_for_cli_entries():
    parser = build_parser()

    ask_args = parser.parse_args(["run", "ask", "hello"])
    eval_run_args = parser.parse_args(["eval", "run"])
    eval_resume_args = parser.parse_args(["eval", "resume", "evalrun_123"])

    assert ask_args.browser_backend == "visible"
    assert eval_run_args.browser_backend == "visible"
    assert eval_resume_args.browser_backend == "visible"


def test_eval_compare_json(tmp_path, capsys):
    eval_db = tmp_path / "eval.db"
    baseline, _ = _seed_eval_run(eval_db, case_id="c1", scores=[0.2])
    candidate, _ = _seed_eval_run(eval_db, case_id="c1", scores=[0.9])

    rc = main(["eval", "compare", baseline, candidate, "--eval-db-path", str(eval_db), "--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["comparisons"][0]["case_id"] == "c1"
    assert payload["comparisons"][0]["verdict"] == "improved"


def test_eval_review_records_human_score(tmp_path):
    eval_db = tmp_path / "eval.db"
    _, epoch_id = _seed_eval_run(eval_db, case_id="c1", scores=[0.8])

    rc = main([
        "eval",
        "review",
        epoch_id,
        "--eval-db-path",
        str(eval_db),
        "--human-score",
        "0.7",
        "--human-label",
        "judge_too_lenient",
        "--human-comment",
        "Too generous.",
    ])

    assert rc == 0
    rows = EvalStore(eval_db).conn.execute(
        "SELECT human_score, human_label, human_comment FROM eval_epochs WHERE epoch_id = ?",
        (epoch_id,),
    ).fetchall()
    assert rows[0]["human_score"] == 0.7
    assert rows[0]["human_label"] == "judge_too_lenient"
    assert rows[0]["human_comment"] == "Too generous."
