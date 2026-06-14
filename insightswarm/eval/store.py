"""Persistence for the evaluation subsystem.

Uses its own SQLite file via the shared connection cache. ``swarm_run_id`` is a
logical pointer into the main DB, not an enforced foreign key.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from insightswarm.db.connection import get_db_connection
from insightswarm.util import new_id

DEFAULT_EVAL_DB_PATH = ".insightswarm/eval.db"


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def init_eval_db(db_path: str | Path) -> None:
    schema_path = Path(__file__).with_name("schema_eval.sql")
    conn = get_db_connection(db_path)
    conn.executescript(schema_path.read_text(encoding="utf-8"))


class EvalStore:
    def __init__(self, db_path: str | Path = DEFAULT_EVAL_DB_PATH):
        self.db_path = str(Path(db_path))
        init_eval_db(self.db_path)

    @property
    def conn(self):
        return get_db_connection(self.db_path)

    def create_eval_run(
        self,
        *,
        suite: str,
        judge_provider: str,
        judge_model: str | None,
        target_provider: str,
        repeat_n: int,
        git_rev: str | None = None,
        notes: str | None = None,
    ) -> str:
        eval_run_id = new_id("evalrun")
        self.conn.execute(
            """
            INSERT INTO eval_runs (
                eval_run_id, suite, judge_provider, judge_model, target_provider,
                repeat_n, git_rev, status, notes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)
            """,
            (eval_run_id, suite, judge_provider, judge_model, target_provider,
             repeat_n, git_rev, notes, _now()),
        )
        return eval_run_id

    def finish_eval_run(self, eval_run_id: str, status: str = "done") -> None:
        self.conn.execute(
            "UPDATE eval_runs SET status = ?, finished_at = ? WHERE eval_run_id = ?",
            (status, _now(), eval_run_id),
        )

    def record_epoch(
        self,
        *,
        eval_run_id: str,
        case_id: str,
        epoch_idx: int,
        swarm_run_id: str | None,
        result_type: str | None,
        score_overall: float | None,
        score_dims: dict[str, Any],
        citation_summary: dict[str, Any],
        grounded_ratio: float | None,
        latency_ms: int | None,
        token_total: int | None,
        status: str,
        error: str | None,
        judge_rationale: str | None,
    ) -> str:
        epoch_id = new_id("epoch")
        self.conn.execute(
            """
            INSERT INTO eval_epochs (
                epoch_id, eval_run_id, case_id, epoch_idx, swarm_run_id,
                result_type, score_overall, score_dims_json, citation_summary_json,
                grounded_ratio, latency_ms, token_total, status, error,
                judge_rationale, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (epoch_id, eval_run_id, case_id, epoch_idx, swarm_run_id,
             result_type, score_overall, json.dumps(score_dims, ensure_ascii=False),
             json.dumps(citation_summary, ensure_ascii=False), grounded_ratio,
             latency_ms, token_total, status, error, judge_rationale, _now()),
        )
        return epoch_id

    def record_citation_checks(self, epoch_id: str, results: list[dict[str, Any]]) -> None:
        rows = [
            (
                new_id("citchk"), epoch_id, str(item.get("source_url") or ""),
                str(item.get("quote") or ""), str(item.get("claim") or ""),
                str(item.get("match_type") or "none"), float(item.get("similarity") or 0.0),
                1 if item.get("matched") else 0, _now(),
            )
            for item in results
        ]
        self.conn.executemany(
            """
            INSERT INTO eval_citation_checks (
                check_id, epoch_id, source_url, quote, claim,
                match_type, similarity, matched, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def upsert_case_agg(
        self,
        *,
        eval_run_id: str,
        case_id: str,
        n_epochs: int,
        mean: float | None,
        std: float | None,
        stderr: float | None,
        min_score: float | None,
        max_score: float | None,
        mean_grounded_ratio: float | None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO eval_case_agg (
                eval_run_id, case_id, n_epochs, mean, std, stderr,
                min_score, max_score, mean_grounded_ratio, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(eval_run_id, case_id) DO UPDATE SET
                n_epochs=excluded.n_epochs, mean=excluded.mean, std=excluded.std,
                stderr=excluded.stderr, min_score=excluded.min_score,
                max_score=excluded.max_score,
                mean_grounded_ratio=excluded.mean_grounded_ratio,
                updated_at=excluded.updated_at
            """,
            (eval_run_id, case_id, n_epochs, mean, std, stderr,
             min_score, max_score, mean_grounded_ratio, _now()),
        )

    def set_human_score(
        self,
        epoch_id: str,
        *,
        human_score: float | None,
        human_label: str | None,
        human_comment: str | None,
    ) -> None:
        self.conn.execute(
            "UPDATE eval_epochs SET human_score = ?, human_label = ?, human_comment = ? WHERE epoch_id = ?",
            (human_score, human_label, human_comment, epoch_id),
        )

    def list_epochs(self, eval_run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM eval_epochs WHERE eval_run_id = ? ORDER BY case_id, epoch_idx",
            (eval_run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_case_aggs(self, eval_run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM eval_case_agg WHERE eval_run_id = ? ORDER BY case_id",
            (eval_run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_eval_run(self, eval_run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM eval_runs WHERE eval_run_id = ?", (eval_run_id,)
        ).fetchone()
        return dict(row) if row else None
