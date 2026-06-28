from __future__ import annotations

import sqlite3

from insightswarm.eval.store import EvalStore


# Minimal subset of the pre-judge_method schema (master's schema_eval.sql at
# the time eval-triage forked). Used to seed an "old" eval.db so the upgrade
# path can be exercised end-to-end.
_OLD_SCHEMA = """
CREATE TABLE eval_runs (
  eval_run_id TEXT PRIMARY KEY,
  suite TEXT NOT NULL,
  judge_provider TEXT NOT NULL,
  judge_model TEXT,
  target_provider TEXT NOT NULL,
  repeat_n INTEGER NOT NULL DEFAULT 1,
  git_rev TEXT,
  status TEXT NOT NULL DEFAULT 'running',
  notes TEXT,
  created_at TEXT NOT NULL,
  finished_at TEXT
);

CREATE TABLE eval_epochs (
  epoch_id TEXT PRIMARY KEY,
  eval_run_id TEXT NOT NULL REFERENCES eval_runs(eval_run_id),
  case_id TEXT NOT NULL,
  epoch_idx INTEGER NOT NULL,
  swarm_run_id TEXT,
  result_type TEXT,
  score_overall REAL,
  score_dims_json TEXT NOT NULL DEFAULT '{}',
  citation_summary_json TEXT NOT NULL DEFAULT '{}',
  grounded_ratio REAL,
  latency_ms INTEGER,
  token_total INTEGER,
  status TEXT NOT NULL DEFAULT 'pending',
  error TEXT,
  judge_rationale TEXT,
  human_score REAL,
  human_label TEXT,
  human_comment TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE eval_case_agg (
  eval_run_id TEXT NOT NULL REFERENCES eval_runs(eval_run_id),
  case_id TEXT NOT NULL,
  n_epochs INTEGER NOT NULL,
  mean REAL,
  std REAL,
  stderr REAL,
  min_score REAL,
  max_score REAL,
  mean_grounded_ratio REAL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (eval_run_id, case_id)
);

CREATE INDEX idx_eval_epochs_run_case ON eval_epochs(eval_run_id, case_id, epoch_idx);
"""


def _build_legacy_db(eval_db) -> None:
    """Create a fresh eval.db using the pre-judge_method schema (no migrations)."""
    con = sqlite3.connect(eval_db)
    try:
        con.executescript(_OLD_SCHEMA)
        con.commit()
    finally:
        con.close()


def test_init_eval_db_migrates_old_schema_without_judge_method(tmp_path):
    """Old eval.db (pre judge_method column) must upgrade transparently.

    Reproduces the upgrade crash: master schema has no judge_method, then
    EvalStore() is constructed on the same db file -> must not raise, and
    record_epoch must be able to write judge_method.
    """
    eval_db = tmp_path / "eval.db"
    _build_legacy_db(eval_db)

    # Upgrade path must not raise. The first EvalStore() call triggers
    # init_eval_db -> _migrate_eval_schema -> ALTER TABLE ADD COLUMN.
    store = EvalStore(eval_db)

    # Writing judge_method must succeed and the value must persist.
    rid = store.create_eval_run(
        suite="s",
        judge_provider="j",
        judge_model=None,
        target_provider="t",
        repeat_n=1,
    )
    eid = store.record_epoch(
        eval_run_id=rid,
        case_id="c",
        epoch_idx=0,
        swarm_run_id=None,
        result_type="report",
        score_overall=0.5,
        score_dims={},
        citation_summary={},
        grounded_ratio=1.0,
        latency_ms=1,
        token_total=1,
        status="ok",
        error=None,
        judge_rationale="r",
        judge_method="fallback",
    )
    row = store.conn.execute(
        "SELECT judge_method FROM eval_epochs WHERE epoch_id=?", (eid,)
    ).fetchone()
    assert row["judge_method"] == "fallback"


def test_migrate_eval_schema_is_idempotent(tmp_path):
    """Running _migrate_eval_schema twice (two EvalStore constructions) is a no-op.

    ALTER TABLE ADD COLUMN raises if the column already exists; the migrator
    must guard via PRAGMA table_info so re-running on an already-migrated db
    does not raise.
    """
    eval_db = tmp_path / "eval.db"
    _build_legacy_db(eval_db)

    EvalStore(eval_db)
    # Second construction re-runs the migrator against the now-migrated db.
    EvalStore(eval_db)

    # Sanity: columns are still present after the second (no-op) pass.
    cols = {row["name"] for row in
            EvalStore(eval_db).conn.execute("PRAGMA table_info(eval_epochs)")}
    assert "judge_method" in cols


def test_migrate_eval_schema_adds_agg_columns_to_legacy_db(tmp_path):
    """Legacy eval_case_agg must gain n_llm/n_fallback/n_no_report/fallback_mean."""
    eval_db = tmp_path / "eval.db"
    _build_legacy_db(eval_db)

    store = EvalStore(eval_db)
    cols = {row["name"] for row in
            store.conn.execute("PRAGMA table_info(eval_case_agg)")}
    assert {"n_llm", "n_fallback", "n_no_report", "fallback_mean"} <= cols

    # upsert_case_agg must succeed end-to-end on the migrated db.
    rid = store.create_eval_run(
        suite="s",
        judge_provider="j",
        judge_model=None,
        target_provider="t",
        repeat_n=1,
    )
    store.upsert_case_agg(
        eval_run_id=rid,
        case_id="c",
        n_epochs=2,
        n_llm=1,
        n_fallback=1,
        n_no_report=0,
        mean=0.7,
        std=0.0,
        stderr=0.0,
        min_score=0.7,
        max_score=0.7,
        fallback_mean=0.0,
        mean_grounded_ratio=1.0,
    )
    agg = store.list_case_aggs(rid)[0]
    assert agg["n_llm"] == 1
    assert agg["n_fallback"] == 1
    assert agg["n_no_report"] == 0
    assert agg["fallback_mean"] == 0.0
