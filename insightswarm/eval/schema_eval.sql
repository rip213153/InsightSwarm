-- Evaluation database schema. Lives in a separate SQLite file (default
-- .insightswarm/eval.db) so evaluation state never mixes with production run
-- state. swarm_run_id columns are logical references into the main DB; there
-- is no cross-database foreign key (SQLite cannot enforce one).

CREATE TABLE IF NOT EXISTS eval_runs (
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

-- One row per (case, epoch): a single complete swarm run scored by the judge.
CREATE TABLE IF NOT EXISTS eval_epochs (
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
  judge_method TEXT NOT NULL DEFAULT 'llm',
  human_score REAL,
  human_label TEXT,
  human_comment TEXT,
  created_at TEXT NOT NULL
);

-- Aggregate over epochs for a (eval_run, case): mean / dispersion.
-- ``mean``/``std``/``stderr``/``min_score``/``max_score`` cover ONLY the
-- LLM-judged epochs (judge_method='llm'); fallback / no_report epochs are
-- excluded from the main score so a judge outage does not pollute the mean.
-- ``n_epochs`` is the total epoch count; ``n_llm``/``n_fallback``/``n_no_report``
-- break that total down by judge_method so callers can see how much of the
-- suite was actually LLM-judged. ``fallback_mean`` summarizes the fallback
-- subset separately (NULL when there were no fallback epochs).
CREATE TABLE IF NOT EXISTS eval_case_agg (
  eval_run_id TEXT NOT NULL REFERENCES eval_runs(eval_run_id),
  case_id TEXT NOT NULL,
  n_epochs INTEGER NOT NULL,
  n_llm INTEGER NOT NULL DEFAULT 0,
  n_fallback INTEGER NOT NULL DEFAULT 0,
  n_no_report INTEGER NOT NULL DEFAULT 0,
  mean REAL,
  std REAL,
  stderr REAL,
  min_score REAL,
  max_score REAL,
  fallback_mean REAL,
  mean_grounded_ratio REAL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (eval_run_id, case_id)
);

-- Per-citation grounding detail, one row per checked quote.
CREATE TABLE IF NOT EXISTS eval_citation_checks (
  check_id TEXT PRIMARY KEY,
  epoch_id TEXT NOT NULL REFERENCES eval_epochs(epoch_id),
  source_url TEXT,
  quote TEXT,
  claim TEXT,
  match_type TEXT NOT NULL,
  similarity REAL NOT NULL,
  matched INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_eval_epochs_run_case ON eval_epochs(eval_run_id, case_id, epoch_idx);
CREATE INDEX IF NOT EXISTS idx_eval_citation_checks_epoch ON eval_citation_checks(epoch_id, matched);
