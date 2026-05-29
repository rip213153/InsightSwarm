CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS phases (
  phase_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  order_index INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
  task_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  phase TEXT NOT NULL,
  agent_name TEXT NOT NULL,
  status TEXT NOT NULL,
  depends_on_json TEXT NOT NULL DEFAULT '[]',
  retry_count INTEGER NOT NULL DEFAULT 0,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  task_id TEXT REFERENCES tasks(task_id),
  artifact_type TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  source_url TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS citations (
  citation_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  task_id TEXT REFERENCES tasks(task_id),
  source_type TEXT NOT NULL,
  artifact_id TEXT REFERENCES artifacts(artifact_id),
  source_url TEXT,
  quote TEXT,
  text_span_json TEXT,
  image_bbox_json TEXT,
  evidence_ids_json TEXT NOT NULL DEFAULT '[]',
  claim TEXT,
  confidence REAL NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  message_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  task_id TEXT REFERENCES tasks(task_id),
  sender TEXT NOT NULL,
  recipient TEXT NOT NULL,
  status TEXT NOT NULL,
  lease_owner TEXT,
  leased_at TEXT,
  lease_expires_at TEXT,
  acked_at TEXT,
  idempotency_key TEXT NOT NULL UNIQUE,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_calls (
  model_call_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  task_id TEXT REFERENCES tasks(task_id),
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  request_json TEXT NOT NULL,
  response_json TEXT NOT NULL,
  usage_json TEXT NOT NULL DEFAULT '{}',
  latency_ms INTEGER NOT NULL,
  status TEXT NOT NULL,
  error TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_events (
  event_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(run_id),
  task_id TEXT REFERENCES tasks(task_id),
  agent_name TEXT NOT NULL,
  event_type TEXT NOT NULL,
  message TEXT NOT NULL,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_run_phase ON tasks(run_id, phase);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_citations_run ON citations(run_id);
CREATE INDEX IF NOT EXISTS idx_messages_status ON messages(run_id, status);
CREATE INDEX IF NOT EXISTS idx_events_run ON agent_events(run_id, created_at);

CREATE TABLE IF NOT EXISTS swarm_run_states (
  run_id TEXT PRIMARY KEY,
  objective TEXT NOT NULL,
  phase TEXT NOT NULL,
  budget_json TEXT NOT NULL DEFAULT '{}',
  stop_reason TEXT,
  delivery_gate INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS swarm_tasks (
  task_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES swarm_run_states(run_id),
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  owner_role TEXT NOT NULL,
  inputs_json TEXT NOT NULL DEFAULT '{}',
  depends_on_json TEXT NOT NULL DEFAULT '[]',
  priority INTEGER NOT NULL DEFAULT 0,
  lease_until TEXT,
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS swarm_messages (
  message_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES swarm_run_states(run_id),
  from_role TEXT NOT NULL,
  to_role TEXT,
  broadcast INTEGER NOT NULL DEFAULT 0,
  intent TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  related_task_id TEXT REFERENCES swarm_tasks(task_id),
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS swarm_artifacts (
  artifact_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES swarm_run_states(run_id),
  type TEXT NOT NULL,
  status TEXT NOT NULL,
  source_task_id TEXT REFERENCES swarm_tasks(task_id),
  payload_ref TEXT NOT NULL,
  summary TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS swarm_evidence (
  evidence_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES swarm_run_states(run_id),
  artifact_id TEXT NOT NULL REFERENCES swarm_artifacts(artifact_id),
  source_url TEXT NOT NULL,
  quote TEXT NOT NULL,
  freshness TEXT,
  confidence REAL NOT NULL,
  qa_state TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS swarm_board_items (
  item_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES swarm_run_states(run_id),
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  title TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  parent_id TEXT REFERENCES swarm_board_items(item_id),
  evidence_id TEXT REFERENCES swarm_evidence(evidence_id),
  artifact_id TEXT REFERENCES swarm_artifacts(artifact_id),
  source_task_id TEXT REFERENCES swarm_tasks(task_id),
  dedupe_key TEXT,
  priority INTEGER NOT NULL DEFAULT 0,
  created_by TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_swarm_tasks_run_owner ON swarm_tasks(run_id, owner_role, status);
CREATE INDEX IF NOT EXISTS idx_swarm_messages_run_target ON swarm_messages(run_id, to_role, broadcast, created_at);
CREATE INDEX IF NOT EXISTS idx_swarm_artifacts_run ON swarm_artifacts(run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_swarm_evidence_run ON swarm_evidence(run_id, qa_state, created_at);
CREATE INDEX IF NOT EXISTS idx_swarm_board_run_kind ON swarm_board_items(run_id, kind, status, priority, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_swarm_board_run_dedupe ON swarm_board_items(run_id, dedupe_key) WHERE dedupe_key IS NOT NULL;
