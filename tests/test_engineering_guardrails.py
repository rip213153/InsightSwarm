from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_code_does_not_read_scripted_fixture_env() -> None:
    offenders: list[str] = []
    for path in (ROOT / "insightswarm").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "INSIGHTSWARM_SCRIPTED_FIXTURE" in text or "_fixture_payload" in text:
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_critic_source_quality_rules_do_not_use_sample_domains_or_primary_labels() -> None:
    text = (ROOT / "insightswarm" / "agents" / "critic_tools.py").read_text(encoding="utf-8").lower()
    forbidden = [
        "openai.com",
        "blog.samaltman.com",
        "reuters.com",
        "nytimes.com",
        "reddit.com",
        "zhihu.com",
        "primary_source",
        "secondary_source",
        "non_primary_source",
    ]

    assert [item for item in forbidden if item in text] == []


def test_runtime_schema_does_not_recreate_legacy_table_family() -> None:
    schema = (ROOT / "insightswarm" / "db" / "schema.sql").read_text(encoding="utf-8").lower()
    forbidden = [
        "create table if not exists runs",
        "create table if not exists phases",
        "create table if not exists tasks",
        "create table if not exists artifacts",
        "create table if not exists citations",
        "create table if not exists messages",
        "create table if not exists agent_events",
        "references runs",
        "references tasks",
        "references artifacts",
    ]

    assert [item for item in forbidden if item in schema] == []
    assert "swarm_task_id text references swarm_tasks(task_id)" in schema


def test_store_does_not_expose_legacy_runtime_methods() -> None:
    text = (ROOT / "insightswarm" / "db" / "store.py").read_text(encoding="utf-8")
    forbidden = [
        "def create_run(",
        "def create_task(",
        "def write_artifact(",
        "def list_citations(",
        "def list_artifacts(",
        "def emit_event(",
        "def create_message(",
        "def lease_messages(",
        "def ack_messages(",
    ]

    assert [item for item in forbidden if item in text] == []


def test_role_prompts_do_not_duplicate_agent_loop_contract() -> None:
    offenders: dict[str, list[str]] = {}
    forbidden = [
        "Return JSON only",
        "Every round, return one JSON object",
        '"tool_call": {',
        '"name": "one exact tool name from tool_specs"',
        "Return exactly one `tool_call` per round",
    ]
    for path in (ROOT / "insightswarm" / "prompts").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        hits = [item for item in forbidden if item in text]
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits

    assert offenders == {}
