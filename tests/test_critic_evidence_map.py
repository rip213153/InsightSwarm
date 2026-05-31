from __future__ import annotations

from pathlib import Path

from insightswarm.agents.critic_tools import CriticToolHandlers, CriticToolState
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore


def _build_store(tmp_path: Path) -> Store:
    db_path = tmp_path / "insightswarm.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    return Store(db_path, artifact_dir)


def _record_evidence(
    store: Store,
    run_id: str,
    *,
    source_task_id: str,
    source_url: str,
    claim: str,
    quote: str,
) -> str:
    artifact_store = ArtifactStore(store)
    board_store = BoardStore(store)
    citation = artifact_store.write_citation(
        run_id,
        source_task_id=source_task_id,
        citation={"source_url": source_url, "claim": claim, "quote": quote, "text": quote},
        summary=claim,
    )
    evidence = artifact_store.create_evidence(
        run_id,
        artifact_id=citation.artifact_id or "",
        source_url=source_url,
        quote=quote,
        freshness=None,
        confidence=0.9,
        qa_state="ready",
    )
    board_store.record_evidence(
        run_id,
        evidence=evidence,
        question_id=None,
        artifact_id=citation.artifact_id,
        source_task_id=source_task_id,
    )
    return evidence.evidence_id or ""


def test_critic_evidence_map_summarizes_large_bundle_without_writing(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="What does OpenAI want to do next?",
        budget={"max_steps": 12},
        phase="review",
    )
    evidence_ids = []
    for index in range(10):
        source_task = task_store.create(
            run_state.run_id,
            kind="raw_document",
            status="done",
            owner_role="extractor",
            inputs={"index": index},
            created_by="test",
        )
        evidence_ids.append(
            _record_evidence(
                store,
                run_state.run_id,
                source_task_id=source_task.task_id or "",
                source_url="https://blog.samaltman.com/reflections" if index < 6 else "https://arstechnica.com/openai-gpt-5-2",
                claim=f"OpenAI future plan claim {index}",
                quote=f"Quote {index} about AI agents, AGI, release, or superintelligence.",
            )
        )
    review_task = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="leased",
        owner_role="critic",
        inputs={"evidence_scope": "run", "evidence_ids": evidence_ids, "question": run_state.objective},
        created_by="test",
    )
    handlers = CriticToolHandlers(
        task=review_task,
        task_store=task_store,
        mailbox=mailbox,
        artifact_store=artifact_store,
        board_store=board_store,
        state=CriticToolState(),
    )

    before_messages = len(store.list_swarm_messages(run_state.run_id))
    result = handlers.read_evidence_map({})
    validation = handlers.validate_evidence_bundle({})

    assert result["ok"] is True
    assert result["evidence_count"] == 10
    assert result["source_count"] == 2
    assert result["coverage"]["primary_sources"] == 1
    assert result["coverage"]["independent_sources"] == 2
    assert len(result["sources"][0]["representative_quotes"]) <= 2
    assert validation == {"ok": True, "passed": True, "must_fix": []}
    assert len(store.list_swarm_messages(run_state.run_id)) == before_messages
