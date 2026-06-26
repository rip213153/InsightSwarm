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
    freshness: str | None = None,
    confidence: float = 0.9,
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
        freshness=freshness,
        confidence=confidence,
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
    assert result["coverage"]["source_count"] == 2
    assert result["coverage"]["largest_source_share"] == 0.6
    assert "source_role_hint_counts" in result["coverage"]
    assert "source_category" not in result["sources"][0]
    assert "source_role_hint" in result["sources"][0]
    dominant_sources = [source for source in result["sources"] if "source_dominates_bundle" in source["risk_flags"]]
    assert len(dominant_sources) == 1
    assert "missing_freshness" in dominant_sources[0]["risk_flags"]
    assert all("non_primary_source" not in source["risk_flags"] for source in result["sources"])
    assert len(result["sources"][0]["representative_quotes"]) <= 2
    assert validation == {"ok": True, "passed": True, "must_fix": []}
    assert len(store.list_swarm_messages(run_state.run_id)) == before_messages


def test_critic_repair_budget_blocks_after_one_review_repair(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="Budget test",
        budget={"max_steps": 12},
        phase="review",
    )
    review_task = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="leased",
        owner_role="critic",
        inputs={"evidence_scope": "run", "evidence_ids": [], "question": run_state.objective},
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
    review_basis = {
        "quote_integrity": "fail",
        "claim_alignment": "unclear",
        "coverage_for_review_scope": "missing",
        "source_concentration": "unclear",
        "freshness_fit": "missing",
        "tensions": "none",
        "review_disposition": "repair",
        "disposition_reason": "No usable evidence exists.",
        "must_fix": ["need quote-backed evidence covering the core question"],
    }

    first = handlers.request_repair(
        {
            "targeted_query": "find quote-backed source covering the core question",
            "must_fix": ["need quote-backed evidence covering the core question"],
            "why_current_evidence_failed": "nothing usable",
            "review_basis": review_basis,
        }
    )
    second = handlers.request_repair(
        {
            "targeted_query": "find another quote-backed source covering the core question",
            "must_fix": ["need another quote-backed source covering the core question"],
            "why_current_evidence_failed": "still nothing usable",
            "review_basis": review_basis,
        }
    )

    assert first["ok"] is True
    assert first["repair_created"] is True
    repair_task = store.get_swarm_task(first["repair_task_id"])
    assert repair_task.inputs["review_basis"]["review_disposition"] == "repair"
    assert second["ok"] is False
    assert "budget" in second["error"]


def test_critic_repair_requires_review_basis(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="Missing review basis test",
        budget={"max_steps": 12},
        phase="review",
    )
    review_task = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="leased",
        owner_role="critic",
        inputs={"evidence_scope": "run", "evidence_ids": [], "question": run_state.objective},
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

    result = handlers.request_repair(
        {
            "targeted_query": "find quote-backed source covering the core question",
            "must_fix": ["need quote-backed evidence covering the core question"],
            "why_current_evidence_failed": "nothing usable",
        }
    )

    assert result["ok"] is False
    assert result["missing_review_basis"] is True
    assert store.list_swarm_tasks(run_state.run_id) == [review_task]


def test_critic_pass_with_caveats_surfaces_verdict(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="Pass caveat test",
        budget={"max_steps": 12},
        phase="review",
    )
    review_task = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="leased",
        owner_role="critic",
        inputs={"evidence_scope": "run", "evidence_ids": [], "question": run_state.objective},
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

    result = handlers.mark_review_passed(
        {
            "reason": "usable but incomplete",
            "verdict": "pass_with_caveats",
            "caveats": ["needs future monitoring"],
            "review_basis": {
                "quote_integrity": "pass",
                "claim_alignment": "aligned",
                "coverage_for_review_scope": "partial",
                "source_concentration": "unclear",
                "freshness_fit": "unclear",
                "tensions": "none",
                "review_disposition": "pass_with_caveats",
                "disposition_reason": "Usable but incomplete.",
                "caveats": ["needs future monitoring"],
            },
        }
    )

    assert result["ok"] is True
    assert mailbox.inbox(run_state.run_id, role="lead")[-1].payload["verdict"] == "pass_with_caveats"


def test_critic_terminal_tools_record_review_basis(tmp_path: Path) -> None:
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="Review basis test",
        budget={"max_steps": 12},
        phase="review",
    )
    review_task = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="leased",
        owner_role="critic",
        inputs={"evidence_scope": "run", "evidence_ids": [], "question": run_state.objective},
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
    review_basis = {
        "quote_integrity": "pass",
        "claim_alignment": "aligned",
        "coverage_for_review_scope": "partial",
        "source_concentration": "single_source",
        "freshness_fit": "missing",
        "tensions": "none",
        "review_disposition": "pass_with_caveats",
        "disposition_reason": "Usable with freshness caveat.",
        "caveats": ["freshness missing"],
    }

    result = handlers.mark_review_passed(
        {
            "reason": "usable with caveat",
            "verdict": "pass_with_caveats",
            "caveats": ["freshness missing"],
            "review_basis": review_basis,
        }
    )

    assert result["ok"] is True
    payload = mailbox.inbox(run_state.run_id, role="lead")[-1].payload
    assert payload["review_basis"]["review_disposition"] == "pass_with_caveats"
    assert payload["review_basis"]["source_concentration"] == "single_source"


def test_critic_review_basis_accepts_non_whitelist_fields(tmp_path: Path) -> None:
    """review_basis must accept any non-empty dict, not just whitelist fields.

    Regression for run-run_ac4eb4e41942 (2026-06-23): the model filled
    verdict/reason/confidence (reasonable fields), but _review_basis() had a
    whitelist that excluded them, returning {} and causing every terminal tool
    to retry forever with missing_review_basis=True.
    """
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="Non-whitelist review_basis test",
        budget={"max_steps": 12},
        phase="review",
    )
    review_task = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="leased",
        owner_role="critic",
        inputs={"evidence_scope": "run", "evidence_ids": [], "question": run_state.objective},
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
    # Model fills verdict/reason/confidence — NOT in the old whitelist.
    review_basis = {
        "verdict": "repair",
        "reason": "No usable evidence for the question.",
        "confidence": 0.95,
        "review_type": "evidence_review",
    }

    result = handlers.request_repair(
        {
            "targeted_query": "find a source covering the question",
            "must_fix": ["need quote-backed evidence"],
            "why_current_evidence_failed": "nothing usable",
            "review_basis": review_basis,
        }
    )

    assert result["ok"] is True
    assert result["repair_created"] is True
    repair_task = store.get_swarm_task(result["repair_task_id"])
    # The full review_basis must be preserved, including non-whitelist fields.
    assert repair_task.inputs["review_basis"]["verdict"] == "repair"
    assert repair_task.inputs["review_basis"]["confidence"] == 0.95


def test_critic_repair_task_inherits_original_question(tmp_path: Path) -> None:
    """Repair task must carry the original review question so researcher keeps language/locale.

    Regression for run-run_ac4eb4e41942 (2026-06-23): critic's targeted_query
    drifted from Chinese "2026年中国航空燃油费调整" to English "Find a source
    covering fuel surcharge policy changes in 2026", causing the researcher to
    search US DOT/FMCSA sources for a China aviation question.
    """
    store = _build_store(tmp_path)
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    artifact_store = ArtifactStore(store)
    board_store = BoardStore(store)
    run_state = store.create_swarm_run_state(
        objective="2026年中国航空燃油费为什么屡次提高",
        budget={"max_steps": 12},
        phase="review",
    )
    original_question = "2026年中国航空燃油费为什么屡次提高"
    review_task = task_store.create(
        run_state.run_id,
        kind="evidence_review",
        status="leased",
        owner_role="critic",
        inputs={"evidence_scope": "run", "evidence_ids": [], "question": original_question},
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
    # Critic writes targeted_query in English (semantic drift).
    drifted_query = "Find a source covering fuel surcharge policy changes in 2026"

    result = handlers.request_repair(
        {
            "targeted_query": drifted_query,
            "must_fix": ["need a source covering 2026 fuel surcharge changes"],
            "why_current_evidence_failed": "document was from 2022",
            "review_basis": {"verdict": "repair", "reason": "temporal mismatch"},
        }
    )

    assert result["ok"] is True
    repair_task = store.get_swarm_task(result["repair_task_id"])
    # The original Chinese question must be preserved in the repair task.
    assert repair_task.inputs["question"] == original_question
    # The drifted targeted_query is also kept as the specific repair directive.
    assert repair_task.inputs["targeted_query"] == drifted_query
