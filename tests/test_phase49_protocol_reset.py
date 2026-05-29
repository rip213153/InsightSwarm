from __future__ import annotations

from pathlib import Path

import pytest

from insightswarm.agents.critic import CriticWorker
from insightswarm.agents.extractor import ExtractorWorker
from insightswarm.agents.sub_researcher import SubResearcherWorker
from insightswarm.agents.writer import WriterWorker
from insightswarm.message_protocol import MESSAGE_TYPES, PAYLOAD_KINDS, validate_message


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_message_protocol_only_allows_five_top_level_types() -> None:
    assert MESSAGE_TYPES == {"request", "response", "observation", "suggestion", "hypothesis"}


def test_validate_message_rejects_legacy_or_unscoped_messages() -> None:
    with pytest.raises(ValueError, match="unsupported message type"):
        validate_message(
            message_type="handoff",
            payload={"kind": "research_subquestion"},
            related_task_id="task_1",
        )

    with pytest.raises(ValueError, match="unsupported payload.kind"):
        validate_message(
            message_type="request",
            payload={"kind": "legacy_handoff"},
            related_task_id="task_1",
        )

    with pytest.raises(ValueError, match="message requires"):
        validate_message(
            message_type="observation",
            payload={"kind": "progress_update"},
            related_task_id=None,
        )


def test_protocol_whitelists_keep_suggestion_non_executable_and_hypothesis_non_evidence() -> None:
    assert "extract_evidence" not in PAYLOAD_KINDS["suggestion"]
    assert "review_evidence" not in PAYLOAD_KINDS["suggestion"]
    assert "evidence" not in PAYLOAD_KINDS["hypothesis"]
    assert "citation" not in PAYLOAD_KINDS["hypothesis"]


def test_non_browser_workers_expose_ooda_structure_methods() -> None:
    for worker_cls in (SubResearcherWorker, ExtractorWorker, CriticWorker, WriterWorker):
        for method_name in ("_assemble_context", "_decide_action", "_write_state"):
            assert hasattr(worker_cls, method_name), f"{worker_cls.__name__} is missing {method_name}"


def test_new_code_no_longer_uses_legacy_message_intent_api() -> None:
    hits: list[str] = []
    legacy_kwarg = "intent" + "="
    legacy_attr = "." + "intent"
    for path in list((REPO_ROOT / "insightswarm").rglob("*.py")) + list((REPO_ROOT / "tests").rglob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if legacy_kwarg in text or legacy_attr in text:
            hits.append(str(path))

    assert hits == []
