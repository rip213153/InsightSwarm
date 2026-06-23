from __future__ import annotations

import json

from insightswarm.agents.agent_loop import AgentLoopState
from insightswarm.agents.trace import build_tool_trace_callback
from insightswarm.schemas.swarm import Task


def test_critic_trace_includes_review_state_summary(tmp_path) -> None:
    trace_path = tmp_path / "steps.jsonl"
    task = Task(
        task_id="task_critic",
        run_id="run_trace",
        kind="evidence_review",
        status="leased",
        owner_role="critic",
        inputs={},
    )
    state = AgentLoopState(
        private_state={
            "review_focus": "coverage",
            "findings_so_far": {"coverage_for_review_scope": "partial"},
            "open_questions": ["freshness"],
            "review_confidence": "medium",
            "likely_disposition": "pass_with_caveats",
            "review_basis": {"review_disposition": "pass_with_caveats"},
            "plan": "mark pass with caveats",
        }
    )
    callback = build_tool_trace_callback(trace_path, role="critic", task=task)

    assert callback is not None
    callback(
        3,
        {"name": "mark_review_passed", "input": {"review_basis": state.private_state["review_basis"]}},
        {"ok": True, "message_id": "msg_1", "review_basis": state.private_state["review_basis"]},
        state,
    )

    record = json.loads(trace_path.read_text(encoding="utf-8").strip())

    assert record["critic_review_state"]["review_focus"] == "coverage"
    assert record["critic_review_state"]["review_basis"]["review_disposition"] == "pass_with_caveats"
    assert record["tool_result"]["review_basis"]["review_disposition"] == "pass_with_caveats"
