from __future__ import annotations

from insightswarm.agents.agent_loop import AgentLoopState, run_agent_loop
from insightswarm.agents.agent_loop_contract import validate_tool_call
from insightswarm.agents.tool_executor import ToolExecutor


TOOLS = [
    {
        "name": "finish_research",
        "description": "Finish explicitly.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {"type": "object", "properties": {"terminal": {"type": "boolean"}}},
        "side_effects": "terminal",
    }
]


def test_validate_tool_call_rejects_invalid_enum_value() -> None:
    """Contract layer enforces enum constraints declared in input_schema.

    This is the shape-layer gate: missing/invalid reason values are rejected
    here, before the handler runs. Handlers only do stateful checks (e.g.
    quick_read history). Keeps the two layers cleanly separated.
    """
    specs = [
        {
            "name": "fetch_source",
            "description": "L2 fetch.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "reason": {
                        "type": "string",
                        "enum": ["verbatim_quote", "numeric_crosscheck", "legal_text", "controversial_claim", "snippet_insufficient"],
                    },
                },
                "required": ["url", "reason"],
            },
            "output_schema": {"type": "object"},
            "side_effects": "none",
        }
    ]
    # Missing reason → missing_required_input.
    v = validate_tool_call({"name": "fetch_source", "input": {"url": "https://x.com"}}, specs)
    assert v is not None
    assert v.failure_kind == "missing_required_input"
    # Invalid reason → invalid_enum_value.
    v = validate_tool_call({"name": "fetch_source", "input": {"url": "https://x.com", "reason": "because"}}, specs)
    assert v is not None
    assert v.failure_kind == "invalid_enum_value"
    # Valid reason → no violation.
    v = validate_tool_call({"name": "fetch_source", "input": {"url": "https://x.com", "reason": "verbatim_quote"}}, specs)
    assert v is None


def test_enforce_fast_path_commitment_rejects_blocked_after_fast_path_ready() -> None:
    """finish_research(blocked) is self-contradictory once a quick_read signaled fast_path_ready.

    Narrow gate: only blocks the "blocked" lie. Other paths stay open so the
    model can still legitimately deliver (finish_with_answer), search more
    (search_web/quick_read), or finish with a non-blocked status.
    """
    from insightswarm.agents.agent_loop_contract import enforce_fast_path_commitment

    # finish_research(blocked) → rejected.
    v = enforce_fast_path_commitment({"name": "finish_research", "input": {"status": "blocked", "reason": "stuck"}})
    assert v is not None
    assert v.failure_kind == "blocked_after_fast_path_ready"

    # finish_research(done) → allowed (not a "blocked" lie).
    v = enforce_fast_path_commitment({"name": "finish_research", "input": {"status": "done", "reason": "answered"}})
    assert v is None

    # finish_with_answer → allowed (intended fast-path terminal).
    v = enforce_fast_path_commitment({"name": "finish_with_answer", "input": {"answer": "x"}})
    assert v is None

    # search_web → allowed (model may want more sources).
    v = enforce_fast_path_commitment({"name": "search_web", "input": {"query": "x"}})
    assert v is None

    # quick_read → allowed (gather more before answering).
    v = enforce_fast_path_commitment({"name": "quick_read", "input": {"url": "https://x.com"}})
    assert v is None


def test_next_private_state_preserves_prior_state_on_model_error() -> None:
    """On model_error/timeout, prior reasoning state must NOT be overwritten.

    Regression for run-run_0ba88c6a5ef3: rounds 5/6 had current_understanding
    replaced by the timeout error text, because _next_private_state fell back
    to assistant_text (which carried the error message). The model then saw
    its own "understanding" as the timeout message and lost the quick_read
    context from round 4, leading to a spurious finish_research(blocked).

    Fix: error turns return previous_state unchanged. Recovery context still
    reaches the model via the model_error tool_result (history layer).
    """
    from insightswarm.agents.agent_loop import _next_private_state

    prior = {
        "current_understanding": "quick_read of caac.gov.cn shows fuel surcharge change",
        "plan": "finish_with_answer next",
    }
    error_turn = {
        "assistant_text": "OpenAI-compatible API request timed out: The read operation timed out",
        "tool_call": None,
        "stop_reason": "model_error",
    }
    new_state = _next_private_state(error_turn, prior)
    # Prior understanding preserved — error text did NOT leak into reasoning.
    assert new_state["current_understanding"] == prior["current_understanding"]
    assert "timed out" not in new_state["current_understanding"]


def test_next_private_state_falls_back_to_assistant_text_on_normal_turn() -> None:
    """Non-error turns without private_state still fall back to assistant_text.

    The fallback is for the rare case where the model returns assistant_text
    but omits the private_state field (a schema violation, but still a real
    model thought — not a runtime error). That path must stay intact.
    """
    from insightswarm.agents.agent_loop import _next_private_state

    normal_turn = {
        "assistant_text": "I will search for recent articles.",
        "tool_call": {"name": "search_web", "input": {"query": "x"}},
        "stop_reason": "ok",
    }
    new_state = _next_private_state(normal_turn, {"current_understanding": "old"})
    # Fallback engaged: assistant_text became current_understanding.
    assert new_state["current_understanding"] == "I will search for recent articles."


class _NoToolThenFinishModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _ModelResult({"assistant_text": "I am done but forgot the tool."})
        return _ModelResult({"tool_call": {"name": "finish_research", "input": {"status": "complete", "reason": "explicit finish"}}})


class _AlwaysNoToolModel:
    def complete(self, *args, **kwargs):
        return _ModelResult({"assistant_text": "No tool."})


class _CaptureMetadataModel:
    def __init__(self) -> None:
        self.metadata = None

    def complete(self, *args, **kwargs):
        self.metadata = dict(kwargs.get("metadata") or {})
        return _ModelResult({"tool_call": {"name": "finish_research", "input": {"status": "complete", "reason": "done"}}})


class _CaptureSystemPromptModel:
    def __init__(self) -> None:
        self.system_prompt = ""

    def complete(self, messages, *args, **kwargs):
        del args, kwargs
        self.system_prompt = str(messages[0]["content"])
        return _ModelResult({"tool_call": {"name": "finish_research", "input": {"status": "complete", "reason": "done"}}})


class _CapturePayloadModel:
    def __init__(self) -> None:
        self.payload = {}

    def complete(self, messages, *args, **kwargs):
        del args, kwargs
        self.payload = __import__("json").loads(messages[1]["content"])
        return _ModelResult({"tool_call": {"name": "finish_research", "input": {"status": "complete", "reason": "done"}}})


class _ModelResult:
    status = "ok"
    text = ""

    def __init__(self, json_data):
        self.json_data = json_data


def test_agent_loop_requires_explicit_finish_tool_after_no_tool_turn() -> None:
    state = AgentLoopState()
    trace, state = run_agent_loop(
        model_client=_NoToolThenFinishModel(),
        system_prompt="Use tools.",
        tool_specs=TOOLS,
        executor=ToolExecutor(TOOLS, {"finish_research": lambda _: {"ok": True, "terminal": True, "status": "done", "reason": "explicit finish"}}),
        initial_user_payload={"task": "demo"},
        state=state,
        safety_cap=5,
    )

    assert state.terminal_status == "done"
    assert trace[0]["failure_kind"] == "model_no_tool"
    assert trace[1]["tool_call"]["name"] == "finish_research"


def test_agent_loop_does_not_treat_repeated_no_tool_as_done() -> None:
    trace, state = run_agent_loop(
        model_client=_AlwaysNoToolModel(),
        system_prompt="Use tools.",
        tool_specs=TOOLS,
        executor=ToolExecutor(TOOLS, {"finish_research": lambda _: {"ok": True, "terminal": True, "status": "done"}}),
        initial_user_payload={"task": "demo"},
        safety_cap=5,
    )

    assert state.terminal_status == "model_no_tool"
    assert state.terminal_status != "done"
    assert len([item for item in trace if item["failure_kind"] == "model_no_tool"]) == 3


def test_agent_loop_attaches_audit_metadata() -> None:
    model = _CaptureMetadataModel()

    run_agent_loop(
        model_client=model,
        system_prompt="Use tools.",
        tool_specs=TOOLS,
        executor=ToolExecutor(TOOLS, {"finish_research": lambda _: {"ok": True, "terminal": True, "status": "done"}}),
        initial_user_payload={"task": "demo"},
        safety_cap=1,
        metadata_role="researcher_tool_loop",
        metadata={"run_id": "run_123", "task_id": "task_456", "operation": "researcher_tool_loop"},
    )

    assert model.metadata == {
        "run_id": "run_123",
        "task_id": "task_456",
        "operation": "researcher_tool_loop",
        "role": "researcher_tool_loop",
    }


def test_agent_loop_injects_shared_contract_into_system_prompt() -> None:
    model = _CaptureSystemPromptModel()

    run_agent_loop(
        model_client=model,
        system_prompt="You are a tiny role prompt.",
        tool_specs=TOOLS,
        executor=ToolExecutor(TOOLS, {"finish_research": lambda _: {"ok": True, "terminal": True, "status": "done"}}),
        initial_user_payload={"task": "demo"},
        safety_cap=1,
    )

    assert "Shared Agent Loop Contract:" in model.system_prompt
    assert "Role Prompt:\nYou are a tiny role prompt." in model.system_prompt


def test_agent_loop_sends_compact_model_facing_tool_specs() -> None:
    model = _CapturePayloadModel()

    run_agent_loop(
        model_client=model,
        system_prompt="Use tools.",
        tool_specs=TOOLS,
        executor=ToolExecutor(TOOLS, {"finish_research": lambda _: {"ok": True, "terminal": True, "status": "done"}}),
        initial_user_payload={"task": "demo"},
        safety_cap=1,
    )

    visible_tool = model.payload["tool_specs"][0]
    assert visible_tool["name"] == "finish_research"
    assert "input" in visible_tool
    assert "output_schema" not in visible_tool
    assert "side_effects" not in visible_tool


def test_agent_loop_summarizes_older_transcript_and_keeps_recent_tool_signal() -> None:
    model = _CapturePayloadModel()
    state = AgentLoopState()
    state.messages = [
        {"role": "tool", "content": {"tool_name": "search_web", "query": "old", "candidates": [{"url": f"https://old.example/{i}", "title": "Old", "snippet": "A" * 500} for i in range(5)]}},
        {"role": "assistant", "content": {"assistant_text": "older assistant " + ("B" * 500)}},
        {"role": "tool", "content": {"tool_name": "fetch_source", "document": {"url": "https://recent.example", "title": "Recent", "usable": True, "usability_reason": "sufficient_text_density", "text_preview": "C" * 900}}},
    ]

    run_agent_loop(
        model_client=model,
        system_prompt="Use tools.",
        tool_specs=TOOLS,
        executor=ToolExecutor(TOOLS, {"finish_research": lambda _: {"ok": True, "terminal": True, "status": "done"}}),
        initial_user_payload={"task": "demo"},
        state=state,
        safety_cap=1,
    )

    transcript = model.payload["recent_tool_transcript"]
    assert transcript[0]["content"]["candidate_count"] == 5
    assert transcript[0]["content"]["omitted_candidate_count"] == 2
    assert len(transcript[0]["content"]["candidates"]) == 3
    assert "older assistant" in transcript[1]["summary"]
    assert transcript[2]["content"]["document"]["url"] == "https://recent.example"
    assert len(transcript[2]["content"]["document"]["text_preview"]) <= 500


_NOOP_TOOLS = [
    {
        "name": "read_task",
        "description": "Read the assigned task.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {"type": "object"},
        "side_effects": "none",
    },
    {
        "name": "read_shared_memory",
        "description": "Read shared memory.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {"type": "object"},
        "side_effects": "none",
    },
    {
        "name": "finish_research",
        "description": "Finish explicitly.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {"type": "object", "properties": {"terminal": {"type": "boolean"}}},
        "side_effects": "terminal",
    },
]


class _AlternatingReadModel:
    """Model that alternates read_task and read_shared_memory — not identical calls but all no-ops.

    This tests the no-op spin detector independently of the repeated-call detector
    (which would catch identical consecutive calls at 2 instead of the stall policy's noop_limit=6).
    """

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        if self.calls % 2 == 1:
            return _ModelResult({"tool_call": {"name": "read_task", "input": {}}})
        return _ModelResult({"tool_call": {"name": "read_shared_memory", "input": {}}})


class _ReadTaskThenFinishModel:
    """Model that calls read_task a few times then finishes."""

    def __init__(self, noop_rounds: int = 3) -> None:
        self.noop_rounds = noop_rounds
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.noop_rounds:
            return _ModelResult({"tool_call": {"name": "read_task", "input": {}}})
        return _ModelResult({"tool_call": {"name": "finish_research", "input": {"status": "complete", "reason": "done"}}})


class _ReadTaskWithNewContentModel:
    """Model that calls read_task but the tool returns new messages — should NOT be flagged as spin."""

    def complete(self, *args, **kwargs):
        return _ModelResult({"tool_call": {"name": "read_task", "input": {}}})


def test_agent_loop_stops_noop_spin_after_limit() -> None:
    """Consecutive read-only calls with no new content must hard-stop at the stall policy's noop_limit."""
    from insightswarm.agents.agent_loop import DEFAULT_STALL_POLICY

    noop_limit = DEFAULT_STALL_POLICY.noop_limit

    def _read_task_handler(_):
        # Empty result — no new content, simulates re-reading unchanged state.
        return {"ok": True, "tool_name": "read_task", "terminal": False}

    trace, state = run_agent_loop(
        model_client=_AlternatingReadModel(),
        system_prompt="Use tools.",
        tool_specs=_NOOP_TOOLS,
        executor=ToolExecutor(_NOOP_TOOLS, {
            "read_task": _read_task_handler,
            "read_shared_memory": _read_task_handler,
            "finish_research": lambda _: {"ok": True, "terminal": True, "status": "done"},
        }),
        initial_user_payload={"task": "demo"},
        safety_cap=50,
    )

    assert state.terminal_status == "blocked"
    assert "no-op spin" in (state.terminal_reason or "")
    # Should stop at exactly noop_limit rounds, not spin to safety_cap.
    assert len(trace) == noop_limit
    assert state.consecutive_noop_count == noop_limit


def test_agent_loop_does_not_flag_read_with_new_content_as_spin() -> None:
    """A read-only tool that returns new messages should reset the noop counter.

    Uses alternating read_task / read_shared_memory to avoid the repeated-call
    detector (which would reject identical consecutive calls at 2).
    """

    def _read_handler(_):
        # Returns new messages — productive read, not a no-op.
        return {"ok": True, "tool_name": "read_task", "terminal": False, "messages": [{"id": "msg_1", "body": "new info"}]}

    class _AlternatingReadWithContentModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            if self.calls > 4:
                return _ModelResult({"tool_call": {"name": "finish_research", "input": {"status": "complete", "reason": "done"}}})
            if self.calls % 2 == 1:
                return _ModelResult({"tool_call": {"name": "read_task", "input": {}}})
            return _ModelResult({"tool_call": {"name": "read_shared_memory", "input": {}}})

    trace, state = run_agent_loop(
        model_client=_AlternatingReadWithContentModel(),
        system_prompt="Use tools.",
        tool_specs=_NOOP_TOOLS,
        executor=ToolExecutor(_NOOP_TOOLS, {
            "read_task": _read_handler,
            "read_shared_memory": _read_handler,
            "finish_research": lambda _: {"ok": True, "terminal": True, "status": "done", "reason": "done"},
        }),
        initial_user_payload={"task": "demo"},
        safety_cap=50,
    )

    # All 4 read calls returned new content, so no spin — should finish cleanly.
    assert state.terminal_status == "done"
    assert state.consecutive_noop_count == 0


# --- Tool-failure spin regression (run-run_ac4eb4e41942, 2026-06-23) ---

_FAILURE_TOOLS = [
    {
        "name": "request_repair",
        "description": "Request repair (terminal tool that may fail recoverably).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {"type": "object"},
        "side_effects": "terminal",
    },
    {
        "name": "finish_research",
        "description": "Finish explicitly.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {"type": "object", "properties": {"terminal": {"type": "boolean"}}},
        "side_effects": "terminal",
    },
]


class _AlwaysFailRequestRepairModel:
    """Model that always calls request_repair with varying args, which always returns missing_review_basis.

    Varies the input to avoid the repeated-call detector (which would reject
    identical consecutive calls). This isolates the tool-failure spin detector.
    """

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        return _ModelResult({"tool_call": {"name": "request_repair", "input": {"attempt": self.calls}}})


def test_agent_loop_stops_tool_failure_spin_at_tightest_limit() -> None:
    """Consecutive non-terminal tool failures with failure_kind must hard-stop at the stall policy's tool_failure_limit (2)."""
    from insightswarm.agents.agent_loop import DEFAULT_STALL_POLICY

    tool_failure_limit = DEFAULT_STALL_POLICY.tool_failure_limit

    def _fail_handler(_):
        return {
            "ok": False,
            "tool_name": "request_repair",
            "terminal": False,
            "error": "request_repair requires review_basis",
            "missing_review_basis": True,
            "failure_kind": "missing_review_basis",
        }

    trace, state = run_agent_loop(
        model_client=_AlwaysFailRequestRepairModel(),
        system_prompt="Use tools.",
        tool_specs=_FAILURE_TOOLS,
        executor=ToolExecutor(_FAILURE_TOOLS, {
            "request_repair": _fail_handler,
            "finish_research": lambda _: {"ok": True, "terminal": True, "status": "done"},
        }),
        initial_user_payload={"task": "demo"},
        safety_cap=50,
    )

    assert state.terminal_status == "blocked"
    assert "no-op spin" in (state.terminal_reason or "")
    assert "tool-failure" in (state.terminal_reason or "")
    # Must stop at tool_failure_limit=2, not spin to safety_cap.
    assert len(trace) == tool_failure_limit
    assert state.consecutive_noop_count == tool_failure_limit


def test_agent_loop_productive_tool_resets_failure_counter() -> None:
    """A successful tool between failures must reset the failure counter."""

    class _FailThenFinishModel:
        def __init__(self):
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            # 1 failure, then finish — should never hit the limit of 2.
            if self.calls == 1:
                return _ModelResult({"tool_call": {"name": "request_repair", "input": {}}})
            return _ModelResult({"tool_call": {"name": "finish_research", "input": {"status": "complete", "reason": "done"}}})

    def _fail_handler(_):
        return {
            "ok": False,
            "tool_name": "request_repair",
            "terminal": False,
            "error": "request_repair requires review_basis",
            "missing_review_basis": True,
            "failure_kind": "missing_review_basis",
        }

    trace, state = run_agent_loop(
        model_client=_FailThenFinishModel(),
        system_prompt="Use tools.",
        tool_specs=_FAILURE_TOOLS,
        executor=ToolExecutor(_FAILURE_TOOLS, {
            "request_repair": _fail_handler,
            "finish_research": lambda _: {"ok": True, "terminal": True, "status": "done", "reason": "done"},
        }),
        initial_user_payload={"task": "demo"},
        safety_cap=50,
    )

    assert state.terminal_status == "done"
    assert state.consecutive_noop_count == 0


def test_agent_loop_productive_tool_resets_recompute_counter() -> None:
    """A productive tool between no-op reads should reset the counter."""

    class _MixedModel:
        def __init__(self):
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            # 3 no-op reads, 1 productive (finish), repeat — should never hit limit.
            if self.calls % 4 == 0:
                return _ModelResult({"tool_call": {"name": "finish_research", "input": {"status": "complete", "reason": "done"}}})
            return _ModelResult({"tool_call": {"name": "read_task", "input": {}}})

    call_count = {"read": 0}

    def _read_task_handler(_):
        call_count["read"] += 1
        return {"ok": True, "tool_name": "read_task", "terminal": False}

    trace, state = run_agent_loop(
        model_client=_MixedModel(),
        system_prompt="Use tools.",
        tool_specs=_NOOP_TOOLS,
        executor=ToolExecutor(_NOOP_TOOLS, {
            "read_task": _read_task_handler,
            "finish_research": lambda _: {"ok": True, "terminal": True, "status": "done", "reason": "done"},
        }),
        initial_user_payload={"task": "demo"},
        safety_cap=50,
    )

    # finish_research is terminal, so the loop ends on the first finish call (round 4).
    assert state.terminal_status == "done"
    # Counter never reached the limit because finish_* reset it.
    assert state.consecutive_noop_count == 0


def test_failure_policy_classifies_noop_spin_as_technical_non_repairable() -> None:
    """noop_spin must not trigger critic review or research repair — it's a technical stall."""
    from insightswarm.agents.failure_policy import normalize_agent_failure

    failure = normalize_agent_failure(
        status="blocked",
        reason="no-op spin detected: 6 consecutive read-only tool calls produced no new information.",
    )
    assert failure.category == "technical"
    assert failure.retryable is False
    assert failure.should_trigger_critic_review is False
    assert failure.should_trigger_research_repair is False


_RANK_TOOLS = [
    {
        "name": "rank_sources",
        "description": "Rank candidate sources.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {"type": "object"},
        "side_effects": "none",
    },
    {
        "name": "finish_research",
        "description": "Finish explicitly.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {"type": "object", "properties": {"terminal": {"type": "boolean"}}},
        "side_effects": "terminal",
    },
]


class _AlwaysRankSourcesModel:
    """Model that calls rank_sources with varying args — simulates the 2026-06-23 re-rank spin.

    Varies the input to avoid the repeated-call detector (which would reject
    identical consecutive calls). This isolates the re-compute spin detector.
    """

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        return _ModelResult({"tool_call": {"name": "rank_sources", "input": {"attempt": self.calls}}})


def test_agent_loop_stops_recompute_spin_at_tighter_limit() -> None:
    """Consecutive rank_sources calls must hard-stop at the stall policy's recompute_limit (3), not noop_limit (6)."""
    from insightswarm.agents.agent_loop import DEFAULT_STALL_POLICY

    recompute_limit = DEFAULT_STALL_POLICY.recompute_limit

    def _rank_handler(_):
        return {"ok": True, "tool_name": "rank_sources", "terminal": False, "ranked_sources": [{"url": "x", "score": 1}]}

    trace, state = run_agent_loop(
        model_client=_AlwaysRankSourcesModel(),
        system_prompt="Use tools.",
        tool_specs=_RANK_TOOLS,
        executor=ToolExecutor(_RANK_TOOLS, {
            "rank_sources": _rank_handler,
            "finish_research": lambda _: {"ok": True, "terminal": True, "status": "done"},
        }),
        initial_user_payload={"task": "demo"},
        safety_cap=50,
    )

    assert state.terminal_status == "blocked"
    assert "no-op spin" in (state.terminal_reason or "")
    assert "re-compute" in (state.terminal_reason or "")
    # Must stop at recompute_limit, not spin to noop_limit or safety_cap.
    assert len(trace) == recompute_limit
    assert state.consecutive_noop_count == recompute_limit


def test_agent_loop_productive_tool_resets_recompute_counter() -> None:
    """A productive tool between rank_sources calls must reset the recompute counter."""

    class _RankThenFinishModel:
        def __init__(self):
            self.calls = 0

        def complete(self, *args, **kwargs):
            self.calls += 1
            # 2 ranks, then finish — should never hit the limit of 3.
            if self.calls <= 2:
                return _ModelResult({"tool_call": {"name": "rank_sources", "input": {}}})
            return _ModelResult({"tool_call": {"name": "finish_research", "input": {"status": "complete", "reason": "done"}}})

    def _rank_handler(_):
        return {"ok": True, "tool_name": "rank_sources", "terminal": False, "ranked_sources": [{"url": "x", "score": 1}]}

    trace, state = run_agent_loop(
        model_client=_RankThenFinishModel(),
        system_prompt="Use tools.",
        tool_specs=_RANK_TOOLS,
        executor=ToolExecutor(_RANK_TOOLS, {
            "rank_sources": _rank_handler,
            "finish_research": lambda _: {"ok": True, "terminal": True, "status": "done", "reason": "done"},
        }),
        initial_user_payload={"task": "demo"},
        safety_cap=50,
    )

    assert state.terminal_status == "done"
    assert state.consecutive_noop_count == 0
