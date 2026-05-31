from __future__ import annotations

from insightswarm.agents.agent_loop import AgentLoopState, run_agent_loop
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
