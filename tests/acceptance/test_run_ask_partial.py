from __future__ import annotations

import json

from insightswarm.cli import main as cli_main

from tests.acceptance.conftest import acceptance_workspace, require_real_model_config


def test_run_ask_partial(monkeypatch, capsys):
    workspace = acceptance_workspace("partial")
    require_real_model_config(monkeypatch, workspace)

    monkeypatch.setenv("INSIGHTSWARM_SCRIPTED_FIXTURE", "partial_missing_evidence")

    rc = cli_main(
        [
            "--db-path",
            str(workspace / "insightswarm.db"),
            "--artifact-dir",
            str(workspace / "artifacts"),
            "run",
            "ask",
            "DeepSeek 下步战略",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    critic_steps = [step for step in payload["steps"] if step["tool_call"]["action"] == "critic"]
    repair_search_steps = [
        step
        for step in payload["steps"]
        if step["tool_call"]["action"] == "search" and step["tool_call"]["arguments"].get("repair_round")
    ]

    assert rc == 0
    assert payload["result_type"] == "report_partial"
    assert payload["critic"]["verdict"] == "repair"
    assert payload["must_fix"]
    assert len(critic_steps) >= 2
    assert repair_search_steps
