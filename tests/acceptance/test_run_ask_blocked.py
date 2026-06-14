from __future__ import annotations

import json

from insightswarm.cli import main as cli_main

from tests.acceptance.conftest import acceptance_workspace, require_real_model_config


def test_run_ask_blocked(monkeypatch, capsys):
    workspace = acceptance_workspace("blocked")
    require_real_model_config(monkeypatch, workspace)

    monkeypatch.setenv("INSIGHTSWARM_SCRIPTED_FIXTURE", "blocked_browser_risk")

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

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert rc == 2
    assert "HumanAuthorizationRequired" in captured.err
    assert payload["result_type"] == "report_blocked"
    assert payload["stop_reason"] == "human_required"
