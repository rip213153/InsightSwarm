from __future__ import annotations

import json

from insightswarm.cli import main as cli_main

from tests.acceptance.conftest import acceptance_workspace, require_real_qwen


def test_run_ask_delivers(monkeypatch, capsys):
    require_real_qwen(monkeypatch)
    workspace = acceptance_workspace("deliver")

    monkeypatch.setenv("INSIGHTSWARM_SCRIPTED_FIXTURE", "deliver_minimal")

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
    run_dir = next((workspace / ".tmp").glob("run-*"), None)

    assert rc == 0
    assert payload["result_type"] == "report"
    assert payload["critic"]["verdict"] == "pass"
    assert payload["report"]["body"]
    assert run_dir is not None
