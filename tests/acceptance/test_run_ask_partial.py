from __future__ import annotations

import json

from insightswarm.cli import main as cli_main

from tests.acceptance.conftest import acceptance_workspace, require_real_qwen


def test_run_ask_partial(monkeypatch, capsys):
    require_real_qwen(monkeypatch)
    workspace = acceptance_workspace("partial")

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

    assert rc == 0
    assert payload["result_type"] == "report_partial"
    assert payload["critic"]["verdict"] == "repair"
    assert payload["must_fix"]
