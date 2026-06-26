from __future__ import annotations

import main
from insightswarm.models.router import build_model_client


def test_main_config_command_prints_runtime_config(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "_discover_model_config", lambda repo_root: None)

    code = main.main(["/config"])

    output = capsys.readouterr().out
    assert code == 0
    assert "provider: qwen" in output
    assert "browser_profile_root:" in output


def test_startup_banner_shows_runtime_identity(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("INSIGHTSWARM_QWEN_TEXT_MODEL", "qwen-test")
    config = main.RuntimeConfig(session_id="session123")

    banner = main._startup_banner(config, tmp_path)

    assert "InsightSwarm" in banner
    assert "local multi-agent research runtime" in banner
    assert "workspace" in banner
    assert "provider   qwen" in banner
    assert "model      qwen-test" in banner
    assert "session    session123" in banner
    assert "/ask <question>" in banner


def test_main_model_command_sets_active_model(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "_discover_model_config", lambda repo_root: None)

    code = main.main(["/model", "qwen3.7-plus"])

    output = capsys.readouterr().out
    assert code == 0
    assert "model set to qwen3.7-plus" in output


def test_main_ask_requires_model_before_running(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "_discover_model_config", lambda repo_root: None)
    monkeypatch.delenv("INSIGHTSWARM_QWEN_TEXT_MODEL", raising=False)
    monkeypatch.delenv("INSIGHTSWARM_TEXT_MODEL", raising=False)

    code = main.main(["/ask", "hello"])

    output = capsys.readouterr().out
    assert code == 2
    assert "No text model is configured" in output


def test_qwen_provider_can_be_built_from_environment_without_config(monkeypatch) -> None:
    monkeypatch.setenv("INSIGHTSWARM_QWEN_TEXT_MODEL", "qwen-test")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    client = build_model_client("qwen")

    assert client.provider == "qwen"
    assert client.model == "qwen-test"
    assert client.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
