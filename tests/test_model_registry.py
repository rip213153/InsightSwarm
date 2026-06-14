from __future__ import annotations

import json

import pytest

from insightswarm.config import load_settings
from insightswarm.models.fake import FakeModelClient
from insightswarm.models.openai_compatible import OpenAICompatibleClient
from insightswarm.models.registry import ModelRegistry, load_model_config
from insightswarm.models.router import build_model_client


def _write_config(path):
    path.write_text(
        json.dumps(
            {
                "providers": {
                    "strong": {
                        "type": "openai_compatible",
                        "base_url": "https://strong.example.com/v1",
                        "api_key_env": "STRONG_MODEL_API_KEY",
                        "models": {"text": "strong-model"},
                    },
                    "fast": {
                        "type": "openai_compatible",
                        "base_url": "https://fast.example.com/v1",
                        "api_key_env": "FAST_MODEL_API_KEY",
                        "models": {"text": "fast-model", "vision": "vision-model"},
                    },
                },
                "agents": {
                    "default": {"provider": "strong"},
                    "researcher": {"provider": "fast", "model": "fast-model"},
                    "vision": {"provider": "fast", "capability": "vision"},
                },
            }
        ),
        encoding="utf-8",
    )


def test_model_config_loads_providers_and_agents(tmp_path):
    path = tmp_path / "models.json"
    _write_config(path)

    config = load_model_config(path)

    assert config.providers["strong"].base_url == "https://strong.example.com/v1"
    assert config.agents["researcher"].provider == "fast"
    assert config.agents["default"].provider == "strong"


def test_registry_builds_agent_clients_and_inherits_default(tmp_path):
    path = tmp_path / "models.json"
    _write_config(path)
    registry = ModelRegistry.from_file(path)

    researcher = registry.for_agent("researcher")
    critic = registry.for_agent("critic")
    vision = registry.for_agent("vision", capability="vision")

    assert isinstance(researcher, OpenAICompatibleClient)
    assert researcher.provider == "fast"
    assert researcher.model == "fast-model"
    assert critic.provider == "strong"
    assert critic.model == "strong-model"
    assert vision.model == "vision-model"


def test_registry_default_without_file_is_fake(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert isinstance(ModelRegistry.from_file(None).for_agent("writer"), FakeModelClient)


def test_invalid_agent_provider_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        '{"providers": {}, "agents": {"writer": {"provider": "missing"}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown provider"):
        load_model_config(path)


def test_router_uses_model_config_path(tmp_path):
    path = tmp_path / "models.json"
    _write_config(path)

    client = build_model_client("strong", model_config_path=str(path))

    assert isinstance(client, OpenAICompatibleClient)
    assert client.provider == "strong"
    assert client.model == "strong-model"


def test_settings_reads_model_config_from_env(monkeypatch, tmp_path):
    path = tmp_path / "models.json"
    monkeypatch.setenv("INSIGHTSWARM_MODEL_CONFIG", str(path))

    settings = load_settings()

    assert settings.model_config_path == path
