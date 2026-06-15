from __future__ import annotations

import http.client

from insightswarm.models.openai_compatible import OpenAICompatibleClient


def test_incomplete_read_returns_model_error(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_API_KEY", "key")

    def _raise_incomplete_read(*args, **kwargs):
        raise http.client.IncompleteRead(b"", 4737)

    monkeypatch.setattr("urllib.request.urlopen", _raise_incomplete_read)
    client = OpenAICompatibleClient(
        provider="test",
        model="model",
        base_url="https://example.com/v1",
        api_key_env="TEST_MODEL_API_KEY",
    )

    result = client.complete([{"role": "user", "content": "hello"}])

    assert result.status == "error"
    assert "response incomplete" in (result.error or "")
