from __future__ import annotations

from insightswarm.objective_runtime import _fixture_payload


def test_fixture_payload_defaults_to_empty_without_env(monkeypatch) -> None:
    monkeypatch.delenv("INSIGHTSWARM_SCRIPTED_FIXTURE", raising=False)

    assert _fixture_payload() == {}
