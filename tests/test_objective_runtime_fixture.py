from __future__ import annotations

from insightswarm.models.clients import ModelResult
from insightswarm.objective_runtime import BrowserCompositeModelClient
from insightswarm.objective_runtime import _fixture_payload


def test_fixture_payload_defaults_to_empty_without_env(monkeypatch) -> None:
    monkeypatch.delenv("INSIGHTSWARM_SCRIPTED_FIXTURE", raising=False)

    assert _fixture_payload() == {}


class _TextClient:
    provider = "text_provider"
    model = "text_model"

    def __init__(self):
        self.complete_called = False

    def complete(self, *args, **kwargs):
        self.complete_called = True
        return ModelResult("", None, self.provider, self.model, {}, 0, {}, "ok")


class _VisionClient:
    provider = "vision_provider"
    model = "vision_model"

    def __init__(self):
        self.analyze_called = False

    def analyze_image(self, *args, **kwargs):
        self.analyze_called = True
        return ModelResult("", None, self.provider, self.model, {}, 0, {}, "ok")


def test_browser_composite_routes_text_and_vision_separately():
    text = _TextClient()
    vision = _VisionClient()
    client = BrowserCompositeModelClient(text_client=text, vision_client=vision)

    client.complete([])
    client.analyze_image([], [])

    assert text.complete_called is True
    assert vision.analyze_called is True
    assert client.provider == "text_provider"
    assert client.model == "text_model"
