from __future__ import annotations

from insightswarm.db.store import Store
from insightswarm.models.audit import AuditedModelClient
from insightswarm.models.fake import FakeModelClient
from insightswarm.models.qwen import QwenOpenAICompatibleClient

SUPPORTED_PROVIDERS = {
    "fake",
    "qwen",
    "deepseek",
    "openai_compatible",
    "qwen_text",
    "aliyun_text",
    "aliyun_vision",
}


class UnimplementedModelClient:
    def __init__(self, provider: str):
        self.provider = provider
        self.model = f"{provider}-not-implemented"

    def complete(self, *args, **kwargs):
        raise NotImplementedError(
            f"Model provider '{self.provider}' is registered but not implemented in Milestone 2."
        )

    def analyze_image(self, *args, **kwargs):
        raise NotImplementedError(
            f"Vision provider '{self.provider}' is registered but not implemented in Milestone 2."
        )


class QwenSwarmClient:
    def __init__(self):
        self.text = QwenOpenAICompatibleClient(
            "qwen_text",
            __import__("os").getenv("INSIGHTSWARM_QWEN_TEXT_MODEL", "qwen3.6-flash"),
        )
        self.vision = QwenOpenAICompatibleClient(
            "aliyun_vision",
            __import__("os").getenv(
                "INSIGHTSWARM_QWEN_OMNI_MODEL",
                "qwen3.5-omni-plus-2026-03-15",
            ),
        )

    def complete(self, *args, **kwargs):
        return self.text.complete(*args, **kwargs)

    def analyze_image(self, *args, **kwargs):
        return self.vision.analyze_image(*args, **kwargs)


def build_model_client(provider: str):
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported model provider '{provider}'. Supported providers: {', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )
    if provider == "qwen":
        return QwenSwarmClient()
    if provider == "qwen_text":
        return QwenSwarmClient()
    if provider == "aliyun_vision":
        return QwenOpenAICompatibleClient(
            "aliyun_vision",
            __import__("os").getenv(
                "INSIGHTSWARM_QWEN_OMNI_MODEL",
                "qwen3.5-omni-plus-2026-03-15",
            ),
        )
    if provider != "fake":
        return UnimplementedModelClient(provider)
    return FakeModelClient()


def build_audited_model_client(provider: str, store: Store):
    return AuditedModelClient(build_model_client(provider), store)
