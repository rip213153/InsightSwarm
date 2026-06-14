from __future__ import annotations

from insightswarm.db.store import Store
from insightswarm.models.audit import AuditedModelClient
from insightswarm.models.fake import FakeModelClient
from insightswarm.models.registry import ModelRegistry


def build_model_client(
    provider: str,
    *,
    model_config_path: str | None = None,
    capability: str = "text",
    model: str | None = None,
):
    """Build a model client.

    Without a model config file, only the deterministic fake client is available.
    Real providers are intentionally config-driven: their base URL, API key env,
    and model names live in ``config.models.json`` rather than in code.
    """
    if provider == "fake" and not model_config_path:
        return FakeModelClient()
    return ModelRegistry.from_file(model_config_path).for_provider(
        provider,
        capability=capability,
        model=model,
    )


def build_audited_model_client(
    provider: str,
    store: Store,
    *,
    model_config_path: str | None = None,
    capability: str = "text",
    model: str | None = None,
):
    return AuditedModelClient(
        build_model_client(
            provider,
            model_config_path=model_config_path,
            capability=capability,
            model=model,
        ),
        store,
    )
