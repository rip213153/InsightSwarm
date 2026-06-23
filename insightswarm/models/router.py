from __future__ import annotations

import os

from insightswarm.db.store import Store
from insightswarm.models.audit import AuditedModelClient
from insightswarm.models.fake import FakeModelClient
from insightswarm.models.openai_compatible import OpenAICompatibleClient
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
    if not model_config_path:
        env_client = _build_env_model_client(provider, capability=capability, model=model)
        if env_client is not None:
            return env_client
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


def _build_env_model_client(provider: str, *, capability: str, model: str | None):
    normalized = str(provider or "").strip().lower()
    if normalized not in {"qwen", "dashscope", "openai_compatible"}:
        return None

    prefix = "INSIGHTSWARM_QWEN" if normalized in {"qwen", "dashscope"} else "INSIGHTSWARM_OPENAI_COMPATIBLE"
    default_base_url = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if normalized in {"qwen", "dashscope"}
        else os.getenv("OPENAI_COMPATIBLE_BASE_URL", "")
    )
    default_key_env = "DASHSCOPE_API_KEY" if normalized in {"qwen", "dashscope"} else "OPENAI_COMPATIBLE_API_KEY"
    selected_model = (
        model
        or os.getenv(f"{prefix}_{capability.upper()}_MODEL")
        or os.getenv(f"{prefix}_TEXT_MODEL")
        or os.getenv("INSIGHTSWARM_TEXT_MODEL")
        or os.getenv("OPENAI_COMPATIBLE_MODEL")
    )
    base_url = os.getenv(f"{prefix}_BASE_URL") or default_base_url
    api_key_env = os.getenv(f"{prefix}_API_KEY_ENV") or default_key_env
    timeout_seconds = float(os.getenv(f"{prefix}_TIMEOUT_SECONDS") or os.getenv("INSIGHTSWARM_MODEL_TIMEOUT_SECONDS") or 180)
    if not selected_model or not base_url:
        return None
    return OpenAICompatibleClient(
        provider=provider,
        model=selected_model,
        base_url=base_url,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
    )
