from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from insightswarm.db.store import Store
from insightswarm.models.audit import AuditedModelClient
from insightswarm.models.fake import FakeModelClient
from insightswarm.models.openai_compatible import OpenAICompatibleClient


DEFAULT_MODEL_CONFIG_PATH = Path("config.models.json")


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    type: str = "openai_compatible"
    base_url: str | None = None
    api_key_env: str | None = None
    models: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    # Capability flags: declared by the operator in config.models.json.
    # ``supports_tool_choice_required`` — provider accepts tool_choice="required"
    # and will always emit at least one tool_call (no model_no_tool escape).
    # ``supports_json_schema_strict`` — provider accepts response_format with
    # json_schema strict mode, making AgentTurn validation a near-passthrough.
    supports_tool_choice_required: bool = False
    supports_json_schema_strict: bool = False


@dataclass(frozen=True)
class AgentModelConfig:
    provider: str
    model: str | None = None
    capability: str = "text"


@dataclass(frozen=True)
class ModelConfig:
    providers: dict[str, ProviderConfig]
    agents: dict[str, AgentModelConfig]


DEFAULT_CONFIG = ModelConfig(
    providers={
        "fake": ProviderConfig(name="fake", type="fake", models={"text": "fake-deterministic-v1"}),
    },
    agents={
        "default": AgentModelConfig(provider="fake", capability="text"),
    },
)


def resolve_model_config_path(path: str | Path | None = None) -> Path | None:
    raw = str(path or os.getenv("INSIGHTSWARM_MODEL_CONFIG") or "").strip()
    if raw:
        return Path(raw)
    if DEFAULT_MODEL_CONFIG_PATH.exists():
        return DEFAULT_MODEL_CONFIG_PATH
    return None


def load_model_config(path: str | Path | None = None) -> ModelConfig:
    resolved = resolve_model_config_path(path)
    if resolved is None:
        return DEFAULT_CONFIG
    if not resolved.exists():
        raise FileNotFoundError(f"model config file not found: {resolved}")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"model config must be a JSON object: {resolved}")
    return _coerce_model_config(data, source=str(resolved))


def _coerce_model_config(data: dict[str, Any], *, source: str) -> ModelConfig:
    providers_data = data.get("providers") or {}
    agents_data = data.get("agents") or {}
    if not isinstance(providers_data, dict):
        raise ValueError(f"{source}: providers must be an object")
    if not isinstance(agents_data, dict):
        raise ValueError(f"{source}: agents must be an object")

    providers: dict[str, ProviderConfig] = {}
    for name, raw in providers_data.items():
        if not isinstance(raw, dict):
            raise ValueError(f"{source}: provider '{name}' must be an object")
        models = raw.get("models") or {}
        if not isinstance(models, dict):
            raise ValueError(f"{source}: provider '{name}'.models must be an object")
        providers[str(name)] = ProviderConfig(
            name=str(name),
            type=str(raw.get("type") or "openai_compatible"),
            base_url=str(raw["base_url"]).rstrip("/") if raw.get("base_url") else None,
            api_key_env=str(raw["api_key_env"]) if raw.get("api_key_env") else None,
            models={str(k): str(v) for k, v in models.items()},
            timeout_seconds=float(raw.get("timeout_seconds") or 60.0),
            supports_tool_choice_required=bool(raw.get("supports_tool_choice_required") or False),
            supports_json_schema_strict=bool(raw.get("supports_json_schema_strict") or False),
        )

    agents: dict[str, AgentModelConfig] = {}
    for role, raw in agents_data.items():
        if not isinstance(raw, dict):
            raise ValueError(f"{source}: agent '{role}' must be an object")
        provider = str(raw.get("provider") or "").strip()
        if not provider:
            raise ValueError(f"{source}: agent '{role}' missing provider")
        agents[str(role)] = AgentModelConfig(
            provider=provider,
            model=str(raw["model"]) if raw.get("model") else None,
            capability=str(raw.get("capability") or "text"),
        )

    merged_providers = dict(DEFAULT_CONFIG.providers)
    merged_providers.update(providers)
    merged_agents = dict(DEFAULT_CONFIG.agents)
    merged_agents.update(agents)
    _validate_config(merged_providers, merged_agents, source=source)
    return ModelConfig(providers=merged_providers, agents=merged_agents)


def _validate_config(
    providers: dict[str, ProviderConfig],
    agents: dict[str, AgentModelConfig],
    *,
    source: str,
) -> None:
    for role, agent in agents.items():
        if agent.provider not in providers:
            raise ValueError(f"{source}: agent '{role}' references unknown provider '{agent.provider}'")
        provider = providers[agent.provider]
        if provider.type == "openai_compatible":
            model = agent.model or provider.models.get(agent.capability) or provider.models.get("text")
            if not model:
                raise ValueError(
                    f"{source}: agent '{role}' needs a model or provider '{agent.provider}' default model"
                )
            if not provider.base_url:
                raise ValueError(f"{source}: provider '{agent.provider}' missing base_url")
            if not provider.api_key_env:
                raise ValueError(f"{source}: provider '{agent.provider}' missing api_key_env")
        elif provider.type != "fake":
            raise ValueError(f"{source}: provider '{agent.provider}' has unsupported type '{provider.type}'")


class ModelRegistry:
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        store: Store | None = None,
        default_provider: str | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        self.store = store
        self.default_provider = default_provider

    @classmethod
    def from_file(
        cls,
        path: str | Path | None = None,
        *,
        store: Store | None = None,
        default_provider: str | None = None,
    ) -> "ModelRegistry":
        return cls(load_model_config(path), store=store, default_provider=default_provider)

    def for_agent(self, role: str, *, capability: str | None = None) -> Any:
        agent = self.config.agents.get(role) or self.config.agents.get("default")
        if agent is None:
            provider = self.default_provider or "fake"
            agent = AgentModelConfig(provider=provider, capability=capability or "text")
        if self.default_provider and agent.provider == "fake" and role == "default":
            agent = AgentModelConfig(provider=self.default_provider, capability=capability or agent.capability)
        if capability and capability != agent.capability:
            agent = AgentModelConfig(provider=agent.provider, model=agent.model, capability=capability)
        client = self._build_client(agent)
        return AuditedModelClient(client, self.store) if self.store is not None else client

    def for_provider(self, provider_name: str, *, capability: str = "text", model: str | None = None) -> Any:
        client = self._build_client(AgentModelConfig(provider=provider_name, model=model, capability=capability))
        return AuditedModelClient(client, self.store) if self.store is not None else client

    def _build_client(self, agent: AgentModelConfig) -> Any:
        provider = self.config.providers.get(agent.provider)
        if provider is None:
            raise ValueError(f"unknown model provider '{agent.provider}'")
        if provider.type == "fake":
            return FakeModelClient()
        model = agent.model or provider.models.get(agent.capability) or provider.models.get("text")
        if not model:
            raise ValueError(f"provider '{provider.name}' has no model for capability '{agent.capability}'")
        return OpenAICompatibleClient(
            provider=provider.name,
            model=model,
            base_url=provider.base_url,
            api_key_env=provider.api_key_env or "OPENAI_API_KEY",
            timeout_seconds=provider.timeout_seconds,
        )
