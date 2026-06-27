from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    db_path: Path
    artifact_dir: Path
    model_provider: str = "fake"
    config_path: Path | None = None
    model_config_path: Path | None = None


# Hand-written dotenv/yaml parsers were replaced by python-dotenv and pyyaml.
# See git history for the original implementations.

def _read_config_yaml(path: Path = Path("config.yaml")) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        return {}
    return {str(key): str(value) for key, value in loaded.items() if value is not None}


def _first(*values: str | None) -> str | None:
    for value in values:
        if value:
            return value
    return None


def load_settings(
    db_path: str | None = None,
    artifact_dir: str | None = None,
    model_provider: str | None = None,
    config_path: str | None = None,
    model_config_path: str | None = None,
) -> Settings:
    load_dotenv()
    yaml_path = Path(config_path) if config_path else Path("config.yaml")
    yaml_config = _read_config_yaml(yaml_path)
    resolved_db = Path(
        _first(
            db_path,
            os.getenv("INSIGHTSWARM_DB_PATH"),
            yaml_config.get("db_path"),
            ".insightswarm/insightswarm.db",
        )
        or ".insightswarm/insightswarm.db"
    )
    resolved_artifacts = Path(
        _first(
            artifact_dir,
            os.getenv("INSIGHTSWARM_ARTIFACT_DIR"),
            yaml_config.get("artifact_dir"),
            ".insightswarm/artifacts",
        )
        or ".insightswarm/artifacts"
    )
    provider = _first(
        model_provider,
        os.getenv("INSIGHTSWARM_MODEL_PROVIDER"),
        yaml_config.get("model_provider"),
        "fake",
    )
    resolved_model_config = _first(
        model_config_path,
        os.getenv("INSIGHTSWARM_MODEL_CONFIG"),
        yaml_config.get("model_config_path"),
        None,
    )
    return Settings(
        resolved_db,
        resolved_artifacts,
        provider or "fake",
        yaml_path,
        Path(resolved_model_config) if resolved_model_config else None,
    )
