from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    db_path: Path
    artifact_dir: Path
    model_provider: str = "fake"
    config_path: Path | None = None
    model_config_path: Path | None = None


def _read_dotenv(path: Path = Path(".env")) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _read_config_yaml(path: Path = Path("config.yaml")) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


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
    dotenv = _read_dotenv()
    yaml_path = Path(config_path) if config_path else Path("config.yaml")
    yaml_config = _read_config_yaml(yaml_path)
    resolved_db = Path(
        _first(
            db_path,
            dotenv.get("INSIGHTSWARM_DB_PATH"),
            os.getenv("INSIGHTSWARM_DB_PATH"),
            yaml_config.get("db_path"),
            ".insightswarm/insightswarm.db",
        )
        or ".insightswarm/insightswarm.db"
    )
    resolved_artifacts = Path(
        _first(
            artifact_dir,
            dotenv.get("INSIGHTSWARM_ARTIFACT_DIR"),
            os.getenv("INSIGHTSWARM_ARTIFACT_DIR"),
            yaml_config.get("artifact_dir"),
            ".insightswarm/artifacts",
        )
        or ".insightswarm/artifacts"
    )
    provider = _first(
        model_provider,
        dotenv.get("INSIGHTSWARM_MODEL_PROVIDER"),
        os.getenv("INSIGHTSWARM_MODEL_PROVIDER"),
        yaml_config.get("model_provider"),
        "fake",
    )
    resolved_model_config = _first(
        model_config_path,
        dotenv.get("INSIGHTSWARM_MODEL_CONFIG"),
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
