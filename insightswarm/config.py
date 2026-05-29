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
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_text_model: str = "qwen3.6-flash"
    qwen_omni_model: str = "qwen3.5-omni-plus-2026-03-15"


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
    qwen_base_url = _first(
        dotenv.get("INSIGHTSWARM_QWEN_BASE_URL"),
        os.getenv("INSIGHTSWARM_QWEN_BASE_URL"),
        yaml_config.get("qwen_base_url"),
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    qwen_text_model = _first(
        dotenv.get("INSIGHTSWARM_QWEN_TEXT_MODEL"),
        os.getenv("INSIGHTSWARM_QWEN_TEXT_MODEL"),
        yaml_config.get("qwen_text_model"),
        "qwen3.6-flash",
    )
    qwen_omni_model = _first(
        dotenv.get("INSIGHTSWARM_QWEN_OMNI_MODEL"),
        os.getenv("INSIGHTSWARM_QWEN_OMNI_MODEL"),
        yaml_config.get("qwen_omni_model"),
        "qwen3.5-omni-plus-2026-03-15",
    )
    return Settings(
        resolved_db,
        resolved_artifacts,
        provider or "fake",
        yaml_path,
        qwen_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        qwen_text_model or "qwen3.6-flash",
        qwen_omni_model or "qwen3.5-omni-plus-2026-03-15",
    )
