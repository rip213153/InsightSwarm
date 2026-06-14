from __future__ import annotations

import os
import json
import shutil
import uuid
from pathlib import Path

import pytest

from insightswarm.config import load_settings


REPO_ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE_TMP_ROOT = REPO_ROOT / ".tmp" / "acceptance"


def require_real_model_config(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    base_url = os.getenv("INSIGHTSWARM_ACCEPTANCE_BASE_URL")
    api_key_env = os.getenv("INSIGHTSWARM_ACCEPTANCE_API_KEY_ENV", "MODEL_API_KEY")
    model = os.getenv("INSIGHTSWARM_ACCEPTANCE_MODEL")
    if not base_url or not model or not os.getenv(api_key_env):
        pytest.fail(
            "Acceptance tests require INSIGHTSWARM_ACCEPTANCE_BASE_URL, "
            "INSIGHTSWARM_ACCEPTANCE_MODEL, and the configured API key env."
        )
    config_path = workspace / "config.models.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {
                    "default": {
                        "type": "openai_compatible",
                        "base_url": base_url,
                        "api_key_env": api_key_env,
                        "models": {"text": model},
                    }
                },
                "agents": {"default": {"provider": "default"}},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("INSIGHTSWARM_MODEL_CONFIG", str(config_path))
    monkeypatch.setenv("INSIGHTSWARM_MODEL_PROVIDER", "default")
    settings = load_settings()
    assert settings.model_config_path == config_path


def acceptance_workspace(name: str) -> Path:
    ACCEPTANCE_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    workspace = ACCEPTANCE_TMP_ROOT / f"{name}-{uuid.uuid4().hex[:8]}"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace
