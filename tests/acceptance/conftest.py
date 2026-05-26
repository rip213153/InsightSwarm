from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pytest

from insightswarm.config import load_settings


REPO_ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE_TMP_ROOT = REPO_ROOT / ".tmp" / "acceptance"
QWEN_TEXT_MODEL = "qwen3.6-flash-2026-04-16"


def require_real_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    if not os.getenv("DASHSCOPE_API_KEY"):
        pytest.fail("DASHSCOPE_API_KEY is required for acceptance tests.")
    monkeypatch.setenv("INSIGHTSWARM_MODEL_PROVIDER", "qwen")
    monkeypatch.setenv("INSIGHTSWARM_QWEN_TEXT_MODEL", QWEN_TEXT_MODEL)
    settings = load_settings()
    assert settings.qwen_text_model == QWEN_TEXT_MODEL


def acceptance_workspace(name: str) -> Path:
    ACCEPTANCE_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    workspace = ACCEPTANCE_TMP_ROOT / f"{name}-{uuid.uuid4().hex[:8]}"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace
