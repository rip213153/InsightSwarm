from __future__ import annotations

import os

import pytest

from insightswarm.browser_backend import BrowserBackendUnavailable, CdpBrowserBackend
from insightswarm.tools import ToolContext, get_tool


pytestmark = pytest.mark.cdp_smoke


def _require_cdp_smoke() -> str:
    if os.getenv("INSIGHTSWARM_CDP_SMOKE") != "1":
        pytest.skip("set INSIGHTSWARM_CDP_SMOKE=1 to run CDP smoke tests")
    cdp_url = os.getenv("INSIGHTSWARM_CDP_URL")
    if not cdp_url:
        pytest.skip("set INSIGHTSWARM_CDP_URL=ws://... to run CDP smoke tests")
    if not CdpBrowserBackend.available():
        pytest.skip("install optional browser extra with websocket-client to run CDP smoke tests")
    return cdp_url


def test_cdp_read_only_snapshot_visible_text_and_screenshot():
    cdp_url = _require_cdp_smoke()
    context = ToolContext(quality_mode="test", metadata={"agent_name": "BrowserAgent"})
    try:
        snapshot = get_tool("browser.snapshot").run({"backend": "cdp", "cdp_url": cdp_url}, context)
        page_state = get_tool("browser.page_state").run({"backend": "cdp", "cdp_url": cdp_url}, context)
        visible_text = get_tool("browser.visible_text").run({"backend": "cdp", "cdp_url": cdp_url}, context)
        screenshot = get_tool("browser.screenshot").run({"backend": "cdp", "cdp_url": cdp_url}, context)
    except BrowserBackendUnavailable as exc:
        pytest.skip(f"CDP endpoint unavailable: {exc}")
    assert snapshot.status == "ok"
    assert snapshot.data["browser_backend"] == "cdp"
    assert snapshot.data["read_only"] is True
    assert "observation" in snapshot.data
    assert page_state.status == "ok"
    assert page_state.data["browser_backend"] == "cdp"
    assert page_state.data["read_only"] is True
    assert "interactable_elements" in page_state.data["observation"]
    assert visible_text.status == "ok"
    assert isinstance(visible_text.data["observation"].get("text"), str)
    assert screenshot.status == "ok"
    assert screenshot.data["observation"]["screenshot_captured"] is True
