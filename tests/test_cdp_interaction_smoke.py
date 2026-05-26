from __future__ import annotations

import os

import pytest

from insightswarm.browser_backend import CdpBrowserBackend
from insightswarm.browser_interaction import approve_browser_action
from insightswarm.browser_sandbox import write_browser_observation
from insightswarm.db.migrations import init_db
from insightswarm.db.store import Store
from insightswarm.tools import ToolContext
from insightswarm.tools.executor import ToolExecutor


pytestmark = pytest.mark.cdp_interaction_smoke


def _require_cdp_interaction_smoke() -> str:
    if os.getenv("INSIGHTSWARM_CDP_INTERACTION_SMOKE") != "1":
        pytest.skip("set INSIGHTSWARM_CDP_INTERACTION_SMOKE=1 to run CDP interaction smoke tests")
    cdp_url = os.getenv("INSIGHTSWARM_CDP_URL")
    if not cdp_url:
        pytest.skip("set INSIGHTSWARM_CDP_URL=ws://... to run CDP interaction smoke tests")
    if not CdpBrowserBackend.available():
        pytest.skip("install optional browser extra with websocket-client to run CDP interaction smoke tests")
    return cdp_url


def test_cdp_real_interaction_approval_flow(tmp_path):
    cdp_url = _require_cdp_interaction_smoke()
    db_path = tmp_path / "cdp-interaction.db"
    artifact_dir = tmp_path / "artifacts"
    init_db(db_path)
    store = Store(db_path, artifact_dir)
    run_id = store.create_run("cdp-interaction", {"quality_mode": "test"})
    task_id = store.create_task(run_id, "Discovery", "BrowserAgent")
    context = ToolContext(run_id, task_id, "test", {"agent_name": "BrowserAgent"})
    page_state_result, _ = ToolExecutor(store).run(
        "browser.page_state",
        {"backend": "cdp", "cdp_url": cdp_url},
        context,
    )
    if page_state_result.status != "ok":
        pytest.skip(f"CDP page state unavailable: {page_state_result.error}")
    page_state_id = write_browser_observation(store, run_id, task_id, "browser.page_state", page_state_result)
    elements = page_state_result.data["observation"].get("interactable_elements") or []
    if not elements:
        pytest.skip("CDP target page has no interactable element for click smoke")
    target_id = elements[0]["stable_node_id"]
    ToolExecutor(store).run(
        "browser.click",
        {"target": "first interactable", "target_id": target_id, "page_state_artifact_id": page_state_id},
        context,
    )
    request = next(row for row in store.list_artifacts(run_id) if row["artifact_type"] == "browser_action_request")
    result = approve_browser_action(store, run_id, request["artifact_id"], execute=True, backend="cdp", cdp_url=cdp_url, quality_mode="test")
    assert result["status"] == "approved"
    assert result["execution"]["status"] in {"ok", "error"}
    assert any(row["artifact_type"] == "browser_action_execution" for row in store.list_artifacts(run_id))
