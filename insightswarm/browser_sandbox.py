from __future__ import annotations

import json
from typing import Any

from insightswarm.db.store import Store
from insightswarm.tools.core import ToolResult


def write_browser_observation(
    store: Store,
    run_id: str,
    task_id: str | None,
    tool_name: str,
    result: ToolResult,
    *,
    source_url: str | None = None,
    session_id: str | None = None,
) -> str:
    diagnostics = result.diagnostics or {}
    observation = ((result.data or {}).get("observation") or {}) if isinstance(result.data, dict) else {}
    is_page_state = tool_name == "browser.page_state"
    payload = {
        "schema": "browser_page_state.v1" if is_page_state else "browser_observation.v1",
        "tool_name": tool_name,
        "status": result.status,
        "data": result.data,
        "diagnostics": diagnostics,
        "warnings": result.warnings,
        "provenance": result.provenance,
    }
    return store.write_artifact(
        run_id,
        task_id,
        "browser_page_state" if is_page_state else "browser_observation",
        "application/json",
        json.dumps(payload, ensure_ascii=True, indent=2),
        source_url=source_url or observation.get("url"),
        metadata={
            "tool": tool_name,
            "browser_session_id": session_id or diagnostics.get("browser_session_id"),
            "risk_status": diagnostics.get("risk_status"),
            "browser_backend": diagnostics.get("browser_backend"),
            "read_only": bool(diagnostics.get("read_only")),
            "cdp_url_present": bool(diagnostics.get("cdp_url_present")),
            "fake_execution": bool(diagnostics.get("fake_execution")),
            "node_count": observation.get("node_count"),
            "interactable_count": observation.get("interactable_count"),
            "truncated": bool(observation.get("truncated")),
            "partial": bool(diagnostics.get("partial")),
            "cdp_methods_used": diagnostics.get("cdp_methods_used", []),
        },
        suffix=".json",
    )
