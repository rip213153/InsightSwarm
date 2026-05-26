from __future__ import annotations

import ast
import json
import math
import re
import statistics
import itertools
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from insightswarm.db.store import Store
from insightswarm.tools.core import ToolContext, ToolResult


BLOCKED_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
}
BLOCKED_MODULES = {"os", "sys", "subprocess", "socket", "requests", "urllib", "httpx", "pandas", "numpy", "pypdf"}
MAX_CODE_CHARS = 8000
MAX_ITEMS = 30
_NAMESPACES: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class CodeSandboxReport:
    status: str
    extracted_items: list[dict[str, Any]]
    candidate_targets: list[dict[str, Any]]
    warnings: list[str]
    namespace_state_summary: dict[str, Any]
    provenance: dict[str, Any]
    error: str | None = None


def run_browser_extract_code(tool_input: dict[str, Any], context: ToolContext | None = None) -> CodeSandboxReport:
    started = time.perf_counter()
    code = str(tool_input.get("code") or "")
    if len(code) > MAX_CODE_CHARS:
        return _error("code is too long", started, tool_input)
    blocked = _validate_code(code)
    if blocked:
        return _error(blocked, started, tool_input)
    session_id = str(tool_input.get("session_id") or "browser-code-default")
    namespace = _NAMESPACES.setdefault(session_id, {})
    page_state = _sanitize_page_state(tool_input.get("page_state") or {})
    html_text = _safe_string(tool_input.get("html_text") or tool_input.get("text") or "", 4000)
    namespace.update(
        {
            "page_state": page_state,
            "html_text": html_text,
            "classify_page_state": classify_page_state,
            "semantic_candidates": classify_page_state(page_state),
        }
    )
    globals_dict = _safe_globals()
    try:
        exec(compile(code, "<browser_extract_code>", "exec"), globals_dict, namespace)
    except Exception as exc:
        return _error(f"{exc.__class__.__name__}: {exc}", started, tool_input, namespace)
    extracted_items = _bounded_records(namespace.get("extracted_items") or [])
    candidate_targets = _bounded_records(namespace.get("candidate_targets") or namespace.get("semantic_candidates") or [])
    warnings = [str(item)[:200] for item in (namespace.get("warnings") or [])[:10]]
    return CodeSandboxReport(
        status="ok",
        extracted_items=extracted_items,
        candidate_targets=candidate_targets,
        warnings=warnings,
        namespace_state_summary=_namespace_summary(namespace),
        provenance=_provenance(started, tool_input),
    )


def classify_page_state(page_state: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    elements = page_state.get("interactable_elements") if isinstance(page_state, dict) else []
    for element in elements or []:
        if not isinstance(element, dict):
            continue
        text = _safe_string(element.get("name") or element.get("text") or element.get("href") or "", 200)
        href = _safe_string(element.get("href") or "", 300)
        role = _safe_string(element.get("role") or "", 60).lower()
        tag = _safe_string(element.get("tag") or "", 40).lower()
        haystack = f"{text} {href} {role} {tag}".lower()
        semantic_type = "unknown"
        confidence = 0.35
        if any(token in haystack for token in ["客服", "customer service", "chat", "咨询"]):
            semantic_type = "customer_service"
            confidence = 0.92
        elif any(token in haystack for token in ["广告", "ad", "sponsored", "推广"]):
            semantic_type = "ad"
            confidence = 0.85
        elif role in {"searchbox"} or tag == "input" or "搜索" in text:
            semantic_type = "search_box"
            confidence = 0.78
        elif any(token in haystack for token in ["￥", "¥", "$", "价格", "price"]):
            semantic_type = "price_text"
            confidence = 0.72
        elif any(token in haystack for token in ["筛选", "filter", "排序"]):
            semantic_type = "filter"
            confidence = 0.7
        elif any(token in haystack for token in ["店", "自营", "旗舰"]):
            semantic_type = "store_link"
            confidence = 0.65
        elif href and any(token in href for token in ["item", "product", "sku", "mall.jd.com"]) and semantic_type == "unknown":
            semantic_type = "product_detail_link"
            confidence = 0.82
        elif role == "link" and any(token in text for token in ["联想", "电脑", "笔记本", "thinkpad", "lenovo"]):
            semantic_type = "product_detail_link"
            confidence = 0.76
        if semantic_type == "customer_service":
            click_priority = 0.05
        elif semantic_type == "product_detail_link":
            click_priority = 0.9
        elif semantic_type == "search_box":
            click_priority = 0.45
        else:
            click_priority = 0.25
        candidates.append(
            {
                "stable_node_id": element.get("stable_node_id"),
                "semantic_type": semantic_type,
                "text": text,
                "href": href or None,
                "role": role,
                "tag": tag,
                "confidence": confidence,
                "click_priority": click_priority,
                "risk_hint": "avoid" if semantic_type in {"customer_service", "ad"} else "normal",
                "bbox": element.get("bbox"),
            }
        )
    return sorted(candidates, key=lambda item: (item["click_priority"], item["confidence"]), reverse=True)[:MAX_ITEMS]


def write_browser_code_result(
    store: Store,
    run_id: str,
    task_id: str | None,
    tool_call_id: str,
    result: ToolResult,
) -> str:
    data = result.data or {}
    payload = {
        "schema": "browser_code_result.v1",
        "status": result.status,
        "tool_call_id": tool_call_id,
        "data": data,
        "diagnostics": result.diagnostics,
        "warnings": result.warnings,
        "error": result.error,
        "provenance": result.provenance,
    }
    artifact_id = store.write_artifact(
        run_id,
        task_id,
        "browser_code_result",
        "application/json",
        json.dumps(payload, ensure_ascii=True, indent=2),
        metadata={
            "schema": "browser_code_result.v1",
            "tool": "browser.extract_code",
            "tool_call_id": tool_call_id,
            "tool_status": result.status,
            "candidate_target_count": len(data.get("candidate_targets") or []),
            "extracted_item_count": len(data.get("extracted_items") or []),
            "namespace_variable_count": (data.get("namespace_state_summary") or {}).get("variable_count", 0),
            "error": result.error,
        },
        suffix=".json",
    )
    event_type = "browser_code_executed" if result.status == "ok" else "browser_code_failed"
    store.emit_event(
        run_id,
        task_id,
        "BrowserAgent",
        event_type,
        f"Browser code sandbox {result.status}.",
        {
            "artifact_id": artifact_id,
            "tool_call_id": tool_call_id,
            "candidate_target_count": len(data.get("candidate_targets") or []),
            "extracted_item_count": len(data.get("extracted_items") or []),
            "error": result.error,
        },
    )
    return artifact_id


def _validate_code(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "import statements are not allowed in browser code sandbox"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
                return f"blocked function: {node.func.id}"
            if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("__"):
                return "dunder attribute calls are not allowed"
        if isinstance(node, ast.Name) and node.id in BLOCKED_MODULES:
            return f"blocked module or name: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder attributes are not allowed"
    return None


def _safe_globals() -> dict[str, Any]:
    return {
        "__builtins__": {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
        },
        "re": re,
        "json": json,
        "math": math,
        "statistics": statistics,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "dataclass": dataclass,
        "itertools": itertools,
    }


def _sanitize_page_state(page_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": _safe_string(page_state.get("url"), 300),
        "title": _safe_string(page_state.get("title"), 200),
        "text_preview": _safe_string(page_state.get("text_preview"), 2000),
        "visible_text_chars": page_state.get("visible_text_chars"),
        "interactable_elements": [
            {
                "stable_node_id": _safe_string(item.get("stable_node_id"), 120),
                "role": _safe_string(item.get("role"), 60),
                "name": _safe_string(item.get("name") or item.get("text"), 200),
                "text": _safe_string(item.get("text") or item.get("name"), 200),
                "tag": _safe_string(item.get("tag"), 40),
                "href": _safe_string(item.get("href"), 300),
                "action_hint": _safe_string(item.get("action_hint"), 80),
                "bbox": item.get("bbox") if isinstance(item.get("bbox"), dict) else None,
                "visibility": _safe_string(item.get("visibility"), 40),
            }
            for item in (page_state.get("interactable_elements") or [])[:200]
            if isinstance(item, dict)
        ],
    }


def _bounded_records(value: Any) -> list[dict[str, Any]]:
    records = value if isinstance(value, list) else []
    bounded = []
    for item in records[:MAX_ITEMS]:
        if isinstance(item, dict):
            bounded.append(_safe_record(item))
    return bounded


def _safe_record(record: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in record.items():
        lowered = str(key).lower()
        if lowered in {"cookie", "token", "authorization", "header", "headers", "password", "localstorage"}:
            continue
        safe[str(key)[:80]] = _safe_value(value)
    return safe


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return _safe_string(value, 300)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return _safe_record(value)
    return _safe_string(value, 120)


def _namespace_summary(namespace: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in namespace.items() if not key.startswith("_")}
    return {
        "variable_count": len(public),
        "variables": [
            {"name": key, "type": type(value).__name__, "preview": _safe_string(value, 120)}
            for key, value in list(public.items())[:20]
        ],
    }


def _error(reason: str, started: float, tool_input: dict[str, Any], namespace: dict[str, Any] | None = None) -> CodeSandboxReport:
    return CodeSandboxReport(
        status="error",
        extracted_items=[],
        candidate_targets=[],
        warnings=[],
        namespace_state_summary=_namespace_summary(namespace or {}),
        provenance=_provenance(started, tool_input),
        error=reason,
    )


def _provenance(started: float, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "sandbox": "browser_code.v0",
        "session_id": tool_input.get("session_id") or "browser-code-default",
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "input_kind": "browser_page_state",
    }


def _safe_string(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\x00", "")
    return text if len(text) <= limit else text[:limit] + "...[truncated]"
