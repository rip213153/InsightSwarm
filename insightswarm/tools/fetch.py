from __future__ import annotations

import html as html_lib
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from insightswarm.tools.core import ToolContext, ToolResult
from insightswarm.tools.safety import validate_public_http_url


class FetchUrlTool:
    name = "fetch.url"

    def run(self, tool_input: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        url = str(tool_input.get("url") or "")
        blocked = validate_public_http_url(url, context)
        if blocked:
            return blocked

        payload = _fixture_payload()
        if tool_input.get("repair_round"):
            fixture_documents = list(payload.get("repair_documents") or [])
        else:
            fixture_documents = list(payload.get("documents") or [])
        for document in fixture_documents:
            if document.get("url") == url:
                raw_html = str(document.get("html") or "")
                text = str(document.get("text") or "")
                if raw_html and not text:
                    text = _clean_html(raw_html)["text"]
                return ToolResult(
                    "ok",
                    data={
                        "source_url": url,
                        "fetcher": "fixture",
                        "status": "ok",
                        "text": text,
                        "html": raw_html or f"<html><body>{text}</body></html>",
                        "title": document.get("title") or _clean_html(raw_html or text)["title"],
                        "metadata": {"fixture": True},
                    },
                    provenance={"tool": self.name, "fetcher": "fixture"},
                )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=float(tool_input.get("timeout") or 20.0)) as response:
                raw = response.read()
                html = raw.decode("utf-8", errors="replace")
                cleaned = _clean_html(html)
        except Exception as exc:
            return ToolResult(
                "error",
                error=f"Fetch failed: {exc}",
                provenance={"tool": self.name, "fetcher": "urllib"},
            )

        return ToolResult(
            "ok",
            data={
                "source_url": url,
                "fetcher": "urllib",
                "status": "ok",
                "text": cleaned["text"],
                "html": html,
                "title": cleaned["title"],
                "metadata": {"latency_ms": int((time.perf_counter() - started) * 1000)},
            },
            provenance={"tool": self.name, "fetcher": "urllib"},
        )


def _fixture_payload() -> dict[str, Any]:
    fixture_name = os.getenv("INSIGHTSWARM_SCRIPTED_FIXTURE")
    if not fixture_name:
        return {}
    fixture_path = Path(__file__).resolve().parent / "fixtures" / f"{fixture_name}.json"
    if not fixture_path.exists():
        return {}
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _clean_html(raw_html: str) -> dict[str, str]:
    html = raw_html or ""
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = _collapse_whitespace(_strip_tags(title_match.group(1))) if title_match else ""
    body = re.sub(
        r"<(script|style|nav|header|footer|svg)\b[^>]*>.*?</\1>",
        " ",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.DOTALL)
    text = _collapse_whitespace(_strip_tags(body))
    return {"title": title, "text": text}


def _strip_tags(value: str) -> str:
    return html_lib.unescape(re.sub(r"<[^>]+>", " ", value))


def _collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
