from __future__ import annotations

import contextlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from insightswarm.agents.browser_agent_tools import (
    BrowserAgentToolHandlers,
    BrowserAgentToolState,
    _execute_code_with_result,
    _format_execution_output,
    _result_for_tool_payload,
    _safe_globals,
    _validate_code,
)
from insightswarm.schemas.swarm import Task


MAX_BROWSER_CELLS = 18
MAX_CELL_CODE_CHARS = 8000
MAX_CELL_OUTPUT_CHARS = 12000


@dataclass
class BrowserCodeCell:
    index: int
    code: str
    success: bool
    output: str
    error: str | None
    browser_state: dict[str, Any]
    model_text: str = ""


@dataclass
class BrowserCodeSessionResult:
    cells: list[BrowserCodeCell] = field(default_factory=list)
    terminal_status: str | None = None
    terminal_reason: str | None = None


class BrowserCodeSession:
    def __init__(
        self,
        *,
        task: Task,
        handlers: BrowserAgentToolHandlers,
        tool_state: BrowserAgentToolState,
        model_client: object,
        trace_path: Path | None = None,
        max_cells: int = MAX_BROWSER_CELLS,
    ):
        self.task = task
        self.handlers = handlers
        self.tool_state = tool_state
        self.model_client = model_client
        self.trace_path = trace_path
        self.max_cells = max_cells
        self.namespace: dict[str, Any] = {}
        self.cells: list[BrowserCodeCell] = []

    def run(self) -> BrowserCodeSessionResult:
        task_context = self.handlers.read_task({})
        self.namespace.update(self._base_namespace())
        last_result: dict[str, Any] = {"ok": True, "output": "BrowserCodeSession started. Write the first code cell."}

        for index in range(1, self.max_cells + 1):
            if self.tool_state.terminal_status:
                break
            prompt = self._assemble_prompt(task_context, last_result)
            model_result = self.model_client.complete(
                [{"role": "user", "content": prompt}],
                max_tokens=1800,
                temperature=0.2,
                metadata={"role": "browser_agent_code_session", "task_id": self.task.task_id},
            )
            model_text = getattr(model_result, "text", "") or ""
            if getattr(model_result, "status", "") != "ok":
                last_result = {"ok": False, "error": getattr(model_result, "error", None) or "model failed"}
                self._append_cell(index, "", False, "", last_result["error"], model_text)
                continue
            code = _extract_code_cell(model_text)
            if not code:
                error = "model did not return a python code cell"
                last_result = {"ok": False, "error": error}
                self._append_cell(index, "", False, "", error, model_text)
                continue
            last_result = self._execute_cell(index, code, model_text)
            if self.tool_state.terminal_status:
                break

        if not self.tool_state.terminal_status:
            self.handlers._finish_browser("blocked", "BrowserCodeSession reached max cells without finishing")
        return BrowserCodeSessionResult(
            cells=list(self.cells),
            terminal_status=self.tool_state.terminal_status,
            terminal_reason=self.tool_state.terminal_reason,
        )

    def _base_namespace(self) -> dict[str, Any]:
        namespace = self.handlers._namespace()
        namespace.update(
            {
                "navigate": namespace["open_url"],
                "click": namespace["click_link"],
                "done": namespace["finish_browser"],
                "inspect_visual_page": lambda why_vision_needed, question=None: self.handlers.inspect_visual_page(
                    {"why_vision_needed": why_vision_needed, "question": question}
                ),
            }
        )
        return namespace

    def _assemble_prompt(self, task_context: dict[str, Any], last_result: dict[str, Any]) -> str:
        prompt_path = Path(__file__).resolve().parent / "prompts" / "browser_code_agent.md"
        system_prompt = prompt_path.read_text(encoding="utf-8")
        recent_cells = [
            {
                "index": cell.index,
                "success": cell.success,
                "output": cell.output[-1200:],
                "error": cell.error,
                "browser_state": cell.browser_state,
            }
            for cell in self.cells[-4:]
        ]
        payload = {
            "task": task_context,
            "browser_state": self.handlers._browser_state_summary(),
            "last_result": last_result,
            "recent_cells": recent_cells,
            "available_functions": [
                "navigate(url)",
                "page_state(max_elements=20, max_text_chars=2000)",
                "visible_text(max_chars=6000)",
                "evaluate(js_expression)",
                "click(dom_index=None, stable_node_id=None, why='')",
                "dismiss_cookie_banner(prefer='reject')",
                "scroll(direction='down')",
                "collect_visible_text(max_scrolls=3, max_chars=12000)",
                "inspect_visual_page(why_vision_needed, question=None)",
                "publish_raw_source(text, url=None, title=None, why_ready='')",
                "request_authorization(reason)",
                "request_login_authorization(login_url=None, reason='')",
                "done(status='complete'|'blocked', reason='...')",
            ],
        }
        return system_prompt + "\n\nSESSION_STATE:\n" + json.dumps(payload, ensure_ascii=False, indent=2)

    def _execute_cell(self, index: int, code: str, model_text: str) -> dict[str, Any]:
        if len(code) > MAX_CELL_CODE_CHARS:
            error = "browser code cell is too long"
            self._append_cell(index, code, False, "", error, model_text)
            return {"ok": False, "error": error}
        blocked = _validate_code(code)
        if blocked:
            self._append_cell(index, code, False, "", blocked, model_text)
            return {"ok": False, "error": blocked}
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                result = _execute_code_with_result(code, _safe_globals(), self.namespace)
            output = _format_execution_output(stdout.getvalue(), result)
            self._append_cell(index, code, True, output, None, model_text)
            return {"ok": True, "output": output, "result": _result_for_tool_payload(result)}
        except Exception as exc:
            output = stdout.getvalue()
            error = f"{type(exc).__name__}: {exc}"
            self._append_cell(index, code, False, output, error, model_text)
            return {"ok": False, "output": output[-MAX_CELL_OUTPUT_CHARS:], "error": error}

    def _append_cell(self, index: int, code: str, success: bool, output: str, error: str | None, model_text: str) -> None:
        cell = BrowserCodeCell(
            index=index,
            code=code,
            success=success,
            output=output[-MAX_CELL_OUTPUT_CHARS:],
            error=error,
            browser_state=self.handlers._browser_state_summary(),
            model_text=model_text[-3000:],
        )
        self.cells.append(cell)
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "role": "browser_agent_code_session",
                            "task_id": self.task.task_id,
                            "cell": cell.__dict__,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def _extract_code_cell(text: str) -> str:
    for pattern in (r"```python\s*(.*?)```", r"```\s*(.*?)```"):
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return ""
        return str(payload.get("code") or "").strip()
    return ""
