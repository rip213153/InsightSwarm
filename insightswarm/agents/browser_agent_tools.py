from __future__ import annotations

import ast
import contextlib
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from insightswarm.authorization_flow import authorization_decision_for_task
from insightswarm.browser_backend import BrowserBackendUnavailable, BrowserSession
from insightswarm.schemas.swarm import Task
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore
from insightswarm.tools.safety import validate_public_http_url


BROWSER_ROLE = "browser_agent"
EXTRACTOR_ROLE = "extractor"

MAX_BROWSER_CODE_CHARS = 5000
MAX_BROWSER_OUTPUT_CHARS = 6000
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
BLOCKED_IMPORTS = {
    "asyncio",
    "httpx",
    "os",
    "pathlib",
    "requests",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
HIGH_RISK_MARKERS = {
    "authorization",
    "cookie",
    "download",
    "header",
    "headers",
    "javascript",
    "js",
    "localstorage",
    "login",
    "log in",
    "password",
    "payment",
    "purchase",
    "submit",
    "token",
    "upload",
}
LOGIN_MARKERS = {"login", "log in", "sign in"}


BROWSER_AGENT_TOOLS = [
    {
        "name": "read_task",
        "description": "Read the assigned hard acquisition goal, target URL, source request context, and safety constraints.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
        "output_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "target_url": {"type": "string"},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "available_browser_functions": {"type": "array", "items": {"type": "string"}},
            },
        },
        "side_effects": "none",
    },
    {
        "name": "execute_browser_code",
        "description": (
            "Execute a small Python snippet in BrowserAgent's persistent browser namespace. "
            "Use this to navigate, inspect page state, extract visible text, publish raw source, "
            "request authorization, or finish. The namespace is restricted and read-mostly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "why_this_code": {"type": "string"},
            },
            "required": ["code", "why_this_code"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "output": {"type": "string"},
                "browser_state": {"type": "object"},
                "published_artifact_ids": {"type": "array"},
                "authorization_requested": {"type": "boolean"},
            },
        },
        "side_effects": "may navigate/read browser, write raw_document artifacts, create extractor tasks, or request authorization",
    },
]


@dataclass
class BrowserAgentToolState:
    task_context: dict[str, Any] | None = None
    session_id: str | None = None
    browser_session: BrowserSession | None = None
    namespace: dict[str, Any] = field(default_factory=dict)
    current_url: str = ""
    last_page_state: dict[str, Any] = field(default_factory=dict)
    last_visible_text: str = ""
    browser_trace: list[dict[str, Any]] = field(default_factory=list)
    trace_message_id: str | None = None
    created_task_ids: list[str] = field(default_factory=list)
    created_message_ids: list[str] = field(default_factory=list)
    created_artifact_ids: list[str] = field(default_factory=list)
    terminal_status: str | None = None
    terminal_reason: str | None = None
    browser_code_counts: dict[str, int] = field(default_factory=dict)
    browser_timeout_counts: dict[str, int] = field(default_factory=dict)


class HumanAuthorizationRequired(RuntimeError):
    pass


class BrowserAgentToolHandlers:
    def __init__(
        self,
        *,
        task: Task,
        task_store: TaskStore,
        mailbox: Mailbox,
        artifact_store: ArtifactStore,
        board_store: BoardStore,
        state: BrowserAgentToolState,
    ):
        self.task = task
        self.task_store = task_store
        self.mailbox = mailbox
        self.artifact_store = artifact_store
        self.board_store = board_store
        self.state = state

    def handlers(self) -> dict[str, Any]:
        return {
            "read_task": self.read_task,
            "execute_browser_code": self.execute_browser_code,
        }

    def read_task(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        del tool_input
        goal = _goal(self.task)
        target_url = _target_url(self.task)
        context = {
            "task_id": self.task.task_id,
            "task_kind": self.task.kind,
            "goal": goal,
            "target_url": target_url,
            "reason": _safe_text(self.task.inputs.get("reason")),
            "constraints": [
                "Only acquire public information relevant to the hard acquisition goal.",
                "Do not type, upload, download, access cookies/storage/headers/tokens/passwords, submit forms, or run arbitrary JavaScript.",
                "Login or account-gated navigation requires an allowlisted domain and explicit human authorization; if approved, the human operator must complete credentials manually in the visible browser.",
                "Publish raw source only when visible/extracted text is relevant and substantial enough for Extractor.",
                "If the goal requires login, call request_login_authorization(login_url, reason) from execute_browser_code.",
                "If the goal requires other high-risk interaction, call request_authorization(reason) from execute_browser_code.",
            ],
            "authorization_status": self._authorization_status(),
            "login_authorization": {
                "allowlist": _login_allowlist(self.task),
                "target_url_allowed": _login_allowed_for_url(target_url, self.task),
            },
            "available_browser_functions": [
                "open_url(url)",
                "page_state(max_elements=20, max_text_chars=2000)",
                "visible_text(max_chars=6000)",
                "assess_page(goal=None)",
                "click_link(dom_index=None, stable_node_id=None, why='')",
                "dismiss_cookie_banner(prefer='reject')",
                "collect_visible_text(max_scrolls=3, max_chars=12000)",
                "scroll(direction='down')",
                "wait(seconds=1)",
                "publish_raw_source(text, url=None, title=None, why_ready='')",
                "request_authorization(reason)",
                "request_login_authorization(login_url=None, reason='')",
                "finish_browser(status='complete'|'blocked', reason='...')",
            ],
        }
        self.state.task_context = context
        return {"ok": True, **context}

    def execute_browser_code(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        if self.state.task_context is None:
            return {"ok": False, "error": "read_task first"}
        code = _safe_text(tool_input.get("code"))
        if len(code) > MAX_BROWSER_CODE_CHARS:
            return {"ok": False, "error": "browser code is too long"}
        blocked = _validate_code(code)
        if blocked:
            return {"ok": False, "error": blocked}
        repeat_block = self._repeat_block_for_code(code)
        if repeat_block:
            return repeat_block

        namespace = self._namespace()
        stdout = io.StringIO()
        result: Any = None
        action_key = _browser_action_key(code)
        try:
            with contextlib.redirect_stdout(stdout):
                result = _execute_code_with_result(code, _safe_globals(), namespace)
        except HumanAuthorizationRequired as exc:
            self._request_authorization(str(exc))
        except Exception as exc:
            self._record_browser_code_failure(action_key, exc)
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "output": _clip(stdout.getvalue(), MAX_BROWSER_OUTPUT_CHARS),
                "browser_state": self._browser_state_summary(),
            }
        finally:
            self._record_browser_code_attempt(code)
            self.state.namespace = {
                key: value
                for key, value in namespace.items()
                if not key.startswith("_") and key not in {"open_url", "page_state", "visible_text", "assess_page", "click_link", "dismiss_cookie_banner", "collect_visible_text", "scroll", "wait", "publish_raw_source", "request_authorization", "request_login_authorization", "finish_browser"}
            }

        return {
            "ok": True,
            "success": True,
            "output": _format_execution_output(stdout.getvalue(), result),
            "result": _result_for_tool_payload(result),
            "browser_state": self._browser_state_summary(),
            "published_artifact_ids": list(self.state.created_artifact_ids),
            "created_task_ids": list(self.state.created_task_ids),
            "authorization_requested": self.state.terminal_status == "blocked" and bool(self.state.terminal_reason and "authorization" in self.state.terminal_reason.lower()),
            "terminal": bool(self.state.terminal_status),
            "status": self.state.terminal_status,
            "reason": self.state.terminal_reason,
        }

    def _repeat_block_for_code(self, code: str) -> dict[str, Any] | None:
        code_key = _normalize_code_for_repeat(code)
        action_key = _browser_action_key(code)
        if self.state.browser_code_counts.get(code_key, 0) >= 2:
            return {
                "ok": False,
                "error": "repeated_browser_code_blocked: same browser code has already been attempted twice",
                "repeat_key": code_key,
                "required_next_strategy": "publish_raw_source, finish_browser(blocked), or choose a materially different browser strategy",
                "browser_state": self._browser_state_summary(),
            }
        if self.state.browser_timeout_counts.get(action_key, 0) >= 2:
            return {
                "ok": False,
                "error": "repeated_browser_timeout_blocked: this browser action family has already timed out twice",
                "repeat_key": action_key,
                "required_next_strategy": "use a lighter read, publish the best available visible text, or finish_browser(blocked)",
                "browser_state": self._browser_state_summary(),
            }
        return None

    def _record_browser_code_attempt(self, code: str) -> None:
        code_key = _normalize_code_for_repeat(code)
        self.state.browser_code_counts[code_key] = self.state.browser_code_counts.get(code_key, 0) + 1

    def _record_browser_code_failure(self, action_key: str, exc: Exception) -> None:
        error_text = f"{type(exc).__name__}: {exc}".lower()
        if "timeout" not in error_text:
            return
        self.state.browser_timeout_counts[action_key] = self.state.browser_timeout_counts.get(action_key, 0) + 1

    def _namespace(self) -> dict[str, Any]:
        namespace = dict(self.state.namespace)
        namespace.update(
            {
                "goal": _goal(self.task),
                "target_url": _target_url(self.task),
                "open_url": self._open_url,
                "page_state": self._page_state,
                "visible_text": self._visible_text,
                "assess_page": self._assess_page,
                "click_link": self._click_link,
                "dismiss_cookie_banner": self._dismiss_cookie_banner,
                "collect_visible_text": self._collect_visible_text,
                "scroll": self._scroll,
                "wait": self._wait,
                "publish_raw_source": self._publish_raw_source,
                "request_authorization": self._request_authorization,
                "request_login_authorization": self._request_login_authorization,
                "finish_browser": self._finish_browser,
            }
        )
        return namespace

    def _open_url(self, url: str | None = None) -> dict[str, Any]:
        url = _safe_text(url) or _target_url(self.task)
        if not url:
            raise ValueError("open_url requires a URL")
        blocked = validate_public_http_url(url, None)
        if blocked:
            raise ValueError(blocked.error or "URL is not allowed")
        marker = _high_risk_marker(url)
        if marker:
            if marker in LOGIN_MARKERS and not _login_allowed_for_url(url, self.task):
                raise ValueError(f"login navigation is not allowlisted for this domain: {_url_host(url) or url}")
            self._require_authorized_interaction(f"opening URL with high-risk marker requires authorization: {marker}")
        session = self._session()
        observation = session.execute("goto", {"url": url, "wait_seconds": 1.0})
        self.state.current_url = url
        self._record_trace("open_url", {"url": url, "status": observation.observation.get("status")})
        return observation.observation

    def _page_state(self, max_elements: int = 20, max_text_chars: int = 2000) -> dict[str, Any]:
        session = self._session()
        observation = session.observe(
            "page_state",
            {
                "max_elements": max_elements,
                "max_text_chars": max_text_chars,
                "max_name_chars": 120,
            },
        )
        self.state.session_id = observation.session_id
        page = dict(observation.observation or {})
        self.state.last_page_state = page
        self.state.current_url = _safe_text(page.get("url")) or self.state.current_url
        self._record_trace(
            "page_state",
            {
                "url": page.get("url"),
                "title": page.get("title"),
                "visible_text_chars": page.get("visible_text_chars"),
                "interactable_count": page.get("interactable_count"),
            },
        )
        return page

    def _visible_text(self, max_chars: int = 6000) -> str:
        session = self._session()
        observation = session.observe("visible_text", {})
        text = _safe_text((observation.observation or {}).get("text"))
        self.state.last_visible_text = text
        self._record_trace("visible_text", {"chars": len(text), "url": self.state.current_url})
        return text[: max(200, min(int(max_chars or 6000), 12000))]

    def _assess_page(self, goal: str | None = None) -> dict[str, Any]:
        if not self.state.last_page_state:
            self._page_state(max_elements=12, max_text_chars=1200)
        if not self.state.last_visible_text:
            self._visible_text(max_chars=3000)
        assessment = _assess_page_quality(
            self.state.last_page_state,
            self.state.last_visible_text,
            goal=_safe_text(goal) or _goal(self.task),
        )
        self._record_trace("assess_page", assessment)
        return assessment

    def _collect_visible_text(self, max_scrolls: int = 3, max_chars: int = 12000) -> dict[str, Any]:
        max_scrolls = max(0, min(int(max_scrolls or 0), 5))
        max_chars = max(1000, min(int(max_chars or 12000), 24000))
        snapshots: list[dict[str, Any]] = []
        chunks: list[str] = []
        seen: set[str] = set()

        for index in range(max_scrolls + 1):
            page = self._page_state(max_elements=12, max_text_chars=1200)
            text = self._visible_text(max_chars=max_chars)
            normalized = " ".join(text.split())
            if normalized and normalized not in seen:
                chunks.append(text)
                seen.add(normalized)
            snapshots.append(
                {
                    "step": index,
                    "url": page.get("url"),
                    "title": page.get("title"),
                    "visible_text_chars": page.get("visible_text_chars"),
                    "text_preview": page.get("text_preview"),
                }
            )
            if index >= max_scrolls:
                break
            self._scroll("down")
            self._wait(0.7)

        combined = "\n\n--- visible scroll boundary ---\n\n".join(chunks)
        if len(combined) > max_chars:
            combined = combined[:max_chars]
        self.state.last_visible_text = combined
        self._record_trace(
            "collect_visible_text",
            {
                "text_chars": len(combined),
                "scrolls_performed": max_scrolls,
                "url": self.state.current_url,
                "title": self.state.last_page_state.get("title"),
            },
        )
        return {
            "text": combined,
            "text_chars": len(combined),
            "snapshots": snapshots,
            "scrolls_performed": max_scrolls,
            "current_url": self.state.current_url,
            "title": self.state.last_page_state.get("title"),
        }

    def _click_link(self, dom_index: int | None = None, stable_node_id: str | None = None, why: str = "") -> dict[str, Any]:
        if not self.state.last_page_state:
            raise ValueError("call page_state before click_link")
        target = _find_interactable(self.state.last_page_state, dom_index=dom_index, stable_node_id=stable_node_id)
        if not target:
            raise ValueError("click_link target not found in the latest page_state")
        role = _safe_text(target.get("role")).lower()
        tag = _safe_text(target.get("tag")).lower()
        href = _safe_text(target.get("href"))
        label = _safe_text(target.get("name") or target.get("text") or target.get("container_context"))
        marker = _high_risk_marker(" ".join([href, label, why]))
        if marker:
            if marker in LOGIN_MARKERS and href and not _login_allowed_for_url(href, self.task):
                raise ValueError(f"login navigation is not allowlisted for this domain: {_url_host(href) or href}")
            self._require_authorized_interaction(f"clicking link with high-risk marker requires authorization: {marker}")
        if tag != "a" and role != "link":
            self._require_authorized_interaction("click_link only allows public link navigation; non-link interaction requires authorization")
        if not href:
            self._require_authorized_interaction("click_link target has no href; ambiguous click requires authorization")
        if href:
            blocked = validate_public_http_url(href, None)
            if blocked:
                raise ValueError(blocked.error or "link URL is not allowed")
        bbox = target.get("bbox") if isinstance(target.get("bbox"), dict) else {}
        session = self._session()
        observation = session.execute(
            "click",
            {
                "target_id": stable_node_id or target.get("stable_node_id") or dom_index,
                "bbox": bbox,
                "wait_seconds": 1.0,
            },
        )
        page = self._page_state(max_elements=20, max_text_chars=2000)
        self._record_trace("click_link", {"href": href, "text": label, "why": _safe_text(why), "new_url": page.get("url"), "new_title": page.get("title")})
        return {
            "clicked": True,
            "target": {"dom_index": target.get("dom_index"), "stable_node_id": target.get("stable_node_id"), "text": label, "href": href},
            "click_observation": observation.observation,
            "page_state_after_click": page,
        }

    def _dismiss_cookie_banner(self, prefer: str = "reject") -> dict[str, Any]:
        if not self.state.last_page_state:
            self._page_state(max_elements=30, max_text_chars=1600)
        target = _find_cookie_banner_target(self.state.last_page_state, prefer=_safe_text(prefer))
        if not target:
            self._record_trace("dismiss_cookie_banner", {"clicked": False, "reason": "no cookie/privacy overlay control found"})
            return {"clicked": False, "reason": "no cookie/privacy overlay control found"}

        bbox = target.get("bbox") if isinstance(target.get("bbox"), dict) else {}
        session = self._session()
        observation = session.execute(
            "click",
            {
                "target_id": target.get("stable_node_id") or target.get("dom_index"),
                "bbox": bbox,
                "wait_seconds": 0.8,
            },
        )
        page = self._page_state(max_elements=30, max_text_chars=2000)
        label = _safe_text(target.get("name") or target.get("text") or target.get("container_context"))
        self._record_trace(
            "dismiss_cookie_banner",
            {"clicked": True, "label": label, "new_url": page.get("url"), "new_title": page.get("title")},
        )
        return {
            "clicked": True,
            "target": {"dom_index": target.get("dom_index"), "stable_node_id": target.get("stable_node_id"), "text": label},
            "click_observation": observation.observation,
            "page_state_after_click": page,
        }

    def _scroll(self, direction: str = "down") -> dict[str, Any]:
        _raise_if_high_risk(direction)
        session = self._session()
        observation = session.execute("scroll", {"direction": direction, "delta": 700})
        self._record_trace("scroll", {"direction": direction})
        return observation.observation

    def _wait(self, seconds: float = 1) -> dict[str, Any]:
        session = self._session()
        observation = session.execute("wait", {"seconds": min(max(float(seconds or 1), 0), 5)})
        self._record_trace("wait", {"seconds": min(max(float(seconds or 1), 0), 5)})
        return observation.observation

    def _publish_raw_source(self, text: str, url: str | None = None, title: str | None = None, why_ready: str = "") -> dict[str, Any]:
        text = _safe_text(text)
        if len(text) < 120:
            raise ValueError("raw source text is too short to publish")
        source_url = _safe_text(url) or self.state.current_url or _target_url(self.task) or "browser://captured"
        artifact = self.artifact_store.write_raw_document(
            self.task.run_id,
            source_task_id=self.task.task_id,
            document={
                "source_url": source_url,
                "url": source_url,
                "title": _safe_text(title) or _safe_text(self.state.last_page_state.get("title")) or source_url,
                "text": text,
                "html": "",
                "metadata": {
                    "produced_by": BROWSER_ROLE,
                    "browser_agent_goal": _goal(self.task),
                    "publish_reason": _safe_text(why_ready),
                    "acquisition_mode": "browser_code_lite",
                },
            },
            summary=_safe_text(title) or f"Browser raw source for {_goal(self.task)}",
        )
        extractor_task = self.task_store.create(
            self.task.run_id,
            kind="raw_document",
            status="pending",
            owner_role=EXTRACTOR_ROLE,
            inputs={
                "artifact_id": artifact.artifact_id,
                "source_task_id": self.task.task_id,
                "board_item_id": self.task.inputs.get("board_item_id"),
                "issue_key": _safe_text(self.task.inputs.get("issue_key")),
            },
            depends_on=[self.task.task_id],
            priority=self.task.priority,
            created_by=BROWSER_ROLE,
        )
        handoff = self.mailbox.send(
            self.task.run_id,
            from_role=BROWSER_ROLE,
            to_role=EXTRACTOR_ROLE,
            message_type="request",
            payload={"kind": "extract_evidence", "task_id": extractor_task.task_id, "artifact_id": artifact.artifact_id},
            related_task_id=extractor_task.task_id,
        )
        self.state.created_artifact_ids.append(artifact.artifact_id or "")
        self.state.created_task_ids.append(extractor_task.task_id or "")
        self.state.created_message_ids.append(handoff.message_id or "")
        self._record_trace("publish_raw_source", {"artifact_id": artifact.artifact_id, "extractor_task_id": extractor_task.task_id, "url": source_url, "text_chars": len(text)})
        board_item_id = _safe_text(self.task.inputs.get("board_item_id"))
        if board_item_id:
            self.board_store.update_status(board_item_id, status="active")
        return {"artifact_id": artifact.artifact_id, "extractor_task_id": extractor_task.task_id}

    def _request_authorization(self, reason: str) -> dict[str, Any]:
        return self._request_authorization_request(reason, authorization_kind="browser_action", target_url=self.state.current_url or _target_url(self.task))

    def _request_login_authorization(self, login_url: str | None = None, reason: str = "") -> dict[str, Any]:
        url = _safe_text(login_url) or self.state.current_url or _target_url(self.task)
        if not url:
            return {"ok": False, "error": "login_url or current target_url is required"}
        blocked = validate_public_http_url(url, None)
        if blocked:
            return {"ok": False, "error": blocked.error or "login URL is not allowed"}
        if not _login_allowed_for_url(url, self.task):
            return {
                "ok": False,
                "error": f"login domain is not allowlisted: {_url_host(url) or url}",
                "allowlist": _login_allowlist(self.task),
                "authorization_requested": False,
            }
        detail = _safe_text(reason) or f"Manual login may be required for {_url_host(url) or url}; operator must complete credentials in the visible browser."
        return self._request_authorization_request(detail, authorization_kind="login", target_url=url)

    def _request_authorization_request(self, reason: str, *, authorization_kind: str, target_url: str | None = None) -> dict[str, Any]:
        decision = self._authorization_status()
        if decision.get("decision") == "allow":
            self._record_trace("authorization_already_allowed", {"reason": _safe_text(reason)})
            return {"authorized": True, "decision": "allow", "reason": decision.get("reason") or ""}
        if decision.get("decision") == "deny":
            self._record_trace("authorization_denied", {"reason": _safe_text(reason)})
            self.state.terminal_status = "blocked"
            self.state.terminal_reason = f"authorization denied: {decision.get('reason') or _safe_text(reason)}"
            return {"authorized": False, "decision": "deny", "reason": decision.get("reason") or ""}
        message = self.mailbox.send(
            self.task.run_id,
            from_role=BROWSER_ROLE,
            to_role="lead",
            message_type="observation",
            payload={
                "kind": "authorization_request",
                "authorization_kind": authorization_kind,
                "task_id": self.task.task_id,
                "goal": _goal(self.task),
                "target_url": _safe_text(target_url),
                "reason": _safe_text(reason),
            },
            related_task_id=self.task.task_id,
        )
        self.state.created_message_ids.append(message.message_id or "")
        board_item_id = _safe_text(self.task.inputs.get("board_item_id"))
        if board_item_id:
            self.board_store.update_status(board_item_id, status="blocked")
        self._record_trace("request_authorization", {"reason": _safe_text(reason)})
        self._publish_trace_observation("authorization required")
        self.state.terminal_status = "blocked"
        self.state.terminal_reason = f"authorization required: {_safe_text(reason)}"
        return {"message_id": message.message_id}

    def _authorization_status(self) -> dict[str, Any]:
        decision = authorization_decision_for_task(self.task_store.store, self.task.run_id, self.task.task_id or "")
        if decision is None:
            return {"decision": "pending_or_not_requested"}
        reason = ""
        for message in self.task_store.store.list_swarm_messages(self.task.run_id):
            if message.type != "observation" or str(message.payload.get("kind") or "") != "authorization_decision":
                continue
            if str(message.payload.get("task_id") or message.related_task_id or "") != (self.task.task_id or ""):
                continue
            reason = _safe_text(message.payload.get("reason"))
        return {"decision": decision, "reason": reason}

    def _require_authorized_interaction(self, reason: str) -> None:
        decision = self._authorization_status()
        if decision.get("decision") == "allow":
            self._record_trace("authorized_interaction", {"reason": _safe_text(reason)})
            return
        if decision.get("decision") == "deny":
            raise HumanAuthorizationRequired(f"authorization denied: {decision.get('reason') or reason}")
        raise HumanAuthorizationRequired(reason)

    def _finish_browser(self, status: str = "complete", reason: str = "") -> dict[str, Any]:
        status = _safe_text(status) or "complete"
        if status == "complete":
            status = "done"
        if status not in {"done", "blocked"}:
            status = "blocked"
        self.state.terminal_status = status
        self.state.terminal_reason = _safe_text(reason) or status
        self._record_trace("finish_browser", {"status": status, "reason": self.state.terminal_reason})
        message = self.mailbox.send(
            self.task.run_id,
            from_role=BROWSER_ROLE,
            to_role="lead",
            message_type="response",
            payload={"kind": "completed" if status == "done" else "blocked", "reason": self.state.terminal_reason},
            related_task_id=self.task.task_id,
        )
        self.state.created_message_ids.append(message.message_id or "")
        self._publish_trace_observation(self.state.terminal_reason)
        return {"status": status, "reason": self.state.terminal_reason, "message_id": message.message_id}

    def _session(self) -> BrowserSession:
        backend = _safe_text(self.task.inputs.get("browser_backend")) or os.getenv("INSIGHTSWARM_BROWSER_BACKEND") or "fake"
        cdp_url = _safe_text(self.task.inputs.get("browser_cdp_url")) or os.getenv("INSIGHTSWARM_BROWSER_CDP_URL") or None
        session = self.state.browser_session
        if session is None or session.backend_type != backend or (cdp_url and session.cdp_url != cdp_url):
            session = BrowserSession(backend=backend, session_id=self.state.session_id, cdp_url=cdp_url)
            self.state.browser_session = session
        self.state.session_id = session.session_id
        session.connect()
        return session

    def _browser_state_summary(self) -> dict[str, Any]:
        return {
            "session_id": self.state.session_id,
            "current_url": self.state.current_url,
            "last_title": self.state.last_page_state.get("title"),
            "last_visible_text_chars": len(self.state.last_visible_text),
            "created_artifact_count": len(self.state.created_artifact_ids),
            "trace_event_count": len(self.state.browser_trace),
        }

    def _record_trace(self, event: str, payload: dict[str, Any]) -> None:
        self.state.browser_trace.append({"event": event, "payload": _safe_trace_payload(payload)})
        self.state.browser_trace = self.state.browser_trace[-40:]

    def _publish_trace_observation(self, reason: str) -> None:
        if self.state.trace_message_id or not self.state.browser_trace:
            return
        message = self.mailbox.send(
            self.task.run_id,
            from_role=BROWSER_ROLE,
            broadcast=True,
            message_type="observation",
            payload={
                "kind": "browser_trace",
                "task_id": self.task.task_id,
                "goal": _goal(self.task),
                "reason": _safe_text(reason),
                "events": list(self.state.browser_trace),
            },
            related_task_id=self.task.task_id,
        )
        self.state.trace_message_id = message.message_id
        self.state.created_message_ids.append(message.message_id or "")


def _validate_code(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "imports are not allowed in BrowserAgent code"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_NAMES:
                return f"blocked function: {node.func.id}"
            if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("__"):
                return "dunder calls are not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder attributes are not allowed"
        if isinstance(node, ast.Name) and node.id in BLOCKED_IMPORTS:
            return f"blocked name: {node.id}"
    lowered = code.lower()
    for marker in ("type(", "input_text", "evaluate(", "document.cookie", "localstorage", "download", "upload"):
        if marker in lowered:
            return f"blocked browser capability in code: {marker}"
    return None


def _normalize_code_for_repeat(code: str) -> str:
    return " ".join(_safe_text(code).split())


def _browser_action_key(code: str) -> str:
    allowed = {
        "open_url",
        "page_state",
        "visible_text",
        "assess_page",
        "click_link",
        "dismiss_cookie_banner",
        "collect_visible_text",
        "scroll",
        "wait",
        "publish_raw_source",
        "request_authorization",
        "request_login_authorization",
        "finish_browser",
    }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "invalid"
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in allowed:
            calls.append(node.func.id)
    return "+".join(calls) if calls else "execute_browser_code"


def _execute_code_with_result(code: str, globals_dict: dict[str, Any], namespace: dict[str, Any]) -> Any:
    tree = ast.parse(code)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
        expression = ast.Expression(body=tree.body[-1].value)
        ast.fix_missing_locations(prefix)
        ast.fix_missing_locations(expression)
        if prefix.body:
            exec(compile(prefix, "<browser_agent_code>", "exec"), globals_dict, namespace)
        return eval(compile(expression, "<browser_agent_code>", "eval"), globals_dict, namespace)
    exec(compile(tree, "<browser_agent_code>", "exec"), globals_dict, namespace)
    return None


def _find_interactable(page_state: dict[str, Any], *, dom_index: int | None, stable_node_id: str | None) -> dict[str, Any] | None:
    for item in page_state.get("interactable_elements") or []:
        if not isinstance(item, dict):
            continue
        if stable_node_id and str(item.get("stable_node_id") or "") == str(stable_node_id):
            return item
        if dom_index is not None:
            try:
                if int(item.get("dom_index") or -1) == int(dom_index):
                    return item
            except (TypeError, ValueError):
                continue
    return None


def _find_cookie_banner_target(page_state: dict[str, Any], *, prefer: str) -> dict[str, Any] | None:
    items = [item for item in page_state.get("interactable_elements") or [] if isinstance(item, dict)]
    if not items:
        return None
    overlay_words = ("cookie", "privacy", "consent", "gdpr", "tracking", "preferences", "your choice")
    safe_words_by_preference = {
        "reject": ("reject", "decline", "refuse", "deny", "necessary only", "continue without", "close"),
        "accept": ("accept", "agree", "allow all", "ok", "got it", "continue", "close"),
        "close": ("close", "dismiss", "continue", "not now", "reject", "decline"),
    }
    unsafe_words = ("login", "log in", "sign in", "password", "payment", "purchase", "submit", "download", "upload")
    preferred_words = safe_words_by_preference.get(prefer.lower(), safe_words_by_preference["reject"])
    fallback_words = tuple(dict.fromkeys(safe_words_by_preference["reject"] + safe_words_by_preference["accept"]))

    def item_text(item: dict[str, Any]) -> str:
        return " ".join(
            _safe_text(item.get(key)).lower()
            for key in ("name", "text", "container_context", "nearby_text", "role", "tag")
        )

    page_text = " ".join([_safe_text(page_state.get("text_preview")).lower()] + [item_text(item) for item in items])
    if not any(word in page_text for word in overlay_words):
        return None

    for words in (preferred_words, fallback_words):
        for item in items:
            text = item_text(item)
            if any(word in text for word in unsafe_words):
                continue
            if any(word in text for word in words) and any(word in text for word in overlay_words + words):
                return item
    return None


def _assess_page_quality(page_state: dict[str, Any], visible_text: str, *, goal: str) -> dict[str, Any]:
    text = _safe_text(visible_text)
    lowered = text.lower()
    title = _safe_text(page_state.get("title"))
    url = _safe_text(page_state.get("url"))
    text_chars = len(text)
    node_count = int(page_state.get("node_count") or 0)
    interactable_count = int(page_state.get("interactable_count") or 0)
    blocked_markers = [
        "captcha",
        "access denied",
        "403 forbidden",
        "verify you are human",
        "enable javascript",
        "unusual traffic",
        "robot check",
    ]
    signals: list[str] = []
    page_type = "unknown"
    likely_publishable = False

    if any(marker in lowered for marker in blocked_markers):
        page_type = "blocked_or_verification"
        signals.append("blocked_or_verification_text")
    elif text_chars < 300:
        page_type = "thin_page"
        signals.append("very_short_visible_text")
    elif interactable_count >= 25 and text_chars < 1200:
        page_type = "navigation_hub"
        signals.append("many_interactables_low_text")
    elif node_count >= 1200 and text_chars < 1500:
        page_type = "spa_shell_low_signal"
        signals.append("large_dom_low_visible_text")
    elif any(word in url.lower() or word in title.lower() for word in ("docs", "documentation", "tutorial", "guide", "manual", "investor", "press", "news", "pricing")):
        page_type = "source_page"
        signals.append("source_like_url_or_title")
        likely_publishable = text_chars >= 800
    elif text_chars >= 1200:
        page_type = "content_page"
        signals.append("substantial_visible_text")
        likely_publishable = True

    if not signals:
        signals.append("insufficient_static_signals")

    return {
        "url": url,
        "title": title,
        "goal": _safe_text(goal),
        "page_type": page_type,
        "text_chars": text_chars,
        "node_count": node_count,
        "interactable_count": interactable_count,
        "information_density": _density_label(text_chars, node_count),
        "likely_publishable": likely_publishable,
        "signals": signals,
        "recommendation": _page_recommendation(page_type, likely_publishable),
    }


def _density_label(text_chars: int, node_count: int) -> str:
    if text_chars >= 2500:
        return "high"
    if text_chars >= 800:
        return "medium"
    if node_count >= 1000 and text_chars < 1500:
        return "low"
    return "low" if text_chars < 800 else "medium"


def _page_recommendation(page_type: str, likely_publishable: bool) -> str:
    if likely_publishable:
        return "publish_if_relevant"
    if page_type in {"thin_page", "navigation_hub", "spa_shell_low_signal"}:
        return "collect_scroll_or_follow_relevant_public_link"
    if page_type == "blocked_or_verification":
        return "request_authorization_or_finish_blocked"
    return "inspect_more_before_publishing"


def _safe_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            safe[key] = value[:500]
        elif isinstance(value, (int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = value[:10]
        elif isinstance(value, dict):
            safe[key] = _safe_trace_payload(value)
        else:
            safe[key] = str(value)[:200]
    return safe


def _safe_globals() -> dict[str, Any]:
    return {
        "__builtins__": {
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
            "print": print,
            "range": range,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
        },
        "json": json,
    }


def _format_execution_output(stdout_text: str, result: Any) -> str:
    chunks: list[str] = []
    if stdout_text:
        chunks.append("stdout:\n" + stdout_text.strip())
    if result is not None:
        chunks.append("result:\n" + _stringify_result(result))
    if not chunks:
        return "(executed successfully, no output)"
    return _clip("\n\n".join(chunks), MAX_BROWSER_OUTPUT_CHARS)


def _stringify_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except TypeError:
        return str(result)


def _result_for_tool_payload(result: Any) -> Any:
    if result is None or isinstance(result, (int, float, bool)):
        return result
    if isinstance(result, str):
        return _clip(result, MAX_BROWSER_OUTPUT_CHARS)
    if isinstance(result, dict):
        return _json_safe_result(result)
    if isinstance(result, list):
        return [_json_safe_result(item) for item in result[:20]]
    return _clip(str(result), 1000)


def _json_safe_result(value: Any) -> Any:
    if isinstance(value, str):
        return _clip(value, 1200)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe_result(item) for key, item in list(value.items())[:30]}
    if isinstance(value, list):
        return [_json_safe_result(item) for item in value[:20]]
    return _clip(str(value), 500)


def _high_risk_marker(text: str) -> str | None:
    lowered = _safe_text(text).lower()
    for marker in HIGH_RISK_MARKERS:
        if marker in lowered:
            return marker
    return None


def _raise_if_high_risk(text: str) -> None:
    marker = _high_risk_marker(text)
    if marker:
        raise HumanAuthorizationRequired(f"High-risk browser action requires approval: {marker}")


def _login_allowed_for_url(url: str, task: Task) -> bool:
    host = _url_host(url)
    if not host:
        return False
    for allowed in _login_allowlist(task):
        if host == allowed or host.endswith("." + allowed):
            return True
    return False


def _login_allowlist(task: Task) -> list[str]:
    values: list[str] = []
    raw_task_value = task.inputs.get("login_allowlist") or task.inputs.get("login_authorization_allowlist")
    if isinstance(raw_task_value, str):
        values.extend(raw_task_value.split(","))
    elif isinstance(raw_task_value, list):
        values.extend(str(item) for item in raw_task_value)
    env_value = os.getenv("INSIGHTSWARM_BROWSER_LOGIN_ALLOWLIST") or ""
    values.extend(env_value.split(","))
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        host = _normalize_host(value)
        if host and host not in seen:
            seen.add(host)
            normalized.append(host)
    return normalized


def _url_host(url: str) -> str:
    return _normalize_host(urlparse(_safe_text(url)).netloc)


def _normalize_host(value: str) -> str:
    host = _safe_text(value).lower()
    if not host:
        return ""
    if "://" in host:
        host = urlparse(host).netloc.lower()
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if ":" in host:
        host = host.split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host.strip(".")


def _goal(task: Task) -> str:
    for key in ("goal", "question", "objective", "targeted_query"):
        value = _safe_text(task.inputs.get(key))
        if value:
            return value
    return f"{task.kind} {task.task_id or ''}".strip()


def _target_url(task: Task) -> str:
    return _safe_text(task.inputs.get("target_url") or task.inputs.get("url"))


def _clip(value: Any, limit: int) -> str:
    text = _safe_text(value)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
