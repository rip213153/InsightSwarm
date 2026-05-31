from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from insightswarm.util import new_id

try:
    import websocket  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    websocket = None  # type: ignore


READ_ONLY_TEXT_EXPRESSION = "document.body ? document.body.innerText : ''"
READ_ONLY_SNAPSHOT_EXPRESSION = """
(() => ({
  url: location.href,
  title: document.title,
  text: document.body ? document.body.innerText : '',
  html_chars: document.documentElement ? document.documentElement.outerHTML.length : 0
}))()
""".strip()
READ_ONLY_PAGE_STATE_EXPRESSION = """
(() => {
  const maxElements = Math.max(1, Math.min(Number(arguments[0]) || 20, 50));
  const maxTextChars = Math.max(80, Math.min(Number(arguments[1]) || 500, 2000));
  const maxNameChars = Math.max(20, Math.min(Number(arguments[2]) || 80, 200));
  const clip = (value, max) => String(value || '').replace(/\\s+/g, ' ').trim().slice(0, max);
  const roleFor = (el) => el.getAttribute('role') || ({
    A: 'link',
    BUTTON: 'button',
    INPUT: 'input',
    SELECT: 'select',
    TEXTAREA: 'textbox',
    SUMMARY: 'button'
  })[el.tagName] || el.tagName.toLowerCase();
  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style && style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 0 && rect.height > 0;
  };
  const candidates = Array.from(document.querySelectorAll('a,button,input,select,textarea,summary,[role],[tabindex]'));
  const elements = [];
  for (const el of candidates) {
    if (!isVisible(el)) continue;
    const rect = el.getBoundingClientRect();
    const text = clip(el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('title') || el.getAttribute('name') || el.href || '', maxNameChars);
    const domIndex = elements.length + 1;
    const container = el.closest('li, article, section, div') || el;
    const containerText = clip(container.innerText || text, maxNameChars * 2);
    elements.push({
      dom_index: domIndex,
      stable_node_id: 'dom-' + elements.length + '-' + el.tagName.toLowerCase(),
      role: roleFor(el),
      name: text,
      text,
      tag: el.tagName.toLowerCase(),
      href: el.href || null,
      action_hint: el.href ? 'navigate' : (['INPUT','TEXTAREA','SELECT'].includes(el.tagName) ? 'input' : 'activate'),
      bbox: {x: Math.round(rect.x), y: Math.round(rect.y), width: Math.round(rect.width), height: Math.round(rect.height)},
      frame_id: 'main',
      visibility: 'visible',
      container_context: containerText,
      nearby_text: containerText,
      preferred_action: el.href ? 'click' : (['INPUT','TEXTAREA','SELECT'].includes(el.tagName) ? 'type' : 'click')
    });
    if (elements.length >= maxElements) break;
  }
  const bodyText = document.body ? document.body.innerText : '';
  return {
    url: location.href,
    title: document.title,
    text_preview: clip(bodyText, maxTextChars),
    visible_text_chars: bodyText.length,
    html_chars: document.documentElement ? document.documentElement.outerHTML.length : 0,
    node_count: document.querySelectorAll('*').length,
    interactable_count: candidates.length,
    interactable_elements: elements,
    truncated: candidates.length > elements.length || bodyText.length > maxTextChars
  };
})()
""".strip()


@dataclass(frozen=True)
class BrowserObservation:
    backend: str
    session_id: str
    observation: dict[str, Any]
    diagnostics: dict[str, Any]


class BrowserBackendUnavailable(RuntimeError):
    def __init__(self, reason: str, diagnostics: dict[str, Any] | None = None):
        super().__init__(reason)
        self.diagnostics = diagnostics or {}


class FakeBrowserBackend:
    backend_type = "fake"

    def __init__(self, session_id: str | None = None, cdp_url: str | None = None):
        self.session_id = session_id or "fake-browser-session"
        self.cdp_url = cdp_url

    def connect(self) -> None:
        return None

    def close(self) -> None:
        return None

    def snapshot(self, tool_input: dict[str, Any]) -> BrowserObservation:
        observation = {
            "url": tool_input.get("url") or "https://example.com/pricing",
            "title": "Example Pricing Page",
            "text_preview": "ExampleCo pricing page. Starter plan costs $49 per month.",
            "html_chars": 128,
            "interactable_elements": [
                {"node_id": "fake-node-pricing", "role": "tab", "text": "Pricing"},
                {"node_id": "fake-node-plan", "role": "text", "text": "Starter plan costs $49 per month."},
            ],
        }
        return self._observation(observation)

    def page_state(self, tool_input: dict[str, Any]) -> BrowserObservation:
        max_elements = _bounded_int(tool_input.get("max_elements"), 20, 1, 50)
        max_text_chars = _bounded_int(tool_input.get("max_text_chars"), 500, 80, 2000)
        text = "ExampleCo pricing page. Starter plan costs $49 per month. Contact sales for enterprise terms."
        elements = [
            {
                "stable_node_id": "fake-link-pricing",
                "dom_index": 1,
                "role": "link",
                "name": "Pricing",
                "text": "Pricing",
                "tag": "a",
                "href": "https://example.com/pricing",
                "action_hint": "navigate",
                "bbox": {"x": 16, "y": 16, "width": 72, "height": 24},
                "frame_id": "main",
                "visibility": "visible",
                "semantic_type": "product_detail_link",
                "container_context": "ExampleCo pricing page",
                "nearby_text": "ExampleCo pricing page",
                "preferred_action": "click",
            },
            {
                "stable_node_id": "fake-button-contact",
                "dom_index": 2,
                "role": "button",
                "name": "Contact sales",
                "text": "Contact sales",
                "tag": "button",
                "href": None,
                "action_hint": "activate",
                "bbox": {"x": 120, "y": 16, "width": 110, "height": 32},
                "frame_id": "main",
                "visibility": "visible",
                "semantic_type": "unknown",
                "container_context": "Contact sales for enterprise terms.",
                "nearby_text": "Contact sales for enterprise terms.",
                "preferred_action": "click",
            },
        ][:max_elements]
        observation = {
            "url": tool_input.get("url") or "https://example.com/pricing",
            "title": "Example Pricing Page",
            "text_preview": text[:max_text_chars],
            "visible_text_chars": len(text),
            "html_chars": 256,
            "node_count": 8,
            "interactable_count": 2,
            "interactable_elements": elements,
            "truncated": max_elements < 2 or len(text) > max_text_chars,
        }
        return self._observation(observation)

    def visible_text(self, tool_input: dict[str, Any]) -> BrowserObservation:
        return self._observation({"text": "ExampleCo pricing page. Starter plan costs $49 per month."})

    def screenshot(self, tool_input: dict[str, Any]) -> BrowserObservation:
        return self._observation({"screenshot_captured": True, "screenshot_base64": None, "screenshot_bytes": 0})

    def execute(self, action: str, tool_input: dict[str, Any]) -> BrowserObservation:
        observation = {
            "status": "fake_action_executed",
            "action": action,
            "target_id": tool_input.get("target_id") or tool_input.get("stable_node_id"),
            "url": tool_input.get("url") or "https://example.com/pricing",
        }
        return self._observation(observation)

    def _observation(self, observation: dict[str, Any]) -> BrowserObservation:
        return BrowserObservation(
            self.backend_type,
            self.session_id,
            observation,
            {"browser_backend": self.backend_type, "read_only": True, "fake_execution": True},
        )


class CdpBrowserBackend:
    backend_type = "cdp"

    def __init__(self, session_id: str | None = None, cdp_url: str | None = None, timeout: float = 5.0):
        self.session_id = session_id or new_id("browser")
        self.cdp_url = cdp_url
        self.timeout = timeout
        self._ws = None
        self._next_id = 1

    @classmethod
    def available(cls) -> bool:
        return websocket is not None

    def connect(self) -> None:
        if websocket is None:
            raise BrowserBackendUnavailable(
                "browser backend unavailable: install optional browser extra",
                {"browser_backend_unavailable": True, "missing_dependency": "websocket-client"},
            )
        if not self.cdp_url:
            raise BrowserBackendUnavailable(
                "browser backend unavailable: cdp_url is required",
                {"browser_backend_unavailable": True, "missing_cdp_url": True},
            )
        parsed = urlparse(self.cdp_url)
        if parsed.scheme not in {"ws", "wss"}:
            raise BrowserBackendUnavailable(
                "browser backend unavailable: cdp_url must be ws:// or wss://",
                {"browser_backend_unavailable": True, "invalid_cdp_url": True},
            )
        self._ws = websocket.create_connection(self.cdp_url, timeout=self.timeout)  # type: ignore[union-attr]

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None

    def snapshot(self, tool_input: dict[str, Any]) -> BrowserObservation:
        payload = self._runtime_evaluate(READ_ONLY_SNAPSHOT_EXPRESSION)
        value = _remote_value(payload)
        text = str(value.get("text") or "") if isinstance(value, dict) else ""
        observation = {
            "url": value.get("url") if isinstance(value, dict) else None,
            "title": value.get("title") if isinstance(value, dict) else None,
            "text_preview": text[:500],
            "html_chars": value.get("html_chars") if isinstance(value, dict) else None,
            "interactable_elements": [],
        }
        return self._observation(observation)

    def page_state(self, tool_input: dict[str, Any]) -> BrowserObservation:
        max_elements = _bounded_int(tool_input.get("max_elements"), 20, 1, 50)
        max_text_chars = _bounded_int(tool_input.get("max_text_chars"), 500, 80, 2000)
        max_name_chars = _bounded_int(tool_input.get("max_name_chars"), 80, 20, 200)
        payload = self._runtime_evaluate(
            READ_ONLY_PAGE_STATE_EXPRESSION,
            [max_elements, max_text_chars, max_name_chars],
        )
        value = _remote_value(payload)
        observation = _normalize_page_state(value if isinstance(value, dict) else {}, max_elements, max_text_chars)
        diagnostics = {
            "cdp_methods_used": ["Runtime.evaluate"],
            "node_count": observation.get("node_count"),
            "interactable_count": observation.get("interactable_count"),
            "page_state_truncated": observation.get("truncated"),
        }
        return self._observation(observation, diagnostics)

    def visible_text(self, tool_input: dict[str, Any]) -> BrowserObservation:
        payload = self._runtime_evaluate(READ_ONLY_TEXT_EXPRESSION)
        return self._observation({"text": str(_remote_value(payload) or "")})

    def screenshot(self, tool_input: dict[str, Any]) -> BrowserObservation:
        payload = self._send("Page.captureScreenshot", {"format": "png", "fromSurface": True})
        data = ((payload.get("result") or {}).get("data") or "")
        return self._observation(
            {
                "screenshot_captured": bool(data),
                "screenshot_base64": data[:120] + "...[truncated]" if data else None,
                "screenshot_bytes": len(base64.b64decode(data)) if data else 0,
            }
        )

    def execute(self, action: str, tool_input: dict[str, Any]) -> BrowserObservation:
        if action == "goto":
            self._send("Page.navigate", {"url": tool_input["url"]})
            self._wait(float(tool_input.get("wait_seconds") or 0.5))
            return self._observation({"status": "executed", "action": action, "url": tool_input["url"]})
        if action == "click":
            x, y = _bbox_center(tool_input.get("bbox") or {})
            self._send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1})
            self._send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1})
            self._wait(float(tool_input.get("wait_seconds") or 0.25))
            return self._observation({"status": "executed", "action": action, "target_id": tool_input.get("target_id"), "x": x, "y": y})
        if action == "type":
            x, y = _bbox_center(tool_input.get("bbox") or {})
            self._send("Input.dispatchMouseEvent", {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1})
            self._send("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1})
            self._send("Input.insertText", {"text": str(tool_input.get("text") or "")[:500]})
            return self._observation({"status": "executed", "action": action, "target_id": tool_input.get("target_id"), "text_chars": len(str(tool_input.get("text") or ""))})
        if action == "scroll":
            direction = str(tool_input.get("direction") or "down").lower()
            delta = int(tool_input.get("delta") or 600)
            if direction in {"up", "left"}:
                delta = -abs(delta)
            self._send("Input.dispatchMouseEvent", {"type": "mouseWheel", "x": 300, "y": 300, "deltaY": delta if direction in {"up", "down"} else 0, "deltaX": delta if direction in {"left", "right"} else 0})
            return self._observation({"status": "executed", "action": action, "direction": direction, "delta": delta})
        if action == "wait":
            seconds = min(max(float(tool_input.get("seconds") or 1), 0), 5)
            self._wait(seconds)
            return self._observation({"status": "executed", "action": action, "seconds": seconds})
        raise BrowserBackendUnavailable(
            f"browser action {action} is not supported by real backend",
            {"browser_backend": self.backend_type, "unsupported_action": action},
        )

    def _runtime_evaluate(self, expression: str, arguments: list[Any] | None = None) -> dict[str, Any]:
        expression_text = expression
        for index, value in enumerate(arguments or []):
            expression_text = expression_text.replace(f"arguments[{index}]", json.dumps(value))
        return self._send(
            "Runtime.evaluate",
            {"expression": expression_text, "returnByValue": True, "awaitPromise": False},
        )

    def _send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._ws is None:
            self.connect()
        request_id = self._next_id
        self._next_id += 1
        self._ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        while True:
            response = json.loads(self._ws.recv())
            if response.get("id") != request_id:
                continue
            if "error" in response:
                raise BrowserBackendUnavailable(
                    f"cdp command failed: {response['error']}",
                    {"browser_backend_unavailable": False, "cdp_error": response["error"], "method": method},
                )
            return response

    def _wait(self, seconds: float) -> None:
        time.sleep(min(max(seconds, 0), 5))

    def _observation(self, observation: dict[str, Any], extra_diagnostics: dict[str, Any] | None = None) -> BrowserObservation:
        return BrowserObservation(
            self.backend_type,
            self.session_id,
            observation,
            {
                "browser_backend": self.backend_type,
                "read_only": True,
                "fake_execution": False,
                "cdp_url_present": bool(self.cdp_url),
                **(extra_diagnostics or {}),
            },
        )


class VisibleCdpBrowserBackend(CdpBrowserBackend):
    backend_type = "visible_cdp"

    def __init__(
        self,
        session_id: str | None = None,
        cdp_url: str | None = None,
        timeout: float = 5.0,
        browser_exe: str | None = None,
        user_data_dir: str | None = None,
    ):
        super().__init__(session_id=session_id, cdp_url=cdp_url, timeout=timeout)
        self.browser_exe = browser_exe
        self.user_data_dir = user_data_dir
        self._process: subprocess.Popen[bytes] | None = None

    def connect(self) -> None:
        if self.cdp_url:
            return super().connect()
        self._launch_visible_browser()
        return super().connect()

    def close(self) -> None:
        super().close()

    def _launch_visible_browser(self) -> None:
        if self._process and self._process.poll() is None:
            return
        browser_exe = self.browser_exe or _find_browser_executable()
        if not browser_exe:
            raise BrowserBackendUnavailable(
                "browser backend unavailable: Chrome/Edge executable not found",
                {"browser_backend_unavailable": True, "missing_browser_executable": True},
            )
        port = _find_free_port()
        self.cdp_url = None
        user_data_dir = self.user_data_dir or str(Path.cwd() / ".tmp" / "browser-profiles" / self.session_id)
        Path(user_data_dir).mkdir(parents=True, exist_ok=True)
        args = [
            browser_exe,
            f"--remote-debugging-port={port}",
            "--remote-allow-origins=*",
            "--new-window",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={user_data_dir}",
            "about:blank",
        ]
        self._process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _wait_for_json_endpoint(port, self.timeout)
        self.cdp_url = _discover_page_ws_url(port, self.timeout)


class BrowserSession:
    def __init__(self, backend: str = "fake", session_id: str | None = None, cdp_url: str | None = None):
        self.backend_type = backend
        self.session_id = session_id or ("fake-browser-session" if backend == "fake" else new_id("browser"))
        self.cdp_url = cdp_url
        self.backend = self._make_backend()

    def _make_backend(self):
        if self.backend_type in {"cdp", "visible", "visible_cdp"}:
            if self.backend_type in {"visible", "visible_cdp"} and not self.cdp_url:
                return VisibleCdpBrowserBackend(self.session_id, self.cdp_url)
            return CdpBrowserBackend(self.session_id, self.cdp_url)
        return FakeBrowserBackend(self.session_id, self.cdp_url)

    def connect(self) -> None:
        self.backend.connect()

    def close(self) -> None:
        self.backend.close()

    def observe(self, action: str, tool_input: dict[str, Any]) -> BrowserObservation:
        if action == "snapshot":
            return self.backend.snapshot(tool_input)
        if action == "visible_text":
            return self.backend.visible_text(tool_input)
        if action == "screenshot":
            return self.backend.screenshot(tool_input)
        if action == "page_state":
            return self.backend.page_state(tool_input)
        if self.backend_type != "fake":
            raise BrowserBackendUnavailable(
                f"browser action {action} is not supported by real read-only backend",
                {"browser_backend": self.backend_type, "unsupported_action": action, "read_only": True},
            )
        return self.backend._observation({"status": "fake_action_observed", "action": action})

    def execute(self, action: str, tool_input: dict[str, Any]) -> BrowserObservation:
        return self.backend.execute(action, tool_input)


def _remote_value(payload: dict[str, Any]) -> Any:
    result = (payload.get("result") or {}).get("result") or {}
    return result.get("value")


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _bbox_center(bbox: dict[str, Any]) -> tuple[float, float]:
    width = float(bbox.get("width") or 0)
    height = float(bbox.get("height") or 0)
    if width <= 0 or height <= 0:
        raise BrowserBackendUnavailable("browser target bbox is missing or not visible", {"error_kind": "bbox_missing"})
    return float(bbox.get("x") or 0) + width / 2, float(bbox.get("y") or 0) + height / 2


def _find_browser_executable() -> str | None:
    candidates = [
        os.getenv("INSIGHTSWARM_BROWSER_EXE"),
        shutil.which("chrome"),
        shutil.which("chrome.exe"),
        shutil.which("msedge"),
        shutil.which("msedge.exe"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_json_endpoint(port: int, timeout: float) -> None:
    deadline = time.time() + max(timeout, 1.0)
    version_url = f"http://127.0.0.1:{port}/json/version"
    while time.time() < deadline:
        try:
            with urlopen(version_url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except (URLError, OSError, TimeoutError):
            time.sleep(0.2)
    raise BrowserBackendUnavailable(
        "browser backend unavailable: visible chrome did not expose the debugging endpoint in time",
        {"browser_backend_unavailable": True, "debug_endpoint_timeout": True, "debug_port": port},
    )


def _discover_page_ws_url(port: int, timeout: float) -> str:
    deadline = time.time() + max(timeout, 1.0)
    while time.time() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1.0) as response:
                if response.status != 200:
                    continue
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
                for target in payload:
                    if isinstance(target, dict) and target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                        return str(target["webSocketDebuggerUrl"])
        except (URLError, OSError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(0.2)
    raise BrowserBackendUnavailable(
        "browser backend unavailable: could not discover a page websocket url",
        {"browser_backend_unavailable": True, "missing_page_websocket": True},
    )


def _normalize_page_state(value: dict[str, Any], max_elements: int, max_text_chars: int) -> dict[str, Any]:
    elements = value.get("interactable_elements") if isinstance(value.get("interactable_elements"), list) else []
    normalized_elements = []
    for item in elements[:max_elements]:
        if not isinstance(item, dict):
            continue
        normalized_elements.append(
            {
                "stable_node_id": str(item.get("stable_node_id") or "")[:80],
                "dom_index": item.get("dom_index"),
                "role": str(item.get("role") or "unknown")[:40],
                "name": str(item.get("name") or item.get("text") or "")[:200],
                "text": str(item.get("text") or item.get("name") or "")[:200],
                "tag": str(item.get("tag") or "")[:40],
                "href": item.get("href"),
                "action_hint": str(item.get("action_hint") or "")[:40],
                "bbox": item.get("bbox") if isinstance(item.get("bbox"), dict) else None,
                "frame_id": str(item.get("frame_id") or "main")[:80],
                "visibility": str(item.get("visibility") or "unknown")[:40],
                "semantic_type": str(item.get("semantic_type") or "unknown")[:80],
                "container_context": str(item.get("container_context") or "")[:300],
                "nearby_text": str(item.get("nearby_text") or "")[:300],
                "negative_signals": item.get("negative_signals") if isinstance(item.get("negative_signals"), list) else [],
                "preferred_action": str(item.get("preferred_action") or "")[:40],
            }
        )
    return {
        "url": value.get("url"),
        "title": value.get("title"),
        "text_preview": str(value.get("text_preview") or "")[:max_text_chars],
        "visible_text_chars": int(value.get("visible_text_chars") or 0),
        "html_chars": value.get("html_chars"),
        "node_count": int(value.get("node_count") or 0),
        "interactable_count": int(value.get("interactable_count") or len(normalized_elements)),
        "interactable_elements": normalized_elements,
        "truncated": bool(value.get("truncated") or len(elements) > len(normalized_elements)),
    }
