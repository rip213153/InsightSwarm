from __future__ import annotations

import os
import socket
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from insightswarm.fetching import fetch_source, should_fallback_to_browser, sync_playwright


pytestmark = pytest.mark.browser_smoke


def _require_browser_smoke() -> None:
    if os.getenv("INSIGHTSWARM_BROWSER_SMOKE") != "1":
        pytest.skip("set INSIGHTSWARM_BROWSER_SMOKE=1 to run browser smoke tests")
    if sync_playwright is None:
        pytest.skip("playwright Python package is unavailable")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002, ANN001
        return


@pytest.fixture
def local_site(tmp_path):
    _require_browser_smoke()
    (tmp_path / "static.html").write_text(
        "<html><body><main><p>ExampleCo pricing is $29 per user per month.</p></main></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "spa.html").write_text(
        """
        <html>
          <body><div id="root"></div>
          <script>
            setTimeout(() => {
              document.getElementById('root').innerText =
                'Delayed browser content shows ExampleCo pricing at $29 per user per month.';
            }, 50);
          </script></body>
        </html>
        """,
        encoding="utf-8",
    )
    port = _free_port()
    handler = partial(QuietHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_static_page_does_not_need_browser(local_site):
    result = fetch_source(f"{local_site}/static.html", timeout=5)
    assert result.ok
    assert result.fetcher == "httpx"
    assert "ExampleCo pricing" in (result.text or "")
    assert result.metadata["fallback"]["reason"] in {"text_sufficient", "short_text_but_static_structure"}


def test_spa_shell_uses_browser_or_returns_diagnostic(local_site):
    fallback, meta = should_fallback_to_browser('<html><body><div id="root"></div></body></html>', "")
    assert fallback is True
    assert meta["reason"] == "short_text_and_spa_shell"
    result = fetch_source(f"{local_site}/spa.html", timeout=8)
    attempts = result.metadata.get("attempts", [])
    browser_attempt = next((item for item in attempts if item.get("fetcher") == "playwright"), None)
    assert browser_attempt is not None
    browser_meta = browser_attempt.get("metadata") or {}
    assert "wait_strategy" in browser_meta
    assert "html_chars" in browser_meta
    assert "text_chars" in browser_meta
    assert "screenshot_captured" in browser_meta
    if result.ok:
        assert result.fetcher == "playwright"
        assert "Delayed browser content" in (result.text or "")
    else:
        assert result.status == "error"
        assert result.metadata.get("error_kind") or browser_meta.get("error_kind")
