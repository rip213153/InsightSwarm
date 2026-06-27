from __future__ import annotations

import http.client
import socket
import urllib.error
import urllib.request
from typing import Any

import orjson


class HttpResponseError(Exception):
    """Raised when an HTTP request returns a non-2xx status code."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"HTTP {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class HttpRequestError(Exception):
    """Raised when an HTTP request fails at the network level.

    Covers timeouts, connection resets, incomplete reads, DNS failures, and
    similar transport-level problems.
    """


def request_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: Any = None,
    timeout: float = 30.0,
) -> Any:
    """Perform an HTTP request and parse the JSON response.

    Args:
        url: Target URL.
        method: HTTP verb (``GET``, ``POST``, ...).
        headers: Request headers.
        body: Request body. ``dict``/``list`` values are serialized with
            orjson. ``bytes`` are sent as-is. ``None`` sends no body.
        timeout: Socket timeout in seconds.

    Returns:
        Parsed JSON response, or ``None`` for an empty body.

    Raises:
        HttpResponseError: Non-2xx response.
        HttpRequestError: Network-level failure.
    """
    if body is None:
        data: bytes | None = None
    elif isinstance(body, bytes):
        data = body
    else:
        data = orjson.dumps(body)
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers or {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            if not raw:
                return None
            return orjson.loads(raw)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HttpResponseError(exc.code, detail) from exc
    except urllib.error.URLError as exc:
        raise HttpRequestError(f"request failed: {exc.reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise HttpRequestError(f"request timed out: {exc}") from exc
    except http.client.IncompleteRead as exc:
        raise HttpRequestError(f"response incomplete: {exc}") from exc
    except (ConnectionResetError, OSError) as exc:
        raise HttpRequestError(f"connection failed: {exc}") from exc
