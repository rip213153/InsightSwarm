from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from insightswarm.tools.core import ToolContext, ToolResult


ALLOWED_URL_SCHEMES = ("http", "https")


def validate_public_http_url(url: str, context: ToolContext | None = None) -> ToolResult | None:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        return ToolResult(
            "blocked",
            error="URL scheme is not allowed",
            diagnostics={"allowed_schemes": list(ALLOWED_URL_SCHEMES)},
        )
    if not parsed.hostname:
        return ToolResult("blocked", error="URL host is required", diagnostics={"url": url})
    if _allows_local_http(context):
        return None
    host = parsed.hostname.lower()
    if host in {"localhost"} or host.endswith(".localhost"):
        return ToolResult("blocked", error="Localhost URLs are blocked in production mode", diagnostics={"host": host})
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None
    if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
        return ToolResult("blocked", error="Internal network URLs are blocked in production mode", diagnostics={"host": host})
    return None


def _allows_local_http(context: ToolContext | None) -> bool:
    return bool(context and context.quality_mode == "test")
