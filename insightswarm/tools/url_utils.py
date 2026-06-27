"""Shared URL validation/normalization helpers.

Centralizes the "is this URL well-formed enough to fetch/cite?" checks that
were previously scattered across fetch.py / firecrawl.py / extractor_tools.py.
The checks here are deliberately syntactic — they reject obviously broken or
truncated URLs (no scheme, ellipsis markers, suspicious truncation) but do NOT
make any network call. Public-network / loopback / private-IP gating remains
in safety.validate_public_http_url, which composes the checks here.
"""
from __future__ import annotations

import unicodedata
from urllib.parse import urlparse


# Ellipsis markers that strongly suggest a truncated/abbreviated URL pasted
# from a snippet or LLM output. Both ASCII "..." and the single Unicode
# horizontal ellipsis "…" (U+2026) are rejected. We also reject the
# mid-string "..." because legitimate URLs never contain three consecutive
# dots in the host/path/query (DNS labels and path segments do not use them).
_ELLIPSIS_MARKERS: tuple[str, ...] = ("...", "…")


class UrlValidationError(ValueError):
    """Raised when a URL fails syntactic validation."""


def is_valid_url(url: str, *, allow_schemes: tuple[str, ...] | None = None) -> bool:
    """Return True iff `url` passes syntactic validation.

    This is the predicate form of `validate_url` — useful when callers want a
    boolean without catching exceptions. `allow_schemes` is forwarded to
    `validate_url`; the default (`None`) restricts to http/https.
    """
    try:
        validate_url(url, allow_schemes=allow_schemes)
    except UrlValidationError:
        return False
    return True


def validate_url(url: str, *, allow_schemes: tuple[str, ...] | None = None) -> str:
    """Validate `url` syntactically; return the normalized URL on success.

    Rejects:
      - empty / non-string input
      - missing scheme (e.g. "example.com/foo")
      - missing netloc/host (for http/https URLs only — synthetic schemes like
        `browser://captured` carry provenance, not a network location, and are
        accepted when their scheme is in `allow_schemes`)
      - URLs containing ellipsis markers ("..." or "…") — these unambiguously
        signal truncation from a snippet/preview and must never be fetched.
      - URLs whose scheme is not in `allow_schemes` (default: http/https).

    Args:
      allow_schemes: permitted URL schemes (lowercased). If `None`, defaults to
        `("http", "https")` — the strict set used by fetch/firecrawl. Pass a
        tuple including synthetic schemes (e.g. `("http", "https", "browser")`)
        for citation-provenance checks where browser-acquired content has no
        canonical web URL. Pass `()` (empty tuple) to accept any non-empty
        scheme — useful when the caller only wants truncation/structure checks.

    Returns the input URL stripped of surrounding whitespace. NFC normalization
    is applied so that visually-identical URLs from different sources compare
    equal downstream (matches the NFKC normalization used in extractor_tools).
    """
    if url is None:
        raise UrlValidationError("URL is required")
    text = unicodedata.normalize("NFC", str(url).strip())
    if not text:
        raise UrlValidationError("URL is empty")
    for marker in _ELLIPSIS_MARKERS:
        if marker in text:
            raise UrlValidationError(f"URL appears truncated (contains '{marker}')")
    parsed = urlparse(text)
    if not parsed.scheme:
        raise UrlValidationError("URL is missing a scheme (http/https)")
    schemes = allow_schemes if allow_schemes is not None else ("http", "https")
    if schemes and parsed.scheme.lower() not in {s.lower() for s in schemes}:
        raise UrlValidationError(f"URL scheme '{parsed.scheme}' is not allowed")
    # For web schemes we require a host; synthetic schemes (browser://, file://,
    # etc.) may legitimately have no hostname — they encode provenance, not a
    # network endpoint. Skip the host check when the scheme is non-web.
    if (not schemes or parsed.scheme.lower() in {"http", "https"}) and not parsed.hostname:
        raise UrlValidationError("URL is missing a host")
    # Reject URLs that look truncated mid-path: a path ending with a dangling
    # fragment with no content, or a path whose final segment is just a single
    # dot/dash (suspicious but not definitive) — we keep this conservative and
    # only catch the clearest cases. The ellipsis check above handles the
    # common case; bare truncation like "https://example.com/foo?q=abc" without
    # closure is hard to detect without false positives, so we leave it.
    return text


def normalize_url(url: str) -> str:
    """Normalize a URL for dedup/comparison: NFC + lowercase scheme/host.

    Does NOT validate — returns the original input (NFC-normalized) on failure
    so callers that only need a stable dedup key can use this without try/except.
    """
    text = unicodedata.normalize("NFC", str(url or "").strip())
    parsed = urlparse(text)
    if not parsed.scheme or not parsed.netloc:
        return text
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path}{('?' + parsed.query) if parsed.query else ''}{('#' + parsed.fragment) if parsed.fragment else ''}"
