"""Tests for the shared URL validation/normalization helpers.

These helpers centralize the "is this URL well-formed enough to fetch/cite?"
checks that were previously scattered across fetch.py / firecrawl.py /
extractor_tools.py. The tests here cover the rejection rules (truncation,
missing scheme, non-web schemes) and the scheme-allowlist escape hatch that
citation-provenance checks need for synthetic schemes like `browser://`.
"""
from __future__ import annotations

import pytest

from insightswarm.tools.url_utils import (
    UrlValidationError,
    is_valid_url,
    normalize_url,
    validate_url,
)


class TestValidateUrl:
    def test_accepts_well_formed_https(self) -> None:
        assert validate_url("https://example.com/path?q=1#frag") == "https://example.com/path?q=1#frag"

    def test_accepts_http(self) -> None:
        assert validate_url("http://example.com") == "http://example.com"

    def test_strips_surrounding_whitespace(self) -> None:
        assert validate_url("  https://example.com  ") == "https://example.com"

    def test_rejects_empty(self) -> None:
        with pytest.raises(UrlValidationError):
            validate_url("")

    def test_rejects_none(self) -> None:
        with pytest.raises(UrlValidationError):
            validate_url(None)  # type: ignore[arg-type]

    def test_rejects_missing_scheme(self) -> None:
        with pytest.raises(UrlValidationError, match="scheme"):
            validate_url("example.com/foo")

    def test_rejects_ftp_scheme_by_default(self) -> None:
        with pytest.raises(UrlValidationError, match="not allowed"):
            validate_url("ftp://example.com/file")

    def test_rejects_ascii_ellipsis(self) -> None:
        with pytest.raises(UrlValidationError, match="truncated"):
            validate_url("https://example.com/foo...")

    def test_rejects_unicode_ellipsis(self) -> None:
        with pytest.raises(UrlValidationError, match="truncated"):
            validate_url("https://example.com/foo…")

    def test_rejects_ellipsis_in_query(self) -> None:
        with pytest.raises(UrlValidationError, match="truncated"):
            validate_url("https://example.com/search?q=abc...")

    def test_rejects_missing_host_for_web_scheme(self) -> None:
        with pytest.raises(UrlValidationError, match="host"):
            validate_url("https:///path-only-no-host")

    def test_allow_schemes_accepts_browser(self) -> None:
        # Synthetic `browser://` provenance must be accepted when the caller
        # explicitly allows it (extractor citation path).
        assert validate_url("browser://captured", allow_schemes=("http", "https", "browser")) == "browser://captured"

    def test_allow_schemes_empty_accepts_any_scheme(self) -> None:
        # Empty tuple = "accept any non-empty scheme" — only truncation/structure
        # checks apply. Useful when the caller only wants to filter truncated URLs.
        assert validate_url("custom://anything", allow_schemes=()) == "custom://anything"

    def test_allow_schemes_still_rejects_ellipsis(self) -> None:
        with pytest.raises(UrlValidationError, match="truncated"):
            validate_url("browser://cap...tured", allow_schemes=("http", "https", "browser"))

    def test_nfc_normalization_applied(self) -> None:
        # NFC folds combining characters — U+0041 (A) + U+0308 (combining diaeresis)
        # becomes U+00C4 (Ä). The validator should accept and return NFC form.
        composed = "https://example.com/\u00c4"
        decomposed = "https://example.com/A\u0308"
        assert validate_url(decomposed) == composed


class TestIsValidUrl:
    def test_returns_true_for_valid(self) -> None:
        assert is_valid_url("https://example.com") is True

    def test_returns_false_for_truncated(self) -> None:
        assert is_valid_url("https://example.com/foo...") is False

    def test_returns_false_for_missing_scheme(self) -> None:
        assert is_valid_url("example.com") is False

    def test_allow_schemes_param_forwarded(self) -> None:
        assert is_valid_url("browser://captured", allow_schemes=("http", "https", "browser")) is True
        assert is_valid_url("browser://captured") is False  # default rejects non-web


class TestNormalizeUrl:
    def test_lowercases_scheme_and_host(self) -> None:
        assert normalize_url("HTTPS://Example.COM/Path") == "https://example.com/Path"

    def test_preserves_query_and_fragment(self) -> None:
        assert normalize_url("https://example.com/p?q=1#f") == "https://example.com/p?q=1#f"

    def test_returns_input_for_no_scheme(self) -> None:
        # No validation — just normalization. Bare strings pass through.
        assert normalize_url("not-a-url") == "not-a-url"

    def test_handles_empty(self) -> None:
        assert normalize_url("") == ""
        assert normalize_url(None) == ""  # type: ignore[arg-type]
