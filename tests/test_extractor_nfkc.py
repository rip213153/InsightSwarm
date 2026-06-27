"""Tests for the NFKC upgrade to `_normalize_for_backcheck` and `_quote_ngrams`.

NFKC compatibility normalization folds visually-identical strings from different
sources (fullwidth digits, ligatures, superscripts) to a single canonical form
so that the deterministic quote backcheck and the n-gram jaccard dedup both
treat them as equal. This catches syndicated copies that differ only in
compatibility form — a common failure mode when one source is a PDF paste and
the other is the original HTML.

These tests pin the new behavior so a future regression to strip+lower would
fail loudly.
"""
from __future__ import annotations

from insightswarm.agents.extractor_tools import (
    _normalize_for_backcheck,
    _quote_jaccard,
    _quote_ngrams,
)


class TestNormalizeForBackcheckNFKC:
    def test_legacy_whitespace_and_lowercase_preserved(self) -> None:
        # The old behavior (strip + collapse whitespace + lower) must still work.
        assert _normalize_for_backcheck("  Hello   World  ") == "hello world"
        assert _normalize_for_backcheck("HELLO") == "hello"

    def test_fullwidth_digits_fold_to_ascii(self) -> None:
        # Fullwidth "Ｑ３" → "Q3" under NFKC. A snippet pasted from a PDF that
        # uses fullwidth digits must match the page text that uses ASCII digits.
        assert _normalize_for_backcheck("Ｑ３ earnings") == "q3 earnings"

    def test_fullwidth_letters_fold_to_ascii(self) -> None:
        assert _normalize_for_backcheck("Ｒｅｖｅｎｕｅ") == "revenue"

    def test_ligature_fi_folds_to_fi(self) -> None:
        # U+FB01 (ﬁ) → "fi" under NFKC.
        assert _normalize_for_backcheck("ﬁnancial report") == "financial report"

    def test_superscript_folds_to_digit(self) -> None:
        # U+00B2 (²) → "2" under NFKC.
        assert _normalize_for_backcheck("Q2 2026") == "q2 2026"

    def test_nonbreaking_space_becomes_regular_space(self) -> None:
        # U+00A0 (non-breaking space) → U+0020 under NFKC, then collapsed.
        assert _normalize_for_backcheck("hello\u00a0world") == "hello world"

    def test_cjk_compatibility_ideograph_folds(self) -> None:
        # U+F900 (豈) is a CJK compatibility ideograph that NFKC folds to its
        # canonical form U+8C48 (豈). Both forms must normalize identically so a
        # copy using the compatibility form matches the canonical form.
        canonical = "豈熊收入"
        compatibility = "\uf900熊收入"
        assert _normalize_for_backcheck(canonical) == _normalize_for_backcheck(compatibility)

    def test_backcheck_substring_match_with_fullwidth(self) -> None:
        """The backcheck does `normalized_quote in normalized_text`. A fullwidth
        quote must be found inside ASCII text after both are NFKC-normalized.
        """
        text = "The company reported Q3 revenue of one hundred billion dollars."
        quote_fullwidth = "Ｑ３ revenue of one hundred billion"
        assert _normalize_for_backcheck(quote_fullwidth) in _normalize_for_backcheck(text)


class TestQuoteNgramsNFKC:
    def test_ngrams_identical_for_fullwidth_and_ascii(self) -> None:
        """A quote with fullwidth digits must produce the same n-gram set as the
        ASCII version, so jaccard dedup catches syndicated copies that differ
        only in compatibility form.
        """
        ascii_quote = "The company reported Q3 revenue of one hundred billion dollars"
        fullwidth_quote = "The company reported Ｑ３ revenue of one hundred billion dollars"
        assert _quote_ngrams(ascii_quote) == _quote_ngrams(fullwidth_quote)

    def test_jaccard_one_for_fullwidth_vs_ascii(self) -> None:
        ascii_quote = "The company reported Q3 revenue of one hundred billion dollars"
        fullwidth_quote = "The company reported Ｑ３ revenue of one hundred billion dollars"
        assert _quote_jaccard(ascii_quote, fullwidth_quote) == 1.0

    def test_legacy_punctuation_stripping_preserved(self) -> None:
        # "earnings." and "earnings" must still tokenize identically.
        assert _quote_ngrams("earnings.") == _quote_ngrams("earnings")
