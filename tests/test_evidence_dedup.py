from __future__ import annotations

from insightswarm.agents.extractor_tools import _quote_jaccard, _quote_ngrams, _normalize_for_backcheck


def test_quote_ngrams_short_quote() -> None:
    grams = _quote_ngrams("short quote")
    # 2 tokens < n=4, so returns the tokens joined
    assert grams == {"short quote"}


def test_quote_ngrams_normal_quote() -> None:
    grams = _quote_ngrams("the company reported revenue of one hundred billion")
    assert "the company reported revenue" in grams
    assert "company reported revenue of" in grams


def test_quote_jaccard_identical_quotes() -> None:
    q = "the company reported revenue of one hundred billion dollars"
    assert _quote_jaccard(q, q) == 1.0


def test_quote_jaccard_near_duplicate_passes_threshold() -> None:
    # Syndicated press release: same text, minor punctuation difference.
    a = "The company reported revenue of one hundred billion dollars in the third quarter."
    b = "The company reported revenue of one hundred billion dollars in the third quarter"
    assert _quote_jaccard(a, b) >= 0.85


def test_quote_jaccard_paraphrase_does_not_pass_threshold() -> None:
    # Same fact, different wording — corroboration, must NOT be flagged as duplicate.
    a = "The company reported revenue of one hundred billion dollars in the third quarter."
    b = "Firm X announced that Q3 top-line exceeded the hundred-billion mark according to their filing."
    assert _quote_jaccard(a, b) < 0.85


def test_quote_jaccard_empty() -> None:
    assert _quote_jaccard("", "something") == 0.0
    assert _quote_jaccard("", "") == 0.0


def test_normalize_for_backcheck_collapses_whitespace() -> None:
    assert _normalize_for_backcheck("  Hello   World  ") == "hello world"


def test_normalize_for_backcheck_lowercase() -> None:
    assert _normalize_for_backcheck("HELLO") == "hello"
