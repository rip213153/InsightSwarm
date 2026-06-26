from __future__ import annotations

from insightswarm.agents.researcher_tools import _dedupe_candidates, _priority_hint, _title_ngrams


def test_dedupe_candidates_drops_exact_url_duplicates() -> None:
    candidates = [
        {"url": "https://example.com/a", "title": "A"},
        {"url": "https://example.com/a/", "title": "Different title same page"},
        {"url": "https://example.com/b", "title": "B"},
    ]
    kept = _dedupe_candidates(candidates)
    assert len(kept) == 2
    assert kept[0]["url"] == "https://example.com/a"
    assert kept[1]["url"] == "https://example.com/b"


def test_dedupe_candidates_drops_near_duplicate_titles() -> None:
    # Syndicated copy: same article, near-identical title, different domain.
    candidates = [
        {"url": "https://news.site.com/article", "title": "Company X announces record Q3 earnings"},
        {"url": "https://mirror.news.com/x-q3", "title": "Company X announces record Q3 earnings."},
        {"url": "https://other.com/different", "title": "Completely unrelated story about weather"},
    ]
    kept = _dedupe_candidates(candidates)
    assert len(kept) == 2
    assert kept[0]["url"] == "https://news.site.com/article"
    assert kept[1]["url"] == "https://other.com/different"


def test_dedupe_candidates_preserves_corroboration() -> None:
    # Two sources reporting the SAME fact with DIFFERENT wording must both be kept.
    # This is corroboration, not duplication — the high threshold must not merge these.
    candidates = [
        {"url": "https://reuters.com/x-earnings", "title": "Company X posts strong Q3 revenue growth"},
        {"url": "https://bloomberg.com/x-results", "title": "X Corporation beats earnings expectations in third quarter"},
        {"url": "https://wsj.com/x-q3", "title": "Firm X delivers solid third-quarter financial performance"},
    ]
    kept = _dedupe_candidates(candidates)
    assert len(kept) == 3, "semantically-similar-but-distinct titles must NOT be deduped"


def test_dedupe_candidates_respects_existing_urls() -> None:
    existing = {"https://example.com/already-seen"}
    candidates = [
        {"url": "https://example.com/already-seen", "title": "Already fetched"},
        {"url": "https://example.com/new", "title": "New"},
    ]
    kept = _dedupe_candidates(candidates, existing_urls=existing)
    assert len(kept) == 1
    assert kept[0]["url"] == "https://example.com/new"


def test_priority_hint_high_overlap() -> None:
    hint = _priority_hint("DeepSeek funding round", "DeepSeek closes new funding round", "Details about the round", "")
    assert hint == "high"


def test_priority_hint_low_overlap() -> None:
    hint = _priority_hint("DeepSeek funding round", "Weather forecast for Tuesday", "Sunny skies expected", "")
    assert hint == "low"


def test_priority_hint_unknown_when_no_query() -> None:
    assert _priority_hint("", "Some title", "Some snippet", "") == "unknown"


def test_title_ngrams_short_text() -> None:
    grams = _title_ngrams("ab")
    assert grams == {"ab"}


def test_title_ngrams_normal_text() -> None:
    grams = _title_ngrams("hello")
    assert "hel" in grams and "ell" in grams and "llo" in grams
