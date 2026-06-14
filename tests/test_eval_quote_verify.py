from __future__ import annotations

from insightswarm.eval.quote_verify import (
    QuoteMatch,
    normalize,
    verify_citations,
    verify_quote,
)


def test_exact_substring_matches_verbatim():
    source = "JD.com announced a strategic investment in instant retail logistics."
    match = verify_quote("a strategic investment in instant retail", source)
    assert match.matched is True
    assert match.match_type == "exact"
    assert match.similarity == 1.0


def test_empty_quote_is_not_grounded():
    match = verify_quote("   ", "any source text")
    assert match.matched is False
    assert match.match_type == "empty"


def test_normalized_match_survives_whitespace_reflow():
    # LLM collapsed a newline + double space into something different.
    source = "OpenAI released\n  the model    in    early 2023."
    quote = "OpenAI released the model in early 2023."
    match = verify_quote(quote, source)
    assert match.matched is True
    assert match.match_type == "normalized"


def test_normalized_match_survives_fullwidth_punctuation():
    # Source uses Chinese full-width punctuation; quote uses ASCII.
    source = "京东宣布，将加大对即时零售的投入（包括物流）。"
    quote = "京东宣布,将加大对即时零售的投入(包括物流)。"
    match = verify_quote(quote, source)
    assert match.matched is True
    assert match.match_type == "normalized"


def test_normalized_match_strips_nbsp_and_zero_width():
    # nbsp stands in for the space; a zero-width char sits inside a word.
    source = "The&nbsp;fuel sur​charge was raised by 5%."
    quote = "The fuel surcharge was raised by 5%."
    match = verify_quote(quote, source)
    assert match.matched is True
    assert match.match_type == "normalized"


def test_case_folding_match():
    source = "The Verge reported the acquisition closed in Q4."
    quote = "the verge reported the acquisition closed in q4."
    match = verify_quote(quote, source)
    assert match.matched is True
    assert match.match_type in {"normalized", "exact"}


def test_fuzzy_match_tolerates_minor_edit():
    # A single dropped word -> not a normalized substring, but very close.
    source = "The company reported quarterly revenue of 1.2 billion dollars in total."
    quote = "The company reported quarterly revenue of 1.2 billion dollars total."
    match = verify_quote(quote, source)
    assert match.matched is True
    assert match.match_type == "fuzzy"
    assert match.similarity >= 0.90
    assert match.best_window


def test_unrelated_quote_is_not_grounded():
    source = "JD.com announced a strategic investment in instant retail logistics."
    quote = "Microsoft acquired a video game studio for ten billion dollars."
    match = verify_quote(quote, source)
    assert match.matched is False
    assert match.match_type == "none"
    assert match.similarity < 0.90


def test_quote_longer_than_source_scored_not_crashed():
    match = verify_quote("a very long quote that exceeds the source entirely", "short")
    assert isinstance(match, QuoteMatch)
    assert match.matched is False


def test_normalize_is_idempotent():
    raw = "京东宣布，\n\n将加大​投入（A）"
    assert normalize(normalize(raw)) == normalize(raw)


def test_verify_citations_routes_by_url_and_aggregates():
    source_by_url = {
        "https://jd.com/ir": "JD.com announced a strategic investment in instant retail logistics.",
        "https://verge.com/openai": "OpenAI released the model in early 2023.",
    }
    citations = [
        {"source_url": "https://jd.com/ir", "quote": "a strategic investment in instant retail", "claim": "c1"},
        {"source_url": "https://verge.com/openai", "quote": "OpenAI released   the model in early 2023.", "claim": "c2"},
        {"source_url": "https://jd.com/ir", "quote": "totally fabricated claim not present", "claim": "c3"},
    ]
    summary = verify_citations(citations, source_by_url)
    assert summary.total == 3
    assert summary.exact == 1
    assert summary.normalized == 1
    assert summary.unmatched == 1
    assert summary.grounded == 2
    assert 0.66 <= summary.grounded_ratio <= 0.67


def test_verify_citations_falls_back_to_corpus_when_url_missing():
    citations = [{"source_url": "", "quote": "instant retail logistics", "claim": "c1"}]
    summary = verify_citations(
        citations,
        source_by_url={},
        corpus="JD.com announced a strategic investment in instant retail logistics.",
    )
    assert summary.grounded == 1
    assert summary.total == 1
