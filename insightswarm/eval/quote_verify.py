"""Deterministic citation grounding check.

A report citation carries a ``quote`` that is supposed to appear in the raw
source text the swarm fetched. Naive ``quote in source`` is brittle: LLMs
silently reflow whitespace, swap full-width/half-width punctuation (common in
Chinese sources), inject ``&nbsp;`` / zero-width characters, or drop a line
break. We verify in three short-circuiting tiers so the common case stays
O(n) substring work and only true misses pay for fuzzy matching:

1. ``exact``      - raw ``quote`` is a substring of raw source.
2. ``normalized`` - both sides normalized (whitespace folded, punctuation
   unified, zero-width stripped), then substring.
3. ``fuzzy``      - sliding window over normalized source scored by normalized
   Levenshtein similarity; matched when similarity >= threshold.

The result is structured (not a bool) so callers can distinguish verbatim from
approximate quoting and surface low-similarity fuzzy hits for human review.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_FUZZY_THRESHOLD = 0.90

# Full-width / typographic punctuation -> ASCII equivalent. Chinese reports
# frequently round-trip through these, which breaks exact matching.
_PUNCT_MAP = {
    "，": ",", "。": ".", "、": ",", "；": ";", "：": ":",
    "！": "!", "？": "?", "（": "(", "）": ")", "【": "[", "】": "]",
    "「": '"', "」": '"', "『": '"', "』": '"',
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "—": "-", "–": "-", "－": "-", "～": "~", "…": "...",
    "　": " ",  # full-width space
}

# Zero-width and other invisible characters that should never affect matching.
_ZERO_WIDTH = ("​", "‌", "‍", "﻿", "­")


@dataclass(frozen=True)
class QuoteMatch:
    matched: bool
    match_type: str  # "exact" | "normalized" | "fuzzy" | "none" | "empty"
    similarity: float
    best_window: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "matched": self.matched,
            "match_type": self.match_type,
            "similarity": round(self.similarity, 4),
            "best_window": self.best_window,
        }


@dataclass
class CitationCheckSummary:
    total: int = 0
    exact: int = 0
    normalized: int = 0
    fuzzy: int = 0
    unmatched: int = 0
    results: list[dict[str, object]] = field(default_factory=list)

    @property
    def grounded(self) -> int:
        return self.exact + self.normalized + self.fuzzy

    @property
    def grounded_ratio(self) -> float:
        return self.grounded / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "exact": self.exact,
            "normalized": self.normalized,
            "fuzzy": self.fuzzy,
            "unmatched": self.unmatched,
            "grounded": self.grounded,
            "grounded_ratio": round(self.grounded_ratio, 4),
            "results": self.results,
        }


def normalize(text: str) -> str:
    """Fold whitespace, unify punctuation, strip zero-width/invisible chars.

    Case is folded too so quoting that only differs by capitalization still
    matches. The transform is idempotent and applied identically to both the
    quote and the source before comparison.
    """
    if not text:
        return ""
    for zero_width in _ZERO_WIDTH:
        text = text.replace(zero_width, "")
    text = text.replace("&nbsp;", " ")
    text = "".join(_PUNCT_MAP.get(char, char) for char in text)
    text = " ".join(text.split())  # collapse all runs of whitespace to one space
    return text.casefold()


def _levenshtein(left: str, right: str) -> int:
    """Iterative Levenshtein edit distance with two rolling rows."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, lchar in enumerate(left, start=1):
        current = [i]
        for j, rchar in enumerate(right, start=1):
            cost = 0 if lchar == rchar else 1
            current.append(min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + cost,  # substitution
            ))
        previous = current
    return previous[-1]


def _similarity(left: str, right: str) -> float:
    longest = max(len(left), len(right))
    if longest == 0:
        return 1.0
    return 1.0 - _levenshtein(left, right) / longest


def _fuzzy_best_window(needle: str, haystack: str, *, threshold: float) -> QuoteMatch:
    """Slide a window of ~len(needle) over haystack, score by similarity.

    Step is a fraction of the needle length so long sources stay tractable
    while still landing close to any true occurrence; the window is then
    refined around the best coarse hit. Only reached after exact and
    normalized tiers miss, and only on the handful of quotes being checked.
    """
    if not needle or not haystack:
        return QuoteMatch(False, "none", 0.0)
    window_size = len(needle)
    if window_size >= len(haystack):
        score = _similarity(needle, haystack)
        return _fuzzy_result(score, haystack, threshold)

    step = max(1, window_size // 4)
    best_score = 0.0
    best_start = 0
    for start in range(0, len(haystack) - window_size + 1, step):
        window = haystack[start : start + window_size]
        score = _similarity(needle, window)
        if score > best_score:
            best_score, best_start = score, start
            if best_score == 1.0:
                break

    # Refine around the coarse best with a small single-step sweep.
    refine_lo = max(0, best_start - step)
    refine_hi = min(len(haystack) - window_size, best_start + step)
    for start in range(refine_lo, refine_hi + 1):
        window = haystack[start : start + window_size]
        score = _similarity(needle, window)
        if score > best_score:
            best_score, best_start = score, start

    best_window = haystack[best_start : best_start + window_size]
    return _fuzzy_result(best_score, best_window, threshold)


def _fuzzy_result(score: float, window: str, threshold: float) -> QuoteMatch:
    return QuoteMatch(
        matched=score >= threshold,
        match_type="fuzzy" if score >= threshold else "none",
        similarity=score,
        best_window=window if score >= threshold else "",
    )


def verify_quote(
    quote: str,
    source_text: str,
    *,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> QuoteMatch:
    """Check whether ``quote`` is grounded in ``source_text`` across three tiers."""
    quote = (quote or "").strip()
    if not quote:
        return QuoteMatch(False, "empty", 0.0)
    source_text = source_text or ""

    if quote in source_text:
        return QuoteMatch(True, "exact", 1.0)

    norm_quote = normalize(quote)
    norm_source = normalize(source_text)
    if norm_quote and norm_quote in norm_source:
        return QuoteMatch(True, "normalized", 1.0)

    return _fuzzy_best_window(norm_quote, norm_source, threshold=fuzzy_threshold)


def verify_citations(
    citations: list[dict[str, object]],
    source_by_url: dict[str, str],
    *,
    corpus: str = "",
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> CitationCheckSummary:
    """Verify a list of citations against per-URL source text.

    ``source_by_url`` maps a citation's ``source_url`` to that source's raw
    text. When a citation's URL is missing from the map (or empty), the check
    falls back to ``corpus`` (the concatenation of all fetched source text for
    the run) so a quote is still credited if it appears in any fetched source.
    """
    summary = CitationCheckSummary()
    for citation in citations:
        quote = str(citation.get("quote") or "")
        url = str(citation.get("source_url") or "")
        source_text = source_by_url.get(url) or corpus
        match = verify_quote(quote, source_text, fuzzy_threshold=fuzzy_threshold)
        summary.total += 1
        if match.match_type == "exact":
            summary.exact += 1
        elif match.match_type == "normalized":
            summary.normalized += 1
        elif match.match_type == "fuzzy":
            summary.fuzzy += 1
        else:
            summary.unmatched += 1
        summary.results.append({
            "source_url": url,
            "quote": quote[:200],
            "claim": str(citation.get("claim") or "")[:200],
            **match.to_dict(),
        })
    return summary
