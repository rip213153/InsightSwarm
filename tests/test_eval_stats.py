from __future__ import annotations

from insightswarm.eval.stats import compare_case, summarize, summarize_split


def test_summarize_empty():
    s = summarize([])
    assert s.n == 0 and s.mean == 0.0 and s.stderr == 0.0


def test_summarize_single_has_zero_dispersion():
    s = summarize([80.0])
    assert s.n == 1
    assert s.mean == 80.0
    assert s.std == 0.0
    assert s.stderr == 0.0


def test_summarize_mean_and_stderr():
    s = summarize([70.0, 80.0, 90.0])
    assert s.n == 3
    assert abs(s.mean - 80.0) < 1e-9
    assert abs(s.std - 10.0) < 1e-9          # sample std of 70,80,90
    assert abs(s.stderr - (10.0 / 3 ** 0.5)) < 1e-9
    assert s.min_score == 70.0 and s.max_score == 90.0


def test_compare_small_delta_is_noise():
    a = summarize([78.0, 80.0, 82.0])   # mean 80
    b = summarize([81.0, 83.0, 85.0])   # mean 83, wide-ish stderr
    cmp = compare_case("c1", a, b)
    # delta 3 is within the 2*combined-stderr band -> noise.
    assert cmp.verdict == "noise"
    assert abs(cmp.delta - 3.0) < 1e-9


def test_compare_large_delta_is_directional():
    a = summarize([50.0, 51.0, 49.0])   # tight around 50
    b = summarize([85.0, 86.0, 84.0])   # tight around 85
    cmp = compare_case("c1", a, b)
    assert cmp.verdict == "improved"
    cmp_rev = compare_case("c1", b, a)
    assert cmp_rev.verdict == "regressed"


def test_compare_new_and_dropped():
    b = summarize([80.0])
    assert compare_case("c1", None, b).verdict == "new"
    a = summarize([80.0])
    assert compare_case("c1", a, None).verdict == "dropped"


def test_summarize_split_keeps_llm_and_fallback_separate():
    """Fallback scores must NOT bleed into the LLM-judged mean.

    This is the regression test for the "fallback pollutes the mean" bug: a
    fallback score of 0.0 must not drag the LLM mean down, and the fallback
    group must be reported on its own.
    """
    split = summarize_split(
        scores_llm=[0.8, 0.9, 0.7],
        scores_fallback=[0.0, 0.0],
        n_no_report=1,
    )
    assert split.llm.n == 3
    assert abs(split.llm.mean - 0.8) < 1e-9
    assert split.fallback.n == 2
    assert split.fallback.mean == 0.0
    assert split.n_no_report == 1
    assert split.n_total == 6


def test_summarize_split_handles_no_llm_epochs():
    """If every epoch fell back, the LLM group is empty and the fallback group
    carries the signal instead of poisoning an empty main mean."""
    split = summarize_split(scores_llm=[], scores_fallback=[0.3, 0.4])
    assert split.llm.n == 0
    assert split.llm.mean == 0.0
    assert split.fallback.n == 2
    assert abs(split.fallback.mean - 0.35) < 1e-9
    assert split.n_total == 2
