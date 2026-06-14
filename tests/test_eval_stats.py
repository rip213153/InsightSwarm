from __future__ import annotations

from insightswarm.eval.stats import compare_case, summarize


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
