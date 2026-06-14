from __future__ import annotations

from pathlib import Path

import pytest

from insightswarm.eval.cases import (
    CaseValidationError,
    load_case_file,
    load_cases,
)

REPO_CASES = Path("evals") / "cases"


def test_golden_cases_load_and_validate():
    cases = load_cases(REPO_CASES)
    assert len(cases) >= 4
    ids = {c.case_id for c in cases}
    assert "jd-next-strategy" in ids
    assert "fuel-surcharge-2026" in ids
    for case in cases:
        assert case.question
        assert case.difficulty in {"light", "heavy"}


def test_filter_by_difficulty():
    light = load_cases(REPO_CASES, difficulty="light")
    assert all(c.difficulty == "light" for c in light)
    assert any(c.case_id == "cartier-site-image" for c in light)


def test_missing_question_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"case_id": "x"}', encoding="utf-8")
    with pytest.raises(CaseValidationError):
        load_case_file(bad)


def test_bad_difficulty_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"case_id": "x", "question": "q", "difficulty": "medium"}', encoding="utf-8")
    with pytest.raises(CaseValidationError):
        load_case_file(bad)


def test_duplicate_case_id_rejected(tmp_path):
    (tmp_path / "a.json").write_text('{"case_id": "dup", "question": "q1"}', encoding="utf-8")
    (tmp_path / "b.json").write_text('{"case_id": "dup", "question": "q2"}', encoding="utf-8")
    with pytest.raises(CaseValidationError):
        load_cases(tmp_path)
