"""Golden eval cases.

A case is a question plus a grading rubric: points the answer must cover, points
it must NOT claim (hallucination traps), a difficulty tier used by routing
evals, and optional input files for multimodal cases. Cases are JSON files
under ``evals/cases/`` (JSON keeps the harness dependency-free; no YAML parser).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CASES_DIR = Path("evals") / "cases"


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    suite: str = "golden"
    difficulty: str = "heavy"          # "light" | "heavy" — feeds routing evals
    must_cover: list[str] = field(default_factory=list)
    must_not_claim: list[str] = field(default_factory=list)
    input_files: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "suite": self.suite,
            "difficulty": self.difficulty,
            "must_cover": list(self.must_cover),
            "must_not_claim": list(self.must_not_claim),
            "input_files": list(self.input_files),
            "notes": self.notes,
        }


class CaseValidationError(ValueError):
    pass


def _coerce_case(data: dict[str, Any], *, source: str) -> EvalCase:
    case_id = str(data.get("case_id") or "").strip()
    question = str(data.get("question") or "").strip()
    if not case_id:
        raise CaseValidationError(f"{source}: missing case_id")
    if not question:
        raise CaseValidationError(f"{source}: case '{case_id}' missing question")
    difficulty = str(data.get("difficulty") or "heavy").strip().lower()
    if difficulty not in {"light", "heavy"}:
        raise CaseValidationError(
            f"{source}: case '{case_id}' difficulty must be 'light' or 'heavy', got '{difficulty}'"
        )
    return EvalCase(
        case_id=case_id,
        question=question,
        suite=str(data.get("suite") or "golden").strip(),
        difficulty=difficulty,
        must_cover=[str(x).strip() for x in (data.get("must_cover") or []) if str(x).strip()],
        must_not_claim=[str(x).strip() for x in (data.get("must_not_claim") or []) if str(x).strip()],
        input_files=[str(x).strip() for x in (data.get("input_files") or []) if str(x).strip()],
        notes=str(data.get("notes") or ""),
    )


def load_case_file(path: str | Path) -> EvalCase:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CaseValidationError(f"{path}: case file must be a JSON object")
    return _coerce_case(data, source=str(path))


def load_cases(
    cases_dir: str | Path = DEFAULT_CASES_DIR,
    *,
    suite: str | None = None,
    difficulty: str | None = None,
) -> list[EvalCase]:
    """Load and validate all case files, optionally filtered by suite/difficulty."""
    cases_dir = Path(cases_dir)
    if not cases_dir.exists():
        raise CaseValidationError(f"cases directory not found: {cases_dir}")
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for path in sorted(cases_dir.glob("*.json")):
        case = load_case_file(path)
        if case.case_id in seen:
            raise CaseValidationError(f"duplicate case_id '{case.case_id}' in {path}")
        seen.add(case.case_id)
        if suite and case.suite != suite:
            continue
        if difficulty and case.difficulty != difficulty:
            continue
        cases.append(case)
    return cases
