from __future__ import annotations

import re
from dataclasses import dataclass, field


CITATION_MARKER_RE = re.compile(r"\[\[(?:doc|img|inf):[^\]\s]+\]\]")


@dataclass(frozen=True)
class ReportValidationResult:
    passed: bool
    missing_citation_markers: list[str] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)
    unsupported_bullets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "missing_citation_markers": self.missing_citation_markers,
            "missing_sections": self.missing_sections,
            "unsupported_bullets": self.unsupported_bullets,
        }


def extract_citation_markers(text: str) -> list[str]:
    return sorted(set(CITATION_MARKER_RE.findall(text or "")))


def validate_writer_report(candidate: str, base_report: str) -> ReportValidationResult:
    base_markers = set(extract_citation_markers(base_report))
    candidate_markers = set(extract_citation_markers(candidate))
    missing_markers = sorted(base_markers - candidate_markers)
    missing_sections: list[str] = []
    if "## Source Health" in base_report and "## Source Health" not in candidate:
        missing_sections.append("## Source Health")
    if "## Degraded Output Warnings" in base_report and "## Degraded Output Warnings" not in candidate:
        missing_sections.append("## Degraded Output Warnings")
    unsupported_bullets = _unsupported_strategy_bullets(candidate)
    return ReportValidationResult(
        passed=not missing_markers and not missing_sections and not unsupported_bullets,
        missing_citation_markers=missing_markers,
        missing_sections=missing_sections,
        unsupported_bullets=unsupported_bullets,
    )


def ensure_quality_section(report: str, quality_section: str) -> str:
    if "## Source Health" in report:
        return report
    return f"{report.rstrip()}\n\n{quality_section}"


def _unsupported_strategy_bullets(text: str) -> list[str]:
    unsupported: list[str] = []
    in_strategic_section = False
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_strategic_section = stripped == "## Strategic Read"
            continue
        if in_strategic_section and stripped.startswith("- ") and "[[" not in stripped:
            unsupported.append(stripped)
    return unsupported
