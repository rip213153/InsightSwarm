from __future__ import annotations

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_decide_next_tool_call_removed_from_insightswarm_sources() -> None:
    hits: list[str] = []
    for path in (REPO_ROOT / "insightswarm").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "decide_next_tool_call" in text:
            hits.append(str(path))

    assert hits == []


def test_objective_runtime_does_not_directly_call_browser_or_extractor() -> None:
    runtime_path = REPO_ROOT / "insightswarm" / "objective_runtime.py"
    text = runtime_path.read_text(encoding="utf-8")

    assert "_decide_runtime_tool_call" not in text
    assert "_dispatch_tool_call" not in text
    assert "RuntimeAction" not in text
    assert "class ToolCall" not in text
    assert "execute_browser_goal" not in text
    assert "extract_citations" not in text
    assert "write_report(" not in text
    assert "from insightswarm.agents.writer import write_report" not in text
    assert "LeadWorker(task_store, mailbox).run_once(" not in text
    assert "LeadWorker(task_store, mailbox).run_until_idle(" not in text
    assert re.search(r"BrowserWorker\([^\)]*\)\.run_once\(", text) is None
    assert re.search(r"ExtractorWorker\([^\)]*\)\.run_once\(", text) is None
    assert "run_researchers(" not in text
    assert "review_evidence(" not in text
    assert re.search(r"WriterWorker\([^\)]*\)\.run_once\(", text) is None
