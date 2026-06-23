from __future__ import annotations

from insightswarm.agents.researcher_tools import _text_preview


def test_text_preview_returns_full_text_when_short() -> None:
    assert _text_preview("hello world") == "hello world"


def test_text_preview_samples_middle_for_long_text() -> None:
    text = "HEAD." + ("x" * 5000) + "MIDDLE_MARKER." + ("y" * 5000) + ".TAIL"
    preview = _text_preview(text, size=600)
    assert "HEAD." in preview
    assert "MIDDLE_MARKER." in preview
    assert "[middle sample]" in preview
    # tail boilerplate should be dropped
    assert ".TAIL" not in preview
    assert len(preview) <= 600 + len("\n…[middle sample]…\n")


def test_text_preview_default_size_larger_than_old_900() -> None:
    text = "A" * 5000
    preview = _text_preview(text)
    assert len(preview) > 900
    assert len(preview) <= 1800 + len("\n…[middle sample]…\n")
