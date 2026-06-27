"""Architecture invariant tests.

These tests enforce the centralized shared-state orchestration model:
- Workers collaborate only through the shared stores (TaskStore, Mailbox,
  Board, Evidence, DeliveryGate), never by directly calling another worker's
  execution path.
- The runtime must not bypass the task queue by invoking a worker's internal
  execution function directly (no ``runtime.invoke_writer()`` style calls).

If a future edit introduces peer-to-peer coupling or central direct-invocation,
these tests fail to flag the architectural drift early.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "insightswarm" / "agents"

# The six worker roles. Collaboration between them must flow through shared
# stores, not direct imports of each other's execution path.
WORKER_MODULES = {
    "lead.py",
    "researcher.py",
    "critic.py",
    "extractor.py",
    "writer.py",
    "browser_agent.py",
}

# Modules that workers MAY import (infrastructure + their own tools + shared
# stores). Anything else from insightswarm.agents is suspect.
ALLOWED_AGENT_IMPORTS = {
    "agent_loop",
    "agent_loop_contract",
    "tool_executor",
    "execution_cell",
    "failure_policy",
    "trace",
    # Each worker may import its own *_tools module.
    # Checked dynamically below.
}

# Pattern: from insightswarm.agents.<module> import ...
# or:     import insightswarm.agents.<module>
_AGENT_IMPORT_RE = re.compile(
    r"^\s*from\s+insightswarm\.agents\.(\w+)\s+import\s",
    re.MULTILINE,
)
_DIRECT_AGENT_IMPORT_RE = re.compile(
    r"^\s*import\s+insightswarm\.agents\.(\w+)\s*$",
    re.MULTILINE,
)


def _own_tools_module(worker_filename: str) -> str:
    """Return the tools module name owned by a worker (e.g. researcher -> researcher_tools)."""
    stem = Path(worker_filename).stem
    return f"{stem}_tools"


def test_workers_do_not_import_peer_workers() -> None:
    """Workers must not import another worker's execution module.

    Collaboration flows through shared stores (TaskStore/Mailbox/Board), not
    direct calls. A worker importing another worker (e.g. researcher importing
    critic) is peer-to-peer coupling that breaks the centralized orchestration
    model.
    """
    violations: list[str] = []
    for worker_file in sorted(AGENTS_DIR.glob("*.py")):
        if worker_file.name not in WORKER_MODULES:
            continue
        own_tools = _own_tools_module(worker_file.name)
        source = worker_file.read_text(encoding="utf-8")
        # Find all insightswarm.agents.<x> imports.
        imported_modules: set[str] = set()
        imported_modules.update(_AGENT_IMPORT_RE.findall(source))
        imported_modules.update(_DIRECT_AGENT_IMPORT_RE.findall(source))
        for mod in imported_modules:
            # A worker may import infrastructure, its own tools module, or its
            # own module (self-import is unusual but not coupling).
            if mod in ALLOWED_AGENT_IMPORTS:
                continue
            if mod == own_tools:
                continue
            if mod + "_tools" == own_tools:
                continue
            # Flag imports of other worker modules.
            if (AGENTS_DIR / f"{mod}.py").name in WORKER_MODULES and mod != Path(worker_file.name).stem:
                violations.append(
                    f"{worker_file.name} imports insightswarm.agents.{mod} — "
                    f"peer worker import violates centralized orchestration"
                )
    assert not violations, "peer-to-peer worker imports detected:\n  " + "\n  ".join(violations)


def test_runtime_does_not_directly_invoke_workers() -> None:
    """objective_runtime.py must not define invoke_<worker> methods.

    The runtime creates tasks and lets workers claim them via the shared
    queue. Direct invocation (runtime.invoke_writer(), runtime.run_critic())
    bypasses the task queue and breaks the centralized shared-state model.
    """
    runtime_file = REPO_ROOT / "insightswarm" / "objective_runtime.py"
    if not runtime_file.exists():
        return  # runtime module renamed/moved; skip
    source = runtime_file.read_text(encoding="utf-8")
    # Match def invoke_<role> or self.<role>_run_direct / similar bypass names.
    forbidden_patterns = [
        r"\bdef\s+invoke_(?:writer|researcher|critic|extractor|lead|browser)\b",
        r"\bdef\s+run_(?:writer|researcher|critic|extractor)_direct\b",
        r"\bdef\s+execute_(?:writer|researcher|critic|extractor)_now\b",
    ]
    violations: list[str] = []
    for pattern in forbidden_patterns:
        matches = re.findall(pattern, source)
        for match in matches:
            violations.append(
                f"objective_runtime.py defines '{match}' — direct worker invocation "
                f"violates centralized shared-state orchestration"
            )
    assert not violations, "direct worker invocation detected:\n  " + "\n  ".join(violations)
