from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Literal

from insightswarm.agents.browser_agent import BrowserWorker
from insightswarm.agents.critic import Critic
from insightswarm.agents.extractor import Extractor
from insightswarm.agents.lead import LeadWorker, bootstrap_lead_objective
from insightswarm.agents.researcher import Researcher
from insightswarm.agents.writer import WriterWorker
from insightswarm.db.store import Store
from insightswarm.delivery_gate import synchronize_delivery_gate
from insightswarm.extraction_batches import synchronize_extraction_batches, synchronize_run_evidence_review
from insightswarm.models.registry import ModelRegistry
from insightswarm.models.router import build_model_client
from insightswarm.multimodal_inputs import ingest_user_input_files
from insightswarm.swarm_store import ArtifactStore, BoardStore, Mailbox, TaskStore
from insightswarm.util import new_id


StopReason = Literal[
    "deliver_called",
    "budget_exhausted",
    "no_progress_budget_exhausted",
    "human_required",
]

DeliveryKind = Literal["report", "report_partial", "report_blocked"]


@dataclass(frozen=True)
class ObjectiveBudget:
    max_steps: int = 12
    max_no_progress_seconds: float = 120.0
    max_runtime_seconds: float = 1800.0
    max_drain_seconds: float = 900.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeliveryResult:
    run_id: str
    status: str
    result_type: DeliveryKind
    stop_reason: StopReason
    final_state: str
    question: str
    steps: list[dict[str, Any]]
    report: dict[str, Any] | None = None
    critic: dict[str, Any] | None = None
    must_fix: list[str] = field(default_factory=list)
    technical_failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeState:
    run_id: str
    run_root: Path
    question: str
    budget: ObjectiveBudget
    step_trace_path: Path | None = None
    task_store: TaskStore | None = None
    mailbox: Mailbox | None = None
    artifact_store: ArtifactStore | None = None
    board_store: BoardStore | None = None
    delivery_gate_status: str = "closed"
    delivery_gate_reasons: list[str] = field(default_factory=list)
    delivery_frontier_hash: str = ""
    stop_reason: StopReason | None = None
    last_progress_at: float = 0.0
    model_client: Any | None = None
    browser_model_client: Any | None = None
    model_registry: ModelRegistry | None = None


class BrowserCompositeModelClient:
    """Text-first browser client with optional vision support."""

    def __init__(self, *, text_client: Any, vision_client: Any | None = None):
        self.text_client = text_client
        self.vision_client = vision_client or text_client
        self.provider = getattr(text_client, "provider", "browser_text")
        self.model = getattr(text_client, "model", "browser_text")

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        return self.text_client.complete(*args, **kwargs)

    def analyze_image(self, *args: Any, **kwargs: Any) -> Any:
        return self.vision_client.analyze_image(*args, **kwargs)


def run_objective(
    store: Store,
    question: str,
    budget: ObjectiveBudget,
    run_root: Path,
    model_client: Any | None = None,
    run_id: str | None = None,
    browser_model_client: Any | None = None,
    model_registry: ModelRegistry | None = None,
) -> DeliveryResult:
    run_id = run_id or new_id("run")
    run_dir = run_root / ".tmp" / f"run-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RuntimeState(
        run_id=run_id,
        run_root=run_root,
        question=question,
        budget=budget,
        step_trace_path=run_dir / "steps.jsonl",
        task_store=TaskStore(store),
        mailbox=Mailbox(store),
        artifact_store=ArtifactStore(store),
        board_store=BoardStore(store),
        last_progress_at=time.monotonic(),
        model_client=model_client,
        browser_model_client=browser_model_client or _browser_model(model_registry),
        model_registry=model_registry,
    )
    state.task_store.recover_expired_leases(run_id)

    stop_event = threading.Event()
    worker_threads = _start_worker_threads(store, state, stop_event)
    try:
        _wait_until_stop(store, state, stop_event)
    finally:
        stop_event.set()
        for thread in worker_threads:
            thread.join(timeout=5.0)

    stop_reason = state.stop_reason or "no_progress_budget_exhausted"
    return _build_delivery_result(store, state, stop_reason)


def create_and_run_objective(
    store: Store,
    *,
    name: str,
    query: str,
    model_provider: str,
    model_config_path: str | Path | None = None,
    artifact_dir: Path,
    max_steps: int = 12,
    max_runtime_seconds: float = 1800.0,
    max_no_progress_seconds: float = 120.0,
    max_drain_seconds: float = 900.0,
    quality_mode: str = "production",
    search_provider: str = "tavily",
    browser_backend: str | None = "visible",
    browser_cdp_url: str | None = None,
    input_files: list[str] | None = None,
    **_: Any,
) -> DeliveryResult:
    if browser_backend:
        os.environ["INSIGHTSWARM_BROWSER_BACKEND"] = browser_backend
    if browser_cdp_url:
        os.environ["INSIGHTSWARM_BROWSER_CDP_URL"] = browser_cdp_url
    run_id = new_id("run")
    metadata = {
        "name": name,
        "query": query,
        "model_provider": model_provider,
        "quality_mode": quality_mode,
        "search_provider": search_provider,
        "browser_backend": browser_backend,
        "browser_cdp_url": browser_cdp_url,
        "user_input_count": 0,
    }
    store.create_swarm_run_state(
        run_id=run_id,
        objective=query,
        budget={"max_steps": max_steps, "metadata": metadata},
        phase="discovery",
    )
    model_registry = (
        ModelRegistry.from_file(model_config_path, store=store)
        if model_config_path
        else None
    )
    vision_model_client = model_registry.for_agent("vision", capability="vision") if model_registry is not None else None
    browser_model_client = _browser_model(model_registry)
    user_input_summaries = ingest_user_input_files(
        store,
        run_id,
        file_paths=input_files,
        vision_model_client=vision_model_client,
    )
    metadata["user_input_count"] = len(user_input_summaries)
    store.update_swarm_run_state(
        run_id,
        budget={"max_steps": max_steps, "metadata": metadata},
    )
    task_store = TaskStore(store)
    mailbox = Mailbox(store)
    bootstrap_lead_objective(
        task_store,
        mailbox,
        run_id=run_id,
        question=query,
        user_inputs=user_input_summaries,
    )
    model_client = None if model_registry else build_model_client(model_provider)
    if browser_model_client is None:
        browser_model_client = model_client
    return run_objective(
        store,
        query,
        ObjectiveBudget(
            max_steps=max_steps,
            max_runtime_seconds=max_runtime_seconds,
            max_no_progress_seconds=max_no_progress_seconds,
            max_drain_seconds=max_drain_seconds,
        ),
        artifact_dir.parent,
        model_client=model_client,
        run_id=run_id,
        browser_model_client=browser_model_client,
        model_registry=model_registry,
    )


def _start_worker_threads(store: Store, state: RuntimeState, stop_event: threading.Event) -> list[threading.Thread]:
    assert state.task_store is not None and state.mailbox is not None and state.artifact_store is not None and state.board_store is not None
    artifact_store = ArtifactStore(store)
    board_store = BoardStore(store)
    threads = [
        threading.Thread(
            target=lambda: LeadWorker(state.task_store, state.mailbox, board_store).run_forever(
                state.run_id,
                stop_event,
                poll_interval=0.05,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=lambda: BrowserWorker(state.task_store, state.mailbox, artifact_store, board_store).run_forever(
                state.run_id,
                stop_event,
                poll_interval=0.05,
                model_client=state.browser_model_client or _browser_model(state.model_registry),
                trace_path=state.step_trace_path,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=lambda: Extractor(state.task_store, state.mailbox, artifact_store, board_store).run_forever(
                state.run_id,
                stop_event,
                poll_interval=0.05,
                model_client=_agent_model(state, "extractor"),
                trace_path=state.step_trace_path,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=lambda: Researcher(state.task_store, state.mailbox, artifact_store, board_store).run_forever(
                state.run_id,
                stop_event,
                poll_interval=0.05,
                model_client=_agent_model(state, "researcher"),
                trace_path=state.step_trace_path,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=lambda: Critic(state.task_store, state.mailbox, artifact_store, board_store).run_forever(
                state.run_id,
                stop_event,
                poll_interval=0.05,
                model_client=_agent_model(state, "critic"),
                trace_path=state.step_trace_path,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=lambda: WriterWorker(state.task_store, state.mailbox, artifact_store, board_store).run_forever(
                state.run_id,
                stop_event,
                poll_interval=0.05,
                model_client=_agent_model(state, "writer"),
            ),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    return threads


def _agent_model(state: RuntimeState, role: str, *, capability: str = "text") -> Any | None:
    if state.model_registry is not None:
        return state.model_registry.for_agent(role, capability=capability)
    return state.model_client


def _browser_model(model_registry: ModelRegistry | None) -> Any | None:
    if model_registry is None:
        return None
    try:
        text_client = model_registry.for_agent("browser", capability="text")
    except ValueError:
        text_client = model_registry.for_agent("browser")
    try:
        vision_client = model_registry.for_agent("vision", capability="vision")
    except ValueError:
        vision_client = text_client
    return BrowserCompositeModelClient(
        text_client=text_client,
        vision_client=vision_client,
    )


def _wait_until_stop(store: Store, state: RuntimeState, stop_event: threading.Event) -> None:
    start = time.monotonic()
    drain_started_at: float | None = None
    last_counts = _progress_counts(store, state.run_id)
    while not stop_event.is_set():
        assert state.task_store is not None
        recovered = state.task_store.recover_expired_leases(state.run_id)
        if recovered:
            _write_runtime_trace(state, "lease_recovered", {"count": recovered})
        updated_batches = synchronize_extraction_batches(store, state.run_id)
        if updated_batches:
            _write_runtime_trace(state, "extraction_batches_ready", {"count": updated_batches})
        created_reviews = synchronize_run_evidence_review(store, state.run_id)
        if created_reviews:
            _write_runtime_trace(state, "run_reviews_created", {"count": created_reviews})
        decision = synchronize_delivery_gate(store, state.run_id)
        state.delivery_gate_status = decision.status
        state.delivery_gate_reasons = list(decision.reasons)
        state.delivery_frontier_hash = decision.frontier_hash

        current_counts = _progress_counts(store, state.run_id)
        if current_counts != last_counts:
            state.last_progress_at = time.monotonic()
            last_counts = current_counts

        report = _load_report_from_store(store, state.run_id, frontier_hash=state.delivery_frontier_hash)
        if report is not None:
            state.stop_reason = "deliver_called"
            _write_runtime_trace(state, "stop", {"stop_reason": state.stop_reason})
            stop_event.set()
            continue

        if decision.status == "blocked":
            state.stop_reason = "human_required"
            _write_runtime_trace(state, "stop", {"stop_reason": state.stop_reason, "delivery_gate_reasons": state.delivery_gate_reasons})
            stop_event.set()
            continue

        if time.monotonic() - start >= state.budget.max_runtime_seconds:
            if _has_active_work(store, state.run_id):
                if drain_started_at is None:
                    drain_started_at = time.monotonic()
                    _write_runtime_trace(
                        state,
                        "drain_started",
                        {"active_task_count": len(_active_tasks(store, state.run_id))},
                    )
                if time.monotonic() - drain_started_at < state.budget.max_drain_seconds:
                    time.sleep(0.05)
                    continue
            state.stop_reason = "budget_exhausted"
            _write_runtime_trace(state, "stop", {"stop_reason": state.stop_reason, "delivery_gate_reasons": state.delivery_gate_reasons})
            stop_event.set()
            continue

        if (
            not _has_active_work(store, state.run_id)
            and time.monotonic() - state.last_progress_at >= state.budget.max_no_progress_seconds
        ):
            state.stop_reason = "no_progress_budget_exhausted"
            _write_runtime_trace(state, "stop", {"stop_reason": state.stop_reason, "delivery_gate_reasons": state.delivery_gate_reasons})
            stop_event.set()
            continue

        time.sleep(0.05)


def _progress_counts(store: Store, run_id: str) -> tuple[int, int, int, int]:
    return (
        len(store.list_swarm_tasks(run_id)),
        len(store.list_swarm_artifacts(run_id)),
        len(store.list_swarm_evidence(run_id)),
        len(store.list_swarm_messages(run_id)),
    )


def _has_active_work(store: Store, run_id: str) -> bool:
    return bool(_active_tasks(store, run_id))


def _active_tasks(store: Store, run_id: str) -> list[Any]:
    return [
        task
        for task in store.list_swarm_tasks(run_id)
        if task.status in {"pending", "leased"}
    ]


def _write_runtime_trace(state: RuntimeState, event: str, payload: dict[str, Any]) -> None:
    if state.step_trace_path is None:
        return
    state.step_trace_path.parent.mkdir(parents=True, exist_ok=True)
    with state.step_trace_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"kind": "runtime_event", "event": event, "run_id": state.run_id, "payload": payload},
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


def _build_delivery_result(store: Store, state: RuntimeState, stop_reason: StopReason) -> DeliveryResult:
    report = _load_report_from_store(store, state.run_id, frontier_hash=state.delivery_frontier_hash)
    critic = _latest_critic_summary(store, state.run_id) or _default_critic_summary(store, state.run_id)
    result_type: DeliveryKind = "report"
    final_state = "completed"
    must_fix: list[str] = []

    if stop_reason == "human_required":
        result_type = "report_blocked"
        final_state = "blocked"
        report = None
        critic = {"verdict": "block"}
        must_fix = list(state.delivery_gate_reasons or ["Human authorization is required before browser execution can continue."])
    elif critic.get("verdict") == "repair":
        result_type = "report_partial"
        final_state = "partial"
        must_fix = list(critic.get("must_fix") or state.delivery_gate_reasons or ["Evidence collection ended before a clean delivery path was reached."])
    elif report is None:
        result_type = "report_partial"
        final_state = "partial"
        if critic.get("verdict") == "block":
            result_type = "report_blocked"
            final_state = "blocked"
        must_fix = list(
            state.delivery_gate_reasons
            or critic.get("must_fix")
            or ["Evidence collection ended before a clean delivery path was reached."]
        )

    return DeliveryResult(
        run_id=state.run_id,
        status="completed",
        result_type=result_type,
        stop_reason=stop_reason,
        final_state=final_state,
        question=state.question,
        steps=_assemble_result_steps(store, state, critic),
        report=report,
        critic=critic,
        must_fix=must_fix,
        technical_failures=_technical_failures(store, state.run_id),
    )


def _load_report_from_store(store: Store, run_id: str, *, frontier_hash: str = "") -> dict[str, Any] | None:
    artifacts = [
        artifact
        for artifact in store.list_swarm_artifacts(run_id)
        if artifact.type in {"report", "report_partial", "report_blocked"}
    ]
    if not artifacts:
        return None
    if frontier_hash:
        report_message_by_artifact = {
            str(message.payload.get("report_artifact_id") or ""): message
            for message in store.list_swarm_messages(run_id)
            if message.from_role == "writer"
            and str(message.payload.get("kind") or "") == "progress_update"
            and str(message.payload.get("frontier_hash") or "") == frontier_hash
        }
        artifacts = [artifact for artifact in artifacts if artifact.artifact_id in report_message_by_artifact]
        if not artifacts:
            return None
    artifact = artifacts[-1]
    path = Path(artifact.payload_ref)
    if not path.exists():
        return None
    return {"body": path.read_text(encoding="utf-8"), "path": str(path)}


def _latest_critic_summary(store: Store, run_id: str) -> dict[str, Any] | None:
    messages = [
        message
        for message in store.list_swarm_messages(run_id)
        if message.from_role == "critic" and "verdict" in message.payload
    ]
    if not messages:
        return None
    return dict(messages[-1].payload)


def _default_critic_summary(store: Store, run_id: str) -> dict[str, Any]:
    if store.list_swarm_evidence(run_id, qa_state="ready"):
        return {"verdict": "pass"}
    return {
        "verdict": "repair",
        "must_fix": ["citation-backed evidence is missing"],
        "targeted_query": store.get_swarm_run_state(run_id).objective,
    }


def _technical_failures(store: Store, run_id: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for message in store.list_swarm_messages(run_id):
        if message.type != "observation":
            continue
        payload = dict(message.payload or {})
        if payload.get("kind") == "technical_failure":
            failures.append(
                {
                    "role": payload.get("role") or message.from_role,
                    "task_id": payload.get("task_id") or message.related_task_id,
                    "task_kind": payload.get("task_kind"),
                    "status": payload.get("status"),
                    "reason": payload.get("reason"),
                    "retryable": payload.get("retryable"),
                }
            )
        elif payload.get("kind") == "extraction_failure" and payload.get("failure_category") == "technical":
            failures.append(
                {
                    "role": message.from_role,
                    "task_id": message.related_task_id,
                    "task_kind": None,
                    "status": "technical_failure",
                    "reason": payload.get("reason"),
                    "retryable": payload.get("retryable"),
                }
            )
    return failures[-20:]


def _assemble_result_steps(store: Store, state: RuntimeState, critic: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    if critic.get("verdict"):
        steps.append(
            {
                "tool_call": {"action": "critic", "arguments": {}},
                "result": {"status": "ok", "progressed": True, "verdict": critic.get("verdict")},
            }
        )
    steps.append(
        {
            "tool_call": {"action": "phase_controller", "arguments": {}},
            "result": {
                "delivery_gate_status": state.delivery_gate_status,
                "delivery_gate_reasons": list(state.delivery_gate_reasons),
                "evidence_count": len(store.list_swarm_evidence(state.run_id)),
                "artifact_count": len(store.list_swarm_artifacts(state.run_id)),
            },
        }
    )
    return steps


