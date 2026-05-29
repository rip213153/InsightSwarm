from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event

from insightswarm.schemas.swarm import Task
from insightswarm.swarm_store import BoardStore, Mailbox, TaskStore


DEFAULT_MAX_REPAIR_ATTEMPTS = 2


@dataclass(frozen=True)
class LeadWorkResult:
    claimed_task_id: str
    created_task_ids: list[str] = field(default_factory=list)
    created_message_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LeadLoopResult:
    iterations: int
    claimed_task_ids: list[str] = field(default_factory=list)
    created_task_ids: list[str] = field(default_factory=list)
    created_message_ids: list[str] = field(default_factory=list)


class LeadWorker:
    def __init__(self, task_store: TaskStore, mailbox: Mailbox, board_store: BoardStore | None = None):
        self.task_store = task_store
        self.mailbox = mailbox
        self.board_store = board_store or BoardStore(task_store.store)

    def run_once(self, run_id: str) -> LeadWorkResult | None:
        self._promote_repair_messages(run_id)
        self._sync_board_tasks(run_id)
        task = self.task_store.claim_next(run_id, owner_role="lead")
        if task is None:
            return None
        result = self._process_task(task)
        self.task_store.complete(task.task_id)
        return result

    def run_until_idle(self, run_id: str, *, max_iterations: int = 32) -> LeadLoopResult:
        iterations = 0
        claimed_task_ids: list[str] = []
        created_task_ids: list[str] = []
        created_message_ids: list[str] = []

        while iterations < max_iterations:
            result = self.run_once(run_id)
            if result is None:
                break
            iterations += 1
            claimed_task_ids.append(result.claimed_task_id)
            created_task_ids.extend(result.created_task_ids)
            created_message_ids.extend(result.created_message_ids)

        return LeadLoopResult(
            iterations=iterations,
            claimed_task_ids=claimed_task_ids,
            created_task_ids=created_task_ids,
            created_message_ids=created_message_ids,
        )

    def run_forever(
        self,
        run_id: str,
        stop_event: Event,
        *,
        poll_interval: float = 0.2,
        max_iterations: int | None = None,
    ) -> LeadLoopResult:
        iterations = 0
        claimed_task_ids: list[str] = []
        created_task_ids: list[str] = []
        created_message_ids: list[str] = []

        while not stop_event.is_set():
            result = self.run_once(run_id)
            if result is None:
                stop_event.wait(poll_interval)
                continue
            iterations += 1
            claimed_task_ids.append(result.claimed_task_id)
            created_task_ids.extend(result.created_task_ids)
            created_message_ids.extend(result.created_message_ids)
            if max_iterations is not None and iterations >= max_iterations:
                break

        return LeadLoopResult(
            iterations=iterations,
            claimed_task_ids=claimed_task_ids,
            created_task_ids=created_task_ids,
            created_message_ids=created_message_ids,
        )

    def _process_task(self, task: Task) -> LeadWorkResult:
        context = self._assemble_context(task)
        if task.kind == "research_question":
            return self._handle_research_question(task, context)
        if task.kind == "repair_request":
            return self._handle_repair_request(task, context)
        raise ValueError(f"unsupported lead task kind: {task.kind}")

    def _assemble_context(self, task: Task) -> dict:
        return {
            "task": task,
            "messages": [
                message
                for message in self.mailbox.inbox(task.run_id, role="lead")
                if message.related_task_id == task.task_id
            ],
        }

    def _handle_research_question(self, task: Task, context: dict) -> LeadWorkResult:
        del context
        created_tasks: list[str] = []
        created_messages: list[str] = []
        question = str(task.inputs.get("question") or "").strip()
        sub_questions = [
            str(item).strip()
            for item in (task.inputs.get("sub_questions") or [])
            if str(item).strip()
        ]
        browser_goal = str(task.inputs.get("browser_goal") or "").strip()

        if not sub_questions and question:
            sub_questions = [question]

        root_question = self.board_store.create_question(
            task.run_id,
            title=question,
            question_type="objective",
            status="active",
            owner_role="lead",
            priority=task.priority,
            created_by="lead",
            dedupe_key=f"objective:{question.lower()}",
        )

        for sub_question in sub_questions[:3]:
            self.board_store.create_question(
                task.run_id,
                title=sub_question,
                question_type="subquestion",
                status="open",
                parent_id=root_question.item_id,
                owner_role="researcher",
                priority=max(task.priority - 1, 0),
                created_by="lead",
            )

        if browser_goal:
            self.board_store.create_question(
                task.run_id,
                title=browser_goal,
                question_type="source_request",
                status="open",
                parent_id=root_question.item_id,
                owner_role="browser_agent",
                priority=task.priority,
                created_by="lead",
                payload={"goal": browser_goal, "reason": "initial browser acquisition request"},
            )

        self.board_store.write_plan(
            task.run_id,
            title="Initial research strategy",
            plan_kind="strategy",
            parent_id=root_question.item_id,
            priority=task.priority,
            created_by="lead",
            dedupe_key=f"plan:strategy:{root_question.item_id}",
            payload={
                "current_focus": question,
                "next_opportunities": sub_questions[:3],
                "browser_goal": browser_goal or None,
            },
        )

        downstream = self._sync_board_tasks(task.run_id)
        created_tasks.extend(downstream["created_task_ids"])
        created_messages.extend(downstream["created_message_ids"])

        status = self.mailbox.send(
            task.run_id,
            from_role="lead",
            broadcast=True,
            message_type="observation",
            payload={
                "kind": "progress_update",
                "source_task_id": task.task_id,
                "created_task_count": len(created_tasks),
                "created_message_count": len(created_messages),
            },
            related_task_id=task.task_id,
        )
        created_messages.append(status.message_id)
        return LeadWorkResult(
            claimed_task_id=task.task_id,
            created_task_ids=created_tasks,
            created_message_ids=created_messages,
        )

    def _handle_repair_request(self, task: Task, context: dict) -> LeadWorkResult:
        del context
        created_tasks: list[str] = []
        created_messages: list[str] = []
        targeted_query = str(task.inputs.get("targeted_query") or task.inputs.get("question") or "").strip()
        escalation_role = str(task.inputs.get("owner_role") or "researcher").strip() or "researcher"
        issue_key = str(task.inputs.get("issue_key") or targeted_query or "repair_issue").strip()
        repair_attempt = int(task.inputs.get("repair_attempt") or 1)
        max_repair_attempts = int(task.inputs.get("max_repair_attempts") or DEFAULT_MAX_REPAIR_ATTEMPTS)
        if repair_attempt > max_repair_attempts:
            delivery_gap = self.mailbox.send(
                task.run_id,
                from_role="lead",
                broadcast=True,
                message_type="observation",
                payload={
                    "kind": "delivery_gap",
                    "issue_key": issue_key,
                    "targeted_query": targeted_query,
                    "repair_attempt": repair_attempt,
                    "max_repair_attempts": max_repair_attempts,
                },
                related_task_id=task.task_id,
            )
            created_messages.append(delivery_gap.message_id)
            return LeadWorkResult(
                claimed_task_id=task.task_id,
                created_task_ids=[],
                created_message_ids=created_messages,
            )
        self.board_store.create_question(
            task.run_id,
            title=targeted_query,
            question_type="repair",
            status="open",
            owner_role=escalation_role,
            priority=task.priority,
            created_by="lead",
            payload={
                "issue_key": issue_key,
                "repair_attempt": repair_attempt,
                "max_repair_attempts": max_repair_attempts,
                "repair_context": dict(task.inputs),
            },
            dedupe_key=f"repair:{issue_key}:{repair_attempt}",
        )
        downstream = self._sync_board_tasks(task.run_id)
        created_tasks.extend(downstream["created_task_ids"])
        created_messages.extend(downstream["created_message_ids"])
        return LeadWorkResult(
            claimed_task_id=task.task_id,
            created_task_ids=created_tasks,
            created_message_ids=created_messages,
        )

    def _promote_repair_messages(self, run_id: str) -> None:
        for message in self.mailbox.inbox(run_id, role="lead"):
            if message.type not in {"request", "observation"}:
                continue
            if str(message.payload.get("kind") or "") not in {"research_repair", "conflict"}:
                continue
            issue_key = str(message.payload.get("issue_key") or message.payload.get("targeted_query") or message.message_id)
            if self._has_active_follow_up(run_id, issue_key):
                continue
            task_kind = "repair_request"
            self.task_store.create(
                run_id,
                kind=task_kind,
                status="pending",
                owner_role="lead",
                inputs={
                    "targeted_query": message.payload.get("targeted_query") or message.payload.get("question") or "",
                    "owner_role": message.payload.get("owner_role") or "researcher",
                    "issue_key": issue_key,
                    "repair_attempt": int(message.payload.get("repair_attempt") or 1),
                    "max_repair_attempts": int(message.payload.get("max_repair_attempts") or DEFAULT_MAX_REPAIR_ATTEMPTS),
                },
                priority=10,
                created_by=message.from_role,
            )

    def _has_active_follow_up(self, run_id: str, issue_key: str) -> bool:
        for task in self.task_store.store.list_swarm_tasks(run_id):
            if task.status not in {"pending", "leased"}:
                continue
            if task.owner_role == "lead" and task.kind == "repair_request" and str(task.inputs.get("issue_key") or "") == issue_key:
                return True
            if task.owner_role not in {"researcher", "browser_agent"}:
                continue
            if str(task.inputs.get("issue_key") or "") == issue_key:
                return True
        return False

    def _sync_board_tasks(self, run_id: str) -> dict[str, list[str]]:
        created_task_ids: list[str] = []
        created_message_ids: list[str] = []
        for item in self.board_store.store.list_swarm_board_items(run_id, kind="question"):
            if item.status not in {"open", "active"}:
                continue
            owner_role = str(item.payload.get("owner_role") or "researcher")
            if owner_role not in {"researcher", "browser_agent"}:
                continue
            if self._has_active_board_task(run_id, item.item_id or "", owner_role):
                continue

            question_type = str(item.payload.get("question_type") or "")
            if owner_role == "browser_agent":
                kind = "hard_acquisition"
                inputs = {
                    "goal": str(item.payload.get("goal") or item.title),
                    "board_item_id": item.item_id,
                    "question_type": question_type,
                    "issue_key": item.payload.get("issue_key"),
                }
                message_kind = "hard_acquisition"
                message_value_key = "goal"
                message_value = str(inputs["goal"])
            else:
                kind = "research_repair" if question_type == "repair" else "research_subquestion"
                inputs = {
                    "question": item.title,
                    "board_item_id": item.item_id,
                    "question_type": question_type or "subquestion",
                    "issue_key": item.payload.get("issue_key"),
                    "repair_context": item.payload.get("repair_context"),
                }
                message_kind = kind
                message_value_key = "question"
                message_value = item.title

            downstream_task = self.task_store.create(
                run_id,
                kind=kind,
                status="pending",
                owner_role=owner_role,
                inputs=inputs,
                priority=item.priority,
                created_by="lead",
            )
            handoff = self.mailbox.send(
                run_id,
                from_role="lead",
                to_role=owner_role,
                message_type="request",
                payload={
                    "kind": message_kind,
                    "task_id": downstream_task.task_id,
                    message_value_key: message_value,
                    "board_item_id": item.item_id,
                    "issue_key": item.payload.get("issue_key"),
                },
                related_task_id=downstream_task.task_id,
            )
            self.board_store.update_status(item.item_id or "", status="active")
            created_task_ids.append(downstream_task.task_id)
            created_message_ids.append(handoff.message_id)
        return {
            "created_task_ids": created_task_ids,
            "created_message_ids": created_message_ids,
        }

    def _has_active_board_task(self, run_id: str, board_item_id: str, owner_role: str) -> bool:
        for task in self.task_store.store.list_swarm_tasks(run_id, owner_role=owner_role):
            if str(task.inputs.get("board_item_id") or "") == board_item_id:
                return True
        return False


def bootstrap_lead_objective(
    task_store: TaskStore,
    mailbox: Mailbox,
    *,
    run_id: str,
    question: str,
    sub_questions: list[str] | None = None,
    browser_goal: str | None = None,
) -> Task:
    root_task = task_store.create(
        run_id,
        kind="research_question",
        status="pending",
        owner_role="lead",
        inputs={
            "question": question,
            "sub_questions": list(sub_questions or []),
            "browser_goal": browser_goal,
        },
        priority=10,
        created_by="run_bootstrap",
    )
    mailbox.send(
        run_id,
        from_role="system",
        to_role="lead",
        message_type="request",
        payload={"kind": "research_subquestion", "task_id": root_task.task_id, "question": question},
        related_task_id=root_task.task_id,
    )
    return root_task
