"""
Enhanced SubResearcher - 放宽约束的原型实现

这个模块提供了增强版的 SubResearcher，支持：
1. 扩展的 Action 空间
2. 动态子问题生成
3. 异步协作协议

使用方式：
    from enhanced_sub_researcher import EnhancedSubResearcherWorker
    
    worker = EnhancedSubResearcherWorker(
        task_store=task_store,
        mailbox=async_mailbox,
        artifact_store=artifact_store,
        board_store=board_store,
        model_client=model_client
    )
    
    result = await worker.run_once_enhanced(run_id)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
from typing import Any, Callable, Awaitable
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class Priority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QuestionStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    MERGED = "merged"
    BLOCKED = "blocked"


class DeliveryMode(Enum):
    IMMEDIATE = "immediate"
    QUEUED = "queued"
    CONDITIONAL = "conditional"


class ActionMode(Enum):
    TRADITIONAL = "traditional"
    COLLABORATIVE = "collaborative"
    ADAPTIVE = "adaptive"


# 扩展的 Action 空间
EXTENDED_ACTIONS = {
    # 原有 actions
    "query_task_context",
    "query_mailbox",
    "query_evidence",
    "search",
    "fetch",
    "publish_raw_document",
    "suggest_browser",
    "complete",
    "blocked",
    
    # 新增：研究策略类
    "spawn_sub_question",
    "merge_questions",
    "split_question",
    "replan",
    
    # 新增：协作类
    "delegate_to_peer",
    "request_help",
    "propose_collaboration",
    "share_knowledge",
    
    # 新增：执行类
    "fetch_multiple",
    "analyze_and_search",
    "parallel_investigate",
    
    # 新增：元认知类
    "reflect",
    "request_feedback",
    "adjust_priority",
}


@dataclass
class SubQuestion:
    """动态生成的子问题"""
    id: str
    question: str
    priority: Priority
    reason: str
    source: str
    estimated_effort: str = "medium"
    dependencies: list[str] = field(default_factory=list)
    status: QuestionStatus = QuestionStatus.PENDING
    assigned_to: str | None = None
    created_by: str = "system"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    can_be_parallel: bool = True
    merging_candidates: list[str] = field(default_factory=list)
    related_questions: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "priority": self.priority.value,
            "reason": self.reason,
            "source": self.source,
            "estimated_effort": self.estimated_effort,
            "status": self.status.value,
            "can_be_parallel": self.can_be_parallel
        }


@dataclass
class AsyncMessage:
    """异步消息"""
    id: str
    type: str
    from_role: str
    to_role: str | None
    broadcast: bool
    payload: dict[str, Any]
    related_task_id: str | None = None
    delivery_mode: DeliveryMode = DeliveryMode.IMMEDIATE
    delivery_condition: dict | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Proposal:
    """协作提案"""
    id: str
    proposer: str
    content: dict
    proposal_type: str
    deadline: datetime | None = None
    votes: dict[str, str] = field(default_factory=dict)  # voter_id -> vote
    status: str = "pending"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "proposer": self.proposer,
            "content": self.content,
            "type": self.proposal_type,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "votes": self.votes,
            "status": self.status
        }


@dataclass
class Delegation:
    """任务委托"""
    id: str
    from_agent: str
    to_role: str
    task: str
    constraints: dict[str, Any]
    status: str = "pending"
    acknowledged: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class VoteResult:
    """投票结果"""
    proposal_id: str
    votes_for: int
    votes_against: int
    votes_abstain: int
    approved: bool
    details: dict[str, str]


class AsyncMailbox:
    """
    异步消息队列
    支持消息订阅、条件投递、队列处理
    """
    
    def __init__(self):
        self._queue: list[AsyncMessage] = []
        self._subscriptions: dict[str, list[Callable]] = {}
        self._proposals: dict[str, Proposal] = {}
        self._delegations: dict[str, Delegation] = {}
        self._messages_history: list[AsyncMessage] = []
    
    async def send_async(
        self,
        message: AsyncMessage,
        delivery: DeliveryMode = DeliveryMode.IMMEDIATE
    ) -> str:
        """异步发送消息"""
        message.delivery_mode = delivery
        self._messages_history.append(message)
        
        if delivery == DeliveryMode.QUEUED:
            self._queue.append(message)
            return message.id
        
        # 立即送达并触发订阅
        await self._deliver_to_subscribers(message)
        return message.id
    
    async def send(
        self,
        run_id: str,
        from_role: str,
        to_role: str | None,
        message_type: str,
        payload: dict,
        broadcast: bool = False,
        related_task_id: str | None = None
    ) -> AsyncMessage:
        """发送消息的便捷方法"""
        message = AsyncMessage(
            id=str(uuid.uuid4()),
            type=message_type,
            from_role=from_role,
            to_role=to_role,
            broadcast=broadcast,
            payload=payload,
            related_task_id=related_task_id
        )
        await self.send_async(message)
        return message
    
    async def subscribe(
        self,
        agent_id: str,
        callback: Callable[[AsyncMessage], Awaitable[None]],
        message_types: list[str] | None = None,
        from_roles: list[str] | None = None
    ) -> str:
        """订阅消息"""
        subscription_id = str(uuid.uuid4())
        
        async def filtered_callback(msg: AsyncMessage):
            if message_types and msg.type not in message_types:
                return
            if from_roles and msg.from_role not in from_roles:
                return
            await callback(msg)
        
        if agent_id not in self._subscriptions:
            self._subscriptions[agent_id] = []
        self._subscriptions[agent_id].append(filtered_callback)
        
        return subscription_id
    
    async def process_queue(self) -> list[AsyncMessage]:
        """处理队列中的消息"""
        processed = []
        while self._queue:
            message = self._queue.pop(0)
            await self._deliver_to_subscribers(message)
            processed.append(message)
        return processed
    
    async def _deliver_to_subscribers(self, message: AsyncMessage):
        """向订阅者投递消息"""
        for agent_id, callbacks in self._subscriptions.items():
            # 跳过发送者
            if agent_id == message.from_role and not message.broadcast:
                continue
            
            for callback in callbacks:
                try:
                    await callback(message)
                except Exception as e:
                    logger.error(f"Error delivering message to {agent_id}: {e}")
    
    async def create_proposal(
        self,
        proposer: str,
        proposal_type: str,
        content: dict,
        deadline: datetime | None = None
    ) -> Proposal:
        """创建提案"""
        proposal = Proposal(
            id=str(uuid.uuid4()),
            proposer=proposer,
            content=content,
            proposal_type=proposal_type,
            deadline=deadline
        )
        self._proposals[proposal.id] = proposal
        return proposal
    
    async def vote(
        self,
        proposal_id: str,
        voter: str,
        vote: str  # "approve", "reject", "abstain"
    ) -> bool:
        """投票"""
        if proposal_id not in self._proposals:
            return False
        
        proposal = self._proposals[proposal_id]
        proposal.votes[voter] = vote
        return True
    
    async def get_proposal_result(self, proposal_id: str) -> VoteResult | None:
        """获取提案投票结果"""
        if proposal_id not in self._proposals:
            return None
        
        proposal = self._proposals[proposal_id]
        votes = proposal.votes
        
        votes_for = sum(1 for v in votes.values() if v == "approve")
        votes_against = sum(1 for v in votes.values() if v == "reject")
        votes_abstain = sum(1 for v in votes.values() if v == "abstain")
        
        return VoteResult(
            proposal_id=proposal_id,
            votes_for=votes_for,
            votes_against=votes_against,
            votes_abstain=votes_abstain,
            approved=votes_for > votes_against,
            details=votes
        )
    
    async def create_delegation(
        self,
        from_agent: str,
        to_role: str,
        task: str,
        constraints: dict | None = None
    ) -> Delegation:
        """创建委托"""
        delegation = Delegation(
            id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_role=to_role,
            task=task,
            constraints=constraints or {}
        )
        self._delegations[delegation.id] = delegation
        return delegation
    
    async def acknowledge_delegation(self, delegation_id: str) -> bool:
        """确认委托"""
        if delegation_id not in self._delegations:
            return False
        self._delegations[delegation_id].acknowledged = True
        return True
    
    def get_pending_delegations(self, to_role: str) -> list[Delegation]:
        """获取待处理的委托"""
        return [
            d for d in self._delegations.values()
            if d.to_role == to_role and not d.acknowledged
        ]


class DynamicQuestionGenerator:
    """
    动态子问题生成器
    
    根据当前研究发现自动生成新的子问题
    """
    
    def __init__(self, model_client=None):
        self.model_client = model_client
        self.generation_history: list[list[SubQuestion]] = []
    
    async def generate_sub_questions(
        self,
        objective: str,
        current_findings: list[dict],
        research_progress: float,
        explored_angles: list[str]
    ) -> list[SubQuestion]:
        """
        根据当前发现生成新的子问题
        
        Args:
            objective: 研究目标
            current_findings: 当前发现列表
            research_progress: 研究进度 (0-1)
            explored_angles: 已探索的角度
        """
        # 使用模型生成（如果有）
        if self.model_client:
            return await self._generate_with_model(
                objective, current_findings, research_progress, explored_angles
            )
        
        # Fallback: 基于规则生成
        return self._generate_rule_based(
            objective, current_findings, research_progress, explored_angles
        )
    
    async def _generate_with_model(
        self,
        objective: str,
        current_findings: list[dict],
        research_progress: float,
        explored_angles: list[str]
    ) -> list[SubQuestion]:
        """使用模型生成子问题"""
        
        prompt = f"""
        目标：{objective}
        
        已发现的信息：
        {json.dumps(current_findings, ensure_ascii=False, indent=2)}
        
        研究进度：{research_progress * 100:.0f}%
        已探索的方向：{explored_angles}
        
        请分析当前研究状态，识别：
        1. 信息缺口（未解决的关键问题）
        2. 新发现带来的新问题（从现有线索衍生的新方向）
        3. 被忽略的重要角度
        
        输出 JSON 格式：
        {{
            "new_questions": [
                {{
                    "question": "问题描述",
                    "priority": "high|medium|low",
                    "reason": "为什么这个问题重要",
                    "source": "哪个发现触发了这个问题",
                    "estimated_effort": "low|medium|high"
                }}
            ],
            "questions_to_merge": [["问题1", "问题2"]],
            "reasoning": "你的推理过程"
        }}
        """
        
        try:
            response = await self.model_client.complete(
                [{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            data = response.json_data if hasattr(response, 'json_data') else json.loads(response)
            
            sub_questions = []
            for q_data in data.get("new_questions", []):
                sub_question = SubQuestion(
                    id=str(uuid.uuid4()),
                    question=q_data["question"],
                    priority=Priority(q_data.get("priority", "medium")),
                    reason=q_data.get("reason", ""),
                    source=q_data.get("source", "model_generated"),
                    estimated_effort=q_data.get("estimated_effort", "medium"),
                    created_by="DynamicQuestionGenerator"
                )
                sub_questions.append(sub_question)
            
            self.generation_history.append(sub_questions)
            return sub_questions
            
        except Exception as e:
            logger.error(f"Error generating questions with model: {e}")
            return self._generate_rule_based(objective, current_findings, research_progress, explored_angles)
    
    def _generate_rule_based(
        self,
        objective: str,
        current_findings: list[dict],
        research_progress: float,
        explored_angles: list[str]
    ) -> list[SubQuestion]:
        """基于规则生成子问题（Fallback）"""
        
        sub_questions = []
        
        # 如果研究进度低，生成探索性问题
        if research_progress < 0.3:
            sub_questions.append(SubQuestion(
                id=str(uuid.uuid4()),
                question=f"{objective} 的基本信息是什么？",
                priority=Priority.HIGH,
                reason="基础信息缺失",
                source="progress_threshold"
            ))
        
        # 检查是否有未完成的关键发现
        for finding in current_findings:
            if "待确认" in str(finding.get("answer", "")):
                sub_questions.append(SubQuestion(
                    id=str(uuid.uuid4()),
                    question=finding.get("question", "") + " 的详细信息",
                    priority=Priority.HIGH,
                    reason="发现待确认信息",
                    source=finding.get("source", "")
                ))
        
        return sub_questions


class CollaborationProtocol:
    """
    协作协议
    
    实现：
    - 提案-投票机制
    - 任务委托协议
    - 知识共享机制
    """
    
    def __init__(self, mailbox: AsyncMailbox):
        self.mailbox = mailbox
        self.negotiation_history: list[dict] = []
    
    async def propose_and_vote(
        self,
        proposer: str,
        proposal_type: str,
        content: dict,
        voters: list[str],
        timeout_seconds: float = 30.0
    ) -> VoteResult:
        """
        提案-投票协议
        
        Args:
            proposer: 提案者
            proposal_type: 提案类型（如 "spawn_sub_question"）
            content: 提案内容
            voters: 投票者列表
            timeout_seconds: 超时时间
        
        Returns:
            VoteResult: 投票结果
        """
        # 1. 创建提案
        deadline = datetime.now(UTC).replace(
            second=datetime.now(UTC).second + int(timeout_seconds)
        )
        proposal = await self.mailbox.create_proposal(
            proposer=proposer,
            proposal_type=proposal_type,
            content=content,
            deadline=deadline
        )
        
        # 2. 广播提案给投票者
        await self.mailbox.send(
            run_id=content.get("run_id", ""),
            from_role=proposer,
            broadcast=False,
            message_type="proposal",
            payload={
                "proposal_id": proposal.id,
                "content": content,
                "type": proposal_type,
                "deadline": deadline.isoformat()
            },
            to_role=",".join(voters)  # 简化处理
        )
        
        # 3. 等待投票（异步）
        start_time = datetime.now(UTC)
        while (datetime.now(UTC) - start_time).total_seconds() < timeout_seconds:
            await asyncio.sleep(0.5)
            result = await self.mailbox.get_proposal_result(proposal.id)
            if result and len(result.details) >= len(voters):
                break
        
        # 4. 返回结果
        final_result = await self.mailbox.get_proposal_result(proposal.id)
        if not final_result:
            return VoteResult(
                proposal_id=proposal.id,
                votes_for=0,
                votes_against=0,
                votes_abstain=len(voters),
                approved=False,
                details={}
            )
        
        return final_result
    
    async def negotiate(
        self,
        parties: list[str],
        topic: str,
        initial_positions: dict[str, str]
    ) -> dict | None:
        """
        协商协议
        
        Args:
            parties: 参与方
            topic: 协商主题
            initial_positions: 各方初始立场
        
        Returns:
            达成的协议或 None
        """
        negotiation = {
            "id": str(uuid.uuid4()),
            "parties": parties,
            "topic": topic,
            "rounds": [],
            "status": "in_progress"
        }
        
        current_positions = dict(initial_positions)
        max_rounds = 5
        
        for round_num in range(max_rounds):
            # 1. 分析立场
            common_points = self._find_common_points(current_positions)
            
            if len(common_points) > 0.7:
                # 高共识，达成协议
                negotiation["status"] = "agreed"
                negotiation["agreement"] = common_points
                self.negotiation_history.append(negotiation)
                return negotiation
            
            # 2. 识别分歧
            disagreements = self._find_disagreements(current_positions)
            
            # 3. 生成妥协方案
            compromise = self._generate_compromise(current_positions, disagreements)
            
            # 4. 广播妥协方案
            await self.mailbox.send(
                run_id="",
                from_role="system",
                to_role=None,
                broadcast=True,
                message_type="negotiation",
                payload={
                    "negotiation_id": negotiation["id"],
                    "round": round_num,
                    "compromise": compromise,
                    "common_points": common_points,
                    "disagreements": disagreements
                }
            )
            
            # 5. 等待回应（简化处理）
            await asyncio.sleep(0.5)
            
            negotiation["rounds"].append({
                "round": round_num,
                "positions": dict(current_positions),
                "compromise": compromise
            })
        
        negotiation["status"] = "failed"
        self.negotiation_history.append(negotiation)
        return None
    
    def _find_common_points(self, positions: dict[str, str]) -> list[str]:
        """找出共同点"""
        if not positions:
            return []
        
        # 简化：检查是否有相同的关键词
        all_words = set()
        for pos in positions.values():
            all_words.update(pos.split())
        
        # 返回出现多次的词
        word_counts = {}
        for word in all_words:
            count = sum(1 for pos in positions.values() if word in pos)
            if count > len(positions) / 2:
                word_counts[word] = count
        
        return list(word_counts.keys())
    
    def _find_disagreements(self, positions: dict[str, str]) -> list[dict]:
        """找出分歧"""
        # 简化实现
        return []
    
    def _generate_compromise(
        self,
        positions: dict[str, str],
        disagreements: list[dict]
    ) -> str:
        """生成妥协方案"""
        # 简化实现
        return "建议采用折中方案"


@dataclass
class EnhancedDecision:
    """增强版决策结果"""
    action: str
    reasoning: str
    confidence: float
    alternatives: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    # 协作相关字段
    spawned_questions: list[SubQuestion] = field(default_factory=list)
    delegation: Delegation | None = None
    merge_candidates: list[str] = field(default_factory=list)


class EnhancedSubResearcherWorker:
    """
    增强版 SubResearcher Worker
    
    主要改进：
    1. 扩展的 Action 空间（27 个 actions）
    2. 动态子问题生成
    3. 异步协作协议
    4. 更宽松的决策验证
    """
    
    def __init__(
        self,
        task_store,
        mailbox: AsyncMailbox,
        artifact_store,
        board_store,
        model_client=None,
        config: dict | None = None
    ):
        self.task_store = task_store
        self.mailbox = mailbox
        self.artifact_store = artifact_store
        self.board_store = board_store
        self.model_client = model_client
        self.config = config or {}
        
        # 初始化组件
        self.question_generator = DynamicQuestionGenerator(model_client)
        self.collaboration_protocol = CollaborationProtocol(mailbox)
        
        # 配置
        self.max_iterations = self.config.get("max_iterations", 0)  # 0 = 无限制
        self.allow_unknown_actions = self.config.get("allow_unknown_actions", True)
        self.max_parallel_questions = self.config.get("max_parallel_questions", 5)
        
        # 状态
        self.current_task = None
        self.scratch: dict[str, Any] = {}
        self.metrics = {
            "unexpected_actions": [],
            "spawned_questions": [],
            "delegations": [],
            "collaborations": []
        }
    
    async def run_once_enhanced(self, run_id: str) -> dict | None:
        """
        增强版单次运行
        
        Returns:
            执行结果或 None（如果没有可执行任务）
        """
        # 1. 认领任务
        task = self.task_store.claim_next(run_id, owner_role="sub_researcher")
        if not task:
            return None
        
        self.current_task = task
        self.scratch = self._init_scratch(task)
        
        # 2. 增强版处理循环
        while self.scratch["iteration_count"] < self.max_iterations or self.max_iterations == 0:
            self.scratch["iteration_count"] += 1
            
            try:
                # 组装上下文
                context = await self._assemble_context_enhanced(task)
                
                # 增强版决策
                decision = await self._decide_action_enhanced(context)
                
                # 验证决策（宽松模式）
                validated = self._validate_action_enhanced(context, decision)
                
                # 执行决策
                result = await self._execute_decision(validated)
                
                # 更新状态
                self._update_scratch(result, decision)
                
                # 广播进度（异步）
                await self._broadcast_progress(task, decision, result)
                
                # 检查是否完成
                if result.get("terminal"):
                    return self._build_result(task, result)
                
            except Exception as e:
                logger.error(f"Error in iteration {self.scratch['iteration_count']}: {e}")
                self.scratch["errors"].append(str(e))
                
                if self.scratch["iteration_count"] >= 9:
                    break
        
        # 达到最大迭代
        return self._build_result(self.current_task, {"action": "max_iterations"})
    
    async def _assemble_context_enhanced(self, task) -> dict:
        """增强版上下文组装"""
        
        # 基础上下文
        context = {
            "task": {
                "task_id": task.task_id,
                "kind": task.kind,
                "run_id": task.run_id
            },
            "objective": self.scratch.get("objective", ""),
            "iteration": self.scratch["iteration_count"],
            
            # 团队状态（新增）
            "team_state": await self._get_team_state(),
            
            # 历史决策（新增）
            "decision_history": self.scratch.get("decision_history", []),
            
            # 资源预算
            "tool_budget": {
                "remaining_searches": max(0, 3 - self.scratch.get("search_count", 0)),
                "remaining_fetches": max(0, 5 - self.scratch.get("fetch_count", 0)),
                "remaining_sub_questions": max(
                    0, 
                    self.max_parallel_questions - len(self.scratch.get("sub_questions", []))
                )
            },
            
            # 工具状态
            "tool_state": {
                "task_context_loaded": bool(self.scratch.get("task_context")),
                "search_results_count": len(self.scratch.get("search_results", [])),
                "documents_count": len(self.scratch.get("documents", [])),
                "sub_questions_count": len(self.scratch.get("sub_questions", []))
            }
        }
        
        return context
    
    async def _decide_action_enhanced(self, context: dict) -> EnhancedDecision:
        """
        增强版决策
        
        关键改进：
        1. 使用扩展的 action 空间
        2. 包含推理过程
        3. 允许未知 action（记录但不 fallback）
        """
        
        if not self.model_client:
            # Fallback 到规则决策
            return self._rule_based_decision(context)
        
        # 构建决策提示
        prompt = self._build_decision_prompt(context)
        
        try:
            response = await self.model_client.complete(
                [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_object",
                    "schema": {
                        "action": "string",
                        "reasoning": "string",
                        "confidence": "float",
                        "alternatives": ["string"],
                        "if_spawn_question": {
                            "question": "string",
                            "priority": "string",
                            "reason": "string"
                        },
                        "if_delegate": {
                            "target_role": "string",
                            "task": "string",
                            "constraints": {}
                        }
                    }
                }
            )
            
            data = response.json_data if hasattr(response, 'json_data') else {}
            
            action = data.get("action", "blocked")
            
            # 宽松验证：记录未知 action 但不 fallback
            if action not in EXTENDED_ACTIONS:
                self.metrics["unexpected_actions"].append({
                    "action": action,
                    "iteration": context["iteration"],
                    "reasoning": data.get("reasoning", "")
                })
                logger.warning(
                    f"Unexpected action '{action}' at iteration {context['iteration']}, "
                    f"but proceeding with model decision"
                )
            
            decision = EnhancedDecision(
                action=action,
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.5),
                alternatives=data.get("alternatives", []),
                metadata=data
            )
            
            # 处理协作相关的决策
            if action == "spawn_sub_question":
                question_data = data.get("if_spawn_question", {})
                decision.spawned_questions = [
                    SubQuestion(
                        id=str(uuid.uuid4()),
                        question=question_data.get("question", ""),
                        priority=Priority(question_data.get("priority", "medium")),
                        reason=question_data.get("reason", ""),
                        source="model_decision",
                        created_by="enhanced_sub_researcher"
                    )
                ]
            
            elif action == "delegate_to_peer":
                delegate_data = data.get("if_delegate", {})
                decision.delegation = await self.mailbox.create_delegation(
                    from_agent="sub_researcher",
                    to_role=delegate_data.get("target_role", "sub_researcher"),
                    task=delegate_data.get("task", ""),
                    constraints=delegate_data.get("constraints", {})
                )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in enhanced decision: {e}")
            return self._rule_based_decision(context)
    
    def _validate_action_enhanced(
        self,
        context: dict,
        decision: EnhancedDecision
    ) -> dict:
        """
        增强版验证
        
        与原版的区别：
        1. 接受更多 action 类型
        2. 协作类 action 有单独的验证逻辑
        """
        
        action = decision.action
        
        # 传统 action 的验证（保持原有逻辑）
        if action in {"search", "fetch", "publish_raw_document"}:
            budget = context["tool_budget"]
            task_context_loaded = context["tool_state"]["task_context_loaded"]
            
            if not task_context_loaded:
                return {"action": "blocked", "reason": "Need task context first"}
            
            if action == "search" and budget["remaining_searches"] <= 0:
                return {"action": "blocked", "reason": "Search budget exhausted"}
            
            if action == "fetch":
                if budget["remaining_fetches"] <= 0:
                    return {"action": "blocked", "reason": "Fetch budget exhausted"}
                if not decision.metadata.get("url"):
                    return {"action": "blocked", "reason": "Fetch needs URL"}
        
        # 协作类 action 的验证（新增）
        elif action in {"spawn_sub_question", "delegate_to_peer"}:
            budget = context["tool_budget"]
            
            if action == "spawn_sub_question":
                if budget["remaining_sub_questions"] <= 0:
                    return {"action": "blocked", "reason": "Max parallel questions reached"}
                if not decision.spawned_questions:
                    return {"action": "blocked", "reason": "No question to spawn"}
            
            if action == "delegate_to_peer":
                if not decision.delegation:
                    return {"action": "blocked", "reason": "No delegation specified"}
        
        # 验证通过
        return {
            "action": decision.action,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
            "alternatives": decision.alternatives,
            "metadata": decision.metadata,
            "spawned_questions": decision.spawned_questions,
            "delegation": decision.delegation
        }
    
    async def _execute_decision(self, validated: dict) -> dict:
        """执行决策"""
        
        action = validated["action"]
        
        # 传统 action
        if action in {"search", "fetch", "publish_raw_document", "query_task_context"}:
            return await self._execute_traditional_action(action, validated)
        
        # 协作 action
        elif action == "spawn_sub_question":
            return await self._execute_spawn_sub_question(validated)
        
        elif action == "delegate_to_peer":
            return await self._execute_delegate(validated)
        
        elif action == "merge_questions":
            return await self._execute_merge_questions(validated)
        
        elif action == "replan":
            return await self._execute_replan(validated)
        
        elif action == "fetch_multiple":
            return await self._execute_fetch_multiple(validated)
        
        elif action == "request_help":
            return await self._execute_request_help(validated)
        
        elif action in {"complete", "blocked"}:
            return {"terminal": True, "action": action}
        
        else:
            # 未知 action，执行 fallback
            logger.warning(f"Executing fallback for action: {action}")
            return await self._execute_fallback(action, validated)
    
    async def _execute_spawn_sub_question(self, validated: dict) -> dict:
        """执行生成子问题"""
        
        spawned_questions = validated.get("spawned_questions", [])
        
        created_ids = []
        for question in spawned_questions:
            # 创建任务
            new_task = self.task_store.create(
                run_id=self.current_task.run_id,
                kind="research_subquestion",
                status="pending",
                owner_role="sub_researcher",
                inputs={"question": question.question},
                priority=self._priority_to_int(question.priority),
                created_by="enhanced_sub_researcher"
            )
            created_ids.append(new_task.task_id)
            self.metrics["spawned_questions"].append(question.to_dict())
            
            # 异步通知 Lead
            await self.mailbox.send_async(
                AsyncMessage(
                    id=str(uuid.uuid4()),
                    type="observation",
                    from_role="sub_researcher",
                    to_role="lead",
                    broadcast=False,
                    payload={
                        "kind": "sub_question_spawned",
                        "question": question.to_dict(),
                        "parent_task_id": self.current_task.task_id
                    }
                ),
                delivery=DeliveryMode.QUEUED
            )
        
        self.scratch["sub_questions"].extend(spawned_questions)
        
        return {
            "terminal": False,
            "action": "spawn_sub_question",
            "created_task_ids": created_ids,
            "spawned_questions": spawned_questions
        }
    
    async def _execute_delegate(self, validated: dict) -> dict:
        """执行任务委托"""
        
        delegation = validated.get("delegation")
        if not delegation:
            return {"terminal": False, "action": "delegate_to_peer", "status": "failed"}
        
        # 创建委托任务
        delegated_task = self.task_store.create(
            run_id=self.current_task.run_id,
            kind="delegated_research",
            status="pending",
            owner_role=delegation.to_role,
            inputs={
                "task": delegation.task,
                "delegation_id": delegation.id,
                "constraints": delegation.constraints,
                "delegated_from": delegation.from_agent
            },
            priority=5,
            created_by="enhanced_sub_researcher"
        )
        
        self.metrics["delegations"].append({
            "delegation_id": delegation.id,
            "to_role": delegation.to_role,
            "task_id": delegated_task.task_id
        })
        
        # 广播委托
        await self.mailbox.send_async(
            AsyncMessage(
                id=str(uuid.uuid4()),
                type="delegation",
                from_role="sub_researcher",
                to_role=delegation.to_role,
                broadcast=False,
                payload={
                    "kind": "task_delegated",
                    "delegation_id": delegation.id,
                    "task_id": delegated_task.task_id,
                    "task": delegation.task
                }
            ),
            delivery=DeliveryMode.IMMEDIATE
        )
        
        return {
            "terminal": False,
            "action": "delegate_to_peer",
            "created_task_ids": [delegated_task.task_id],
            "delegation_id": delegation.id
        }
    
    async def _execute_fetch_multiple(self, validated: dict) -> dict:
        """执行批量获取"""
        
        urls = validated.get("metadata", {}).get("urls", [])
        if not urls:
            return {"terminal": False, "action": "fetch_multiple", "status": "no_urls"}
        
        results = []
        for url in urls[:3]:  # 限制批量数量
            # 这里简化处理，实际应该调用 fetch tool
            self.scratch["fetch_count"] += 1
            results.append({"url": url, "status": "fetched"})
        
        return {
            "terminal": False,
            "action": "fetch_multiple",
            "results": results
        }
    
    async def _execute_request_help(self, validated: dict) -> dict:
        """执行请求帮助"""
        
        reason = validated.get("reasoning", "")
        needed_capabilities = validated.get("metadata", {}).get("needed_capabilities", [])
        
        # 向 Lead 发送帮助请求
        help_request = await self.mailbox.send_async(
            AsyncMessage(
                id=str(uuid.uuid4()),
                type="help_request",
                from_role="sub_researcher",
                to_role="lead",
                broadcast=False,
                payload={
                    "kind": "request_help",
                    "reason": reason,
                    "needed_capabilities": needed_capabilities,
                    "task_id": self.current_task.task_id
                }
            ),
            delivery=DeliveryMode.IMMEDIATE
        )
        
        return {
            "terminal": False,
            "action": "request_help",
            "help_request_id": help_request
        }
    
    async def _execute_replan(self, validated: dict) -> dict:
        """执行重新规划"""
        
        reason = validated.get("reasoning", "")
        
        # 广播重新规划
        await self.mailbox.send_async(
            AsyncMessage(
                id=str(uuid.uuid4()),
                type="observation",
                from_role="sub_researcher",
                to_role=None,
                broadcast=True,
                payload={
                    "kind": "replan",
                    "reason": reason,
                    "task_id": self.current_task.task_id,
                    "new_plan": validated.get("metadata", {}).get("new_plan", {})
                }
            ),
            delivery=DeliveryMode.QUEUED
        )
        
        # 重置部分状态
        self.scratch["search_results"] = []
        self.scratch["documents"] = []
        self.scratch["attempted_queries"] = []
        self.scratch["attempted_urls"] = []
        
        return {
            "terminal": False,
            "action": "replan"
        }
    
    async def _broadcast_progress(
        self,
        task,
        decision: EnhancedDecision,
        result: dict
    ):
        """异步广播进度"""
        
        await self.mailbox.send_async(
            AsyncMessage(
                id=str(uuid.uuid4()),
                type="observation",
                from_role="sub_researcher",
                to_role=None,
                broadcast=True,
                payload={
                    "kind": "enhanced_progress",
                    "source_task_id": task.task_id,
                    "round": self.scratch["iteration_count"],
                    "action": decision.action,
                    "reasoning": decision.reasoning,
                    "confidence": decision.confidence,
                    "team_state": await self._get_team_state(),
                    "sub_questions_count": len(self.scratch.get("sub_questions", [])),
                    "delegations_count": len(self.metrics["delegations"])
                },
                related_task_id=task.task_id
            ),
            delivery=DeliveryMode.QUEUED
        )
    
    async def _get_team_state(self) -> dict:
        """获取团队状态"""
        
        # 查询活跃任务
        pending_tasks = self.task_store.list_pending_tasks()
        active_researchers = sum(
            1 for t in pending_tasks 
            if t.owner_role == "sub_researcher"
        )
        
        return {
            "active_researchers": active_researchers,
            "pending_tasks": len(pending_tasks),
            "pending_delegations": len(
                self.mailbox.get_pending_delegations("sub_researcher")
            )
        }
    
    def _init_scratch(self, task) -> dict:
        """初始化 scratch 状态"""
        return {
            "objective": getattr(task, 'objective', ''),
            "task_context": None,
            "search_count": 0,
            "fetch_count": 0,
            "attempted_queries": [],
            "attempted_urls": [],
            "search_results": [],
            "documents": [],
            "sub_questions": [],
            "errors": [],
            "iteration_count": 0,
            "decision_history": []
        }
    
    def _update_scratch(self, result: dict, decision: EnhancedDecision):
        """更新 scratch 状态"""
        
        self.scratch["decision_history"].append({
            "iteration": self.scratch["iteration_count"],
            "action": decision.action,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence
        })
        
        if result.get("action") == "search":
            self.scratch["search_count"] += 1
        elif result.get("action") == "fetch":
            self.scratch["fetch_count"] += 1
    
    def _build_decision_prompt(self, context: dict) -> str:
        """构建决策提示"""
        
        return f"""
        任务：{context['objective']}
        当前轮次：{context['iteration']}
        
        团队状态：
        - 活跃研究员：{context['team_state']['active_researchers']}
        - 待处理任务：{context['team_state']['pending_tasks']}
        
        资源预算：
        - 剩余搜索次数：{context['tool_budget']['remaining_searches']}
        - 剩余获取次数：{context['tool_budget']['remaining_fetches']}
        - 剩余子问题配额：{context['tool_budget']['remaining_sub_questions']}
        
        当前状态：
        - 搜索结果数：{context['tool_state']['search_results_count']}
        - 文档数：{context['tool_state']['documents_count']}
        - 已生成子问题：{context['tool_state']['sub_questions_count']}
        
        决策历史：
        {json.dumps(context.get('decision_history', [])[-3:], ensure_ascii=False, indent=2)}
        
        请做出决策。你可以选择以下任何 action（不只是 search/fetch）：
        {json.dumps(list(EXTENDED_ACTIONS), ensure_ascii=False, indent=2)}
        
        如果选择 spawn_sub_question，请提供问题内容。
        如果选择 delegate_to_peer，请指定目标和任务。
        
        输出 JSON 格式。
        """
    
    def _get_system_prompt(self) -> str:
        """获取系统提示"""
        
        return f"""你是一个自主的研究 Agent。

目标：完成分配的研究任务

可用 Actions（{len(EXTENDED_ACTIONS)} 种）：
- 基础：query_task_context, query_mailbox, query_evidence
- 搜索：search, fetch, fetch_multiple, analyze_and_search
- 研究：publish_raw_document, spawn_sub_question, merge_questions, split_question
- 协作：delegate_to_peer, request_help, propose_collaboration, share_knowledge
- 元认知：replan, reflect, request_feedback, adjust_priority
- 完成：complete, blocked

规则：
1. 如果没有任务上下文，先 query_task_context
2. 可以动态生成新的子问题（spawn_sub_question）
3. 可以委托任务给其他 Agent（delegate_to_peer）
4. 如果资源耗尽或无法继续，使用 blocked
5. 包含你的推理过程（reasoning）
6. 提供置信度（confidence）

决策时要考虑：
- 当前研究进度
- 团队协作机会
- 资源使用效率
- 发现的新线索
"""
    
    def _rule_based_decision(self, context: dict) -> EnhancedDecision:
        """基于规则的决策（当没有模型时）"""
        
        if not context["tool_state"]["task_context_loaded"]:
            return EnhancedDecision(
                action="query_task_context",
                reasoning="Need task context first",
                confidence=0.9
            )
        
        if context["tool_state"]["search_results_count"] == 0:
            return EnhancedDecision(
                action="search",
                reasoning="No search results yet",
                confidence=0.7
            )
        
        if context["tool_budget"]["remaining_sub_questions"] > 0:
            # 考虑生成子问题
            return EnhancedDecision(
                action="spawn_sub_question",
                reasoning="Can spawn sub question to explore new direction",
                confidence=0.6
            )
        
        return EnhancedDecision(
            action="blocked",
            reasoning="No more actions available",
            confidence=0.5
        )
    
    async def _execute_traditional_action(self, action: str, validated: dict) -> dict:
        """执行传统 action"""
        return {"terminal": False, "action": action}
    
    async def _execute_merge_questions(self, validated: dict) -> dict:
        """执行合并问题"""
        return {"terminal": False, "action": "merge_questions"}
    
    async def _execute_fallback(self, action: str, validated: dict) -> dict:
        """执行 fallback"""
        return {"terminal": True, "action": "fallback", "original_action": action}
    
    def _build_result(self, task, result: dict) -> dict:
        """构建结果"""
        return {
            "claimed_task_id": task.task_id,
            "action": result.get("action"),
            "created_task_ids": result.get("created_task_ids", []),
            "metrics": self.metrics
        }
    
    def _priority_to_int(self, priority: Priority) -> int:
        """优先级转整数"""
        return {
            Priority.HIGH: 10,
            Priority.MEDIUM: 5,
            Priority.LOW: 1
        }.get(priority, 5)


# ============ 示例用法 ============

async def example_usage():
    """示例用法"""
    
    # 初始化组件
    mailbox = AsyncMailbox()
    
    # 创建增强版 worker（需要真实的 task_store, artifact_store, board_store）
    # worker = EnhancedSubResearcherWorker(
    #     task_store=task_store,
    #     mailbox=mailbox,
    #     artifact_store=artifact_store,
    #     board_store=board_store,
    #     model_client=model_client,
    #     config={
    #         "allow_unknown_actions": True,
    #         "max_iterations": 0,  # 无限制
    #         "max_parallel_questions": 5
    #     }
    # )
    
    # 运行
    # result = await worker.run_once_enhanced(run_id="test_run")
    
    print("EnhancedSubResearcherWorker 示例")
    print(f"支持的 Actions 数量: {len(EXTENDED_ACTIONS)}")
    print(f"Actions: {sorted(EXTENDED_ACTIONS)}")


if __name__ == "__main__":
    asyncio.run(example_usage())
