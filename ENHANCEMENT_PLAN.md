# InsightSwarm 放宽约束实现方案

## 1. 当前问题的根本原因

### 1.1 硬编码的 Action 集合
```python
# 当前的严格约束
VALID_ACTIONS = {
    "query_task_context",
    "query_mailbox",
    "query_evidence",
    "search",
    "fetch",
    "publish_raw_document",
    "suggest_browser",
    "complete",
    "blocked"
}
```

**问题**：
- Agent 只能做这 9 件事
- 模型推理被限制在这 9 个选项内
- 无法产生"涌现"行为

### 1.2 中心化的任务分配
```python
# Lead 静态分配所有任务
for sub_question in sub_questions[:3]:
    board_store.create_question(
        owner_role="sub_researcher",  # 固定角色
        ...
    )
```

**问题**：
- Agent 无法动态生成新任务
- 无法根据发现调整研究方向
- 无法委托给其他 Agent

### 1.3 同步阻塞的执行模式
```python
# 当前：线性等待
result = extractor.extract(raw_document)
citation = citation_generator.create(result)
qa_passed = qa_agent.validate(citation)
```

**问题**：
- 必须等待前一阶段完成
- 无法并行探索多个方向
- 浪费等待时间

---

## 2. 放宽约束的 3 阶段方案

### 阶段 1：扩展 Action 空间（立即生效）

#### 新增的 Dynamic Actions：
```python
DYNAMIC_ACTIONS = {
    # === 研究策略类 ===
    "spawn_sub_question",      # 生成新的子问题
    "merge_questions",        # 合并相关问题
    "split_question",         # 拆分复杂问题
    
    # === 协作类 ===
    "delegate_to_peer",        # 委托给其他 Agent
    "request_help",            # 向 Lead 请求帮助
    "propose_collaboration",   # 向其他 Agent 提议协作
    "share_knowledge",        # 分享知识给团队
    
    # === 推理类 ===
    "replan",                 # 根据新信息重新规划
    "revise_objective",       # 修改目标
    "question_assumption",     # 质疑当前假设
    
    # === 执行类 ===
    "fetch_multiple",          # 批量获取多个 URL
    "analyze_and_search",      # 搜索后立即分析
    "parallel_investigate",    # 并行调查多个方向
    
    # === 元认知类 ===
    "reflect",                # 反思当前策略
    "request_feedback",        # 请求反馈
    "adjust_priority",         # 调整优先级
}
```

#### 扩展的 Prompt Template：
```markdown
# 新增的 Action 说明

### 9. spawn_sub_question
- Description: 根据当前发现，动态生成新的研究子问题
- Use when: 发现新的相关线索，需要深入调查
- Example: {"action": "spawn_sub_question", "question": "DeepSeek 最新融资情况", "priority": "high", "reason": "从官方博客发现的新线索"}

### 10. delegate_to_peer
- Description: 将部分研究委托给其他 Agent
- Use when: 当前任务可以被并行处理，或其他 Agent 更擅长
- Example: {"action": "delegate_to_peer", "target_role": "sub_researcher", "task": "调查竞品定价策略", "constraints": {"max_sources": 5}}

### 11. merge_questions
- Description: 合并多个相关问题以提高效率
- Use when: 多个子问题指向同一方向
- Example: {"action": "merge_questions", "questions": ["问题1", "问题2"], "merged_question": "统一的问题描述"}

### 12. replan
- Description: 根据新信息重新制定计划
- Use when: 原始计划不再适用，需要调整方向
- Example: {"action": "replan", "reason": "发现了更权威的信息源", "new_approach": "优先调查官方网站"}

### 13. fetch_multiple
- Description: 批量获取多个 URL
- Use when: 有多个高优先级 URL 需要获取
- Example: {"action": "fetch_multiple", "urls": ["url1", "url2", "url3"], "priority": "high"}

### 14. parallel_investigate
- Description: 并行调查多个方向
- Use when: 不确定哪个方向最有效
- Example: {"action": "parallel_investigate", "directions": ["官方文档", "媒体报道", "社区讨论"], "budget_per_direction": 2}
```

---

## 3. 动态子问题生成机制

### 3.1 核心设计

```python
class DynamicQuestionGenerator:
    """动态子问题生成器"""
    
    def __init__(self, model_client):
        self.model = model_client
    
    async def generate_sub_questions(
        self,
        objective: str,
        current_findings: list[Finding],
        research_context: ResearchContext
    ) -> list[SubQuestion]:
        """
        根据当前发现动态生成新的子问题
        """
        prompt = f"""
        目标：{objective}
        
        已发现的信息：
        {self._format_findings(current_findings)}
        
        当前研究进度：{research_context.progress}%
        已探索的方向：{research_context.explored_angles}
        
        请分析当前研究状态，识别：
        1. 信息缺口（未解决的关键问题）
        2. 新发现带来的新问题（从现有线索衍生的新方向）
        3. 被忽略的重要角度
        4. 可以合并的重复问题
        
        输出 JSON 格式：
        {{
            "new_questions": [
                {{
                    "question": "问题描述",
                    "priority": "high|medium|low",
                    "reason": "为什么这个问题重要",
                    "source": "哪个发现触发了这个问题",
                    "estimated_effort": "low|medium|high",
                    "dependencies": ["前置问题ID"]
                }}
            ],
            "questions_to_merge": [["问题1", "问题2"]],
            "questions_to_reprioritize": [{{"id": "xxx", "new_priority": "high"}}]
        }}
        """
        
        result = await self.model.complete(prompt, response_format="json")
        return self._parse_sub_questions(result)
```

### 3.2 SubQuestion 数据结构

```python
@dataclass
class SubQuestion:
    id: str
    question: str
    priority: Priority
    reason: str
    source: str  # 触发这个问题的发现
    estimated_effort: EffortLevel
    dependencies: list[str]
    status: QuestionStatus
    assigned_to: str | None
    created_by: str  # 哪个 Agent 创建的
    created_at: datetime
    
    # 新增的协作字段
    can_be_parallel: bool
    merging_candidates: list[str]
    related_questions: list[str]
```

### 3.3 与 Lead 的动态交互

```python
class DynamicLeadProtocol:
    """动态 Lead 协议"""
    
    async def handle_agent_request(
        self,
        agent_id: str,
        request: AgentRequest
    ) -> LeadResponse:
        """处理 Agent 的动态请求"""
        
        if request.type == "spawn_sub_question":
            return await self._handle_spawn_request(agent_id, request)
        
        elif request.type == "delegate_to_peer":
            return await self._handle_delegation(agent_id, request)
        
        elif request.type == "merge_questions":
            return await self._handle_merge(agent_id, request)
        
        elif request.type == "request_help":
            return await self._handle_help_request(agent_id, request)
    
    async def _handle_spawn_request(
        self,
        agent_id: str,
        request: SpawnRequest
    ) -> LeadResponse:
        """
        处理子问题生成请求
        
        1. 验证请求的合理性
        2. 检查是否与现有问题重复
        3. 创建新问题并分配
        4. 更新研究计划
        """
        # 检查重复
        existing = self.board.find_similar(request.question)
        if existing.similarity > 0.8:
            return LeadResponse(
                status="duplicate",
                message="类似问题已存在",
                existing_question_id=existing.id
            )
        
        # 创建新问题
        new_question = self.board.create_question(
            title=request.question,
            question_type="dynamic_subquestion",
            owner_role=request.preferred_role or "sub_researcher",
            priority=request.priority,
            created_by=agent_id,
            payload={
                "reason": request.reason,
                "source": request.source,
                "estimated_effort": request.estimated_effort,
                "can_be_parallel": True  # 动态生成的问题可以并行
            }
        )
        
        # 通知相关 Agent
        await self.broadcast(QuestionCreated(
            question=new_question,
            created_by=agent_id,
            related_to=request.source
        ))
        
        return LeadResponse(
            status="created",
            question_id=new_question.id,
            message=f"已创建新子问题：{request.question}"
        )
```

---

## 4. 异步协作协议设计

### 4.1 消息类型扩展

```python
# 新的消息类型
EXTENDED_MESSAGE_TYPES = {
    # 原有类型
    "request", "response", "observation", "suggestion", "hypothesis",
    
    # 新增类型
    "proposal",        # 提案（需要投票）
    "vote",            # 投票
    "negotiation",     # 协商中
    "agreement",       # 达成协议
    "delegation",      # 委托
    "knowledge_share",  # 知识共享
    "help_request",    # 请求帮助
    "capability_announce",  # 能力声明
}

# 新增的 Payload 类型
EXTENDED_PAYLOAD_KINDS = {
    "proposal": {
        "spawn_sub_question",
        "merge_questions",
        "change_priority",
        "delegate_work",
        "share_findings",
        "request_review",
    },
    "vote": {
        "approve",
        "reject",
        "abstain",
        "request_modification",
    },
    "delegation": {
        "transfer_task",
        "share_workload",
        "expert_referral",
    }
}
```

### 4.2 异步消息处理

```python
class AsyncMailbox:
    """异步消息队列"""
    
    async def send_async(
        self,
        message: Message,
        delivery: DeliveryMode = DeliveryMode.IMMEDIATE
    ) -> MessageId:
        """
        发送异步消息
        - IMMEDIATE: 立即送达
        - QUEUED: 进入队列，等待处理
        - CONDITIONAL: 满足条件后送达
        """
        if delivery == DeliveryMode.QUEUED:
            await self.queue.enqueue(message)
            return MessageId(queue_position=await self.queue.size())
        
        elif delivery == DeliveryMode.CONDITIONAL:
            condition = message.delivery_condition
            if await self._check_condition(condition):
                await self._deliver(message)
            else:
                await self._register_condition(message, condition)
            return message.id
    
    async def subscribe(
        self,
        agent_id: str,
        callback: Callable[[Message], Awaitable[None]],
        filters: MessageFilter | None = None
    ) -> SubscriptionId:
        """订阅消息流"""
        subscription = Subscription(
            agent_id=agent_id,
            callback=callback,
            filters=filters
        )
        await self.subscriptions.add(subscription)
        return subscription.id
```

### 4.3 协作协议实现

```python
class CollaborationProtocol:
    """协作协议"""
    
    async def propose_and_vote(
        self,
        proposer: AgentId,
        proposal: Proposal
    ) -> VoteResult:
        """
        提案-投票协议
        
        流程：
        1. Agent A 提出提案
        2. 系统广播给所有相关 Agent
        3. 每个 Agent 投票（可以自动或人工）
        4. 统计结果并执行
        """
        # 1. 创建提案
        proposal_record = await self.create_proposal(proposer, proposal)
        
        # 2. 确定投票者
        voters = await self._determine_voters(proposal)
        
        # 3. 广播提案
        for voter in voters:
            await self.mailbox.send(Message(
                type="proposal",
                to_role=voter,
                payload={
                    "proposal_id": proposal_record.id,
                    "content": proposal.content,
                    "deadline": proposal.deadline
                }
            ))
        
        # 4. 收集投票（异步等待）
        votes = await self._collect_votes(
            proposal_record.id,
            voters,
            timeout=proposal.deadline
        )
        
        # 5. 统计结果
        return self._tally_votes(votes)
    
    async def negotiate(
        self,
        parties: list[AgentId],
        topic: str,
        initial_positions: dict[AgentId, Position]
    ) -> Agreement | None:
        """
        协商协议
        
        流程：
        1. 各方陈述立场
        2. 识别共同点和分歧
        3. 迭代提出妥协方案
        4. 达成协议或放弃
        """
        negotiation = await self._init_negotiation(parties, topic)
        
        for round_num in range(MAX_NEGOTIATION_ROUNDS):
            # 交换立场
            positions = await self._gather_positions(negotiation)
            
            # 分析共同点和分歧
            analysis = self._analyze_positions(positions)
            
            if analysis.common_ground > 0.8:
                # 高共识，达成协议
                return await self._create_agreement(negotiation, analysis)
            
            # 提出妥协方案
            compromise = await self._propose_compromise(
                negotiation,
                positions,
                analysis
            )
            
            # 检查是否接受
            acceptances = await self._check_acceptances(
                compromise,
                parties
            )
            
            if acceptances.all_accepted():
                return await self._create_agreement(negotiation, compromise)
        
        return None  # 协商失败
```

---

## 5. 完整实现示例

### 5.1 增强版 SubResearcher

```python
class EnhancedSubResearcherWorker:
    """
    增强版 SubResearcher
    支持动态子问题生成和异步协作
    """
    
    def __init__(self, task_store, mailbox, artifact_store, board_store):
        self.task_store = task_store
        self.mailbox = mailbox
        self.artifact_store = artifact_store
        self.board_store = board_store
        self.question_generator = DynamicQuestionGenerator(self.model_client)
        self.collaboration_protocol = CollaborationProtocol(self.mailbox)
    
    async def _process_task_enhanced(self, task: Task) -> SubResearcherWorkResult:
        """增强版任务处理"""
        
        scratch = self._new_scratch(task)
        
        while scratch["iteration_count"] < self.max_iterations:
            scratch["iteration_count"] += 1
            
            # 1. 组装上下文（包括团队状态）
            context = await self._assemble_context_enhanced(task, scratch)
            
            # 2. 模型决策（更宽松的 action 空间）
            decision = await self._decide_action_enhanced(context)
            
            # 3. 验证和执行
            validated = self._validate_action_enhanced(context, decision)
            
            # 4. 根据 action 类型分发
            if validated["action"] in {"search", "fetch", "publish_raw_document"}:
                result = await self._execute_traditional(validated)
            elif validated["action"] in {"spawn_sub_question", "delegate_to_peer"}:
                result = await self._execute_collaborative(validated)
            elif validated["action"] == "replan":
                result = await self._execute_replan(validated, scratch)
            else:
                result = await self._execute_fallback(validated)
            
            # 5. 更新状态
            scratch = self._update_scratch(scratch, result, decision)
            
            # 6. 广播进度（异步）
            await self._broadcast_progress(task, scratch, decision)
            
            # 7. 检查是否完成
            if result.get("terminal"):
                return self._build_result(task, scratch, result)
    
    async def _decide_action_enhanced(self, context: dict) -> dict:
        """
        增强版决策
        支持更宽松的 action 空间
        """
        
        prompt = self._build_decision_prompt(context)
        
        # 使用更强的推理模型
        response = await self.model.complete(
            prompt,
            response_format={
                "type": "json_object",
                "schema": {
                    "action": "string",
                    "reasoning": "string",  # 推理过程
                    "confidence": "float",
                    "alternatives": ["action1", "action2"],  # 备选方案
                    "if_spawn_question": {"question": "string", "priority": "string"},
                    "if_delegate": {"target_role": "string", "task": "string"}
                }
            }
        )
        
        # 宽松的验证：记录但不强制 fallback
        if response.action not in ALL_ACTIONS:
            self.logger.warning(
                f"Unexpected action: {response.action}, "
                f"but proceeding with model reasoning"
            )
            # 记录异常但不 fallback，让模型决策生效
            self.metrics.record_unexpected_action(response.action)
        
        return {
            "action": response.action,
            "reasoning": response.reasoning,
            "confidence": response.confidence,
            "alternatives": response.alternatives,
            "metadata": response  # 原始响应
        }
    
    async def _execute_collaborative(self, decision: dict) -> dict:
        """执行协作类 action"""
        
        action = decision["action"]
        
        if action == "spawn_sub_question":
            # 生成并注册新问题
            question_data = decision.get("if_spawn_question", {})
            new_question = await self._spawn_sub_question(
                question=question_data.get("question"),
                priority=question_data.get("priority", "medium"),
                reason=decision.get("reasoning"),
                source_task=self.current_task
            )
            
            return {
                "terminal": False,
                "created_task_ids": [new_question.id],
                "created_message_ids": [],
                "action": "spawn_sub_question",
                "new_question": new_question
            }
        
        elif action == "delegate_to_peer":
            # 委托任务给其他 Agent
            delegation = decision.get("if_delegate", {})
            task_id = await self._delegate_to_peer(
                target_role=delegation.get("target_role", "sub_researcher"),
                task=delegation.get("task"),
                constraints=decision.get("constraints", {})
            )
            
            return {
                "terminal": False,
                "created_task_ids": [task_id],
                "created_message_ids": [],
                "action": "delegate_to_peer",
                "delegated_to": delegation.get("target_role")
            }
        
        elif action == "merge_questions":
            # 合并问题
            merged = await self._merge_questions(
                questions=decision.get("questions_to_merge", [])
            )
            
            return {
                "terminal": False,
                "action": "merge_questions",
                "merged_question": merged
            }
        
        elif action == "request_help":
            # 向 Lead 请求帮助
            help_request = await self._request_help(
                reason=decision.get("reasoning"),
                needed_capabilities=decision.get("needed_capabilities", [])
            )
            
            return {
                "terminal": False,
                "action": "request_help",
                "help_request_id": help_request.id
            }
    
    async def _broadcast_progress(
        self,
        task: Task,
        scratch: dict,
        decision: dict
    ):
        """异步广播进度"""
        
        await self.mailbox.send_async(
            Message(
                type="observation",
                broadcast=True,
                payload={
                    "kind": "enhanced_progress",
                    "source_task_id": task.task_id,
                    "round": scratch["iteration_count"],
                    "current_understanding": decision.get("current_understanding"),
                    "reasoning": decision.get("reasoning"),  # 新增：推理过程
                    "action": decision["action"],
                    "confidence": decision.get("confidence"),
                    "created_sub_questions": scratch.get("sub_questions_created", []),
                    "delegations": scratch.get("delegations", []),
                    "team_state": await self._get_team_state()
                },
                related_task_id=task.task_id
            ),
            delivery=DeliveryMode.QUEUED  # 异步，不阻塞
        )
    
    async def _get_team_state(self) -> dict:
        """获取团队状态（用于决策上下文）"""
        
        # 查询其他 Agent 的状态
        peer_states = []
        
        for agent_id in self._get_active_agents():
            agent_tasks = await self.task_store.list_tasks(agent_id)
            agent_progress = self._calculate_progress(agent_tasks)
            
            peer_states.append({
                "agent_id": agent_id,
                "active_tasks": len(agent_tasks),
                "progress": agent_progress,
                "current_focus": await self._get_agent_current_focus(agent_id)
            })
        
        return {
            "active_agents": len(peer_states),
            "peer_states": peer_states,
            "total_pending_tasks": await self.task_store.count_pending(),
            "collaboration_opportunities": self._identify_collab_opportunities(peer_states)
        }
```

---

## 6. 配置文件

```python
# config/enhanced_sub_researcher.py

ENHANCED_SUB_RESEARCHER_CONFIG = {
    # === Action 空间配置 ===
    "action_space": {
        "enabled_actions": [
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
            
            # 新增 actions（默认启用）
            "spawn_sub_question",      # 动态生成子问题
            "delegate_to_peer",        # 委托给其他 Agent
            "merge_questions",         # 合并问题
            "replan",                  # 重新规划
            "fetch_multiple",          # 批量获取
            "parallel_investigate",    # 并行调查
            "share_knowledge",         # 分享知识
            "request_help",            # 请求帮助
        ],
        
        # 是否允许未知 action（记录但不 fallback）
        "allow_unknown_actions": True,
        
        # 未知 action 的处理策略
        "unknown_action_strategy": "log_and_proceed",  # log_and_proceed | fallback | reject
    },
    
    # === 协作配置 ===
    "collaboration": {
        # 最大并行子问题数
        "max_parallel_sub_questions": 5,
        
        # 最大委托次数
        "max_delegations_per_task": 3,
        
        # 是否允许跨角色委托
        "cross_role_delegation": True,
        
        # 委托时是否需要同意
        "delegation_requires_ack": False,  # 异步委托，不需要立即确认
        
        # 最大协商轮次
        "max_negotiation_rounds": 5,
    },
    
    # === 推理配置 ===
    "reasoning": {
        # 是否包含推理过程
        "include_reasoning": True,
        
        # 是否包含备选方案
        "include_alternatives": True,
        
        # 最小置信度阈值
        "min_confidence_threshold": 0.5,
        
        # 是否允许低置信度 action
        "allow_low_confidence_actions": True,
    },
    
    # === 预算配置 ===
    "budget": {
        # 最大迭代次数（0 = 无限制）
        "max_iterations": 0,  # 改为 0，让模型自己决定
        
        # 是否启用自适应预算
        "adaptive_budget": True,
        
        # 自适应预算参数
        "adaptive_params": {
            "base_iterations": 12,
            "complexity_multiplier": 1.5,
            "low_confidence_threshold": 0.6,
            "high_complexity_threshold": 0.7
        }
    }
}
```

---

## 7. 测试场景

### 场景 1：动态子问题生成

```python
async def test_dynamic_sub_questions():
    """测试动态子问题生成"""
    
    # 场景：研究"DeepSeek 最新动态"
    # Agent 发现了 DeepSeek 官方博客上的融资新闻
    
    context = {
        "objective": "研究 DeepSeek 最新动态",
        "current_findings": [
            Finding(
                question="DeepSeek 融资情况",
                answer="发现 DeepSeek 获得新一轮融资",
                source="官方博客"
            )
        ],
        "model_client": MockModelClient()
    }
    
    generator = DynamicQuestionGenerator(context["model_client"])
    sub_questions = await generator.generate_sub_questions(
        objective=context["objective"],
        current_findings=context["current_findings"],
        research_context=ResearchContext(progress=30)
    )
    
    # 期望：自动生成追问
    assert any(
        "融资金额" in q.question or 
        "投资方" in q.question or
        "估值" in q.question
        for q in sub_questions
    )
    
    print(f"生成的子问题：{sub_questions}")
```

### 场景 2：异步协作

```python
async def test_async_collaboration():
    """测试异步协作"""
    
    mailbox = AsyncMailbox()
    
    # Agent A 提出提案
    proposal = Proposal(
        content="生成新的子问题：DeepSeek 融资金额",
        type="spawn_sub_question"
    )
    
    # 异步广播
    message_id = await mailbox.send_async(
        Message(
            type="proposal",
            payload={"proposal": proposal}
        ),
        delivery=DeliveryMode.QUEUED
    )
    
    # 订阅消息
    received = []
    await mailbox.subscribe(
        agent_id="agent_b",
        callback=lambda msg: received.append(msg)
    )
    
    # 处理消息
    await mailbox.process_queue()
    
    assert len(received) == 1
    assert received[0].payload["proposal"] == proposal
```

---

## 8. 迁移路径

### Phase 1: 最小改动（1天）
- 修改 `sub_researcher.py` 的 action 验证逻辑
- 添加 `spawn_sub_question` action
- 启用 `allow_unknown_actions=True`

### Phase 2: 核心功能（3天）
- 实现 `DynamicQuestionGenerator`
- 实现 `DynamicLeadProtocol`
- 修改 Lead 的任务分配逻辑

### Phase 3: 协作增强（1周）
- 实现 `AsyncMailbox`
- 实现 `CollaborationProtocol`
- 添加提案-投票机制

### Phase 4: 优化（持续）
- 添加学习机制
- 优化决策质量
- 添加监控和指标

---

## 9. 风险和缓解

### 风险 1：Agent 过度生成子问题
**缓解**：
- 设置 `max_parallel_sub_questions` 限制
- 添加重复检测
- 实施优先级衰减

### 风险 2：协作开销过大
**缓解**：
- 使用异步消息
- 限制协商轮次
- 自动投票（低优先级提案）

### 风险 3：决策质量下降
**缓解**：
- 保留 fallback 机制（可选）
- 添加置信度阈值
- 记录异常决策供分析
