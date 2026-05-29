"""
增强版 SubResearcher 演示脚本

这个脚本演示增强版功能如何工作：
1. 动态子问题生成
2. 扩展的 Action 空间
3. 异步协作

运行方式：
    python demo_enhanced.py

问题：DeepSeek 下一步的战略规划
"""

import asyncio
import json
from datetime import datetime, UTC
from enhanced_sub_researcher import (
    EnhancedSubResearcherWorker,
    EnhancedDecision,
    AsyncMailbox,
    DynamicQuestionGenerator,
    CollaborationProtocol,
    SubQuestion,
    Priority,
    DeliveryMode,
    EXTENDED_ACTIONS
)


class SimulatedModelClient:
    """
    模拟模型客户端
    模拟真实的 LLM 推理过程
    """
    
    def __init__(self):
        self.call_count = 0
        self.iteration = 0
    
    async def complete(self, messages, response_format=None):
        self.call_count += 1
        
        # 获取用户消息内容
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break
        
        # 提取迭代次数
        self.iteration = 0
        for line in user_content.split('\n'):
            if '当前轮次' in line:
                try:
                    self.iteration = int(line.split('：')[1].split('\n')[0].strip())
                except:
                    pass
        
        print(f"\n  🤖 模型推理 #{self.call_count} (迭代 #{self.iteration})")
        
        # 模拟模型根据上下文做出不同的决策
        if self.iteration == 1:
            # 第一轮：获取任务上下文
            print("     → 选择: query_task_context")
            return MockResponse({
                "action": "query_task_context",
                "reasoning": "需要先了解任务范围和目标",
                "confidence": 0.9
            })
        
        elif self.iteration == 2:
            # 第二轮：搜索
            print("     → 选择: search")
            return MockResponse({
                "action": "search",
                "reasoning": "收集 DeepSeek 相关的信息源",
                "confidence": 0.85,
                "query": "DeepSeek 战略规划 2024"
            })
        
        elif self.iteration == 3:
            # 第三轮：获取到结果，考虑生成子问题
            print("     → 选择: spawn_sub_question")
            print("     💡 增强版功能：动态生成子问题！")
            return MockResponse({
                "action": "spawn_sub_question",
                "reasoning": "发现了 DeepSeek 融资信息，生成追问",
                "confidence": 0.8,
                "if_spawn_question": {
                    "question": "DeepSeek 最新融资金额和估值是多少？",
                    "priority": "high",
                    "reason": "从搜索结果发现的新线索"
                }
            })
        
        elif self.iteration == 4:
            # 第四轮：委托给其他 agent
            print("     → 选择: delegate_to_peer")
            print("     💡 增强版功能：委托任务！")
            return MockResponse({
                "action": "delegate_to_peer",
                "reasoning": "将技术细节调查委托给专家 agent",
                "confidence": 0.75,
                "if_delegate": {
                    "target_role": "sub_researcher",
                    "task": "调查 DeepSeek 的技术路线和竞争优势",
                    "constraints": {"max_sources": 5}
                }
            })
        
        elif self.iteration == 5:
            # 第五轮：批量获取
            print("     → 选择: fetch_multiple")
            print("     💡 增强版功能：批量获取多个 URL！")
            return MockResponse({
                "action": "fetch_multiple",
                "reasoning": "并行获取多个信息源",
                "confidence": 0.7,
                "urls": ["url1", "url2", "url3"]
            })
        
        elif self.iteration == 6:
            # 第六轮：合并问题
            print("     → 选择: merge_questions")
            print("     💡 增强版功能：合并重复问题！")
            return MockResponse({
                "action": "merge_questions",
                "reasoning": "合并相似的子问题以提高效率",
                "confidence": 0.65
            })
        
        elif self.iteration == 7:
            # 第七轮：重新规划
            print("     → 选择: replan")
            print("     💡 增强版功能：动态重新规划！")
            return MockResponse({
                "action": "replan",
                "reasoning": "根据新发现调整研究方向",
                "confidence": 0.7
            })
        
        else:
            # 默认：完成
            print("     → 选择: complete")
            return MockResponse({
                "action": "complete",
                "reasoning": "已收集足够信息，可以总结",
                "confidence": 0.8
            })


class MockResponse:
    def __init__(self, data):
        self.json_data = data


class MockTaskStore:
    """模拟任务存储"""
    
    def __init__(self):
        self.tasks = []
        self.next_id = 1
    
    def create(self, run_id, kind, status, owner_role, inputs, priority, created_by):
        task = MockTask(
            task_id=f"task_{self.next_id}",
            run_id=run_id,
            kind=kind,
            status=status,
            owner_role=owner_role,
            inputs=inputs,
            priority=priority,
            created_by=created_by
        )
        self.tasks.append(task)
        self.next_id += 1
        return task
    
    def claim_next(self, run_id, owner_role):
        for task in self.tasks:
            if task.status == "pending" and task.owner_role == owner_role:
                task.status = "leased"
                return task
        return None
    
    def complete(self, task_id):
        for task in self.tasks:
            if task.task_id == task_id:
                task.status = "completed"
    
    def list_pending_tasks(self):
        return [t for t in self.tasks if t.status == "pending"]
    
    def get_swarm_run_state(self, run_id):
        return MockRunState(objective="DeepSeek 下一步的战略规划")


class MockTask:
    def __init__(self, task_id, run_id, kind, status, owner_role, inputs, priority, created_by):
        self.task_id = task_id
        self.run_id = run_id
        self.kind = kind
        self.status = status
        self.owner_role = owner_role
        self.inputs = inputs
        self.priority = priority
        self.created_by = created_by


class MockRunState:
    def __init__(self, objective):
        self.objective = objective


class MockArtifactStore:
    """模拟工件存储"""
    pass


async def run_demo():
    """运行演示"""
    
    print("="*70)
    print("  InsightSwarm 增强版演示")
    print("  问题：DeepSeek 下一步的战略规划")
    print("="*70)
    
    # 初始化组件
    model_client = SimulatedModelClient()
    task_store = MockTaskStore()
    artifact_store = MockArtifactStore()
    mailbox = AsyncMailbox()
    
    # 创建初始任务
    initial_task = task_store.create(
        run_id="demo_run",
        kind="research_subquestion",
        status="pending",
        owner_role="sub_researcher",
        inputs={
            "question": "DeepSeek 下一步的战略规划",
            "board_item_id": "root"
        },
        priority=10,
        created_by="lead"
    )
    
    print(f"\n📋 创建初始任务: {initial_task.task_id}")
    print(f"   问题: {initial_task.inputs['question']}")
    
    # 创建增强版 worker
    worker = EnhancedSubResearcherWorker(
        task_store=task_store,
        mailbox=mailbox,
        artifact_store=artifact_store,
        board_store=None,  # 简化演示
        model_client=model_client,
        config={
            "allow_unknown_actions": True,  # 允许未知 action
            "max_iterations": 0,           # 无限制
            "max_parallel_questions": 5
        }
    )
    
    print(f"\n🔧 配置:")
    print(f"   - 允许未知 Action: True")
    print(f"   - 最大迭代: 无限制")
    print(f"   - 最大并行子问题: 5")
    print(f"   - 扩展 Action 空间: {len(EXTENDED_ACTIONS)} 种")
    
    print("\n" + "-"*70)
    print("开始研究循环...")
    print("-"*70)
    
    # 运行研究循环
    iteration = 0
    max_iterations = 7  # 演示最多 7 轮
    
    while iteration < max_iterations:
        iteration += 1
        scratch_iteration = iteration  # 保存当前迭代次数用于模型推理
        
        print(f"\n{'='*70}")
        print(f"  第 {iteration} 轮迭代")
        print(f"{'='*70}")
        
        # 认领任务
        task = task_store.claim_next("demo_run", "sub_researcher")
        if not task:
            print("\n✅ 没有更多任务，演示结束")
            break
        
        print(f"\n📌 认领任务: {task.task_id}")
        print(f"   类型: {task.kind}")
        
        # 运行增强版 worker
        worker.current_task = task
        worker.scratch = worker._init_scratch(task)
        worker.scratch["iteration_count"] = scratch_iteration  # 设置迭代次数
        
        # 执行增强版决策
        context = await worker._assemble_context_enhanced(task)
        decision = await worker._decide_action_enhanced(context)
        
        print(f"\n🤔 决策结果:")
        print(f"   Action: {decision.action}")
        print(f"   推理: {decision.reasoning}")
        print(f"   置信度: {decision.confidence:.0%}")
        
        if decision.action == "spawn_sub_question":
            print(f"   ✨ 生成的子问题: {decision.spawned_questions[0].question if decision.spawned_questions else 'N/A'}")
        
        if decision.action == "delegate_to_peer" and decision.delegation:
            print(f"   ✨ 委托给: {decision.delegation.to_role}")
            print(f"   ✨ 委托任务: {decision.delegation.task}")
        
        # 验证决策
        validated = worker._validate_action_enhanced(context, decision)
        
        # 执行决策
        result = await worker._execute_decision(validated)
        
        # 广播进度
        await worker._broadcast_progress(task, decision, result)
        
        # 检查是否完成
        if result.get("terminal"):
            print(f"\n✅ 任务完成: {task.task_id}")
            task_store.complete(task.task_id)
            break
        
        # 限制迭代次数
        if iteration >= 7:
            print(f"\n⏹️ 达到演示迭代上限")
            break
        
        task_store.complete(task.task_id)
    
    # 显示统计
    print("\n" + "="*70)
    print("  演示统计")
    print("="*70)
    
    print(f"\n📊 执行统计:")
    print(f"   总迭代次数: {iteration}")
    print(f"   模型调用次数: {model_client.call_count}")
    print(f"   生成子问题: {len(worker.metrics['spawned_questions'])}")
    print(f"   委托任务: {len(worker.metrics['delegations'])}")
    print(f"   未知 Action: {len(worker.metrics['unexpected_actions'])}")
    
    print(f"\n🎯 Action 分布:")
    action_counts = {}
    for decision in worker.scratch.get("decision_history", []):
        action = decision.get("action", "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
    
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
        marker = "✨" if action in [
            "spawn_sub_question", "delegate_to_peer", 
            "fetch_multiple", "merge_questions", "replan"
        ] else "  "
        print(f"   {marker} {action}: {count}")
    
    print("\n" + "="*70)
    print("  演示完成！")
    print("="*70)
    
    print("\n💡 增强版功能总结:")
    print("   1. ✨ 扩展的 Action 空间 (9 → 23 种)")
    print("   2. ✨ 动态子问题生成 - 根据发现自动创建新问题")
    print("   3. ✨ 任务委托 - 将部分工作委托给其他 Agent")
    print("   4. ✨ 批量获取 - 并行获取多个信息源")
    print("   5. ✨ 异步协作 - 非阻塞的团队通信")
    print("   6. ✨ 宽松验证 - 允许模型创新决策")
    
    print("\n📝 下一步:")
    print("   1. 配置真实的 API keys (TAVILY_API_KEY, DASHSCOPE_API_KEY)")
    print("   2. 集成到主项目:")
    print("      from enhanced_sub_researcher import EnhancedSubResearcherWorker")
    print("   3. 运行真实研究任务")


async def run_action_space_demo():
    """演示扩展的 Action 空间"""
    
    print("\n" + "="*70)
    print("  扩展的 Action 空间")
    print("="*70)
    
    categories = {
        "📋 传统类 (9)": [
            "query_task_context", "query_mailbox", "query_evidence",
            "search", "fetch", "publish_raw_document", "suggest_browser",
            "complete", "blocked"
        ],
        "🎯 研究策略类 (4) ✨": [
            "spawn_sub_question", "merge_questions", "split_question", "replan"
        ],
        "🤝 协作类 (4) ✨": [
            "delegate_to_peer", "request_help", "propose_collaboration", "share_knowledge"
        ],
        "⚡ 执行类 (3) ✨": [
            "fetch_multiple", "analyze_and_search", "parallel_investigate"
        ],
        "🧠 元认知类 (3) ✨": [
            "reflect", "request_feedback", "adjust_priority"
        ]
    }
    
    for category, actions in categories.items():
        print(f"\n{category}:")
        for action in actions:
            marker = "✨" if "✨" in category else "  "
            desc = get_action_description(action)
            print(f"   {marker} {action:25s} - {desc}")
    
    print(f"\n总计: 9 (原有) + 14 (新增) = {len(EXTENDED_ACTIONS)} 种 Action")


def get_action_description(action: str) -> str:
    """获取 Action 描述"""
    descriptions = {
        "query_task_context": "查询任务上下文",
        "query_mailbox": "查询消息队列",
        "query_evidence": "查询已有证据",
        "search": "搜索信息源",
        "fetch": "获取 URL 内容",
        "publish_raw_document": "发布原始文档",
        "suggest_browser": "建议使用浏览器",
        "complete": "完成任务",
        "blocked": "任务阻塞",
        "spawn_sub_question": "✨ 动态生成子问题",
        "merge_questions": "✨ 合并相关问题",
        "split_question": "✨ 拆分复杂问题",
        "replan": "✨ 重新制定计划",
        "delegate_to_peer": "✨ 委托给其他 Agent",
        "request_help": "✨ 请求帮助",
        "propose_collaboration": "✨ 提议协作",
        "share_knowledge": "✨ 分享知识",
        "fetch_multiple": "✨ 批量获取多个 URL",
        "analyze_and_search": "✨ 搜索后分析",
        "parallel_investigate": "✨ 并行调查",
        "reflect": "✨ 反思当前策略",
        "request_feedback": "✨ 请求反馈",
        "adjust_priority": "✨ 调整优先级"
    }
    return descriptions.get(action, "")


async def main():
    """主函数"""
    
    print("\n" + "="*70)
    print("  InsightSwarm 增强版功能演示")
    print("  问题：DeepSeek 下一步的战略规划")
    print("="*70)
    
    # 演示 1: Action 空间
    await run_action_space_demo()
    
    # 演示 2: 完整研究循环
    await run_demo()
    
    print("\n" + "="*70)
    print("  所有演示完成！")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
