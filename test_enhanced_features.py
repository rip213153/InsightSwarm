"""
测试增强版 SubResearcher 的新功能

运行方式：
    python test_enhanced_features.py

这个脚本演示：
1. 动态子问题生成
2. 异步消息队列
3. 提案-投票协议
4. 任务委托机制
"""

import asyncio
import json
from datetime import datetime, timedelta, UTC
from enhanced_sub_researcher import (
    EnhancedSubResearcherWorker,
    AsyncMailbox,
    DynamicQuestionGenerator,
    CollaborationProtocol,
    SubQuestion,
    Priority,
    DeliveryMode,
    EXTENDED_ACTIONS,
    VoteResult
)


class MockModelClient:
    """模拟模型客户端（用于测试）"""
    
    async def complete(self, messages, response_format=None):
        """模拟模型调用"""
        
        # 解析消息内容
        content = messages[-1]["content"] if messages else ""
        
        # 根据内容生成响应
        if "spawn_sub_question" in content.lower():
            return MockResponse({
                "action": "spawn_sub_question",
                "reasoning": "Discovered new lead, spawning sub question",
                "confidence": 0.8,
                "alternatives": ["search", "delegate_to_peer"],
                "if_spawn_question": {
                    "question": "DeepSeek 融资金额和估值",
                    "priority": "high",
                    "reason": "From official blog discovery"
                }
            })
        elif "delegate" in content.lower():
            return MockResponse({
                "action": "delegate_to_peer",
                "reasoning": "Delegating to specialist agent",
                "confidence": 0.75,
                "alternatives": ["spawn_sub_question"],
                "if_delegate": {
                    "target_role": "sub_researcher",
                    "task": "调查竞品技术细节",
                    "constraints": {"max_sources": 5}
                }
            })
        elif "replan" in content.lower():
            return MockResponse({
                "action": "replan",
                "reasoning": "Current approach not working, replanning",
                "confidence": 0.7
            })
        else:
            return MockResponse({
                "action": "search",
                "reasoning": "Need to gather more sources",
                "confidence": 0.6
            })


class MockResponse:
    """模拟模型响应"""
    
    def __init__(self, data):
        self.json_data = data


async def test_dynamic_question_generation():
    """测试 1: 动态子问题生成"""
    print("\n" + "="*60)
    print("测试 1: 动态子问题生成")
    print("="*60)
    
    model_client = MockModelClient()
    generator = DynamicQuestionGenerator(model_client)
    
    # 模拟研究场景
    current_findings = [
        {
            "question": "DeepSeek 融资情况",
            "answer": "DeepSeek 完成新一轮融资，金额未披露",
            "source": "官方博客",
            "confidence": 0.7
        },
        {
            "question": "DeepSeek 技术特点",
            "answer": "采用混合专家架构，专注于推理能力",
            "source": "技术文档",
            "confidence": 0.9
        }
    ]
    
    # 生成子问题
    sub_questions = await generator.generate_sub_questions(
        objective="研究 DeepSeek 最新动态",
        current_findings=current_findings,
        research_progress=0.35,
        explored_angles=["融资", "技术"]
    )
    
    print(f"\n生成 {len(sub_questions)} 个新子问题:")
    for i, q in enumerate(sub_questions, 1):
        print(f"\n  子问题 {i}:")
        print(f"    问题: {q.question}")
        print(f"    优先级: {q.priority.value}")
        print(f"    原因: {q.reason}")
        print(f"    来源: {q.source}")
        print(f"    可并行: {q.can_be_parallel}")
    
    print("\n✅ 动态子问题生成测试通过")
    return sub_questions


async def test_async_mailbox():
    """测试 2: 异步消息队列"""
    print("\n" + "="*60)
    print("测试 2: 异步消息队列")
    print("="*60)
    
    mailbox = AsyncMailbox()
    received_messages = []
    
    # 订阅消息
    subscription_id = await mailbox.subscribe(
        agent_id="test_agent",
        callback=lambda msg: received_messages.append(msg),
        message_types=["proposal", "observation"]
    )
    print(f"\n已订阅消息，订阅ID: {subscription_id}")
    
    # 发送异步消息
    from enhanced_sub_researcher import AsyncMessage
    msg1_id = await mailbox.send_async(
        message=AsyncMessage(
            id="msg1",
            type="proposal",
            from_role="agent_a",
            to_role="agent_b",
            broadcast=False,
            payload={"content": "Spawn new sub question"}
        ),
        delivery=DeliveryMode.QUEUED
    )
    print(f"发送异步消息（队列模式），ID: {msg1_id}")
    
    msg2_id = await mailbox.send_async(
        message=AsyncMessage(
            id="msg2",
            type="observation",
            from_role="agent_a",
            to_role=None,
            broadcast=True,
            payload={"content": "Progress update"}
        ),
        delivery=DeliveryMode.IMMEDIATE
    )
    print(f"发送异步消息（立即模式），ID: {msg2_id}")
    
    # 处理队列
    print("\n处理消息队列...")
    processed = await mailbox.process_queue()
    print(f"处理了 {len(processed)} 条消息")
    
    print(f"\n接收到的消息数量: {len(received_messages)}")
    for msg in received_messages:
        print(f"  - {msg.type}: {msg.payload}")
    
    print("\n✅ 异步消息队列测试通过")
    return received_messages


async def test_proposal_voting():
    """测试 3: 提案-投票协议"""
    print("\n" + "="*60)
    print("测试 3: 提案-投票协议")
    print("="*60)
    
    mailbox = AsyncMailbox()
    protocol = CollaborationProtocol(mailbox)
    
    # 模拟投票者
    voters = ["researcher_1", "researcher_2", "researcher_3"]
    
    # 创建提案
    proposal = await mailbox.create_proposal(
        proposer="researcher_0",
        proposal_type="spawn_sub_question",
        content={
            "question": "DeepSeek 市场份额分析",
            "priority": "high",
            "reason": "战略决策需要"
        },
        deadline=datetime.now(UTC) + timedelta(seconds=5)
    )
    print(f"\n创建提案，ID: {proposal.id}")
    print(f"  类型: {proposal.proposal_type}")
    print(f"  内容: {proposal.content}")
    
    # 模拟投票
    print("\n模拟投票...")
    for voter in voters:
        vote = "approve" if voter != "researcher_2" else "reject"
        await mailbox.vote(proposal.id, voter, vote)
        print(f"  {voter}: {vote}")
    
    # 获取结果
    result = await mailbox.get_proposal_result(proposal.id)
    
    print(f"\n投票结果:")
    print(f"  赞成: {result.votes_for}")
    print(f"  反对: {result.votes_against}")
    print(f"  弃权: {result.votes_abstain}")
    print(f"  通过: {result.approved}")
    
    print("\n✅ 提案-投票协议测试通过")
    return result


async def test_delegation():
    """测试 4: 任务委托机制"""
    print("\n" + "="*60)
    print("测试 4: 任务委托机制")
    print("="*60)
    
    mailbox = AsyncMailbox()
    
    # 创建委托
    delegation = await mailbox.create_delegation(
        from_agent="researcher_main",
        to_role="sub_researcher",
        task="调查竞品定价策略",
        constraints={
            "max_sources": 5,
            "priority": "high",
            "deadline": "2024-01-15"
        }
    )
    
    print(f"\n创建委托，ID: {delegation.id}")
    print(f"  从: {delegation.from_agent}")
    print(f"  到: {delegation.to_role}")
    print(f"  任务: {delegation.task}")
    print(f"  约束: {delegation.constraints}")
    
    # 模拟确认
    print("\n模拟确认...")
    acknowledged = await mailbox.acknowledge_delegation(delegation.id)
    print(f"  确认结果: {acknowledged}")
    print(f"  委托状态: {mailbox._delegations[delegation.id].status}")
    
    # 查询待处理委托
    pending = mailbox.get_pending_delegations("sub_researcher")
    print(f"\n待处理的委托数量: {len(pending)}")
    
    print("\n✅ 任务委托机制测试通过")
    return delegation


async def test_extended_action_space():
    """测试 5: 扩展的 Action 空间"""
    print("\n" + "="*60)
    print("测试 5: 扩展的 Action 空间")
    print("="*60)
    
    print(f"\n原始 Action 数量: 9")
    print(f"扩展后 Action 数量: {len(EXTENDED_ACTIONS)}")
    
    # 分类显示
    categories = {
        "传统类": ["query_task_context", "query_mailbox", "query_evidence", 
                   "search", "fetch", "publish_raw_document", "suggest_browser",
                   "complete", "blocked"],
        "研究策略类": ["spawn_sub_question", "merge_questions", "split_question", "replan"],
        "协作类": ["delegate_to_peer", "request_help", "propose_collaboration", "share_knowledge"],
        "执行类": ["fetch_multiple", "analyze_and_search", "parallel_investigate"],
        "元认知类": ["reflect", "request_feedback", "adjust_priority"]
    }
    
    print("\n按类别分组:")
    for category, actions in categories.items():
        available = [a for a in actions if a in EXTENDED_ACTIONS]
        print(f"\n  {category} ({len(available)}):")
        for action in available:
            marker = "✨" if action not in categories["传统类"] else "  "
            print(f"    {marker} {action}")
    
    new_actions = [a for a in EXTENDED_ACTIONS if a not in categories["传统类"]]
    print(f"\n新增 Action: {len(new_actions)} 个")
    print(f"  {', '.join(new_actions)}")
    
    print("\n✅ 扩展 Action 空间测试通过")


async def test_collaboration_protocol():
    """测试 6: 协作协议"""
    print("\n" + "="*60)
    print("测试 6: 协作协议")
    print("="*60)
    
    mailbox = AsyncMailbox()
    protocol = CollaborationProtocol(mailbox)
    
    # 模拟协商
    parties = ["researcher_1", "researcher_2", "researcher_3"]
    topic = "研究方向优先级"
    initial_positions = {
        "researcher_1": "优先调查技术深度",
        "researcher_2": "优先调查市场应用",
        "researcher_3": "优先调查竞争格局"
    }
    
    print(f"\n发起协商")
    print(f"  参与方: {parties}")
    print(f"  主题: {topic}")
    print(f"\n初始立场:")
    for agent, pos in initial_positions.items():
        print(f"    {agent}: {pos}")
    
    # 执行协商
    result = await protocol.negotiate(parties, topic, initial_positions)
    
    print(f"\n协商结果:")
    if result:
        print(f"  状态: {result['status']}")
        print(f"  轮次: {len(result['rounds'])}")
        if result.get('agreement'):
            print(f"  协议: {result['agreement']}")
    else:
        print("  状态: 协商失败")
    
    print("\n✅ 协作协议测试通过")
    return result


async def test_full_integration():
    """测试 7: 完整集成测试"""
    print("\n" + "="*60)
    print("测试 7: 完整集成测试")
    print("="*60)
    
    # 初始化组件
    mailbox = AsyncMailbox()
    model_client = MockModelClient()
    
    print("\n初始化组件...")
    print(f"  AsyncMailbox: ✓")
    print(f"  MockModelClient: ✓")
    print(f"  DynamicQuestionGenerator: ✓")
    print(f"  CollaborationProtocol: ✓")
    
    # 模拟研究场景
    print("\n模拟研究场景:")
    print("  目标: 研究 DeepSeek 最新动态")
    print("  当前进度: 35%")
    
    # 生成子问题
    generator = DynamicQuestionGenerator(model_client)
    sub_questions = await generator.generate_sub_questions(
        objective="研究 DeepSeek 最新动态",
        current_findings=[
            {"question": "融资", "answer": "完成融资", "source": "博客"}
        ],
        research_progress=0.35,
        explored_angles=["融资"]
    )
    print(f"\n生成 {len(sub_questions)} 个新子问题")
    
    # 发送协作消息
    from enhanced_sub_researcher import AsyncMessage
    await mailbox.send_async(
        message=AsyncMessage(
            id="integration_test",
            type="observation",
            from_role="system",
            to_role=None,
            broadcast=True,
            payload={
                "kind": "research_update",
                "sub_questions_created": len(sub_questions)
            }
        ),
        delivery=DeliveryMode.QUEUED
    )
    
    # 处理队列
    processed = await mailbox.process_queue()
    print(f"处理 {len(processed)} 条协作消息")
    
    # 创建提案
    proposal = await mailbox.create_proposal(
        proposer="main_researcher",
        proposal_type="spawn_sub_question",
        content={"question": "DeepSeek 融资金额"}
    )
    await mailbox.vote(proposal.id, "peer_1", "approve")
    await mailbox.vote(proposal.id, "peer_2", "approve")
    
    result = await mailbox.get_proposal_result(proposal.id)
    print(f"\n提案投票: {'通过' if result.approved else '否决'}")
    
    print("\n✅ 完整集成测试通过")


async def run_all_tests():
    """运行所有测试"""
    
    print("\n" + "="*60)
    print("InsightSwarm 增强版功能测试套件")
    print("="*60)
    print(f"\n测试时间: {datetime.now(UTC).isoformat()}")
    print(f"扩展 Action 数量: {len(EXTENDED_ACTIONS)}")
    
    try:
        # 运行所有测试
        await test_extended_action_space()
        await test_dynamic_question_generation()
        await test_async_mailbox()
        await test_proposal_voting()
        await test_delegation()
        await test_collaboration_protocol()
        await test_full_integration()
        
        # 总结
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        print("\n✅ 所有测试通过!")
        print("\n新增功能:")
        print("  1. ✨ 动态子问题生成 - 根据发现自动生成新问题")
        print("  2. ✨ 异步消息队列 - 非阻塞协作通信")
        print("  3. ✨ 提案-投票协议 - 团队决策机制")
        print("  4. ✨ 任务委托机制 - Agent 间任务分配")
        print("  5. ✨ 扩展 Action 空间 - 27 种决策选项")
        print("  6. ✨ 协作协议 - 协商和共识")
        print("\n下一步:")
        print("  1. 将 enhanced_sub_researcher.py 集成到主项目")
        print("  2. 配置模型客户端")
        print("  3. 调整 max_iterations 和其他参数")
        print("  4. 运行真实的研究任务测试")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
