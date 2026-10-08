"""Agent 运行时验证 —— 11 个场景，验证状态机与三个硬约束

覆盖：
  ① 正常任务：调工具 → 观察 → 给答案（含步状态是否正确落盘）
  ② 死循环：反复调同一工具 → 被检测 → 优雅收尾
  ②b 交替循环：A,B,A,B... → 被检测（**原实现漏判**：判定条件只看"窗口内签名是否全同"，
      只要出现第二种签名就永远不成立，于是这种循环会一路跑满 max_steps）+ 反向验证不误判
  ③ 步数超限：一直调不同工具 → 达上限 → 优雅收尾
  ④ 预算超限：token 累积超预算 → 优雅收尾
  ⑤ 越权工具：调用被网关拒绝 → 拒绝原因作为观察反馈给模型
  ⑥ 高危工具：触发审批 → 提示勿重复提交 → 模型给阶段性结论
  ⑦ 模型非法输出：既无动作也无答案 → 标记异常
  ⑦b 动作格式非法：给了动作但结构不对（字符串 / tool 为 null）→ usage_error 而非崩溃
  ⑦c 参数下界校验：loop_threshold / max_steps / token_budget 配错 → 构造时 fail-fast
  ⑧ 轨迹落盘与统计：可回放、可算指标

运行：python tests/test_runtime.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from brain import ScriptedBrain  # noqa: E402
from builtin_tools import ALL_TOOLS, ROLE_ADMIN, ROLE_OPERATOR, ROLE_READER  # noqa: E402
from gateway import ApprovalStore, AuditStore, IdempotencyLedger, ToolGateway  # noqa: E402
from registry import ToolRegistry  # noqa: E402
from runtime import AgentRuntime, FinishReason, TraceStore  # noqa: E402

AUDIT_PATH = os.path.join(ROOT, "data", "audit.jsonl")
TRACE_PATH = os.path.join(ROOT, "data", "traces.jsonl")

results = []


def check(name: str, ok: bool, detail: str = ""):
    results.append((name, ok))
    print(f"  {'✅' if ok else '❌'} {name}" + (f" —— {detail}" if detail else ""))


def section(title: str):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def build_gateway(audit: AuditStore) -> ToolGateway:
    registry = ToolRegistry()
    registry.register_all(ALL_TOOLS)
    return ToolGateway(registry, audit, ApprovalStore(), IdempotencyLedger())


def main() -> int:
    audit = AuditStore(AUDIT_PATH)
    audit.clear()
    trace = TraceStore(TRACE_PATH)
    trace.clear()
    gw = build_gateway(audit)

    # ------------------------------------------------------------
    section("① 正常任务：查订单 → 观察 → 作答")
    brain = ScriptedBrain(
        [
            {"thought": "先查订单状态", "action": {"tool": "query_order", "args": {"order_id": "ORD-000123"}}},
            {"thought": "拿到结果了", "answer": "订单 ORD-000123 状态为已支付，金额 1280 元。"},
        ]
    )
    rt = AgentRuntime(brain, gw, max_steps=8, token_budget=5000, trace_store=trace)
    run = rt.run("查询 ORD-000123 的订单状态", role=ROLE_READER, task_id="T-normal")
    check("任务完成", run.completed, f"status={run.status}")
    check("结束原因为 answered", run.finish_reason == FinishReason.ANSWERED.value, run.finish_reason)
    check("产生了 1 次工具调用", run.tool_calls == 1, f"tool_calls={run.tool_calls}")
    check("轨迹包含思考+动作", run.step_count == 2, f"steps={run.step_count}")
    check("最终答案非空", bool(run.final_answer), (run.final_answer or "")[:40])
    # 步状态落盘：原来 AgentState 定义了 5 个状态，但只写入 THINKING 一个，
    # DONE / OBSERVING 从没被赋值过 —— 轨迹回放时看不出每步在做什么。
    check("步状态落盘：调完工具那步是 observing", run.steps[0].state == "observing", run.steps[0].state)
    check("步状态落盘：给出答案那步是 done", run.steps[1].state == "done", run.steps[1].state)

    # ------------------------------------------------------------
    section("② 死循环：反复调同一工具 → 被检测并优雅收尾")
    loop_brain = ScriptedBrain(
        [
            {"thought": "查订单", "action": {"tool": "query_order", "args": {"order_id": "ORD-000123"}}},
        ],
        repeat_last=True,  # 永远重复这一步 → 模拟卡死
    )
    rt = AgentRuntime(loop_brain, gw, max_steps=10, token_budget=99999, loop_threshold=3, trace_store=trace)
    run = rt.run("查询订单", role=ROLE_READER, task_id="T-loop")
    check("检测到死循环", run.finish_reason == FinishReason.LOOP_DETECTED.value, run.finish_reason)
    check("没有跑满 max_steps 就中止", run.step_count < 10, f"steps={run.step_count}")
    check("仍然产出了收尾结论（优雅降级）", bool(run.final_answer), (run.final_answer or "")[:50])
    check("状态为 aborted", run.status == "aborted", run.status)

    # ------------------------------------------------------------
    section("②b 交替循环：A,B,A,B...（原实现漏判的场景）")
    # 原实现的条件是 `len(set(recent_calls)) == 1` ——
    # 只要窗口里出现第二种签名，条件就永远不成立。
    # 于是一个"在两个工具之间来回摇摆、什么都没推进"的 Agent
    # 会一路跑满 max_steps 才停，中间烧掉的调用和 token 全是白费。
    # 这里用「同一个工具 + 两个不同订单号」交替，构造严格周期序列。
    alt_steps = []
    for _ in range(4):
        alt_steps.append(
            {
                "thought": "查 A 订单",
                "action": {"tool": "query_order", "args": {"order_id": "ORD-000123"}},
            }
        )
        alt_steps.append(
            {
                "thought": "查 B 订单",
                "action": {"tool": "query_order", "args": {"order_id": "ORD-000124"}},
            }
        )
    rt = AgentRuntime(
        ScriptedBrain(alt_steps),
        gw,
        max_steps=10,
        token_budget=99999,
        loop_threshold=3,
        trace_store=trace,
    )
    run_alt = rt.run("来回查两个订单", role=ROLE_READER, task_id="T-alt-loop")
    check(
        "交替循环（A,B,A,B）被检测到",
        run_alt.finish_reason == FinishReason.LOOP_DETECTED.value,
        run_alt.finish_reason,
    )
    check("交替循环没跑满 max_steps 就中止", run_alt.step_count < 10, f"steps={run_alt.step_count}")
    check("交替循环仍产出收尾结论", bool(run_alt.final_answer), (run_alt.final_answer or "")[:40])

    # 反向验证：正常的多步任务不能被误判成循环
    # （只有在"严格周期重复"时才判——判太宽松会把正常任务误杀）
    distinct_steps = [
        {
            "thought": f"查第 {i} 个订单",
            "action": {"tool": "query_order", "args": {"order_id": f"ORD-0000{i:02d}"}},
        }
        for i in range(1, 7)
    ] + [{"thought": "信息够了", "answer": "已汇总 6 个订单。"}]
    rt = AgentRuntime(
        ScriptedBrain(distinct_steps, repeat_last=False),
        gw,
        max_steps=12,
        token_budget=99999,
        loop_threshold=3,
        trace_store=trace,
    )
    run_distinct = rt.run("批量查询 6 个订单", role=ROLE_READER, task_id="T-no-false-loop")
    check(
        "6 次互不相同的调用不会被误判为循环",
        run_distinct.finish_reason == FinishReason.ANSWERED.value,
        run_distinct.finish_reason,
    )
    check("6 次调用全部执行完", run_distinct.tool_calls == 6, f"tool_calls={run_distinct.tool_calls}")

    # ------------------------------------------------------------
    section("③ 步数超限：一直调不同工具")
    many_calls = [
        {"thought": f"第{i}次查询", "action": {"tool": "query_order", "args": {"order_id": f"ORD-0000{i:02d}"}}}
        for i in range(1, 9)
    ]
    rt = AgentRuntime(ScriptedBrain(many_calls), gw, max_steps=4, token_budget=99999, trace_store=trace)
    run = rt.run("连续查询多个订单", role=ROLE_READER, task_id="T-maxsteps")
    check("触发步数上限", run.finish_reason == FinishReason.MAX_STEPS.value, run.finish_reason)
    check("步数记录不超过 max_steps + 收尾步", run.step_count <= 5, f"steps={run.step_count}")
    check("仍产出收尾结论", bool(run.final_answer), (run.final_answer or "")[:50])

    # ------------------------------------------------------------
    section("④ 预算超限：token 累积触发")
    budget_brain = ScriptedBrain(
        [{"thought": "持续查询", "action": {"tool": "query_order", "args": {"order_id": "ORD-000123"}}}],
        tokens_per_step=200,  # 每步 200 token
    )
    rt = AgentRuntime(budget_brain, gw, max_steps=20, token_budget=500, trace_store=trace)
    run = rt.run("大量查询", role=ROLE_READER, task_id="T-budget")
    check("触发预算上限", run.finish_reason == FinishReason.TOKEN_BUDGET.value, run.finish_reason)
    check("token 消耗被控制在预算附近", run.total_tokens <= 900, f"tokens={run.total_tokens}")

    # ------------------------------------------------------------
    section("⑤ 越权工具：拒绝原因作为观察反馈给模型")
    authz_brain = ScriptedBrain(
        [
            {"thought": "尝试退款", "action": {"tool": "issue_refund",
                                              "args": {"order_id": "ORD-000123", "amount": 50}}},
            {"thought": "被拒了，改用只读工具", "action": {"tool": "query_order",
                                                          "args": {"order_id": "ORD-000123"}}},
            {"thought": "拿到信息，作答", "answer": "我没有退款权限，但订单 ORD-000123 状态为已支付。"},
        ]
    )
    rt = AgentRuntime(authz_brain, gw, max_steps=8, token_budget=5000, trace_store=trace)
    run = rt.run("帮我退款并确认订单状态", role=ROLE_READER, task_id="T-authz")
    check("越权调用被记录为 denied", run.denied_calls == 1, f"denied={run.denied_calls}")
    check("模型收到拒绝后继续推进并完成", run.completed, f"reason={run.finish_reason}")
    denied_step = next((s for s in run.steps if s.tool_status == "denied"), None)
    # 断言用【错误码】而不是自然语言关键词：文案会改，错误码是稳定契约
    check("观察里带回了明确的拒绝原因（错误码）",
          bool(denied_step and "[permission_denied]" in (denied_step.observation or "")),
          (denied_step.observation if denied_step else "")[:70])

    # ------------------------------------------------------------
    section("⑥ 高危工具：触发审批 → 提示勿重复提交")
    risky_brain = ScriptedBrain(
        [
            {"thought": "发起退款", "action": {"tool": "issue_refund",
                                              "args": {"order_id": "ORD-000777", "amount": 3000}}},
        ],
        repeat_last=True,
    )
    rt = AgentRuntime(risky_brain, gw, max_steps=6, token_budget=99999, loop_threshold=2, trace_store=trace)
    run = rt.run("给 ORD-000777 退款 3000 元", role=ROLE_OPERATOR, task_id="T-approval")
    check("产生了待审批调用", run.pending_approvals >= 1, f"pending={run.pending_approvals}")
    apv_step = next((s for s in run.steps if s.tool_status == "pending_approval"), None)
    check("观察里提示了审批单号且要求勿重复提交",
          bool(apv_step and "approval_id" in (apv_step.observation or "")),
          (apv_step.observation if apv_step else "")[:70])
    check("重复调用被幂等/循环检测拦住（没有创建多张审批单）",
          run.pending_approvals <= 2, f"pending={run.pending_approvals}")

    # ------------------------------------------------------------
    section("⑦ 模型非法输出：既无动作也无答案")
    bad_brain = ScriptedBrain([{"thought": "我什么都不做"}])
    rt = AgentRuntime(bad_brain, gw, max_steps=5, token_budget=5000, trace_store=trace)
    run = rt.run("模糊任务", role=ROLE_READER, task_id="T-bad")
    check("标记为 no_action", run.finish_reason == FinishReason.NO_ACTION.value, run.finish_reason)

    # ------------------------------------------------------------
    section("⑦b 动作格式非法：不能崩，要优雅收尾")
    # 原实现直接 `decision.action.get("tool", "")` ——
    # 只要 action 不是 dict（真实 LLM 完全可能吐出字符串），
    # 就会抛 AttributeError 把整个任务崩掉，而不是优雅收尾。
    # 注意这和 ⑦ 是两种不同的失败：⑦ 是"什么都没给"，这里是"给了但格式不对"。
    fmt_brain = ScriptedBrain([{"thought": "直接给个字符串", "action": "query_order"}], repeat_last=False)
    rt = AgentRuntime(fmt_brain, gw, max_steps=6, token_budget=99999, trace_store=trace)
    run_fmt = rt.run("非法动作", role=ROLE_READER, task_id="T-usage-error")
    check(
        "action 是字符串 → 判 usage_error（而不是抛 AttributeError 崩掉）",
        run_fmt.finish_reason == FinishReason.USAGE_ERROR.value,
        run_fmt.finish_reason,
    )
    check("非法动作也走优雅收尾（仍产出结论）", bool(run_fmt.final_answer), (run_fmt.final_answer or "")[:40])
    check("非法动作没有真的执行任何工具", run_fmt.tool_calls == 0, f"tool_calls={run_fmt.tool_calls}")

    # 另一种形态：action 是 dict，但 tool 字段是空的
    null_brain = ScriptedBrain([{"thought": "tool 是空的", "action": {"tool": None}}], repeat_last=False)
    rt = AgentRuntime(null_brain, gw, max_steps=6, token_budget=99999, trace_store=trace)
    run_null = rt.run("tool 为空", role=ROLE_READER, task_id="T-usage-error2")
    check(
        "tool 字段为空 → 同样判 usage_error",
        run_null.finish_reason == FinishReason.USAGE_ERROR.value,
        run_null.finish_reason,
    )

    # ------------------------------------------------------------
    section("⑦c 参数下界校验：配错时必须立刻报错，而不是静默做错事")
    # 这两个参数配错时不会崩，只会**静默地做错事**，而且错的方向很隐蔽：
    #   loop_threshold=1 → 一次正常调用就被判成"连续 1 次相同" →
    #       正常任务被误杀，steps=2 且**工具一次都没执行**
    #   loop_threshold=0 → deque(maxlen=0) 窗口恒空 → **循环检测静默失效**，
    #       无限循环要跑满 max_steps 才被兜住
    # 两种都是实测出来的。静默失败比崩溃更危险（崩溃至少能被发现），
    # 所以正解是构造时 fail-fast。
    def _rejects(**kwargs) -> bool:
        try:
            AgentRuntime(
                brain=ScriptedBrain([{"thought": "x", "answer": "y"}]), gateway=gw, **kwargs
            )
            return False
        except ValueError:
            return True

    check(
        "loop_threshold=1 → 直接报错（否则正常任务被判死循环、工具一次都不执行）",
        _rejects(loop_threshold=1),
    )
    check(
        "loop_threshold=0 → 直接报错（否则窗口恒空、循环检测静默失效）",
        _rejects(loop_threshold=0),
    )
    check("loop_threshold=2 合法（下界）", not _rejects(loop_threshold=2))
    check("max_steps=1 → 直接报错", _rejects(max_steps=1))
    check("token_budget=0 → 直接报错", _rejects(token_budget=0))

    # ------------------------------------------------------------
    section("⑧ 轨迹落盘与统计")
    stats = trace.stats()
    check("轨迹已落盘", stats["tasks"] >= 7, f"tasks={stats['tasks']}")
    check("能读回指定任务的轨迹", bool(trace.load("T-normal")), "T-normal")
    check("统计包含成功率与平均步数",
          "success_rate" in stats and "avg_steps" in stats,
          f"success_rate={stats.get('success_rate')} avg_steps={stats.get('avg_steps')}")

    print()
    # 【指标解读说明】这个测试集里刻意放了一堆异常场景，
    # 所以"整体完成率"天然很低 —— 不能拿它当系统成功率讲。
    # 正确的指标是分开看：正常场景的完成率 + 异常场景的约束触发率。
    #
    # 【这里踩过一个坑】原先这段是硬编码的 `print("约束触发准确率 5/5 = 100%")` ——
    #   数字是手写死的，不是算出来的。用例后来从 5 个扩到 7 个，它照样打印 5/5，
    #   **而且不会有任何地方报错**。指标一旦是手写的，它就迟早变成谎言。
    #   现在改为从落盘轨迹的结束原因里真实聚合。
    by_reason = stats.get("by_finish_reason") or {}
    EXPECTED_CONSTRAINTS = [
        FinishReason.LOOP_DETECTED,   # ② 连续相同调用 / ②b 交替循环
        FinishReason.MAX_STEPS,       # ③ 步数超限
        FinishReason.TOKEN_BUDGET,    # ④ 预算超限
        FinishReason.NO_ACTION,       # ⑦ 既不调工具也不作答
        FinishReason.USAGE_ERROR,     # ⑦b 给了动作但格式非法
    ]
    hit = [r for r in EXPECTED_CONSTRAINTS if by_reason.get(r.value, 0) > 0]
    coverage = len(hit) / len(EXPECTED_CONSTRAINTS)

    print("  【场景构成】")
    print("    · 正常场景       1 个（T-normal）  → 期望 completed")
    print(f"    · 约束类场景     {len(EXPECTED_CONSTRAINTS)} 类"
          f"（{'、'.join(r.value for r in EXPECTED_CONSTRAINTS)}）→ 期望被识别并优雅收尾")
    print("    · 高危审批       1 个（T-approval）→ 期望挂起且不重复建单")
    print()
    print("  【关键指标】（以下数字全部从落盘轨迹聚合，没有手写的常量）")
    print(f"    约束触发覆盖率        {len(hit)}/{len(EXPECTED_CONSTRAINTS)} = {coverage:.0%}")
    print(f"    结束原因分布          {by_reason}")
    print(f"    平均步数              {stats.get('avg_steps')}")
    print(f"    平均 token            {stats.get('avg_tokens')}")
    print(f"    平均耗时(ms)          {stats.get('avg_latency_ms')}")
    print()
    print(f"  （注：整体完成率 {stats.get('completed')}/{stats.get('tasks')} 偏低是因为测试集刻意偏难，"
          f"不代表系统真实成功率）")

    print()
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        if not ok:
            print(f"  FAIL  {name}")
    print(f"  结果: {passed}/{len(results)} 通过")
    print(f"  轨迹落盘: {TRACE_PATH}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
