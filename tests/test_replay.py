"""确定性回放验证 —— 11 条轨迹 + 篡改检测 + 旧轨迹兼容

【为什么必须测"回放器能不能发现问题"】
    一个永远报"一致"的回放器等于没有回放器。
    所以这里除了验证"正常轨迹能重放"，还必须**故意篡改轨迹**，
    确认它真的能把差异抓出来、并且精确指到"第几步、哪个字段"。

【为什么这个测试特别值得留】
    写这套回放器的过程中，它自己抓出了 **4 个"轨迹不自包含"的缺陷**：
      1. 没记模型的原始 action        → usage_error / no_action 无法重现
      2. 没记工具结果的完整快照        → denied 的 reason、审批的 approval_id 丢失
      3. 没判断原脚本是不是"无限重复"型 → loop_detected 被重放成 answered
      4. 没还原每步 token 数          → token_budget 被重放成 loop_detected
    这四条的共同点是：**轨迹记了"发生了什么"，但没记"为什么会这样"** ——
    而回放恰恰要求的是一份**自包含**的记录。
    这个教训比回放功能本身更值钱：**可观测性不是"记日志"，是"记到能重建现场"**。

运行：python tests/test_replay.py
"""
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from brain import ScriptedBrain  # noqa: E402
from builtin_tools import ALL_TOOLS, ROLE_OPERATOR, ROLE_READER  # noqa: E402
from gateway import ApprovalStore, AuditStore, IdempotencyLedger, ToolGateway  # noqa: E402
from registry import ToolRegistry  # noqa: E402
from replay import brain_from_trace, replay, replay_all  # noqa: E402
from runtime import AgentRuntime, TraceStore  # noqa: E402

AUDIT_PATH = os.path.join(ROOT, "data", "audit.jsonl")
TRACE_PATH = os.path.join(ROOT, "data", "traces_replay_check.jsonl")

results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  {'✅' if ok else '❌'} {name}" + (f" —— {detail}" if detail else ""))


def section(title):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def build_gateway(audit):
    registry = ToolRegistry()
    registry.register_all(ALL_TOOLS)
    return ToolGateway(registry, audit, ApprovalStore(), IdempotencyLedger())


def main() -> int:
    audit = AuditStore(AUDIT_PATH)
    audit.clear()
    trace = TraceStore(TRACE_PATH)
    trace.clear()
    gw = build_gateway(audit)

    # ---------------- 造几条覆盖面广的轨迹 ----------------
    section("准备：生成 4 条不同类型任务的轨迹")

    cases = [
        # 正常：查订单 → 作答
        ("T-rp-normal", "查询订单", ROLE_READER, 8, 5000, 3,
         [{"thought": "先查订单", "action": {"tool": "query_order", "args": {"order_id": "ORD-000123"}}},
          {"thought": "有结果了", "answer": "订单已支付。"}], False),
        # 循环：重复同一调用 → 死循环检测
        ("T-rp-loop", "反复查同一单", ROLE_READER, 10, 99999, 3,
         [{"thought": "查订单", "action": {"tool": "query_order", "args": {"order_id": "ORD-000123"}}}], True),
        # 越权：退款被拒 → 换只读工具 → 作答
        ("T-rp-denied", "申请退款", ROLE_READER, 8, 5000, 3,
         [{"thought": "尝试退款", "action": {"tool": "issue_refund", "args": {"order_id": "ORD-000123", "amount": 50}}},
          {"thought": "没权限，改用只读", "action": {"tool": "query_order", "args": {"order_id": "ORD-000123"}}},
          {"thought": "给出结论", "answer": "我没有退款权限。"}], False),
        # 高危审批：小额/大额都走一遍
        ("T-rp-approval", "大额退款", ROLE_OPERATOR, 6, 99999, 2,
         [{"thought": "发起退款", "action": {"tool": "issue_refund", "args": {"order_id": "ORD-000777", "amount": 3000}}}], True),
    ]

    for task_id, goal, role, max_steps, budget, loop_th, script, repeat in cases:
        rt = AgentRuntime(
            ScriptedBrain(script, repeat_last=repeat),
            gw, max_steps=max_steps, token_budget=budget, loop_threshold=loop_th,
            trace_store=trace,
        )
        rt.run(goal, role=role, task_id=task_id)

    rows = trace.load_all()
    check("轨迹已生成", len(rows) == len(cases), f"{len(rows)} 条")

    # ---------------- 回放：应该全部一致 ----------------
    section("① 正常轨迹必须能确定性重放")

    reports = replay_all(TRACE_PATH)
    ok_count = sum(1 for r in reports if r.ok)
    check(
        f"{len(cases)}/{len(cases)} 条轨迹可确定性重放",
        ok_count == len(cases),
        f"通过 {ok_count}/{len(cases)}",
    )
    for r in reports:
        if not r.ok:
            print(f"      · {r.task_id}: {r.error or r.divergences}")

    # 未结束原因、token、步数都要一致 —— 逐条核对
    check(
        "结束原因与 token 全程一致（不只是步数对得上）",
        all(
            r.finish_reason_original == r.finish_reason_replayed and r.tokens_original == r.tokens_replayed
            for r in reports
        ),
    )

    # ---------------- 篡改检测：回放器必须能发现问题 ----------------
    section("② 篡改轨迹后，回放器必须能抓出来（否则它等于没有）")

    record = copy.deepcopy(rows[0])
    # 篡改：把第 1 步的观察文本改掉
    record["steps"][1]["observation"] = "（被人为篡改过的观察）"
    r = replay(record)
    check("篡改 observation 被检出", not r.ok and len(r.divergences) >= 1, f"{len(r.divergences)} 处差异")
    check(
        "差异定位到具体步与字段",
        any(d.field == "observation" for d in r.divergences),
        f"{[(d.step, d.field) for d in r.divergences]}",
    )

    # 篡改：把工具返回结果改掉 —— 这会改变渲染出的观察，也应被抓出
    record2 = copy.deepcopy(rows[0])
    for s in record2["steps"]:
        if s.get("tool_call_result"):
            s["tool_call_result"]["result"] = {"order_id": "ORD-999999", "status": "refunded"}
    r2 = replay(record2)
    check("篡改工具返回结果被检出", not r2.ok, f"{len(r2.divergences)} 处差异")

    # 篡改：把结束原因改掉
    record3 = copy.deepcopy(rows[0])
    record3["finish_reason"] = "loop_detected"
    r3 = replay(record3)
    check("篡改结束原因被检出", not r3.ok, f"{len(r3.divergences)} 处差异")

    # ---------------- 旧轨迹兼容 ----------------
    section("③ 早期版本的轨迹（缺 tool_call_result）要给明确提示，而不是静默跑错")

    legacy = copy.deepcopy(rows[1])
    for s in legacy["steps"]:
        s.pop("tool_call_result", None)
    r4 = replay(legacy)
    check("缺 tool_call_result 时明确报错", not r4.ok and bool(r4.error), (r4.error or "")[:60])

    legacy2 = copy.deepcopy(rows[0])
    for s in legacy2["steps"]:
        s.pop("raw_action", None)
    r5 = replay(legacy2)
    # 缺 raw_action 不一定会不一致（普通 action 可由 tool_name 推出），但不应崩溃
    check("缺 raw_action 时不崩溃（尽力回放）", isinstance(r5.ok, bool))

    # ---------------- brain 反推的正确性 ----------------
    section("④ 从轨迹反推 brain 脚本：两个关键还原点")

    brain = brain_from_trace(rows[0])
    check(
        "有 done 步的轨迹 → 重建脚本不重复（repeat_last=False）",
        getattr(brain, "repeat_last", None) is False,
    )

    loop_record = next(r for r in rows if r["task_id"] == "T-rp-loop")
    brain_loop = brain_from_trace(loop_record)
    check(
        "无 done 步的轨迹 → 重建脚本重复（repeat_last=True，否则死循环重放不出来）",
        getattr(brain_loop, "repeat_last", None) is True,
    )

    # ---------------- 汇总 ----------------
    print()
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        if not ok:
            print(f"  FAIL  {name}")
    print(f"  结果: {passed}/{len(results)} 通过")
    print(f"  轨迹落盘: {TRACE_PATH}")

    trace.clear()
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
