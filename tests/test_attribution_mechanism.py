"""tests/test_attribution_mechanism.py —— 「机制失效」四类归因的单元测试

测的是 attribution.py 里 B 组的四个判据：
    IDEMPOTENCY_FAILURE      幂等失效（同一个键被真实执行两次）
    RECOVERY_FAILURE         恢复失败（结果未知后直接重放）
    STATE_TRANSITION_ERROR   状态转换错误（状态机被破坏）
    LOOP_DETECTION_FAILURE   循环检测失效（判据没抓住明显重复）

【为什么要直接构造 TaskRun，而不是跑一遍 runtime】
    这四类都是"保障机制失效"，**正常跑是跑不出来的** ——
    要构造出失效的场景，必须手工拼轨迹。
    直接构造还有一个好处：**测的是判据本身**，不受 runtime 行为变化影响
    （runtime 改了不会让这些断言莫名其妙地红/绿）。

【每个判据都必须配反向验证】
    一个"什么都判失效"的检测器能让所有正例通过 —— 但它比不检测更糟：
    误报多了人就不看报告了。所以每组都配了"合法情况绝不能误报"的断言。

运行：python tests/test_attribution_mechanism.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import attribution as attr_mod  # noqa: E402
from attribution import Attribution, attribute, mechanism_failure_rate  # noqa: E402
from runtime import FinishReason, Step, TaskRun  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK  {name}")
    else:
        FAIL += 1
        print(f"  XX  {name}  {detail}")


def section(title):
    print(f"\n{'=' * 66}\n  {title}\n{'=' * 66}")


# ============================================================
# 构造工具
# ============================================================
def snap(status="executed", key="", idempotent=True, side_effect=True):
    """构造一份工具结果快照（字段和 runtime 里写进轨迹的一致）"""
    return {
        "status": status,
        "tool_name": "",
        "result": None,
        "reason": "",
        "approval_id": None,
        "idempotency_key": key,
        "idempotent": idempotent,
        "side_effect": side_effect,
    }


def call_step(i, tool, args=None, status="executed", sn=None, obs=None):
    """一步"调用了工具并拿到观察"的轨迹（state=observing）"""
    args = args if args is not None else {"order_id": "ORD-000123"}
    return Step(
        index=i,
        state="observing",
        thought="调工具",
        tool_name=tool,
        args=args,
        tool_status=status,
        observation=obs if obs is not None else f"工具 {tool} 执行结果",
        tool_call_result=sn if sn is not None else snap(status=status),
    )


def think_step(i, obs=None, tool=None):
    """一步"只想没动手"的轨迹（state=thinking，被约束中止时带观察）"""
    return Step(
        index=i, state="thinking", thought="想一想",
        tool_name=tool, observation=obs,
    )


def done_step(i):
    """正常作答步（state=done，有观察文本但没有工具调用）"""
    return Step(index=i, state="done", thought="作答", observation="(模型直接作答)")


def finish_step(i, reason="max_steps"):
    """优雅收尾步（state=force_answer）"""
    return Step(
        index=i, state="force_answer", thought="收尾",
        observation="(约束触发，进入优雅收尾) token 预算已耗尽",
        tool_call_result=None,
    )


def make_run(steps, finish_reason="answered", completed=True, **kw):
    return TaskRun(
        task_id="T-test",
        goal="测试任务",
        role="agent_operator",
        status="completed" if completed else "aborted",
        finish_reason=finish_reason,
        final_answer="结论" if completed else None,
        steps=steps,
        loop_threshold=kw.pop("loop_threshold", 3),
        **kw,
    )


CASE = {"case_id": "C-test"}


def main():
    global PASS, FAIL

    # ============================================================
    section("① 幂等失效 —— 同一个幂等键被真实执行两次")
    key = "abc123def456"
    run = make_run([
        think_step(0),
        call_step(1, "issue_refund", sn=snap("executed", key=key)),
        call_step(2, "issue_refund", sn=snap("executed", key=key)),
        done_step(3),
    ])
    r = attribute(run, CASE)
    check("重复执行同一幂等键 → IDEMPOTENCY_FAILURE",
          r.category is Attribution.IDEMPOTENCY_FAILURE, f"实际 {r.category}")

    # 反向验证 1：第二次命中缓存（幂等生效）→ 不能报失效
    run = make_run([
        think_step(0),
        call_step(1, "issue_refund", sn=snap("executed", key=key)),
        call_step(2, "issue_refund", status="cached", sn=snap("cached", key=key)),
        done_step(3),
    ])
    r = attribute(run, CASE)
    check("幂等命中（cached）不算失效",
          r.category is Attribution.SUCCESS, f"实际 {r.category}")

    # 反向验证 2：非幂等工具的重复执行是合法的 → 不能报失效
    run = make_run([
        think_step(0),
        call_step(1, "send_email", sn=snap("executed", key=key, idempotent=False)),
        call_step(2, "send_email", sn=snap("executed", key=key, idempotent=False)),
        done_step(3),
    ])
    r = attribute(run, CASE)
    check("非幂等工具重复执行不报失效",
          r.category is Attribution.SUCCESS, f"实际 {r.category}")

    # 反向验证 3：不同幂等键 → 本来就不是重复
    run = make_run([
        think_step(0),
        call_step(1, "issue_refund", args={"order_id": "ORD-000001"},
                  sn=snap("executed", key="key-a")),
        call_step(2, "issue_refund", args={"order_id": "ORD-000002"},
                  sn=snap("executed", key="key-b")),
        done_step(3),
    ])
    r = attribute(run, CASE)
    check("不同幂等键（两笔不同业务）不报失效",
          r.category is Attribution.SUCCESS, f"实际 {r.category}")

    # 反向验证 4：旧轨迹缺 idempotent 字段 → 保守不判定
    old = snap("executed", key=key)
    old.pop("idempotent")
    run = make_run([
        think_step(0),
        call_step(1, "issue_refund", sn=old),
        call_step(2, "issue_refund", sn=dict(old)),
        done_step(3),
    ])
    r = attribute(run, CASE)
    check("旧轨迹缺 idempotent 字段 → 保守不判定（宁漏勿误）",
          r.category is Attribution.SUCCESS, f"实际 {r.category}")

    # ============================================================
    section("② 恢复失败 —— 结果未知后未经状态查询就重放")
    args = {"order_id": "ORD-000777", "amount": 3000}
    run = make_run([
        think_step(0),
        call_step(1, "issue_refund", args=args, status="timeout",
                  sn=snap("timeout", key="k-refund"), obs="工具执行超时"),
        call_step(2, "issue_refund", args=args, sn=snap("executed", key="k-refund")),
        done_step(3),
    ], finish_reason="answered")
    r = attribute(run, CASE)
    check("超时后直接重放 → RECOVERY_FAILURE",
          r.category is Attribution.RECOVERY_FAILURE, f"实际 {r.category}")

    # 反向验证：中间先查了状态 → 这是**正确**的恢复流程，不能报失败
    run = make_run([
        think_step(0),
        call_step(1, "issue_refund", args=args, status="timeout",
                  sn=snap("timeout", key="k-refund"), obs="工具执行超时"),
        call_step(2, "query_refund", args=args, sn=snap("executed", key="k-query")),
        call_step(3, "issue_refund", args=args, sn=snap("executed", key="k-refund")),
        done_step(4),
    ])
    r = attribute(run, CASE)
    check("超时 → 查状态 → 再重放：不升级为恢复失败（超时本身仍归环境异常）",
          r.category is Attribution.ENVIRONMENT_ERROR, f"实际 {r.category}")

    # 反向验证：超时之后执行的是**另一个操作** → 不构成盲目重放
    run = make_run([
        think_step(0),
        call_step(1, "issue_refund", args=args, status="timeout",
                  sn=snap("timeout", key="k-a"), obs="工具执行超时"),
        call_step(2, "query_order", args={"order_id": "ORD-000123"},
                  sn=snap("executed", key="k-b")),
        done_step(3),
    ])
    r = attribute(run, CASE)
    check("超时后做别的操作：不升级为恢复失败",
          r.category is Attribution.ENVIRONMENT_ERROR, f"实际 {r.category}")

    # ============================================================
    section("③ 状态转换错误 —— 状态机不变量被破坏")
    # ③-1 瞬态状态落盘
    bad = call_step(1, "query_order")
    bad.state = "acting"
    run = make_run([think_step(0), bad, done_step(2)])
    check("ACTING 落盘（瞬态不该出现）被检出",
          attribute(run, CASE).category is Attribution.STATE_TRANSITION_ERROR)

    # ③-2 observing 但没有调用
    bad = think_step(1)
    bad.state = "observing"
    run = make_run([think_step(0), bad, done_step(2)])
    check("observing 却没有工具调用被检出",
          attribute(run, CASE).category is Attribution.STATE_TRANSITION_ERROR)

    # ③-3 有调用但状态不是 observing
    bad = call_step(1, "query_order")
    bad.state = "thinking"
    run = make_run([think_step(0), bad, done_step(2)])
    check("有工具调用但状态是 thinking 被检出",
          attribute(run, CASE).category is Attribution.STATE_TRANSITION_ERROR)

    # ③-4 调用了工具但没有观察
    bad = call_step(1, "query_order", obs="")  # 空字符串 = 没观察
    run = make_run([think_step(0), bad, done_step(2)])
    check("调用成功但没有观察结果被检出",
          attribute(run, CASE).category is Attribution.STATE_TRANSITION_ERROR)

    # ③-5 终态之后还有步骤
    run = make_run([think_step(0), done_step(1), call_step(2, "query_order")])
    check("终态之后还有步骤被检出",
          attribute(run, CASE).category is Attribution.STATE_TRANSITION_ERROR)

    # ③-6 未知状态
    bad = think_step(1)
    bad.state = "some_unknown_state"
    run = make_run([think_step(0), bad, done_step(2)])
    check("未知状态值被检出",
          attribute(run, CASE).category is Attribution.STATE_TRANSITION_ERROR)

    # ---- 反向验证：真实轨迹里最常见、也最容易被误报的几种形态 ----
    run = make_run([think_step(0), call_step(1, "query_order"), done_step(2)])
    check("正常轨迹（thinking→observing→done）不报错",
          attribute(run, CASE).category is Attribution.SUCCESS,
          f"实际 {attribute(run, CASE).category}")

    # 约束中止步：thinking + 有观察 + 无工具调用 —— 这是合法形态（runtime 真的这么写）
    run = make_run([
        think_step(0),
        call_step(1, "query_order"),
        think_step(2, obs="(检测到死循环，已中止: 连续 3 次完全相同的调用（query_order）)"),
        finish_step(3),
    ], finish_reason="loop_detected", completed=False)
    got = attribute(run, CASE).category
    check("约束中止步（thinking + 有观察）不被误报为状态错误",
          got is Attribution.LOOP_DETECTED, f"实际 {got}")

    # 优雅收尾步：force_answer + 有观察 + 无工具调用 —— 合法
    run = make_run([
        think_step(0),
        call_step(1, "query_order"),
        finish_step(2),
    ], finish_reason="max_steps", completed=False)
    got = attribute(run, CASE).category
    check("优雅收尾步（force_answer）不被误报为状态错误",
          got is Attribution.MAX_STEPS, f"实际 {got}")

    # 预算耗尽的中止步同样是 thinking + 无 tool
    run = make_run([
        think_step(0, obs="(token 预算已耗尽，进入优雅收尾)"),
        finish_step(1),
    ], finish_reason="token_budget", completed=False)
    got = attribute(run, CASE).category
    check("预算耗尽形态不被误报为状态错误",
          got is Attribution.BUDGET_EXHAUSTED, f"实际 {got}")

    # ============================================================
    section("④ 循环检测失效 —— 判据没抓住明显重复")
    # ④-1 连续三次完全相同的调用，却以 max_steps 结束
    run = make_run([
        call_step(0, "query_order", args={"order_id": "ORD-000123"}),
        call_step(1, "query_order", args={"order_id": "ORD-000123"}),
        call_step(2, "query_order", args={"order_id": "ORD-000123"}),
        finish_step(3),
    ], finish_reason="max_steps", completed=False)
    r = attribute(run, CASE)
    check("连续 3 次相同调用但未判循环 → LOOP_DETECTION_FAILURE",
          r.category is Attribution.LOOP_DETECTION_FAILURE, f"实际 {r.category}")

    # ④-2 交替循环 A,B,A,B 却以 token_budget 结束（这正是当初漏判的那类）
    run = make_run([
        call_step(0, "query_order", args={"order_id": "ORD-A"}),
        call_step(1, "query_user_profile", args={"user_id": "U-1"}),
        call_step(2, "query_order", args={"order_id": "ORD-A"}),
        call_step(3, "query_user_profile", args={"user_id": "U-1"}),
        finish_step(4),
    ], finish_reason="token_budget", completed=False)
    r = attribute(run, CASE)
    check("交替循环 A,B,A,B 未被判循环 → LOOP_DETECTION_FAILURE",
          r.category is Attribution.LOOP_DETECTION_FAILURE, f"实际 {r.category}")

    # ---- 反向验证 ----
    # ④-3 检测器真的工作了（finish_reason=loop_detected）→ 不算失效
    run = make_run([
        call_step(0, "query_order", args={"order_id": "ORD-000123"}),
        call_step(1, "query_order", args={"order_id": "ORD-000123"}),
        call_step(2, "query_order", args={"order_id": "ORD-000123"}),
        finish_step(3),
    ], finish_reason="loop_detected", completed=False)
    got = attribute(run, CASE).category
    check("检测器已触发（loop_detected）→ 不算失效",
          got is Attribution.LOOP_DETECTED, f"实际 {got}")

    # ④-4 六次互不相同的调用 → 正常批量任务，绝不能误报
    run = make_run([
        call_step(i, "query_order", args={"order_id": f"ORD-00000{i}"})
        for i in range(6)
    ] + [done_step(6)])
    got = attribute(run, CASE).category
    check("6 次互不相同的调用（正常批量）不误报",
          got is Attribution.SUCCESS, f"实际 {got}")

    # ④-5 只有两次调用 → 不够判据，不报
    run = make_run([
        call_step(0, "query_order", args={"order_id": "ORD-000123"}),
        call_step(1, "query_order", args={"order_id": "ORD-000123"}),
        done_step(2),
    ])
    got = attribute(run, CASE).category
    check("只有两次相同调用 → 不报（判据要求连续 3 次）",
          got is Attribution.SUCCESS, f"实际 {got}")

    # ============================================================
    section("⑤ 优先级 —— 机制失效必须压过它掩盖的那个现象")
    # 幂等失效 + 越权并存 → 报幂等失效（重复扣钱比越权尝试损失更大）
    run = make_run([
        think_step(0),
        call_step(1, "issue_refund", status="denied",
                  sn=snap("denied", key="k1"), obs="[permission_denied] 无权调用"),
        call_step(2, "issue_refund", sn=snap("executed", key="k2")),
        call_step(3, "issue_refund", sn=snap("executed", key="k2")),
        done_step(4),
    ])
    r = attribute(run, CASE)
    check("幂等失效 优先于 权限拒绝",
          r.category is Attribution.IDEMPOTENCY_FAILURE, f"实际 {r.category}")

    # 恢复失败 + 环境异常并存 → 报恢复失败（盲目重放比"超时"这个现象严重）
    run = make_run([
        think_step(0),
        call_step(1, "issue_refund", args=args, status="timeout",
                  sn=snap("timeout", key="k"), obs="工具执行超时"),
        call_step(2, "issue_refund", args=args, sn=snap("executed", key="k")),
        done_step(3),
    ])
    r = attribute(run, CASE)
    check("恢复失败 优先于 环境异常",
          r.category is Attribution.RECOVERY_FAILURE, f"实际 {r.category}")

    # 策略违规 + 循环检测失效并存 → 报策略违规（安全问题永远最高）
    case_policy = {"case_id": "C2", "forbidden_tools": ["delete_record"]}
    run = make_run([
        call_step(0, "delete_record", sn=snap("executed", key="k1")),
        call_step(1, "delete_record", sn=snap("executed", key="k1")),
        call_step(2, "delete_record", sn=snap("executed", key="k1")),
        finish_step(3),
    ], finish_reason="max_steps", completed=False)
    r = attribute(run, case_policy)
    check("策略违规 优先于 循环检测失效",
          r.category is Attribution.POLICY_VIOLATION, f"实际 {r.category}")

    # ============================================================
    section("⑥ 机制失效率 —— 这个指标正常值就该是 0")
    items = [
        attribute(make_run([think_step(0), call_step(1, "query_order"), done_step(2)]), CASE),
        attribute(make_run([think_step(0), done_step(1)]), CASE),
        attribute(make_run([
            think_step(0),
            call_step(1, "issue_refund", sn=snap("executed", key="kk")),
            call_step(2, "issue_refund", sn=snap("executed", key="kk")),
            done_step(3),
        ]), CASE),
    ]
    stat = mechanism_failure_rate(items)
    check("总样本数正确", stat["total"] == 3, f"实际 {stat['total']}")
    check("机制失效数为 1", stat["mechanism_failures"] == 1,
          f"实际 {stat['mechanism_failures']}")
    check("失效率 ≈ 0.3333", abs(stat["rate"] - 0.3333) < 0.001, f"实际 {stat['rate']}")
    check("按类别计数正确",
          stat["by_category"] == {"idempotency_failure": 1}, f"实际 {stat['by_category']}")

    check("is_mechanism_failure 只对 B 组为真",
          all(
              (it.category in attr_mod.MECHANISM_CATEGORIES) == it.is_mechanism_failure
              for it in items
          ))

    # ============================================================
    print(f"\n{'=' * 66}")
    print(f"  结果: {PASS} 通过 / {FAIL} 失败")
    print(f"{'=' * 66}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
