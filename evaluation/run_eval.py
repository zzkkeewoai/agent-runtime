"""Agent 任务级评测运行器

跑法：
    加载任务集 → 每条任务用脚本化 brain 跑一次完整的 Agent 流程（经过工具网关）
    → 判定通过/失败 → 归因 → 汇总报告

为什么用"脚本化 brain"而不是真实 LLM：
    评测要能**稳定复现**。真实 LLM 每次都不同，无法定位"这次失败是模型随机性
    还是我的系统问题"。脚本化 brain 把模型这个变量固定住，测的才是
    **运行时 + 工具网关 + 约束** 这一整条链路。
    接真实 LLM 时只需换 brain 实现，评测逻辑一行不改。

运行：
    cd agent_runtime
    python evaluation/run_eval.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, EVAL_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import attribution as attr_mod  # noqa: E402
from brain import ScriptedBrain  # noqa: E402
from builtin_tools import ALL_TOOLS  # noqa: E402
from gateway import ApprovalStore, AuditStore, IdempotencyLedger, ToolGateway  # noqa: E402
from registry import ToolRegistry  # noqa: E402
from runtime import AgentRuntime, FinishReason, TraceStore  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
AUDIT_PATH = os.path.join(DATA_DIR, "audit.jsonl")
TRACE_PATH = os.path.join(DATA_DIR, "traces.jsonl")


def load_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================================================
# 判定：一次执行是否符合期望
# ============================================================
def judge(run, case) -> tuple:
    """返回 (passed, reasons)。

    判定四个维度：
      1. outcome     结果形态是否符合期望（完成 / 挂审批 / 被拒 / 任意）
      2. tools_used  期望用到的工具是否都用了
      3. must_not_use 禁止的工具是否一次都没碰
      4. max_steps   步数是否在预期内
    """
    expect = case.get("expect", {})
    reasons = []
    ok = True

    outcome = expect.get("outcome", "any")
    if outcome == "completed":
        if not run.completed:
            ok = False
            reasons.append(f"期望 completed，实际 {run.finish_reason}")
    elif outcome == "pending_approval":
        if run.pending_approvals < 1:
            ok = False
            reasons.append("期望产生待审批，实际没有")
    elif outcome == "denied_attempt":
        if run.denied_calls < 1:
            ok = False
            reasons.append("期望出现被拒调用，实际没有")
    # outcome == "any"：不校验结果形态（这类用例关心的是"约束是否正确触发"）

    used = [s.tool_name for s in run.steps if s.tool_name]
    for tool in expect.get("tools_used", []) or []:
        if tool not in used:
            ok = False
            reasons.append(f"期望使用工具 {tool}，实际用了 {used}")

    for tool in expect.get("must_not_use", []) or []:
        if tool in used:
            ok = False
            reasons.append(f"触碰了禁止的工具 {tool}")

    max_steps = expect.get("max_steps")
    if max_steps and run.step_count > max_steps:
        ok = False
        reasons.append(f"步数 {run.step_count} 超过预期 {max_steps}")

    return ok, reasons


# ============================================================
# 报告
# ============================================================
def print_case_results(rows):
    print("\n" + "=" * 88)
    print("逐条结果")
    print("=" * 88)
    print(f"{'用例':<6}{'判定':<7}{'归因':<5}{'结果':<16}{'步数':<6}{'工具':<6}{'token':<8}归因类别")
    print("-" * 88)
    for r in rows:
        run, attribution, passed = r["run"], r["attribution"], r["passed"]
        attr_mark = "—" if r["attr_match"] is None else ("✓" if r["attr_match"] else "✗")
        print(
            f"{r['case_id']:<6}{'PASS' if passed else 'FAIL':<7}{attr_mark:<5}"
            f"{run.finish_reason:<16}{run.step_count:<6}{run.tool_calls:<6}"
            f"{run.total_tokens:<8}{attribution.category.value}"
        )
    traps = [r for r in rows if r["is_trap"] or r["is_model_error"]]
    if traps:
        print()
        for r in traps:
            if r["is_trap"]:
                kind = "故意违规用例"
                note = "模型被故意引导去碰禁止工具，验证系统能否拦住 + 归因能否标出"
            else:
                kind = "模型行为错误用例"
                note = "模型被故意引导用错工具，任务【应当失败】，验证归因能否定位到规划问题"
            print(f"  ⚠️ {r['case_id']} 是【{kind}】：{note}")
            print(
                f"     实际：任务判定={'PASS' if r['passed'] else 'FAIL'}，"
                f"归因={r['attribution'].category.value}，"
                f"归因与标注{'一致 ✅' if r['attr_match'] else '不一致 ❌'}"
            )


def print_metrics(rows):
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    completed = sum(1 for r in rows if r["run"].completed)
    policy_violations = sum(1 for r in rows if r["attribution"].category.value == "policy_violation")
    denied = sum(r["run"].denied_calls for r in rows)
    pending = sum(r["run"].pending_approvals for r in rows)
    avg_steps = round(sum(r["run"].step_count for r in rows) / total, 2)
    avg_tokens = round(sum(r["run"].total_tokens for r in rows) / total, 1)

    # 归因准确率：与人工设定的期望归因比对（这是"归因器准不准"的指标）
    checked = [r for r in rows if r["attr_match"] is not None]
    attr_ok = sum(1 for r in checked if r["attr_match"])

    print("\n" + "=" * 88)
    print("指标汇总")
    print("=" * 88)
    print(f"  任务通过率（符合期望）      {passed}/{total} = {passed / total:.0%}")
    print(f"  任务完成率（正常收尾）      {completed}/{total} = {completed / total:.0%}")
    print(f"  归因准确率（与标注一致）     {attr_ok}/{len(checked)} = {(attr_ok / len(checked) if checked else 0):.0%}")
    print(f"  安全违规检出                {policy_violations} 次（故意违规用例）")
    print(f"  越权拦截次数                {denied}")
    print(f"  高危审批挂起次数            {pending}")
    print(f"  平均步数 / 平均 token       {avg_steps} / {avg_tokens}")

    # 约束触发准确率：本测试集里 a06~a10 是"异常/边界"用例
    abnormal = [r for r in rows if r["case_id"] in ("a06", "a07", "a08", "a09", "a10")]
    if abnormal:
        triggered = sum(1 for r in abnormal if r["attribution"].category.value != "success")
        print(f"  异常用例归因识别率          {triggered}/{len(abnormal)}")


def print_attribution(rows):
    items = [r["attribution"] for r in rows]
    dist = attr_mod.summarize_attributions(items)
    print("\n" + "=" * 84)
    print("归因分布（按严重程度排序）")
    print("=" * 84)
    for cat, cnt in dist.items():
        print(f"  {cat:<22} {cnt}")

    fails = attr_mod.failure_categories(items)
    if fails:
        print("\n  ── 优化建议（按归因类别）──")
        seen = set()
        for r in rows:
            a = r["attribution"]
            if a.category.value == "success" or a.category in seen:
                continue
            seen.add(a.category)
            print(f"  · {a.category.value:<22} {a.detail}")
            print(f"    → {a.fix_hint}")


def main() -> int:
    cases = load_jsonl(os.path.join(DATA_DIR, "eval_tasks.jsonl"))

    registry = ToolRegistry()
    registry.register_all(ALL_TOOLS)
    audit = AuditStore(AUDIT_PATH)
    audit.clear()
    trace = TraceStore(TRACE_PATH)
    trace.clear()
    gw = ToolGateway(registry, audit, ApprovalStore(), IdempotencyLedger())

    print("=" * 84)
    print(f"Agent 任务评测 · {len(cases)} 条任务（脚本化 brain，可稳定复现）")
    print("=" * 84)

    rows = []
    for case in cases:
        brain = ScriptedBrain(
            case["script"],
            repeat_last=case.get("repeat_last", False),
        )
        rt = AgentRuntime(
            brain,
            gw,
            max_steps=case.get("max_steps", 8),
            token_budget=case.get("token_budget", 5000),
            loop_threshold=case.get("loop_threshold", 3),
            trace_store=trace,
        )
        run = rt.run(case["goal"], role=case["role"], task_id=case["case_id"])
        passed, reasons = judge(run, case)
        attribution = attr_mod.attribute(run, case)

        # 归因准确率：与人工标注的期望归因比对（独立于"任务通过率"的指标）
        expect_attr = case.get("expect_attribution")
        attr_match = (attribution.category.value == expect_attr) if expect_attr else None

        rows.append(
            {
                "case_id": case["case_id"],
                "run": run,
                "passed": passed,
                "reasons": reasons,
                "attribution": attribution,
                "attr_match": attr_match,
                "is_trap": case.get("trap", False),
                "is_model_error": case.get("model_error", False),
            }
        )

    print_case_results(rows)
    print_metrics(rows)
    print_attribution(rows)

    print("\n" + "=" * 84)
    print(f"轨迹落盘: {TRACE_PATH}")
    print(f"审计落盘: {AUDIT_PATH}")
    print("=" * 84)

    # 退出码语义：判"是否符合预期"，而不是"是否全部通过"。
    #   · model_error 用例【应当失败】—— 它失败才说明模型确实犯了错、可被检出
    #   · 其它用例应当通过
    #   · 所有用例的归因都应与人工标注一致
    unexpected = [
        r for r in rows
        if (r["is_model_error"] and r["passed"]) or (not r["is_model_error"] and not r["passed"])
    ]
    attr_mismatch = [r for r in rows if r["attr_match"] is False]
    if unexpected:
        print("\n  ⚠️ 判定不符合预期的用例: " + ", ".join(r["case_id"] for r in unexpected))
    if attr_mismatch:
        print("\n  ⚠️ 归因与标注不一致的用例: " + ", ".join(r["case_id"] for r in attr_mismatch))
    return 0 if (not unexpected and not attr_mismatch) else 1


if __name__ == "__main__":
    sys.exit(main())
