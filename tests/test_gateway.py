"""工具网关验证 —— 13 个场景，产出可量化的安全指标

运行的指标（面试可直接报）：
  · 越权拦截率            N/N
  · 高危操作审批覆盖率     N/N
  · 幂等正确性            重复调用副作用只发生一次
  · 超时保护              0.5s 超时的工具在 ~0.5s 返回（而不是等任务真的跑完）

运行：python tests/test_gateway.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from builtin_tools import (  # noqa: E402
    ALL_TOOLS,
    ROLE_ADMIN,
    ROLE_OPERATOR,
    ROLE_READER,
)
from gateway import (  # noqa: E402
    ApprovalStore,
    AuditStore,
    IdempotencyLedger,
    ToolCallRequest,
    ToolGateway,
)
from registry import ToolRegistry  # noqa: E402

AUDIT_PATH = os.path.join(ROOT, "data", "audit.jsonl")

results = []


def check(name: str, ok: bool, detail: str = ""):
    results.append((name, ok))
    print(f"  {'✅' if ok else '❌'} {name}" + (f" —— {detail}" if detail else ""))


def section(title: str):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def main() -> int:
    registry = ToolRegistry()
    registry.register_all(ALL_TOOLS)
    audit = AuditStore(AUDIT_PATH)
    audit.clear()
    approvals = ApprovalStore()
    ledger = IdempotencyLedger()
    gw = ToolGateway(registry, audit, approvals, ledger)

    builder = ToolCallRequest
    sec_count = {"authz_total": 0, "authz_blocked": 0, "risk_total": 0, "risk_gated": 0}

    # ------------------------------------------------------------
    section("① 正常调用：只读工具，任何角色都可以")
    r = gw.call(builder(task_id="T1", role=ROLE_READER, tool_name="query_order", args={"order_id": "ORD-000123"}))
    check("reader 查询订单 → executed", r.status == "executed", f"status={r.status} result={r.result}")

    r = gw.call(builder(task_id="T1", role=ROLE_READER, tool_name="query_user_profile", args={"user_id": "U00001"}))
    check("reader 查用户资料 → denied（含隐私，仅 admin）", r.status == "denied", r.reason)

    r = gw.call(builder(task_id="T1", role=ROLE_ADMIN, tool_name="query_user_profile", args={"user_id": "U00001"}))
    check("admin 查用户资料 → executed", r.status == "executed", f"result={r.result}")

    # ------------------------------------------------------------
    section("② 越权拦截：角色不在白名单")
    sec_count["authz_total"] += 1
    r = gw.call(builder(task_id="T1", role=ROLE_READER, tool_name="issue_refund",
                        args={"order_id": "ORD-000123", "amount": 50}))
    blocked = r.status == "denied" and "permission_denied" in r.reason
    sec_count["authz_blocked"] += int(blocked)
    check("reader 发起退款 → denied", blocked, r.reason)

    sec_count["authz_total"] += 1
    r = gw.call(builder(task_id="T1", role=ROLE_OPERATOR, tool_name="delete_record",
                        args={"table": "orders", "record_id": "ORD-000123"}))
    blocked = r.status == "denied" and "permission_denied" in r.reason
    sec_count["authz_blocked"] += int(blocked)
    check("operator 删除记录 → denied（仅 admin）", blocked, r.reason)

    sec_count["authz_total"] += 1
    r = gw.call(builder(task_id="T1", role="unknown_role", tool_name="query_order",
                        args={"order_id": "ORD-000123"}))
    blocked = r.status == "denied" and "permission_denied" in r.reason
    sec_count["authz_blocked"] += int(blocked)
    check("未知角色 → denied", blocked, r.reason)

    # ------------------------------------------------------------
    section("③ 参数校验：挡住模型幻觉/注入的参数")
    r = gw.call(builder(task_id="T1", role=ROLE_READER, tool_name="query_order",
                        args={"order_id": "'; DROP TABLE orders; --"}))
    check("订单号注入字符串 → denied（格式不匹配）", r.status == "denied" and "invalid_arguments" in r.reason, r.reason)

    r = gw.call(builder(task_id="T1", role=ROLE_OPERATOR, tool_name="issue_refund",
                        args={"order_id": "ORD-000123", "amount": -100}))
    check("退款金额为负 → denied", r.status == "denied" and "invalid_arguments" in r.reason, r.reason)

    r = gw.call(builder(task_id="T1", role=ROLE_ADMIN, tool_name="delete_record",
                        args={"table": "users", "record_id": "x"}))
    check("删除非白名单表 → denied", r.status == "denied" and "invalid_arguments" in r.reason, r.reason)

    # ------------------------------------------------------------
    section("④ 高危审批：动态规则 + 静态规则")
    sec_count["risk_total"] += 1
    r = gw.call(builder(task_id="T2", role=ROLE_OPERATOR, tool_name="issue_refund",
                        args={"order_id": "ORD-000456", "amount": 2000}))
    gated = r.status == "pending_approval"
    sec_count["risk_gated"] += int(gated)
    check("退款 2000 元（>500）→ pending_approval", gated, f"approval_id={r.approval_id}")
    big_refund_apv = r.approval_id

    r = gw.call(builder(task_id="T2", role=ROLE_OPERATOR, tool_name="issue_refund",
                        args={"order_id": "ORD-000457", "amount": 50}))
    check("退款 50 元（≤500）→ executed（无需审批）", r.status == "executed", f"result={r.result}")

    sec_count["risk_total"] += 1
    r = gw.call(builder(task_id="T2", role=ROLE_OPERATOR, tool_name="send_email",
                        args={"to": "a@example.com", "subject": "hi", "body": "test"}))
    gated = r.status == "pending_approval"
    sec_count["risk_gated"] += int(gated)
    check("发邮件 → pending_approval（静态高危）", gated, f"approval_id={r.approval_id}")

    # 人工审批 → 通过后执行
    r = gw.approve_and_run(big_refund_apv, approver="manager@corp", run=True)
    check("审批通过后执行 → executed", r.status == "executed", f"result={r.result}")

    # 人工审批 → 拒绝
    r2 = gw.call(builder(task_id="T3", role=ROLE_ADMIN, tool_name="delete_record",
                         args={"table": "logs", "record_id": "L-1"}))
    rejected = gw.approve_and_run(r2.approval_id, approver="manager@corp", run=False)
    check("审批拒绝 → denied", rejected.status == "denied", rejected.reason)

    # ------------------------------------------------------------
    section("⑤ 幂等：同一任务重复调用只产生一次副作用")
    call_args = {"order_id": "ORD-000888", "amount": 300}
    first = gw.call(builder(task_id="T4", role=ROLE_OPERATOR, tool_name="issue_refund", args=call_args))
    second = gw.call(builder(task_id="T4", role=ROLE_OPERATOR, tool_name="issue_refund", args=call_args))
    check("第 1 次调用 → executed", first.status == "executed", f"refund_id={first.result.get('refund_id') if first.result else None}")
    check("第 2 次（同任务同参数）→ cached，不重复执行", second.status == "cached", second.reason)
    same = (
        first.result
        and second.result
        and first.result.get("refund_id") == second.result.get("refund_id")
    )
    check("两次返回同一个 refund_id（副作用只发生一次）", bool(same),
          f"{first.result.get('refund_id') if first.result else None} vs {second.result.get('refund_id') if second.result else None}")

    third = gw.call(builder(task_id="T5", role=ROLE_OPERATOR, tool_name="issue_refund", args=call_args))
    check("不同任务、相同参数 → 正常执行（是新业务）", third.status == "executed",
          f"refund_id={third.result.get('refund_id') if third.result else None}")

    # ------------------------------------------------------------
    section("⑥ 工具不存在 + 超时保护")
    r = gw.call(builder(task_id="T6", role=ROLE_ADMIN, tool_name="drop_database", args={}))
    check("未注册工具 → denied", r.status == "denied" and "tool_not_found" in r.reason, r.reason)

    t0 = time.time()
    r = gw.call(builder(task_id="T7", role=ROLE_READER, tool_name="slow_task", args={"seconds": 2}))
    elapsed = time.time() - t0
    check("慢工具（2s）在超时 0.5s 后返回 → timeout", r.status == "timeout", r.reason)
    check("超时保护真实生效（耗时 < 1s，而不是等 2s）", elapsed < 1.0, f"实际耗时 {elapsed:.2f}s")

    # ------------------------------------------------------------
    section("⑦ 审批单幂等：重复的高危调用只创建一个审批单")
    high_args = {"order_id": "ORD-000999", "amount": 3000}
    a1 = gw.call(builder(task_id="T8", role=ROLE_OPERATOR, tool_name="issue_refund", args=high_args))
    pending_after_first = len(approvals.list_pending())
    a2 = gw.call(builder(task_id="T8", role=ROLE_OPERATOR, tool_name="issue_refund", args=high_args))
    pending_after_second = len(approvals.list_pending())

    check("首次高危调用 → pending_approval", a1.status == "pending_approval", f"approval_id={a1.approval_id}")
    check("重试同一调用 → 复用同一个审批单", a1.approval_id == a2.approval_id,
          f"{a1.approval_id} vs {a2.approval_id}")
    check("待审批队列没有重复增长", pending_after_first == pending_after_second,
          f"{pending_after_first} → {pending_after_second}")

    # 审批通过并执行后，再次重试 → 应该命中幂等（不重复执行）
    gw.approve_and_run(a1.approval_id, approver="manager@corp", run=True)
    a3 = gw.call(builder(task_id="T8", role=ROLE_OPERATOR, tool_name="issue_refund", args=high_args))
    check("审批执行后再次重试 → cached（不重复退款）", a3.status == "cached", a3.reason)

    # ------------------------------------------------------------
    section("指标汇总")

    authz_rate = sec_count["authz_blocked"] / sec_count["authz_total"] if sec_count["authz_total"] else 0
    risk_rate = sec_count["risk_gated"] / sec_count["risk_total"] if sec_count["risk_total"] else 0
    stats = audit.stats()

    print(f"  越权拦截率            {sec_count['authz_blocked']}/{sec_count['authz_total']} = {authz_rate:.0%}")
    print(f"  高危操作审批覆盖率     {sec_count['risk_gated']}/{sec_count['risk_total']} = {risk_rate:.0%}")
    print(f"  审计记录总数          {stats['total']}")
    print(f"  审计状态分布          {stats['by_status']}")
    print(f"  拒绝原因分布          {stats['by_reason']}")
    print(f"  待审批队列剩余        {len(approvals.list_pending())}")

    print()
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        if not ok:
            print(f"  FAIL  {name}")
    print(f"  结果: {passed}/{len(results)} 通过")
    print(f"  审计落盘: {AUDIT_PATH}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
