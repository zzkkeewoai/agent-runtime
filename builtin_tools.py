"""智能客服 Agent 的工具集 —— 6 个工具覆盖三类风险

【场景】本项目以**电商智能客服**为落地场景：客服 Agent 替用户
查订单、查资料、发起退款、发通知、清理记录。

这些动作里既有只读查询，也有**一旦执行错就造成真实资金/数据损失**的写操作 ——
这正是"Agent 敢不敢上生产"最直观的战场：
    退款执行两次 → 赔钱；通知发错人 → 收不回；记录误删 → 数据丢失。

设计意图：这 6 个工具刻意覆盖了企业里最常见的三类风险，
这样后面的测试才能测出真实的拦截效果：

  只读低危      query_order          任何角色可调（客服日常最高频操作）
  只读但敏感    query_user_profile   仅主管（含手机号等个人信息）
  写 + 涉及钱   issue_refund         动态高危：金额 > 500 需审批
  对外副作用    send_email           静态高危：永远需审批（发出去收不回）
  极高危        delete_record        静态高危 + 仅主管
  慢接口        slow_task            用于验证超时保护（防止单次调用拖垮服务）

参数 schema 都加了强约束（格式/范围）——这是挡"模型幻觉参数"的第一道防线。
"""
import uuid
from typing import Any

from pydantic import BaseModel, Field

from registry import ToolSpec

# ============================================================
# 角色定义（简化版 RBAC）—— 对应客服团队的分级权限
# ============================================================
# 注意：这里的字符串值是**契约**（评测用例、审计记录都依赖它），不要改；
#      中文注释只是为了让"谁能做什么"一眼看清。
ROLE_READER = "agent_reader"      # 初级客服：只能查订单等只读操作
ROLE_OPERATOR = "agent_operator"  # 客服专员：可发起退款，大额需主管审批
ROLE_ADMIN = "agent_admin"        # 客服主管：可查敏感信息、可删记录
ALL_ROLES = {ROLE_READER, ROLE_OPERATOR, ROLE_ADMIN}


# ============================================================
# ① 查订单（只读，低危）
# ============================================================
class QueryOrderArgs(BaseModel):
    """订单号有严格格式约束：不符合格式的（包括注入字符串）在参数校验阶段就被挡住。"""

    order_id: str = Field(..., pattern=r"^ORD-\d{6}$", description="订单号，格式 ORD-123456")


def _query_order(args: QueryOrderArgs) -> dict:
    # 模拟数据源
    return {"order_id": args.order_id, "status": "paid", "amount": 1280.0}


TOOL_QUERY_ORDER = ToolSpec(
    name="query_order",
    description="按订单号查询订单的状态与金额。仅查询，不修改任何数据。",
    args_model=QueryOrderArgs,
    handler=_query_order,
    allowed_roles=ALL_ROLES,
    idempotent=True,
    side_effect=False,
)


# ============================================================
# ② 查用户资料（只读，但含敏感信息 → 权限收紧）
# ============================================================
class QueryProfileArgs(BaseModel):
    user_id: str = Field(..., pattern=r"^U\d{5}$", description="用户 ID，格式 U12345")


def _query_user_profile(args: QueryProfileArgs) -> dict:
    return {"user_id": args.user_id, "phone": "138****0000", "level": "vip"}


TOOL_QUERY_PROFILE = ToolSpec(
    name="query_user_profile",
    description="查询用户资料（含手机号等个人信息）。涉及隐私，仅管理员可调用。",
    args_model=QueryProfileArgs,
    handler=_query_user_profile,
    allowed_roles={ROLE_ADMIN},  # ← 含个人信息，权限收紧
    idempotent=True,
    side_effect=False,
)


# ============================================================
# ③ 发起退款（写 + 涉及钱 → 动态高危）
# ============================================================
class IssueRefundArgs(BaseModel):
    order_id: str = Field(..., pattern=r"^ORD-\d{6}$")
    amount: float = Field(..., gt=0, le=100000, description="退款金额（元），必须为正且不超过 10 万")


def _issue_refund(args: IssueRefundArgs) -> dict:
    return {
        "refund_id": f"RF-{uuid.uuid4().hex[:8]}",
        "order_id": args.order_id,
        "amount": args.amount,
        "status": "submitted",
    }


# 动态审批规则：金额超过 500 元才需要人工审批
REFUND_APPROVAL_THRESHOLD = 500.0

TOOL_ISSUE_REFUND = ToolSpec(
    name="issue_refund",
    description="对指定订单发起退款。会产生真实的资金操作。",
    args_model=IssueRefundArgs,
    handler=_issue_refund,
    allowed_roles={ROLE_OPERATOR, ROLE_ADMIN},
    requires_approval=False,  # 不是"永远高危"
    approval_rule=lambda a: a.amount > REFUND_APPROVAL_THRESHOLD,  # ← 按金额动态判断
    idempotent=True,
    side_effect=True,
)


# ============================================================
# ④ 发邮件（对外副作用 → 静态高危）
# ============================================================
class SendEmailArgs(BaseModel):
    to: str = Field(..., pattern=r"^[\w.+-]+@[\w-]+\.[\w.]+$", description="收件人邮箱")
    subject: str = Field(..., min_length=1, max_length=100)
    body: str = Field(..., min_length=1, max_length=2000)


def _send_email(args: SendEmailArgs) -> dict:
    return {"message_id": f"MSG-{uuid.uuid4().hex[:8]}", "to": args.to, "sent": True}


TOOL_SEND_EMAIL = ToolSpec(
    name="send_email",
    description="发送邮件给指定收件人。一旦发出无法撤回。",
    args_model=SendEmailArgs,
    handler=_send_email,
    allowed_roles={ROLE_OPERATOR, ROLE_ADMIN},
    requires_approval=True,  # ← 对外不可撤回，永远要审批
    idempotent=True,
    side_effect=True,
)


# ============================================================
# ⑤ 删记录（极高危 → 静态高危 + 权限收紧）
# ============================================================
class DeleteRecordArgs(BaseModel):
    table: str = Field(..., pattern=r"^(orders|refunds|logs)$", description="表名（白名单）")
    record_id: str = Field(..., min_length=1, max_length=32)


def _delete_record(args: DeleteRecordArgs) -> dict:
    return {"deleted": True, "table": args.table, "record_id": args.record_id}


TOOL_DELETE_RECORD = ToolSpec(
    name="delete_record",
    description="删除指定表中的一条记录。不可恢复。",
    args_model=DeleteRecordArgs,
    handler=_delete_record,
    allowed_roles={ROLE_ADMIN},
    requires_approval=True,
    idempotent=False,  # ← 删除不幂等：重放语义不同（第二次可能删的是别的东西）
    side_effect=True,
)


# ============================================================
# 一个"会挂死"的工具 —— 用来测超时保护
# ============================================================
class SlowTaskArgs(BaseModel):
    seconds: float = Field(..., gt=0, le=60)


def _slow_task(args: SlowTaskArgs) -> dict:  # pragma: no cover - 测试用
    import time

    time.sleep(args.seconds)
    return {"slept": args.seconds}


TOOL_SLOW_TASK = ToolSpec(
    name="slow_task",
    description="模拟耗时任务（测试超时保护用）。",
    args_model=SlowTaskArgs,
    handler=_slow_task,
    allowed_roles=ALL_ROLES,
    timeout_seconds=0.5,  # ← 声明 0.5 秒超时
    idempotent=False,
    side_effect=False,
)


ALL_TOOLS: list[ToolSpec] = [
    TOOL_QUERY_ORDER,
    TOOL_QUERY_PROFILE,
    TOOL_ISSUE_REFUND,
    TOOL_SEND_EMAIL,
    TOOL_DELETE_RECORD,
    TOOL_SLOW_TASK,
]
